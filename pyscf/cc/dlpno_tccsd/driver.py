"""
Top-level driver for DMRG-DLPNO-TCCSD(T).

Orchestrates the full pipeline in the correct order so that DMRG and DLPNO
work in the SAME localized orbital basis:

  Stage 1: RHF with density fitting
  Stage 2: Localize occupied + active virtual MOs (Pipek-Mezey / Boys)
  Stage 3: DMRG-CI (no orbital optimization) in the localized active space
  Stage 4: PAO + PNO construction in the same localized basis
  Stage 5: Pair screening and CAS-pair identification
  Stage 6: DLPNO-TCCSD (tailored CC with DMRG amplitudes injected)
  Stage 7: External-space (T) correction

Key design principle: orbitals are localized BEFORE DMRG so that the CAS
cluster amplitudes extracted from DMRG-CI live in the same local MO basis
as the DLPNO pair correlation. Running DMRG-CASSCF (with orbital optimization)
first and localizing after would put DMRG amplitudes in CASSCF-optimized MOs
while DLPNO works in a different localized basis — making the tailoring wrong.

References:
    Lang et al., JCTC 2020, 16, 3028
    Riplinger & Neese, JCP 2013, 138, 034106
    Ye & Berkelbach, JCTC 2024, 20, 8948
"""

import numpy as np
from pyscf import scf, lib
from pyscf.lib import logger

from pyscf.cc.dlpno_tccsd.dmrg_interface import run_dmrg_casci, run_dmrg_casscf, get_cas_amplitudes
from pyscf.cc.dlpno_tccsd.local_orbs import split_localize_orbitals, make_paos
from pyscf.cc.dlpno_tccsd.pno import make_pnos
from pyscf.cc.dlpno_tccsd.screening import classify_pairs
from pyscf.cc.dlpno_tccsd.lccsd import run_lccsd
from pyscf.cc.dlpno_tccsd.lccsd_t import run_lccsd_t_ext


def run_dlpno_tccsd_t(mol, ncas, nelec_cas,
                      basis=None,
                      maxM=1000,
                      T_CutPNO=1e-7,
                      T_CutPairs=1e-4,
                      T_CutPairs_MP2=1e-6,
                      T_CutDO=0.02,
                      S_cut_domain=1e-6,
                      cas_pno_proj_thresh=0.99,
                      lmo_method='pipek-mezey',
                      frozen=0,
                      ccsd_conv_tol=1e-7,
                      ccsd_max_cycle=50,
                      scratch='./dmrg_scratch',
                      verbose=4):
    """Run the full DMRG-DLPNO-TCCSD(T) pipeline.

    Args:
        mol: PySCF Mole object (already built).
        ncas (int): Number of active (CAS) orbitals.
        nelec_cas (int or tuple): Number of active electrons. If int, assumed
            to be equal number of alpha and beta electrons: (nelec//2, nelec//2).
        basis (str, optional): Ignored (basis already set in mol). Kept for
            API compatibility.
        maxM (int): Maximum DMRG bond dimension.
        T_CutPNO (float): PNO occupation truncation threshold.
            Use 1e-7 (TightPNO) or 3.33e-7 (NormalPNO).
        T_CutPairs (float): Strong/weak pair separation threshold.
        T_CutPairs_MP2 (float): Weak/negligible pair separation threshold.
        T_CutDO (float): PAO domain differential overlap threshold (Boughton-Pulay).
        S_cut_domain (float): Eigenvalue cutoff for PAO canonical orthogonalization.
        cas_pno_proj_thresh (float): Threshold for PNO-in-CAS-vir test.
        lmo_method (str): 'pipek-mezey' or 'boys'.
        frozen (int): Number of frozen-core orbitals.
        ccsd_conv_tol (float): Convergence threshold for pair CCSD.
        ccsd_max_cycle (int): Maximum pair CCSD iterations.
        scratch (str): DMRG scratch directory.
        verbose (int): Verbosity level.

    Returns:
        result (dict): Energy components:
            'e_hf'       : RHF total energy
            'e_dmrg_ci'  : DMRG-CI total energy (localized active space)
            'e_lmp2_weak': LMP2 correction from weak pairs
            'e_tccsd'    : DLPNO-TCCSD correlation energy (strong pairs)
            'e_t'        : External (T) correction
            'e_total'    : e_hf + e_tccsd + e_lmp2_weak + e_t
    """
    # Normalize nelec_cas
    if isinstance(nelec_cas, int):
        nalpha = nelec_cas // 2 + nelec_cas % 2
        nbeta = nelec_cas // 2
        nelec_cas = (nalpha, nbeta)

    # -----------------------------------------------------------------------
    # Stage 1: RHF with density fitting
    # -----------------------------------------------------------------------
    log = logger.new_logger(mol, verbose)
    log.note('\n' + '='*70)
    log.note('DMRG-DLPNO-TCCSD(T)')
    log.note('='*70)
    log.note('Stage 1: RHF')

    mf = scf.RHF(mol).density_fit()
    mf.verbose = verbose - 1
    mf.run()

    if not mf.converged:
        log.warn('RHF did not converge!')

    log.note('E(HF) = %.15g', mf.e_tot)

    # Index bookkeeping (from RHF, before any active space treatment)
    nocc_full = mol.nelectron // 2
    nocc_cas_a = nelec_cas[0]   # alpha electrons in CAS
    ncore = nocc_full - nocc_cas_a
    nvir_cas = ncas - nocc_cas_a

    # MO index ranges in the full canonical MO list
    occ_cas_idx = np.arange(ncore, nocc_full)          # active occupied
    vir_cas_idx = np.arange(nocc_full, nocc_full + nvir_cas)  # active virtual

    log.info('Index summary: ncore=%d nocc_cas=%d nvir_cas=%d nocc_full=%d',
             ncore, nocc_cas_a, nvir_cas, nocc_full)
    log.info('  occ_cas_idx = %s', occ_cas_idx)
    log.info('  vir_cas_idx = %s', vir_cas_idx)

    # AO overlap matrix (needed throughout)
    s1e = mf.get_ovlp()

    # -----------------------------------------------------------------------
    # Stage 2: CASSCF orbital preparation
    # -----------------------------------------------------------------------
    # Orbital optimization is required before localization to get the correct
    # active space shape. Without it, canonical HF virtuals are poor active
    # orbitals and CASCI gives wrong energies.
    # For small CAS (ncas <= ~16): use standard FCI solver (no DMRG needed).
    # For large CAS: use run_dmrg_casscf here instead.
    log.note('\nStage 2: CASSCF orbital preparation')

    from pyscf import mcscf
    mc_prep = mcscf.CASSCF(mf, ncas, nelec_cas)
    mc_prep.verbose = verbose - 1
    mc_prep.conv_tol = 1e-9
    mc_prep.max_cycle_macro = 50
    mc_prep.kernel()
    if not mc_prep.converged:
        log.warn('CASSCF orbital preparation did not converge.')
    log.note('CASSCF E = %.15g (orbital preparation only)', mc_prep.e_tot)

    # -----------------------------------------------------------------------
    # Stage 3: Split-localize the CASSCF-optimized orbital subspaces
    # -----------------------------------------------------------------------
    # Key property: FCI/DMRG energy is invariant under unitary rotations
    # within the active space. So DMRG-CI in localized CASSCF orbitals gives
    # exactly the same energy as CASSCF. This is what Lang et al. 2020 exploit.
    log.note('\nStage 3: Split-localization of CASSCF-optimized subspaces')
    log.note('  Subspaces localized independently (no inter-subspace mixing):')
    log.note('  inactive occupied | active occupied | active virtual')

    # Build a temporary mf-like object with CASSCF MOs for split_localize
    import copy
    mf_casscf = copy.copy(mf)
    mf_casscf.mo_coeff = mc_prep.mo_coeff
    mf_casscf.mo_occ = mf.mo_occ   # use original occupations for nocc detection

    C_mo_loc, C_lmo = split_localize_orbitals(
        mf_casscf, ncas, nelec_cas,
        method=lmo_method, frozen=frozen)

    log.note('Split localization complete.')

    # -----------------------------------------------------------------------
    # Stage 4: DMRG-CI in the localized active space (NO orbital optimization)
    # -----------------------------------------------------------------------
    # FCI/DMRG in the localized basis gives the same energy as CASSCF because
    # the CI problem is invariant under active-space unitary rotations.
    log.note('\nStage 4: DMRG-CI in localized active space')

    mc = run_dmrg_casci(mol, mf, ncas, nelec_cas, C_mo_loc,
                        maxM=maxM, scratch=scratch, verbose=verbose)

    log.note('E(DMRG-CI) = %.15g  (should match CASSCF E = %.15g)',
             mc.e_tot, mc_prep.e_tot)

    # Extract cluster amplitudes in the localized MO basis
    t1_cas, t2_cas, occ_cas_idx, vir_cas_idx = get_cas_amplitudes(
        mc, verbose=verbose)

    log.info('CAS amplitude norms: |t1|=%.4g  |t2|=%.4g',
             np.linalg.norm(t1_cas), np.linalg.norm(t2_cas))

    # -----------------------------------------------------------------------
    # Stage 5: PAO construction (same localized basis as DMRG-CI)
    # -----------------------------------------------------------------------
    log.note('\nStage 5: PAO + PNO construction')

    # PAO projection uses mc.mo_coeff (= the localized MO basis from DMRG-CI).
    # This ensures PAOs are orthogonal to the same localized occupied space
    # that DMRG uses, so the correlation and CAS treatments are consistent.
    C_pao, pao_domains, S_pao, F_pao = make_paos(
        mc, C_lmo, T_CutDO=T_CutDO, s1e=s1e)
    log.info('PAO construction: %d PAOs, avg domain size=%.1f',
             C_pao.shape[1], np.mean([len(d) for d in pao_domains]))

    # -----------------------------------------------------------------------
    # Stage 5: PNO construction + pair classification
    # -----------------------------------------------------------------------
    log.note('\nStage 5: PNO construction')

    pno_spaces, strong_pairs_raw, weak_pairs_raw, e_lmp2_raw = make_pnos(
        mf, C_lmo, C_pao, pao_domains, S_pao, F_pao,
        T_CutPNO=T_CutPNO, T_CutPairs=T_CutPairs,
        S_cut_domain=S_cut_domain, verbose=verbose)

    log.note('LMP2 energy (before CAS classification) = %.15g', e_lmp2_raw)

    # Refine pair classification with CAS-pair check (including PNO projection)
    (cas_pairs, strong_pairs, weak_pairs, negligible_pairs,
     e_lmp2_weak, e_lmp2_strong) = classify_pairs(
        pno_spaces, occ_cas_idx, vir_cas_idx,
        mc.mo_coeff, s1e,
        T_CutPairs=T_CutPairs, T_CutPairs_MP2=T_CutPairs_MP2,
        cas_pno_proj_thresh=cas_pno_proj_thresh,
        verbose=verbose)

    log.note('Pair classification: %d CAS, %d strong, %d weak, %d negligible',
             len(cas_pairs), len(strong_pairs), len(weak_pairs),
             len(negligible_pairs))
    log.note('E(LMP2) weak pairs = %.15g', e_lmp2_weak)

    # -----------------------------------------------------------------------
    # Stage 6: DLPNO-TCCSD
    # -----------------------------------------------------------------------
    log.note('\nStage 6: DLPNO-TCCSD (fragment CCSD with CAS injection)')

    e_tccsd, t2_pno_all, t1_singles = run_lccsd(
        mf, C_lmo, pno_spaces,
        strong_pairs=strong_pairs + list(cas_pairs),  # include CAS pairs
        cas_pairs=cas_pairs,
        t1_cas=t1_cas, t2_cas=t2_cas,
        occ_cas_idx=occ_cas_idx, vir_cas_idx=vir_cas_idx,
        mo_coeff_cas=mc.mo_coeff, s1e=s1e,
        conv_tol=ccsd_conv_tol, max_cycle=ccsd_max_cycle,
        verbose=verbose)

    log.note('E(TCCSD) correlation = %.15g', e_tccsd)

    # -----------------------------------------------------------------------
    # Stage 7: External (T)
    # -----------------------------------------------------------------------
    log.note('\nStage 7: External (T) correction')

    # CAS virtual MOs in AO basis (for T2 zeroing in TCC (T))
    C_cas_vir = mc.mo_coeff[:, vir_cas_idx]   # (nao, n_cas_vir)

    e_t = run_lccsd_t_ext(
        mf, C_lmo, pno_spaces,
        strong_pairs=strong_pairs,
        t2_pno_all=t2_pno_all,
        occ_cas_idx=occ_cas_idx,
        C_cas_vir=C_cas_vir,
        vir_cas_idx=vir_cas_idx,
        verbose=verbose)

    log.note('E(T) external = %.15g', e_t)

    # -----------------------------------------------------------------------
    # Total energy
    # -----------------------------------------------------------------------
    e_total = mf.e_tot + e_tccsd + e_lmp2_weak + e_t

    log.note('\n' + '='*70)
    log.note('DMRG-DLPNO-TCCSD(T) RESULTS')
    log.note('='*70)
    log.note('E(HF)         = %20.15g', mf.e_tot)
    log.note('E(DMRG-CI)    = %20.15g  (localized active space)', mc.e_tot)
    log.note('E(TCCSD)      = %20.15g  (corr, strong pairs)', e_tccsd)
    log.note('E(LMP2 weak)  = %20.15g  (corr, weak pairs)', e_lmp2_weak)
    log.note('E(T) ext      = %20.15g', e_t)
    log.note('E(total)      = %20.15g', e_total)
    log.note('='*70)

    return {
        'e_hf':        mf.e_tot,
        'e_dmrg_ci':   mc.e_tot,
        'e_lmp2_weak': e_lmp2_weak,
        'e_tccsd':     e_tccsd,
        'e_t':         e_t,
        'e_total':     e_total,
        # Store objects for analysis / restart
        'mc':          mc,
        'mf':          mf,
        't1_cas':      t1_cas,
        't2_cas':      t2_cas,
        'occ_cas_idx': occ_cas_idx,
        'vir_cas_idx': vir_cas_idx,
        'cas_pairs':   cas_pairs,
        'strong_pairs': strong_pairs,
        'weak_pairs':  weak_pairs,
    }


def run_dlpno_tccsd_t_from_mc(mc, mf,
                               T_CutPNO=1e-7,
                               T_CutPairs=1e-4,
                               T_CutPairs_MP2=1e-6,
                               T_CutDO=0.02,
                               S_cut_domain=1e-6,
                               cas_pno_proj_thresh=0.99,
                               lmo_method='pipek-mezey',
                               frozen=0,
                               ccsd_conv_tol=1e-7,
                               ccsd_max_cycle=50,
                               verbose=4):
    """Run DLPNO-TCCSD(T) starting from a pre-converged CASSCF object.

    Useful when the DMRG-CASSCF calculation has already been done and the
    cluster amplitudes need to be extracted and injected.

    Args:
        mc: Converged CASSCF/DMRGSCF object.
        mf: RHF object with density fitting (mf.with_df must be set).
        (remaining args same as run_dlpno_tccsd_t)

    Returns:
        result (dict): Same as run_dlpno_tccsd_t.
    """
    log = logger.new_logger(mc, verbose)

    mol = mc.mol

    # Extract CAS amplitudes
    t1_cas, t2_cas, occ_cas_idx, vir_cas_idx = get_cas_amplitudes(
        mc, verbose=verbose)

    # Update mf to use CASSCF MOs
    mf.mo_coeff = mc.mo_coeff
    s1e = mf.get_ovlp()

    # LMO + PAO
    C_lmo, _ = make_lmos(mc, method=lmo_method, frozen=frozen)
    C_pao, pao_domains, S_pao, F_pao = make_paos(
        mc, C_lmo, T_CutDO=T_CutDO, s1e=s1e)

    # PNO construction
    pno_spaces, _, _, e_lmp2_raw = make_pnos(
        mf, C_lmo, C_pao, pao_domains, S_pao, F_pao,
        T_CutPNO=T_CutPNO, T_CutPairs=T_CutPairs,
        S_cut_domain=S_cut_domain, verbose=verbose)

    # Pair classification
    (cas_pairs, strong_pairs, weak_pairs, negligible_pairs,
     e_lmp2_weak, e_lmp2_strong) = classify_pairs(
        pno_spaces, occ_cas_idx, vir_cas_idx,
        mc.mo_coeff, s1e,
        T_CutPairs=T_CutPairs, T_CutPairs_MP2=T_CutPairs_MP2,
        cas_pno_proj_thresh=cas_pno_proj_thresh,
        verbose=verbose)

    # TCCSD
    e_tccsd, t2_pno_all, _ = run_lccsd(
        mf, C_lmo, pno_spaces,
        strong_pairs=strong_pairs + list(cas_pairs),
        cas_pairs=cas_pairs,
        t1_cas=t1_cas, t2_cas=t2_cas,
        occ_cas_idx=occ_cas_idx, vir_cas_idx=vir_cas_idx,
        mo_coeff_cas=mc.mo_coeff, s1e=s1e,
        conv_tol=ccsd_conv_tol, max_cycle=ccsd_max_cycle,
        verbose=verbose)

    # (T)
    C_cas_vir = mc.mo_coeff[:, vir_cas_idx]   # (nao, n_cas_vir)
    e_t = run_lccsd_t_ext(
        mf, C_lmo, pno_spaces,
        strong_pairs=strong_pairs,
        t2_pno_all=t2_pno_all,
        occ_cas_idx=occ_cas_idx,
        C_cas_vir=C_cas_vir,
        vir_cas_idx=vir_cas_idx,
        verbose=verbose)

    e_total = mf.e_tot + e_tccsd + e_lmp2_weak + e_t

    log.note('E(total DLPNO-TCCSD(T)) = %.15g', e_total)

    return {
        'e_hf':        mf.e_tot,
        'e_dmrg_ci':   mc.e_tot,
        'e_lmp2_weak': e_lmp2_weak,
        'e_tccsd':     e_tccsd,
        'e_t':         e_t,
        'e_total':     e_total,
        'mc':          mc,
        'mf':          mf,
        't1_cas':      t1_cas,
        't2_cas':      t2_cas,
        'occ_cas_idx': occ_cas_idx,
        'vir_cas_idx': vir_cas_idx,
        'cas_pairs':   cas_pairs,
        'strong_pairs': strong_pairs,
        'weak_pairs':  weak_pairs,
    }
