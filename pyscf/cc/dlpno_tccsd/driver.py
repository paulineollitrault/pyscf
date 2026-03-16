"""
Top-level driver for DMRG-DLPNO-TCCSD(T).

The driver takes a converged RHF object, an active space specification
(ncas, nelec), and an initial MO ordering (from the caller's active-space
selection protocol, e.g. MP2 natural orbitals) and runs the full pipeline:

  Stage 1: Split-localise each orbital subspace independently
  Stage 2: DMRG-CI in the localised active space → tailoring amplitudes
  Stage 3: PAO + PNO construction in the localised basis
  Stage 4: Pair screening and CAS-pair identification
  Stage 5: DLPNO-TCCSD (with DMRG tailoring)
  Stage 6: External (T) correction

The split-localisation (Stage 1) is performed BEFORE DMRG-CI (Stage 2) so
that the CAS cluster amplitudes extracted from DMRG live in the same local
MO basis as the DLPNO pair correlation treatment.

Typical usage
-------------

    from pyscf import gto, scf, mp
    from pyscf.cc.dlpno_tccsd import run_dlpno_tccsd_t

    mol = gto.M(...)
    mf  = scf.RHF(mol).density_fit(); mf.kernel()

    # Active space selection (caller's choice — here MP2 NOONs)
    mymp2 = mp.MP2(mf); mymp2.kernel()
    natocc, natorb = ...   # diagonalise MP2 1-RDM
    # arrange natorb columns as: core | active-occ | active-vir | external-vir
    mo_init = natorb_reordered

    result = run_dlpno_tccsd_t(mf, ncas=7, nelec=10, mo_init=mo_init,
                               dmrg_maxM=1000, dmrg_scratch='/tmp/dmrg',
                               frozen=6)

Frozen-core treatment
---------------------
frozen=N (int) freezes the N lowest occupied MOs from the correlation
treatment.  frozen can also be a list of MO indices to freeze.

References:
    Lang et al., JCTC 2020, 16, 3028
    Riplinger & Neese, JCP 2013, 138, 034106
    Ye & Berkelbach, JCTC 2024, 20, 8948
"""

import copy as _copy
import numpy as np
from pyscf import mcscf
from pyscf import dmrgscf
from pyscf.lib import logger

from pyscf.cc.dlpno_tccsd.dmrg_interface import get_cas_amplitudes
from pyscf.cc.dlpno_tccsd.local_orbs import split_localize_orbitals, make_paos
from pyscf.cc.dlpno_tccsd.pno import make_pnos
from pyscf.cc.dlpno_tccsd.screening import classify_pairs
from pyscf.cc.dlpno_tccsd.lccsd import run_lccsd
from pyscf.cc.dlpno_tccsd.lccsd_t import run_lccsd_t_ext


def run_dlpno_tccsd_t(mf, ncas, nelec, mo_init=None,
                      frozen=0,
                      ncores=1,
                      lmo_method='pipek-mezey',
                      dmrg_maxM=1000,
                      dmrg_tol=1e-10,
                      dmrg_scratch='/tmp/dmrg',
                      T_CutPNO=1e-7,
                      T_CutPairs=1e-4,
                      T_CutPairs_MP2=1e-6,
                      T_CutDO=0.02,
                      S_cut_domain=1e-6,
                      cas_pno_proj_thresh=0.99,
                      ccsd_conv_tol=1e-7,
                      ccsd_max_cycle=50,
                      verbose=4):
    """Run DMRG-DLPNO-TCCSD(T).

    The caller provides:
      - mf:      converged DF-RHF
      - ncas:    number of active (CAS) orbitals
      - nelec:   number of active electrons (int or (nalpha, nbeta))
      - mo_init: (nao, nmo) initial MO matrix with subspaces already in the
                 intended order: frozen-core | inactive-occ | active-occ |
                 active-vir | external-vir.  Typically the MP2 natural orbital
                 arrangement from active-space selection.  If None, mf.mo_coeff
                 is used (canonical HF ordering).

    The driver then:
      1. Split-localises each orbital subspace independently (Pipek-Mezey by
         default) so that CAS and inactive occupied are separately localised.
      2. Runs DMRG-CI in the localised active space to obtain tailoring
         amplitudes.
      3. Builds PAOs and PNOs in the same local basis.
      4. Classifies pairs (CAS / strong / weak / negligible).
      5. Runs DLPNO-TCCSD tailored by DMRG.
      6. Adds the external (T) correction.

    Args:
        mf: Converged DF-RHF (scf.RHF(mol).density_fit()).
        ncas (int): Number of CAS orbitals.
        nelec (int or tuple): Active electrons.  If int, assumed closed-shell
            (nelec//2, nelec//2).
        mo_init (np.ndarray, optional): Initial MO coefficient matrix
            (nao, nmo) with subspaces arranged as described above.
        frozen (int or list): Frozen-core MOs excluded from correlation.
        ncores (int): Number of CPU cores to use.  Sets both the PySCF/BLAS
            thread count (used by RHF, MP2, PNO construction, LCCSD matrix
            operations) and the DMRG thread count.  Default 1 (serial).
        lmo_method (str): Localisation method: 'pipek-mezey' or 'boys'.
        dmrg_maxM (int): Maximum DMRG bond dimension.
        dmrg_tol (float): DMRG convergence tolerance.
        dmrg_scratch (str): DMRG scratch/runtime directory.
        T_CutPNO (float): PNO truncation threshold.
        T_CutPairs (float): Strong/weak pair threshold.
        T_CutPairs_MP2 (float): Weak/negligible pair threshold.
        T_CutDO (float): PAO domain differential overlap threshold.
        S_cut_domain (float): PAO canonical orthogonalisation cutoff.
        cas_pno_proj_thresh (float): PNO-in-CAS-virtual projection threshold.
        ccsd_conv_tol (float): Pair CCSD convergence threshold.
        ccsd_max_cycle (int): Maximum pair CCSD iterations.
        verbose (int): Verbosity.

    Returns:
        result (dict):
            'e_hf'        : HF total energy
            'e_dmrg'      : DMRG-CI total energy (localised active space)
            'e_lmp2_weak' : LMP2 correction from weak pairs
            'e_tccsd'     : DLPNO-TCCSD correlation energy (strong pairs)
            'e_t'         : External (T) correction
            'e_total'     : e_hf + e_tccsd + e_lmp2_weak + e_t
    """
    if not hasattr(mf, 'with_df') or mf.with_df is None:
        raise ValueError('mf must be a density-fitted RHF '
                         '(use scf.RHF(mol).density_fit()).')

    mol = mf.mol
    log = logger.new_logger(mf, verbose)

    nmo = mf.mo_coeff.shape[1]
    nocc = int(np.count_nonzero(mf.mo_occ > 1e-10))
    print(f'  Basis: {mol.basis}  |  nMO: {nmo}  (occ: {nocc}  vir: {nmo - nocc})'
          f'  |  ncores: {ncores}')

    # Set PySCF/BLAS thread count for all subsequent numpy/scipy/BLAS calls
    from pyscf import lib as _pyscf_lib
    _pyscf_lib.num_threads(ncores)

    # Normalise nelec to (nalpha, nbeta)
    mol_spin = mol.spin   # = 2*S; 0 for singlet, 2 for triplet, etc.
    if isinstance(nelec, (int, np.integer)):
        n = int(nelec)
        if mol_spin > 0:
            nelec_cas = ((n + mol_spin) // 2, (n - mol_spin) // 2)
        else:
            nelec_cas = (n // 2, n // 2)
    else:
        nelec_cas = (int(nelec[0]), int(nelec[1]))
    # ------------------------------------------------------------------
    # Frozen-core bookkeeping
    # ------------------------------------------------------------------
    if isinstance(frozen, (list, tuple, np.ndarray)):
        frozen_idx = np.asarray(frozen, dtype=int)
        n_frozen = len(frozen_idx)
    else:
        n_frozen = int(frozen)
        frozen_idx = np.arange(n_frozen)

    if n_frozen > 0:
        log.info('Frozen-core: %d orbital(s)', n_frozen)

    # ------------------------------------------------------------------
    # Stage 1: Split-localise orbital subspaces
    # ------------------------------------------------------------------
    print('  Stage 1: Split-localising orbital subspaces...', flush=True)

    if mo_init is None:
        mo_init = mf.mo_coeff

    # Build a lightweight copy of mf with mo_init as mo_coeff so that
    # split_localize_orbitals sees the caller's initial orbital ordering.
    mf_init = _copy.copy(mf)
    mf_init.mo_coeff = mo_init

    # For open-shell (ROHF), provide natural orbital occupations so that
    # the active occupied subspace is split into doubly/singly occupied
    # blocks before localization (Lang et al. JCTC 2020).
    # In the MP2 NO ordering (descending NOON), the last mol_spin active
    # occupied orbitals are the SOMOs (NOON ≈ 1.0).
    occ_natural = None
    if mol_spin > 0:
        nocc_cas_a = nelec_cas[0]
        n_docc = max(0, nocc_cas_a - mol_spin)
        occ_natural = np.array([2.0] * n_docc + [1.0] * mol_spin)

    mo_loc, C_lmo = split_localize_orbitals(
        mf_init, ncas, nelec_cas,
        method=lmo_method, frozen=n_frozen,
        occ_natural=occ_natural)

    # ------------------------------------------------------------------
    # Stage 2: DMRG-CI in the localised active space
    # ------------------------------------------------------------------
    print(f'  Stage 2: DMRG-CI({ncas},{sum(nelec_cas)}) maxM={dmrg_maxM}...',
          flush=True)

    mc = mcscf.CASCI(mf, ncas, nelec_cas)
    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=dmrg_maxM, tol=dmrg_tol)
    mc.fcisolver.runtimeDir = dmrg_scratch
    mc.fcisolver.scratchDirectory = dmrg_scratch
    mc.fcisolver.threads = ncores
    mc.fcisolver.memory = int(mol.max_memory * 0.8 / 1000)

    # Ramp-up schedule based on dmrg_maxM
    M1 = min(dmrg_maxM // 4, 100)
    M2 = min(dmrg_maxM // 2, 250)
    M3 = min(3 * dmrg_maxM // 4, 500)
    mc.fcisolver.scheduleSweeps  = [0,    4,    8,    12,   16,         20        ]
    mc.fcisolver.scheduleMaxMs   = [M1,   M2,   M3,   dmrg_maxM, dmrg_maxM, dmrg_maxM]
    mc.fcisolver.scheduleNoises  = [1e-4, 1e-4, 1e-5, 1e-5, 0.0,        0.0       ]
    mc.fcisolver.scheduleTols    = [1e-5, 1e-5, 1e-6, 1e-7, 1e-8,       dmrg_tol  ]
    mc.fcisolver.twodot_to_onedot = 18
    mc.verbose = verbose
    mc.kernel(mo_loc)

    print(f'  E(DMRG-CI) = {mc.e_tot:.10f}')

    # Extract CAS tailoring amplitudes
    t1_cas, t2_cas, occ_cas_idx, vir_cas_idx = get_cas_amplitudes(
        mc, verbose=verbose)

    log.info('CAS amplitude norms: |t1|=%.4g  |t2|=%.4g',
             np.linalg.norm(t1_cas), np.linalg.norm(t2_cas))

    # Convert occ_cas_idx from full-MO space to LMO-local space.
    # get_cas_amplitudes returns indices like ncore + 0..nocc_cas-1
    # (e.g. [17,18,19,20,21] for ncore=17).  LMO pair keys (i,j) in
    # pno_spaces are 0-based from the frozen boundary (LMO 0 = full-MO
    # n_frozen).  Subtracting n_frozen aligns the two index spaces.
    occ_cas_idx = occ_cas_idx - n_frozen

    # ------------------------------------------------------------------
    # Stage 3: PAO + PNO construction
    # ------------------------------------------------------------------
    print('  Stage 3: PAO + PNO construction...', flush=True)

    s1e = mf.get_ovlp()
    C_pao, pao_domains, S_pao, F_pao = make_paos(
        mc, C_lmo, T_CutDO=T_CutDO, s1e=s1e)

    nlmo = C_lmo.shape[1]
    domain_sizes = [len(pao_domains[i]) for i in range(nlmo)]
    print(f'  PAO domains: {nlmo} LMOs  |  sizes (AOs): '
          f'min={min(domain_sizes)}  max={max(domain_sizes)}  '
          f'avg={np.mean(domain_sizes):.1f}')
    print('    ' + '  '.join(f'LMO{i}:{s}' for i, s in enumerate(domain_sizes)))

    pno_spaces, _, _, _ = make_pnos(
        mf, C_lmo, C_pao, pao_domains, S_pao, F_pao,
        T_CutPNO=T_CutPNO, T_CutPairs=T_CutPairs,
        S_cut_domain=S_cut_domain, verbose=verbose)

    # ------------------------------------------------------------------
    # Stage 4: Pair screening
    # ------------------------------------------------------------------
    (cas_pairs, strong_pairs, weak_pairs, negligible_pairs,
     e_lmp2_weak, e_lmp2_strong) = classify_pairs(
        pno_spaces, occ_cas_idx, vir_cas_idx,
        mc.mo_coeff, s1e,
        T_CutPairs=T_CutPairs, T_CutPairs_MP2=T_CutPairs_MP2,
        cas_pno_proj_thresh=cas_pno_proj_thresh,
        verbose=verbose)

    print(f'  Pairs: {len(cas_pairs)} CAS  {len(strong_pairs)} strong  '
          f'{len(weak_pairs)} weak  {len(negligible_pairs)} negligible')
    if strong_pairs:
        pno_counts = [len(pno_spaces[p]['n_pno']) for p in strong_pairs]
        print(f'  Strong-pair PNOs: min={min(pno_counts)}  '
              f'max={max(pno_counts)}  avg={np.mean(pno_counts):.1f}')

    # ------------------------------------------------------------------
    # Stage 5: DLPNO-TCCSD
    # ------------------------------------------------------------------
    print('  Stage 5: DLPNO-TCCSD...', flush=True)

    e_tccsd, t2_pno_all, t1_singles = run_lccsd(
        mf, C_lmo, pno_spaces,
        strong_pairs=strong_pairs + list(cas_pairs),
        cas_pairs=cas_pairs,
        t1_cas=t1_cas, t2_cas=t2_cas,
        occ_cas_idx=occ_cas_idx, vir_cas_idx=vir_cas_idx,
        mo_coeff_cas=mc.mo_coeff, s1e=s1e,
        conv_tol=ccsd_conv_tol, max_cycle=ccsd_max_cycle,
        ncores=ncores, verbose=verbose)

    log.info('E(TCCSD) correlation = %.15g', e_tccsd)

    # ------------------------------------------------------------------
    # Stage 6: External (T) correction
    # ------------------------------------------------------------------
    print('  Stage 6: (T) correction...', flush=True)

    C_cas_vir = mc.mo_coeff[:, vir_cas_idx]

    e_t = run_lccsd_t_ext(
        mf, C_lmo, pno_spaces,
        strong_pairs=strong_pairs,
        t2_pno_all=t2_pno_all,
        occ_cas_idx=occ_cas_idx,
        C_cas_vir=C_cas_vir,
        vir_cas_idx=vir_cas_idx,
        ncores=ncores, verbose=verbose)

    log.info('E(T) external = %.15g', e_t)

    # ------------------------------------------------------------------
    # Total energy
    # ------------------------------------------------------------------
    e_total = mf.e_tot + e_tccsd + e_lmp2_weak + e_t

    return {
        'e_hf':         mf.e_tot,
        'e_dmrg':       mc.e_tot,
        'e_lmp2_weak':  e_lmp2_weak,
        'e_tccsd':      e_tccsd,
        'e_t':          e_t,
        'e_total':      e_total,
        # Objects for analysis / diagnostics
        'mc':           mc,
        'mf':           mf,
        't1_cas':       t1_cas,
        't2_cas':       t2_cas,
        'occ_cas_idx':  occ_cas_idx,
        'vir_cas_idx':  vir_cas_idx,
        'cas_pairs':    cas_pairs,
        'strong_pairs': strong_pairs,
        'weak_pairs':   weak_pairs,
        'C_lmo':        C_lmo,
        'pno_spaces':   pno_spaces,
        't2_pno_all':   t2_pno_all,
    }
