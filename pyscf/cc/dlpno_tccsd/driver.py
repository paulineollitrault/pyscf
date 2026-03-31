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
import time as _time
import numpy as np
from pyscf import mcscf
from pyscf.lib import logger

from pyscf.cc.dlpno_tccsd.dmrg_interface import extract_amplitudes_from_mps
from pyscf.cc.dlpno_tccsd.local_orbs import split_localize_orbitals, make_paos
from pyscf.cc.dlpno_tccsd.pno import make_pnos
from pyscf.cc.dlpno_tccsd.screening import classify_pairs
from pyscf.cc.dlpno_tccsd.lccsd import run_lccsd
from pyscf.cc.dlpno_tccsd.lccsd_t import run_lccsd_t_ext


def run_dlpno_tccsd_t(mf, ncas=None, nelec=None, mo_init=None,
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
                      verbose=4,
                      _pool=None):
    """Run DMRG-DLPNO-TCCSD(T), or plain DLPNO-CCSD(T) when ncas is None.

    When ncas is None (or 0), the DMRG/CAS stages are skipped entirely and
    the calculation reduces to a standard DLPNO-CCSD(T).

    The caller provides:
      - mf:      converged DF-RHF
      - ncas:    number of active (CAS) orbitals, or None/0 for plain DLPNO-CCSD
      - nelec:   number of active electrons (int or (nalpha, nbeta));
                 ignored when ncas is None
      - mo_init: (nao, nmo) initial MO matrix with subspaces already in the
                 intended order: frozen-core | inactive-occ | active-occ |
                 active-vir | external-vir.  Typically the MP2 natural orbital
                 arrangement from active-space selection.  If None, mf.mo_coeff
                 is used (canonical HF ordering).

    The driver then:
      1. Split-localises each orbital subspace independently (Pipek-Mezey by
         default) so that CAS and inactive occupied are separately localised.
         For plain DLPNO-CCSD, all occupied orbitals are localised together.
      2. Runs DMRG-CI in the localised active space → tailoring amplitudes.
         (Skipped for plain DLPNO-CCSD.)
      3. Builds PAOs and PNOs in the same local basis.
      4. Classifies pairs (CAS / strong / weak / negligible).
      5. Runs DLPNO-TCCSD (with DMRG tailoring) or DLPNO-CCSD.
      6. Adds the external (T) correction.

    Args:
        mf: Converged DF-RHF (scf.RHF(mol).density_fit()).
        ncas (int or None): Number of CAS orbitals.  None or 0 for plain
            DLPNO-CCSD(T) without active space.
        nelec (int, tuple, or None): Active electrons.  If int, assumed
            closed-shell (nelec//2, nelec//2).  Ignored when ncas is None.
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
            'e_dmrg'      : DMRG-CI total energy (None for plain DLPNO-CCSD)
            'e_lmp2_weak' : LMP2 correction from weak pairs
            'e_tccsd'     : DLPNO-TCCSD correlation energy (strong pairs)
            'e_t'         : External (T) correction
            'e_total'     : e_hf + e_tccsd + e_lmp2_weak + e_t
    """
    # ------------------------------------------------------------------
    # Determine whether this is a plain DLPNO-CCSD or TCCSD calculation
    # ------------------------------------------------------------------
    no_cas = (ncas is None or ncas == 0)
    if no_cas:
        ncas = 0

    if not hasattr(mf, 'with_df') or mf.with_df is None:
        import warnings
        warnings.warn(
            'mf does not have density fitting (with_df is None). '
            'Exact 4-index integrals will be used for LCCSD — correct but slow '
            'for large systems. Note: the (T) correction requires density '
            'fitting and will return 0.',
            UserWarning, stacklevel=2)

    mol = mf.mol
    log = logger.new_logger(mf, verbose)

    nmo = mf.mo_coeff.shape[1]
    nocc = int(np.count_nonzero(mf.mo_occ > 1e-10))
    method_label = 'DLPNO-CCSD(T)' if no_cas else 'DMRG-DLPNO-TCCSD(T)'
    print(f'  {method_label}')
    print(f'  Basis: {mol.basis}  |  nMO: {nmo}  (occ: {nocc}  vir: {nmo - nocc})'
          f'  |  ncores: {ncores}')

    # Set BLAS threads once. Never toggle after this point.
    from pyscf import lib as _pyscf_lib
    _pyscf_lib.num_threads(ncores)

    # Use caller-provided pool if available (avoids creating new thread
    # IDs across molecules that exceed OpenBLAS MAX_THREADS), otherwise
    # create one.
    from concurrent.futures import ThreadPoolExecutor
    _owns_pool = _pool is None
    if _owns_pool:
        _n_pool = max(1, ncores // 2)
        _shared_pool = ThreadPoolExecutor(max_workers=_n_pool) if _n_pool > 1 else None
    else:
        _shared_pool = _pool

    # Normalise nelec to (nalpha, nbeta)
    mol_spin = mol.spin   # = 2*S; 0 for singlet, 2 for triplet, etc.
    if no_cas:
        nelec_cas = (0, 0)
    elif isinstance(nelec, (int, np.integer)):
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
    _t_loc_start = _time.time()

    if mo_init is None:
        mo_init = mf.mo_coeff

    # Build a lightweight copy of mf with mo_init as mo_coeff so that
    # split_localize_orbitals sees the caller's initial orbital ordering.
    mf_init = _copy.copy(mf)
    mf_init.mo_coeff = mo_init

    if no_cas:
        # Plain DLPNO-CCSD: localise all occupied orbitals together
        from pyscf import lo
        nocc_full = np.count_nonzero(mf.mo_occ > 1e-10)
        C_occ = mo_init[:, n_frozen:nocc_full]
        if C_occ.shape[1] > 1:
            if lmo_method.lower() in ('pipek-mezey', 'pm'):
                mlo = lo.PipekMezey(mol, C_occ)
            elif lmo_method.lower() == 'boys':
                mlo = lo.Boys(mol, C_occ)
            else:
                mlo = None
            if mlo is not None:
                mlo.verbose = 0
                C_lmo = mlo.kernel()
            else:
                C_lmo = C_occ
        else:
            C_lmo = C_occ
        mo_loc = mo_init.copy()
        mo_loc[:, n_frozen:nocc_full] = C_lmo
    else:
        # TCCSD: split-localise each orbital subspace independently
        # For open-shell (ROHF), provide natural orbital occupations so that
        # the active occupied subspace is split into doubly/singly occupied
        # blocks before localization (Lang et al. JCTC 2020).
        occ_natural = None
        if mol_spin > 0:
            nocc_cas_a = nelec_cas[0]
            n_docc = max(0, nocc_cas_a - mol_spin)
            occ_natural = np.array([2.0] * n_docc + [1.0] * mol_spin)

        mo_loc, C_lmo = split_localize_orbitals(
            mf_init, ncas, nelec_cas,
            method=lmo_method, frozen=n_frozen,
            occ_natural=occ_natural)

    _t_loc = _time.time() - _t_loc_start
    print(f'  Stage 1 wall time: {_t_loc:.2f} s', flush=True)

    # ------------------------------------------------------------------
    # Stage 2: DMRG-CI in the localised active space (pyblock2)
    #          + exact CI→CC amplitude extraction from the MPS
    #          (Skipped for plain DLPNO-CCSD)
    # ------------------------------------------------------------------
    if no_cas:
        print('  Stage 2: Skipped (no active space — plain DLPNO-CCSD)', flush=True)
        t1_cas = None
        t2_cas = None
        occ_cas_idx = np.array([], dtype=int)
        vir_cas_idx = np.array([], dtype=int)
        mc = None
        e_dmrg = None
    else:
        print(f'  Stage 2: DMRG-CI({ncas},{sum(nelec_cas)}) maxM={dmrg_maxM}...',
              flush=True)

        # Build a CASCI to get the effective h1e (with core contributions)
        mc = mcscf.CASCI(mf, ncas, nelec_cas)
        mc.mo_coeff = mo_loc
        mc.frozen = n_frozen
        mc.verbose = verbose

        ncore = mc.ncore
        nocc_cas = nelec_cas[0]
        nvir_cas = ncas - nocc_cas
        mo_cas = mo_loc[:, ncore:ncore + ncas]
        h1e_cas, ecore = mc.h1e_for_cas()

        from pyscf import ao2mo as _ao2mo
        h2e_cas = _ao2mo.kernel(mol, mo_cas, compact=False).reshape(
            ncas, ncas, ncas, ncas)

        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
        import os as _os
        _os.makedirs(dmrg_scratch, exist_ok=True)

        nalpha, nbeta = nelec_cas
        n_elec = nalpha + nbeta
        spin = nalpha - nbeta

        # Run DMRG-CI in SU2 mode (spin-adapted, more efficient)
        su2_driver = DMRGDriver(scratch=dmrg_scratch, symm_type=SymmetryTypes.SU2,
                                n_threads=ncores, stack_mem=int(100e9))
        su2_driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin)
        mpo = su2_driver.get_qc_mpo(h1e_cas, h2e_cas, ecore=ecore, iprint=0)

        M1 = min(dmrg_maxM // 4, 100)
        M2 = min(dmrg_maxM // 2, 250)
        M3 = min(3 * dmrg_maxM // 4, 500)
        ket_su2 = su2_driver.get_random_mps(tag="KET_SU2", bond_dim=M1, nroots=1)
        e_dmrg = su2_driver.dmrg(
            mpo, ket_su2,
            bond_dims=[M1, M2, M3, dmrg_maxM, dmrg_maxM],
            noises=[1e-4, 1e-4, 1e-5, 1e-5, 0],
            thrds=[1e-5, 1e-5, 1e-6, 1e-7, dmrg_tol],
            tol=1e-6, n_sweeps=30, twosite_to_onesite=18, iprint=1)

        mc.e_tot = e_dmrg  # store for downstream use
        print(f'  E(DMRG-CI) = {e_dmrg:.10f}')

        # Convert SU2 MPS → SZ MPS for determinant-based amplitude extraction
        print('  Converting SU2 MPS to SZ for amplitude extraction...', flush=True)
        ket_sz = su2_driver.mps_change_to_sz(ket_su2, tag="KET_SZ", sz=spin)

        # Create SZ driver (same scratch dir) and load the converted MPS
        sz_driver = DMRGDriver(scratch=dmrg_scratch, symm_type=SymmetryTypes.SZ,
                               n_threads=ncores, stack_mem=int(100e9))
        sz_driver.initialize_system(n_sites=ncas, n_elec=n_elec, spin=spin)
        ket_sz_loaded = sz_driver.load_mps(tag="KET_SZ")

        t1_cas, t2_cas = extract_amplitudes_from_mps(
            sz_driver, ket_sz_loaded, ncas, nelec_cas, nocc_cas, nvir_cas, log)

        occ_cas_idx = np.arange(ncore, ncore + nocc_cas)
        vir_cas_idx = np.arange(ncore + nocc_cas, ncore + ncas)

        log.info('CAS amplitude norms: |t1|=%.4g  |t2|=%.4g',
                 np.linalg.norm(t1_cas), np.linalg.norm(t2_cas))

        # Convert occ_cas_idx from full-MO space to LMO-local space.
        occ_cas_idx = occ_cas_idx - n_frozen

    # ------------------------------------------------------------------
    # Stage 3: PAO + PNO construction
    # ------------------------------------------------------------------
    print('  Stage 3: PAO + PNO construction...', flush=True)

    s1e = mf.get_ovlp()
    mf_or_mc = mf if no_cas else mc
    C_pao, pao_domains, S_pao, F_pao = make_paos(
        mf_or_mc, C_lmo, T_CutDO=T_CutDO, s1e=s1e)

    nlmo = C_lmo.shape[1]
    domain_sizes = [len(pao_domains[i]) for i in range(nlmo)]
    print(f'  PAO domains: {nlmo} LMOs  |  sizes (AOs): '
          f'min={min(domain_sizes)}  max={max(domain_sizes)}  '
          f'avg={np.mean(domain_sizes):.1f}')
    print('    ' + '  '.join(f'LMO{i}:{s}' for i, s in enumerate(domain_sizes)))

    # CAS virtual MO coefficients in AO basis (for extended PNO construction)
    if no_cas:
        C_cas_vir = None
        nvir_cas_loc = 0
    else:
        C_cas_vir = mo_loc[:, vir_cas_idx]  # vir_cas_idx is in full-MO space
        nvir_cas_loc = len(vir_cas_idx)

    pno_spaces, _, _, _ = make_pnos(
        mf, C_lmo, C_pao, pao_domains, S_pao, F_pao,
        T_CutPNO=T_CutPNO, T_CutPairs=T_CutPairs,
        S_cut_domain=S_cut_domain,
        occ_cas_idx=occ_cas_idx, C_cas_vir=C_cas_vir,
        nvir_cas=nvir_cas_loc, s1e=s1e,
        verbose=verbose)

    # ------------------------------------------------------------------
    # Stage 4: Pair screening
    # ------------------------------------------------------------------
    mo_coeff_ref = mo_loc if no_cas else mc.mo_coeff
    (cas_pairs, strong_pairs, weak_pairs, negligible_pairs,
     e_lmp2_weak, e_lmp2_strong) = classify_pairs(
        pno_spaces, occ_cas_idx, vir_cas_idx,
        mo_coeff_ref, s1e,
        T_CutPairs=T_CutPairs, T_CutPairs_MP2=T_CutPairs_MP2,
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
    stage5_label = 'DLPNO-CCSD' if no_cas else 'DLPNO-TCCSD'
    print(f'  Stage 5: {stage5_label}...', flush=True)
    _t_ccsd_start = _time.time()

    mo_coeff_cas_arg = mo_loc if no_cas else mc.mo_coeff
    e_tccsd, t2_pno_all, t1_pno = run_lccsd(
        mf, C_lmo, pno_spaces,
        strong_pairs=strong_pairs,
        cas_pairs=cas_pairs,
        t1_cas=t1_cas, t2_cas=t2_cas,
        occ_cas_idx=occ_cas_idx, vir_cas_idx=vir_cas_idx,
        mo_coeff_cas=mo_coeff_cas_arg, s1e=s1e,
        conv_tol=ccsd_conv_tol, max_cycle=ccsd_max_cycle,
        ncores=ncores, C_pao=C_pao, verbose=verbose,
        _pool=_shared_pool)

    _t_ccsd = _time.time() - _t_ccsd_start
    log.info('E(%s) correlation = %.15g', stage5_label, e_tccsd)
    print(f'  Stage 5 wall time: {_t_ccsd:.2f} s', flush=True)

    # ------------------------------------------------------------------
    # Stage 6: External (T) correction
    # ------------------------------------------------------------------
    print('  Stage 6: (T) correction...', flush=True)
    _t_triples_start = _time.time()

    C_cas_vir_t = None if no_cas else mo_loc[:, vir_cas_idx]

    e_t = run_lccsd_t_ext(
        mf, C_lmo, pno_spaces,
        strong_pairs=strong_pairs,
        t2_pno_all=t2_pno_all,
        t1_pno=t1_pno,
        occ_cas_idx=occ_cas_idx,
        C_cas_vir=C_cas_vir_t,
        vir_cas_idx=vir_cas_idx,
        ncores=ncores, verbose=verbose,
        _pool=_shared_pool)

    _t_triples = _time.time() - _t_triples_start
    if _owns_pool and _shared_pool is not None:
        _shared_pool.shutdown(wait=True)
    log.info('E(T) external = %.15g', e_t)
    print(f'  Stage 6 wall time: {_t_triples:.2f} s', flush=True)

    # ------------------------------------------------------------------
    # Total energy
    # ------------------------------------------------------------------
    e_total = mf.e_tot + e_tccsd + e_lmp2_weak + e_t

    print(f'\n  Timings:  localization={_t_loc:.2f}s  '
          f'CCSD={_t_ccsd:.2f}s  (T)={_t_triples:.2f}s  '
          f'total={_t_loc + _t_ccsd + _t_triples:.2f}s', flush=True)

    return {
        'e_hf':         mf.e_tot,
        'e_dmrg':       e_dmrg,
        'e_lmp2_weak':  e_lmp2_weak,
        'e_tccsd':      e_tccsd,
        'e_t':          e_t,
        'e_total':      e_total,
        # Timings (wall time in seconds)
        't_localization': _t_loc,
        't_ccsd':         _t_ccsd,
        't_triples':      _t_triples,
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
        't1_pno':       t1_pno,
    }


def run_dlpno_ccsd_t(mf, frozen=0, **kwargs):
    """Convenience wrapper for plain DLPNO-CCSD(T) (no active space).

    Equivalent to ``run_dlpno_tccsd_t(mf, ncas=None, frozen=frozen, **kwargs)``.

    Args:
        mf: Converged DF-RHF (scf.RHF(mol).density_fit()).
        frozen (int or list): Frozen-core MOs excluded from correlation.
        **kwargs: All other keyword arguments forwarded to run_dlpno_tccsd_t.

    Returns:
        result (dict): Same as run_dlpno_tccsd_t.
    """
    return run_dlpno_tccsd_t(mf, ncas=None, nelec=None, frozen=frozen, **kwargs)
