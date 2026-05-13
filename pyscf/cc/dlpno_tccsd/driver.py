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
import os as _os
import time as _time
import numpy as np
from pyscf import mcscf
from pyscf.lib import logger

from pyscf.cc.dlpno_tccsd.dmrg_interface import extract_amplitudes_from_mps


def _malloc_trim():
    """Force glibc to release free chunks back to the OS. Without this,
    pymalloc / malloc keep large arenas pooled and RSS stays at the
    high-water mark even after refcounts drop."""
    import gc
    gc.collect()
    try:
        import ctypes as _ct
        _ct.CDLL('libc.so.6').malloc_trim(0)
    except Exception:
        pass


def _log_mem(label):
    """Print [MEM/<label>] RSS=X GiB tmpfs=Y GiB if DLPNO_MEM_PROBE=1."""
    if not _os.environ.get('DLPNO_MEM_PROBE'):
        return
    try:
        with open('/proc/self/status') as _f:
            _txt = _f.read()
        _rss = 0
        for _ln in _txt.splitlines():
            if _ln.startswith('VmRSS:'):
                _rss = int(_ln.split()[1]) // 1024  # MB
                break
        _st = _os.statvfs('/tmp')
        _tmpfs = (_st.f_blocks - _st.f_bfree) * _st.f_frsize // (1024 ** 2)
        print(f'  [MEM/{label}] RSS={_rss/1024:.2f} GiB  tmpfs={_tmpfs/1024:.2f} GiB',
              flush=True)
    except Exception:
        pass


def _free_triples_caches():
    """Drop the function-attribute caches that the (T) orchestrator
    accumulates across triples (pair arena, t1 cache, globals).  Intended
    for cross-triple reuse within ONE run_dlpno_tccsd_t call; if left in
    place between calls the next molecule inherits 30+ GiB of stale flat
    arenas (water-49 → water-64 transition was carrying ~45 GiB of these).
    """
    try:
        from pyscf.cc.dlpno_tccsd import lccsd_t as _lt
        for func_name in ('_orch', '_run_triples_omp'):
            func = getattr(_lt, func_name, None)
            if func is None:
                continue
            for attr in ('_pair_arena', '_t1_cache', '_globals_cache'):
                try:
                    delattr(func, attr)
                except AttributeError:
                    pass
        # Per-LMO partner-set cache from _build_partners (id-keyed).
        if hasattr(_lt, '_partners_cache'):
            _lt._partners_cache.clear()
    except Exception as _e:
        print(f'  [_free_triples_caches] failed: {_e}', flush=True)
    _malloc_trim()


def _free_ccsd_plan_caches():
    """Drop the cycle-invariant plan caches that residual / t1_residual
    functions accumulate during cycle 1 and reuse for cycles 2+.

    These are reachable only via function attributes set as a perf cache;
    after Stage 5 returns the converged amplitudes, nothing else needs
    them — Stage 6 (T) and the public-energy reduction don't touch the
    residual plans. Clearing them between Stage 5 and Stage 6 frees
    O(N²)-by-pair-count memory that would otherwise stay resident through
    the rest of the calculation.
    """
    import gc
    try:
        from pyscf.cc.dlpno_tccsd import residual as _r
        from pyscf.cc.dlpno_tccsd import lccsd as _l
        targets = [
            (_r.compute_G_term_batched, '_plan_cache'),
            (_r.build_D_tilde_batched, '_plan_cache'),
            (_r.compute_C_tilde_batched, '_plan_cache'),
            (_r.compute_B_E_batched, '_plan_cache'),
            (_r.compute_CD_terms_batched, '_plan_cache'),
            (_l._compute_t1_residual, '_per_kl_plan_cache'),
            (_l._compute_t1_residual, '_per_kl_batched_scratch'),
        ]
        for func, attr in targets:
            if hasattr(func, attr):
                # Always delattr — dict.clear() leaves the attr as an
                # empty dict, but downstream code re-uses
                # `_per_kl_batched_scratch['num_threads']` and similar
                # without a None / empty-dict guard, which would KeyError.
                try:
                    delattr(func, attr)
                except AttributeError:
                    pass
    except Exception as _e:
        print(f'  [_free_ccsd_plan_caches] failed: {_e}', flush=True)
    _malloc_trim()


def _walk_bytes(obj, _seen=None, _depth=0, _max_depth=4):
    """Sum nbytes of numpy arrays reachable from obj (best-effort)."""
    if _seen is None:
        _seen = set()
    if id(obj) in _seen or _depth > _max_depth:
        return 0
    _seen.add(id(obj))
    if hasattr(obj, 'nbytes') and hasattr(obj, 'shape'):
        return int(obj.nbytes)
    if isinstance(obj, dict):
        return sum(_walk_bytes(v, _seen, _depth + 1, _max_depth)
                    for v in obj.values())
    if isinstance(obj, (list, tuple, set)):
        return sum(_walk_bytes(x, _seen, _depth + 1, _max_depth)
                    for x in obj)
    return 0


def _dump_plan_cache_sizes(label):
    """If DLPNO_MEM_PROBE=1, sum the nbytes of every function-attribute
    `_plan_cache` / `_per_kl_plan_cache` we know about and print."""
    if not _os.environ.get('DLPNO_MEM_PROBE'):
        return
    try:
        from pyscf.cc.dlpno_tccsd import residual as _r
        from pyscf.cc.dlpno_tccsd import lccsd as _l
        targets = [
            ('G_term', getattr(_r.compute_G_term_batched, '_plan_cache', None)),
            ('D_tilde', getattr(_r.build_D_tilde_batched, '_plan_cache', None)),
            ('C_tilde', getattr(_r.compute_C_tilde_batched, '_plan_cache', None)),
            ('B_E', getattr(_r.compute_B_E_batched, '_plan_cache', None)),
            ('CD', getattr(_r.compute_CD_terms_batched, '_plan_cache', None)),
            ('per_kl', getattr(_l._compute_t1_residual,
                                '_per_kl_plan_cache', None)),
            ('per_kl_scratch', getattr(_l._compute_t1_residual,
                                        '_per_kl_batched_scratch', None)),
        ]
        total = 0
        rows = []
        for name, c in targets:
            if c is None:
                rows.append((name, 0))
                continue
            sz = _walk_bytes(c)
            rows.append((name, sz))
            total += sz
        print(f'  [PLAN_CACHE/{label}] total={total/2**30:.2f} GiB  '
              + '  '.join(f'{n}={s/2**30:.2f}G' for n, s in rows),
              flush=True)
    except Exception as _e:
        print(f'  [PLAN_CACHE/{label}] dump failed: {_e}', flush=True)
from pyscf.cc.dlpno_tccsd.local_orbs import split_localize_orbitals, make_paos
from pyscf.cc.dlpno_tccsd.pno import make_pnos
from pyscf.cc.dlpno_tccsd.screening import classify_pairs
from pyscf.cc.dlpno_tccsd.lccsd import run_lccsd
from pyscf.cc.dlpno_tccsd.lccsd_t import run_lccsd_t_ext, run_lccsd_t1_iterations


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
                      S_cut_domain=1e-8,
                      T_CutEnergy=0.97,
                      T_CutTrace=1.0,
                      cas_pno_proj_thresh=0.99,
                      ccsd_conv_tol=1e-7,
                      ccsd_max_cycle=50,
                      use_t1_iterations=False,
                      verbose=4,
                      _C_lmo_override=None,
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

    # Pool sizing: half of ncores for ThreadPool workers, the other half
    # available as BLAS threads. We then use threadpool_limits to clamp
    # BLAS to 1 thread inside parallel sections — total OS thread usage
    # ≈ pool_size, well under OpenBLAS' per-process region cap.
    # Setup phases (SCF, etc.) outside the parallel region keep the
    # default BLAS thread count, so full BLAS parallelism is available
    # there.
    from concurrent.futures import ThreadPoolExecutor
    _owns_pool = _pool is None
    if _owns_pool:
        # Pool size depends on BLAS config. If caller has pinned BLAS to 1
        # thread (OMP_NUM_THREADS=1, MKL_NUM_THREADS=1) — typical
        # apples-to-apples Psi4-equivalent setup — we can use ncores
        # workers for true parallel pairs. Otherwise BLAS may use up to
        # ncores threads internally per pair, so half-pool avoids
        # oversubscription.
        _omp_n = int(_os.environ.get('OMP_NUM_THREADS', '0') or 0)
        _mkl_n = int(_os.environ.get('MKL_NUM_THREADS', '0') or 0)
        _blas_pinned = (_omp_n == 1) and (_mkl_n == 1)
        if _blas_pinned:
            # Empirical sweet spot on EC2 64-core: pool=32 beats pool=64 by
            # ~2s on water-15 CCSD. Above ~32 workers, malloc contention +
            # brief GIL holds in numpy dispatch start to dominate. Cap at
            # 32 unless caller overrides via env DLPNO_POOL_MAX_WORKERS.
            _pool_cap = int(_os.environ.get('DLPNO_POOL_MAX_WORKERS', '32'))
            _n_pool = max(1, min(ncores, _pool_cap))
        else:
            _n_pool = max(1, ncores // 2)
        _shared_pool = (
            ThreadPoolExecutor(max_workers=_n_pool) if _n_pool > 1
            else None)
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
        if _C_lmo_override is not None:
            C_lmo = _C_lmo_override
            log.info('Using externally provided LMOs (override)')
        elif C_occ.shape[1] > 1:
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
    _log_mem('stage3_enter')
    print('  Stage 3: PAO + PNO construction...', flush=True)
    import time as _time_s3
    _t_s3_0 = _time_s3.perf_counter()

    s1e = mf.get_ovlp()
    mf_or_mc = mf if no_cas else mc
    # Use grid-based DOI to match Psi4 (Jiang Eq 58 PAO-based DOI gives
    # much larger domains and is not compatible with Psi4's truncation).
    _with_df = getattr(mf, 'with_df', None)
    _t_paos = _time_s3.perf_counter()
    C_pao, pao_domains, S_pao, F_pao, doi_iu = make_paos(
        mf_or_mc, C_lmo, T_CutDO=T_CutDO, s1e=s1e, with_df=_with_df,
        doi_method='grid')
    print(f'  [STAGE3-PROF] make_paos: {_time_s3.perf_counter() - _t_paos:.2f}s',
          flush=True)
    _malloc_trim()
    _log_mem('after_make_paos')

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

    _t_pnos = _time_s3.perf_counter()
    pno_spaces, _, _, _, e_mp2_prescreened = make_pnos(
        mf, C_lmo, C_pao, pao_domains, S_pao, F_pao,
        T_CutPNO=T_CutPNO, T_CutPairs=T_CutPairs,
        T_CutPairs_MP2=T_CutPairs_MP2,
        S_cut_domain=S_cut_domain,
        T_CutEnergy=T_CutEnergy, T_CutTrace=T_CutTrace,
        occ_cas_idx=occ_cas_idx, C_cas_vir=C_cas_vir,
        nvir_cas=nvir_cas_loc, s1e=s1e,
        _pool=_shared_pool,
        verbose=verbose)
    print(f'  [STAGE3-PROF] make_pnos: {_time_s3.perf_counter() - _t_pnos:.2f}s',
          flush=True)
    print(f'  [STAGE3-PROF] Stage 3 total: '
          f'{_time_s3.perf_counter() - _t_s3_0:.2f}s', flush=True)
    # Drop the dense AO-basis C_pno (nao × npno per pair) — it's a legacy
    # representation; the X_pno + pair_paos pair (PAO-domain basis) carries
    # the same information at ~1/13th the size for typical water clusters.
    # All production-path consumers prefer X_pno when available and only
    # fall back to C_pno for CAS pairs (where X_pno is None). See Psi4's
    # equivalent storage in DLPNO::pno_construction.
    _n_dropped = 0
    _bytes_dropped = 0
    for _k, _v in pno_spaces.items():
        if _v.get('is_cas_pair', False):
            continue  # CAS pairs need C_pno (X_pno is None)
        if _v.get('X_pno') is None or _v.get('pair_paos') is None:
            continue
        _cp = _v.get('C_pno')
        if _cp is not None and hasattr(_cp, 'nbytes'):
            _bytes_dropped += int(_cp.nbytes)
            _v['C_pno'] = None
            _n_dropped += 1
    if _n_dropped > 0:
        print(f'  [pno_spaces] dropped C_pno from {_n_dropped} pairs '
              f'({_bytes_dropped/2**30:.2f} GiB freed)', flush=True)

    # Stage 3 has consumed the dense DF tensor twice: once for SCF
    # (DF-RHF J/K) and once to build ovL via with_df.loop().  Stage 5 and
    # Stage 6 of the production path never read cderi: Stage 5 uses
    # cc_ints (built fresh by compute_cc_integrals_sparse, libcint
    # shell-by-shell) and Stage 6 uses the sparse-DF stack
    # (build_sparse_df_arrays).  The remaining with_df.loop() /
    # with_df.ao2mo() call sites in residual.py / lccsd.py are explicit
    # fallbacks only hit when _term2_precomputed is None / K_pno is
    # None (CAS pairs only).  Releasing the cderi backing now frees
    # ~50–100 GB at water-64 (the (naux × nao_pair) tensor) and avoids
    # holding it on /scratch disk for the rest of the calculation.
    if hasattr(mf, 'with_df') and mf.with_df is not None:
        try:
            _cderi = getattr(mf.with_df, '_cderi', None)
            if _cderi is not None:
                # Cache F_AO regardless of whether we release cderi — the
                # cost is tiny and it future-proofs Stage 5's get_fock.
                try:
                    mf._dlpno_fock_ao = mf.get_fock()
                except Exception as _ef:
                    print(f'  [with_df] cache F_AO failed: {_ef}', flush=True)
                # Releasing cderi was originally for the water-64 memory
                # budget.  But after the release, both Stage 5 (mf.get_fock,
                # with_df.get_naoaux) and Stage 6 ((T) DF tensors) trigger
                # cderi rebuilds — at water-34 the rebuild cost was ~47s
                # in CCSD and another ~50s in (T).  Default: keep cderi
                # alive (sits in RAM/page-cache, ~10 GB at water-49).
                # Set DLPNO_RELEASE_CDERI=1 to release for water-64+ runs
                # where the cderi exceeds RAM.
                _release_cderi = bool(int(_os.environ.get(
                    'DLPNO_RELEASE_CDERI', '0'))) if False else (
                    __import__('os').environ.get('DLPNO_RELEASE_CDERI', '0')
                    not in ('0', '', 'false', 'False'))
                if _release_cderi:
                    if hasattr(mf.with_df, 'reset'):
                        mf.with_df.reset()
                    else:
                        mf.with_df._cderi = None
                    print(f'  [with_df] released cderi backing '
                          f'(DLPNO_RELEASE_CDERI=1)', flush=True)
                else:
                    print(f'  [with_df] keeping cderi alive '
                          f'(set DLPNO_RELEASE_CDERI=1 to release)',
                          flush=True)
        except Exception as _e:
            print(f'  [with_df] cderi release failed: {_e}', flush=True)
    _malloc_trim()
    _log_mem('after_make_pnos')

    # ------------------------------------------------------------------
    # Stage 4: Pair screening
    # ------------------------------------------------------------------
    _t_s4 = _time_s3.perf_counter()
    mo_coeff_ref = mo_loc if no_cas else mc.mo_coeff
    (cas_pairs, strong_pairs, weak_pairs, negligible_pairs,
     e_lmp2_weak, e_lmp2_strong, e_lmp2_negligible) = classify_pairs(
        pno_spaces, occ_cas_idx, vir_cas_idx,
        mo_coeff_ref, s1e,
        T_CutPairs=T_CutPairs, T_CutPairs_MP2=T_CutPairs_MP2,
        verbose=verbose)
    print(f'  [STAGE4-PROF] classify_pairs: '
          f'{_time_s3.perf_counter() - _t_s4:.2f}s', flush=True)

    print(f'  Pairs: {len(cas_pairs)} CAS  {len(strong_pairs)} strong  '
          f'{len(weak_pairs)} weak  {len(negligible_pairs)} negligible')
    print(f'  Eliminated-pair SC-MP2 correction: {e_lmp2_negligible:.6e} Eh',
          flush=True)
    if strong_pairs:
        pno_counts = [pno_spaces[p]['n_pno'] for p in strong_pairs]
        print(f'  Strong-pair PNOs: min={min(pno_counts)}  '
              f'max={max(pno_counts)}  avg={np.mean(pno_counts):.1f}')

    # Filter pno_spaces to drop negligible pairs from CCSD (Psi4 algorithm:
    # eliminated pairs contribute only the static SC-MP2 term added back to
    # the final energy below). This shrinks the per-pair iteration set
    # across all downstream phases (cc_ints, S_pno_cache, C/D/G plans,
    # cycles) — fixing the N^2 vs Psi4 N^1.43 pair-count scaling gap.
    # Override with DLPNO_KEEP_NEGLIGIBLE=1 to restore old behavior.
    _drop_neg_env = not bool(int(
        _os.environ.get('DLPNO_KEEP_NEGLIGIBLE', '0')))
    if _drop_neg_env and negligible_pairs:
        _negl_set = set((min(p), max(p)) for p in negligible_pairs)
        pno_spaces = {k: v for k, v in pno_spaces.items()
                       if k not in _negl_set}
        print(f'  Dropped {len(_negl_set)} negligible pairs from CCSD '
              f'(kept {len(pno_spaces)} pairs).', flush=True)

    # ------------------------------------------------------------------
    # Stage 5: DLPNO-TCCSD
    # ------------------------------------------------------------------
    # Limit BLAS threads to 1 during the DLPNO loops: each ThreadPool
    # worker calls into BLAS for small matmuls; without this constraint
    # ncores workers × default 64 BLAS threads = N×64 thread requests,
    # severe oversubscription. With 1 BLAS thread per worker × ncores
    # workers, total thread requests == ncores ≤ physical cores.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        threadpool_limits = None

    stage5_label = 'DLPNO-CCSD' if no_cas else 'DLPNO-TCCSD'
    print(f'  Stage 5: {stage5_label}...', flush=True)
    _log_mem('stage5_enter')
    _t_ccsd_start = _time.time()

    mo_coeff_cas_arg = mo_loc if no_cas else mc.mo_coeff

    # Stage 5: CCSD with BLAS threads limited to 1 (workers do the parallelism).
    _blas_ctx = (threadpool_limits(limits=1, user_api='blas')
                 if threadpool_limits is not None and _shared_pool is not None
                 else None)
    if _blas_ctx is not None:
        _blas_ctx.__enter__()
    try:
        e_tccsd, t2_pno_all, t1_pno = run_lccsd(
            mf, C_lmo, pno_spaces,
            strong_pairs=strong_pairs,
            cas_pairs=cas_pairs,
            t1_cas=t1_cas, t2_cas=t2_cas,
            occ_cas_idx=occ_cas_idx, vir_cas_idx=vir_cas_idx,
            mo_coeff_cas=mo_coeff_cas_arg, s1e=s1e,
            conv_tol=ccsd_conv_tol, max_cycle=ccsd_max_cycle,
            ncores=ncores, C_pao=C_pao, verbose=verbose,
            negligible_pairs=negligible_pairs,
            _pool=_shared_pool)
    finally:
        if _blas_ctx is not None:
            _blas_ctx.__exit__(None, None, None)

    _t_ccsd = _time.time() - _t_ccsd_start
    log.info('E(%s) correlation = %.15g', stage5_label, e_tccsd)
    print(f'  Stage 5 wall time: {_t_ccsd:.2f} s', flush=True)
    _log_mem('stage5_exit')

    # Free cycle-invariant residual plan caches before (T): they're
    # reachable only via function attributes and (T) never touches them.
    _free_ccsd_plan_caches()
    _log_mem('after_free_ccsd_plans')

    # ------------------------------------------------------------------
    # Stage 6: External (T) correction. Same BLAS-thread limit as CCSD —
    # without it, ncores workers × 64 BLAS threads exceeds OpenBLAS'
    # internal memory-region cap and triggers BLAS allocation errors.
    # ------------------------------------------------------------------
    print('  Stage 6: (T) correction...', flush=True)
    _log_mem('stage6_enter')
    _t_triples_start = _time.time()

    C_cas_vir_t = None if no_cas else mo_loc[:, vir_cas_idx]

    _blas_ctx_t = (threadpool_limits(limits=1, user_api='blas')
                   if threadpool_limits is not None and _shared_pool is not None
                   else None)
    if _blas_ctx_t is not None:
        _blas_ctx_t.__enter__()
    try:
        if use_t1_iterations:
            e_t = run_lccsd_t1_iterations(
                mf, C_lmo, pno_spaces,
                strong_pairs=strong_pairs,
                t2_pno_all=t2_pno_all,
                t1_pno=t1_pno,
                occ_cas_idx=occ_cas_idx,
                C_cas_vir=C_cas_vir_t,
                T_CutTNO=1e-9,
                ncores=ncores, verbose=verbose,
                _pool=_shared_pool)
        else:
            e_t = run_lccsd_t_ext(
                mf, C_lmo, pno_spaces,
                strong_pairs=strong_pairs,
                t2_pno_all=t2_pno_all,
                t1_pno=t1_pno,
                occ_cas_idx=occ_cas_idx,
                C_cas_vir=C_cas_vir_t,
                vir_cas_idx=vir_cas_idx,
                negligible_pairs=negligible_pairs,
                weak_pairs=weak_pairs,
                C_pao=C_pao,
                doi_iu=doi_iu,
                ncores=ncores, verbose=verbose,
                _pool=_shared_pool)
    finally:
        if _blas_ctx_t is not None:
            _blas_ctx_t.__exit__(None, None, None)

    _t_triples = _time.time() - _t_triples_start
    if _owns_pool and _shared_pool is not None:
        _shared_pool.shutdown(wait=True)
    log.info('E(T) external = %.15g', e_t)
    print(f'  Stage 6 wall time: {_t_triples:.2f} s', flush=True)

    # ------------------------------------------------------------------
    # Total energy
    # ------------------------------------------------------------------
    # e_lmp2_negligible: static SC-MP2 correction from pairs eliminated at
    # screening (Psi4's "Eliminated Pair dE"). Was always implicitly
    # included before via running CCSD over negligibles; now added back
    # explicitly since we drop them from pno_spaces.
    # e_mp2_prescreened: SC-MP2 contribution from pairs eliminated at the
    # crude SC-MP2 prescreen step BEFORE LMP2 iteration (Psi4's
    # "Crude Prescreening" eliminated pairs).
    e_total = (mf.e_tot + e_tccsd + e_lmp2_weak + e_lmp2_negligible
               + e_mp2_prescreened + e_t)
    print(f'  Crude prescreen SC-MP2 correction: '
          f'{e_mp2_prescreened:.6e} Eh', flush=True)

    print(f'\n  Timings:  localization={_t_loc:.2f}s  '
          f'CCSD={_t_ccsd:.2f}s  (T)={_t_triples:.2f}s  '
          f'total={_t_loc + _t_ccsd + _t_triples:.2f}s', flush=True)

    # Drop cross-call function-attribute caches before returning to the
    # caller. Without this, the next system inherits ~30+ GiB of stale
    # flat pair arenas / t1 caches from the (T) orchestrator.
    _free_triples_caches()

    return {
        'e_hf':         mf.e_tot,
        'e_dmrg':       e_dmrg,
        'e_lmp2_weak':  e_lmp2_weak,
        'e_lmp2_negligible': e_lmp2_negligible,
        'e_mp2_prescreened': e_mp2_prescreened,
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


def count_frozen_core(mol):
    """Count frozen-core orbitals using ORCA's default frozen-core convention.

    ORCA does NOT freeze 1s for Li and Be — their 1s electrons are valence.
    For Na/Mg, only 1s is frozen (2s2p are valence).

    Args:
        mol: PySCF Mole object.

    Returns:
        int: Number of frozen-core orbitals.
    """
    # Number of frozen core orbitals per element (by atomic number)
    # Z=1-2: 0, Z=3-4: 0 (Li/Be 1s is valence),
    # Z=5-10: 1 (freeze 1s), Z=11-12: 1 (Na/Mg: freeze 1s only),
    # Z=13-18: 5 (Al-Ar: freeze 1s2s2p), Z=19-36: 9, etc.
    frozen_per_z = {
        1: 0, 2: 0,                                       # H, He
        3: 0, 4: 0,                                        # Li, Be
        5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1,              # B-Ne
        11: 1, 12: 1,                                       # Na, Mg
        13: 5, 14: 5, 15: 5, 16: 5, 17: 5, 18: 5,         # Al-Ar
        19: 9, 20: 9,                                       # K, Ca
        21: 9, 22: 9, 23: 9, 24: 9, 25: 9, 26: 9, 27: 9,  # Sc-Co
        28: 9, 29: 9, 30: 9,                                # Ni, Cu, Zn
        31: 14, 32: 14, 33: 14, 34: 14, 35: 14, 36: 14,   # Ga-Kr
    }
    nfrozen = 0
    for z in mol.atom_charges():
        nfrozen += frozen_per_z.get(int(z), 0)
    return nfrozen
