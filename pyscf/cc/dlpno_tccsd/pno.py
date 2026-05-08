"""
Pair Natural Orbital (PNO) construction for DLPNO-TCCSD(T).

Implements the PNO generation algorithm from Neese/Riplinger (JCP 2013,
Pinski 2015) as described in Jiang JCP 2024 and translated from the Psi4
DLPNO implementation (andyj10224/psi4 dlpno_ccsd_t_jan_24 branch).

Key steps per pair (i,j):
  1. Build pair domain = union(pao_domain[i], pao_domain[j])
  2. Orthogonalize PAOs within pair domain (canonical orthogonalization)
  3. Get DF integrals (ia|jb) in LMO/ortho-PAO basis via ovL from Ye's ERIS
  4. Semicanonical MP2: diagonalize F_vv in pair domain → semicanonical PAOs
  5. T2_ij = -K_ij / D_ij  (K = exchange integrals, D = Fock denominator)
  6. Pair density D_ij = (T2+T2.T) @ T2.T + h.c., diagonalize → PNOs
  7. Truncate PNOs by occupation number threshold T_CutPNO
  8. LMP2 energy e_ij, pair classification: strong / weak
  9. CAS pair classification: both i,j in CAS occ AND PNOs in CAS vir subspace

References:
    Neese et al., JCP 2009, 130, 114108
    Riplinger & Neese, JCP 2013, 138, 034106
    Pinski et al., JCP 2015, 143, 034108
    Jiang, JCP 2024 (Psi4 implementation)
    Ye & Berkelbach, JCTC 2024 (for ovL integral infrastructure)
"""

import os
import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.lib import logger

from pyscf.cc.dlpno_tccsd.local_orbs import (
    pao_domain_union, orthogonalize_pao_domain
)


# ---------------------------------------------------------------------------
# DF integral builder (ported from Ye's lnocc/lno.py _init_mp_df_eris)
# ---------------------------------------------------------------------------

def _build_ovL(with_df, C_occ, C_vir, max_memory=4000):
    """Build (occ,vir|L) three-index DF tensor.

    Returns ovL[i,a,L] of shape (nocc, nvir, naux).
    This is the core routine from Ye lnocc/lno.py _init_mp_df_eris, adapted
    to accept arbitrary occ/vir coefficient matrices.

    Args:
        with_df: PySCF DF object (mf.with_df).
        C_occ (np.ndarray): (nao, nocc). Occupied orbital coefficients.
        C_vir (np.ndarray): (nao, nvir). Virtual orbital coefficients.
        max_memory (int): Memory limit in MB.

    Returns:
        ovL (np.ndarray): (nocc, nvir, naux).
    """
    from pyscf.ao2mo import _ao2mo

    nao, nocc = C_occ.shape
    nvir = C_vir.shape[1]
    nmo = nocc + nvir
    naux = with_df.get_naoaux()

    mo = np.asarray(np.hstack((C_occ, C_vir)), order='F')
    ijslice = (0, nocc, nocc, nmo)

    ovL = np.empty((nocc, nvir, naux))
    buf = None
    p1 = 0
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]
        p0 = p1
        p1 = p0 + nL
        buf = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
        ovL[:, :, p0:p1] = buf.reshape(nL, nocc, nvir).transpose(1, 2, 0)
        Lpq = None

    return ovL




def _pair_K_iajb(ovL_i, ovL_j):
    """Compute exchange integral K_iajb = (ia|jb) from DF 3-index tensors.

    Args:
        ovL_i (np.ndarray): (nvir_i, naux) — row i of ovL[i,:,:]
        ovL_j (np.ndarray): (nvir_j, naux) — row j of ovL[j,:,:]

    Returns:
        K (np.ndarray): (nvir_i, nvir_j) — K[a,b] = (ia|jb)
    """
    return np.dot(ovL_i, ovL_j.T)   # (nvir_i, nvir_j)



def _iterative_lmp2(pair_data, F_lmo, s1e, nocc_lmo,
                    max_iter=50, e_conv=1e-7, r_conv=5e-7,
                    fock_cutoff=1e-5, log=None):
    """Run iterative local MP2 with inter-pair occupied Fock coupling.

    This matches ORCA's "full local MP2" used for PNO generation (Pass-2).
    The inter-pair coupling via off-diagonal F_oo[i,k] produces ~2.4% larger
    correlation energy than semicanonical MP2, yielding bigger pair densities
    and more PNOs surviving the T_CutPNO threshold.

    All amplitudes are stored and updated in each pair's SEMICANONICAL basis
    where the virtual Fock matrix is diagonal.  This makes the Jacobi
    preconditioner exact for the intra-pair part, ensuring stable convergence.
    Inter-pair coupling is projected between different pairs' SC bases via
    P_sc = U_sc_ij.T @ C_orth_ij.T @ S_ao @ C_orth_kj @ U_sc_kj.

    Args:
        pair_data (dict): key=(i,j), value=dict with:
            'U_sc': (n_orth, n_orth) orth->SC transformation
            'eps_sc': (n_orth,) SC orbital energies (eigenvalues of F_orth)
            'K_sc': (n_orth, n_orth) exchange integrals in SC basis
            'C_orth': (nao, n_orth) orthogonal PAO coefficients
            'T2_orth': (n_orth, n_orth) SC-MP2 initial guess (in orth basis)
        F_lmo (np.ndarray): (nocc, nocc) Fock matrix in LMO basis.
        s1e (np.ndarray): (nao, nao) AO overlap matrix.
        nocc_lmo (int): Number of occupied LMOs.
        max_iter (int): Max L-MP2 iterations.
        e_conv (float): Energy convergence tolerance.
        r_conv (float): Residual convergence tolerance.
        fock_cutoff (float): Skip F_oo[i,k] coupling if |F_oo[i,k]| < cutoff.
        log: Logger object.

    Returns:
        t2 (dict): Converged T2 amplitudes in ORTH basis, key=(i,j).
        e_lmp2 (float): Converged L-MP2 correlation energy.
    """
    keys = sorted(pair_data.keys())

    # Initialize T2 in SC basis from the SC-MP2 solution
    # T2_orth = U_sc @ T2_sc @ U_sc.T, so T2_sc = U_sc.T @ T2_orth @ U_sc
    t2_sc = {}
    for ij in keys:
        U = pair_data[ij]['U_sc']
        t2_sc[ij] = reduce(np.dot, (U.T, pair_data[ij]['T2_orth'], U))

    # Precompute SC-basis projection matrices between coupled pair domains
    # P_sc_{ij,kj} = U_sc_ij.T @ C_orth_ij.T @ S_ao @ C_orth_kj @ U_sc_kj
    proj_sc_cache = {}
    for ij in keys:
        i, j = ij
        C_ij = pair_data[ij]['C_orth']
        U_ij = pair_data[ij]['U_sc']
        SC_ij = U_ij.T @ C_ij.T @ s1e  # (n_sc_ij, nao)
        for k in range(nocc_lmo):
            if k != i and abs(F_lmo[i, k]) >= fock_cutoff:
                kj = (min(k, j), max(k, j))
                if kj in pair_data and (ij, kj) not in proj_sc_cache:
                    C_kj = pair_data[kj]['C_orth']
                    U_kj = pair_data[kj]['U_sc']
                    proj_sc_cache[(ij, kj)] = SC_ij @ C_kj @ U_kj
            if k != j and abs(F_lmo[k, j]) >= fock_cutoff:
                ik = (min(i, k), max(i, k))
                if ik in pair_data and (ij, ik) not in proj_sc_cache:
                    C_ik = pair_data[ik]['C_orth']
                    U_ik = pair_data[ik]['U_sc']
                    proj_sc_cache[(ij, ik)] = SC_ij @ C_ik @ U_ik

    # Precompute Jacobi preconditioner in SC basis:
    # D_ij[a,b] = eps_sc_a + eps_sc_b - F_ii - F_jj  (exact diagonal)
    D_all = {}
    for ij in keys:
        i, j = ij
        eps = pair_data[ij]['eps_sc']
        D = eps[:, None] + eps[None, :] - F_lmo[i, i] - F_lmo[j, j]
        D_all[ij] = np.where(np.abs(D) > 1e-12, D, 1.0)

    e_prev = 0.0
    for iteration in range(max_iter):
        e_lmp2 = 0.0
        r_max = 0.0

        for ij in keys:
            i, j = ij
            K_sc = pair_data[ij]['K_sc']
            T2_ij = t2_sc[ij]

            # Residual in SC basis (F_vv is diagonal = eps_sc):
            # R = K_sc + D*T2 - coupling
            # where D = eps_a + eps_b - eps_i - eps_j
            R = K_sc + D_all[ij] * T2_ij

            # Inter-pair coupling: -sum_k F_oo[i,k] * P @ T2_kj @ P.T
            for k in range(nocc_lmo):
                if k != i and abs(F_lmo[i, k]) >= fock_cutoff:
                    kj = (min(k, j), max(k, j))
                    if kj not in t2_sc:
                        continue
                    P = proj_sc_cache.get((ij, kj))
                    if P is None:
                        continue
                    T2_kj = t2_sc[kj]
                    if k > j:
                        T2_kj_used = T2_kj.T
                    else:
                        T2_kj_used = T2_kj
                    R -= F_lmo[i, k] * (P @ T2_kj_used @ P.T)

            # Inter-pair coupling: -sum_k F_oo[k,j] * P @ T2_ik @ P.T
            for k in range(nocc_lmo):
                if k != j and abs(F_lmo[k, j]) >= fock_cutoff:
                    ik = (min(i, k), max(i, k))
                    if ik not in t2_sc:
                        continue
                    P = proj_sc_cache.get((ij, ik))
                    if P is None:
                        continue
                    T2_ik = t2_sc[ik]
                    if i > k:
                        T2_ik_used = T2_ik.T
                    else:
                        T2_ik_used = T2_ik
                    R -= F_lmo[k, j] * (P @ T2_ik_used @ P.T)

            # Jacobi update: T2_new = T2 - R/D = -K/D + coupling/D
            t2_sc[ij] = T2_ij - R / D_all[ij]

            r_max = max(r_max, np.max(np.abs(R)))

            # Pair energy
            Tt = 2.0 * t2_sc[ij] - t2_sc[ij].T
            e_ij = np.einsum('ab,ab->', K_sc, Tt)
            e_lmp2 += e_ij * (1.0 if i == j else 2.0)

        dE = abs(e_lmp2 - e_prev)
        if log is not None and iteration % 2 == 0:
            log.info('LMP2-Iter=%3d: E_LMP2=%.12f  dE=%.1e  Rmax=%.1e',
                     iteration, e_lmp2, dE, r_max)
        if dE < e_conv and r_max < r_conv:
            if log is not None:
                log.info('LMP2-Iter=%3d: E_LMP2=%.12f  dE=%.1e  Rmax=%.1e => CONVERGED',
                         iteration, e_lmp2, dE, r_max)
            break
        e_prev = e_lmp2

    # Transform converged T2 back to orthogonal basis for downstream use
    t2_orth = {}
    for ij in keys:
        U = pair_data[ij]['U_sc']
        t2_orth[ij] = reduce(np.dot, (U, t2_sc[ij], U.T))

    return t2_orth, e_lmp2


# ---------------------------------------------------------------------------
# Main PNO construction
# ---------------------------------------------------------------------------

def make_pnos(mf, C_lmo, C_pao, pao_domains, S_pao, F_pao,
              T_CutPNO=1e-7, T_CutPairs=1e-4, T_CutPairs_MP2=1e-6,
              S_cut_domain=1e-8,
              T_CutEnergy=1.0, T_CutTrace=1.0,
              T_CutPNO_MP2=None, T_CutTrace_MP2=0.9999, T_CutEnergy_MP2=0.999,
              occ_cas_idx=None, C_cas_vir=None, nvir_cas=0, s1e=None,
              _pool=None, verbose=None):
    """Construct PNO spaces for all LMO pairs and compute LMP2 energies.

    For each pair (i,j) with i <= j:
      1. Build pair PAO domain = union(domain_i, domain_j)
      2. Orthogonalize PAOs within the pair domain (canonical orthogonalization)
      3. Build F_vv and K_iajb in the orthogonalized domain basis using DF
      4. Semicanonicalize: diagonalize F_vv → get diagonal orbital energies
      5. Compute semicanonical MP2 amplitudes T2 = -K / D
      6. Build pair density, diagonalize → PNOs (U_ij, n_ij)
      7. Truncate PNOs using three criteria (Jiang et al. 2024):
         a. Occupation: n_ij > T_CutPNO
         b. Energy: cumulative pair energy / total > T_CutEnergy
         c. Trace: cumulative occupation sum / total > T_CutTrace
         A PNO is kept if ANY criterion requires it.
      8. Compute LMP2 energy for this pair
      9. Classify pair as strong / weak based on |e_ij| vs T_CutPairs

    Args:
        mf: RHF object with mf.with_df set (density-fitted).
        C_lmo (np.ndarray): (nao, nocc_lmo). LMO coefficients.
        C_pao (np.ndarray): (nao, nao). PAO coefficients.
        pao_domains (list): pao_domains[i] = np.array of AO domain indices for LMO i.
        S_pao (np.ndarray): (nao, nao). PAO overlap.
        F_pao (np.ndarray): (nao, nao). Fock in PAO basis.
        T_CutPNO (float): PNO occupation truncation threshold.
        T_CutPairs (float): |e_ij| threshold: strong if > T_CutPairs, else weak.
        S_cut_domain (float): Eigenvalue cutoff for canonical orthogonalization.
        T_CutEnergy (float): Energy criterion: keep PNOs until cumulative
            pair energy ratio exceeds this (default 0.997, Jiang et al. 2024).
        T_CutTrace (float): Trace criterion: keep PNOs until cumulative
            occupation fraction exceeds this (default 0.999, Jiang et al. 2024).
        verbose: Verbosity.

    Returns:
        pno_spaces (dict): Key = (i,j) with i<=j. Value = dict with keys:
            'C_pno'  : (nao, n_pno_ij) PNO coefficients in AO basis
            'n_pno'  : (n_pno_ij,) PNO occupation numbers
            'e_pno'  : (n_pno_ij,) semicanonical PNO orbital energies
            'K_pno'  : (n_pno_ij, n_pno_ij) exchange integrals in PNO basis
            'T2_pno' : (n_pno_ij, n_pno_ij) MP2 amplitudes in PNO basis
            'e_mp2'  : float, LMP2 pair energy
        strong_pairs (list): List of (i,j) pairs with |e_ij| > T_CutPairs.
        weak_pairs (list): List of (i,j) pairs with 0 < |e_ij| <= T_CutPairs.
        e_lmp2 (float): Total LMP2 correlation energy (strong + weak pairs).
    """
    log = logger.new_logger(mf, verbose)

    nocc_lmo = C_lmo.shape[1]
    nao = C_pao.shape[0]

    # We need DF integrals in LMO/PAO basis.
    # Strategy: build global ovL[i,a,L] where 'a' runs over ALL PAOs.
    # This allows us to extract pair domains by slicing columns.
    # Memory: nocc_lmo * nao * naux doubles — may be large. For large systems,
    # implement blocked (occ,domain) approach. Here we use the simple incore path.
    if hasattr(mf, 'with_df') and mf.with_df is not None:
        log.info('Building LMO/PAO DF 3-index integrals...')
        import time as _t_pno_pre
        _pno_pre_prof = bool(int(os.environ.get('DLPNO_PNO_PRE_PROF', '0')))
        def _pmark_pre(label, t0):
            if _pno_pre_prof:
                print(f'  [PNO-PRE] {label}: '
                      f'{_t_pno_pre.perf_counter() - t0:.2f}s', flush=True)
        _t = _t_pno_pre.perf_counter()
        ovL = _build_ovL(mf.with_df, C_lmo, C_pao, max_memory=mf.max_memory)
        _pmark_pre('_build_ovL', _t)
        # ovL[i, a, L]: i = LMO index, a = PAO index (global), L = aux index
        use_df = True
        # Also build raw 3-center integrals for local DF K (matching Psi4)
        _t = _t_pno_pre.perf_counter()
        _auxmol = mf.with_df.auxmol
        _naux = _auxmol.nao_nr()
        _pmol = mf.mol + _auxmol
        _shls = (0, mf.mol.nbas, 0, mf.mol.nbas,
                 mf.mol.nbas, mf.mol.nbas + _auxmol.nbas)
        _raw_3c = _pmol.intor('int3c2e', shls_slice=_shls)  # (nao, nao, naux)
        _pmark_pre("intor('int3c2e')", _t)
        _t = _t_pno_pre.perf_counter()
        _j2c = _auxmol.intor('int2c2e')  # (naux, naux) Coulomb metric
        _pmark_pre("intor('int2c2e')", _t)
        # Precompute raw_3c @ C_pao for half-transform.
        # _raw_3c is (nao, nao, naux); we contract its axis-1 with C_pao
        # axis-0 → output (nao, naux, npao).  np.tensordot reduces to
        # ONE single-threaded BLAS dgemm here (~3 s on water-15) because
        # MKL is pinned to 1 thread by DLPNO_POOL_PIN_BLAS.  Split the
        # output's leading u-axis across the shared pool so each worker
        # processes a thin slab; this releases the GIL inside the BLAS
        # call and yields ~30× speedup on water-15.
        _t = _t_pno_pre.perf_counter()
        nao_loc = _raw_3c.shape[0]
        _naux_loc = _raw_3c.shape[2]
        _npao_loc = C_pao.shape[1]
        _raw_half_pao = np.empty((nao_loc, _naux_loc, _npao_loc),
                                  dtype=_raw_3c.dtype)
        if _pool is not None and nao_loc > 16:
            n_workers = 32
            chunk = max(1, (nao_loc + n_workers - 1) // n_workers)
            ranges = [(s, min(s + chunk, nao_loc))
                       for s in range(0, nao_loc, chunk)]
            def _slab_half(rng):
                s, e = rng
                # _raw_3c[s:e]: (n, nao, naux); contract axis 1 with C_pao
                # axis 0 → (n, naux, npao).
                _raw_half_pao[s:e] = np.tensordot(
                    _raw_3c[s:e], C_pao, axes=([1], [0]))
            list(_pool.map(_slab_half, ranges))
        else:
            _raw_half_pao[:] = np.tensordot(
                _raw_3c, C_pao, axes=([1], [0]))
        del _raw_3c
        _pmark_pre('tensordot raw_3c @ C_pao', _t)
        # Hoist the LMO contraction out of the per-pair Phase 1 loop.
        # _raw_lmo_pao[i, Q, b] = sum_u C_lmo[u, i] * _raw_half_pao[u, Q, b].
        # Same parallelisation: split the output occupied axis i across the
        # pool so each worker drives one BLAS call (much smaller than the
        # half-transform — typically ~0.3 s — but free win).
        _t = _t_pno_pre.perf_counter()
        _nocc_lmo_loc = C_lmo.shape[1]
        _raw_lmo_pao = np.empty((_nocc_lmo_loc, _naux_loc, _npao_loc),
                                 dtype=_raw_half_pao.dtype)
        if _pool is not None and _nocc_lmo_loc > 8:
            n_workers = 16
            chunk = max(1, (_nocc_lmo_loc + n_workers - 1) // n_workers)
            ranges = [(s, min(s + chunk, _nocc_lmo_loc))
                       for s in range(0, _nocc_lmo_loc, chunk)]
            def _slab_lmo(rng):
                s, e = rng
                # C_lmo[:, s:e] is (nao, n); contract its axis-0 with
                # _raw_half_pao axis-0 → (n, naux, npao).
                _raw_lmo_pao[s:e] = np.tensordot(
                    C_lmo[:, s:e], _raw_half_pao, axes=([0], [0]))
            list(_pool.map(_slab_lmo, ranges))
        else:
            _raw_lmo_pao[:] = np.tensordot(
                C_lmo, _raw_half_pao, axes=([0], [0]))
        _pmark_pre('tensordot C_lmo @ raw_half_pao', _t)
    else:
        log.info('Building LMO/PAO exact 4-index integrals (no density fitting)...')
        from pyscf import ao2mo as _ao2mo_mod
        nocc_lmo_tmp, npao_tmp = C_lmo.shape[1], C_pao.shape[1]
        nmo_tmp = nocc_lmo_tmp + npao_tmp
        mo_tmp = np.hstack((C_lmo, C_pao))
        eri_tmp = _ao2mo_mod.kernel(
            mf.mol, mo_tmp, compact=False).reshape(nmo_tmp, nmo_tmp, nmo_tmp, nmo_tmp)
        # K_iajb_exact[i, a, j, b] = (ia|jb)
        K_iajb_exact = eri_tmp[:nocc_lmo_tmp, nocc_lmo_tmp:, :nocc_lmo_tmp, nocc_lmo_tmp:]
        use_df = False

    # Fock matrix diagonal in LMO basis (needed for denominator)
    fock_ao = mf.get_fock()
    F_lmo = reduce(np.dot, (C_lmo.T, fock_ao, C_lmo))
    eps_i = F_lmo.diagonal().real   # (nocc_lmo,)

    # Per-LMO local aux domains (for local DF K in PNO construction).
    # Mulliken formula matching Psi4 dlpnobase.cc:667-704 exactly:
    # P_i[u,v] = C[u,i] * S[u,v] * C[v,i]
    # mkn_pop[atom_u] += p_uv * p_uu / (p_uu + p_vv)
    # mkn_pop[atom_v] += p_uv * p_vv / (p_uu + p_vv)
    # (asymmetric split based on diagonal weights, NOT a row sum)
    _lmo_aux_mask = None
    if use_df and s1e is not None:
        _t_mkn = _t_pno_pre.perf_counter()
        _ao_labels = mf.mol.ao_labels(fmt=False)
        _atom_ids = np.array([lbl[0] for lbl in _ao_labels])
        _aux_atom_ids = np.array([lbl[0] for lbl in _auxmol.ao_labels(fmt=False)])
        _T_CutMKN = 1e-3
        _natm = mf.mol.natm
        _atom_id_masks = [(_atom_ids == a) for a in range(_natm)]
        def _mkn_per_lmo(ii):
            c_i = C_lmo[:, ii]
            P_i = s1e * c_i[:, None] * c_i[None, :]
            p_diag = np.diag(P_i)
            sum_diag = p_diag[:, None] + p_diag[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                w_u = np.where(sum_diag > 1e-15,
                               p_diag[:, None] / sum_diag, 0.0)
                w_v = np.where(sum_diag > 1e-15,
                               p_diag[None, :] / sum_diag, 0.0)
            contrib_u = P_i * w_u
            contrib_v = P_i * w_v
            mkn_pop = np.empty(_natm)
            for a in range(_natm):
                m = _atom_id_masks[a]
                mkn_pop[a] = (np.sum(contrib_u[m, :])
                              + np.sum(contrib_v[:, m]))
            return np.isin(
                _aux_atom_ids,
                np.where(np.abs(mkn_pop) > _T_CutMKN)[0])
        if _pool is not None and nocc_lmo > 1:
            _lmo_aux_mask = list(_pool.map(_mkn_per_lmo, range(nocc_lmo)))
        else:
            _lmo_aux_mask = [_mkn_per_lmo(ii) for ii in range(nocc_lmo)]
        _pmark_pre('Mulliken aux loop', _t_mkn)

    pno_spaces = {}
    strong_pairs = []
    weak_pairs = []
    e_lmp2_total = 0.0

    n_pairs_strong = 0
    n_pairs_weak = 0

    occ_cas_set = set(occ_cas_idx.tolist()) if occ_cas_idx is not None else set()

    # ===================================================================
    # Phase 0 — Psi4-style dipole prescreen (matches dlpnobase.cc::
    # compute_dipole_pair_energies). Drops pairs whose dipole-bound MP2
    # estimate is below T_CutPairs_MP2 BEFORE the expensive Phase 1
    # (orthogonalization + per-pair DF + SC-MP2). Was previously applied
    # AFTER Phase 1+2a (crude prescreen on SC-MP2 e_ij), which on water-42
    # made Phase 1 process all 14196 pairs and then drop 76% of them.
    # ===================================================================
    _t_p0_start = _pno_time_p1.perf_counter() if False else None
    import time as _pno_time_p0
    _t_p0_start = _pno_time_p0.perf_counter()
    e_dipole_dropped = 0.0
    keep_pairs_set = None
    _do_dipole_prescreen = bool(int(os.environ.get(
        'DLPNO_DIPOLE_PRESCREEN', '1')))
    if _do_dipole_prescreen and use_df:
        from pyscf.cc.dlpno_tccsd.screening import compute_dipole_pair_energies
        # Compute dipole-bound pair-energy estimates. Drop only when the
        # *bound* (which is a true upper bound on |e_ij|) is below the
        # SC-MP2 prescreen threshold — guaranteeing surviving pairs are a
        # superset of those the post-Phase-2a prescreen would have kept.
        dipole_e, dipole_e_bound = compute_dipole_pair_energies(
            C_lmo, C_pao, mf.mol, F_lmo, pao_domains,
            S_pao, F_pao, with_df=mf.with_df)
        keep_pairs_set = set()
        for i in range(nocc_lmo):
            for j in range(i, nocc_lmo):
                if i == j or abs(dipole_e_bound[i, j]) >= T_CutPairs_MP2:
                    keep_pairs_set.add((i, j))
                else:
                    fac = 1.0 if i == j else 2.0
                    e_dipole_dropped += fac * dipole_e[i, j]
        _n_total = nocc_lmo * (nocc_lmo + 1) // 2
        _n_kept = len(keep_pairs_set)
        _pno_dbg = bool(int(os.environ.get('DLPNO_PNO_DBG', '0')))
        _pno_dbg and print(
            f'[PNO_DBG] Phase 0 dipole prescreen: '
            f'{_pno_time_p0.perf_counter() - _t_p0_start:.2f}s, '
            f'kept {_n_kept}/{_n_total} pairs '
            f'(e_dropped={e_dipole_dropped:.3e} Eh)', flush=True)

    # ===================================================================
    # Phase 1: Build per-pair domain data and SC-MP2 initial guess
    # (parallelized via _pool.map — each (i,j) pair is independent)
    # ===================================================================
    import time as _pno_time_p1
    _t_p1_start = _pno_time_p1.perf_counter()
    import threading as _p1_threading
    _p1_lock = _p1_threading.Lock()
    _p1_sub = {'orth': 0.0, 'df': 0.0, 'solve': 0.0, 'eigh': 0.0, 'mp2': 0.0}
    def _p1_add(k, v):
        with _p1_lock:
            _p1_sub[k] += v
    pair_domain_data = {}   # intermediate data for L-MP2 iteration

    def _phase1_one(ij):
        i, j = ij
        domain_ij = pao_domain_union(pao_domains[i], pao_domains[j])
        if len(domain_ij) == 0:
            return ij, None

        _t = _pno_time_p1.perf_counter()
        C_orth_ij, X_orth_ij = orthogonalize_pao_domain(
            C_pao, S_pao, domain_ij, S_cut=S_cut_domain, method='psi4')
        n_orth = C_orth_ij.shape[1]
        if getattr(make_pnos, '_dump_n_orth', False):
            print(f'NORTH pair({i},{j}): npao_raw={len(domain_ij)} '
                  f'npao_ortho={n_orth}', flush=True)
        if n_orth == 0:
            return ij, None
        _p1_add('orth', _pno_time_p1.perf_counter() - _t)

        F_dom = F_pao[np.ix_(domain_ij, domain_ij)]
        F_orth = reduce(np.dot, (X_orth_ij.T, F_dom, X_orth_ij))

        _t = _pno_time_p1.perf_counter()
        if use_df:
            if _lmo_aux_mask is not None:
                if getattr(make_pnos, '_force_full_aux', False):
                    _pair_aux = np.arange(_j2c.shape[0])
                else:
                    _pair_aux = np.where(_lmo_aux_mask[i] | _lmo_aux_mask[j])[0]
                # Slice the pre-contracted LMO×AUX×PAO tensor (cheap copy
                # of (naux, ndomain) per pair vs (nao, naux, ndomain) before).
                _raw_i_dom = _raw_lmo_pao[i][:, domain_ij]
                _raw_j_dom = _raw_lmo_pao[j][:, domain_ij]
                _raw_i_orth = _raw_i_dom @ X_orth_ij
                _raw_j_orth = _raw_j_dom @ X_orth_ij
                _raw_i_local = _raw_i_orth[_pair_aux, :]
                _raw_j_local = _raw_j_orth[_pair_aux, :]
                _j2c_local = _j2c[np.ix_(_pair_aux, _pair_aux)]
                _p1_add('df', _pno_time_p1.perf_counter() - _t)
                _t = _pno_time_p1.perf_counter()
                _fitted_j = np.linalg.solve(_j2c_local, _raw_j_local)
                _p1_add('solve', _pno_time_p1.perf_counter() - _t)
                K_ij = _raw_i_local.T @ _fitted_j
            else:
                ovL_i_dom = ovL[i][domain_ij, :]
                ovL_j_dom = ovL[j][domain_ij, :]
                ovL_i_orth = np.dot(X_orth_ij.T, ovL_i_dom)
                ovL_j_orth = np.dot(X_orth_ij.T, ovL_j_dom)
                K_ij = _pair_K_iajb(ovL_i_orth, ovL_j_orth)
                _p1_add('df', _pno_time_p1.perf_counter() - _t)
        else:
            K_dom_ij = K_iajb_exact[i, :, j, :][np.ix_(domain_ij, domain_ij)]
            K_ij = X_orth_ij.T @ K_dom_ij @ X_orth_ij
            _p1_add('df', _pno_time_p1.perf_counter() - _t)

        _t = _pno_time_p1.perf_counter()
        eps_sc, U_sc = np.linalg.eigh(F_orth)
        _p1_add('eigh', _pno_time_p1.perf_counter() - _t)
        _t = _pno_time_p1.perf_counter()
        K_sc = reduce(np.dot, (U_sc.T, K_ij, U_sc))
        D_ij_sc = (eps_i[i] + eps_i[j]
                   - eps_sc[:, None] - eps_sc[None, :])
        D_safe = np.where(np.abs(D_ij_sc) > 1e-12, D_ij_sc, 1.0)
        T2_sc = K_sc / D_safe
        T2_sc = np.where(np.abs(D_ij_sc) > 1e-12, T2_sc, 0.0)
        T2_orth = reduce(np.dot, (U_sc, T2_sc, U_sc.T))
        _p1_add('mp2', _pno_time_p1.perf_counter() - _t)

        return ij, {
            'C_orth': C_orth_ij, 'X_orth': X_orth_ij, 'F_orth': F_orth,
            'K_orth': K_ij, 'T2_orth': T2_orth,
            'U_sc': U_sc, 'eps_sc': eps_sc, 'K_sc': K_sc,
            'domain_ij': domain_ij,
        }

    if keep_pairs_set is not None:
        _p1_keys = sorted(keep_pairs_set)
    else:
        _p1_keys = [(i, j) for i in range(nocc_lmo)
                    for j in range(i, nocc_lmo)]
    _p1_iter = (_pool.map(_phase1_one, _p1_keys)
                if _pool is not None
                else (_phase1_one(k) for k in _p1_keys))
    for ij, data in _p1_iter:
        if data is not None:
            pair_domain_data[ij] = data

    _pno_dbg = bool(int(os.environ.get('DLPNO_PNO_DBG', '0')))
    _pno_dbg and print(f'[PNO_DBG] Phase 1: {_pno_time_p1.perf_counter() - _t_p1_start:.2f}s '
          f'CPU sum={sum(_p1_sub.values()):.1f}s '
          f'orth={_p1_sub["orth"]:.1f} df={_p1_sub["df"]:.1f} '
          f'solve={_p1_sub["solve"]:.1f} eigh={_p1_sub["eigh"]:.1f} '
          f'mp2={_p1_sub["mp2"]:.1f}', flush=True)
    _t_p1_collect = _pno_time_p1.perf_counter()
    _t_p2a_start = _pno_time_p1.perf_counter()
    # ===================================================================
    # Phase 2a: Build INITIAL PNOs from direct SC-MP2 T2
    # (matching Psi4 compute_pair_energies<false>() lines 489-609)
    # ===================================================================
    if s1e is None:
        s1e = mf.get_ovlp()

    # Build initial PNOs from direct SC-MP2 T2 for each pair (parallel)
    initial_pno_data = {}

    def _phase2a_one(ij_pdata):
        ij, pdata = ij_pdata
        i, j = ij
        U_sc = pdata['U_sc']
        eps_sc = pdata['eps_sc']
        K_sc = pdata['K_sc']
        K_ij = pdata['K_orth']
        domain_ij = pdata['domain_ij']
        C_orth_ij = pdata['C_orth']
        X_orth_ij = pdata['X_orth']

        # Direct SC-MP2 T2
        D_sc_ij = (eps_i[i] + eps_i[j] - eps_sc[:, None] - eps_sc[None, :])
        D_safe = np.where(np.abs(D_sc_ij) > 1e-12, D_sc_ij, 1.0)
        T2_sc_init = K_sc / D_safe
        T2_sc_init = np.where(np.abs(D_sc_ij) > 1e-12, T2_sc_init, 0.0)

        Tt_sc_init = 2.0 * T2_sc_init - T2_sc_init.T
        e_ij_init = np.einsum('ab,ab->', K_sc, Tt_sc_init)
        if getattr(make_pnos, '_dump_e_ij_init', False):
            print(f'EIJ_INIT pair({i},{j}): e_ij={e_ij_init:.10e} '
                  f'norm_K={np.linalg.norm(K_sc):.4e}', flush=True)

        # Pair density from direct SC-MP2
        D_pair = np.dot(Tt_sc_init, T2_sc_init.T) + np.dot(Tt_sc_init.T, T2_sc_init)
        if i == j:
            D_pair *= 0.5

        pno_occ_init, U_pno_init = np.linalg.eigh(D_pair)
        # Sort by VALUE descending (matching Psi4's descending diagonalize)
        order = np.argsort(pno_occ_init)[::-1]
        pno_occ_init = pno_occ_init[order]
        U_pno_init = U_pno_init[:, order]

        # Initial PNO selection: Psi4 uses three OR'd criteria (Psi4 ccsd.cc
        # lines 516-523): keep PNO if eigenvalue >= cutoff OR cumulative trace
        # < target OR cumulative energy < target. Eigenvalue-only is too
        # aggressive for off-diagonal pairs with flat eigenvalue spectra.
        _t_cut_mp2 = T_CutPNO_MP2 if T_CutPNO_MP2 is not None else T_CutPNO * 0.01
        _t_cut_mp2_ij = _t_cut_mp2 * 1e-3 if i == j else _t_cut_mp2
        n_sc = len(pno_occ_init)
        # Cumulative trace and energy criteria
        occ_total_init = np.sum(pno_occ_init)
        # K and Tt in pno basis (for energy cumulation)
        K_pno_init_full = U_pno_init.T @ K_sc @ U_pno_init
        Tt_pno_init_full = U_pno_init.T @ Tt_sc_init @ U_pno_init
        keep = np.zeros(n_sc, dtype=bool)
        e_pno_cum = 0.0
        occ_pno_cum = 0.0
        for a in range(n_sc):
            cond_occ = abs(pno_occ_init[a]) >= _t_cut_mp2_ij
            cond_trace = (occ_pno_cum / occ_total_init < T_CutTrace_MP2
                          if abs(occ_total_init) > 1e-15 and T_CutTrace_MP2 < 1.0
                          else False)
            cond_energy = (abs(e_pno_cum) < T_CutEnergy_MP2 * abs(e_ij_init)
                           if abs(e_ij_init) > 1e-15 and T_CutEnergy_MP2 < 1.0
                           else False)
            if cond_occ or cond_trace or cond_energy:
                keep[a] = True
                # Update cumulative quantities using submatrix [0..a]
                e_pno_cum = np.einsum(
                    'ab,ab->',
                    K_pno_init_full[:a+1, :a+1],
                    Tt_pno_init_full[:a+1, :a+1])
                occ_pno_cum += pno_occ_init[a]
        if not np.any(keep):
            keep[np.argmax(np.abs(pno_occ_init))] = True
        n_pno_init = int(np.sum(keep))
        if getattr(make_pnos, '_debug_npno_init', False):
            print(f'NPNO_INIT pair({i},{j}): n_sc={n_sc} n_pno_init={n_pno_init}', flush=True)

        U_pno_kept = U_pno_init[:, keep]
        # X_pno in SC basis: columns of U_pno that are kept
        # Transform to orthogonal PAO basis: X_orth @ U_sc @ U_pno_kept
        X_pno_sc = U_sc @ U_pno_kept  # SC → PNO transform

        # K and T2 in initial PNO basis
        K_pno = U_pno_kept.T @ K_sc @ U_pno_kept
        T2_pno = U_pno_kept.T @ T2_sc_init @ U_pno_kept
        # PNO orbital energies (diagonal of F_pno)
        F_sc = np.diag(eps_sc)
        F_pno = U_pno_kept.T @ F_sc @ U_pno_kept
        e_pno = np.diag(F_pno)
        # Semicanonicalize PNOs (Psi4 convention: descending eigenvalues)
        e_pno_sc, V_pno = np.linalg.eigh(F_pno)
        e_pno_sc = e_pno_sc[::-1]
        V_pno = V_pno[:, ::-1]
        # Apply semicanonicalization
        X_pno_final = X_pno_sc @ V_pno
        K_pno = V_pno.T @ K_pno @ V_pno
        T2_pno = V_pno.T @ T2_pno @ V_pno

        # C_pno in AO basis
        C_pno = C_orth_ij @ X_pno_final

        return ij, {
            'C_pno': C_pno, 'e_pno': e_pno_sc, 'K_pno': K_pno,
            'T2_pno': T2_pno, 'n_pno': n_pno_init,
            'e_ij': e_ij_init, 'domain_ij': domain_ij,
            'X_pno_final': X_pno_final,
            'C_orth': C_orth_ij, 'X_orth': X_orth_ij,
            'F_orth': pdata['F_orth'],
            'U_sc': pdata['U_sc'], 'eps_sc': pdata['eps_sc'],
        }

    _p2a_items = list(pair_domain_data.items())
    _p2a_iter = (_pool.map(_phase2a_one, _p2a_items)
                 if _pool is not None
                 else (_phase2a_one(x) for x in _p2a_items))
    for ij, val in _p2a_iter:
        initial_pno_data[ij] = val

    _pno_dbg = bool(int(os.environ.get('DLPNO_PNO_DBG', '0')))
    _pno_dbg and print(f'[PNO_DBG] Phase 2a: {_pno_time_p1.perf_counter() - _t_p2a_start:.2f}s',
          flush=True)

    # Crude prescreen (Psi4-style): drop pairs whose initial SC-MP2 |e_ij|
    # is below T_CutPairs_MP2 BEFORE the LMP2 iteration. Currently we
    # iterate ALL pairs through Phase 2b then classify+drop later in
    # driver.py. For water-15 that means iterating 2850 pairs when only
    # 984 would survive — and the LMP2-residual plan-build for the dropped
    # pairs alone takes ~25s.
    # Match Psi4 (mp2.cc:compute_pair_energies "Crude Prescreening Step"):
    # use SC-MP2 e_ij (which we already have) for early elimination.
    _prescreen_thresh = float(os.environ.get(
        'DLPNO_LMP2_PRESCREEN_THRESH', str(T_CutPairs_MP2)))
    _e_mp2_prescreened = 0.0
    if _prescreen_thresh > 0.0:
        _to_drop = []
        for _key, _val in initial_pno_data.items():
            _ii, _jj = _key
            _e = float(_val.get('e_ij', 0.0))
            if abs(_e) < _prescreen_thresh:
                _to_drop.append(_key)
                _fac = 1.0 if _ii == _jj else 2.0
                _e_mp2_prescreened += _fac * _e
        for _key in _to_drop:
            del initial_pno_data[_key]
        if _to_drop:
            _pno_dbg and print(
                f'[PNO_DBG] Crude prescreen: dropped {len(_to_drop)} pairs '
                f'(|e_ij|<{_prescreen_thresh:.1e}), kept {len(initial_pno_data)} '
                f'(SC-MP2 prescreen energy = {_e_mp2_prescreened:.6e} Eh)',
                flush=True)
    # ===================================================================
    # Phase 2b: Iterative LMP2 in PNO space
    # (matching Psi4 pno_lmp2_iterations() lines 690-802)
    # ===================================================================
    log.info('Running iterative LMP2 in PNO space...')
    import time as _pno_time
    _t_p2b_start = _pno_time.perf_counter()
    _t_p2b_residual = 0.0
    _t_p2b_jacobi = 0.0
    _t_p2b_diis = 0.0
    _t_p2b_energy = 0.0

    # Build PNO overlap matrices for inter-pair coupling.
    # Iterate F-neighbors only (not full nocc) — same set the LMP2
    # residual loop touches. This eliminates O(npairs * nocc) setup
    # scan that was matching every k to F_CUT for every pair.
    _t_spno_start = _pno_time.perf_counter()
    pno_S_cache = {}
    F_CUT = 1e-5  # Fock coupling threshold
    _F_neigh_pre = [
        [k for k in range(nocc_lmo)
         if k != ii and abs(F_lmo[ii, k]) > F_CUT]
        for ii in range(nocc_lmo)
    ]

    # Pre-compute s1e @ C_pno_k for all k once (reused across all pair_ij
    # overlap computations). This turns the 3-matmul per-pair pattern
    # `C_pno_ij.T @ s1e @ C_pno_kj` into a single 2-matmul `C_pno_ij.T @ Z_kj`
    # where Z_kj = s1e @ C_pno_kj, halving FLOPs and reducing memory traffic.
    # Skip empty-PNO pairs (npno=0) which would trigger MKL DGEMM LDA warnings.
    _s1e_C_cache = {k: s1e @ initial_pno_data[k]['C_pno']
                    for k in initial_pno_data
                    if initial_pno_data[k]['C_pno'].shape[1] > 0}

    def _spno_one(key_ij):
        i, j = key_ij
        C_pno_ij = initial_pno_data[key_ij]['C_pno']
        out = {}
        if C_pno_ij.shape[1] == 0:
            return out
        C_pno_ij_T = C_pno_ij.T
        for k in _F_neigh_pre[i]:
            key_kj = (min(k, j), max(k, j))
            Z_kj = _s1e_C_cache.get(key_kj)
            if Z_kj is not None:
                out[(key_ij, key_kj)] = C_pno_ij_T @ Z_kj
        for k in _F_neigh_pre[j]:
            key_ik = (min(i, k), max(i, k))
            Z_ik = _s1e_C_cache.get(key_ik)
            if Z_ik is not None:
                out[(key_ij, key_ik)] = C_pno_ij_T @ Z_ik
        return out

    _key_list = list(initial_pno_data.keys())
    if _pool is not None:
        _all_outs = list(_pool.map(_spno_one, _key_list))
    else:
        _all_outs = [_spno_one(k) for k in _key_list]
    for _o in _all_outs:
        pno_S_cache.update(_o)
    _t_spno_build = _pno_time.perf_counter() - _t_spno_start
    _pno_dbg = bool(int(os.environ.get('DLPNO_PNO_DBG', '0')))
    _pno_dbg and print(f'[PNO_DBG] pno_S_cache build: {_t_spno_build:.2f}s '
          f'({len(pno_S_cache)} entries)', flush=True)

    # Iterative LMP2 in PNO space with DIIS (matching Psi4 lines 690-802)
    T2_pno_all = {k: d['T2_pno'].copy() for k, d in initial_pno_data.items()}
    max_lmp2_iter = 50
    e_conv_lmp2 = 1e-9
    r_conv_lmp2 = 5e-9
    e_prev_lmp2 = 0.0

    # Precompute F-coupling neighbors per LMO.  The inner loop in the
    # residual previously ran over all nocc with a per-iter F_CUT check;
    # this scales O(N_pairs * nocc).  Psi4 iterates only the pair's LMO
    # neighborhood (lmopair_to_lmos_), giving O(N_pairs * nlmo_ij) — much
    # better scaling for large systems (water-10: nlmo_ij~11 vs nocc=50).
    _F_neighbors = [
        [k for k in range(nocc_lmo)
         if k != i and abs(F_lmo[i, k]) > F_CUT]
        for i in range(nocc_lmo)
    ]

    # Ordered pair keys for consistent flattening
    _lmp2_keys = [k for k in initial_pno_data if initial_pno_data[k]['n_pno'] > 0]
    _lmp2_sizes = {k: initial_pno_data[k]['n_pno'] ** 2 for k in _lmp2_keys}

    # ------------------------------------------------------------------
    # Batched-C plan for inter-pair F-coupling residual update.
    # Replaces the Python loop that calls S @ T2 @ S.T thousands of times
    # per LMP2 iteration. Plan-cached: tasks list built once, kernel called
    # per LMP2 iteration with current T2.
    # ------------------------------------------------------------------
    _use_c_lmp2_resid = True
    _c_resid_plan = None
    _t_plan_start = _pno_time.perf_counter()
    if _use_c_lmp2_resid and _lmp2_keys:
        import ctypes as _ct_lmp2
        from pyscf import lib as _lib_lmp2
        _libcc_lmp2 = _lib_lmp2.load_library('libcc')
        _libcc_lmp2.DLPNOlmp2_residual_batched.restype = None
        _libcc_lmp2.DLPNOlmp2_residual_batched.argtypes = [
            _ct_lmp2.c_void_p, _ct_lmp2.c_void_p, _ct_lmp2.c_void_p,
            _ct_lmp2.c_void_p, _ct_lmp2.c_void_p, _ct_lmp2.c_void_p,
            _ct_lmp2.c_void_p, _ct_lmp2.c_void_p, _ct_lmp2.c_void_p,
            _ct_lmp2.c_void_p, _ct_lmp2.c_void_p,
            _ct_lmp2.c_int, _ct_lmp2.c_size_t, _ct_lmp2.c_int,
        ]
        _t_plan_setup = _pno_time.perf_counter() - _t_plan_start

        _ordered_keys = list(_lmp2_keys)
        _pair_idx_lut = {k: p for p, k in enumerate(_ordered_keys)}
        N_p = len(_ordered_keys)
        n_pno_arr = np.array(
            [initial_pno_data[k]['n_pno'] for k in _ordered_keys],
            dtype=np.int32)
        _t_plan_keys = _pno_time.perf_counter() - _t_plan_start - _t_plan_setup

        # T2 flat layout: per-pair (n_pno, n_pno) concatenated.
        T2_offsets = np.empty(N_p + 1, dtype=np.int64)
        T2_offsets[0] = 0
        T2_offsets[1:] = np.cumsum(n_pno_arr.astype(np.int64) ** 2)
        T2_flat = np.empty(int(T2_offsets[-1]))
        R_flat = np.empty(int(T2_offsets[-1]))

        # Enumerate tasks per target pair, fill S_flat from pno_S_cache.
        _task_partner = []
        _task_F = []
        _task_S_off = []
        _task_T = []
        _S_chunks = []      # collect S arrays in task order
        _S_offsets_per_task = []
        _s_total = 0
        n_tasks_per_pair = np.zeros(N_p, dtype=np.int64)

        for p, key_ij in enumerate(_ordered_keys):
            i_lmo, j_lmo = key_ij
            n_p_target = int(n_pno_arr[p])
            # First condition: F[i,k] > FCUT → use S(ij, kj) and T2[kj]
            for k in _F_neighbors[i_lmo]:
                key_kj = (min(k, j_lmo), max(k, j_lmo))
                if key_kj not in _pair_idx_lut:
                    continue
                p_partner = _pair_idx_lut[key_kj]
                if int(n_pno_arr[p_partner]) == 0:
                    continue
                S = pno_S_cache.get((key_ij, key_kj))
                if S is None:
                    continue
                _S_chunks.append(np.ascontiguousarray(S).ravel())
                _S_offsets_per_task.append(_s_total)
                _s_total += S.size
                _task_partner.append(p_partner)
                _task_F.append(F_lmo[i_lmo, k])
                _task_T.append(1 if k > j_lmo else 0)
                n_tasks_per_pair[p] += 1
            # Second condition: F[k,j] > FCUT → use S(ij, ik) and T2[ik]
            for k in _F_neighbors[j_lmo]:
                key_ik = (min(i_lmo, k), max(i_lmo, k))
                if key_ik not in _pair_idx_lut:
                    continue
                p_partner = _pair_idx_lut[key_ik]
                if int(n_pno_arr[p_partner]) == 0:
                    continue
                S = pno_S_cache.get((key_ij, key_ik))
                if S is None:
                    continue
                _S_chunks.append(np.ascontiguousarray(S).ravel())
                _S_offsets_per_task.append(_s_total)
                _s_total += S.size
                _task_partner.append(p_partner)
                _task_F.append(F_lmo[k, j_lmo])
                _task_T.append(1 if i_lmo > k else 0)
                n_tasks_per_pair[p] += 1

        _t_plan_enum = (_pno_time.perf_counter() - _t_plan_start
                        - _t_plan_setup - _t_plan_keys)

        target_task_starts = np.empty(N_p + 1, dtype=np.int64)
        target_task_starts[0] = 0
        target_task_starts[1:] = np.cumsum(n_tasks_per_pair)

        # Vectorised S_flat assembly: np.concatenate is C-implemented and
        # much faster than per-chunk slice assignment in a Python loop.
        if _S_chunks:
            S_flat = np.concatenate(_S_chunks)
        else:
            S_flat = np.empty(_s_total)
        _t_plan_sflat = (_pno_time.perf_counter() - _t_plan_start
                         - _t_plan_setup - _t_plan_keys - _t_plan_enum)

        # Per-pair K_pno + D constants for fast per-iter R0 build.
        K_pno_flat = np.empty(int(T2_offsets[-1]))
        D_flat = np.empty(int(T2_offsets[-1]))
        for p, key_ij in enumerate(_ordered_keys):
            i_l, j_l = key_ij
            pdata_ = initial_pno_data[key_ij]
            n_p = int(n_pno_arr[p])
            e_p = pdata_['e_pno']
            D = e_p[:, None] + e_p[None, :] - F_lmo[i_l, i_l] - F_lmo[j_l, j_l]
            K_pno_flat[T2_offsets[p]:T2_offsets[p + 1]] = pdata_['K_pno'].ravel()
            D_flat[T2_offsets[p]:T2_offsets[p + 1]] = D.ravel()
        _t_plan_kpno = (_pno_time.perf_counter() - _t_plan_start
                        - _t_plan_setup - _t_plan_keys - _t_plan_enum
                        - _t_plan_sflat)

        _c_resid_plan = {
            'ordered_keys': _ordered_keys,
            'n_pno_arr': n_pno_arr,
            'T2_offsets': T2_offsets,
            'T2_flat': T2_flat,
            'R_flat': R_flat,
            'K_pno_flat': K_pno_flat,
            'D_flat': D_flat,
            'target_task_starts': target_task_starts,
            'task_partner_idx': np.array(_task_partner, dtype=np.int64),
            'task_F_coeff': np.array(_task_F, dtype=np.float64),
            'task_S_off': np.array(_S_offsets_per_task, dtype=np.int64),
            'task_transpose': np.array(_task_T, dtype=np.int8),
            'S_flat': S_flat,
            'max_n_pno': int(n_pno_arr.max()) if N_p > 0 else 0,
            'N_p': N_p,
            'n_threads': min(16, N_p) if N_p > 0 else 1,
        }
        _pno_dbg and print(
            f'[PNO_DBG] LMP2 residual C plan: {N_p} pairs, '
            f'{int(target_task_starts[-1])} tasks, '
            f'S_flat={_s_total*8/1024/1024:.1f} MB '
            f'(setup={_t_plan_setup*1000:.0f}ms '
            f'keys={_t_plan_keys*1000:.0f}ms '
            f'enum={_t_plan_enum*1000:.0f}ms '
            f'sflat={_t_plan_sflat*1000:.0f}ms '
            f'kpno={_t_plan_kpno*1000:.0f}ms)',
            flush=True)

    # DIIS setup (matching Psi4 line 700)
    from pyscf.lib.diis import DIIS
    lmp2_diis = DIIS()
    lmp2_diis.space = 8

    for lmp2_iter in range(max_lmp2_iter):
        # Step 1: Compute residuals for ALL pairs.
        _t0 = _pno_time.perf_counter()
        R_all = {}
        r_max = 0.0

        if _c_resid_plan is not None:
            # Pack T2 flat in plan order; build R0 = K_pno + D*T2 in flat.
            P = _c_resid_plan
            ordered_keys = P['ordered_keys']
            T2_flat = P['T2_flat']
            R_flat = P['R_flat']
            T2_offsets = P['T2_offsets']
            n_pno_arr = P['n_pno_arr']
            for p, key_ij in enumerate(ordered_keys):
                T2 = T2_pno_all[key_ij]
                T2_flat[T2_offsets[p]:T2_offsets[p + 1]] = T2.ravel()
            R_flat[:] = P['K_pno_flat'] + P['D_flat'] * T2_flat

            _libcc_lmp2.DLPNOlmp2_residual_batched(
                P['target_task_starts'].ctypes.data_as(_ct_lmp2.c_void_p),
                P['task_partner_idx'].ctypes.data_as(_ct_lmp2.c_void_p),
                P['task_F_coeff'].ctypes.data_as(_ct_lmp2.c_void_p),
                P['task_S_off'].ctypes.data_as(_ct_lmp2.c_void_p),
                P['task_transpose'].ctypes.data_as(_ct_lmp2.c_void_p),
                n_pno_arr.ctypes.data_as(_ct_lmp2.c_void_p),
                T2_flat.ctypes.data_as(_ct_lmp2.c_void_p),
                T2_offsets.ctypes.data_as(_ct_lmp2.c_void_p),
                P['S_flat'].ctypes.data_as(_ct_lmp2.c_void_p),
                R_flat.ctypes.data_as(_ct_lmp2.c_void_p),
                T2_offsets.ctypes.data_as(_ct_lmp2.c_void_p),
                int(P['max_n_pno']),
                int(P['N_p']),
                int(P['n_threads']),
            )

            for p, key_ij in enumerate(ordered_keys):
                n = int(n_pno_arr[p])
                R = R_flat[T2_offsets[p]:T2_offsets[p + 1]].reshape(n, n).copy()
                R_all[key_ij] = R
                rmx = float(np.max(np.abs(R)))
                if rmx > r_max:
                    r_max = rmx
        else:
            for key_ij, pdata in initial_pno_data.items():
                i, j = key_ij
                n = pdata['n_pno']
                if n == 0:
                    continue
                K_pno = pdata['K_pno']
                e_pno = pdata['e_pno']
                T2 = T2_pno_all[key_ij]

                D = (e_pno[:, None] + e_pno[None, :]
                     - F_lmo[i, i] - F_lmo[j, j])
                R = K_pno + D * T2

                for k in _F_neighbors[i]:
                    key_kj = (min(k, j), max(k, j))
                    if key_kj in initial_pno_data:
                        S = pno_S_cache.get((key_ij, key_kj))
                        if S is not None and initial_pno_data[key_kj]['n_pno'] > 0:
                            T2_kj = T2_pno_all.get(key_kj)
                            if T2_kj is not None:
                                if k > j:
                                    T2_kj = T2_kj.T
                                R -= F_lmo[i, k] * S @ T2_kj @ S.T
                for k in _F_neighbors[j]:
                    if k == j:
                        continue
                    key_ik = (min(i, k), max(i, k))
                    if key_ik in initial_pno_data:
                        S = pno_S_cache.get((key_ij, key_ik))
                        if S is not None and initial_pno_data[key_ik]['n_pno'] > 0:
                            T2_ik = T2_pno_all.get(key_ik)
                            if T2_ik is not None:
                                if i > k:
                                    T2_ik = T2_ik.T
                                R -= F_lmo[k, j] * S @ T2_ik @ S.T

                R_all[key_ij] = R
                r_max = max(r_max, np.max(np.abs(R)))

        _t_p2b_residual += _pno_time.perf_counter() - _t0
        _t0 = _pno_time.perf_counter()
        # Step 2: Jacobi update (matching Psi4 lines 757-767)
        for key_ij, pdata in initial_pno_data.items():
            n = pdata['n_pno']
            if n == 0:
                continue
            i, j = key_ij
            e_pno = pdata['e_pno']
            D = e_pno[:, None] + e_pno[None, :] - F_lmo[i, i] - F_lmo[j, j]
            D_safe = np.where(np.abs(D) > 1e-12, D, 1.0)
            T2_pno_all[key_ij] -= R_all[key_ij] / D_safe

        _t_p2b_jacobi += _pno_time.perf_counter() - _t0
        _t0 = _pno_time.perf_counter()
        # Step 3: DIIS extrapolation (matching Psi4 lines 769-781)
        t2_flat = np.concatenate([T2_pno_all[k].ravel() for k in _lmp2_keys])
        r_flat = np.concatenate([R_all[k].ravel() for k in _lmp2_keys])
        t2_flat = lmp2_diis.update(t2_flat, r_flat)
        # Unflatten
        offset = 0
        for k in _lmp2_keys:
            sz = _lmp2_sizes[k]
            n = initial_pno_data[k]['n_pno']
            T2_pno_all[k] = t2_flat[offset:offset + sz].reshape(n, n)
            offset += sz

        _t_p2b_diis += _pno_time.perf_counter() - _t0
        _t0 = _pno_time.perf_counter()
        # Step 4: Build Tt and energy (matching Psi4 lines 783-799)
        e_curr_lmp2 = 0.0
        for key_ij, pdata in initial_pno_data.items():
            i, j = key_ij
            T2 = T2_pno_all[key_ij]
            K_pno = pdata['K_pno']
            Tt = 2.0 * T2 - T2.T
            e_pair = np.einsum('ab,ab->', K_pno, Tt)
            e_curr_lmp2 += e_pair if i == j else 2.0 * e_pair

        _t_p2b_energy += _pno_time.perf_counter() - _t0
        dE = abs(e_curr_lmp2 - e_prev_lmp2)
        if lmp2_iter > 0:
            log.info('LMP2-Iter=%3d: E_LMP2=%.12f  dE=%.1e  Rmax=%.1e',
                     lmp2_iter, e_curr_lmp2, dE, r_max)
        if lmp2_iter > 0 and dE < e_conv_lmp2 and r_max < r_conv_lmp2:
            log.info('LMP2-Iter=%3d: E_LMP2=%.12f  dE=%.1e  Rmax=%.1e => CONVERGED',
                     lmp2_iter, e_curr_lmp2, dE, r_max)
            break
        e_prev_lmp2 = e_curr_lmp2

    log.info('L-MP2 correlation energy = %.15g', e_curr_lmp2)
    _t_p2b = _pno_time.perf_counter() - _t_p2b_start
    _pno_dbg = bool(int(os.environ.get('DLPNO_PNO_DBG', '0')))
    _pno_dbg and print(f'[PNO_DBG] Phase 2b: {_t_p2b:.2f}s '
          f'(residual={_t_p2b_residual:.2f} jacobi={_t_p2b_jacobi:.2f} '
          f'diis={_t_p2b_diis:.2f} energy={_t_p2b_energy:.2f})', flush=True)

    # ===================================================================
    # Phase 3: Recompute PNOs from converged PNO-LMP2 amplitudes
    # (matching Psi4 pno_lmp2_iterations lines 816-900)
    # ===================================================================
    _t_p3_start = _pno_time.perf_counter()
    # Parallelised via _pool.map — each pair is independent. Worker returns
    # (key, pno_spaces_entry_or_None, e_ij, is_strong_flag).  is_strong flag
    # also carries -1 for "neither strong nor weak" (n_pno==0 or e_ij==0).
    fock_ao_local_p3 = (mf._fock_cache if hasattr(mf, '_fock_cache')
                        else mf.get_fock())
    occ_cas_set_p3 = set(occ_cas_idx.tolist()) if occ_cas_idx is not None else set()

    def _phase3_one(item):
        (i, j), pdata = item
        n = pdata['n_pno']
        if n == 0:
            return (i, j), None, 0.0, -1
        K_pno = pdata['K_pno']
        T2 = T2_pno_all[(i, j)]
        Tt = 2.0 * T2 - T2.T

        # Pair density from converged PNO-LMP2 T2 (in PNO space, matching Psi4 line 831)
        D_pair = np.dot(Tt, T2.T) + np.dot(Tt.T, T2)
        if i == j:
            D_pair *= 0.5

        # Eigendecomposition in PNO space (matching Psi4 line 840)
        # Sort by VALUE descending (matching Psi4's 'descending' diagonalize)
        pno_occ, U_pno = np.linalg.eigh(D_pair)
        order = np.argsort(pno_occ)[::-1]  # value descending
        pno_occ = pno_occ[order]
        U_pno = U_pno[:, order]
        if getattr(make_pnos, '_debug_pno_occ', False):
            print(f'PNOOCC pair({i},{j}): n={len(pno_occ)} top: '
                  f'{pno_occ[:5].tolist()}', flush=True)

        # Total pair energy (matching Psi4 line 848)
        e_ij = np.einsum('ab,ab->', K_pno, Tt)

        domain_ij = pdata['domain_ij']
        C_orth_ij = pdata['C_orth']
        X_pno_old = pdata['X_pno_final']  # orth → old PNO

        # Transform K, Tt to new PNO basis for energy criterion (Psi4 lines 851-852)
        K_pno_new = reduce(np.dot, (U_pno.T, K_pno, U_pno))
        Tt_pno_new = reduce(np.dot, (U_pno.T, Tt, U_pno))

        # --- PNO selection (matching Psi4 lines 856-873) ---
        is_cas_pair = (i in occ_cas_set_p3 and j in occ_cas_set_p3
                       and C_cas_vir is not None and nvir_cas > 0)

        nvir_cas_local = 0

        T_CutPNO_ij = T_CutPNO * 1e-3 if i == j else T_CutPNO

        # Combined loop matching Psi4 exactly (lines 856-873):
        # Keep PNO if ANY of: occ >= threshold, trace < target, energy < target
        occ_total = np.sum(pno_occ)  # signed sum (Psi4 line 846)
        n_pno_total = len(pno_occ)
        e_pno_cum = 0.0
        occ_pno_cum = 0.0
        n_pno = 0
        _trace_dbg = (getattr(make_pnos, '_disable_debug', False)
                      and i == 0 and j == 1)
        for p in range(n_pno_total):
            cond_occ = abs(pno_occ[p]) >= T_CutPNO_ij
            cond_trace = (occ_pno_cum / occ_total < T_CutTrace
                          if abs(occ_total) > 1e-15 and T_CutTrace < 1.0
                          else False)
            cond_energy = (abs(e_pno_cum) < T_CutEnergy * abs(e_ij)
                           if abs(e_ij) > 1e-15 and T_CutEnergy < 1.0
                           else False)
            if _trace_dbg:
                print(f'  p={p} occ={pno_occ[p]:+.4e} '
                      f'occ_cum={occ_pno_cum:+.4e}/{occ_total:+.4e}={occ_pno_cum/max(abs(occ_total),1e-15):+.4f} '
                      f'e_cum={e_pno_cum:+.4e}/{e_ij:+.4e}={e_pno_cum/max(abs(e_ij),1e-15):+.4f} '
                      f'cond_o={cond_occ} cond_t={cond_trace} cond_e={cond_energy}',
                      flush=True)
            if cond_occ or cond_trace or cond_energy:
                # Update cumulative energy using submatrix [0..p] (Psi4 line 869)
                K_sub = K_pno_new[:p+1, :p+1]
                Tt_sub = Tt_pno_new[:p+1, :p+1]
                e_pno_cum = np.einsum('ab,ab->', K_sub, Tt_sub)
                occ_pno_cum += pno_occ[p]
                n_pno += 1
            # Once all three conditions fail, no more PNOs can be kept
            # (eigenvalues decrease, cumulative values only grow)

        n_pno = max(n_pno, 1)
        if getattr(make_pnos, '_debug_phase3', False):
            print(f'PHASE3 pair({i},{j}): n_total={n_pno_total} n_kept={n_pno} '
                  f'top5_occ={pno_occ[:5].tolist()} '
                  f'occ_total={occ_total:.4e} e_ij={e_ij:.4e}', flush=True)

        U_pno_kept = U_pno[:, :n_pno]
        n_pno_kept = pno_occ[:n_pno]

        # Semicanonicalize: F in PNO space (Psi4 line 828/886)
        # Psi4 sorts eigenvalues DESCENDING (canonicalizer uses descending)
        F_pno_old = reduce(np.dot, (X_pno_old.T, pdata['F_orth'], X_pno_old))
        F_pno_new = reduce(np.dot, (U_pno_kept.T, F_pno_old, U_pno_kept))
        e_pno_sc, V_pno = np.linalg.eigh(F_pno_new)
        # Reverse to match Psi4's descending order
        e_pno_sc = e_pno_sc[::-1]
        V_pno = V_pno[:, ::-1]
        U_pno_kept = np.dot(U_pno_kept, V_pno)

        # Get SC-space quantities for CAS pair handling
        U_sc = pdata.get('U_sc', np.eye(X_pno_old.shape[0]))
        eps_sc = pdata.get('eps_sc', np.zeros(X_pno_old.shape[0]))

        if is_cas_pair:
            nvir_cas_local = nvir_cas
            X_new_cas = np.dot(X_pno_old, U_pno_kept)
            C_ext_pno = np.dot(C_orth_ij, X_new_cas)

            if s1e is not None and C_ext_pno.shape[1] > 0:
                overlap = C_cas_vir.T @ (s1e @ C_ext_pno)
                C_ext_pno = C_ext_pno - C_cas_vir @ overlap
                col_norms = np.sqrt(np.maximum(
                    np.einsum('ip,ip->p', C_ext_pno, s1e @ C_ext_pno), 0.0))
                C_ext_pno = C_ext_pno[:, col_norms > 1e-8]
                if C_ext_pno.shape[1] > 0:
                    C_ext_pno /= col_norms[col_norms > 1e-8][None, :]
                    S_ext = C_ext_pno.T @ (s1e @ C_ext_pno)
                    sv, Vext = np.linalg.eigh(S_ext)
                    C_ext_pno = (C_ext_pno @ Vext[:, sv > 1e-8]
                                 / np.sqrt(sv[sv > 1e-8])[None, :])

            C_pno_ij = np.hstack([C_cas_vir, C_ext_pno])

            fock_ao_local = fock_ao_local_p3
            F_cas_vir = reduce(np.dot, (C_cas_vir.T, fock_ao_local, C_cas_vir))
            e_cas_vir = np.diag(F_cas_vir).real
            if C_ext_pno.shape[1] > 0:
                F_ext = C_ext_pno.T @ fock_ao_local @ C_ext_pno
                e_ext_sc, V_ext = np.linalg.eigh(F_ext)
                C_ext_pno = C_ext_pno @ V_ext
                C_pno_ij = np.hstack([C_cas_vir, C_ext_pno])
            else:
                e_ext_sc = np.array([])
            e_pno_sc = np.concatenate([e_cas_vir, e_ext_sc])
            n_pno_kept = np.ones(C_pno_ij.shape[1])

            K_pno = None
            T2_pno = None
        else:
            # New PNO in AO basis: C_orth @ X_pno_old @ U_pno_kept
            X_new = np.dot(X_pno_old, U_pno_kept)  # orth → new PNO
            C_pno_ij = np.dot(C_orth_ij, X_new)
            U_full = X_new  # for backward compat

            K_pno = reduce(np.dot, (U_pno_kept.T, pdata['K_pno'], U_pno_kept))
            T2_pno = reduce(np.dot, (U_pno_kept.T, T2, U_pno_kept))

        is_strong = abs(e_ij) > T_CutPairs

        # Psi4-style PAO-domain storage: X_pno = (|domain_ij|, npno) is the
        # transform from raw PAO domain to PNO basis, and pair_paos lists the
        # PAO AO indices that span the pair domain. Reconstruction:
        #     C_pno_ij == C_pao[:, pair_paos] @ X_pno
        # CAS pairs use the full nao basis (CAS virt extends beyond domain),
        # so X_pno / pair_paos are not defined for them.
        if is_cas_pair:
            X_pno_pair = None
            pair_paos = None
        else:
            X_pno_pair = pdata['X_orth'] @ X_new
            pair_paos = domain_ij

        entry = {
            'C_pno': C_pno_ij,
            'X_pno': X_pno_pair,
            'pair_paos': pair_paos,
            'n_pno': n_pno_kept,
            'e_pno': e_pno_sc,
            'K_pno': K_pno,
            'T2_pno': T2_pno,
            'e_mp2': e_ij,
            'domain_ij': domain_ij,
            'X_orth': pdata['X_orth'],
            'U_full': U_full if not is_cas_pair else None,
            'domain_idx': domain_ij,
            'is_cas_pair': is_cas_pair,
            'nvir_cas_local': nvir_cas_local,
        }
        # Status code: 1 = strong, 0 = weak (e_ij != 0), -1 = neither.
        status = 1 if is_strong else (0 if abs(e_ij) > 0.0 else -1)
        return (i, j), entry, e_ij, status

    _p3_iter = (_pool.map(_phase3_one, initial_pno_data.items())
                if _pool is not None
                else (_phase3_one(item) for item in initial_pno_data.items()))
    for key, entry, e_ij, status in _p3_iter:
        if entry is None:
            continue
        i, j = key
        pno_spaces[key] = entry
        e_lmp2_total += e_ij * (1 if i == j else 2)
        if status == 1:
            strong_pairs.append(key)
            n_pairs_strong += 1
        elif status == 0:
            weak_pairs.append(key)
            n_pairs_weak += 1

    # Add the SC-MP2 contribution from pairs eliminated by the crude
    # prescreen (analogous to Psi4's "Crude Prescreening (Eliminated)" step).
    # These pairs never went through the iterative LMP2 — their initial
    # SC-MP2 estimate is added to the total to preserve correctness.
    e_lmp2_total += _e_mp2_prescreened
    # Add the dipole-prescreen contribution (Phase 0 — pairs that never
    # reached Phase 1 because the dipole-bound estimate was below
    # T_CutPairs_MP2). Their dipole estimate is the closed-form approx
    # to e_ij and is added back to the total energy, exactly mirroring
    # Psi4's compute_dipole_pair_energies bookkeeping.
    e_lmp2_total += e_dipole_dropped
    log.info('PNO construction complete: %d strong pairs, %d weak pairs',
             n_pairs_strong, n_pairs_weak)
    log.info('Total LMP2 energy = %.15g (incl prescreen %.6e dipole %.6e)',
             e_lmp2_total, _e_mp2_prescreened, e_dipole_dropped)
    _pno_dbg = bool(int(os.environ.get('DLPNO_PNO_DBG', '0')))
    _pno_dbg and print(f'[PNO_DBG] Phase 3: {_pno_time.perf_counter() - _t_p3_start:.2f}s',
          flush=True)

    return (pno_spaces, strong_pairs, weak_pairs, e_lmp2_total,
            _e_mp2_prescreened)


def classify_cas_pairs(pno_spaces, strong_pairs, occ_cas_idx, vir_cas_idx,
                       cas_pno_proj_thresh=0.99):
    """Identify CAS pairs: both LMOs in CAS occ AND PNOs in CAS vir subspace.

    A pair (i,j) is a CAS pair if:
      (a) i ∈ occ_cas_idx and j ∈ occ_cas_idx
      (b) Every retained PNO can be expressed within the CAS virtual subspace:
          ||P_cas_vir @ pno_k|| > cas_pno_proj_thresh  for all k

    For CAS pairs, DMRG cluster amplitudes are injected into the CC equations
    and the LCCSD solve is skipped for these pairs.

    Args:
        pno_spaces (dict): From make_pnos.
        strong_pairs (list): Strong pair index list.
        occ_cas_idx (np.ndarray): CAS occupied orbital indices in full MO list.
        vir_cas_idx (np.ndarray): CAS virtual orbital indices in full MO list.
        cas_pno_proj_thresh (float): Projection norm threshold (default 0.99).

    Returns:
        cas_pairs (set): Set of (i,j) tuples that are CAS pairs.
    """
    occ_cas_set = set(occ_cas_idx.tolist())
    cas_pairs = set()

    for (i, j) in strong_pairs:
        # Criterion (a): both in CAS occupied
        if i not in occ_cas_set or j not in occ_cas_set:
            continue

        if (i, j) not in pno_spaces:
            continue

        # Criterion (b): PNOs lie within CAS virtual subspace
        # The PNOs are given in AO basis as C_pno[:, k].
        # The CAS virtual space is spanned by the columns C_cas_vir = mo_coeff[:, vir_cas_idx].
        # We check that each PNO is close to the CAS virtual span by computing
        # the projection norm ||C_cas_vir @ (C_cas_vir^T @ S @ pno_k)||_S / ||pno_k||_S
        # Since we don't have S here, we use a simpler overlap metric.
        # The check is deferred to the driver which has access to mo_coeff and S.
        # Here we set a placeholder: all (i,j) with both in CAS occ are CAS pairs
        # by default; the projection check happens in the driver.
        cas_pairs.add((i, j))

    return cas_pairs


def check_pno_in_cas_vir(C_pno_ij, mo_coeff, vir_cas_idx, s1e,
                          proj_thresh=0.99):
    """Check if all PNOs in C_pno_ij lie within the CAS virtual subspace.

    Args:
        C_pno_ij (np.ndarray): (nao, n_pno) PNO coefficients.
        mo_coeff (np.ndarray): Full MO coefficient matrix (nao, nmo).
        vir_cas_idx (np.ndarray): CAS virtual MO indices.
        s1e (np.ndarray): AO overlap matrix.
        proj_thresh (float): Minimum projection norm.

    Returns:
        bool: True if all PNOs are well within the CAS virtual subspace.
    """
    if len(vir_cas_idx) == 0:
        return False

    C_cas_vir = mo_coeff[:, vir_cas_idx]   # (nao, nvir_cas)

    # Overlap of CAS vir with each PNO: P[a,k] = C_cas_vir[:,a]^T @ S @ C_pno[:,k]
    SC_pno = np.dot(s1e, C_pno_ij)         # (nao, n_pno)
    # Norm of each PNO in S-metric
    pno_norms = np.einsum('pk,pk->k', C_pno_ij, SC_pno)   # (n_pno,)

    # Projection of each PNO onto CAS vir: sum_a |<a|S|k>|^2 = ||P_cas_vir k||^2
    SC_cas = np.dot(s1e, C_cas_vir)        # (nao, nvir_cas)
    proj = np.dot(C_cas_vir.T, SC_pno)    # (nvir_cas, n_pno): <a|S|k>
    proj_sq = np.einsum('ak,ak->k', proj, proj)  # (n_pno,) norm^2

    # Projection fraction
    frac = np.where(pno_norms > 1e-14, proj_sq / pno_norms, 0.0)

    return bool(np.all(frac > proj_thresh))
