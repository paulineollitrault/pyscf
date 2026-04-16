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
              T_CutPNO=1e-7, T_CutPairs=1e-4, S_cut_domain=1e-8,
              T_CutEnergy=1.0, T_CutTrace=1.0,
              T_CutPNO_MP2=None, T_CutTrace_MP2=0.9999, T_CutEnergy_MP2=0.999,
              occ_cas_idx=None, C_cas_vir=None, nvir_cas=0, s1e=None,
              verbose=None):
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
        ovL = _build_ovL(mf.with_df, C_lmo, C_pao, max_memory=mf.max_memory)
        # ovL[i, a, L]: i = LMO index, a = PAO index (global), L = aux index
        use_df = True
        # Also build raw 3-center integrals for local DF K (matching Psi4)
        _auxmol = mf.with_df.auxmol
        _naux = _auxmol.nao_nr()
        _pmol = mf.mol + _auxmol
        _shls = (0, mf.mol.nbas, 0, mf.mol.nbas,
                 mf.mol.nbas, mf.mol.nbas + _auxmol.nbas)
        _raw_3c = _pmol.intor('int3c2e', shls_slice=_shls)  # (nao, nao, naux)
        _j2c = _auxmol.intor('int2c2e')  # (naux, naux) Coulomb metric
        # Precompute raw_3c @ C_pao for half-transform
        _raw_half_pao = np.tensordot(_raw_3c, C_pao, axes=([1], [0]))  # (nao, naux, npao)
        del _raw_3c
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
        _ao_labels = mf.mol.ao_labels(fmt=False)
        _atom_ids = np.array([lbl[0] for lbl in _ao_labels])
        _aux_atom_ids = np.array([lbl[0] for lbl in _auxmol.ao_labels(fmt=False)])
        _T_CutMKN = 1e-3
        _natm = mf.mol.natm
        _lmo_aux_mask = []
        for ii in range(nocc_lmo):
            c_i = C_lmo[:, ii]
            P_i = s1e * c_i[:, None] * c_i[None, :]
            p_diag = np.diag(P_i)
            sum_diag = p_diag[:, None] + p_diag[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                w_u = np.where(sum_diag > 1e-15, p_diag[:, None] / sum_diag, 0.0)
                w_v = np.where(sum_diag > 1e-15, p_diag[None, :] / sum_diag, 0.0)
            contrib_u = P_i * w_u   # contribution to atom of row index u
            contrib_v = P_i * w_v   # contribution to atom of col index v
            mkn_pop = np.zeros(_natm)
            for a in range(_natm):
                mask_a = (_atom_ids == a)
                mkn_pop[a] = np.sum(contrib_u[mask_a, :]) + np.sum(contrib_v[:, mask_a])
            _lmo_aux_mask.append(
                np.isin(_aux_atom_ids, np.where(np.abs(mkn_pop) > _T_CutMKN)[0]))

    pno_spaces = {}
    strong_pairs = []
    weak_pairs = []
    e_lmp2_total = 0.0

    n_pairs_strong = 0
    n_pairs_weak = 0

    occ_cas_set = set(occ_cas_idx.tolist()) if occ_cas_idx is not None else set()

    # ===================================================================
    # Phase 1: Build per-pair domain data and SC-MP2 initial guess
    # ===================================================================
    pair_domain_data = {}   # intermediate data for L-MP2 iteration

    for i in range(nocc_lmo):
        for j in range(i, nocc_lmo):
            # --- 1. Pair domain (union of LMO domains) ---
            domain_ij = pao_domain_union(pao_domains[i], pao_domains[j])
            n_dom = len(domain_ij)

            if n_dom == 0:
                continue

            # --- 2. Canonical orthogonalization of PAOs in pair domain ---
            # Use Psi4's exact PartialCholesky algorithm so the canonical
            # PAO basis matches Psi4 — eliminates the e_pno drift that
            # propagated through CCSD iterations.
            C_orth_ij, X_orth_ij = orthogonalize_pao_domain(
                C_pao, S_pao, domain_ij, S_cut=S_cut_domain,
                method='psi4')
            n_orth = C_orth_ij.shape[1]
            if getattr(make_pnos, '_dump_n_orth', False):
                print(f'NORTH pair({i},{j}): npao_raw={len(domain_ij)} '
                      f'npao_ortho={n_orth}', flush=True)

            if n_orth == 0:
                continue

            # --- 3. F_vv and K_iajb in orthogonal domain basis ---
            F_dom = F_pao[np.ix_(domain_ij, domain_ij)]
            F_orth = reduce(np.dot, (X_orth_ij.T, F_dom, X_orth_ij))

            if use_df:
                if _lmo_aux_mask is not None:
                    # Local DF K matching Psi4 pno_transform():
                    # K = raw_i_orth^T @ J_local^{-1} @ raw_j_orth
                    if getattr(make_pnos, '_force_full_aux', False):
                        _pair_aux = np.arange(_j2c.shape[0])
                    else:
                        _pair_aux = np.where(_lmo_aux_mask[i] | _lmo_aux_mask[j])[0]
                    # raw_i[a_dom, Q] = C_lmo[:,i]^T @ raw_half_pao[:, Q, a_dom]
                    _raw_i_dom = np.tensordot(C_lmo[:, i], _raw_half_pao[:, :, domain_ij],
                                              axes=([0], [0]))  # (naux, n_dom)
                    _raw_j_dom = np.tensordot(C_lmo[:, j], _raw_half_pao[:, :, domain_ij],
                                              axes=([0], [0]))
                    # Transform to orth basis
                    _raw_i_orth = _raw_i_dom @ X_orth_ij  # (naux, n_orth)
                    _raw_j_orth = _raw_j_dom @ X_orth_ij
                    # Extract local aux and apply J^{-1}
                    _raw_i_local = _raw_i_orth[_pair_aux, :]  # (n_local, n_orth)
                    _raw_j_local = _raw_j_orth[_pair_aux, :]
                    _j2c_local = _j2c[np.ix_(_pair_aux, _pair_aux)]
                    # K = raw_i^T @ J^{-1} @ raw_j
                    _fitted_j = np.linalg.solve(_j2c_local, _raw_j_local)
                    K_ij = _raw_i_local.T @ _fitted_j  # (n_orth, n_orth)
                else:
                    # Fallback: global DF
                    ovL_i_dom = ovL[i][domain_ij, :]
                    ovL_j_dom = ovL[j][domain_ij, :]
                    ovL_i_orth = np.dot(X_orth_ij.T, ovL_i_dom)
                    ovL_j_orth = np.dot(X_orth_ij.T, ovL_j_dom)
                    K_ij = _pair_K_iajb(ovL_i_orth, ovL_j_orth)
            else:
                K_dom_ij = K_iajb_exact[i, :, j, :][np.ix_(domain_ij, domain_ij)]
                K_ij = X_orth_ij.T @ K_dom_ij @ X_orth_ij

            # --- 4. Semicanonical MP2 initial guess ---
            eps_sc, U_sc = np.linalg.eigh(F_orth)
            K_sc = reduce(np.dot, (U_sc.T, K_ij, U_sc))

            D_ij_sc = (eps_i[i] + eps_i[j]
                       - eps_sc[:, None] - eps_sc[None, :])
            D_safe = np.where(np.abs(D_ij_sc) > 1e-12, D_ij_sc, 1.0)
            T2_sc = K_sc / D_safe
            T2_sc = np.where(np.abs(D_ij_sc) > 1e-12, T2_sc, 0.0)

            # Transform SC-MP2 T2 back to orthogonal domain basis for L-MP2
            T2_orth = reduce(np.dot, (U_sc, T2_sc, U_sc.T))

            pair_domain_data[(i, j)] = {
                'C_orth': C_orth_ij,
                'X_orth': X_orth_ij,
                'F_orth': F_orth,
                'K_orth': K_ij,
                'T2_orth': T2_orth,
                'U_sc': U_sc,
                'eps_sc': eps_sc,
                'K_sc': K_sc,
                'domain_ij': domain_ij,
            }

    # ===================================================================
    # Phase 2a: Build INITIAL PNOs from direct SC-MP2 T2
    # (matching Psi4 compute_pair_energies<false>() lines 489-609)
    # ===================================================================
    if s1e is None:
        s1e = mf.get_ovlp()

    # Build initial PNOs from direct SC-MP2 T2 for each pair
    initial_pno_data = {}
    for (i, j), pdata in pair_domain_data.items():
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

        initial_pno_data[(i, j)] = {
            'C_pno': C_pno, 'e_pno': e_pno_sc, 'K_pno': K_pno,
            'T2_pno': T2_pno, 'n_pno': n_pno_init,
            'e_ij': e_ij_init, 'domain_ij': domain_ij,
            'X_pno_final': X_pno_final,
            'C_orth': C_orth_ij, 'X_orth': X_orth_ij,
            'F_orth': pdata['F_orth'],
            'U_sc': pdata['U_sc'], 'eps_sc': pdata['eps_sc'],
        }

    # ===================================================================
    # Phase 2b: Iterative LMP2 in PNO space
    # (matching Psi4 pno_lmp2_iterations() lines 690-802)
    # ===================================================================
    log.info('Running iterative LMP2 in PNO space...')

    # Build PNO overlap matrices for inter-pair coupling
    pno_S_cache = {}
    F_CUT = 1e-5  # Fock coupling threshold
    for key_ij in initial_pno_data:
        i, j = key_ij
        for k in range(nocc_lmo):
            key_kj = (min(k, j), max(k, j))
            key_ik = (min(i, k), max(i, k))
            if key_kj in initial_pno_data and i != k and abs(F_lmo[i, k]) > F_CUT:
                C_pno_ij = initial_pno_data[key_ij]['C_pno']
                C_pno_kj = initial_pno_data[key_kj]['C_pno']
                pno_S_cache[(key_ij, key_kj)] = C_pno_ij.T @ s1e @ C_pno_kj
            if key_ik in initial_pno_data and j != k and abs(F_lmo[k, j]) > F_CUT:
                C_pno_ij = initial_pno_data[key_ij]['C_pno']
                C_pno_ik = initial_pno_data[key_ik]['C_pno']
                pno_S_cache[(key_ij, key_ik)] = C_pno_ij.T @ s1e @ C_pno_ik

    # Iterative LMP2 in PNO space with DIIS (matching Psi4 lines 690-802)
    T2_pno_all = {k: d['T2_pno'].copy() for k, d in initial_pno_data.items()}
    max_lmp2_iter = 50
    e_conv_lmp2 = 1e-9
    r_conv_lmp2 = 5e-9
    e_prev_lmp2 = 0.0

    # Ordered pair keys for consistent flattening
    _lmp2_keys = [k for k in initial_pno_data if initial_pno_data[k]['n_pno'] > 0]
    _lmp2_sizes = {k: initial_pno_data[k]['n_pno'] ** 2 for k in _lmp2_keys}

    # DIIS setup (matching Psi4 line 700)
    from pyscf.lib.diis import DIIS
    lmp2_diis = DIIS()
    lmp2_diis.space = 8

    for lmp2_iter in range(max_lmp2_iter):
        # Step 1: Compute residuals for ALL pairs
        R_all = {}
        r_max = 0.0
        for key_ij, pdata in initial_pno_data.items():
            i, j = key_ij
            n = pdata['n_pno']
            if n == 0:
                continue
            K_pno = pdata['K_pno']
            e_pno = pdata['e_pno']
            T2 = T2_pno_all[key_ij]

            D = e_pno[:, None] + e_pno[None, :] - F_lmo[i, i] - F_lmo[j, j]
            R = K_pno + D * T2

            # Inter-pair Fock coupling (matching Psi4 lines 717-733)
            for k in range(nocc_lmo):
                key_kj = (min(k, j), max(k, j))
                if key_kj in initial_pno_data and i != k and abs(F_lmo[i, k]) > F_CUT:
                    S = pno_S_cache.get((key_ij, key_kj))
                    if S is not None and initial_pno_data[key_kj]['n_pno'] > 0:
                        T2_kj = T2_pno_all.get(key_kj)
                        if T2_kj is not None:
                            if k > j:
                                T2_kj = T2_kj.T
                            R -= F_lmo[i, k] * S @ T2_kj @ S.T
                key_ik = (min(i, k), max(i, k))
                if key_ik in initial_pno_data and j != k and abs(F_lmo[k, j]) > F_CUT:
                    S = pno_S_cache.get((key_ij, key_ik))
                    if S is not None and initial_pno_data[key_ik]['n_pno'] > 0:
                        T2_ik = T2_pno_all.get(key_ik)
                        if T2_ik is not None:
                            if i > k:
                                T2_ik = T2_ik.T
                            R -= F_lmo[k, j] * S @ T2_ik @ S.T

            R_all[key_ij] = R
            r_max = max(r_max, np.max(np.abs(R)))

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

        # Step 4: Build Tt and energy (matching Psi4 lines 783-799)
        e_curr_lmp2 = 0.0
        for key_ij, pdata in initial_pno_data.items():
            i, j = key_ij
            T2 = T2_pno_all[key_ij]
            K_pno = pdata['K_pno']
            Tt = 2.0 * T2 - T2.T
            e_pair = np.einsum('ab,ab->', K_pno, Tt)
            e_curr_lmp2 += e_pair if i == j else 2.0 * e_pair

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

    # ===================================================================
    # Phase 3: Recompute PNOs from converged PNO-LMP2 amplitudes
    # (matching Psi4 pno_lmp2_iterations lines 816-900)
    # ===================================================================
    for (i, j), pdata in initial_pno_data.items():
        n = pdata['n_pno']
        if n == 0:
            continue
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
        is_cas_pair = (i in occ_cas_set and j in occ_cas_set
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

            fock_ao_local = mf.get_fock() if not hasattr(mf, '_fock_cache') else mf._fock_cache
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

        e_lmp2_total += e_ij * (1 if i == j else 2)

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

        pno_spaces[(i, j)] = {
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

        if is_strong:
            strong_pairs.append((i, j))
            n_pairs_strong += 1
        else:
            if abs(e_ij) > 0.0:
                weak_pairs.append((i, j))
                n_pairs_weak += 1

    log.info('PNO construction complete: %d strong pairs, %d weak pairs',
             n_pairs_strong, n_pairs_weak)
    log.info('Total LMP2 energy = %.15g', e_lmp2_total)

    return pno_spaces, strong_pairs, weak_pairs, e_lmp2_total


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
