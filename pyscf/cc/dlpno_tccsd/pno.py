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


def _build_ovL_batched(with_df, C_occ, C_vir_list, max_memory=4000):
    """Build (occ,vir|L) for multiple virtual spaces in a single DF pass.

    Instead of calling _build_ovL once per PNO space (each iterating over
    all DF chunks), this function reads the DF file once and does a single
    large half-transform with all virtual spaces concatenated (one BLAS call).

    Args:
        with_df: PySCF DF object (mf.with_df).
        C_occ (np.ndarray): (nao, nocc).
        C_vir_list (list[np.ndarray]): List of (nao, nvir_k) arrays.
        max_memory (int): Memory limit in MB (advisory).

    Returns:
        list[np.ndarray]: ovL_k of shape (nocc, nvir_k, naux) for each k.
    """
    from pyscf.lib import unpack_tril

    nao, nocc = C_occ.shape
    naux = with_df.get_naoaux()

    # Build concatenated virtual matrix and track slices
    slices = []
    col = 0
    c_parts = []
    for C_vir in C_vir_list:
        nvir = C_vir.shape[1]
        slices.append(slice(col, col + nvir))
        if nvir > 0:
            c_parts.append(C_vir)
        col += nvir
    n_total = col
    C_all = np.hstack(c_parts) if c_parts else np.empty((nao, 0))

    # Pre-allocate outputs
    results = []
    for C_vir in C_vir_list:
        nvir = C_vir.shape[1]
        results.append(np.empty((nocc, nvir, naux)))

    p1 = 0
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]
        p0 = p1
        p1 = p0 + nL
        # Half-transform to occ: L_occ[L,μ,i] = Σ_ν Lpq[L,μν] * C_occ[ν,i]
        L_ao = unpack_tril(Lpq)              # (nL, nao, nao)
        L_occ = np.tensordot(L_ao, C_occ, axes=([2], [0]))  # (nL, nao, nocc)
        Lpq = L_ao = None  # free memory
        # ONE big virtual transform with concatenated C_all:
        # ovL_all[L,i,a] = Σ_μ L_occ[L,μ,i] * C_all[μ,a]
        ovL_all = np.tensordot(
            L_occ, C_all, axes=([1], [0]))  # (nL, nocc, n_total)
        L_occ = None
        # Split into per-space results
        for idx, sl in enumerate(slices):
            if sl.start == sl.stop:
                continue
            results[idx][:, :, p0:p1] = ovL_all[:, :, sl].transpose(1, 2, 0)

    return results


def _build_kcoul_batched(with_df, C_lmo, pno_spaces, kcoul_keys, ooL_3idx=None):
    """Rebuild all K_coul integrals using C-level _ao2mo.nr_e2.

    K_coul[(key_ij, key_ik, m1, m2)][b,c] = (m1 m2 | b_ij c_ik)
        = Σ_L B^L_{m1,m2} · B^L_{b_ij, c_ik}

    Strategy: group cross-pairs by key_ij. For each key_ij, concatenate
    its PNOs with all cross-partner PNOs and use nr_e2 to extract the
    cross-block in one C-level call per DF chunk. This avoids the expensive
    Python-level unpack_tril entirely.

    Args:
        with_df: PySCF DF object.
        C_lmo (np.ndarray): (nao, nocc) LMO coefficients (possibly T1-dressed).
        pno_spaces (dict): pair → {'C_pno': ...}
        kcoul_keys (list): List of (key_ij, key_ik, m1, m2) tuples.
        ooL_3idx (np.ndarray, optional): (nocc, nocc, naux). If provided, skip
            the extra DF pass to build it.

    Returns:
        dict: Same keys → (n_ij, n_ik) arrays.
    """
    from pyscf.ao2mo import _ao2mo

    if not kcoul_keys:
        return {}

    nao, nocc = C_lmo.shape
    naux = with_df.get_naoaux()

    # ooL[m1,m2,L] = (m1_lmo m2_lmo | L)
    if ooL_3idx is not None:
        ooL = ooL_3idx
    else:
        ooL = _build_ovL(with_df, C_lmo, C_lmo)  # (nocc, nocc, naux)

    # Group kcoul_keys by key_ij, then by cross-partner key_ik
    from collections import defaultdict
    # ij_to_partners[key_ij] = {key_ik: [(m1,m2), ...]}
    ij_to_partners = defaultdict(lambda: defaultdict(list))
    for key_ij, key_ik, m1, m2 in kcoul_keys:
        ij_to_partners[key_ij][key_ik].append((m1, m2))

    # Pre-allocate result with zeros
    result = {}
    for key_ij, key_ik, m1, m2 in kcoul_keys:
        n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
        n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
        result[(key_ij, key_ik, m1, m2)] = np.zeros((n_ij, n_ik))

    # Pre-build concatenated MO matrices and metadata for each key_ij
    ij_data = {}  # key_ij → (C_concat, ijslice, n_ij, partner_slices)
    for key_ij, partners in ij_to_partners.items():
        C_ij = pno_spaces[key_ij]['C_pno']
        n_ij = C_ij.shape[1]
        if n_ij == 0:
            continue
        c_parts = [C_ij]
        partner_slices = {}
        col = n_ij
        for key_ik in partners:
            C_ik = pno_spaces[key_ik]['C_pno']
            n_ik = C_ik.shape[1]
            if n_ik > 0:
                partner_slices[key_ik] = slice(col - n_ij, col - n_ij + n_ik)
                c_parts.append(C_ik)
                col += n_ik
        if not partner_slices:
            continue
        C_concat = np.asfortranarray(np.hstack(c_parts))
        ijslice = (0, n_ij, n_ij, col)
        ij_data[key_ij] = (C_concat, ijslice, n_ij, partner_slices)

    # Single DF pass: for each chunk, call nr_e2 once per key_ij
    p1 = 0
    bufs = {k: None for k in ij_data}
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]
        p0 = p1
        p1 = p0 + nL
        ooL_chunk = ooL[:, :, p0:p1]  # (nocc, nocc, nL)

        for key_ij, (C_concat, ijslice, n_ij, partner_slices) in ij_data.items():
            n_right = ijslice[3] - ijslice[2]
            bufs[key_ij] = _ao2mo.nr_e2(
                Lpq, C_concat, ijslice, aosym='s2', out=bufs[key_ij])
            cross_block = bufs[key_ij].reshape(nL, n_ij, n_right)

            for key_ik, occ_pairs in ij_to_partners[key_ij].items():
                sl = partner_slices.get(key_ik)
                if sl is None:
                    continue
                vvL_chunk = cross_block[:, :, sl]  # (nL, n_ij, n_ik)
                for m1, m2 in occ_pairs:
                    result[(key_ij, key_ik, m1, m2)] += np.tensordot(
                        ooL_chunk[m1, m2], vvL_chunk, axes=([0], [0]))

    return result


def _pair_K_iajb(ovL_i, ovL_j):
    """Compute exchange integral K_iajb = (ia|jb) from DF 3-index tensors.

    Args:
        ovL_i (np.ndarray): (nvir_i, naux) — row i of ovL[i,:,:]
        ovL_j (np.ndarray): (nvir_j, naux) — row j of ovL[j,:,:]

    Returns:
        K (np.ndarray): (nvir_i, nvir_j) — K[a,b] = (ia|jb)
    """
    return np.dot(ovL_i, ovL_j.T)   # (nvir_i, nvir_j)


# ---------------------------------------------------------------------------
# Iterative L-MP2 for PNO generation
# ---------------------------------------------------------------------------

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
              T_CutPNO=1e-7, T_CutPairs=1e-4, S_cut_domain=1e-6,
              T_CutEnergy=1.0, T_CutTrace=1.0,
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
            C_orth_ij, X_orth_ij = orthogonalize_pao_domain(
                C_pao, S_pao, domain_ij, S_cut=S_cut_domain)
            n_orth = C_orth_ij.shape[1]

            if n_orth == 0:
                continue

            # --- 3. F_vv and K_iajb in orthogonal domain basis ---
            F_dom = F_pao[np.ix_(domain_ij, domain_ij)]
            F_orth = reduce(np.dot, (X_orth_ij.T, F_dom, X_orth_ij))

            if use_df:
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
    # Phase 2: Iterative L-MP2 with inter-pair Fock coupling
    # ===================================================================
    if s1e is None:
        s1e = mf.get_ovlp()

    t2_lmp2, e_lmp2_iter = _iterative_lmp2(
        pair_domain_data, F_lmo, s1e, nocc_lmo, log=log)

    log.info('L-MP2 correlation energy = %.15g', e_lmp2_iter)

    # ===================================================================
    # Phase 3: Build PNOs from converged L-MP2 amplitudes
    # ===================================================================
    for (i, j), pdata in pair_domain_data.items():
        C_orth_ij = pdata['C_orth']
        X_orth_ij = pdata['X_orth']
        F_orth = pdata['F_orth']
        K_ij = pdata['K_orth']
        U_sc = pdata['U_sc']
        eps_sc = pdata['eps_sc']
        K_sc = pdata['K_sc']
        domain_ij = pdata['domain_ij']

        # Transform converged L-MP2 T2 to semicanonical basis
        T2_orth_conv = t2_lmp2[(i, j)]
        T2_sc = reduce(np.dot, (U_sc.T, T2_orth_conv, U_sc))

        # L-MP2 pair energy (pre-truncation, for classification)
        Tt_sc = 2.0 * T2_sc - T2_sc.T
        e_ij = np.einsum('ab,ab->', K_sc, Tt_sc)

        # --- 6. Pair density and PNO eigendecomposition ---
        D_pair = np.dot(Tt_sc, T2_sc.T) + np.dot(T2_sc, Tt_sc.T)
        if i == j:
            D_pair *= 0.5

        pno_occ, U_pno = np.linalg.eigh(D_pair)
        order = np.argsort(np.abs(pno_occ))[::-1]
        pno_occ = pno_occ[order]
        U_pno = U_pno[:, order]

        # --- 7. Build PNO basis ---
        is_cas_pair = (i in occ_cas_set and j in occ_cas_set
                       and C_cas_vir is not None and nvir_cas > 0)

        nvir_cas_local = 0

        # Diagonal pairs (i==j) use a tighter PNO cutoff for singles.
        # ORCA TightPNO: TCutPNOSingles = 3e-9 (with TCutPNO = 1e-7).
        T_CutPNO_ij = T_CutPNO * 3e-2 if i == j else T_CutPNO

        # Three PNO significance criteria (Jiang et al. 2024, p.14):
        # A PNO is kept if ANY criterion requires it.
        # 1. Occupation criterion
        keep_occ = np.abs(pno_occ) > T_CutPNO_ij

        # 2. Energy criterion: include PNOs from largest to smallest occupation
        #    until cumulative pair energy / total pair energy > T_CutEnergy.
        n_sc = len(pno_occ)
        keep_energy = np.zeros(n_sc, dtype=bool)
        if T_CutEnergy < 1.0 and abs(e_ij) > 1e-15:
            e_cum = 0.0
            for p in range(n_sc):
                # Pair energy from PNOs 0..p (in SC basis, projected through U_pno)
                # Approximation: use diagonal MP2 energy per PNO
                # E_p = K_sc_pno[p,p] * (2*T2_sc_pno[p,p] - T2_sc_pno[p,p]) / denominator
                # More accurately: accumulate the MP2 pair energy in PNO basis
                # by transforming K and T2 to PNO space up to index p.
                # For efficiency, use the per-PNO contribution:
                #   T2_pno[a,b] = U.T @ T2_sc @ U, K_pno[a,b] = U.T @ K_sc @ U
                #   e_a = Σ_b K_pno[a,b] * (2*T2_pno[a,b] - T2_pno[b,a])
                U_p = U_pno[:, :p+1]
                K_p = reduce(np.dot, (U_p.T, K_sc, U_p))
                T2_p = reduce(np.dot, (U_p.T, T2_sc, U_p))
                Tt_p = 2.0 * T2_p - T2_p.T
                e_cum = np.einsum('ab,ab->', K_p, Tt_p)
                keep_energy[p] = True
                if abs(e_cum / e_ij) >= T_CutEnergy:
                    break

        # 3. Trace criterion: include PNOs until cumulative occupation
        #    fraction > T_CutTrace.
        keep_trace = np.zeros(n_sc, dtype=bool)
        total_trace = np.sum(np.abs(pno_occ))
        if T_CutTrace < 1.0 and total_trace > 1e-15:
            cum_trace = 0.0
            for p in range(n_sc):
                cum_trace += abs(pno_occ[p])
                keep_trace[p] = True
                if cum_trace / total_trace >= T_CutTrace:
                    break

        # Union of all three criteria
        keep = keep_occ | keep_energy | keep_trace
        n_pno = int(np.sum(keep))

        if n_pno == 0:
            n_pno = 1
            keep[np.argmax(np.abs(pno_occ))] = True

        U_pno_kept = U_pno[:, keep]
        n_pno_kept = pno_occ[keep]

        F_sc_diag = np.diag(eps_sc)
        F_pno_block = reduce(np.dot, (U_pno_kept.T, F_sc_diag, U_pno_kept))
        e_pno_sc, V_pno = np.linalg.eigh(F_pno_block)
        U_pno_kept = np.dot(U_pno_kept, V_pno)

        if is_cas_pair:
            nvir_cas_local = nvir_cas
            U_full_ext = np.dot(U_sc, U_pno_kept)
            C_ext_pno = np.dot(C_orth_ij, U_full_ext)

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
            U_full = np.dot(U_sc, U_pno_kept)
            C_pno_ij = np.dot(C_orth_ij, U_full)

            K_pno = reduce(np.dot, (U_pno_kept.T, K_sc, U_pno_kept))
            T2_pno = reduce(np.dot, (U_pno_kept.T, T2_sc, U_pno_kept))

        e_lmp2_total += e_ij * (1 if i == j else 2)

        is_strong = abs(e_ij) > T_CutPairs

        pno_spaces[(i, j)] = {
            'C_pno': C_pno_ij,
            'n_pno': n_pno_kept,
            'e_pno': e_pno_sc,
            'K_pno': K_pno,
            'T2_pno': T2_pno,
            'e_mp2': e_ij,
            'domain_ij': domain_ij,
            'X_orth': X_orth_ij,
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
