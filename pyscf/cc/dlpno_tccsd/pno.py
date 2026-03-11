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


# ---------------------------------------------------------------------------
# Main PNO construction
# ---------------------------------------------------------------------------

def make_pnos(mf, C_lmo, C_pao, pao_domains, S_pao, F_pao,
              T_CutPNO=1e-7, T_CutPairs=1e-4, S_cut_domain=1e-6,
              verbose=None):
    """Construct PNO spaces for all LMO pairs and compute LMP2 energies.

    For each pair (i,j) with i <= j:
      1. Build pair PAO domain = union(domain_i, domain_j)
      2. Orthogonalize PAOs within the pair domain (canonical orthogonalization)
      3. Build F_vv and K_iajb in the orthogonalized domain basis using DF
      4. Semicanonicalize: diagonalize F_vv → get diagonal orbital energies
      5. Compute semicanonical MP2 amplitudes T2 = -K / D
      6. Build pair density, diagonalize → PNOs (U_ij, n_ij)
      7. Truncate at T_CutPNO, canonicalize PNOs w.r.t. Fock
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
    if not hasattr(mf, 'with_df') or mf.with_df is None:
        raise ValueError('DF integrals required. Use mf = scf.RHF(mol).density_fit()')

    log.info('Building LMO/PAO DF 3-index integrals...')
    ovL = _build_ovL(mf.with_df, C_lmo, C_pao, max_memory=mf.max_memory)
    # ovL[i, a, L]: i = LMO index, a = PAO index (global), L = aux index

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
            # F in orthogonal domain: X_orth^T @ F_dom @ X_orth
            F_dom = F_pao[np.ix_(domain_ij, domain_ij)]  # (n_dom, n_dom)
            F_orth = reduce(np.dot, (X_orth_ij.T, F_dom, X_orth_ij))  # (n_orth, n_orth)

            # K_iajb in orthogonal domain: (ovL_i_dom) @ X_orth, etc.
            # ovL[i, domain_ij, :] → transform to orthogonal basis
            ovL_i_dom = ovL[i][domain_ij, :]   # (n_dom, naux)
            ovL_j_dom = ovL[j][domain_ij, :]   # (n_dom, naux)

            # Transform to orthogonal basis: ovL_i_orth[a,L] = sum_mu X_orth[mu,a] ovL_i[mu,L]
            ovL_i_orth = np.dot(X_orth_ij.T, ovL_i_dom)   # (n_orth, naux)
            ovL_j_orth = np.dot(X_orth_ij.T, ovL_j_dom)   # (n_orth, naux)

            K_ij = _pair_K_iajb(ovL_i_orth, ovL_j_orth)   # (n_orth, n_orth)

            # --- 4. Semicanonicalize: diagonalize F_orth ---
            # In semicanonical basis, F_vv is diagonal → denominators are exact
            eps_sc, U_sc = np.linalg.eigh(F_orth)   # eigenvalues, (n_orth, n_orth)

            # K in semicanonical basis
            K_sc = reduce(np.dot, (U_sc.T, K_ij, U_sc))    # (n_orth, n_orth)

            # --- 5. Semicanonical MP2 amplitudes ---
            # T2[a,b] = -K[a,b] / (eps_i + eps_j - eps_a - eps_b)
            D_ij = (eps_i[i] + eps_i[j]
                    - eps_sc[:, None] - eps_sc[None, :])   # (n_orth, n_orth)
            # Avoid division by zero (should not occur in well-behaved systems)
            D_safe = np.where(np.abs(D_ij) > 1e-12, D_ij, 1.0)
            T2_sc = K_sc / D_safe   # (n_orth, n_orth)  T2 = K/D < 0 (D<0)
            T2_sc = np.where(np.abs(D_ij) > 1e-12, T2_sc, 0.0)

            # --- 6. Pair density and PNO eigendecomposition ---
            # Antisymmetrized amplitude: Tt = 2*T2 - T2.T
            Tt_sc = 2.0 * T2_sc - T2_sc.T
            # Pair density: D_ij = Tt @ T2.T + T2 @ Tt.T (symmetrized)
            D_pair = np.dot(Tt_sc, T2_sc.T) + np.dot(T2_sc, Tt_sc.T)
            if i == j:
                D_pair *= 0.5

            # Diagonalize: sort by abs value descending (most important first).
            # For i==j the density is PSD; for i!=j it can have negative
            # eigenvalues that still carry significant pair-correlation energy.
            pno_occ, U_pno = np.linalg.eigh(D_pair)
            order = np.argsort(np.abs(pno_occ))[::-1]
            pno_occ = pno_occ[order]
            U_pno = U_pno[:, order]

            # --- 7. Truncate PNOs ---
            # Use abs(occupation) for threshold: off-diagonal pairs (i≠j) have
            # a non-PSD pair density with negative eigenvalues that still carry
            # significant energy and must be retained.
            keep = np.abs(pno_occ) > T_CutPNO
            n_pno = int(np.sum(keep))

            if n_pno == 0:
                # Include at least 1 PNO to avoid empty pair
                n_pno = 1
                keep[np.argmax(np.abs(pno_occ))] = True

            U_pno_kept = U_pno[:, keep]        # (n_orth, n_pno)
            n_pno_kept = pno_occ[keep]         # (n_pno,)

            # Canonicalize PNOs w.r.t. Fock: diagonalize F in PNO subspace
            F_pno_block = reduce(np.dot, (U_pno_kept.T, F_orth, U_pno_kept))
            e_pno_sc, V_pno = np.linalg.eigh(F_pno_block)
            U_pno_kept = np.dot(U_pno_kept, V_pno)   # (n_orth, n_pno)

            # Full transformation: PAO(domain) → semicanonical → PNO
            # C_pno[:,k] in AO basis: C_pno = C_orth @ U_sc @ U_pno_kept
            U_full = np.dot(U_sc, U_pno_kept)          # (n_orth, n_pno)
            C_pno_ij = np.dot(C_orth_ij, U_full)       # (nao, n_pno)

            # K and T2 in PNO basis (for LCCSD)
            K_pno = reduce(np.dot, (U_pno_kept.T, K_sc, U_pno_kept))
            T2_pno = reduce(np.dot, (U_pno_kept.T, T2_sc, U_pno_kept))

            # --- 8. LMP2 pair energy ---
            Tt_pno = 2.0 * T2_pno - T2_pno.T
            # e_ij = sum_ab K[a,b] * (2*T2[a,b] - T2[b,a])
            e_ij = np.einsum('ab,ab->', K_pno, Tt_pno)

            e_lmp2_total += e_ij * (1 if i == j else 2)

            # --- 9. Classify pair ---
            is_strong = abs(e_ij) > T_CutPairs

            pno_spaces[(i, j)] = {
                'C_pno': C_pno_ij,
                'n_pno': n_pno_kept,
                'e_pno': e_pno_sc,
                'K_pno': K_pno,
                'T2_pno': T2_pno,
                'e_mp2': e_ij,
                'domain_ij': domain_ij,
                # Store transformation info for LCCSD integral rebuild
                'X_orth': X_orth_ij,   # (n_dom, n_orth)
                'U_full': U_full,      # (n_orth, n_pno)
                'domain_idx': domain_ij,
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
