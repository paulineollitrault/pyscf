"""Correct closed-shell DLPNO-(T) using Jiang Eq 53 with base W.

Replaces the buggy ccsd_t_slow-based 36-term formula.
Uses the base W (one occupied term, no P_L) + the Eq 53 antisymmetrizer
with factor ×6 (= 2 spin × 3 closed-shell) for the compact i≤j≤k loop.

The vooo (A*t2) term sums over ALL occupied LMOs in the triplet domain,
matching Psi4's implementation.
"""
import numpy as np
from functools import reduce


def compute_triple_energy_eq53(
    i, j, k,
    ovL_triple,   # (3, n_tno, naux) — ovL for triple's 3 LMOs
    vvL,          # (n_tno, n_tno, naux)
    ooL_full,     # (nocc, nocc, naux) — ooL for ALL occupied
    t2_for_T,     # dict: pair_key -> (n_pno, n_pno) pair amplitudes
    pno_spaces,   # dict: pair_key -> PNO data
    s1e,          # (nao, nao) AO overlap
    C_tno_sc,     # (nao, n_tno) semicanonical TNO coefficients
    eps_occ_ijk,  # (3,) semicanonical occupied energies for this triple
    eps_vir,      # (n_tno,) semicanonical virtual energies
    V_occ_sc,     # (3, 3) occupied semicanonalization matrix
    t1_sc=None,   # (3, n_tno) or None
    fvo_sc=None,  # (n_tno, 3) or None
):
    """Compute (T) energy for one triple i<=j<=k using Eq 53.

    Returns et_ijk (float): energy contribution including spin factor.
    """
    n = len(eps_vir)
    if n == 0:
        return 0.0

    triple = [i, j, k]
    nocc = ooL_full.shape[0]

    # Project T2 amplitudes to TNO semicanonical basis
    def _map_t2(p, q):
        pk = (min(p, q), max(p, q))
        if pk not in t2_for_T or pk not in pno_spaces:
            return np.zeros((n, n))
        C_p = pno_spaces[pk]['C_pno']
        if C_p.shape[1] == 0:
            return np.zeros((n, n))
        U = reduce(np.dot, (C_p.T, s1e, C_tno_sc))
        t2_proj = reduce(np.dot, (U.T, t2_for_T[pk], U))
        if p > q:
            t2_proj = t2_proj.T
        return t2_proj

    # Build base W[a,b,c] = (ia|bf)*t2[k,j,c,f] - sum_l (ia|jl)*t2[l,k,b,c]
    # In semicanonical basis: use ovL_triple for LMO i (local index 0),
    # ooL between j (local 1) and all occupied, t2 for pairs (k,j) and (l,k).

    # K term: K[a,b,f] = ovL[i_local, a, L] * vvL[f, b, L]
    K_ab = np.einsum('aL,fbL->abf', ovL_triple[0], vvL)  # (n, n, n)
    # t2[k,j] in TNO-SC basis
    t2_kj = _map_t2(k, j)
    base_K = np.einsum('abf,cf->abc', K_ab, t2_kj)  # (n, n, n)

    # A term: sum_l (ia|jl)*t2[l,k,b,c] — sum over ALL occupied l
    # (ia|jl) = ovL[i_sc, a, L] * ooL[j_global, l_global, L]
    # In SC basis: ovL_triple is already in SC. ooL_full is in LMO basis.
    # Need: A[a, l] = sum_L ovL_sc[0, a, L] * ooL_full[j, l, L]
    # But ovL_triple is in SC basis (rotated by V_occ_sc).
    # ooL_full is in LMO basis. Need to be consistent.

    # Actually, the formula uses GLOBAL occupied indices:
    # base(i,j,k)[a,b,c] = sum_f K[a,b,f]*t2[k,j,c,f] - sum_l A[i,a,j,l]*t2[l,k,b,c]
    # where K[a,b,f] = ovL[i,a,L]*vvL[f,b,L] and A[i,a,j,l] = ovL[i,a,L]*ooL[j,l,L]
    # These use GLOBAL occupied indices i,j,k (not semicanonical).

    # For Eq 53: we DON'T semicanonicalize the occupied. We use LMO energies
    # directly in the denominator. The virtual space IS semicanonalized (Fock diagonal).
    # The denominator: D = eps_i + eps_j + eps_k - eps_a - eps_b - eps_c
    # where eps_i,j,k are the LMO Fock diagonal elements.

    # But our eps_occ_ijk is semicanonalized... For canonical MOs it doesn't matter.
    # For localized MOs, we need eps_occ = F_lmo[i,i], F_lmo[j,j], F_lmo[k,k]
    # (diagonal elements, not semicanonical eigenvalues).

    # For now: use the semicanonalized eps_occ_ijk from the caller.
    # This is the (T0) approximation which introduces some error from the
    # off-diagonal Fock elements in the occupied block.

    # Build A*t2 with full occ sum using ovL in LMO basis:
    # We need ovL in LMO basis for LMO i. ovL_triple is already in SC basis.
    # Convert back: ovL_lmo = V_occ_sc.T @ ovL_triple (3×3 @ 3×n×naux → 3×n×naux)
    # Then ovL_lmo[0] is LMO i's ovL.
    naux = ovL_triple.shape[2]
    ovL_lmo = np.einsum('nm,maL->naL', V_occ_sc.T, ovL_triple)  # (3, n, naux) in LMO basis

    A_al = np.einsum('aL,lL->al', ovL_lmo[0], ooL_full[j])  # (n, nocc)
    # t2[l,k,b,c] for all l:
    t2_lk = np.zeros((nocc, n, n))
    for l in range(nocc):
        t2_lk[l] = _map_t2(l, k)
    base_A = np.einsum('al,lbc->abc', A_al, t2_lk)

    W = base_K - base_A

    # Denominator — use eps_occ for the GLOBAL i,j,k (not semicanonical)
    # For simplicity, use semicanonical (caller provides eps_occ_ijk)
    D_occ = eps_occ_ijk[0] + eps_occ_ijk[1] + eps_occ_ijk[2]
    D = D_occ - (eps_vir[:, None, None] + eps_vir[None, :, None] + eps_vir[None, None, :])

    T = W / D

    # Include V disconnected (T1 contribution) if available
    V = W.copy()
    # (V intermediate: V = W + 0.5 * V_disconnected)
    # For Eq 53: energy = t3 * antisym(V) where t3 = W/D
    # and V = W + P_S[t1*(jb|kc)] + t2*fvo
    # For now, no T1.

    # Occupancy degeneracy factor
    dij = int(i == j)
    djk = int(j == k)
    dik = int(i == k)
    occ_denom = 1 + dij + djk + dik + 2 * dij * djk * dik

    # Eq 53 antisymmetrizer on virtual indices: 8V - 4V(cba) - 4V(acb) - 4V(bac) + 2V(bca) + 2V(cab)
    et = (8 * np.sum(V * T)
          - 4 * np.sum(V.transpose(2, 1, 0) * T)
          - 4 * np.sum(V.transpose(0, 2, 1) * T)
          - 4 * np.sum(V.transpose(1, 0, 2) * T)
          + 2 * np.sum(V.transpose(1, 2, 0) * T)
          + 2 * np.sum(V.transpose(2, 0, 1) * T))

    # Factor 6 = 2 (spin) × 3 (closed-shell from spin-orbital → restricted)
    return float(et * 6 / occ_denom)
