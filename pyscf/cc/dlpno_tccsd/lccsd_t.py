"""
External-space perturbative triples (T) for DLPNO-TCCSD(T).

Computes the (T) energy correction restricted to triples that have at least
one index outside the CAS space. Pure-CAS triples are excluded because they
are already contained in the DMRG wavefunction.

The triple PNO space for each triplet (i,j,k) is the intersection of the
pair PNO spaces: PNO(ijk) = PNOs shared between pairs (ij), (ik), (jk).
This follows the TNO/triple-PNO approach of Jiang JCP 2024 / Psi4 lccsd_t.cc.

The energy formula is the standard CCSD(T) perturbative triples expression:
    e_T = sum_{i>j>k, abc} W[a,b,c] * r3(W)[a,b,c] / D[a,b,c]
where W is the W3 intermediate and r3 is the cyclic antisymmetrizer.

W[a,b,c] is built from the 6-term canonical formula using DF 3-index integrals:
    W[a,b,c] = sum_f (ia|bf)*t2_jk[c,f] + sum_f (jb|af)*t2_ik[c,f]
             + sum_f (kc|af)*t2_ij[b,f] + sum_f (ia|cf)*t2_kj[b,f]
             + sum_f (jb|cf)*t2_ki[a,f] + sum_f (kc|bf)*t2_ji[a,f]
where (xa|bf) = sum_L ovL_x[a,L] * vvL[b,f,L].

CAS double-counting prevention (Lang et al. 2020, Sec. II.C):
    All active T1 and T2 amplitudes are zeroed before computing (T).
    All triples (i,j,k) are included — no triples are skipped.

References:
    Lee & Head-Gordon, JCTC 2019, 15, 4594  (TCCSD(T) CAS exclusion)
    Jiang, JCP 2024  (DLPNO-(T) triple PNO intersection)
    Stanton, Chem. Phys. Lett. 1997, 281, 130  ((T) energy formula)
    Ye & Berkelbach, JCTC 2024  (LNO-(T) for PySCF infrastructure)
    ccsd_t_slow.py: canonical reference formula
"""

import numpy as np
from functools import reduce
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo


def _triple_pno_union(pno_spaces, i, j, k, s1e, t2_for_T=None,
                      T_CutTNO=1e-9, S_cut=1e-6):
    """Compute the TNO space from the averaged triplet density (Jiang 2024 eq.62).

    1. Pool PNO columns from pairs ij, ik, jk.
    2. Orthogonalize via canonical orthogonalization in S metric.
    3. Project pair densities into this common space, average → D_ijk.
    4. Diagonalize D_ijk, truncate at T_CutTNO → final TNO basis.

    Args:
        pno_spaces (dict): Output of pno.make_pnos.
        i, j, k (int): Triple LMO indices.
        s1e (np.ndarray): (nao, nao) AO overlap matrix S.
        t2_for_T (dict, optional): T2 amplitudes in PNO basis for each pair.
            If provided, pair densities are built from these (CCSD amplitudes).
            Otherwise, initial PNO T2 from pno_spaces is used.
        T_CutTNO (float): TNO occupation number truncation threshold.
        S_cut (float): Eigenvalue threshold for canonical orthogonalization.

    Returns:
        C_tno (np.ndarray): (nao, n_tno) S-orthonormal TNO coefficients.
        n_tno (int): Number of TNOs.
    """
    ij = (min(i,j), max(i,j))
    ik = (min(i,k), max(i,k))
    jk = (min(j,k), max(j,k))

    if ij not in pno_spaces or ik not in pno_spaces or jk not in pno_spaces:
        return None, 0

    C_ij = pno_spaces[ij]['C_pno']
    C_ik = pno_spaces[ik]['C_pno']
    C_jk = pno_spaces[jk]['C_pno']

    if C_ij.shape[1] == 0 and C_ik.shape[1] == 0 and C_jk.shape[1] == 0:
        return np.zeros((s1e.shape[0], 0)), 0

    # Pool all PNO columns from the three pairs
    cols = [c for c in (C_ij, C_ik, C_jk) if c.shape[1] > 0]
    C_pool = np.hstack(cols)   # (nao, n_ij + n_ik + n_jk)

    # Canonical orthogonalization in S metric: diagonalize S_pool, keep > S_cut
    S_pool = reduce(np.dot, (C_pool.T, s1e, C_pool))
    eigvals, eigvecs = np.linalg.eigh(S_pool)
    keep = eigvals > S_cut
    n_union = int(np.sum(keep))

    if n_union == 0:
        return np.zeros((s1e.shape[0], 0)), 0

    X = eigvecs[:, keep] / np.sqrt(eigvals[keep])
    C_union = np.dot(C_pool, X)   # (nao, n_union), S-orthonormal

    # If no T2 available or T_CutTNO <= 0, return the full union
    if T_CutTNO <= 0:
        return C_union, n_union

    # Build averaged triplet density D_ijk = (D_ij + D_ik + D_jk) / 3
    # Project each pair's T2 into the union basis, compute pair density
    D_avg = np.zeros((n_union, n_union))
    for pair_key, ii, jj in [(ij, i, j), (ik, i, k), (jk, j, k)]:
        C_p = pno_spaces[pair_key]['C_pno']
        if C_p.shape[1] == 0:
            continue
        # Projection matrix: PNO(pair) -> union TNO
        U = reduce(np.dot, (C_p.T, s1e, C_union))   # (n_pno, n_union)
        # Get T2 amplitudes
        if t2_for_T is not None and pair_key in t2_for_T:
            T2_p = t2_for_T[pair_key]
        elif pno_spaces[pair_key].get('T2_pno') is not None:
            T2_p = pno_spaces[pair_key]['T2_pno']
        else:
            continue
        # Project T2 to union basis
        T2_u = reduce(np.dot, (U.T, T2_p, U))   # (n_union, n_union)
        Tt_u = 2.0 * T2_u - T2_u.T
        # Pair density in union basis
        D_p = np.dot(Tt_u, T2_u.T) + np.dot(T2_u, Tt_u.T)
        if ii == jj:
            D_p *= 0.5
        D_avg += D_p

    D_avg /= 3.0

    # Diagonalize triplet density and truncate at T_CutTNO
    tno_occ, tno_vecs = np.linalg.eigh(D_avg)
    keep_tno = np.abs(tno_occ) > T_CutTNO
    n_tno = int(np.sum(keep_tno))

    if n_tno == 0:
        # Keep at least the largest
        n_tno = 1
        keep_tno[np.argmax(np.abs(tno_occ))] = True

    # Transform union → truncated TNO
    C_tno = np.dot(C_union, tno_vecs[:, keep_tno])   # (nao, n_tno)

    return C_tno, n_tno


def _preload_df_integrals(with_df):
    """Preload all DF 3-index integrals into memory.

    Returns Lpq_full as a contiguous (naux, nao_pair) array in packed
    triangular format, ready for _ao2mo.nr_e2.  Reading the HDF5 file
    once up-front makes all subsequent integral transforms thread-safe
    and eliminates redundant I/O.
    """
    naux = with_df.get_naoaux()
    chunks = []
    for Lpq in with_df.loop():
        chunks.append(Lpq.copy())
    return np.vstack(chunks)   # (naux, nao_pair)


def _build_ovL_tno(Lpq_full, C_lmo, C_tno, lmo_indices):
    """Build (LMO_i, TNO_a | L) 3-index DF tensor for selected LMOs.

    Args:
        Lpq_full (np.ndarray): (naux, nao_pair) preloaded DF integrals.
        C_lmo (np.ndarray): (nao, nocc_lmo) LMO coefficients.
        C_tno (np.ndarray): (nao, n_tno) TNO coefficients in AO basis.
        lmo_indices (list): LMO orbital indices to compute (e.g. [i, j, k]).

    Returns:
        ovL (np.ndarray): (len(lmo_indices), n_tno, naux)
    """
    n_tno = C_tno.shape[1]
    naux = Lpq_full.shape[0]
    n_sel = len(lmo_indices)

    C_occ_sel = C_lmo[:, lmo_indices]   # (nao, n_sel)
    nmo_sel = n_sel + n_tno
    mo = np.asarray(np.hstack((C_occ_sel, C_tno)), order='F')
    ijslice = (0, n_sel, n_sel, nmo_sel)

    buf = _ao2mo.nr_e2(Lpq_full, mo, ijslice, aosym='s2')
    ovL = buf.reshape(naux, n_sel, n_tno).transpose(1, 2, 0).copy()
    return ovL


def _build_vvL_tno(Lpq_full, C_tno):
    """Build (TNO_a, TNO_b | L) 3-index DF tensor in TNO basis.

    Args:
        Lpq_full (np.ndarray): (naux, nao_pair) preloaded DF integrals.
        C_tno (np.ndarray): (nao, n_tno) TNO coefficients.

    Returns:
        vvL (np.ndarray): (n_tno, n_tno, naux)
    """
    n_tno = C_tno.shape[1]
    naux = Lpq_full.shape[0]

    mo = np.asfortranarray(C_tno)
    ijslice = (0, n_tno, 0, n_tno)

    buf = _ao2mo.nr_e2(Lpq_full, mo, ijslice, aosym='s2')
    vvL = buf.reshape(naux, n_tno, n_tno).transpose(1, 2, 0).copy()
    return vvL


def _build_ooL_triple(Lpq_full, C_lmo, lmo_indices):
    """Build (n_sel, n_sel, naux) occ-occ DF integrals for selected LMO indices.

    Args:
        Lpq_full (np.ndarray): (naux, nao_pair) preloaded DF integrals.
        C_lmo (np.ndarray): (nao, nocc_lmo) LMO coefficients.
        lmo_indices (list): LMO orbital indices [i, j, k] or [i, k].

    Returns:
        ooL (np.ndarray): (n_sel, n_sel, naux) occ-occ DF tensor in LMO basis.
    """
    n_sel = len(lmo_indices)
    naux = Lpq_full.shape[0]
    C_occ_sel = C_lmo[:, lmo_indices]   # (nao, n_sel)
    mo = np.asfortranarray(C_occ_sel)
    ijslice = (0, n_sel, 0, n_sel)

    buf = _ao2mo.nr_e2(Lpq_full, mo, ijslice, aosym='s2')
    ooL = buf.reshape(naux, n_sel, n_sel).transpose(1, 2, 0).copy()
    return ooL


def _w3_intermediate(t2_sc, ovL_sc, ooL_sc, vvL_sc, eps_occ, eps_vir,
                     t1_sc=None, fvo_sc=None, sum_all_occ=False,
                     occ_indices=None,
                     ooL_sc_full=None, t2_sc_full=None):
    """Compute (T) energy for one triple i<=j<=k using Eq 53 (Jiang 2024).

    Uses the base W (single occupied assignment, no P_L permutation) with the
    Eq 53 closed-shell antisymmetrizer acting on virtual indices. The factor 6
    (= 2 spin × 3 closed-shell) converts from the spin-orbital formula.

    The vooo (A*t2) term sums over ALL occupied LMOs when ooL_sc_full and
    t2_sc_full are provided, matching Psi4's triplet-domain approach.

    The occupied degeneracy factor 1/(1+δ_ij+δ_jk+δ_ik+2δ_ijδ_jkδ_ik)
    is applied when occ_indices=(i,j,k) is given.

    Args:
        t2_sc: (nocc_t, nocc_t, n, n) T2 in SC occ × SC vir.
        ovL_sc: (nocc_t, n, naux) SC-occ × TNO DF integrals.
        ooL_sc: (nocc_t, nocc_t, naux) or larger for full-occ A*t2.
        vvL_sc: (n, n, naux) TNO × TNO DF integrals.
        eps_occ: SC occupied orbital energies.
        eps_vir: (n,) SC virtual orbital energies.
        occ_indices: (i, j, k) global occupied indices for degeneracy factor.
        ooL_sc_full: (nocc_t, nocc_all, naux) for full-occ vooo term.
        t2_sc_full: (nocc_all, nocc_t, n, n) for full-occ vooo term.

    Returns:
        et_ijk (float): (T) energy contribution including spin factor.
    """
    n = len(eps_vir)
    if n == 0:
        return 0.0

    nocc_triple = ovL_sc.shape[0]
    naux = ovL_sc.shape[2]

    # --- Denominator ---
    D_vir = eps_vir[:, None, None] + eps_vir[None, :, None] + eps_vir[None, None, :]
    D_occ = eps_occ[0] + eps_occ[1] + eps_occ[2]
    D = D_vir - D_occ  # eps_abc - eps_ijk (positive for occupied < virtual)

    # --- Occupied degeneracy factor ---
    if occ_indices is not None:
        i, j, k = occ_indices
        dij = int(i == j); djk = int(j == k); dik = int(i == k)
        occ_denom = 1 + dij + djk + dik + 2 * dij * djk * dik
    else:
        occ_denom = 1

    # --- Build P_L-assembled W (Jiang Eq 47+49) ---
    # P_L simultaneously permutes occupied (i,j,k) and virtual (a,b,c).
    # For each S_3 permutation σ: W[a,b,c] += base(σ(i),σ(j),σ(k))[σ(a),σ(b),σ(c)]
    # The transpose maps base[x,y,z] → W[a,b,c] by inverting the virtual permutation.
    trans = [lambda x: x,                     # (i,j,k) → identity
             lambda x: x.transpose(0,2,1),    # (i,k,j) → (a,c,b)
             lambda x: x.transpose(1,0,2),    # (j,i,k) → (b,a,c)
             lambda x: x.transpose(2,0,1),    # (j,k,i) → (b,c,a)
             lambda x: x.transpose(1,2,0),    # (k,i,j) → (c,a,b)
             lambda x: x.transpose(2,1,0)]    # (k,j,i) → (c,b,a)

    W = np.zeros((n, n, n))
    for pidx in range(6):
        p_map = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)][pidx]
        ip, iq, ir = p_map

        K_ab = np.einsum('aL,fbL->abf', ovL_sc[ip], vvL_sc)
        base = np.einsum('abf,cf->abc', K_ab, t2_sc[ir, iq])

        if ooL_sc_full is not None and t2_sc_full is not None:
            A_al = np.einsum('aL,mL->am', ovL_sc[ip], ooL_sc_full[iq])
            base -= np.einsum('am,mbc->abc', A_al, t2_sc_full[:, ir])
        else:
            A_al = np.einsum('aL,mL->am', ovL_sc[ip], ooL_sc[iq])
            base -= np.einsum('am,mbc->abc', A_al, t2_sc[:, ir])

        W += trans[pidx](base)

    # --- T = -W/D ---
    T = -W / D

    # --- V = W + T1 disconnected (Psi4 lines 930-962) ---
    V = W.copy()
    if t1_sc is not None and fvo_sc is not None:
        K_jk = np.einsum('bL,cL->bc', ovL_sc[1], ovL_sc[2])
        K_ik = np.einsum('aL,cL->ac', ovL_sc[0], ovL_sc[2])
        K_ij = np.einsum('aL,bL->ab', ovL_sc[0], ovL_sc[1])
        V += (np.einsum('a,bc->abc', t1_sc[0], K_jk)
              + np.einsum('b,ac->abc', t1_sc[1], K_ik)
              + np.einsum('c,ab->abc', t1_sc[2], K_ij))

    # --- Energy: Eq 53 antisymmetrizer on virtual indices ---
    et = (8 * np.sum(V * T)
          - 4 * np.sum(V.transpose(2, 1, 0) * T)
          - 4 * np.sum(V.transpose(0, 2, 1) * T)
          - 4 * np.sum(V.transpose(1, 0, 2) * T)
          + 2 * np.sum(V.transpose(1, 2, 0) * T)
          + 2 * np.sum(V.transpose(2, 0, 1) * T))

    return float(et / occ_denom)


def _zero_cas_t2_amplitudes(t2_pno_all, pno_spaces, occ_cas_idx, C_cas_vir,
                            s1e, cas_proj_thresh=0.5):
    """Zero T2 amplitudes with CAS virtual character for TCC (T) calculation.

    Per Lang et al. 2020, Sec. II.C: to prevent double-counting of static
    correlation, all active single and double amplitudes are set to zero
    before computing the (T) correction.

    The PNO construction (pno.py) structures CAS-occupied pair PNOs as
    C_pno = [C_cas_vir | C_ext_pno]  (Lang et al. eq. 10: S^ij = I_NCAS ⊕ d^ij).
    The first n_cas_vir PNOs ARE the CAS virtuals; we zero exactly those
    rows/columns.  For non-CAS pairs, the PNOs are pure external and
    nothing is zeroed.

    Args:
        t2_pno_all (dict): {(i,j): t2_array} T2 amplitudes in PNO basis.
        pno_spaces (dict): PNO information from make_pnos.
        occ_cas_idx (array-like): Indices of CAS occupied LMOs.
        C_cas_vir (np.ndarray): (nao, n_cas_vir) CAS virtual MO coefficients.
        s1e (np.ndarray): (nao, nao) AO overlap matrix.
        cas_proj_thresh (float): Unused (kept for API compatibility).

    Returns:
        t2_zeroed (dict): Copy of t2_pno_all with CAS components zeroed.
    """
    occ_cas_set = set(int(x) for x in occ_cas_idx) if occ_cas_idx is not None else set()
    n_cas_vir = C_cas_vir.shape[1] if C_cas_vir is not None else 0
    t2_zeroed = {}

    for pair_key, t2 in t2_pno_all.items():
        i, j = pair_key

        # Only CAS-occupied pairs have the [CAS_vir | ext_PNO] structure.
        # Zero only the pure CAS-CAS block (both virtual indices in CAS).
        # Mixed CAS-ext amplitudes are part of T_ext and must be kept —
        # they have at least one index outside the CAS.
        if (n_cas_vir > 0
                and i in occ_cas_set and j in occ_cas_set
                and t2.shape[0] >= n_cas_vir):
            t2_new = t2.copy()
            t2_new[:n_cas_vir, :n_cas_vir] = 0.0
            t2_zeroed[pair_key] = t2_new
        else:
            t2_zeroed[pair_key] = t2.copy()

    return t2_zeroed


def _process_one_triple(i, j, k,
                        pno_spaces, t2_for_T,
                        Lpq_full, C_lmo, fock_ao, F_lmo, s1e,
                        t1_pno=None, T_CutTNO=1e-9):
    """Compute (T) energy contribution for one triple (i,j,k). Thread-safe.

    All inputs are read-only.  Lpq_full is the preloaded DF array (naux, nao_pair)
    so no HDF5 reads happen here — fully thread-safe.

    Returns et_ijk (float), or 0.0 if the triple should be skipped.
    """
    ij = (min(i, j), max(i, j))
    ik = (min(i, k), max(i, k))
    jk = (min(j, k), max(j, k))

    if (ij not in pno_spaces or ik not in pno_spaces or jk not in pno_spaces):
        return 0.0
    if (ij not in t2_for_T or ik not in t2_for_T or jk not in t2_for_T):
        return 0.0

    C_tno, n_tno = _triple_pno_union(pno_spaces, i, j, k, s1e,
                                      t2_for_T=t2_for_T,
                                      T_CutTNO=T_CutTNO)
    if n_tno == 0:
        return 0.0

    F_tno_full = reduce(np.dot, (C_tno.T, fock_ao, C_tno))
    eps_tno_sc, V_sc = np.linalg.eigh(F_tno_full)
    C_tno_sc = np.dot(C_tno, V_sc)

    def _map_t2(pk):
        C_p = pno_spaces[pk]['C_pno']
        U = reduce(np.dot, (C_p.T, s1e, C_tno_sc))
        return reduce(np.dot, (U.T, t2_for_T[pk], U))

    t2_ij_sc = _map_t2(ij)
    t2_ik_sc = _map_t2(ik)
    t2_jk_sc = _map_t2(jk)

    ovL_ijk    = _build_ovL_tno(Lpq_full, C_lmo, C_tno_sc, [i, j, k])
    vvL_sc     = _build_vvL_tno(Lpq_full, C_tno_sc)
    triple_lmo = [i, j, k]
    nocc_lmo = C_lmo.shape[1]

    # No occupied semicanonalization — use diagonal LMO Fock for the denominator.
    # This matches Psi4's (T0) implementation.  The off-diagonal occupied Fock
    # elements are neglected in (T0); the (T1) iterative approach corrects for
    # them via inter-triple coupling.
    eps_occ = np.array([F_lmo[ii, ii] for ii in triple_lmo])

    # Full-occ ooL for the vooo (A*t2) term
    ooL_lmo_full = _build_ooL_triple(Lpq_full, C_lmo, list(range(nocc_lmo)))

    # T2 for all occupied paired with triple LMOs: t2_mr[l, r_local] = t2[l, triple[r]]
    def _proj_t2(p, q):
        pk = (min(p, q), max(p, q))
        if pk not in t2_for_T or pk not in pno_spaces:
            return np.zeros((n_tno, n_tno))
        C_p = pno_spaces[pk]['C_pno']
        if C_p.shape[1] == 0:
            return np.zeros((n_tno, n_tno))
        U = reduce(np.dot, (C_p.T, s1e, C_tno_sc))
        t2_proj = reduce(np.dot, (U.T, t2_for_T[pk], U))
        if p > q:
            t2_proj = t2_proj.T
        return t2_proj

    t2_mr = np.zeros((nocc_lmo, 3, n_tno, n_tno))
    for r_local, r_global in enumerate(triple_lmo):
        for m in range(nocc_lmo):
            t2_mr[m, r_local] = _proj_t2(m, r_global)

    # T2 block for the 3 triple LMOs (used by _w3_intermediate)
    t2_block = np.zeros((3, 3, n_tno, n_tno))
    for p in range(3):
        for q in range(3):
            t2_block[p, q] = t2_mr[triple_lmo[p], q]

    # Project local T1 to TNO basis for the V intermediate.
    t1_lmo = None
    fvo = None
    if t1_pno is not None:
        t1_lmo_tno = np.zeros((3, n_tno))
        for idx_local, r_global in enumerate(triple_lmo):
            key_rr = (r_global, r_global)
            if key_rr in pno_spaces and t1_pno.get(r_global) is not None:
                C_pno_rr = pno_spaces[key_rr]['C_pno']
                if C_pno_rr.shape[1] > 0 and t1_pno[r_global].size > 0:
                    U_rr = reduce(np.dot, (C_pno_rr.T, s1e, C_tno_sc))
                    t1_lmo_tno[idx_local] = U_rr.T @ t1_pno[r_global]
        t1_lmo = t1_lmo_tno

        C_lmo_triple = C_lmo[:, triple_lmo]
        fvo = reduce(np.dot, (C_tno_sc.T, fock_ao, C_lmo_triple))

    return _w3_intermediate(t2_block, ovL_ijk, None, vvL_sc,
                            eps_occ, eps_tno_sc,
                            t1_sc=t1_lmo, fvo_sc=fvo,
                            occ_indices=(i, j, k),
                            ooL_sc_full=ooL_lmo_full[triple_lmo],
                            t2_sc_full=t2_mr)


def _process_degenerate_pair(i, k,
                             pno_spaces, t2_for_T,
                             Lpq_full, C_lmo, fock_ao, F_lmo, s1e,
                             t1_pno=None, T_CutTNO=1e-9):
    """Compute (T) energy from degenerate occupied triples {i,i,k} and {i,k,k}.

    With the Eq 53 formula, _process_one_triple handles degenerate triples
    correctly via the occ_indices degeneracy factor. We just call it for both
    (i,i,k) and (i,k,k).
    """
    kwargs = dict(pno_spaces=pno_spaces, t2_for_T=t2_for_T,
                  Lpq_full=Lpq_full, C_lmo=C_lmo, fock_ao=fock_ao,
                  F_lmo=F_lmo, s1e=s1e, t1_pno=t1_pno, T_CutTNO=T_CutTNO)
    et_iik = _process_one_triple(i, i, k, **kwargs)
    et_ikk = _process_one_triple(i, k, k, **kwargs)
    return et_iik + et_ikk


def _process_degenerate_pair_old(i, k,
                             pno_spaces, t2_for_T,
                             Lpq_full, C_lmo, fock_ao, F_lmo, s1e,
                             t1_pno=None, T_CutTNO=1e-9):
    """OLD: Compute (T) energy from degenerate occupied triples {i,i,k} and {i,k,k}.

    For a pair (i,k) with i<k, the 2^3=8 occupied combinations in the
    2-LMO block cover: {i,i,k} (3 entries), {i,k,k} (3 entries), and
    {i,i,i}/{k,k,k} (2 entries, zero after r3).

    This captures all "two-equal" occupied contributions missing from the
    all-distinct (i<j<k) loop.

    Returns et (float), or 0.0 if the pair should be skipped.
    """
    ii = (i, i)
    kk = (k, k)
    ik = (min(i, k), max(i, k))

    # Need diagonal pairs (i,i), (k,k) and off-diagonal (i,k)
    if ik not in pno_spaces or ik not in t2_for_T:
        return 0.0
    # Diagonal pairs may not exist if they have no PNOs — treat as zero
    has_ii = ii in pno_spaces and ii in t2_for_T
    has_kk = kk in pno_spaces and kk in t2_for_T

    # Build TNO space from available pair PNO spaces
    cols = []
    pair_keys_available = []
    if has_ii:
        C_ii = pno_spaces[ii]['C_pno']
        if C_ii.shape[1] > 0:
            cols.append(C_ii)
            pair_keys_available.append((ii, i, i))
    C_ik = pno_spaces[ik]['C_pno']
    if C_ik.shape[1] > 0:
        cols.append(C_ik)
        pair_keys_available.append((ik, i, k))
    if has_kk:
        C_kk = pno_spaces[kk]['C_pno']
        if C_kk.shape[1] > 0:
            cols.append(C_kk)
            pair_keys_available.append((kk, k, k))

    if not cols:
        return 0.0

    C_pool = np.hstack(cols)
    S_pool = reduce(np.dot, (C_pool.T, s1e, C_pool))
    eigvals, eigvecs = np.linalg.eigh(S_pool)
    S_cut = 1e-6
    keep = eigvals > S_cut
    n_union = int(np.sum(keep))
    if n_union == 0:
        return 0.0
    X = eigvecs[:, keep] / np.sqrt(eigvals[keep])
    C_union = np.dot(C_pool, X)

    # Build averaged triplet density and truncate at T_CutTNO
    if T_CutTNO > 0 and pair_keys_available:
        D_avg = np.zeros((n_union, n_union))
        n_pairs_used = 0
        for pair_key, p, q in pair_keys_available:
            C_p = pno_spaces[pair_key]['C_pno']
            if C_p.shape[1] == 0:
                continue
            U = reduce(np.dot, (C_p.T, s1e, C_union))
            T2_p = t2_for_T[pair_key]
            T2_u = reduce(np.dot, (U.T, T2_p, U))
            Tt_u = 2.0 * T2_u - T2_u.T
            D_p = np.dot(Tt_u, T2_u.T) + np.dot(T2_u, Tt_u.T)
            if p == q:
                D_p *= 0.5
            D_avg += D_p
            n_pairs_used += 1
        if n_pairs_used > 0:
            D_avg /= max(n_pairs_used, 1)
            tno_occ, tno_vecs = np.linalg.eigh(D_avg)
            keep_tno = np.abs(tno_occ) > T_CutTNO
            n_tno = int(np.sum(keep_tno))
            if n_tno == 0:
                n_tno = 1
                keep_tno[np.argmax(np.abs(tno_occ))] = True
            C_tno = np.dot(C_union, tno_vecs[:, keep_tno])
        else:
            C_tno = C_union
            n_tno = n_union
    else:
        C_tno = C_union
        n_tno = n_union

    # Semicanonicalize TNO
    F_tno_full = reduce(np.dot, (C_tno.T, fock_ao, C_tno))
    eps_tno_sc, V_sc = np.linalg.eigh(F_tno_full)
    C_tno_sc = np.dot(C_tno, V_sc)

    # Map T2 amplitudes to TNO-SC basis
    def _map_t2(pk):
        if pk not in pno_spaces or pk not in t2_for_T:
            return np.zeros((n_tno, n_tno))
        C_p = pno_spaces[pk]['C_pno']
        if C_p.shape[1] == 0:
            return np.zeros((n_tno, n_tno))
        U = reduce(np.dot, (C_p.T, s1e, C_tno_sc))
        return reduce(np.dot, (U.T, t2_for_T[pk], U))

    # Build 2×2 occupied block (0→i, 1→k)
    pair_lmo = [i, k]
    t2_ii = _map_t2(ii)
    t2_kk = _map_t2(kk)
    t2_ik = _map_t2(ik)

    t2_lmo_block = np.zeros((2, 2, n_tno, n_tno))
    t2_lmo_block[0, 0] = t2_ii
    t2_lmo_block[1, 1] = t2_kk
    t2_lmo_block[0, 1] = t2_ik
    t2_lmo_block[1, 0] = t2_ik.T

    # Semicanonicalize occupied
    F_occ_2x2 = F_lmo[np.ix_(pair_lmo, pair_lmo)]
    eps_occ_sc, V_occ_sc = np.linalg.eigh(F_occ_2x2)
    t2_sc_block = np.einsum('pm,qn,pqAB->mnAB', V_occ_sc, V_occ_sc, t2_lmo_block)

    # Build integrals
    ovL_pair = _build_ovL_tno(Lpq_full, C_lmo, C_tno_sc, pair_lmo)
    ovL_sc_occ = np.einsum('nm,naL->maL', V_occ_sc, ovL_pair)
    vvL_sc = _build_vvL_tno(Lpq_full, C_tno_sc)
    ooL_lmo = _build_ooL_triple(Lpq_full, C_lmo, pair_lmo)
    ooL_sc = np.einsum('pm,qn,pqL->mnL', V_occ_sc, V_occ_sc, ooL_lmo)

    # Project T1 if available
    t1_sc = None
    fvo_sc = None
    if t1_pno is not None:
        nocc_lmo = C_lmo.shape[1]
        t1_lmo_tno = np.zeros((nocc_lmo, n_tno))
        for r in pair_lmo:
            key_rr = (r, r)
            if key_rr in pno_spaces and t1_pno.get(r) is not None:
                C_pno_rr = pno_spaces[key_rr]['C_pno']
                if C_pno_rr.shape[1] > 0 and t1_pno[r].size > 0:
                    U_rr = reduce(np.dot, (C_pno_rr.T, s1e, C_tno_sc))
                    t1_lmo_tno[r] = U_rr.T @ t1_pno[r]
        t1_pair = t1_lmo_tno[pair_lmo, :]
        t1_sc = np.dot(V_occ_sc.T, t1_pair)
        C_lmo_pair = C_lmo[:, pair_lmo]
        F_vo = reduce(np.dot, (C_tno_sc.T, fock_ao, C_lmo_pair))
        fvo_sc = np.dot(F_vo, V_occ_sc)

    return _w3_intermediate(t2_sc_block, ovL_sc_occ, ooL_sc, vvL_sc,
                            eps_occ_sc, eps_tno_sc,
                            t1_sc=t1_sc, fvo_sc=fvo_sc,
                            sum_all_occ=True)


def run_lccsd_t_ext(mf, C_lmo, pno_spaces, strong_pairs,
                    t2_pno_all, occ_cas_idx,
                    t1_pno=None,
                    C_cas_vir=None,
                    vir_cas_idx=None,
                    cas_proj_thresh=0.5,
                    T_CutTNO=1e-9,
                    ncores=1,
                    verbose=None,
                    _pool=None):
    """Compute the (T) energy correction for DLPNO-TCCSD(T).

    Loops over all distinct triples (i<j<k).  Per Lang et al. 2020
    Sec. II.C, double-counting is prevented by zeroing all active T2
    amplitudes before computing (T), NOT by skipping CAS triples.

    The disconnected V intermediate (T1 × vvoo + T2 × Fock) is included
    following the canonical formula from ccsd_t_slow.py:
        z = r3(w + 0.5*v) / D;  E = sum w * z

    Args:
        mf: RHF object (for Fock matrix and with_df).
        C_lmo (np.ndarray): (nao, nocc_lmo) LMO coefficients.
        pno_spaces (dict): Output of pno.make_pnos.
        strong_pairs (list): Strong pairs from pair classification.
        t2_pno_all (dict): Converged T2 amplitudes per pair from run_lccsd.
        occ_cas_idx (np.ndarray): CAS occupied indices.
        t1_pno (dict): i → (n_pno_ii,) local T1 amplitudes in diagonal PNO
            basis (Jiang et al. JCP 2024).  If None, the V intermediate is omitted.
        C_cas_vir (np.ndarray): (nao, n_cas_vir) CAS virtual MO coefficients.
            Required for TCC; if None, T2 zeroing is skipped (warning issued).
        vir_cas_idx (array-like): CAS virtual indices (optional, for logging).
        cas_proj_thresh (float): Zero PNO if CAS-virtual overlap > threshold.
        T_CutTNO (float): TNO eigenvalue threshold (Lang et al. default 1e-9).
        verbose: Verbosity.

    Returns:
        e_t (float): (T) energy correction.
    """
    log = logger.new_logger(mf, verbose)

    if not hasattr(mf, 'with_df') or mf.with_df is None:
        import warnings
        warnings.warn(
            'mf.with_df is None — (T) correction requires density fitting. '
            'Returning e_t = 0.',
            UserWarning, stacklevel=2)
        return 0.0

    occ_cas_set = set(occ_cas_idx.tolist())
    nocc_lmo = C_lmo.shape[1]

    # AO overlap matrix (needed for S-metric PNO overlaps and CAS zeroing)
    s1e = mf.get_ovlp()

    # Zero CAS amplitudes to prevent double-counting static correlation
    # (Lang et al. Section II.A: set all active amplitudes to zero for (T))
    if C_cas_vir is not None and C_cas_vir.shape[1] > 0:
        log.info('TCC (T): zeroing T2 with CAS character (n_cas_vir=%d, thresh=%.2f)',
                 C_cas_vir.shape[1], cas_proj_thresh)
        t2_for_T = _zero_cas_t2_amplitudes(
            t2_pno_all, pno_spaces, occ_cas_idx, C_cas_vir, s1e, cas_proj_thresh)
    else:
        if C_cas_vir is None and len(occ_cas_set) > 0:
            log.warn('TCC (T): C_cas_vir not provided — T2 amplitudes not zeroed!')
        t2_for_T = t2_pno_all

    # Fock diagonal in LMO basis for occupied orbital energies
    fock_ao = mf.get_fock()
    F_lmo = reduce(np.dot, (C_lmo.T, fock_ao, C_lmo))

    occ_list = list(range(nocc_lmo))

    # Enumerate all distinct triples (i <= j <= k).
    # Per Lang et al. 2020 Sec. II.C: all triples are included — double-counting
    # is prevented by zeroing the CAS T2 amplitudes (done above), NOT by
    # skipping triples with occupied indices in the CAS.
    # Strictly ordered i < j < k — the energy formula in _w3_intermediate
    # already sums over all 6 occupied permutations, so each distinct
    # triple must appear exactly once.
    valid_triples = []
    for i in range(nocc_lmo):
        for j in range(i + 1, nocc_lmo):
            for k in range(j + 1, nocc_lmo):
                valid_triples.append((i, j, k))

    # Preload all DF 3-index integrals into memory (single HDF5 read).
    # This makes all subsequent nr_e2 calls thread-safe and eliminates
    # redundant I/O (~252 DF reads → 1).
    import time as _time
    _t0 = _time.perf_counter()
    Lpq_full = _preload_df_integrals(mf.with_df)
    _dt_preload = _time.perf_counter() - _t0
    log.info('(T) preloaded DF integrals: shape=%s, %.1f MB, %.2f s',
             Lpq_full.shape,
             Lpq_full.nbytes / 1e6,
             _dt_preload)

    triple_kwargs = dict(
        pno_spaces=pno_spaces, t2_for_T=t2_for_T,
        Lpq_full=Lpq_full, C_lmo=C_lmo,
        fock_ao=fock_ao, F_lmo=F_lmo, s1e=s1e,
        t1_pno=t1_pno,
        T_CutTNO=T_CutTNO,
    )

    def _do_triple(ijk):
        return _process_one_triple(ijk[0], ijk[1], ijk[2], **triple_kwargs)

    # Degenerate occupied triples: pairs (i,k) with i<k capture
    # {i,i,k} and {i,k,k} contributions (60% of canonical (T)).
    valid_pairs = [(i, k) for i in range(nocc_lmo)
                           for k in range(i + 1, nocc_lmo)]

    def _do_degen(ik):
        return _process_degenerate_pair(ik[0], ik[1], **triple_kwargs)

    # Reuse the shared pool from the driver (same threads as LCCSD stage).
    # Set BLAS to single-thread during pool phase, restore after.
    if _pool is not None:
        et_values = list(_pool.map(_do_triple, valid_triples))
        et_degen_values = list(_pool.map(_do_degen, valid_pairs))
    else:
        et_values = [_do_triple(ijk) for ijk in valid_triples]
        et_degen_values = [_do_degen(ik) for ik in valid_pairs]

    e_t_distinct = sum(et_values)
    n_triples = sum(1 for v in et_values if v != 0.0)

    e_t_degen = sum(et_degen_values)
    n_degen = sum(1 for v in et_degen_values if v != 0.0)

    e_t = e_t_distinct + e_t_degen

    log.info('(T) correction: %d distinct triples, %d degenerate pairs '
             '(%d + %d candidates)',
             n_triples, n_degen, len(valid_triples), len(valid_pairs))
    log.info('E(T) distinct = %.15g', e_t_distinct)
    log.info('E(T) degenerate = %.15g', e_t_degen)
    log.info('E(T) external = %.15g', e_t)

    return e_t


# =============================================================================
# (T1) Iterative Triples — Jiang JCP 2024 / Psi4 lccsd_t_iterations
# =============================================================================

def _build_triple_W_V_T0(i, j, k, pno_spaces, t2_for_T,
                          Lpq_full, C_lmo, fock_ao, F_lmo, s1e,
                          ooL_full, t1_pno=None, T_CutTNO=1e-9):
    """Compute W, V, T0 and TNO data for one triple.

    No occupied semicanonalization — uses diagonal LMO Fock for the denominator.
    The virtual space is semicanonalized (Fock-diagonal TNOs).

    Args:
        ooL_full: (nocc, nocc, naux) prebuilt ooL for all occupied pairs.

    Returns:
        dict with W, V, T, C_tno_sc, eps_tno, n_tno, or None if skipped.
    """
    ij = (min(i, j), max(i, j))
    ik = (min(i, k), max(i, k))
    jk = (min(j, k), max(j, k))

    if (ij not in pno_spaces or ik not in pno_spaces or jk not in pno_spaces):
        return None
    if (ij not in t2_for_T or ik not in t2_for_T or jk not in t2_for_T):
        return None

    C_tno, n_tno = _triple_pno_union(pno_spaces, i, j, k, s1e,
                                      t2_for_T=t2_for_T, T_CutTNO=T_CutTNO)
    if n_tno == 0:
        return None

    n = n_tno
    triple_lmo = [i, j, k]
    nocc = C_lmo.shape[1]

    # Virtual semicanonalization only
    F_tno_full = reduce(np.dot, (C_tno.T, fock_ao, C_tno))
    eps_tno, V_sc = np.linalg.eigh(F_tno_full)
    C_tno_sc = np.dot(C_tno, V_sc)

    # DF integrals in LMO-occ × SC-vir basis
    ovL = _build_ovL_tno(Lpq_full, C_lmo, C_tno_sc, triple_lmo)  # (3, n, naux)
    vvL = _build_vvL_tno(Lpq_full, C_tno_sc)  # (n, n, naux)

    # T2 for all occ paired with triple LMOs: t2_mr[l, r_local] = t2[l, triple[r]]
    def _proj_t2(p, q):
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

    t2_mr = np.zeros((nocc, 3, n, n))
    for r_local, r_global in enumerate(triple_lmo):
        for m in range(nocc):
            t2_mr[m, r_local] = _proj_t2(m, r_global)

    # P_L-assembled W (corrected virtual transposes)
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    trans_axes = [(0,1,2),(0,2,1),(1,0,2),(2,0,1),(1,2,0),(2,1,0)]

    W = np.zeros((n, n, n))
    for pidx, (ip, iq, ir) in enumerate(perms):
        # vvov: sum_f (ip_a|b f) * t2[r,q,c,f]
        K_abf = np.einsum('aL,fbL->abf', ovL[ip], vvL)
        t2_rq = t2_mr[triple_lmo[ir], iq]  # t2[triple[ir], triple[iq]]
        base = np.einsum('abf,cf->abc', K_abf, t2_rq)

        # vooo: -sum_l (ip_a | iq_l) * t2[l, ir, b, c]  (Eq47 factorization)
        A_al = np.einsum('aL,lL->al', ovL[ip], ooL_full[triple_lmo[iq]])
        base -= np.einsum('al,lbc->abc', A_al, t2_mr[:, ir])

        W += base.transpose(trans_axes[pidx])

    # Denominator: diagonal LMO Fock (no occupied SC)
    eps_occ = np.array([F_lmo[ii, ii] for ii in triple_lmo])
    D_occ = eps_occ[0] + eps_occ[1] + eps_occ[2]
    D = eps_tno[:, None, None] + eps_tno[None, :, None] + eps_tno[None, None, :] - D_occ

    T = -W / D

    # V = W + T1 disconnected
    V = W.copy()
    if t1_pno is not None:
        t1_tno = np.zeros((3, n))
        for idx_local, r_global in enumerate(triple_lmo):
            key_rr = (r_global, r_global)
            if key_rr in pno_spaces and t1_pno.get(r_global) is not None:
                C_pno_rr = pno_spaces[key_rr]['C_pno']
                if C_pno_rr.shape[1] > 0 and t1_pno[r_global].size > 0:
                    U_rr = reduce(np.dot, (C_pno_rr.T, s1e, C_tno_sc))
                    t1_tno[idx_local] = U_rr.T @ t1_pno[r_global]
        K_jk = np.einsum('bL,cL->bc', ovL[1], ovL[2])
        K_ik = np.einsum('aL,cL->ac', ovL[0], ovL[2])
        K_ij = np.einsum('aL,bL->ab', ovL[0], ovL[1])
        V += (np.einsum('a,bc->abc', t1_tno[0], K_jk)
              + np.einsum('b,ac->abc', t1_tno[1], K_ik)
              + np.einsum('c,ab->abc', t1_tno[2], K_ij))

    return {
        'W': W, 'V': V, 'T': T,
        'C_tno_sc': C_tno_sc,
        'eps_tno': eps_tno,
        'D': D,
        'n_tno': n_tno,
        'i': i, 'j': j, 'k': k,
    }


def _triples_permuter(X, i_perm, j_perm, k_perm):
    """Permute virtual indices of X[a,b,c] based on occupied ordering.

    X is stored with canonical ordering (i<=j<=k). Given a target occupied
    ordering (i_perm, j_perm, k_perm), return X with virtual indices permuted
    to match. Follows Psi4's triples_permuter() logic.

    The mapping: determine which S_3 element maps the canonical ordering
    to the target ordering, then apply the same permutation to (a,b,c).
    """
    # Determine permutation index from the sorted ordering
    if i_perm <= j_perm and j_perm <= k_perm:
        return X                                        # identity
    elif i_perm <= k_perm and k_perm <= j_perm:
        return X.transpose(0, 2, 1)                     # swap b↔c
    elif j_perm <= i_perm and i_perm <= k_perm:
        return X.transpose(1, 0, 2)                     # swap a↔b
    elif j_perm <= k_perm and k_perm <= i_perm:
        # cycle: (j,k,i) stored as (i',j',k') where i'<=j'<=k'
        # perm_idx 3 → (b,c,a) in Psi4
        return X.transpose(2, 0, 1)                     # a→c, b→a, c→b
    elif k_perm <= i_perm and i_perm <= j_perm:
        # cycle: (k,i,j) stored as (i',j',k') where i'<=j'<=k'
        # perm_idx 4 → (c,a,b) in Psi4
        return X.transpose(1, 2, 0)                     # a→b, b→c, c→a
    else:
        return X.transpose(2, 1, 0)                     # swap a↔c


def _project_t3(T_src, S):
    """Project T3 amplitudes from source TNO basis to target TNO basis.

    Implements: T_tgt[a,b,c] = sum_{a',b',c'} S[a,a'] S[b,b'] S[c,c'] T_src[a',b',c']
    Uses BLAS matmul for efficiency (3 dgemm calls).

    Args:
        T_src: (n_src, n_src, n_src) T3 in source TNO basis.
        S: (n_tgt, n_src) overlap matrix between target and source TNOs.

    Returns:
        T_tgt: (n_tgt, n_tgt, n_tgt) T3 in target TNO basis.
    """
    n_tgt, n_src = S.shape
    # Contract first index (a'): result[i, b', c'] = sum_a' S[i,a'] T[a',b',c']
    tmp = np.dot(S, T_src.reshape(n_src, n_src * n_src))   # (n_tgt, n_src^2)
    tmp = tmp.reshape(n_tgt, n_src, n_src)                  # (i, b', c')
    # Contract second index (b'): put b' as column for matmul
    tmp = tmp.transpose(0, 2, 1).reshape(n_tgt * n_src, n_src)  # (i*n_src+c', b')
    tmp = np.dot(tmp, S.T)                                  # (i*n_src+c', j)
    # Contract third index (c'): put c' as column for matmul
    tmp = tmp.reshape(n_tgt, n_src, n_tgt).transpose(0, 2, 1)  # (i, j, c')
    return np.dot(tmp.reshape(n_tgt * n_tgt, n_src), S.T).reshape(n_tgt, n_tgt, n_tgt)


def _compute_t1_energy(triple_data_list, triple_idx_map, F_lmo, nocc):
    """Compute (T) energy from stored V and T using Eq53 antisymmetrizer.

    For each triple, applies: e_ijk = prefactor * (8V·T - 4V(cba)·T - ...)

    Args:
        triple_data_list: list of dicts from _build_triple_W_V_T0 (or None).
        triple_idx_map: dict (i,j,k) → index into triple_data_list.
        F_lmo: (nocc, nocc) LMO Fock matrix.
        nocc: number of occupied.

    Returns:
        e_t (float), e_ijk_list (list of per-triple energies).
    """
    e_ijk_list = [0.0] * len(triple_data_list)

    for idx, td in enumerate(triple_data_list):
        if td is None:
            continue
        i, j, k = td['i'], td['j'], td['k']
        V = td['V']
        T = td['T']

        # Degeneracy factor
        dij = int(i == j); djk = int(j == k); dik = int(i == k)
        occ_denom = 1 + dij + djk + dik + 2 * dij * djk * dik

        # Eq53 antisymmetrizer: 8V·T - 4V(kji)·T - 4V(ikj)·T - 4V(jik)·T
        #                        + 2V(jki)·T + 2V(kij)·T
        et = (8.0 * np.sum(V * T)
              - 4.0 * np.sum(V.transpose(2, 1, 0) * T)    # V(k,j,i)
              - 4.0 * np.sum(V.transpose(0, 2, 1) * T)    # V(i,k,j)
              - 4.0 * np.sum(V.transpose(1, 0, 2) * T)    # V(j,i,k)
              + 2.0 * np.sum(V.transpose(1, 2, 0) * T)    # V(j,k,i) → transpose(2,0,1) for (b,c,a)
              + 2.0 * np.sum(V.transpose(2, 0, 1) * T))   # V(k,i,j) → transpose(1,2,0) for (c,a,b)

        # Wait: the antisymmetrizer permutes V's virtual indices based on
        # occupied permutation. For V stored as V[a,b,c] (a↔i,b↔j,c↔k),
        # permute(V, k,j,i) means virtual perm matching (k,j,i):
        #   perm_idx=5 → (c,b,a) → V.transpose(2,1,0) ✓
        # permute(V, i,k,j) → perm_idx=1 → (a,c,b) → V.transpose(0,2,1) ✓
        # permute(V, j,i,k) → perm_idx=2 → (b,a,c) → V.transpose(1,0,2) ✓
        # permute(V, j,k,i) → perm_idx=3 → (b,c,a) → V.transpose(2,0,1) ✓
        # permute(V, k,i,j) → perm_idx=4 → (c,a,b) → V.transpose(1,2,0) ✓

        e_ijk_list[idx] = float(et / occ_denom)

    return sum(e_ijk_list), e_ijk_list


def run_lccsd_t1_iterations(mf, C_lmo, pno_spaces, strong_pairs,
                             t2_pno_all, occ_cas_idx,
                             t1_pno=None,
                             C_cas_vir=None,
                             T_CutTNO=1e-9,
                             ncores=1,
                             verbose=None,
                             _pool=None,
                             max_iter=50,
                             e_conv=1e-8,
                             r_conv=1e-6,
                             F_CUT_T=1e-5,
                             T_CUT_ITER=0.0):
    """Compute (T1) iterative triples correction (Jiang JCP 2024).

    Improves upon the (T0) semicanonical approximation by iteratively
    solving for T3 amplitudes including off-diagonal occupied Fock coupling
    between neighboring triples.

    The algorithm (following Psi4's lccsd_t_iterations):
    1. Compute W, V for all triples; initialize T = -W/D (T0).
    2. Iterate:
       R = W + T*D - sum_l F_lmo[l,k] * S_overlap * T3_neighbor
       T -= R/D
    3. Compute energy from converged V and T.

    Args:
        Same as run_lccsd_t_ext, plus:
        max_iter (int): Maximum number of iterations (default 50).
        e_conv (float): Energy convergence threshold (default 1e-8).
        r_conv (float): Residual convergence threshold (default 1e-6).
        F_CUT_T (float): Off-diagonal Fock cutoff for coupling (default 1e-5).
        T_CUT_ITER (float): Skip triples whose energy changed less than
            this fraction (default 0.0 = no skipping).

    Returns:
        e_t (float): Converged (T1) energy correction.
    """
    import time as _time
    log = logger.new_logger(mf, verbose)

    if not hasattr(mf, 'with_df') or mf.with_df is None:
        import warnings
        warnings.warn('mf.with_df is None — (T) correction requires DF.', UserWarning)
        return 0.0

    occ_cas_set = set(occ_cas_idx.tolist()) if occ_cas_idx is not None else set()
    nocc = C_lmo.shape[1]
    s1e = mf.get_ovlp()
    fock_ao = mf.get_fock()
    F_lmo = reduce(np.dot, (C_lmo.T, fock_ao, C_lmo))

    # Zero CAS T2 amplitudes
    if C_cas_vir is not None and C_cas_vir.shape[1] > 0:
        t2_for_T = _zero_cas_t2_amplitudes(
            t2_pno_all, pno_spaces, occ_cas_idx, C_cas_vir, s1e)
    else:
        t2_for_T = t2_pno_all

    # Preload DF integrals
    _t0 = _time.perf_counter()
    Lpq_full = _preload_df_integrals(mf.with_df)
    log.info('(T1) preloaded DF integrals: %.1f MB, %.2f s',
             Lpq_full.nbytes / 1e6, _time.perf_counter() - _t0)

    # Build full ooL (nocc, nocc, naux) once for all triples
    _t0 = _time.perf_counter()
    ooL_full = _build_ooL_triple(Lpq_full, C_lmo, list(range(nocc)))
    log.info('(T1) built ooL_full: shape=%s, %.2f s',
             ooL_full.shape, _time.perf_counter() - _t0)

    # =========================================================================
    # Phase 1: Enumerate triples (i<=j<=k) and compute W, V, T0
    # =========================================================================
    all_triples = []
    for i in range(nocc):
        for j in range(i, nocc):
            for k in range(j, nocc):
                all_triples.append((i, j, k))

    triple_idx_map = {ijk: idx for idx, ijk in enumerate(all_triples)}
    n_triples = len(all_triples)

    _t0 = _time.perf_counter()
    triple_data = [None] * n_triples

    common_kwargs = dict(
        pno_spaces=pno_spaces, t2_for_T=t2_for_T,
        Lpq_full=Lpq_full, C_lmo=C_lmo, fock_ao=fock_ao,
        F_lmo=F_lmo, s1e=s1e, ooL_full=ooL_full,
        t1_pno=t1_pno, T_CutTNO=T_CutTNO,
    )

    def _build_one(ijk):
        return _build_triple_W_V_T0(ijk[0], ijk[1], ijk[2], **common_kwargs)

    if _pool is not None:
        results = list(_pool.map(_build_one, all_triples))
    else:
        results = [_build_one(ijk) for ijk in all_triples]

    for idx, td in enumerate(results):
        triple_data[idx] = td

    n_active = sum(1 for td in triple_data if td is not None)
    _dt = _time.perf_counter() - _t0
    log.info('(T1) Phase 1: %d/%d active triples, %.2f s', n_active, n_triples, _dt)

    if n_active == 0:
        log.info('(T1) No active triples — returning 0.')
        return 0.0

    # Compute initial T0 energy
    e_t0, e_ijk_list = _compute_t1_energy(triple_data, triple_idx_map, F_lmo, nocc)
    log.info('(T1) E(T0) = %.12f', e_t0)

    # =========================================================================
    # Phase 2: Build coupling structure and TNO overlap matrices
    # =========================================================================
    # For each active triple (i,j,k), find neighbors (i,j,l), (i,l,k), (l,j,k)
    # with F_lmo[l, replaced_idx] >= F_CUT_T.
    # Pre-compute TNO→TNO overlap S and cache.

    _t0 = _time.perf_counter()
    # Cache: coupling_info[idx] = list of (neighbor_idx, F_coupling, S_overlap, perm_occ)
    coupling_info = [[] for _ in range(n_triples)]

    for idx, td in enumerate(triple_data):
        if td is None:
            continue
        i, j, k = td['i'], td['j'], td['k']
        n_tgt = td['n_tno']
        C_tgt = td['C_tno_sc']  # (nao, n_tgt)

        for l in range(nocc):
            # Channel 1: (i,j,l) replaces k, coupling F_lmo[l,k]
            if l != k and abs(F_lmo[l, k]) >= F_CUT_T:
                # Canonical ordering of (i,j,l)
                sorted_ijl = tuple(sorted([i, j, l]))
                if sorted_ijl in triple_idx_map:
                    nbr_idx = triple_idx_map[sorted_ijl]
                    nbr_td = triple_data[nbr_idx]
                    if nbr_td is not None:
                        # TNO overlap: S(n_tgt, n_src)
                        S_ov = reduce(np.dot, (C_tgt.T, s1e, nbr_td['C_tno_sc']))
                        coupling_info[idx].append((
                            nbr_idx, -F_lmo[l, k], S_ov,
                            (i, j, l)  # target occ ordering for permuter
                        ))

            # Channel 2: (i,l,k) replaces j, coupling F_lmo[l,j]
            if l != j and abs(F_lmo[l, j]) >= F_CUT_T:
                sorted_ilk = tuple(sorted([i, l, k]))
                if sorted_ilk in triple_idx_map:
                    nbr_idx = triple_idx_map[sorted_ilk]
                    nbr_td = triple_data[nbr_idx]
                    if nbr_td is not None:
                        S_ov = reduce(np.dot, (C_tgt.T, s1e, nbr_td['C_tno_sc']))
                        coupling_info[idx].append((
                            nbr_idx, -F_lmo[l, j], S_ov,
                            (i, l, k)
                        ))

            # Channel 3: (l,j,k) replaces i, coupling F_lmo[l,i]
            if l != i and abs(F_lmo[l, i]) >= F_CUT_T:
                sorted_ljk = tuple(sorted([l, j, k]))
                if sorted_ljk in triple_idx_map:
                    nbr_idx = triple_idx_map[sorted_ljk]
                    nbr_td = triple_data[nbr_idx]
                    if nbr_td is not None:
                        S_ov = reduce(np.dot, (C_tgt.T, s1e, nbr_td['C_tno_sc']))
                        coupling_info[idx].append((
                            nbr_idx, -F_lmo[l, i], S_ov,
                            (l, j, k)
                        ))

    n_couplings = sum(len(ci) for ci in coupling_info)
    _dt = _time.perf_counter() - _t0
    log.info('(T1) Phase 2: %d coupling terms, %.2f s', n_couplings, _dt)

    # =========================================================================
    # Phase 3: Jacobi iterations
    # =========================================================================
    log.info('')
    log.info('  ==> Local CCSD(T1) Iterations <==')
    log.info('')
    log.info('  E_CONVERGENCE = %.2e', e_conv)
    log.info('  R_CONVERGENCE = %.2e', r_conv)
    log.info('  F_CUT_T       = %.2e', F_CUT_T)
    log.info('')
    log.info('  %5s %18s %12s %12s %8s',
             'Iter', 'Corr. Energy', 'Delta E', 'Max R', 'Time')

    e_prev = e_t0
    e_ijk_old = list(e_ijk_list)

    for iteration in range(1, max_iter + 1):
        _t_iter = _time.perf_counter()
        r_max_list = [0.0] * n_triples

        # Snapshot current T for Jacobi semantics (read from old, write to new)
        T_snapshot = [td['T'].copy() if td is not None else None
                      for td in triple_data]

        for idx, td in enumerate(triple_data):
            if td is None:
                continue

            if T_CUT_ITER > 0 and abs(e_ijk_list[idx] - e_ijk_old[idx]) < abs(e_ijk_old[idx] * T_CUT_ITER):
                continue

            W = td['W']
            T_old = T_snapshot[idx]
            D = td['D']

            R = W + T_old * D

            for nbr_idx, f_coupling, S_ov, perm_occ in coupling_info[idx]:
                T_nbr = T_snapshot[nbr_idx]
                if T_nbr is None:
                    continue
                T_perm = _triples_permuter(T_nbr, *perm_occ)
                T_proj = _project_t3(T_perm, S_ov)
                R += f_coupling * T_proj

            td['T'] = T_old - R / D
            r_max_list[idx] = float(np.sqrt(np.mean(R**2)))

        # Compute energy
        e_ijk_old = list(e_ijk_list)
        e_curr, e_ijk_list = _compute_t1_energy(triple_data, triple_idx_map, F_lmo, nocc)

        r_max = max(r_max_list)
        delta_e = e_curr - e_prev
        _dt_iter = _time.perf_counter() - _t_iter

        log.info('  %5d %18.12f %12.3e %12.3e %8.1f',
                 iteration, e_curr, delta_e, r_max, _dt_iter)

        e_converged = abs(delta_e) < e_conv
        r_converged = abs(r_max) < r_conv

        if e_converged and r_converged:
            log.info('')
            log.info('  (T1) converged in %d iterations.', iteration)
            break

        e_prev = e_curr
    else:
        log.warn('  (T1) NOT converged after %d iterations!', max_iter)

    e_t = e_curr
    log.info('E(T1) = %.15g', e_t)
    return e_t
