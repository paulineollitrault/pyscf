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


def _triple_pno_union_psi4(pno_spaces, i, j, k, C_pao, S_pao_full, F_pao_full,
                            t2_for_T=None, T_CutTNO=1e-9, S_cut_domain=1e-8,
                            pao_domains_triple=None):
    """Psi4-style TNO construction in orthogonalized triple-PAO basis.

    Mirrors Psi4 DLPNOCCSD_T::tno_transform (triples.cc:431):
    (1) triple_paos = union of pair_paos for (ij), (jk), (ik) in PAO
        index space (no orthogonalization yet).
    (2) Canonical orthogonalization of S_pao[triple_paos, triple_paos]
        → X_pao_ijk (npao_ijk, npao_can_ijk).
    (3) Project each pair's density D_ij = Tt_ij @ T_ij.T + Tt_ij.T @ T_ij
        into the triple's orthogonal-PAO basis.
    (4) D_ijk = (D_ij + D_jk + D_ik) / 3; diagonalize → TNOs.
    (5) Truncate at T_CutTNO, canonicalize in F basis → X_tno_canonical.
    (6) X_tno_ijk = X_pao_ijk @ X_tno_canonical
        shape (npao_ijk, n_tno) — maps triple's RAW PAO domain to
        canonical TNO. This is what Psi4 stores as X_tno_[ijk] and what
        compute_lccsd_t0 uses to slice sparse qia[Q]/qij[Q]/qab[Q].

    Returns:
        C_tno_sc   : (nao, n_tno) TNO in AO basis (S-orthonormal)
        n_tno      : int
        X_tno_ijk  : (npao_ijk, n_tno) triple-PAO → canonical-TNO
        triple_paos: (npao_ijk,) int PAO indices (sorted)
        eps_tno_sc : (n_tno,) TNO orbital energies
    """
    ij = (min(i, j), max(i, j))
    ik = (min(i, k), max(i, k))
    jk = (min(j, k), max(j, k))
    nao = C_pao.shape[0]

    if not all(key in pno_spaces for key in (ij, ik, jk)):
        return None, 0, None, None, None

    # (1) triple_paos: union of per-LMO PAO domains [i, j, k]
    # Psi4 lmotriplet_to_paos_[ijk] = union of lmo_to_paos[i, j, k] rebuilt
    # at the triples stage with T_CUT_DO_TRIPLES (triples.cc:316-329).
    # Fallback: union of pair_paos for (ij, jk, ik) — the CCSD-stage domain.
    if pao_domains_triple is not None:
        pao_set = set()
        for lmo in (i, j, k):
            pao_set.update(int(x) for x in np.asarray(pao_domains_triple[lmo]).tolist())
    else:
        pao_set = set()
        for key in (ij, jk, ik):
            pp = pno_spaces[key].get('pair_paos')
            if pp is not None:
                pao_set.update(int(x) for x in np.asarray(pp).tolist())
    if not pao_set:
        return np.zeros((nao, 0)), 0, None, None, None
    triple_paos = np.array(sorted(pao_set), dtype=np.int64)

    # (2) Canonical orthogonalization of triple's PAO domain
    from pyscf.cc.dlpno_tccsd.local_orbs import orthogonalize_pao_domain
    C_orth_ijk, X_pao_ijk = orthogonalize_pao_domain(
        C_pao, S_pao_full, triple_paos,
        S_cut=S_cut_domain, method='psi4')
    npao_can_ijk = X_pao_ijk.shape[1]
    if npao_can_ijk == 0:
        return np.zeros((nao, 0)), 0, None, None, None

    # F_pao in orthogonalized triple basis
    F_dom = F_pao_full[np.ix_(triple_paos, triple_paos)]
    F_orth_ijk = X_pao_ijk.T @ F_dom @ X_pao_ijk

    # (3) Pair-PNO to triple-orth-PAO projections (S-weighted)
    # S_proj[key] has shape (n_pno_pair, npao_can_ijk):
    #   S_proj = X_pno[pair].T @ S_pao[pair_paos, triple_paos] @ X_pao_ijk
    S_projs = {}
    for key in (ij, jk, ik):
        pp_pair = np.asarray(pno_spaces[key]['pair_paos'])
        X_pno_pair = pno_spaces[key].get('X_pno')
        if X_pno_pair is None or X_pno_pair.shape[1] == 0:
            S_projs[key] = None
            continue
        S_pair_triple = S_pao_full[np.ix_(pp_pair, triple_paos)]
        S_projs[key] = X_pno_pair.T @ S_pair_triple @ X_pao_ijk

    # (4) Sum projected pair densities
    D_ijk = np.zeros((npao_can_ijk, npao_can_ijk))
    for key, ii_lmo, jj_lmo in [(ij, i, j), (jk, j, k), (ik, i, k)]:
        S = S_projs[key]
        if S is None:
            continue
        if t2_for_T is not None and key in t2_for_T:
            T2_p = t2_for_T[key]
        elif pno_spaces[key].get('T2_pno') is not None:
            T2_p = pno_spaces[key]['T2_pno']
        else:
            continue
        Tt_p = 2.0 * T2_p - T2_p.T
        D_pair = Tt_p @ T2_p.T + Tt_p.T @ T2_p
        if ii_lmo == jj_lmo:
            D_pair *= 0.5
        D_ijk += S.T @ D_pair @ S
    D_ijk /= 3.0

    # (5) Diagonalize, sort, truncate
    tno_occ, tno_vecs = np.linalg.eigh(D_ijk)
    order = np.argsort(tno_occ)[::-1]
    tno_occ = tno_occ[order]
    tno_vecs = tno_vecs[:, order]
    keep = np.abs(tno_occ) >= T_CutTNO
    n_tno = int(np.sum(keep))
    if n_tno == 0:
        n_tno = 1
        keep[0] = True
    X_tno_initial = tno_vecs[:, keep]

    # (6) Canonicalize in F basis
    F_in_tno = X_tno_initial.T @ F_orth_ijk @ X_tno_initial
    eps_tno_sc, tno_canon = np.linalg.eigh(F_in_tno)
    X_tno_canonical = X_tno_initial @ tno_canon

    # (7) X_tno_ijk maps triple's raw PAO domain → canonical TNO
    X_tno_ijk = X_pao_ijk @ X_tno_canonical              # (npao_ijk, n_tno)

    # TNO in AO basis (S-orthonormal by construction)
    C_tno_sc = C_pao[:, triple_paos] @ X_tno_ijk         # (nao, n_tno)

    return C_tno_sc, n_tno, X_tno_ijk, triple_paos, eps_tno_sc


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
                     ooL_sc_full=None, t2_sc_full=None,
                     K_ooov=None):
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

    # Precompute K_ab[ip] = Σ_L ovL_sc[ip,a,L] * vvL_sc[b,f,L] for the 3
    # distinct occupied-index slots. The perm loop visits each ip twice,
    # so caching here avoids 3 redundant einsums per triple (×442 triples).
    # tensordot is ~3-5× faster than einsum for simple 3-axis contractions
    # since it goes straight to BLAS without the einsum path planner.
    K_ab_cache = [None, None, None]
    for ip in range(3):
        # shape (n, n, n): indexed [a, f, b]
        t = np.tensordot(ovL_sc[ip], vvL_sc, axes=([1], [2]))
        # reorder to [a, b, f]
        K_ab_cache[ip] = t.transpose(0, 2, 1)

    # Permutation index tables — shared by K_ovvv phase and K_ooov phase.
    p_table = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    _ip_arr = np.array([p[0] for p in p_table])   # (6,) ovL-slot of each perm
    _iq_arr = np.array([p[1] for p in p_table])   # (6,) oo-slot
    _ir_arr = np.array([p[2] for p in p_table])   # (6,) t2-slot

    # --- Phase 1: K_ovvv contribution (Psi4 triples.cc:827-830) ---
    # Per perm (A, B, C): base += K_Avvv[a, b, f] * t2_{CB}[c, f]
    W = np.zeros((n, n, n))
    for pidx in range(6):
        ip, iq, ir = p_table[pidx]
        K_ab = K_ab_cache[ip]
        t2_rq = t2_sc[ir, iq]
        base = (K_ab.reshape(n * n, n) @ t2_rq.T).reshape(n, n, n)
        W += trans[pidx](base)

    # --- Phase 2: K_ooov subtraction (Psi4 triples.cc:832-848) ---
    # Psi4 streams m one at a time: per perm (A, B, C), per m, build
    # T_Am[a,b] from pair (A, m) and subtract T_Am[a,b] × K_{Bo Cv}[m, c].
    # We vectorize the m-sum (and batch across the 6 perms) into one
    # matmul: sub[p, a, b, c] = Σ_m K_ooov[ip,iq,a,m] · t2[m, ir, b, c].
    if K_ooov is not None and t2_sc_full is not None:
        K_batch = K_ooov[_ip_arr, _iq_arr]                         # (6, n, m_dom)
        t2_batch = t2_sc_full[:, _ir_arr].transpose(1, 0, 2, 3)    # (6, m_dom, n, n)
        m_dom = K_batch.shape[2]
        sub = np.matmul(
            K_batch, t2_batch.reshape(6, m_dom, n * n)
        ).reshape(6, n, n, n)
        # Apply per-perm virtual-index transpose (inverts P_L) and subtract.
        W -= sub[0]
        W -= sub[1].transpose(0, 2, 1)
        W -= sub[2].transpose(1, 0, 2)
        W -= sub[3].transpose(2, 0, 1)
        W -= sub[4].transpose(1, 2, 0)
        W -= sub[5].transpose(2, 1, 0)
    else:
        # Fallback for legacy callers (no pre-built K_ooov, or in-block-only).
        for pidx in range(6):
            ip, iq, ir = p_table[pidx]
            if ooL_sc_full is not None and t2_sc_full is not None:
                A_al = ovL_sc[ip] @ ooL_sc_full[iq].T
                t2_mbc = t2_sc_full[:, ir]
            else:
                A_al = ovL_sc[ip] @ ooL_sc[iq].T
                t2_mbc = t2_sc[:, ir]
            m = t2_mbc.shape[0]
            W -= trans[pidx](
                (A_al @ t2_mbc.reshape(m, n * n)).reshape(n, n, n))

    # --- T = -W/D ---
    T = -W / D

    # --- V = W + T1 disconnected (Psi4 lines 930-962) ---
    V = W.copy()
    if t1_sc is not None and fvo_sc is not None:
        K_jk = ovL_sc[1] @ ovL_sc[2].T
        K_ik = ovL_sc[0] @ ovL_sc[2].T
        K_ij = ovL_sc[0] @ ovL_sc[1].T
        V += (t1_sc[0][:, None, None] * K_jk[None, :, :]
              + t1_sc[1][None, :, None] * K_ik[:, None, :]
              + t1_sc[2][None, None, :] * K_ij[:, :, None])

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


def _build_triple_local_DF(i, j, k, X_tno_ijk, triple_paos, triple_domain,
                            sparse_df, screening, j2c_full,
                            lmo_aux_mask):
    """Psi4-style per-triple DF integrals on a local aux domain.

    Mirrors Psi4 DLPNOCCSD_T::compute_lccsd_t0 (triples.cc:667+). Aux Q's
    are grouped by atom center; each center's Q-stack is contracted in
    one batched matmul, amortizing BLAS-3 across the n_aux_at_A Q's of
    each atom (same pattern as compute_cc_integrals for CCSD pairs).

      q_iv[q, a_tno] = Σ_{u ∈ paos_ext[Q] ∩ triple_paos}
                          qia[Q][i_sparse, u_in_Q]
                          * X_tno_ijk[u_in_triple_paos, a_tno]

      q_vv[q, a, b]  = Σ_{u, v ∈ paos_ext[Q] ∩ triple_paos}
                          X_tno_ijk[u_tp, a]
                          * qab[Q][u_in_Q, v_in_Q]
                          * X_tno_ijk[v_tp, b]

      q_io[q, m]     = qij[Q][i_sparse, m_sparse]

    After the per-center loop, apply local J^{-1/2}.

    Args:
        X_tno_ijk: (npao_ijk, n_tno) — triple's raw-PAO → canonical-TNO
            transform (from _triple_pno_union_psi4).
        triple_paos: (npao_ijk,) global PAO indices in the triple.
        triple_domain: list of global LMO indices (sorted) for ooL rows.

    Returns:
        ovL_sc  (3, n_tno, naux_ijk), vvL_sc (n_tno, n_tno, naux_ijk),
        ooL_sc  (3, n_domain, naux_ijk), all fitted with local J^{-1/2}.
    """
    qij_atom = sparse_df.get('qij_atom')
    qia_atom = sparse_df.get('qia_atom')
    qab_atom = sparse_df.get('qab_atom')
    aux_pos_in_atom = sparse_df.get('aux_pos_in_atom')
    aux_atom_ids = sparse_df.get('aux_atom_ids')
    riatom_to_lmos_ext_dense = screening['riatom_to_lmos_ext_dense']
    riatom_to_paos_ext_dense = screening['riatom_to_paos_ext_dense']

    aux_mask = lmo_aux_mask[i] | lmo_aux_mask[j] | lmo_aux_mask[k]
    aux_idx = np.where(aux_mask)[0]
    naux_ijk = aux_idx.size
    n_tno = X_tno_ijk.shape[1]
    n_domain = len(triple_domain)

    if naux_ijk == 0 or n_tno == 0:
        return (np.zeros((3, n_tno, 0)),
                np.zeros((n_tno, n_tno, 0)),
                np.zeros((3, n_domain, 0)))

    dom_arr = np.asarray(triple_domain, dtype=np.int64)
    ijk_global = np.array([i, j, k], dtype=np.int64)

    # Raw (pre-metric) tensors
    ovL_raw = np.zeros((3, n_tno, naux_ijk))
    vvL_raw = np.zeros((n_tno, n_tno, naux_ijk))
    ooL_raw = np.zeros((3, n_domain, naux_ijk))

    # Group by center: aux Q's of the same atom share paos_ext/lmos_ext,
    # so we can do one batched BLAS-3 operation per center.
    centers_of_aux = aux_atom_ids[aux_idx]
    unique_centers = np.unique(centers_of_aux)

    for centerQ in unique_centers:
        centerQ = int(centerQ)
        mask_c = centers_of_aux == centerQ
        local_Q = np.where(mask_c)[0]              # (nQ_c,) positions in naux_ijk
        global_Q = aux_idx[local_Q]                # global aux indices
        atom_pos = aux_pos_in_atom[global_Q]       # positions within centerQ's stack
        nQ_c = local_Q.size

        # Occupied-index sparse positions (same for all Q's in this center)
        i_s = int(riatom_to_lmos_ext_dense[centerQ, i])
        j_s = int(riatom_to_lmos_ext_dense[centerQ, j])
        k_s = int(riatom_to_lmos_ext_dense[centerQ, k])
        occ_sp = (i_s, j_s, k_s)

        # ------------- ooL (qij path, PAO-free) -------------
        # Slice LMO axes FIRST (nl_c is O(1)), then Q: avoids copying the
        # (nQ_A, nl_c, nl_c) full stack when atom_pos is a subset.
        if qij_atom[centerQ] is not None:
            m_sparse_arr = riatom_to_lmos_ext_dense[centerQ, dom_arr]  # (n_dom,)
            m_valid_mask = m_sparse_arr >= 0
            dom_local = np.where(m_valid_mask)[0]       # positions in triple_domain
            m_sparse_kept = m_sparse_arr[m_valid_mask]  # positions in centerQ's lmo stack
            if dom_local.size > 0:
                qij_A = qij_atom[centerQ]               # (nQ_A, nl, nl) — shared
                for idx in range(3):
                    occ = occ_sp[idx]
                    if occ >= 0:
                        # qij_A[:, occ, m_sparse_kept] — (nQ_A, n_valid), then pick atom_pos
                        vals_full = qij_A[:, occ, m_sparse_kept]  # (nQ_A, n_valid)
                        ooL_raw[idx][np.ix_(dom_local, local_Q)] = vals_full[atom_pos].T

        # ------------- ovL & vvL (PAO paths) -------------
        if qia_atom[centerQ] is None and qab_atom[centerQ] is None:
            continue
        # Triple PAO → centerQ's pao stack position (-1 if outside)
        tp_pos_in_Q = riatom_to_paos_ext_dense[centerQ, triple_paos]  # (npao_ijk,)
        valid_tp = np.where(tp_pos_in_Q >= 0)[0]        # positions in triple_paos
        valid_u_Q = tp_pos_in_Q[valid_tp]               # positions in centerQ's pao stack
        nu = valid_tp.size
        if nu == 0:
            continue
        X_Q = X_tno_ijk[valid_tp]                       # (nu, n_tno) — shared by all Q

        # ovL: slice PAO axis FIRST on the full (nQ_A, nl, np_A) stack so
        # the (copy-triggering) atom_pos slice sees only (nQ_A, nl, nu)
        # — scales like nu, not np_A.
        if qia_atom[centerQ] is not None:
            qia_A = qia_atom[centerQ]                           # (nQ_A, nl, np_A)
            qia_A_u = qia_A[:, :, valid_u_Q]                    # (nQ_A, nl, nu)
            qia_stack_cut = qia_A_u[atom_pos]                   # (nQ_c, nl, nu)
            for idx in range(3):
                occ = occ_sp[idx]
                if occ >= 0:
                    block = qia_stack_cut[:, occ, :] @ X_Q      # (nQ_c, n_tno)
                    ovL_raw[idx][:, local_Q] = block.T          # (n_tno, nQ_c)

        # vvL: same trick — PAO-PAO slice first (nu² << np_A²), then Q-slice.
        if qab_atom[centerQ] is not None:
            qab_A = qab_atom[centerQ]                           # (nQ_A, np_A, np_A)
            qab_A_uu = qab_A[:, valid_u_Q[:, None],
                             valid_u_Q[None, :]]                # (nQ_A, nu, nu)
            qab_stack_cut = qab_A_uu[atom_pos]                  # (nQ_c, nu, nu)
            tmp = qab_stack_cut @ X_Q                           # (nQ_c, nu, n_tno)
            vvL_c = np.matmul(X_Q.T, tmp)                       # (nQ_c, n_tno, n_tno)
            vvL_raw[:, :, local_Q] = vvL_c.transpose(1, 2, 0)

    # Local J^{-1/2}
    j_loc = j2c_full[np.ix_(aux_idx, aux_idx)]
    evals, evecs = np.linalg.eigh(j_loc)
    keep_e = evals > 1e-14
    jhi = (evecs[:, keep_e] * (1.0 / np.sqrt(evals[keep_e]))
           ) @ evecs[:, keep_e].T

    ovL_sc = ovL_raw @ jhi
    vvL_sc = vvL_raw @ jhi
    ooL_sc = ooL_raw @ jhi

    return ovL_sc, vvL_sc, ooL_sc


def _process_one_triple(i, j, k,
                        pno_spaces, t2_for_T,
                        Lpq_full, C_lmo, fock_ao, F_lmo, s1e,
                        t1_pno=None, T_CutTNO=1e-9,
                        nonneg_set=None,
                        sparse_df=None, screening=None,
                        j2c_full=None, lmo_aux_mask=None, C_pao=None,
                        S_pao_full=None, F_pao_full=None,
                        pao_domains_triple=None):
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

    # Psi4-style TNO build in orthogonalized triple-PAO basis when the
    # sparse-DF data is available — gives us X_tno_ijk and triple_paos
    # needed by _build_triple_local_DF. Falls back to the legacy
    # AO-basis construction otherwise.
    _use_psi4_tno = (sparse_df is not None and screening is not None
                     and j2c_full is not None and lmo_aux_mask is not None
                     and C_pao is not None
                     and S_pao_full is not None and F_pao_full is not None)
    if _use_psi4_tno:
        C_tno_sc, n_tno, X_tno_ijk, triple_paos, eps_tno_sc = \
            _triple_pno_union_psi4(
                pno_spaces, i, j, k, C_pao, S_pao_full, F_pao_full,
                t2_for_T=t2_for_T, T_CutTNO=T_CutTNO,
                pao_domains_triple=pao_domains_triple)
        if n_tno == 0:
            return 0.0
    else:
        C_tno, n_tno = _triple_pno_union(pno_spaces, i, j, k, s1e,
                                          t2_for_T=t2_for_T,
                                          T_CutTNO=T_CutTNO)
        if n_tno == 0:
            return 0.0
        F_tno_full = reduce(np.dot, (C_tno.T, fock_ao, C_tno))
        eps_tno_sc, V_sc = np.linalg.eigh(F_tno_full)
        C_tno_sc = np.dot(C_tno, V_sc)
        X_tno_ijk = None
        triple_paos = None

    # Triple-local aux domain: raw 3-center @ local J^{-1/2}_local gives
    # a proper local DF fit for this triple. Matches CCSD pair approach
    # (compute_cc_integrals_sparse) and dramatically reduces the naux
    # dimension of per-triple ovL / vvL / ooL tensor transforms from
    # O(N) to O(1).
    triple_lmo = [i, j, k]
    nocc_lmo = C_lmo.shape[1]

    # Triple-local LMO domain: m contributes to the vooo (A*t2) term only
    # if pairs (m, i), (m, j), (m, k) all survive (non-negligible).
    _tr_pair = lambda a, b: (min(a, b), max(a, b))
    _domain_set = nonneg_set if nonneg_set is not None else set(t2_for_T.keys())
    triple_domain = sorted(
        m for m in range(nocc_lmo)
        if _tr_pair(m, i) in _domain_set
        and _tr_pair(m, j) in _domain_set
        and _tr_pair(m, k) in _domain_set)
    _m_pos = {m_global: m_local for m_local, m_global in enumerate(triple_domain)}

    eps_occ = np.array([F_lmo[ii, ii] for ii in triple_lmo])

    # DF integrals: Psi4-style triple-local aux path when available.
    if _use_psi4_tno and X_tno_ijk is not None:
        ovL_ijk, vvL_sc, ooL_lmo_full = _build_triple_local_DF(
            i, j, k, X_tno_ijk, triple_paos, triple_domain,
            sparse_df, screening, j2c_full, lmo_aux_mask)
        _sparse_path_prebuilt_rows = True
    else:
        ovL_ijk = _build_ovL_tno(Lpq_full, C_lmo, C_tno_sc, [i, j, k])
        vvL_sc = _build_vvL_tno(Lpq_full, C_tno_sc)
        ooL_lmo_full = _build_ooL_triple(Lpq_full, C_lmo, triple_domain)
        _sparse_path_prebuilt_rows = False

    # --- Pair PNO → triple TNO overlap matrix U[pk] in PAO basis ---
    # Psi4-style (O(1) per pair): U = X_pno[pk].T @ S_pao[pair_paos, triple_paos]
    #                                  @ X_tno_ijk.
    # Precompute W = S_pao[:, triple_paos] @ X_tno_ijk once per triple
    # (shape: (nao_pao, n_tno)) so each pair lookup reduces to one int-slice
    # + one matmul (avoids O(nao²) s1e matmul AND np.ix_ fancy indexing).
    # Fallback (AO-basis C_pno path): O(nao²) per pair via s1e.
    _pao_basis = (_use_psi4_tno and X_tno_ijk is not None
                  and S_pao_full is not None and triple_paos is not None)
    if _pao_basis:
        _W_pao_tno = S_pao_full[:, triple_paos] @ X_tno_ijk  # (nao_pao, n_tno)
    else:
        _W_pao_tno = None
    _U_cache = {}

    def _U_for(pk):
        if pk in _U_cache:
            return _U_cache[pk]
        if pk not in pno_spaces:
            _U_cache[pk] = None
            return None
        if _pao_basis:
            X_pno = pno_spaces[pk].get('X_pno')
            pp = pno_spaces[pk].get('pair_paos')
            if X_pno is None or pp is None or X_pno.shape[1] == 0:
                _U_cache[pk] = None
                return None
            # W_pk = W_pao_tno[pair_paos, :], shape (n_pao_pk, n_tno)
            # U    = X_pno.T @ W_pk, shape (n_pno_pk, n_tno)
            U = X_pno.T @ _W_pao_tno[pp]
        else:
            C_p = pno_spaces[pk]['C_pno']
            if C_p.shape[1] == 0:
                _U_cache[pk] = None
                return None
            U = reduce(np.dot, (C_p.T, s1e, C_tno_sc))
        _U_cache[pk] = U
        return U

    # T2 projection: pair-PNO → triple-TNO.
    def _proj_t2(p, q):
        pk = (min(p, q), max(p, q))
        if pk not in t2_for_T:
            return np.zeros((n_tno, n_tno))
        U = _U_for(pk)
        if U is None:
            return np.zeros((n_tno, n_tno))
        t2_proj = U.T @ t2_for_T[pk] @ U
        if p > q:
            t2_proj = t2_proj.T
        return t2_proj

    m_dom_size = len(triple_domain)
    t2_mr = np.zeros((m_dom_size, 3, n_tno, n_tno))
    for r_local, r_global in enumerate(triple_lmo):
        for m_local, m_global in enumerate(triple_domain):
            t2_mr[m_local, r_local] = _proj_t2(m_global, r_global)

    # T2 block for the 3 triple LMOs (used by _w3_intermediate)
    t2_block = np.zeros((3, 3, n_tno, n_tno))
    for p in range(3):
        for q in range(3):
            t2_block[p, q] = t2_mr[_m_pos[triple_lmo[p]], q]

    # Project local T1 to TNO basis for the V intermediate.
    t1_lmo = None
    fvo = None
    if t1_pno is not None:
        t1_lmo_tno = np.zeros((3, n_tno))
        for idx_local, r_global in enumerate(triple_lmo):
            key_rr = (r_global, r_global)
            if key_rr in pno_spaces and t1_pno.get(r_global) is not None:
                if t1_pno[r_global].size == 0:
                    continue
                U_rr = _U_for(key_rr)
                if U_rr is not None:
                    t1_lmo_tno[idx_local] = U_rr.T @ t1_pno[r_global]
        t1_lmo = t1_lmo_tno

        C_lmo_triple = C_lmo[:, triple_lmo]
        fvo = reduce(np.dot, (C_tno_sc.T, fock_ao, C_lmo_triple))

    # ooL_sc_full shape (3, n_domain, naux).  In the sparse path it is
    # already indexed by (i,j,k) rows; in the full-naux path we still
    # need to pick out the rows corresponding to i,j,k.
    if _sparse_path_prebuilt_rows:
        _ooL_for_w3 = ooL_lmo_full
    else:
        _triple_rows = np.array([_m_pos[x] for x in triple_lmo])
        _ooL_for_w3 = ooL_lmo_full[_triple_rows]

    # Pre-build the 6 K_ooov permutation matrices once per triple (Psi4
    # triples.cc:775-809). Each K_ooov[p, q] = Σ_Q ovL_ijk[p, :, Q] *
    # ooL[q, :, Q], shape (n_tno, n_domain). One batched matmul is
    # cheaper than recomputing the same thing inside the 6-perm loop of
    # _w3_intermediate, and it's the prerequisite for the per-m vooo
    # restructure (step 2).
    naux_ijk = ovL_ijk.shape[2]
    ov_flat = ovL_ijk.reshape(3 * n_tno, naux_ijk)
    oo_flat = _ooL_for_w3.reshape(3 * m_dom_size, naux_ijk)
    K_ooov = (ov_flat @ oo_flat.T).reshape(
        3, n_tno, 3, m_dom_size).transpose(0, 2, 1, 3)

    return _w3_intermediate(t2_block, ovL_ijk, None, vvL_sc,
                            eps_occ, eps_tno_sc,
                            t1_sc=t1_lmo, fvo_sc=fvo,
                            occ_indices=(i, j, k),
                            ooL_sc_full=_ooL_for_w3,
                            t2_sc_full=t2_mr,
                            K_ooov=K_ooov)


def _process_degenerate_pair(i, k,
                             pno_spaces, t2_for_T,
                             Lpq_full, C_lmo, fock_ao, F_lmo, s1e,
                             t1_pno=None, T_CutTNO=1e-9,
                             nonneg_set=None,
                             sparse_df=None, screening=None,
                             j2c_full=None, lmo_aux_mask=None,
                             C_pao=None,
                             S_pao_full=None, F_pao_full=None,
                             pao_domains_triple=None):
    """Compute (T) energy from degenerate occupied triples {i,i,k} and {i,k,k}.

    With the Eq 53 formula, _process_one_triple handles degenerate triples
    correctly via the occ_indices degeneracy factor. We just call it for both
    (i,i,k) and (i,k,k).
    """
    kwargs = dict(pno_spaces=pno_spaces, t2_for_T=t2_for_T,
                  Lpq_full=Lpq_full, C_lmo=C_lmo, fock_ao=fock_ao,
                  F_lmo=F_lmo, s1e=s1e, t1_pno=t1_pno, T_CutTNO=T_CutTNO,
                  nonneg_set=nonneg_set,
                  sparse_df=sparse_df, screening=screening,
                  j2c_full=j2c_full, lmo_aux_mask=lmo_aux_mask,
                  C_pao=C_pao,
                  S_pao_full=S_pao_full, F_pao_full=F_pao_full,
                  pao_domains_triple=pao_domains_triple)
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
                    T_CutTriplesWeak=1e-7,
                    ncores=1,
                    negligible_pairs=None,
                    weak_pairs=None,
                    C_pao=None,
                    doi_iu=None,
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

    # Enumerate distinct triples (i<j<k) following Psi4's triples_sparsity
    # (dlpno/triples.cc::triples_sparsity prescreening block).
    #   (1) k iterated over pair_lmo_idx[ij] (= lmopair_to_lmos_[ij] in Psi4)
    #       — the pair's interacting-LMO domain, NOT all of nocc.
    #   (2) max-2-weak-pair constraint: at least one of (ij), (ik), (jk)
    #       must be strong, else the triple is dropped.
    _negl_set = set((min(p), max(p)) for p in (negligible_pairs or []))
    _weak_set = set((min(p), max(p)) for p in (weak_pairs or []))
    _tT_set = set(k for k in t2_for_T.keys() if k not in _negl_set)

    # Build pair_lmo_idx locally — m is in pair (i,j)'s domain iff both
    # (i,m) and (j,m) are non-negligible.  (Same construction as in
    # lccsd.py _run_dlpno_lccsd.)
    pair_lmo_idx = {}
    for key in _tT_set:
        i, j = key
        dom = [m for m in range(nocc_lmo)
               if (min(i, m), max(i, m)) in _tT_set
               and (min(j, m), max(j, m)) in _tT_set]
        pair_lmo_idx[key] = dom

    valid_triples = []
    _n_before_weak_screen = 0
    for ij in _tT_set:
        i, j = ij
        if i >= j:
            continue        # strict i < j; degenerate (i,i,k) handled below
        for k in pair_lmo_idx.get(ij, []):
            if k <= j:                         # strict i < j < k
                continue
            ik = (i, k)
            jk = (j, k)
            if ik not in _tT_set or jk not in _tT_set:
                continue
            _n_before_weak_screen += 1
            weak_count = ((ij in _weak_set) + (ik in _weak_set) +
                          (jk in _weak_set))
            if weak_count > 2:
                continue
            valid_triples.append((i, j, k))
    _n_all = nocc_lmo * (nocc_lmo - 1) * (nocc_lmo - 2) // 6
    print(f'  (T) triples after Psi4-style screening: '
          f'{len(valid_triples)} / {_n_all} '
          f'({100.0 * len(valid_triples) / max(_n_all, 1):.1f}%) '
          f'[pre-weak-count: {_n_before_weak_screen}]', flush=True)

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

    # --- Build sparse-DF infrastructure for triple-local aux path ---
    # Mirrors Psi4 DLPNOCCSD_T::compute_lccsd_t0: per-triple integrals
    # are constructed by slicing pre-computed sparse arrays qij[Q],
    # qia[Q], qab[Q] by the triple's aux_ijk/LMO_ijk/PAO_ijk domains,
    # then applying a local J^{-1/2}.  _process_one_triple will use
    # this path when all of {sparse_df, screening, j2c_full,
    # lmo_aux_mask, C_pao} are available.
    from pyscf.cc.dlpno_tccsd.local_df import (
        build_screening_maps as _build_screen,
        build_sparse_df_arrays as _build_sparse,
    )
    # Psi4/Jiang defaults for the TRIPLES stage (distinct from CCSD T_CUT_MKN=1e-3):
    #   T_CUT_MKN_TRIPLES = 1e-2   (read_options.cc line 2573)
    #   T_CUT_DO_TRIPLES  = 1e-2   (read_options.cc line 2575; applied to PAO domains)
    # The triples-stage thresholds are 10× LOOSER than CCSD to keep per-triple
    # aux/PAO domains small while not hurting (T) accuracy.
    # Psi4 (T) thresholds — separate defaults for tight pass and (T0) prescreen.
    # read_options.cc L2565-2575. The PRE thresholds are deliberately looser so
    # the prescreen pass runs on a SMALLER sparse-DF infrastructure (cheap (T0)),
    # while the tight pass on surviving triples uses the full-accuracy thresholds.
    _T_CUT_MKN_TRIPLES     = 1e-2
    _T_CUT_DO_TRIPLES      = 1e-2
    _T_CUT_MKN_TRIPLES_PRE = 1e-1    # 10× looser than tight
    _T_CUT_DO_TRIPLES_PRE  = 2e-2    # 2× looser than tight
    _T_CUT_CLMO = 1e-3
    _auxmol = mf.with_df.auxmol if hasattr(mf.with_df, 'auxmol') else None
    _j2c_full = None
    if _auxmol is not None and C_pao is not None:
        _natm = mf.mol.natm
        _ao_labels = mf.mol.ao_labels(fmt=False)
        _atom_ids = np.array([lbl[0] for lbl in _ao_labels])
        _aux_atom_ids = np.array(
            [lbl[0] for lbl in _auxmol.ao_labels(fmt=False)])
        _atom_to_ao = [np.where(_atom_ids == a)[0] for a in range(_natm)]

        def _build_triples_infrastructure(T_CUT_MKN, T_CUT_DO, label):
            """Build lmo_aux_mask, pao_domains, screening, and sparse_df stacks
            at the given thresholds. Mirrors Psi4 triples_sparsity(prescreening).
            """
            # --- lmo_aux_mask: per-LMO Mulliken-weighted aux-atom mask ---
            mask_rows = []
            for ii in range(nocc_lmo):
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
                mkn_pop = np.zeros(_natm)
                for _a in range(_natm):
                    mask_a = (_atom_ids == _a)
                    mkn_pop[_a] = (np.sum(contrib_u[mask_a, :])
                                   + np.sum(contrib_v[:, mask_a]))
                mask_rows.append(
                    np.isin(_aux_atom_ids,
                            np.where(np.abs(mkn_pop) > T_CUT_MKN)[0]))
            lmo_aux_mask = np.array(mask_rows)

            # --- pao_domains: DOI > T_CUT_DO, atom-complete (triples.cc:316-329) ---
            pao_domains = []
            if doi_iu is not None:
                for ii in range(nocc_lmo):
                    doi = doi_iu[ii]
                    pao_inds = np.where(doi > T_CUT_DO)[0]
                    if pao_inds.size == 0:
                        pao_inds = np.array([int(np.argmax(doi))])
                    atoms_in = np.unique(_atom_ids[pao_inds])
                    domain_i = np.concatenate([_atom_to_ao[a] for a in atoms_in])
                    pao_domains.append(np.sort(domain_i))
            else:
                for ii in range(nocc_lmo):
                    key_ii = (ii, ii)
                    if (key_ii in pno_spaces
                            and pno_spaces[key_ii].get('pair_paos') is not None):
                        pao_domains.append(
                            np.asarray(pno_spaces[key_ii]['pair_paos']))
                    else:
                        pao_domains.append(np.zeros(0, dtype=int))

            _dom_sizes = [len(d) for d in pao_domains]
            log.info('(T) [%s] T_CUT_MKN=%.1e T_CUT_DO=%.1e: '
                     'avg PAOs/LMO=%.1f (min=%d, max=%d)', label,
                     T_CUT_MKN, T_CUT_DO,
                     float(np.mean(_dom_sizes)),
                     int(min(_dom_sizes)), int(max(_dom_sizes)))

            # --- screening + sparse_df + per-atom stacks ---
            strong_pair_keys = list(t2_for_T.keys())
            screening = _build_screen(
                mf.mol, _auxmol, C_lmo, pao_domains, s1e, strong_pair_keys,
                T_CUT_MKN=T_CUT_MKN, T_CUT_CLMO=_T_CUT_CLMO, C_pao=C_pao)
            sparse_df = _build_sparse(mf.mol, _auxmol, C_lmo, C_pao, screening)
            aux_atom_ids_arr = screening['aux_atom_ids']
            naux_total = len(sparse_df['qij'])
            aux_at_atom_list = [np.where(aux_atom_ids_arr == A)[0]
                                for A in range(_natm)]
            qij_stack = [None] * _natm
            qia_stack = [None] * _natm
            qab_stack = [None] * _natm
            ri_lmos_ext = screening['riatom_to_lmos_ext']
            ri_paos_ext = screening['riatom_to_paos_ext']
            for A in range(_natm):
                Qs_A = aux_at_atom_list[A]
                if len(Qs_A) == 0:
                    continue
                nl_A = len(ri_lmos_ext[A])
                np_A = len(ri_paos_ext[A])
                if nl_A > 0:
                    qij_stack[A] = np.stack(
                        [sparse_df['qij'][Q] for Q in Qs_A])
                if nl_A > 0 and np_A > 0:
                    qia_stack[A] = np.stack(
                        [sparse_df['qia'][Q] for Q in Qs_A])
                if np_A > 0:
                    qab_stack[A] = np.stack(
                        [sparse_df['qab'][Q] for Q in Qs_A])
            aux_pos_in_atom = -np.ones(naux_total, dtype=np.int64)
            for A in range(_natm):
                for pos, Q in enumerate(aux_at_atom_list[A]):
                    aux_pos_in_atom[Q] = pos
            sparse_df['qij_atom'] = qij_stack
            sparse_df['qia_atom'] = qia_stack
            sparse_df['qab_atom'] = qab_stack
            sparse_df['aux_at_atom'] = aux_at_atom_list
            sparse_df['aux_pos_in_atom'] = aux_pos_in_atom
            sparse_df['aux_atom_ids'] = aux_atom_ids_arr
            return lmo_aux_mask, pao_domains, screening, sparse_df

        # Tight infrastructure (used for the final (T) pass on surviving triples).
        _lmo_aux_mask, _pao_domains, _screening, _sparse_df = \
            _build_triples_infrastructure(
                _T_CUT_MKN_TRIPLES, _T_CUT_DO_TRIPLES, 'tight')
        _j2c_full = _auxmol.intor('int2c2e')
        _S_pao_full = C_pao.T @ s1e @ C_pao
        _F_pao_full = C_pao.T @ fock_ao @ C_pao

        # Prescreen infrastructure (used by the loose (T0) pass).
        if T_CutTriplesWeak > 0.0:
            _lmo_aux_mask_pre, _pao_domains_pre, _screening_pre, _sparse_df_pre = \
                _build_triples_infrastructure(
                    _T_CUT_MKN_TRIPLES_PRE, _T_CUT_DO_TRIPLES_PRE, 'prescreen')
        else:
            _lmo_aux_mask_pre = _lmo_aux_mask
            _pao_domains_pre = _pao_domains
            _screening_pre = _screening
            _sparse_df_pre = _sparse_df
    else:
        _lmo_aux_mask = None
        _sparse_df = None
        _screening = None
        _pao_domains = None
        _lmo_aux_mask_pre = None
        _sparse_df_pre = None
        _screening_pre = None
        _pao_domains_pre = None
        _S_pao_full = None
        _F_pao_full = None

    # Diagnostic: per-triple screening tightness. naux_ijk / naux_full and
    # len(triple_domain) / nocc tell us whether aux-Q and occupied domains
    # are O(1) per triple (good scaling) or growing with system size (bad).
    if valid_triples:
        _naux_tot = _lmo_aux_mask.shape[1] if _lmo_aux_mask is not None else 0
        _ss_aux, _ss_dom = [], []
        for _i, _j, _k in valid_triples:
            if _lmo_aux_mask is not None:
                _ss_aux.append(int((_lmo_aux_mask[_i] | _lmo_aux_mask[_j]
                                   | _lmo_aux_mask[_k]).sum()))
            _dom = [m for m in range(nocc_lmo)
                    if (min(m, _i), max(m, _i)) in _tT_set
                    and (min(m, _j), max(m, _j)) in _tT_set
                    and (min(m, _k), max(m, _k)) in _tT_set]
            _ss_dom.append(len(_dom))
        if _ss_aux:
            print(f'  (T) naux_ijk: avg={np.mean(_ss_aux):.1f}/{_naux_tot} '
                  f'({100.0*np.mean(_ss_aux)/max(_naux_tot,1):.1f}%)  '
                  f'min={min(_ss_aux)}  max={max(_ss_aux)}',
                  flush=True)
        print(f'  (T) triple_domain: avg={np.mean(_ss_dom):.1f}/{nocc_lmo} '
              f'({100.0*np.mean(_ss_dom)/max(nocc_lmo,1):.1f}%)  '
              f'min={min(_ss_dom)}  max={max(_ss_dom)}',
              flush=True)

    triple_kwargs = dict(
        pno_spaces=pno_spaces, t2_for_T=t2_for_T,
        Lpq_full=Lpq_full, C_lmo=C_lmo,
        fock_ao=fock_ao, F_lmo=F_lmo, s1e=s1e,
        t1_pno=t1_pno,
        T_CutTNO=T_CutTNO,
        nonneg_set=_tT_set,
        sparse_df=_sparse_df, screening=_screening,
        j2c_full=_j2c_full, lmo_aux_mask=_lmo_aux_mask,
        C_pao=C_pao,
        S_pao_full=_S_pao_full, F_pao_full=_F_pao_full,
        pao_domains_triple=_pao_domains,
    )

    def _do_triple(ijk):
        return _process_one_triple(ijk[0], ijk[1], ijk[2], **triple_kwargs)

    # ------------------------------------------------------------------
    # (3) SC-MP2 (T0) prescreen to drop low-contribution triples.
    # Psi4 (triples.cc:1296-1326) runs the prescreen on a SEPARATE sparse-DF
    # infrastructure built with LOOSER thresholds (T_CUT_MKN_TRIPLES_PRE,
    # T_CUT_DO_TRIPLES_PRE, T_CUT_TNO_PRE). Surviving triples are then
    # rerun at the tight thresholds. This makes the (T0) pass genuinely
    # cheap while preserving accuracy on the triples that matter.
    # ------------------------------------------------------------------
    _T_CUT_TNO_PRE = max(T_CutTNO, 1e-7)   # Psi4 read_options.cc:2565
    _T_CUT_TRIPLES_WEAK = T_CutTriplesWeak
    e_t_screened = 0.0   # dropped-triple contribution, ADDED BACK at end
    if (len(valid_triples) > 0 and _T_CUT_TNO_PRE > T_CutTNO
            and _T_CUT_TRIPLES_WEAK > 0.0):
        # Build kwargs for the prescreen pass: LOOSE thresholds on every
        # axis — PAO domain, aux-Q mask, sparse-DF stacks, screening maps,
        # and TNO cutoff. Everything else (triple-paos union, F/S matrices,
        # t2 amplitudes) is shared with the tight pass.
        _pre_kwargs = dict(triple_kwargs)
        _pre_kwargs['T_CutTNO'] = _T_CUT_TNO_PRE
        _pre_kwargs['sparse_df'] = _sparse_df_pre
        _pre_kwargs['screening'] = _screening_pre
        _pre_kwargs['lmo_aux_mask'] = _lmo_aux_mask_pre
        _pre_kwargs['pao_domains_triple'] = _pao_domains_pre

        def _pre_triple(ijk):
            return _process_one_triple(ijk[0], ijk[1], ijk[2], **_pre_kwargs)

        if _pool is not None:
            _et_pre = list(_pool.map(_pre_triple, valid_triples))
        else:
            _et_pre = [_pre_triple(ijk) for ijk in valid_triples]

        _kept = [ijk for ijk, e in zip(valid_triples, _et_pre)
                 if abs(e) >= _T_CUT_TRIPLES_WEAK]
        # Psi4 (triples.cc:235, 252): the prescreen energy of *dropped*
        # triples is tracked as de_lccsd_t_screened_ and ADDED to the
        # final (T) correction — that way the prescreen error is
        # bounded by T_CUT_TRIPLES_WEAK × n_triples rather than the
        # full per-triple contribution.
        e_t_screened = sum(e for e in _et_pre
                           if abs(e) < _T_CUT_TRIPLES_WEAK)
        print(f'  (T) SC-MP2 prescreen: kept {len(_kept)} / '
              f'{len(valid_triples)} ({100.0*len(_kept)/len(valid_triples):.1f}%)'
              f', screened-back energy = {e_t_screened:.3e} Eh',
              flush=True)
        valid_triples = _kept

    # Degenerate occupied triples: pairs (i,k) with i<k capture
    # {i,i,k} and {i,k,k} contributions (60% of canonical (T)).
    valid_pairs = [(i, k) for i in range(nocc_lmo)
                           for k in range(i + 1, nocc_lmo)
                           if (i, k) in _tT_set]

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

    # Add back the SC-MP2 prescreen energy of dropped triples — the error
    # vs a full tight-TNO calculation is bounded by T_CUT_TRIPLES_WEAK per
    # dropped triple (Psi4 de_lccsd_t_screened_ convention).
    e_t = e_t_distinct + e_t_degen + e_t_screened

    log.info('(T) correction: %d distinct triples, %d degenerate pairs '
             '(%d + %d candidates)',
             n_triples, n_degen, len(valid_triples), len(valid_pairs))
    log.info('E(T) distinct = %.15g', e_t_distinct)
    log.info('E(T) degenerate = %.15g', e_t_degen)
    log.info('E(T) screened-back = %.15g', e_t_screened)
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
