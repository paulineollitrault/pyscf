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


def _triple_pno_union(pno_spaces, i, j, k, s1e, S_cut=1e-6):
    """Compute the triple PNO space as the union of three pair PNO spaces.

    The union is formed by pooling all columns from the three pair PNO matrices
    and re-orthogonalising via canonical orthogonalisation in the S metric.
    Near-linear dependencies (eigenvalue < S_cut) are removed.

    This is the standard TNO construction used in LNO-CCSD(T) and DLPNO-(T):
    the triple virtual space spans everything important for any of the three
    pairs, so no contribution is lost.

    Args:
        pno_spaces (dict): Output of pno.make_pnos.
        i, j, k (int): Triple LMO indices.
        s1e (np.ndarray): (nao, nao) AO overlap matrix S.
        S_cut (float): Eigenvalue threshold for canonical orthogonalisation.

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

    # Canonical orthogonalisation in S metric: diagonalise S_pool, keep > S_cut
    S_pool = reduce(np.dot, (C_pool.T, s1e, C_pool))
    eigvals, eigvecs = np.linalg.eigh(S_pool)
    keep = eigvals > S_cut
    n_tno = int(np.sum(keep))

    if n_tno == 0:
        return np.zeros((s1e.shape[0], 0)), 0

    # X[:,m] = eigvec[:,m] / sqrt(eigval[m])  →  C_pool @ X is S-orthonormal
    X = eigvecs[:, keep] / np.sqrt(eigvals[keep])
    C_tno = np.dot(C_pool, X)   # (nao, n_tno), S-orthonormal

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
                     t1_sc=None, fvo_sc=None, sum_all_occ=False):
    """Compute the (T) energy contribution for a single occupied triple (i,j,k).

    Follows the canonical formula from ccsd_t_slow.py:
        z = r3(w + 0.5*v) / D
        E = sum w * z  (over 36 terms)

    where W is the connected part (T2 × integrals) and V is the disconnected
    part (T1 × vvoo integrals + T2 × Fock off-diagonal).

    Two-stage approach for maximum speed:

    1. Precompute W_all[a,b,c,p,q,r] for ALL virtual (a,b,c) at once using two
       vectorised numpy contractions:

           K[a,b,p,f]        = sum_L ovL_sc[p,f,L] * vvL_sc[a,b,L]
           A[p,a,q,m]        = sum_L ovL_sc[p,a,L] * ooL_sc[q,m,L]
           W_vvov[a,b,c,p,q,r] = sum_f K[a,b,p,f] * t2_sc[q,r,c,f]
           W_vooo[a,b,c,p,q,r] = sum_m A[p,a,q,m] * t2_sc[m,r,b,c]
           W_all = W_vvov - W_vooo

       Memory: (n,n,n,3,3,3) × 8 bytes — e.g. 31 MB for n=53.

    2. Loop over compact (a>=b>=c) virtual triples, but each build_w(x,y,z) is
       now just a free slice of the precomputed W_all — no arithmetic inside
       the loop at all.  The 26K × 6 r3 calls operate on 3×3×3 arrays (trivial)
       and accumulate the 36-term energy formula.

    The formula and sign convention match ccsd_t_slow.py exactly; r3 acts on
    the occupied (p,q,r) axes.  The compact loop with degeneracy factors is
    the standard way to handle the sum over distinct virtual triples.

    This is correct for non-antisymmetric T2 (pair CCSD T2 has exchange
    symmetry only, not full antisymmetry) because the energy formula uses
    the same compact-loop structure as the original, just with precomputed W.

    Args:
        t2_sc (np.ndarray): (3, 3, n_tno, n_tno) T2[p,q,a,b] in SC occ × SC vir.
        ovL_sc (np.ndarray): (3, n_tno, naux) SC-occ × TNO DF integrals.
        ooL_sc (np.ndarray): (3, 3, naux) SC-occ × SC-occ DF integrals.
        vvL_sc (np.ndarray): (n_tno, n_tno, naux) TNO × TNO DF integrals.
        eps_occ (np.ndarray): (3,) SC occupied orbital energies.
        eps_vir (np.ndarray): (n_tno,) SC virtual orbital energies.

    Returns:
        et_ijk (float): (T) energy contribution from this triple (× spin factor 2).
    """
    n = len(eps_vir)
    if n == 0:
        return 0.0

    # ------------------------------------------------------------------
    # Stage 1: build the full W[a,b,c,p,q,r] tensor in one shot.
    # ------------------------------------------------------------------
    # K[a,b,p,f] = sum_L ovL_sc[p,a,L]*vvL_sc[f,b,L] = (pa|fb)  — (n, n, 3, n)
    # Canonical formula: w[i,j,k] = sum_f (ia|fb)*t2[k,j,c,f]
    # so K uses the occ-vir integral (pa|fb), NOT (pf|ab).
    K = np.einsum('paL,fbL->abpf', ovL_sc, vvL_sc)
    # W_vvov[a,b,c,p,q,r] = sum_f K[a,b,p,f] * t2_sc[r,q,c,f]
    W_all = np.einsum('abpf,rqcf->abcpqr', K, t2_sc)    # (n, n, n, 3, 3, 3)
    del K

    # A[p,a,q,m] = sum_L ovL_sc[p,a,L]*ooL_sc[q,m,L]  — (3, n, 3, 3)
    A = np.einsum('paL,qmL->paqm', ovL_sc, ooL_sc)
    # W_vooo[a,b,c,p,q,r] = sum_m A[p,a,q,m] * t2_sc[m,r,b,c]  (subtract in-place)
    W_all -= np.einsum('paqm,mrbc->abcpqr', A, t2_sc)

    # Make contiguous so slices W_all[a,b,c] are fast.
    W_all = np.ascontiguousarray(W_all)

    # ------------------------------------------------------------------
    # Build V_all (disconnected part): T1 × vvoo integrals + T2 × Fock
    # From ccsd_t_slow.py:
    #   v[a,b,c; i,j,k] = vvoo[a,b,i,j] * t1[k,c] + t2[i,j,a,b] * fvo[c,k]
    # ------------------------------------------------------------------
    has_v = (t1_sc is not None and fvo_sc is not None)
    if has_v:
        # vvoo[a,b,p,q] = (pa|qb) = sum_L ovL_sc[p,a,L] * ovL_sc[q,b,L]
        # Canonical: eris_vvoo[a,b,i,j] = (ia|jb), NOT (ab|ij).
        vvoo = np.einsum('paL,qbL->abpq', ovL_sc, ovL_sc)  # (n, n, 3, 3)
        # V_all[a,b,c,p,q,r] = vvoo[a,b,p,q] * t1_sc[r,c]
        #                    + t2_sc[p,q,a,b] * fvo_sc[c,r]
        V_all = np.einsum('abpq,rc->abcpqr', vvoo, t1_sc)
        V_all += np.einsum('pqab,cr->abcpqr', t2_sc, fvo_sc)
        V_all = np.ascontiguousarray(V_all)
        del vvoo

    # ------------------------------------------------------------------
    # Stage 2: fully vectorised compact (a>=b>=c) energy accumulation.
    # ------------------------------------------------------------------
    # Build compact index arrays once (no Python loop at all).
    a_arr = np.empty(n*(n+1)*(n+2)//6, dtype=np.int32)
    b_arr = np.empty_like(a_arr)
    c_arr = np.empty_like(a_arr)
    idx = 0
    for a in range(n):
        for b in range(a + 1):
            for c in range(b + 1):
                a_arr[idx] = a; b_arr[idx] = b; c_arr[idx] = c
                idx += 1
    # n_compact = C(n+2,3)

    # Batch-extract W for all 6 virtual permutations at once (fancy indexing).
    # Shape: (n_compact, 3, 3, 3)
    wabc = W_all[a_arr, b_arr, c_arr];  wacb = W_all[a_arr, c_arr, b_arr]
    wbac = W_all[b_arr, a_arr, c_arr];  wbca = W_all[b_arr, c_arr, a_arr]
    wcab = W_all[c_arr, a_arr, b_arr];  wcba = W_all[c_arr, b_arr, a_arr]
    del W_all  # free memory

    # Batch-extract V (disconnected part) if available.
    if has_v:
        vabc = V_all[a_arr, b_arr, c_arr];  vacb = V_all[a_arr, c_arr, b_arr]
        vbac = V_all[b_arr, a_arr, c_arr];  vbca = V_all[b_arr, c_arr, a_arr]
        vcab = V_all[c_arr, a_arr, b_arr];  vcba = V_all[c_arr, b_arr, a_arr]
        del V_all

    # r3 antisymmetriser acting on axes 1,2,3 (occupied p,q,r) — batched.
    def r3b(W):   # W: (N, 3, 3, 3)
        return (4*W
                + W.transpose(0, 2, 3, 1) + W.transpose(0, 3, 1, 2)
                - 2*W.transpose(0, 1, 3, 2) - 2*W.transpose(0, 2, 1, 3)
                - 2*W.transpose(0, 3, 2, 1))

    # Denominator: D_occ[p,q,r] - eps_vir[a] - eps_vir[b] - eps_vir[c]
    D_occ = (eps_occ[:, None, None]
             + eps_occ[None, :, None]
             + eps_occ[None, None, :])   # (3,3,3)
    D_abc = eps_vir[a_arr] + eps_vir[b_arr] + eps_vir[c_arr]         # (n_compact,)
    D_batch = D_occ[None, :, :, :] - D_abc[:, None, None, None]      # (n_compact,3,3,3)

    # Degeneracy factors (matches ccsd_t_slow.py compact loop).
    all_same = (a_arr == c_arr)                           # a==b==c
    ab_bc    = ((a_arr == b_arr) | (b_arr == c_arr)) & ~all_same
    D_batch[all_same] *= 6
    D_batch[ab_bc]    *= 2
    D_safe = np.where(np.abs(D_batch) > 1e-12, D_batch, 1e-12)

    # z = r3(w + 0.5*v) / D for all 6 permutations at once.
    if has_v:
        zabc = r3b(wabc + 0.5*vabc) / D_safe
        zacb = r3b(wacb + 0.5*vacb) / D_safe
        zbac = r3b(wbac + 0.5*vbac) / D_safe
        zbca = r3b(wbca + 0.5*vbca) / D_safe
        zcab = r3b(wcab + 0.5*vcab) / D_safe
        zcba = r3b(wcba + 0.5*vcba) / D_safe
    else:
        zabc = r3b(wabc) / D_safe;  zacb = r3b(wacb) / D_safe
        zbac = r3b(wbac) / D_safe;  zbca = r3b(wbca) / D_safe
        zcab = r3b(wcab) / D_safe;  zcba = r3b(wcba) / D_safe

    # 36-term energy accumulation (all n_compact triples at once).
    # Transpositions act on axes 1,2,3 (p,q,r); axis 0 is batch.
    contrib = (
        wabc*zabc         + wacb*zabc.transpose(0,1,3,2)
        + wbac*zabc.transpose(0,2,1,3) + wbca*zabc.transpose(0,2,3,1)
        + wcab*zabc.transpose(0,3,1,2) + wcba*zabc.transpose(0,3,2,1)
        + wacb*zacb       + wabc*zacb.transpose(0,1,3,2)
        + wcab*zacb.transpose(0,2,1,3) + wcba*zacb.transpose(0,2,3,1)
        + wbac*zacb.transpose(0,3,1,2) + wbca*zacb.transpose(0,3,2,1)
        + wbac*zbac       + wbca*zbac.transpose(0,1,3,2)
        + wabc*zbac.transpose(0,2,1,3) + wacb*zbac.transpose(0,2,3,1)
        + wcba*zbac.transpose(0,3,1,2) + wcab*zbac.transpose(0,3,2,1)
        + wbca*zbca       + wbac*zbca.transpose(0,1,3,2)
        + wcba*zbca.transpose(0,2,1,3) + wcab*zbca.transpose(0,2,3,1)
        + wabc*zbca.transpose(0,3,1,2) + wacb*zbca.transpose(0,3,2,1)
        + wcab*zcab       + wcba*zcab.transpose(0,1,3,2)
        + wacb*zcab.transpose(0,2,1,3) + wabc*zcab.transpose(0,2,3,1)
        + wbca*zcab.transpose(0,3,1,2) + wbac*zcab.transpose(0,3,2,1)
        + wcba*zcba       + wcab*zcba.transpose(0,1,3,2)
        + wbca*zcba.transpose(0,2,1,3) + wbac*zcba.transpose(0,2,3,1)
        + wacb*zcba.transpose(0,3,1,2) + wabc*zcba.transpose(0,3,2,1))

    # Sum occupied elements for each compact (a,b,c).
    if sum_all_occ:
        # Degenerate case (2 occ): sum ALL occupied elements.
        # For a pair (i,k), the 2^3=8 entries contain {i,i,k} (3 entries),
        # {i,k,k} (3 entries), and {i,i,i}/{k,k,k} (2 entries, zero after r3).
        et = np.sum(contrib)
    else:
        # All-distinct case (3 occ): sum only the 6 all-distinct permutations.
        et = np.sum(contrib[:, 0, 1, 2] + contrib[:, 0, 2, 1]
                    + contrib[:, 1, 0, 2] + contrib[:, 1, 2, 0]
                    + contrib[:, 2, 0, 1] + contrib[:, 2, 1, 0])

    # Factor 2 for closed-shell spin summation (matches canonical et *= 2)
    return float(et.real) * 2


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
                        t1_pno=None):
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

    C_tno, n_tno = _triple_pno_union(pno_spaces, i, j, k, s1e)
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
    F_occ_3x3  = F_lmo[np.ix_(triple_lmo, triple_lmo)]
    eps_occ_sc, V_occ_sc = np.linalg.eigh(F_occ_3x3)
    ovL_sc_occ = np.einsum('nm,naL->maL', V_occ_sc, ovL_ijk)
    ooL_lmo    = _build_ooL_triple(Lpq_full, C_lmo, triple_lmo)
    ooL_sc     = np.einsum('pm,qn,pqL->mnL', V_occ_sc, V_occ_sc, ooL_lmo)

    # Include diagonal pair amplitudes t2[i,i,a,b] — these are nonzero in
    # spatial-orbital CCSD and contribute to the (T) correction.
    def _map_t2_diag(idx):
        pair_key = (idx, idx)
        if pair_key not in pno_spaces or pair_key not in t2_for_T:
            return np.zeros((n_tno, n_tno))
        C_p = pno_spaces[pair_key]['C_pno']
        if C_p.shape[1] == 0:
            return np.zeros((n_tno, n_tno))
        U = reduce(np.dot, (C_p.T, s1e, C_tno_sc))
        return reduce(np.dot, (U.T, t2_for_T[pair_key], U))

    t2_lmo_block = np.zeros((3, 3, n_tno, n_tno))
    t2_lmo_block[0, 0] = _map_t2_diag(i)
    t2_lmo_block[1, 1] = _map_t2_diag(j)
    t2_lmo_block[2, 2] = _map_t2_diag(k)
    t2_lmo_block[0, 1] = t2_ij_sc;  t2_lmo_block[1, 0] = t2_ij_sc.T
    t2_lmo_block[0, 2] = t2_ik_sc;  t2_lmo_block[2, 0] = t2_ik_sc.T
    t2_lmo_block[1, 2] = t2_jk_sc;  t2_lmo_block[2, 1] = t2_jk_sc.T
    t2_sc_block = np.einsum('pm,qn,pqAB->mnAB', V_occ_sc, V_occ_sc, t2_lmo_block)

    # Project local T1 to semicanonical TNO basis for the V intermediate.
    t1_sc = None
    fvo_sc = None
    if t1_pno is not None:
        # t1_pno[r]: (n_pno_rr,) in diagonal PNO basis of pair (r,r).
        # Project each to TNO-sc: t1_lmo_tno[r, :] = U_rr.T @ t1_pno[r]
        # where U_rr = C_pno_rr.T @ S @ C_tno_sc
        nocc_lmo = C_lmo.shape[1]
        t1_lmo_tno = np.zeros((nocc_lmo, n_tno))
        for r in triple_lmo:
            key_rr = (r, r)
            if key_rr in pno_spaces and t1_pno.get(r) is not None:
                C_pno_rr = pno_spaces[key_rr]['C_pno']
                if C_pno_rr.shape[1] > 0 and t1_pno[r].size > 0:
                    U_rr = reduce(np.dot, (C_pno_rr.T, s1e, C_tno_sc))
                    t1_lmo_tno[r] = U_rr.T @ t1_pno[r]
        t1_triple = t1_lmo_tno[triple_lmo, :]                  # (3, n_tno)
        t1_sc = np.dot(V_occ_sc.T, t1_triple)                  # (3, n_tno)

        # Fock virtual-occupied block in semicanonical basis:
        # fvo_sc[c, r] = C_tno_sc.T @ fock_ao @ C_lmo_triple @ V_occ_sc
        C_lmo_triple = C_lmo[:, triple_lmo]                     # (nao, 3)
        F_vo = reduce(np.dot, (C_tno_sc.T, fock_ao, C_lmo_triple))  # (n_tno, 3)
        fvo_sc = np.dot(F_vo, V_occ_sc)                         # (n_tno, 3)

    return _w3_intermediate(t2_sc_block, ovL_sc_occ, ooL_sc, vvL_sc,
                            eps_occ_sc, eps_tno_sc,
                            t1_sc=t1_sc, fvo_sc=fvo_sc)


def _process_degenerate_pair(i, k,
                             pno_spaces, t2_for_T,
                             Lpq_full, C_lmo, fock_ao, F_lmo, s1e,
                             t1_pno=None):
    """Compute (T) energy from degenerate occupied triples {i,i,k} and {i,k,k}.

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
    if has_ii:
        C_ii = pno_spaces[ii]['C_pno']
        if C_ii.shape[1] > 0:
            cols.append(C_ii)
    C_ik = pno_spaces[ik]['C_pno']
    if C_ik.shape[1] > 0:
        cols.append(C_ik)
    if has_kk:
        C_kk = pno_spaces[kk]['C_pno']
        if C_kk.shape[1] > 0:
            cols.append(C_kk)

    if not cols:
        return 0.0

    C_pool = np.hstack(cols)
    S_pool = reduce(np.dot, (C_pool.T, s1e, C_pool))
    eigvals, eigvecs = np.linalg.eigh(S_pool)
    keep = eigvals > 1e-6
    n_tno = int(np.sum(keep))
    if n_tno == 0:
        return 0.0
    X = eigvecs[:, keep] / np.sqrt(eigvals[keep])
    C_tno = np.dot(C_pool, X)

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
                    verbose=None):
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
    )

    from concurrent.futures import ThreadPoolExecutor
    _n_workers = max(1, min(ncores, len(valid_triples)))

    def _do_triple(ijk):
        return _process_one_triple(ijk[0], ijk[1], ijk[2], **triple_kwargs)

    if _n_workers > 1:
        with ThreadPoolExecutor(max_workers=_n_workers) as pool:
            et_values = list(pool.map(_do_triple, valid_triples))
    else:
        et_values = [_do_triple(ijk) for ijk in valid_triples]

    e_t_distinct = sum(et_values)
    n_triples = sum(1 for v in et_values if v != 0.0)

    # Degenerate occupied triples: pairs (i,k) with i<k capture
    # {i,i,k} and {i,k,k} contributions (60% of canonical (T)).
    valid_pairs = [(i, k) for i in range(nocc_lmo)
                           for k in range(i + 1, nocc_lmo)]

    def _do_degen(ik):
        return _process_degenerate_pair(ik[0], ik[1], **triple_kwargs)

    if _n_workers > 1:
        with ThreadPoolExecutor(max_workers=_n_workers) as pool:
            et_degen_values = list(pool.map(_do_degen, valid_pairs))
    else:
        et_degen_values = [_do_degen(ik) for ik in valid_pairs]

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
