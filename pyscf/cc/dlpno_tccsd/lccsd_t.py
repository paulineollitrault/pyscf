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

CAS exclusion (from Lee & Head-Gordon tccsd_t.py, ecCC-TCC):
    Skip triple (i,j,k) if ALL of {i,j,k} in occ_cas_idx.

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


def _build_ovL_tno(with_df, C_lmo, C_tno, lmo_indices):
    """Build (LMO_i, TNO_a | L) 3-index DF tensor for selected LMOs.

    Args:
        with_df: PySCF DF object.
        C_lmo (np.ndarray): (nao, nocc_lmo) LMO coefficients.
        C_tno (np.ndarray): (nao, n_tno) TNO coefficients in AO basis.
        lmo_indices (list): LMO orbital indices to compute (e.g. [i, j, k]).

    Returns:
        ovL (np.ndarray): (len(lmo_indices), n_tno, naux)
    """
    nao, nocc_lmo = C_lmo.shape
    n_tno = C_tno.shape[1]
    naux = with_df.get_naoaux()
    n_sel = len(lmo_indices)

    # Build C_occ for selected LMOs only
    C_occ_sel = C_lmo[:, lmo_indices]   # (nao, n_sel)
    nmo_sel = n_sel + n_tno
    mo = np.asarray(np.hstack((C_occ_sel, C_tno)), order='F')
    ijslice = (0, n_sel, n_sel, nmo_sel)

    ovL = np.empty((n_sel, n_tno, naux))
    buf = None
    p1 = 0
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]
        p0 = p1
        p1 = p0 + nL
        buf = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
        ovL[:, :, p0:p1] = buf.reshape(nL, n_sel, n_tno).transpose(1, 2, 0)
        Lpq = None

    return ovL


def _build_vvL_tno(with_df, C_tno):
    """Build (TNO_a, TNO_b | L) 3-index DF tensor in TNO basis.

    Args:
        with_df: PySCF DF object.
        C_tno (np.ndarray): (nao, n_tno) TNO coefficients.

    Returns:
        vvL (np.ndarray): (n_tno, n_tno, naux)
    """
    nao, n_tno = C_tno.shape
    naux = with_df.get_naoaux()

    mo = np.asfortranarray(C_tno)
    ijslice = (0, n_tno, 0, n_tno)

    vvL = np.empty((n_tno, n_tno, naux))
    buf = None
    p1 = 0
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]
        p0 = p1
        p1 = p0 + nL
        buf = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
        vvL[:, :, p0:p1] = buf.reshape(nL, n_tno, n_tno).transpose(1, 2, 0)
        Lpq = None

    return vvL


def _build_ooL_triple(with_df, C_lmo, lmo_indices):
    """Build (3, 3, naux) occ-occ DF integrals for 3 selected LMO indices.

    Args:
        with_df: PySCF DF object.
        C_lmo (np.ndarray): (nao, nocc_lmo) LMO coefficients.
        lmo_indices (list): Three LMO orbital indices [i, j, k].

    Returns:
        ooL (np.ndarray): (3, 3, naux) occ-occ DF tensor in LMO basis.
    """
    n_sel = len(lmo_indices)
    naux = with_df.get_naoaux()
    C_occ_sel = C_lmo[:, lmo_indices]   # (nao, n_sel)
    mo = np.asfortranarray(C_occ_sel)
    ijslice = (0, n_sel, 0, n_sel)

    ooL = np.empty((n_sel, n_sel, naux))
    buf = None
    p1 = 0
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]
        p0 = p1
        p1 = p0 + nL
        buf = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
        ooL[:, :, p0:p1] = buf.reshape(nL, n_sel, n_sel).transpose(1, 2, 0)
        Lpq = None

    return ooL


def _w3_intermediate(t2_sc, ovL_sc, ooL_sc, vvL_sc, eps_occ, eps_vir):
    """Compute the (T) energy contribution for a single occupied triple (i,j,k).

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
    # K[a,b,p,f] = sum_L ovL_sc[p,f,L]*vvL_sc[a,b,L]  — (n, n, 3, n)
    K = np.einsum('pfL,abL->abpf', ovL_sc, vvL_sc)
    # W_vvov[a,b,c,p,q,r] = sum_f K[a,b,p,f] * t2_sc[q,r,c,f]
    W_all = np.einsum('abpf,qrcf->abcpqr', K, t2_sc)    # (n, n, n, 3, 3, 3)
    del K

    # A[p,a,q,m] = sum_L ovL_sc[p,a,L]*ooL_sc[q,m,L]  — (3, n, 3, 3)
    A = np.einsum('paL,qmL->paqm', ovL_sc, ooL_sc)
    # W_vooo[a,b,c,p,q,r] = sum_m A[p,a,q,m] * t2_sc[m,r,b,c]  (subtract in-place)
    W_all -= np.einsum('paqm,mrbc->abcpqr', A, t2_sc)

    # Make contiguous so slices W_all[a,b,c] are fast.
    W_all = np.ascontiguousarray(W_all)

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

    # z = r3(w) / D for all 6 permutations at once.
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

    # Sum the 6 all-distinct occupied elements for each compact (a,b,c).
    et = np.sum(contrib[:, 0, 1, 2] + contrib[:, 0, 2, 1]
                + contrib[:, 1, 0, 2] + contrib[:, 1, 2, 0]
                + contrib[:, 2, 0, 1] + contrib[:, 2, 1, 0])

    # Factor 2 for closed-shell spin summation (matches canonical et *= 2)
    return float(et.real) * 2


def _zero_cas_t2_amplitudes(t2_pno_all, pno_spaces, occ_cas_idx, C_cas_vir,
                            s1e, cas_proj_thresh=0.5):
    """Zero T2 amplitudes with CAS character for TCC (T) calculation.

    Per Lang et al. Section II.A: to prevent double-counting of static
    correlation, all active (CAS) single and double amplitudes must be set
    to zero before computing the (T) correction.

    - Pairs (i,j) where BOTH i,j are CAS occupied: zero the entire T2 matrix.
    - All pairs: zero PNO components whose overlap with CAS virtuals exceeds
      cas_proj_thresh (i.e. PNOs that live mostly in the CAS virtual space).

    Args:
        t2_pno_all (dict): {(i,j): t2_array} T2 amplitudes in PNO basis.
        pno_spaces (dict): PNO information from make_pnos.
        occ_cas_idx (array-like): Indices of CAS occupied LMOs.
        C_cas_vir (np.ndarray): (nao, n_cas_vir) CAS virtual MO coefficients.
        s1e (np.ndarray): (nao, nao) AO overlap matrix.
        cas_proj_thresh (float): Zero PNO if sum of squared S-overlaps with
            CAS virtuals exceeds this value (default 0.5).

    Returns:
        t2_zeroed (dict): Copy of t2_pno_all with CAS components zeroed.
    """
    occ_cas_set = set(int(x) for x in occ_cas_idx)
    n_cas_vir = C_cas_vir.shape[1] if C_cas_vir is not None else 0
    t2_zeroed = {}

    for pair_key, t2 in t2_pno_all.items():
        i, j = pair_key

        # Both occupied indices in CAS → zero entire amplitude
        if i in occ_cas_set and j in occ_cas_set:
            t2_zeroed[pair_key] = np.zeros_like(t2)
            continue

        t2_new = t2.copy()

        # Zero PNO components with significant CAS virtual character
        if n_cas_vir > 0 and pair_key in pno_spaces:
            C_pno = pno_spaces[pair_key]['C_pno']   # (nao, n_pno)
            n_pno = C_pno.shape[1]
            if n_pno > 0:
                # |<PNO_a | S | CAS_v>|^2 summed over CAS virtuals
                overlap = reduce(np.dot, (C_pno.T, s1e, C_cas_vir))   # (n_pno, n_cas_vir)
                pno_cas_proj = np.sum(overlap**2, axis=1)               # (n_pno,)
                for a in range(n_pno):
                    if pno_cas_proj[a] > cas_proj_thresh:
                        t2_new[a, :] = 0.0
                        t2_new[:, a] = 0.0

        t2_zeroed[pair_key] = t2_new

    return t2_zeroed


def run_lccsd_t_ext(mf, C_lmo, pno_spaces, strong_pairs,
                    t2_pno_all, occ_cas_idx,
                    C_cas_vir=None,
                    vir_cas_idx=None,
                    cas_proj_thresh=0.5,
                    T_CutTNO=1e-9,
                    verbose=None):
    """Compute external-space (T) energy correction for DLPNO-TCCSD(T).

    Loops over all distinct triples (i<j<k), skipping pure-CAS triples.
    Before computing (T), zeros all T2 amplitudes with CAS character to
    prevent double-counting static correlation (Lang et al. Section II.A).

    Args:
        mf: RHF object (for Fock matrix and with_df).
        C_lmo (np.ndarray): (nao, nocc_lmo) LMO coefficients.
        pno_spaces (dict): Output of pno.make_pnos.
        strong_pairs (list): Strong pairs from pair classification.
        t2_pno_all (dict): Converged T2 amplitudes per pair from run_lccsd.
        occ_cas_idx (np.ndarray): CAS occupied indices.
        C_cas_vir (np.ndarray): (nao, n_cas_vir) CAS virtual MO coefficients.
            Required for TCC; if None, T2 zeroing is skipped (warning issued).
        vir_cas_idx (array-like): CAS virtual indices (optional, for logging).
        cas_proj_thresh (float): Zero PNO if CAS-virtual overlap > threshold.
        T_CutTNO (float): TNO eigenvalue threshold (Lang et al. default 1e-9).
        verbose: Verbosity.

    Returns:
        e_t (float): (T) energy correction (external triples only).
    """
    log = logger.new_logger(mf, verbose)

    if not hasattr(mf, 'with_df') or mf.with_df is None:
        raise ValueError('DF integrals required (mf.with_df must be set).')

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
        log.warn('TCC (T): C_cas_vir not provided — T2 amplitudes not zeroed!')
        t2_for_T = t2_pno_all

    # Fock diagonal in LMO basis for occupied orbital energies
    fock_ao = mf.get_fock()
    F_lmo = reduce(np.dot, (C_lmo.T, fock_ao, C_lmo))

    occ_list = list(range(nocc_lmo))

    e_t = 0.0
    n_triples = 0
    n_cas_skip = 0

    # Loop over all triples i <= j <= k from the strong pair occupied set
    for idx_i, i in enumerate(occ_list):
        for idx_j, j in enumerate(occ_list[idx_i:], idx_i):
            for idx_k, k in enumerate(occ_list[idx_j:], idx_j):

                # CAS exclusion: skip pure-CAS triples
                if (i in occ_cas_set) and (j in occ_cas_set) and (k in occ_cas_set):
                    n_cas_skip += 1
                    continue

                # Check that all three pair PNO spaces exist
                ij = (min(i,j), max(i,j))
                ik = (min(i,k), max(i,k))
                jk = (min(j,k), max(j,k))

                if (ij not in pno_spaces or ik not in pno_spaces
                        or jk not in pno_spaces):
                    continue

                if (ij not in t2_for_T or ik not in t2_for_T
                        or jk not in t2_for_T):
                    continue

                # Build triple PNO space as union of three pair PNO spaces
                C_tno, n_tno = _triple_pno_union(pno_spaces, i, j, k, s1e)
                if n_tno == 0:
                    continue

                # Semi-canonical transformation: diagonalise F in TNO subspace.
                # The diagonal of F_tno is NOT the correct virtual energy (TNOs are
                # not eigenstates of F); use eigenvalues instead so the denominator
                # D = eps_i + eps_j + eps_k - eps_a - eps_b - eps_c is correct.
                F_tno_full = reduce(np.dot, (C_tno.T, fock_ao, C_tno))
                eps_tno_sc, V_sc = np.linalg.eigh(F_tno_full)

                # Rotate TNO coefficients to semi-canonical basis.
                # C_tno_sc[:,m] are eigenvectors of F; still S-orthonormal.
                C_tno_sc = np.dot(C_tno, V_sc)   # (nao, n_tno)

                # Map each pair T2 to the semi-canonical TNO basis:
                #   U = C_pair.T @ S @ C_tno_sc   (PNO → SC-TNO overlap)
                #   t2_sc = U.T @ t2_pair @ U
                def _map_t2_to_sc(pair_key):
                    C_pair = pno_spaces[pair_key]['C_pno']               # (nao, n_pair)
                    U = reduce(np.dot, (C_pair.T, s1e, C_tno_sc))        # (n_pair, n_tno)
                    t2 = t2_for_T[pair_key]                               # (n_pair, n_pair)
                    return reduce(np.dot, (U.T, t2, U))                  # (n_tno, n_tno)

                t2_ij_sc = _map_t2_to_sc(ij)
                t2_ik_sc = _map_t2_to_sc(ik)
                t2_jk_sc = _map_t2_to_sc(jk)

                # Build DF integrals directly in the semi-canonical TNO basis
                ovL_ijk = _build_ovL_tno(mf.with_df, C_lmo, C_tno_sc, [i, j, k])
                vvL_sc = _build_vvL_tno(mf.with_df, C_tno_sc)

                # Semi-canonical occupied transformation for this triple.
                # Diagonalise the 3×3 LMO Fock block to get proper triple energies.
                # Off-diagonal F_LMO elements (e.g. 0.27 Eh for H2O bonds) make the
                # diagonal approximation fail; eigenvalues give the correct denominator.
                triple_lmo = [i, j, k]
                F_occ_3x3 = F_lmo[np.ix_(triple_lmo, triple_lmo)]
                eps_occ_sc, V_occ_sc = np.linalg.eigh(F_occ_3x3)

                # Rotate ovL to SC occupied basis:
                #   ovL_sc[m,a,L] = sum_n V_occ_sc[n,m] * ovL_ijk[n,a,L]
                ovL_sc_occ = np.einsum('nm,naL->maL', V_occ_sc, ovL_ijk)

                # Build occ-occ DF integrals for the 3 LMOs, then rotate to SC basis.
                # DLPNO approximation: vooo sum restricted to these 3 LMOs.
                ooL_lmo = _build_ooL_triple(mf.with_df, C_lmo, triple_lmo)
                ooL_sc = np.einsum('pm,qn,pqL->mnL', V_occ_sc, V_occ_sc, ooL_lmo)

                # Build T2 in SC occupied × SC virtual basis.
                # Use exchange symmetry T2[q,p,b,a] = T2[p,q,a,b] (pyscf RHF convention).
                t2_lmo_block = np.zeros((3, 3, n_tno, n_tno))
                t2_lmo_block[0, 1] = t2_ij_sc;  t2_lmo_block[1, 0] = t2_ij_sc.T
                t2_lmo_block[0, 2] = t2_ik_sc;  t2_lmo_block[2, 0] = t2_ik_sc.T
                t2_lmo_block[1, 2] = t2_jk_sc;  t2_lmo_block[2, 1] = t2_jk_sc.T

                # T2_sc[m,n] = sum_{pq} V[p,m]*V[q,n] * T2_lmo[p,q]
                t2_sc_block = np.einsum('pm,qn,pqAB->mnAB', V_occ_sc, V_occ_sc,
                                        t2_lmo_block)

                # (T) energy in SC occupied × SC virtual basis (returns contribution × 2)
                et_ijk = _w3_intermediate(
                    t2_sc_block, ovL_sc_occ, ooL_sc, vvL_sc,
                    eps_occ_sc, eps_tno_sc)

                e_t += et_ijk
                n_triples += 1

    log.info('(T) correction: %d triples computed, %d CAS triples skipped',
             n_triples, n_cas_skip)
    log.info('E(T) external = %.15g', e_t)

    return e_t
