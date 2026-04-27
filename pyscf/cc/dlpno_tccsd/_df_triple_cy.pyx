# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Per-center DF kernels for _build_triple_local_DF (Step C2).

Replaces three numpy-heavy operations inside the per-center inner loop
of ``_build_triple_local_DF``:

* ``vvL`` path: two fancy-index gathers + two batched matmuls + transpose
  + scatter → one nogil Cython function that fuses the gather with two
  cython_blas.dgemm calls.
* ``ovL`` path: one fancy-index gather + batched matmul + scatter → one
  nogil Cython function with a hand-rolled (nu, n_tno) small matmul.
* ``ooL`` path: fancy-index gather + scatter → one plain nogil loop.

The per-center Python dispatch costs that we skip:

* Fancy indexing ``qab_A[:, vu[:, None], vu[None, :]]`` — O(nQ_A·nu²) copy.
* Fancy indexing ``qab_A_uu[atom_pos]`` — O(nQ_c·nu²) copy.
* ``np.matmul`` dispatch × 2 per center (vvL) + 1 per center × 3 idx (ovL).
* Extra transpose + scatter assignments.
"""
from scipy.linalg.cython_blas cimport dgemm

import numpy as np


def vvL_center_kernel(
    double[:, :, ::1] qab_A,            # (nQ_A, np_A, np_A)
    long[::1] atom_pos,                 # (nQ_c,)
    long[::1] valid_u_Q,                # (nu,) positions in centerQ's PAO stack
    double[:, ::1] X_Q,                 # (nu, n_tno) triple-PAO → TNO transform
    long[::1] local_Q,                  # (nQ_c,) target slots in naux_ijk axis
    double[:, :, ::1] vvL_raw,          # (n_tno, n_tno, naux_ijk) in-place
    int nQ_c, int nu, int n_tno,
):
    """Accumulate this center's vvL contribution into ``vvL_raw`` in place.

    Per-Q work (one dgemm pair):
        qab_cut[u, v] = qab_A[atom_pos[q], valid_u_Q[u], valid_u_Q[v]]
        tmp           = qab_cut @ X_Q                 # (nu, n_tno)
        vvL_c         = X_Q.T @ tmp                   # (n_tno, n_tno)
        vvL_raw[:, :, local_Q[q]] = vvL_c
    """
    cdef Py_ssize_t q, u, v, a, b
    cdef long atom_q, vu_u, vu_v, lq
    cdef char N_ = b'N', T_ = b'T'
    cdef double one = 1.0, zero = 0.0
    cdef int int_nu = nu, int_n_tno = n_tno

    if nQ_c == 0 or nu == 0 or n_tno == 0:
        return

    cdef double[:, ::1] qab_cut = np.empty((nu, nu))
    cdef double[:, ::1] tmp = np.empty((nu, n_tno))
    cdef double[:, ::1] vvL_c = np.empty((n_tno, n_tno))

    with nogil:
        for q in range(nQ_c):
            atom_q = atom_pos[q]
            # Gather qab_cut (nu, nu) from the (np_A, np_A) slab at qab_A[atom_q].
            for u in range(nu):
                vu_u = valid_u_Q[u]
                for v in range(nu):
                    vu_v = valid_u_Q[v]
                    qab_cut[u, v] = qab_A[atom_q, vu_u, vu_v]

            # tmp (nu, n_tno) = qab_cut (nu, nu) @ X_Q (nu, n_tno).
            dgemm(&N_, &N_,
                  &int_n_tno, &int_nu, &int_nu,
                  &one,
                  &X_Q[0, 0], &int_n_tno,
                  &qab_cut[0, 0], &int_nu,
                  &zero,
                  &tmp[0, 0], &int_n_tno)

            # vvL_c (n_tno, n_tno) = X_Q.T (n_tno, nu) @ tmp (nu, n_tno).
            # X_Q is row-major (nu, n_tno). Using transb='T' reads X_Q as
            # its natural C-layout transposed (i.e. X_Q_F.T = X_Q).
            dgemm(&N_, &T_,
                  &int_n_tno, &int_n_tno, &int_nu,
                  &one,
                  &tmp[0, 0], &int_n_tno,
                  &X_Q[0, 0], &int_n_tno,
                  &zero,
                  &vvL_c[0, 0], &int_n_tno)

            # Scatter vvL_c into vvL_raw[:, :, local_Q[q]].
            lq = local_Q[q]
            for a in range(n_tno):
                for b in range(n_tno):
                    vvL_raw[a, b, lq] = vvL_c[a, b]


def ovL_center_kernel(
    double[:, :, ::1] qia_A,            # (nQ_A, nl, np_A)
    long[::1] atom_pos,                 # (nQ_c,)
    long[::1] valid_u_Q,                # (nu,)
    double[:, ::1] X_Q,                 # (nu, n_tno)
    long[::1] local_Q,                  # (nQ_c,)
    long[::1] occ_sp,                   # (3,) — -1 means "skip this idx"
    double[:, :, ::1] ovL_raw,          # (3, n_tno, naux_ijk) in-place
    int nQ_c, int nu, int n_tno,
):
    """Accumulate the 3 ovL contributions (one per occ slot) at this center.

    Per (q, idx) with occ_sp[idx] >= 0:
        block_row[u] = qia_A[atom_pos[q], occ_sp[idx], valid_u_Q[u]]
        ovL_raw[idx, :, local_Q[q]] = block_row @ X_Q
    """
    cdef Py_ssize_t q, u, a, idx
    cdef long atom_q, vu_u, occ, lq
    cdef double acc

    if nQ_c == 0 or nu == 0 or n_tno == 0:
        return

    cdef double[::1] block_row = np.empty(nu)

    with nogil:
        for q in range(nQ_c):
            atom_q = atom_pos[q]
            lq = local_Q[q]
            for idx in range(3):
                occ = occ_sp[idx]
                if occ < 0:
                    continue
                # Gather block_row (nu,).
                for u in range(nu):
                    vu_u = valid_u_Q[u]
                    block_row[u] = qia_A[atom_q, occ, vu_u]
                # ovL_raw[idx, a, lq] = Σ_u block_row[u] * X_Q[u, a].
                # nu ~ 30, n_tno ~ 25; hand-rolled wins on dispatch here
                # and the compiler autovectorises the inner u loop.
                for a in range(n_tno):
                    acc = 0.0
                    for u in range(nu):
                        acc = acc + block_row[u] * X_Q[u, a]
                    ovL_raw[idx, a, lq] = acc


def ooL_center_kernel(
    double[:, :, ::1] qij_A,            # (nQ_A, nl, nl)
    long[::1] atom_pos,                 # (nQ_c,)
    long[::1] local_Q,                  # (nQ_c,)
    long[::1] m_sparse_kept,            # (n_dom_kept,)
    long[::1] dom_local,                # (n_dom_kept,)
    long[::1] occ_sp,                   # (3,)
    double[:, :, ::1] ooL_raw,          # (3, n_domain, naux_ijk) in-place
    int nQ_c, int n_dom_kept,
):
    """Scatter qij contributions into ooL_raw for one center.

    Per (q, idx, m) with occ_sp[idx] >= 0:
        ooL_raw[idx, dom_local[m], local_Q[q]] = qij_A[atom_pos[q],
                                                        occ_sp[idx],
                                                        m_sparse_kept[m]]
    """
    cdef Py_ssize_t q, idx, m
    cdef long atom_q, lq, occ, m_sp, dl

    if nQ_c == 0 or n_dom_kept == 0:
        return

    with nogil:
        for q in range(nQ_c):
            atom_q = atom_pos[q]
            lq = local_Q[q]
            for idx in range(3):
                occ = occ_sp[idx]
                if occ < 0:
                    continue
                for m in range(n_dom_kept):
                    m_sp = m_sparse_kept[m]
                    dl = dom_local[m]
                    ooL_raw[idx, dl, lq] = qij_A[atom_q, occ, m_sp]
