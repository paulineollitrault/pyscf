# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Batched Cython kernel for ``build_D_tilde_batched._process_ik_t12``.

Same path-(b) pattern as `_c_tilde_ph1_batched_cy`. Processes all
ordered pairs (i, k) in one prange over an isolated OpenMP team, BLAS
forced to 1 thread. Static per-pair data cached across CCSD iterations.

Math per pair (from residual.py:881 _process_ik_t12):

  Term 2 (3 BLAS + 1 transpose-accumulate):
    X[L, a]      = sum_c Qab[L, a, c] * t1[c]              (dgemv on Qab flat)
    D[a, b]     += 2 * sum_L X[L, a] * k_Qa[L, b]          (dgemm: X.T @ k_Qa)
    w[L]         = sum_a k_Qa[L, a] * t1[a]                (dgemv)
    temp[b, a]   = sum_L w[L] * Qab[L, b, a]               (dgemv on Qab flat)
    D[a, b]     -= temp[b, a]                              (transpose-add loop)

  Term 1 (2 BLAS + 1 elementwise + 1 BLAS):
    ilkc[l, c]   = sum_L ooL_il_all[L, l] * ovL_k[L, c]    (dgemm)
    iklc[l, c]   = sum_L Qma_sub[L, l, c] * ooL_ik[L]      (dgemv)
    M[l, c]      = 2*ilkc[l, c] - iklc[l, c]               (elementwise)
    D[a, b]     -= sum_l T1_rows[l, a] * M[l, b]           (dgemm)

Total: 6 BLAS ops per pair (plus two tiny loops). No Qab pre-transpose
needed — Term 2a and 2d share the same raw Qab buffer; Term 2d's
transpose is done via a cheap scalar loop on the (n_pno, n_pno) output.

Shapes per pair:
  k_Qa:        (n_local, n_pno)
  Qab:         (n_local, n_pno, n_pno)
  Qma_sub:     (n_local, n_domain, n_pno)
  ooL_il_all:  (n_local, n_domain)
  ooL_ik:      (n_local,)
  ovL_k:       (n_local, n_pno)
  t1_i_ik:     (n_pno,)
  T1_rows:     (n_domain, n_pno)
  D_out:       (n_pno, n_pno)
"""
from scipy.linalg.cython_blas cimport dgemm, dgemv
from cython.parallel cimport prange
cimport openmp

import numpy as np


def d_tilde_ph1_batched(
    # Static flat per-pair buffers
    double[::1] k_Qa_flat,
    long[::1]   k_Qa_offsets,
    double[::1] Qab_flat,
    long[::1]   Qab_offsets,
    double[::1] Qma_sub_flat,
    long[::1]   Qma_sub_offsets,
    double[::1] ooL_il_all_flat,
    long[::1]   ooL_il_all_offsets,
    double[::1] ooL_ik_flat,
    long[::1]   ooL_ik_offsets,
    double[::1] ovL_k_flat,
    long[::1]   ovL_k_offsets,

    # Per-iter flat buffers
    double[::1] t1_flat,
    long[::1]   t1_offsets,
    double[::1] T1_rows_flat,
    long[::1]   T1_rows_offsets,

    # Per-pair shapes
    int[::1] n_pno_arr,
    int[::1] n_local_arr,
    int[::1] n_domain_arr,

    # Per-thread scratch
    double[:, ::1] X_scratch,         # (num_threads, max_n_local * max_n_pno)
    double[:, ::1] w_scratch,         # (num_threads, max_n_local)
    double[:, ::1] temp_scratch,      # (num_threads, max_n_pno * max_n_pno)
    double[:, ::1] ilkc_scratch,      # (num_threads, max_n_domain * max_n_pno)
    double[:, ::1] iklc_scratch,      # (num_threads, max_n_domain * max_n_pno)

    # Output
    double[::1] D_flat,
    long[::1]   D_offsets,

    int num_threads,
):
    cdef Py_ssize_t N = n_pno_arr.shape[0]
    cdef Py_ssize_t p
    cdef int tid
    cdef Py_ssize_t n_pno, n_local, n_domain
    cdef Py_ssize_t a, b, l, c

    cdef char N_flag = b'N', T_flag = b'T'
    cdef double one = 1.0, zero = 0.0, two = 2.0, neg_one = -1.0
    cdef int int_one = 1
    cdef int int_n_pno, int_n_local, int_n_domain
    cdef int int_npno2, int_dom_npno, int_local_npno

    cdef double *k_Qa
    cdef double *Qab
    cdef double *Qma_sub
    cdef double *ooL_il_all
    cdef double *ooL_ik
    cdef double *ovL_k
    cdef double *t1
    cdef double *T1_rows
    cdef double *D_out
    cdef double *X
    cdef double *w
    cdef double *temp
    cdef double *ilkc
    cdef double *iklc

    for p in prange(N, schedule='dynamic', nogil=True, num_threads=num_threads):
        tid = openmp.omp_get_thread_num()
        n_pno    = n_pno_arr[p]
        n_local  = n_local_arr[p]
        n_domain = n_domain_arr[p]

        k_Qa       = &k_Qa_flat[k_Qa_offsets[p]]
        Qab        = &Qab_flat[Qab_offsets[p]]
        Qma_sub    = &Qma_sub_flat[Qma_sub_offsets[p]]
        ooL_il_all = &ooL_il_all_flat[ooL_il_all_offsets[p]]
        ooL_ik     = &ooL_ik_flat[ooL_ik_offsets[p]]
        ovL_k      = &ovL_k_flat[ovL_k_offsets[p]]
        t1         = &t1_flat[t1_offsets[p]]
        T1_rows    = &T1_rows_flat[T1_rows_offsets[p]]
        D_out      = &D_flat[D_offsets[p]]

        X    = &X_scratch[tid, 0]
        w    = &w_scratch[tid, 0]
        temp = &temp_scratch[tid, 0]
        ilkc = &ilkc_scratch[tid, 0]
        iklc = &iklc_scratch[tid, 0]

        int_n_pno      = <int>n_pno
        int_n_local    = <int>n_local
        int_n_domain   = <int>n_domain
        int_npno2      = <int>(n_pno * n_pno)
        int_dom_npno   = <int>(n_domain * n_pno)
        int_local_npno = <int>(n_local * n_pno)

        # Init D = 0
        for a in range(n_pno):
            for b in range(n_pno):
                D_out[a * n_pno + b] = 0.0

        # ================================================================
        # Term 2a: X[L, a] = sum_c Qab[L, a, c] * t1[c]
        # Qab row-major (n_local, n_pno, n_pno); flat (n_local*n_pno, n_pno)
        # with Qab_flat[L*n_pno + a, c] = Qab[L, a, c].
        # X[L*n_pno + a] = sum_c Qab_flat[L*n_pno + a, c] * t1[c]
        #               = (Qab_flat @ t1)[L*n_pno + a]
        # Col-major Qab_F shape (n_pno, n_local*n_pno), lda=n_pno.
        # Row-major X = Qab_flat @ t1 ⇔ col-major X_col = Qab_F^T @ t1.
        # dgemv('T', n_pno, n_local*n_pno, 1, Qab, n_pno, t1, 1, 0, X, 1)
        # ================================================================
        dgemv(&T_flag, &int_n_pno, &int_local_npno,
              &one, Qab, &int_n_pno,
              t1, &int_one,
              &zero, X, &int_one)

        # ================================================================
        # Term 2b: D[a, b] += 2 * sum_L X[L, a] * k_Qa[L, b]
        # = 2 * X.T @ k_Qa (row-major (n_pno, n_pno))
        # Same col-major trick as C_tilde Term 1 last step:
        # dgemm('N', 'T', n_pno, n_pno, n_local, 2, k_Qa, n_pno, X, n_pno, 1, D, n_pno)
        # ================================================================
        dgemm(&N_flag, &T_flag,
              &int_n_pno, &int_n_pno, &int_n_local,
              &two, k_Qa, &int_n_pno,
              X, &int_n_pno,
              &one, D_out, &int_n_pno)

        # ================================================================
        # Term 2c: w[L] = sum_a k_Qa[L, a] * t1[a]
        # (Same as C_tilde Term 2a — row-major k_Qa @ t1.)
        # ================================================================
        dgemv(&T_flag, &int_n_pno, &int_n_local,
              &one, k_Qa, &int_n_pno,
              t1, &int_one,
              &zero, w, &int_one)

        # ================================================================
        # Term 2d: D[a, b] -= sum_L w[L] * Qab[L, b, a]
        # Compute temp[b, a] = sum_L w[L] * Qab[L, b, a] via dgemv on Qab flat:
        #   Qab_flat[L, k] where k = b*n_pno + a gives Qab[L, b, a].
        #   temp_flat[k] = (Qab_flat.T @ w)[k]
        # Then scalar loop: D[a, b] -= temp_flat[b*n_pno + a].
        # ================================================================
        dgemv(&N_flag, &int_npno2, &int_n_local,
              &one, Qab, &int_npno2,
              w, &int_one,
              &zero, temp, &int_one)
        for a in range(n_pno):
            for b in range(n_pno):
                D_out[a * n_pno + b] -= temp[b * n_pno + a]

        # ================================================================
        # Term 1a: ilkc[l, c] = sum_L ooL_il_all[L, l] * ovL_k[L, c]
        # Row-major ilkc (n_domain, n_pno) = ooL_il_all.T @ ovL_k
        # Col-major trick (same shape as before):
        # dgemm('N', 'T', n_pno, n_domain, n_local, 1, ovL_k, n_pno, ooL_il_all, n_domain, 0, ilkc, n_pno)
        # ================================================================
        dgemm(&N_flag, &T_flag,
              &int_n_pno, &int_n_domain, &int_n_local,
              &one, ovL_k, &int_n_pno,
              ooL_il_all, &int_n_domain,
              &zero, ilkc, &int_n_pno)

        # ================================================================
        # Term 1b: iklc[l, c] = sum_L Qma_sub[L, l, c] * ooL_ik[L]
        # Same dgemv pattern as C_tilde Term 1a.
        # ================================================================
        dgemv(&N_flag, &int_dom_npno, &int_n_local,
              &one, Qma_sub, &int_dom_npno,
              ooL_ik, &int_one,
              &zero, iklc, &int_one)

        # ================================================================
        # M = 2*ilkc - iklc (elementwise, stored back into ilkc)
        # ================================================================
        for l in range(n_domain):
            for c in range(n_pno):
                ilkc[l * n_pno + c] = 2.0 * ilkc[l * n_pno + c] - iklc[l * n_pno + c]

        # ================================================================
        # Term 1c: D[a, b] -= sum_l T1_rows[l, a] * M[l, b]
        # Same as C_tilde Term 1 last step.
        # ================================================================
        dgemm(&N_flag, &T_flag,
              &int_n_pno, &int_n_pno, &int_n_domain,
              &neg_one, ilkc, &int_n_pno,
              T1_rows, &int_n_pno,
              &one, D_out, &int_n_pno)
