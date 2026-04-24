# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Batched Cython kernel for ``compute_C_tilde_batched._process_ki_terms12``.

Path (b) pattern — same as `_t1_fock_batched_cy`. Processes all ordered
pairs (k, i) in one prange loop over an isolated OpenMP team, with
BLAS forced to 1 thread. Static per-pair data (k_Qa, Qab, Qma_sub,
ooL_ki) is cached across CCSD iterations; per-iter T1 inputs are
gathered fresh.

Math per pair (from residual.py:1609 _process_ki_terms12):

  Term 2:
    z[L]           = sum_a k_Qa[L, a] * t1_i_ki[a]          (dgemv)
    C[a, b]       += sum_L z[L] * Qab[L, a, b]              (dgemv on flat Qab)

  Term 1:
    K_bar[l, c]    = sum_L Qma_sub[L, l, c] * ooL_ki[L]     (dgemv on flat Qma)
    C[a, b]       -= sum_l T1_local[l, a] * K_bar[l, b]     (dgemm)

Shapes per pair:
  k_Qa:     (n_local, n_pno)                C-contig
  Qab:      (n_local, n_pno, n_pno)          C-contig
  Qma_sub:  (n_local, n_domain, n_pno)       C-contig (caller-gathered slice)
  ooL_ki:   (n_local,)                       C-contig
  t1_i_ki:  (n_pno,)                         C-contig
  T1_local: (n_domain, n_pno)                C-contig
  C_out:    (n_pno, n_pno)                   C-contig
"""
from scipy.linalg.cython_blas cimport dgemm, dgemv
from cython.parallel cimport prange
cimport openmp

import numpy as np


def c_tilde_ph1_batched(
    # Static flat per-pair buffers
    double[::1] k_Qa_flat,
    long[::1]   k_Qa_offsets,
    double[::1] Qab_flat,
    long[::1]   Qab_offsets,
    double[::1] Qma_sub_flat,
    long[::1]   Qma_sub_offsets,
    double[::1] ooL_ki_flat,
    long[::1]   ooL_ki_offsets,

    # Per-iter flat buffers
    double[::1] t1_ki_flat,
    long[::1]   t1_ki_offsets,
    double[::1] T1_local_flat,
    long[::1]   T1_local_offsets,

    # Per-pair shapes
    int[::1] n_pno_arr,
    int[::1] n_local_arr,
    int[::1] n_domain_arr,

    # Per-thread scratch
    double[:, ::1] z_scratch,       # (num_threads, max_n_local)
    double[:, ::1] K_bar_scratch,   # (num_threads, max_n_domain * max_n_pno)

    # Output
    double[::1] C_flat,
    long[::1]   C_offsets,

    int num_threads,
):
    cdef Py_ssize_t N = n_pno_arr.shape[0]
    cdef Py_ssize_t p
    cdef int tid
    cdef Py_ssize_t n_pno, n_local, n_domain

    cdef char N_flag = b'N', T_flag = b'T'
    cdef double one = 1.0, zero = 0.0, neg_one = -1.0
    cdef int int_one = 1
    cdef int int_n_pno, int_n_local, int_n_domain
    cdef int int_npno2, int_dom_npno

    cdef double *k_Qa
    cdef double *Qab
    cdef double *Qma_sub
    cdef double *ooL_ki
    cdef double *t1
    cdef double *T1_local
    cdef double *C_out
    cdef double *z
    cdef double *K_bar

    for p in prange(N, schedule='dynamic', nogil=True, num_threads=num_threads):
        tid = openmp.omp_get_thread_num()
        n_pno   = n_pno_arr[p]
        n_local = n_local_arr[p]
        n_domain = n_domain_arr[p]

        k_Qa    = &k_Qa_flat[k_Qa_offsets[p]]
        Qab     = &Qab_flat[Qab_offsets[p]]
        Qma_sub = &Qma_sub_flat[Qma_sub_offsets[p]]
        ooL_ki  = &ooL_ki_flat[ooL_ki_offsets[p]]
        t1      = &t1_ki_flat[t1_ki_offsets[p]]
        T1_local = &T1_local_flat[T1_local_offsets[p]]
        C_out   = &C_flat[C_offsets[p]]

        z     = &z_scratch[tid, 0]
        K_bar = &K_bar_scratch[tid, 0]

        int_n_pno    = <int>n_pno
        int_n_local  = <int>n_local
        int_n_domain = <int>n_domain
        int_npno2    = <int>(n_pno * n_pno)
        int_dom_npno = <int>(n_domain * n_pno)

        # ---- Term 2a: z = k_Qa @ t1_i_ki  (dgemv 'T' on row-major k_Qa) ----
        # k_Qa row-major (n_local, n_pno); col-major (n_pno, n_local), lda=n_pno.
        # Row-major y = A @ x ≡ col-major y = A_F^T @ x.
        dgemv(&T_flag, &int_n_pno, &int_n_local,
              &one, k_Qa, &int_n_pno,
              t1, &int_one,
              &zero, z, &int_one)

        # ---- Term 2b: C_out = Qab_flat.T @ z (initializes C, beta=0) ----
        # Qab row-major (n_local, n_pno*n_pno); col-major (n_pno², n_local),
        # lda=n_pno². Want C.ravel() = Qab.T @ z (row-major (n_pno²,)).
        # Col-major: C_col = Qab_col @ z (no trans).
        dgemv(&N_flag, &int_npno2, &int_n_local,
              &one, Qab, &int_npno2,
              z, &int_one,
              &zero, C_out, &int_one)

        # ---- Term 1a: K_bar = Qma_sub.T @ ooL_ki (dgemv 'N') ----
        # Qma_sub row-major (n_local, n_domain, n_pno), flat (n_local, n_dom*n_pno);
        # col-major (n_dom*n_pno, n_local), lda=n_dom*n_pno.
        # Row-major K_bar.ravel() = Qma_sub_flat.T @ ooL = col-major Qma_col @ ooL.
        dgemv(&N_flag, &int_dom_npno, &int_n_local,
              &one, Qma_sub, &int_dom_npno,
              ooL_ki, &int_one,
              &zero, K_bar, &int_one)

        # ---- Term 1b: C_out -= T1_local.T @ K_bar (dgemm) ----
        # Row-major: C (n_pno, n_pno) -= T1_local.T (n_pno, n_dom) @ K_bar (n_dom, n_pno)
        # Col-major view:
        #   T1_local row-major (n_dom, n_pno) → col-major (n_pno, n_dom), lda_F=n_pno.
        #   K_bar    row-major (n_dom, n_pno) → col-major (n_pno, n_dom), lda_F=n_pno.
        #   C_out    row-major (n_pno, n_pno) → col-major (n_pno, n_pno), ldc_F=n_pno.
        #   C_col[b, a] = C_row[a, b] -= sum_l T1_local[l, a] * K_bar[l, b]
        #              = sum_l T1_col[a, l] * K_col[b, l]
        #              = (K_col @ T1_col.T)[b, a]
        # dgemm('N', 'T', n_pno, n_pno, n_dom, -1, K_bar, n_pno, T1_local, n_pno, 1, C, n_pno)
        dgemm(&N_flag, &T_flag,
              &int_n_pno, &int_n_pno, &int_n_domain,
              &neg_one, K_bar, &int_n_pno,
              T1_local, &int_n_pno,
              &one, C_out, &int_n_pno)
