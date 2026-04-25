# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Batched Cython kernel for ``build_D_tilde_batched._process_ik_t12``.

Version 2: algorithmic parity with Psi4 — Terms 1 and 2 both use
precomputed static quantities (K_tilde_chem, K_bar_ij/ji, K_bar_chem),
cutting per-iteration FLOPs by 7-13× vs the prior n_local-summing
implementation.

Math per pair (from residual.py:881 _process_ik_t12, matches Psi4
ccsd.cc:1876 compute_D_tilde):

  Term 2 (2 dgemv + 1 accumulate loop):
    part1[j*n_pno + p]  = sum_q K_tilde_chem_k[j, p*n_pno + q] * T1[q]
    part2[a*n_pno + b]  = sum_c T1[c] * K_tilde_chem_k[c, a*n_pno + b]
    D[a, b]            += 2 * part1[b*n_pno + a] - part2[b*n_pno + a]
    (matches Psi4 compute_D_tilde L1898-1908.)

  Term 1 (1 dgemm):
    M_static[l, c] = 2 * K_bar_ij_or_ji[l, c] - K_bar_chem[l, c]   (precomputed)
    D[a, b]       -= sum_l T1_rows[l, a] * M_static[l, b]
    (matches Psi4 compute_D_tilde L1910-1913 L_bar_temp + doublet.)

Compare to prior 6-BLAS-op version: Term 2 was (dgemv Qab×T1)
→ (dgemm X×k_Qa) → (dgemv k_Qa×T1) → (dgemv Qab×w) + transpose loop
— all four dgemvs ran over n_local. Term 1 was (dgemm ooL_il×ovL_k)
→ (dgemv Qma×ooL_ik) + elementwise + (dgemm T1×M). All n_local sums.
New kernel: 3 BLAS ops on pure (n_pno) / (n_domain) tensors.

Shapes per pair:
  K_tilde_chem_k:  (n_pno, n_pno²)          C-contig
  M_static:        (n_domain, n_pno)        C-contig
  t1_i_ik:         (n_pno,)                 C-contig
  T1_rows:         (n_domain, n_pno)        C-contig
  D_out:           (n_pno, n_pno)           C-contig
"""
from scipy.linalg.cython_blas cimport dgemm, dgemv
from cython.parallel cimport prange
cimport openmp

import numpy as np


def d_tilde_ph1_batched(
    # Static flat per-pair buffers (L already summed out)
    double[::1] K_tilde_chem_flat,    # per pair: (n_pno, n_pno²)
    long[::1]   K_tilde_chem_offsets,
    double[::1] M_static_flat,        # per pair: (n_domain, n_pno)
    long[::1]   M_static_offsets,

    # Per-iter flat buffers
    double[::1] t1_flat,
    long[::1]   t1_offsets,
    double[::1] T1_rows_flat,
    long[::1]   T1_rows_offsets,

    # Per-pair shapes
    int[::1] n_pno_arr,
    int[::1] n_domain_arr,

    # Per-thread scratch (for Term 2)
    double[:, ::1] part1_scratch,     # (num_threads, max_n_pno²)
    double[:, ::1] part2_scratch,     # (num_threads, max_n_pno²)

    # Output
    double[::1] D_flat,
    long[::1]   D_offsets,

    int num_threads,
):
    cdef Py_ssize_t N = n_pno_arr.shape[0]
    cdef Py_ssize_t p
    cdef int tid
    cdef Py_ssize_t n_pno, n_domain
    cdef Py_ssize_t a, b

    cdef char N_flag = b'N', T_flag = b'T'
    cdef double one = 1.0, zero = 0.0, neg_one = -1.0
    cdef int int_one = 1
    cdef int int_n_pno, int_n_domain, int_npno2

    cdef double *K_tilde_chem
    cdef double *M_static
    cdef double *t1
    cdef double *T1_rows
    cdef double *D_out
    cdef double *part1
    cdef double *part2

    for p in prange(N, schedule='dynamic', nogil=True, num_threads=num_threads):
        tid = openmp.omp_get_thread_num()
        n_pno    = n_pno_arr[p]
        n_domain = n_domain_arr[p]

        K_tilde_chem = &K_tilde_chem_flat[K_tilde_chem_offsets[p]]
        M_static     = &M_static_flat[M_static_offsets[p]]
        t1           = &t1_flat[t1_offsets[p]]
        T1_rows      = &T1_rows_flat[T1_rows_offsets[p]]
        D_out        = &D_flat[D_offsets[p]]

        part1 = &part1_scratch[tid, 0]
        part2 = &part2_scratch[tid, 0]

        int_n_pno    = <int>n_pno
        int_n_domain = <int>n_domain
        int_npno2    = <int>(n_pno * n_pno)

        # Init D = 0
        for a in range(n_pno):
            for b in range(n_pno):
                D_out[a * n_pno + b] = 0.0

        # ================================================================
        # Term 2 part 1: part1[j*n_pno + p] = sum_q K_tilde[j, p*n_pno + q] * T1[q]
        # K_tilde row-major (n_pno, n_pno²) as (n_pno², n_pno) after reshape.
        # Matches Psi4 L1899-1903: reshape + doublet with T_i.
        # Col-major view of K_tilde with lda=n_pno: shape (n_pno, n_pno²).
        # dgemv('T', m=n_pno, n=n_pno², 1, K_tilde, n_pno, T1, 1, 0, part1, 1)
        # ================================================================
        dgemv(&T_flag, &int_n_pno, &int_npno2,
              &one, K_tilde_chem, &int_n_pno,
              t1, &int_one,
              &zero, part1, &int_one)

        # ================================================================
        # Term 2 part 2: part2[a*n_pno + b] = sum_c T1[c] * K_tilde[c, a*n_pno + b]
        # Row-major (T1.T @ K_tilde) giving (n_pno²,).
        # Col-major view of K_tilde with lda=n_pno² gives col-major (n_pno², n_pno).
        # dgemv('N', m=n_pno², n=n_pno, 1, K_tilde, n_pno², T1, 1, 0, part2, 1)
        # ================================================================
        dgemv(&N_flag, &int_npno2, &int_n_pno,
              &one, K_tilde_chem, &int_npno2,
              t1, &int_one,
              &zero, part2, &int_one)

        # ================================================================
        # Combine: D[a, b] += 2 * part1[b*n_pno + a] - part2[b*n_pno + a]
        # (matches Psi4 L_temp.transpose() for both parts → transpose-add.)
        # ================================================================
        for a in range(n_pno):
            for b in range(n_pno):
                D_out[a * n_pno + b] += (2.0 * part1[b * n_pno + a]
                                         - part2[b * n_pno + a])

        # ================================================================
        # Term 1: D[a, b] -= sum_l T1_rows[l, a] * M_static[l, b]
        # M_static = 2 * K_bar_ij_or_ji[ll_idx] - K_bar_chem[ll_idx],
        # precomputed once in the plan (matches Psi4 L_bar_temp).
        # dgemm('N', 'T', n_pno, n_pno, n_domain, -1, M_static, n_pno, T1_rows, n_pno, 1, D, n_pno)
        # ================================================================
        dgemm(&N_flag, &T_flag,
              &int_n_pno, &int_n_pno, &int_n_domain,
              &neg_one, M_static, &int_n_pno,
              T1_rows, &int_n_pno,
              &one, D_out, &int_n_pno)
