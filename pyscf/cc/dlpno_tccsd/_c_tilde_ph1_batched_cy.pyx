# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Batched Cython kernel for ``compute_C_tilde_batched._process_ki_terms12``.

Version 2: algorithmic parity with Psi4 — Terms 1 and 2 both use
precomputed static quantities (K_tilde_chem and K_bar_chem_slice),
cutting per-iteration FLOPs by ~7-13× vs the prior n_local-summing
implementation.

Math per pair (from residual.py:1609 _process_ki_terms12):

  Term 2 (1 dgemv):
    C[a, b] = sum_a' K_tilde_chem_k[a', a*n_pno + b] * T1[a']
    where K_tilde_chem_k[a', ab] = sum_L k_Qa[L, a'] * Qab[L, a, b]
    (precomputed once in the plan; ~n_pno³ per pair)

  Term 1 (1 dgemm):
    C[a, b] -= sum_l T1_local[l, a] * K_bar_chem_slice[l, b]
    where K_bar_chem_slice = ci['K_bar_chem'][ll_idx]
    (precomputed once — it's just a row-slice of the stored tensor)

Compare to the prior 4-BLAS-op version: Term 2 did `z = k_Qa @ T1` then
`C = Qab_flat.T @ z` (two dgemvs touching n_local-sized axes every
cycle); Term 1 did `K_bar = Qma_sub.T @ ooL_ki` (another dgemv over
n_local) then the dgemm. All three n_local-summing ops are now
L-pre-summed via the cc_ints static tensors (Psi4's design since 2013).

Shapes per pair:
  K_tilde_chem_k:   (n_pno, n_pno²)         C-contig
  K_bar_chem_slice: (n_domain, n_pno)       C-contig
  t1_i_ki:          (n_pno,)                C-contig
  T1_local:         (n_domain, n_pno)       C-contig
  C_out:            (n_pno, n_pno)          C-contig
"""
from scipy.linalg.cython_blas cimport dgemm, dgemv
from cython.parallel cimport prange
cimport openmp

import numpy as np


def c_tilde_ph1_batched(
    # Static flat per-pair buffers (L already summed out)
    double[::1] K_tilde_chem_flat,    # per pair: (n_pno, n_pno²)
    long[::1]   K_tilde_chem_offsets,
    double[::1] K_bar_chem_slice_flat,  # per pair: (n_domain, n_pno)
    long[::1]   K_bar_chem_slice_offsets,

    # Per-iter flat buffers
    double[::1] t1_ki_flat,
    long[::1]   t1_ki_offsets,
    double[::1] T1_local_flat,
    long[::1]   T1_local_offsets,

    # Per-pair shapes
    int[::1] n_pno_arr,
    int[::1] n_domain_arr,

    # Output
    double[::1] C_flat,
    long[::1]   C_offsets,

    int num_threads,
):
    cdef Py_ssize_t N = n_pno_arr.shape[0]
    cdef Py_ssize_t p
    cdef Py_ssize_t n_pno, n_domain

    cdef char N_flag = b'N', T_flag = b'T'
    cdef double one = 1.0, zero = 0.0, neg_one = -1.0
    cdef int int_one = 1
    cdef int int_n_pno, int_n_domain, int_npno2

    cdef double *K_tilde_chem
    cdef double *K_bar_chem_slice
    cdef double *t1
    cdef double *T1_local
    cdef double *C_out

    for p in prange(N, schedule='dynamic', nogil=True, num_threads=num_threads):
        n_pno    = n_pno_arr[p]
        n_domain = n_domain_arr[p]

        K_tilde_chem     = &K_tilde_chem_flat[K_tilde_chem_offsets[p]]
        K_bar_chem_slice = &K_bar_chem_slice_flat[K_bar_chem_slice_offsets[p]]
        t1               = &t1_ki_flat[t1_ki_offsets[p]]
        T1_local         = &T1_local_flat[T1_local_offsets[p]]
        C_out            = &C_flat[C_offsets[p]]

        int_n_pno    = <int>n_pno
        int_n_domain = <int>n_domain
        int_npno2    = <int>(n_pno * n_pno)

        # ================================================================
        # Term 2: C[ab] = sum_a' K_tilde_chem_k[a', ab] * T1[a']
        # K_tilde_chem row-major (n_pno, n_pno²) → col-major (n_pno², n_pno),
        # lda=n_pno².
        # Row-major C.ravel() = K_tilde.T @ T1 ≡ col-major K_col @ T1 (no trans).
        # dgemv('N', m=n_pno², n=n_pno, 1, K_tilde, lda=n_pno², T1, 1, 0, C, 1)
        # ================================================================
        dgemv(&N_flag, &int_npno2, &int_n_pno,
              &one, K_tilde_chem, &int_npno2,
              t1, &int_one,
              &zero, C_out, &int_one)

        # ================================================================
        # Term 1: C[a, b] -= sum_l T1_local[l, a] * K_bar_chem_slice[l, b]
        # Both (n_domain, n_pno) row-major; result (n_pno, n_pno).
        # Col-major trick (same as prior version):
        # dgemm('N', 'T', n_pno, n_pno, n_domain, -1, K_bar, n_pno, T1, n_pno, 1, C, n_pno)
        # ================================================================
        dgemm(&N_flag, &T_flag,
              &int_n_pno, &int_n_pno, &int_n_domain,
              &neg_one, K_bar_chem_slice, &int_n_pno,
              T1_local, &int_n_pno,
              &one, C_out, &int_n_pno)
