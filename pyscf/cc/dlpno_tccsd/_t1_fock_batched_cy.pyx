# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Batched Cython kernel for ``t1_fock._per_pair``.

Path (b) from HANDOFF_CCSD_PARITY_V3: isolated OpenMP team that does
NOT share threads with the Python ThreadPoolExecutor. Processes all
strong pairs in one prange loop. Requires the caller to set BLAS to
single-threaded for the duration of this call (e.g. via
``threadpool_limits(limits=1, user_api='blas')``) so each thread's
BLAS calls don't oversubscribe.

Math per pair (matches local_df.py:993 _per_pair):

  Step 1:
    d_ij = 2 * sum(T1 * K_chem_l) - sum(T1 * K_ji_l)
    d_ji = 2 * sum(T1 * K_chem_l) - sum(T1 * K_ij_l)  (if i != j)

  Step 2:
    Fab = diag(e_pno)
    gamma[L] = sum_{m, a} Qma[L, m, a] * T1[m, a]
    Fab[a, b] += 2 * sum_L gamma[L] * Qab[L, a, b]
    Y[L, b, m] = sum_c Qab[L, b, c] * T1[m, c]       (via dgemm)
    Y_alt[L, m, b] = Y[L, b, m]                       (transpose copy)
    Fab[b, c] -= sum_{L, m} Y_alt[L, m, b] * Qma[L, m, c]  (via dgemm)
    Fia[m, a] = 2 * sum_L gamma[L] * Qma[L, m, a]
    Z[L, j, i] = sum_c Qma[L, j, c] * T1[i, c]        (via dgemm)
    Z_xxx[L, i, j] = Z[L, j, i]                        (transpose copy)
    Fia[m_out, a] -= sum_{L, i} Z_xxx[L, i, m_out] * Qma[L, i, a]  (via dgemm)
    Fab[a, b] -= sum_m T1[m, a] * Fia[m, b]            (via dgemm)

BLAS calls use row-major ↔ col-major "swap-and-interpret" convention:
a row-major (p, q) buffer is a col-major (q, p) matrix with lda=q.
"""
from scipy.linalg.cython_blas cimport dgemm, dgemv
from cython.parallel cimport prange
cimport openmp

import numpy as np


def t1_fock_batched(
    # Flat per-pair input buffers + offsets
    double[::1] T1_flat,
    long[::1]   T1_offsets,       # (N+1,)
    double[::1] K_chem_flat,
    long[::1]   K_chem_offsets,
    double[::1] K_ji_flat,
    long[::1]   K_ji_offsets,
    double[::1] K_ij_flat,
    long[::1]   K_ij_offsets,
    double[::1] Qma_flat,
    long[::1]   Qma_offsets,
    double[::1] Qab_flat,
    long[::1]   Qab_offsets,
    double[::1] e_pno_flat,
    long[::1]   e_pno_offsets,

    # Per-pair shapes (all int32, length N)
    int[::1] nlmo_arr,
    int[::1] npno_arr,
    int[::1] n_local_arr,
    int[::1] need_dji_arr,

    # Per-thread scratch (pre-allocated, sized for max shape × num_threads)
    # Sizes from the Python wrapper: gamma max_n_local,
    # Y/Y_alt max_n_local*max_npno*max_nlmo,
    # Fia max_nlmo*max_npno,
    # Z/Z_xxx max_n_local*max_nlmo*max_nlmo
    double[:, ::1] gamma_scratch,
    double[:, ::1] Y_trans_scratch,
    double[:, ::1] Y_alt_scratch,
    double[:, ::1] Fia_scratch,
    double[:, ::1] Z_stacked_scratch,
    double[:, ::1] Z_xxx_scratch,

    # Outputs
    double[::1] d_flat,           # (N * 2,)
    double[::1] Fab_flat,
    long[::1]   Fab_offsets,

    int num_threads,
):
    cdef Py_ssize_t N = nlmo_arr.shape[0]
    cdef Py_ssize_t p
    cdef int tid
    cdef Py_ssize_t nlmo, npno, n_local
    cdef int need_dji
    cdef Py_ssize_t m, a, b, mp, L

    cdef char N_flag = b'N', T_flag = b'T'
    cdef double one = 1.0, zero = 0.0, two = 2.0, neg_one = -1.0
    cdef int int_one = 1
    cdef int int_nlmo, int_npno, int_n_local
    cdef int int_nlmo_npno, int_npno2
    cdef int int_n_local_npno, int_n_local_nlmo

    cdef double *T1
    cdef double *K_chem
    cdef double *K_ji
    cdef double *K_ij
    cdef double *Qma
    cdef double *Qab
    cdef double *e_pno
    cdef double *d_out
    cdef double *Fab
    cdef double *gamma
    cdef double *Y_trans
    cdef double *Y_alt
    cdef double *Fia_bar
    cdef double *Z_stacked
    cdef double *Z_xxx
    cdef double s, acc_chem, acc_ji, acc_ij

    for p in prange(N, schedule='dynamic', nogil=True, num_threads=num_threads):
        tid = openmp.omp_get_thread_num()
        nlmo    = nlmo_arr[p]
        npno    = npno_arr[p]
        n_local = n_local_arr[p]
        need_dji = need_dji_arr[p]

        T1     = &T1_flat[T1_offsets[p]]
        K_chem = &K_chem_flat[K_chem_offsets[p]]
        K_ji   = &K_ji_flat[K_ji_offsets[p]]
        K_ij   = &K_ij_flat[K_ij_offsets[p]]
        Qma    = &Qma_flat[Qma_offsets[p]]
        Qab    = &Qab_flat[Qab_offsets[p]]
        e_pno  = &e_pno_flat[e_pno_offsets[p]]
        d_out  = &d_flat[p * 2]
        Fab    = &Fab_flat[Fab_offsets[p]]

        gamma     = &gamma_scratch[tid, 0]
        Y_trans   = &Y_trans_scratch[tid, 0]
        Y_alt     = &Y_alt_scratch[tid, 0]
        Fia_bar   = &Fia_scratch[tid, 0]
        Z_stacked = &Z_stacked_scratch[tid, 0]
        Z_xxx     = &Z_xxx_scratch[tid, 0]

        int_nlmo         = <int>nlmo
        int_npno         = <int>npno
        int_n_local      = <int>n_local
        int_nlmo_npno    = <int>(nlmo * npno)
        int_npno2        = <int>(npno * npno)
        int_n_local_npno = <int>(n_local * npno)
        int_n_local_nlmo = <int>(n_local * nlmo)

        # -------- Step 1: d_ij, d_ji scalars --------
        acc_chem = 0.0
        acc_ji = 0.0
        for m in range(nlmo):
            for a in range(npno):
                acc_chem = acc_chem + T1[m * npno + a] * K_chem[m * npno + a]
                acc_ji   = acc_ji   + T1[m * npno + a] * K_ji[m * npno + a]
        d_out[0] = 2.0 * acc_chem - acc_ji
        if need_dji:
            acc_ij = 0.0
            for m in range(nlmo):
                for a in range(npno):
                    acc_ij = acc_ij + T1[m * npno + a] * K_ij[m * npno + a]
            d_out[1] = 2.0 * acc_chem - acc_ij

        # -------- Step 2: Fab init to diag(e_pno) --------
        for a in range(npno):
            for b in range(npno):
                Fab[a * npno + b] = 0.0
            Fab[a * npno + a] = e_pno[a]

        # -------- gamma = Qma_flat @ T1.ravel() (via dgemv 'T') --------
        dgemv(&T_flag, &int_nlmo_npno, &int_n_local,
              &one, Qma, &int_nlmo_npno,
              T1, &int_one,
              &zero, gamma, &int_one)

        # -------- Fab += 2 * Qab_flat.T @ gamma (dgemv 'N', beta=1 accum) --------
        dgemv(&N_flag, &int_npno2, &int_n_local,
              &two, Qab, &int_npno2,
              gamma, &int_one,
              &one, Fab, &int_one)

        # -------- Y_trans_flat = Qab_flat @ T1.T (dgemm with swap-trick) --------
        # Row-major: Y(n_local*npno, nlmo) = Qab(n_local*npno, npno) @ T1.T(npno, nlmo)
        # Col-major view: Y_col(nlmo, n_local*npno) = T1_col^T @ Qab_col
        dgemm(&T_flag, &N_flag,
              &int_nlmo, &int_n_local_npno, &int_npno,
              &one, T1, &int_npno,
              Qab, &int_npno,
              &zero, Y_trans, &int_nlmo)

        # -------- Transpose Y_trans[L, b, m] → Y_alt[L, m, b] --------
        for L in range(n_local):
            for m in range(nlmo):
                for b in range(npno):
                    Y_alt[L * nlmo * npno + m * npno + b] = (
                        Y_trans[L * npno * nlmo + b * nlmo + m])

        # -------- Fab -= Y_alt_flat.T @ Qma_flat --------
        dgemm(&N_flag, &T_flag,
              &int_npno, &int_npno, &int_n_local_nlmo,
              &neg_one, Qma, &int_npno,
              Y_alt, &int_npno,
              &one, Fab, &int_npno)

        # -------- Fia_bar = 2 * Qma_flat.T @ gamma (dgemv 'N') --------
        dgemv(&N_flag, &int_nlmo_npno, &int_n_local,
              &two, Qma, &int_nlmo_npno,
              gamma, &int_one,
              &zero, Fia_bar, &int_one)

        # -------- Z_stacked = Qma_flat @ T1.T (dgemm) --------
        dgemm(&T_flag, &N_flag,
              &int_nlmo, &int_n_local_nlmo, &int_npno,
              &one, T1, &int_npno,
              Qma, &int_npno,
              &zero, Z_stacked, &int_nlmo)

        # -------- Transpose Z_stacked[L, j, i] → Z_xxx[L, i, j] --------
        for L in range(n_local):
            for m in range(nlmo):
                for mp in range(nlmo):
                    Z_xxx[L * nlmo * nlmo + mp * nlmo + m] = (
                        Z_stacked[L * nlmo * nlmo + m * nlmo + mp])

        # -------- Fia_bar -= Z_xxx_flat.T @ Qma_flat --------
        dgemm(&N_flag, &T_flag,
              &int_npno, &int_nlmo, &int_n_local_nlmo,
              &neg_one, Qma, &int_npno,
              Z_xxx, &int_nlmo,
              &one, Fia_bar, &int_npno)

        # -------- Fab -= T1.T @ Fia_bar --------
        dgemm(&N_flag, &T_flag,
              &int_npno, &int_npno, &int_nlmo,
              &neg_one, Fia_bar, &int_npno,
              T1, &int_npno,
              &one, Fab, &int_npno)
