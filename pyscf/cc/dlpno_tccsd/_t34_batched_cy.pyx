# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Single-call batched Cython kernels for compute_C_tilde / build_D_tilde
Phase 2 (Terms 3 + 4).

Replaces the per-shape-bucket calls to ``t3_kernel`` / ``t4_kernel``
(``_c_tilde_cy.pyx``) with two single nogil ``prange`` calls that
process all items at once via per-item shape + offset arrays. Same
two-stage structure (parallel compute → serial scatter-add) — output
``tiles_flat`` per item, caller scatters into per-(n_ki, side) output
buffers.

Per-item math matches the reference t3_kernel / t4_kernel exactly:

  t3 (per item n):
    Kt1[a]    = sum_b K[b, a] * t1i[b]            (n_kl,)
    Kt1_ki[a] = sum_b S[a, b] * Kt1[b]            (n_ki,)
    contrib[a, c] = -T1l[a] * Kt1_ki[c]           (n_ki, n_ki) — rank-1

  t4 (per item n):
    tmp1[a, b] = sum_c S_ki_li[a, c] * t2[c, b]    (n_ki, n_li)
    tmp2[a, b] = sum_c tmp1[a, c]   * S_li_kl[c, b](n_ki, n_kl)
    tmp3[a, b] = sum_c tmp2[a, c]   * K[c, b]      (n_ki, n_kl)
    contrib[a, b] = scale *
                    sum_c tmp3[a, c] * S_kl_ki[c, b]  (n_ki, n_ki)
"""
from cython.parallel cimport prange
cimport openmp

import numpy as np


def t3_kernel_batched(
    int N,
    int max_n_ki, int max_n_kl,

    # Per-item shapes
    int[::1] n_kl_arr,
    int[::1] n_ki_arr,

    # Per-item flat offsets into static and dynamic input buffers
    long[::1] K_off,                    # (N,) into K_flat
    long[::1] S_off,                    # (N,) into S_flat
    long[::1] t1i_off,                  # (N,) into t1_flat (length n_kl per item)
    long[::1] T1l_off,                  # (N,) into t1_flat (length n_ki per item)
    long[::1] tile_off,                 # (N+1,) into tiles_flat

    # Static flat buffers
    double[::1] K_flat,
    double[::1] S_flat,

    # Dynamic flat buffer (rebuilt per cycle): per-item t1 vectors
    double[::1] t1_flat,

    # Per-thread scratch
    double[:, ::1] Kt1_scratch,         # (num_threads, max_n_kl)
    double[:, ::1] Kt1_ki_scratch,      # (num_threads, max_n_ki)

    # Output: per-item contrib tiles (caller scatters)
    double[::1] tiles_flat,

    int num_threads,
):
    cdef Py_ssize_t n
    cdef int tid
    cdef int n_kl, n_ki
    cdef Py_ssize_t a, b, c
    cdef double s, v
    cdef double *K
    cdef double *S
    cdef double *t1i
    cdef double *T1l
    cdef double *Kt1
    cdef double *Kt1_ki
    cdef double *contrib

    for n in prange(N, schedule='dynamic',
                    nogil=True, num_threads=num_threads):
        tid = openmp.omp_get_thread_num()
        n_kl = n_kl_arr[n]
        n_ki = n_ki_arr[n]

        K = &K_flat[K_off[n]]              # (n_kl, n_ki)
        S = &S_flat[S_off[n]]              # (n_ki, n_kl)
        t1i = &t1_flat[t1i_off[n]]         # (n_kl,)
        T1l = &t1_flat[T1l_off[n]]         # (n_ki,)
        Kt1 = &Kt1_scratch[tid, 0]
        Kt1_ki = &Kt1_ki_scratch[tid, 0]
        contrib = &tiles_flat[tile_off[n]]  # (n_ki, n_ki)

        # K is (n_kl, n_kl), S is (n_ki, n_kl).
        # Kt1[a] = sum_b K[b, a] * t1i[b]   (= K.T @ t1i, shape (n_kl,))
        for a in range(n_kl):
            s = 0.0
            for b in range(n_kl):
                s = s + K[b * n_kl + a] * t1i[b]
            Kt1[a] = s

        # Kt1_ki[a] = sum_b S[a, b] * Kt1[b]   for a in [0, n_ki)
        for a in range(n_ki):
            s = 0.0
            for b in range(n_kl):
                s = s + S[a * n_kl + b] * Kt1[b]
            Kt1_ki[a] = s

        # contrib[a, c] = -T1l[a] * Kt1_ki[c]
        for a in range(n_ki):
            v = -T1l[a]
            for c in range(n_ki):
                contrib[a * n_ki + c] = v * Kt1_ki[c]


def t4_kernel_batched(
    int N,
    int max_n_ki, int max_n_li, int max_n_kl,

    # Per-item shapes
    int[::1] n_ki_arr,
    int[::1] n_li_arr,
    int[::1] n_kl_arr,

    # Per-item flat offsets
    long[::1] S_ki_li_off,
    long[::1] t2_off,
    long[::1] S_li_kl_off,
    long[::1] K_off,
    long[::1] S_kl_ki_off,
    long[::1] tile_off,

    # Static flat buffers (S projections, K)
    double[::1] S_ki_li_flat,
    double[::1] S_li_kl_flat,
    double[::1] K_flat,
    double[::1] S_kl_ki_flat,

    # Dynamic flat buffer (rebuilt per cycle): t2 stacked per item
    double[::1] t2_flat,

    # Per-thread scratch
    double[:, ::1] tmp1_scratch,        # (num_threads, max_n_ki * max_n_li)
    double[:, ::1] tmp2_scratch,        # (num_threads, max_n_ki * max_n_kl)
    double[:, ::1] tmp3_scratch,        # (num_threads, max_n_ki * max_n_kl)

    # Output
    double[::1] tiles_flat,
    double scale,

    int num_threads,
):
    cdef Py_ssize_t n
    cdef int tid
    cdef int n_ki, n_li, n_kl
    cdef Py_ssize_t a, b, c
    cdef double s
    cdef double *S_ki_li
    cdef double *t2
    cdef double *S_li_kl
    cdef double *K
    cdef double *S_kl_ki
    cdef double *tmp1
    cdef double *tmp2
    cdef double *tmp3
    cdef double *contrib

    for n in prange(N, schedule='dynamic',
                    nogil=True, num_threads=num_threads):
        tid = openmp.omp_get_thread_num()
        n_ki = n_ki_arr[n]
        n_li = n_li_arr[n]
        n_kl = n_kl_arr[n]

        S_ki_li = &S_ki_li_flat[S_ki_li_off[n]]    # (n_ki, n_li)
        t2      = &t2_flat[t2_off[n]]              # (n_li, n_li)
        S_li_kl = &S_li_kl_flat[S_li_kl_off[n]]    # (n_li, n_kl)
        K       = &K_flat[K_off[n]]                # (n_kl, n_kl)
        S_kl_ki = &S_kl_ki_flat[S_kl_ki_off[n]]    # (n_kl, n_ki)
        tmp1    = &tmp1_scratch[tid, 0]
        tmp2    = &tmp2_scratch[tid, 0]
        tmp3    = &tmp3_scratch[tid, 0]
        contrib = &tiles_flat[tile_off[n]]         # (n_ki, n_ki)

        # tmp1[a, b] = sum_c S_ki_li[a, c] * t2[c, b]
        for a in range(n_ki):
            for b in range(n_li):
                s = 0.0
                for c in range(n_li):
                    s = s + S_ki_li[a * n_li + c] * t2[c * n_li + b]
                tmp1[a * n_li + b] = s

        # tmp2[a, b] = sum_c tmp1[a, c] * S_li_kl[c, b]
        for a in range(n_ki):
            for b in range(n_kl):
                s = 0.0
                for c in range(n_li):
                    s = s + tmp1[a * n_li + c] * S_li_kl[c * n_kl + b]
                tmp2[a * n_kl + b] = s

        # tmp3[a, b] = sum_c tmp2[a, c] * K[c, b]
        for a in range(n_ki):
            for b in range(n_kl):
                s = 0.0
                for c in range(n_kl):
                    s = s + tmp2[a * n_kl + c] * K[c * n_kl + b]
                tmp3[a * n_kl + b] = s

        # contrib[a, b] = scale * sum_c tmp3[a, c] * S_kl_ki[c, b]
        for a in range(n_ki):
            for b in range(n_ki):
                s = 0.0
                for c in range(n_kl):
                    s = s + tmp3[a * n_kl + c] * S_kl_ki[c * n_ki + b]
                contrib[a * n_ki + b] = scale * s
