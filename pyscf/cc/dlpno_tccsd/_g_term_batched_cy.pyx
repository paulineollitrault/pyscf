# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Single-call batched Cython kernel for ``compute_G_term_batched``.

Replaces the per-bucket loop dispatching ~900 ``np.matmul`` calls per
cycle (one per (n_ij, n_ik, side) shape group) with a single nogil
prange call that processes all ~16k items at once via per-item shape +
offset arrays. Same two-stage structure as the cd / t34 batched paths
(parallel compute → serial scatter).

Per-item math (matches reference in residual.py:402-429):

  S    : (n_ij, n_ik)   from S_arr_flat at per-item S_off
  t2   : (n_ik, n_ik)   from t2_pno_all._buffer at per-item t2_canon_off,
                        with optional transpose handled by the gather
                        Cython kernel before this kernel runs
  scalar = G_tilde[k_idx[n], scalar_lmo[n]]

  tmp[a, b]  = sum_c S[a, c] * t2[c, b]       (n_ij, n_ik)
  Cc[a, d]   = scalar * sum_b tmp[a, b] * S[d, b]  (n_ij, n_ij)

  out[target_off[n] : target_off[n] + n_ij²] -= Cc   (caller scatters)
"""
from cython.parallel cimport prange
cimport openmp

import numpy as np


def g_term_batched(
    int N,
    int max_n_ij, int max_n_ik,

    # Per-item metadata
    int[::1] n_ij_arr,
    int[::1] n_ik_arr,
    long[::1] S_off,
    long[::1] t2_off,
    long[::1] tile_off,                 # (N+1,) per-item Cc tile offsets

    # Per-item scalar lookups into G_tilde
    long[::1] k_idx,                    # (N,) k value
    long[::1] scalar_lmo,               # (N,) other LMO (j for ik-side, i for jk-side)

    # Static + dynamic flat buffers
    double[::1] S_flat,                 # static
    double[::1] t2_flat,                # dynamic (filled by gather kernel)
    double[:, ::1] G_tilde,             # (nocc, nocc) — scalar source

    # Per-thread scratch
    double[:, ::1] tmp_scratch,         # (num_threads, max_n_ij * max_n_ik)

    # Output: per-item Cc tiles (caller scatters into per-n_ij output)
    double[::1] tiles_flat,

    int num_threads,
):
    cdef Py_ssize_t n
    cdef int tid
    cdef int n_ij, n_ik
    cdef Py_ssize_t a, b, c, d
    cdef double s, scalar
    cdef double *S
    cdef double *t2
    cdef double *tmp
    cdef double *Cc

    for n in prange(N, schedule='dynamic',
                    nogil=True, num_threads=num_threads):
        tid = openmp.omp_get_thread_num()
        n_ij = n_ij_arr[n]
        n_ik = n_ik_arr[n]

        S = &S_flat[S_off[n]]              # (n_ij, n_ik)
        t2 = &t2_flat[t2_off[n]]           # (n_ik, n_ik)
        tmp = &tmp_scratch[tid, 0]         # (n_ij, n_ik)
        Cc = &tiles_flat[tile_off[n]]      # (n_ij, n_ij)

        scalar = G_tilde[k_idx[n], scalar_lmo[n]]

        # tmp[a, b] = sum_c S[a, c] * t2[c, b]
        for a in range(n_ij):
            for b in range(n_ik):
                s = 0.0
                for c in range(n_ik):
                    s = s + S[a * n_ik + c] * t2[c * n_ik + b]
                tmp[a * n_ik + b] = s

        # Cc[a, d] = scalar * sum_b tmp[a, b] * S[d, b]
        for a in range(n_ij):
            for d in range(n_ij):
                s = 0.0
                for b in range(n_ik):
                    s = s + tmp[a * n_ik + b] * S[d * n_ik + b]
                Cc[a * n_ij + d] = scalar * s
