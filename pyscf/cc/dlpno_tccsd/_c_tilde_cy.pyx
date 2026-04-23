# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Cython kernels for compute_C_tilde Terms 3 + 4 — Phase 4 canary.

Signatures match the corresponding functions in ``_c_tilde_numba.py``
exactly so the Python driver can swap one for the other without
changing the caller.  Per-triple work uses hand-rolled loops (the
matrices are small — n_pno ≈ 25 — so direct loops beat BLAS dispatch
overhead; the C compiler vectorises them at ``-O3 -march=native``).

Two-stage structure (race-free):

  1. ``prange`` over triples with ``nogil``.  Each thread computes its
     own (n_ki, n_ki) contribution into a pre-allocated per-triple
     scratch buffer.  All per-triple arithmetic is in private, per-
     thread stack-local scalars — no synchronisation needed.

  2. Sequential scatter-add into the flat output using the ``idx``
     lookup.  This runs after the ``prange`` barrier and is cheap
     (O(N × n_ki²) floating-point adds with no dispatch overhead).
"""
from cython.parallel cimport prange

import numpy as np


def t3_kernel(double[:, :, ::1] K,
              double[:, :, ::1] S,
              double[:, ::1] t1i,
              double[:, ::1] T1l,
              long[::1] idx,
              double[:, :, ::1] out):
    """Accumulate Term 3 contributions into ``out``.

    For each triple n in [0, N):
        Kt1    = K[n].T  @ t1i[n]            (n_kl,)
        Kt1_ki = S[n]    @ Kt1               (n_ki,)
        out[idx[n], a, c] -= T1l[n, a] * Kt1_ki[c]

    Matches ``_c_tilde_numba.t3_kernel`` bit-for-bit up to FP reordering.
    """
    cdef Py_ssize_t N = K.shape[0]
    cdef Py_ssize_t n_kl = K.shape[1]
    cdef Py_ssize_t n_ki = S.shape[1]
    cdef Py_ssize_t n, a, b, c
    cdef double v, s
    cdef long target

    if N == 0:
        return

    # N-sized scratch: each triple writes its own row.  These are shared
    # memoryviews, but prange iterations address disjoint slices so
    # there's no race.  Buffers fall out of scope after the kernel.
    cdef double[:, ::1] wks_kt1    = np.empty((N, n_kl))
    cdef double[:, ::1] wks_kt1_ki = np.empty((N, n_ki))
    cdef double[:, :, ::1] contrib = np.empty((N, n_ki, n_ki))

    # --- Stage 1: parallel per-triple compute ---
    with nogil:
        for n in prange(N, schedule='dynamic'):
            # Kt1 = K[n].T @ t1i[n]    (GEMV with transpose)
            for a in range(n_kl):
                s = 0.0
                for b in range(n_kl):
                    s = s + K[n, b, a] * t1i[n, b]
                wks_kt1[n, a] = s
            # Kt1_ki = S[n] @ Kt1      (GEMV, no transpose)
            for a in range(n_ki):
                s = 0.0
                for b in range(n_kl):
                    s = s + S[n, a, b] * wks_kt1[n, b]
                wks_kt1_ki[n, a] = s
            # contrib[n] = -outer(T1l[n], Kt1_ki)   (rank-1 update)
            for a in range(n_ki):
                v = -T1l[n, a]
                for c in range(n_ki):
                    contrib[n, a, c] = v * wks_kt1_ki[n, c]

    # --- Stage 2: sequential scatter-add into flat output ---
    for n in range(N):
        target = idx[n]
        for a in range(n_ki):
            for c in range(n_ki):
                out[target, a, c] += contrib[n, a, c]


def t4_kernel(double[:, :, ::1] S_ki_li,
              double[:, :, ::1] t2,
              double[:, :, ::1] S_li_kl,
              double[:, :, ::1] K,
              double[:, :, ::1] S_kl_ki,
              long[::1] idx,
              double[:, :, ::1] out,
              double scale=-0.5):
    """Accumulate Term 4 contributions into ``out``.

    For each triple n:
        tmp1 = S_ki_li[n] @ t2[n]          (n_ki, n_li)
        tmp2 = tmp1       @ S_li_kl[n]     (n_ki, n_kl)
        tmp3 = tmp2       @ K[n]           (n_ki, n_kl)
        C    = tmp3       @ S_kl_ki[n]     (n_ki, n_ki)
        out[idx[n]] += scale * C

    ``scale`` defaults to -0.5 (C_tilde convention); D_tilde Term 4 uses
    ``scale=+0.5`` with ``K = L_lk`` and ``t2 = u_il``.
    """
    cdef Py_ssize_t N = S_ki_li.shape[0]
    cdef Py_ssize_t n_ki = S_ki_li.shape[1]
    cdef Py_ssize_t n_li = S_ki_li.shape[2]
    cdef Py_ssize_t n_kl = S_li_kl.shape[2]
    cdef Py_ssize_t n, a, b, c, d
    cdef double s
    cdef long target

    if N == 0:
        return

    cdef double[:, :, ::1] wks_tmp1 = np.empty((N, n_ki, n_li))
    cdef double[:, :, ::1] wks_tmp2 = np.empty((N, n_ki, n_kl))
    cdef double[:, :, ::1] wks_tmp3 = np.empty((N, n_ki, n_kl))
    cdef double[:, :, ::1] contrib  = np.empty((N, n_ki, n_ki))

    with nogil:
        for n in prange(N, schedule='dynamic'):
            # tmp1 = S_ki_li[n] @ t2[n]
            for a in range(n_ki):
                for b in range(n_li):
                    s = 0.0
                    for c in range(n_li):
                        s = s + S_ki_li[n, a, c] * t2[n, c, b]
                    wks_tmp1[n, a, b] = s
            # tmp2 = tmp1 @ S_li_kl[n]
            for a in range(n_ki):
                for b in range(n_kl):
                    s = 0.0
                    for c in range(n_li):
                        s = s + wks_tmp1[n, a, c] * S_li_kl[n, c, b]
                    wks_tmp2[n, a, b] = s
            # tmp3 = tmp2 @ K[n]
            for a in range(n_ki):
                for b in range(n_kl):
                    s = 0.0
                    for c in range(n_kl):
                        s = s + wks_tmp2[n, a, c] * K[n, c, b]
                    wks_tmp3[n, a, b] = s
            # contrib = scale * tmp3 @ S_kl_ki[n]
            for a in range(n_ki):
                for b in range(n_ki):
                    s = 0.0
                    for c in range(n_kl):
                        s = s + wks_tmp3[n, a, c] * S_kl_ki[n, c, b]
                    contrib[n, a, b] = scale * s

    for n in range(N):
        target = idx[n]
        for a in range(n_ki):
            for b in range(n_ki):
                out[target, a, b] += contrib[n, a, b]
