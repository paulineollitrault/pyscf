# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Cython kernel for compute_G_term_batched (T2 residual Eq 81).

Per-item semantics:

    tmp     = S @ T2                                  (n_ij × n_ik)
    out     = tmp @ S.T                               (n_ij × n_ij)
    result  = -scalar * out
    out_G[idx] += result

Shapes::

    S       : (N, n_ij, n_ik)
    T2      : (N, n_ik, n_ik)   — already transposed where needed
    scalars : (N,)              — G_tilde[k, j] or G_tilde[k, i]
    idx     : (N,) int64        — target pair slot in out_G
    out_G   : (n_slots, n_ij, n_ij)

Inner matmul chain uses ``scipy.linalg.cython_blas.dgemm`` from within
the nogil prange region so each thread does its own dgemm calls at
single-thread BLAS.  This beats hand-rolled loops at n_pno >= 20 (BLAS
is cache-blocked + SIMD-optimised) and beats numpy batched matmul for
this bucket-parallel pattern because we avoid the per-bucket batched
BLAS dispatch and get nogil-prange parallelism over items.

Two-stage structure matches the other Phase 4+ kernels:

  1. Parallel ``prange`` produces per-item contrib tiles ``C[n]`` into
     a pre-allocated (N, n_ij, n_ij) scratch via two dgemm calls per
     item (row-major-on-Fortran-BLAS trick).
  2. Short sequential phase scatter-adds into the shared output.
"""
from cython.parallel cimport prange
from scipy.linalg.cython_blas cimport dgemm

import numpy as np


def g_kernel(double[:, :, ::1] S,
             double[:, :, ::1] T2,
             double[::1] scalars,
             long[::1] idx,
             double[:, :, ::1] out_G):
    """Scatter-accumulate G contributions for a bucket.

    Output shape bucket (n_ij, n_ik) is uniform across items in this
    call; the caller buckets by (n_ij, n_ik) and invokes once per shape.
    """
    cdef Py_ssize_t N = S.shape[0]
    cdef Py_ssize_t n_ij = S.shape[1]
    cdef Py_ssize_t n_ik = S.shape[2]
    cdef Py_ssize_t n, a, d
    cdef long target

    if N == 0:
        return

    # Per-item scratch.  ``tmp`` holds S @ T2 (shape n_ij × n_ik), ``C``
    # holds the scaled output tile (shape n_ij × n_ij).  Allocated
    # outside nogil so prange iterations index their own row without
    # allocator contention.
    cdef double[:, :, ::1] tmp_all = np.empty((N, n_ij, n_ik))
    cdef double[:, :, ::1] C_all   = np.empty((N, n_ij, n_ij))

    # BLAS dgemm args (shared per item but local to each thread).
    cdef char N_ = b'N'
    cdef char T_ = b'T'
    cdef double one = 1.0
    cdef double zero = 0.0
    cdef double neg_scalar
    cdef int int_n_ij = <int>n_ij
    cdef int int_n_ik = <int>n_ik

    # --- Stage 1: parallel per-item compute via dgemm ---
    # For row-major (C-order) arrays A (M, K), B (K, N) → C (M, N),
    # call dgemm('N', 'N', N, M, K, 1.0, B, N, A, K, 0.0, C, N).  This
    # computes C^T = B^T @ A^T which is C = A @ B in row-major.  Use
    # scalar fused into C = alpha * A @ B to avoid a separate scale.
    with nogil:
        for n in prange(N, schedule='dynamic'):
            # tmp[n] = S[n] @ T2[n]
            #   A = S[n] (n_ij × n_ik, lda=n_ik in row-major)
            #   B = T2[n] (n_ik × n_ik, ldb=n_ik)
            #   C = tmp[n] (n_ij × n_ik, ldc=n_ik)
            dgemm(&N_, &N_,
                  &int_n_ik, &int_n_ij, &int_n_ik,
                  &one,
                  &T2[n, 0, 0], &int_n_ik,
                  &S[n, 0, 0], &int_n_ik,
                  &zero,
                  &tmp_all[n, 0, 0], &int_n_ik)

            # C[n] = (-scalar) * tmp[n] @ S[n].T
            #   A = tmp[n] (n_ij × n_ik, lda=n_ik)
            #   B = S[n]   (n_ij × n_ik, accessed as its transpose)
            #   C = C[n]   (n_ij × n_ij, ldc=n_ij)
            # For A (M=n_ij, K=n_ik) @ B.T (where B has shape (n_ij,
            # n_ik) in row-major, so B.T is (n_ik, n_ij)), the result
            # is (n_ij, n_ij).  Row-major trick: dgemm('T', 'N', N, M,
            # K, alpha, B, K, A, K, 0.0, C, N) for A @ B.T where A is
            # (M, K) row-major and B is (N, K) row-major.
            neg_scalar = -scalars[n]
            dgemm(&T_, &N_,
                  &int_n_ij, &int_n_ij, &int_n_ik,
                  &neg_scalar,
                  &S[n, 0, 0], &int_n_ik,
                  &tmp_all[n, 0, 0], &int_n_ik,
                  &zero,
                  &C_all[n, 0, 0], &int_n_ij)

    # --- Stage 2: sequential scatter-add into flat output ---
    for n in range(N):
        target = idx[n]
        for a in range(n_ij):
            for d in range(n_ij):
                out_G[target, a, d] += C_all[n, a, d]
