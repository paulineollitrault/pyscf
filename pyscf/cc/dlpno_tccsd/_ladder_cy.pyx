# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: language_level=3
"""Cython inner loop for ladder accumulation: dress + contract + accumulate.

Called after AO→MO transforms are done in Python (nr_e2 is already C).
This handles the per-pair: B_tilde = B - corr; X = B_tilde @ tau; ladder += X' @ B_tilde'
using direct BLAS calls without Python overhead.
"""
cimport numpy as cnp
import numpy as np
from scipy.linalg.cython_blas cimport dgemm

cnp.import_array()


def process_ladder_batch(
    list B_L_list,     # list of (nL, n_pno, n_pno) arrays from nr_e2
    list corr_list,    # list of (nL, n_pno, n_pno) corr batch slices
    list tau_list,     # list of (n_pno, n_pno) tau arrays
    list ladder_list,  # list of (n_pno, n_pno) arrays (accumulated in-place)
    list n_pno_list,   # list of int PNO sizes
):
    """Process dress+contract+accumulate for all pairs in one batch.

    For each pair p:
      B_tilde = B_L[p] - corr[p]       (nL, n, n)
      X = B_tilde @ tau[p]              (nL, n, n) batched matmul
      ladder[p] += X' @ B_tilde'        reshape to (n, nL*n) then dgemm
    """
    cdef int n_pairs = len(B_L_list)
    cdef int p, n, nL, nLn
    cdef double alpha = 1.0, beta = 1.0  # accumulate
    cdef char transN = b'N'
    cdef char transT = b'T'

    cdef cnp.ndarray[double, ndim=3] B_L, corr_b, X_L, B_tilde
    cdef cnp.ndarray[double, ndim=2] tau, ladder, X_r, B_r

    for p in range(n_pairs):
        n = n_pno_list[p]
        if n == 0:
            continue

        B_L = B_L_list[p]
        corr_b = corr_list[p]
        tau = tau_list[p]
        ladder = ladder_list[p]
        nL = B_L.shape[0]

        # B_tilde = B_L - corr (in-place to avoid allocation)
        B_tilde = np.subtract(B_L, corr_b)

        # X = B_tilde @ tau (batch matmul)
        X_L = np.matmul(B_tilde, tau)

        # ladder += X_r @ B_r.T where X_r, B_r are (n, nL*n) reshaped views
        # Use BLAS dgemm directly: C = alpha * A @ B^T + beta * C
        X_r = np.ascontiguousarray(X_L.transpose(1, 0, 2).reshape(n, nL * n))
        B_r = np.ascontiguousarray(B_tilde.transpose(1, 0, 2).reshape(n, nL * n))
        nLn = nL * n

        # dgemm: ladder(n,n) += X_r(n,nLn) @ B_r(n,nLn)^T
        dgemm(&transN, &transT, &n, &n, &nLn,
               &alpha, &X_r[0,0], &n,
               &B_r[0,0], &n,
               &beta, &ladder[0,0], &n)
