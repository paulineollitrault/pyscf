# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Cython kernel for compute_B_E_batched (Phase 5c).

Per-item semantics (one item per (ij, kl) with kl in domain_ij):

    TB  = beta_kl * T                               if same
        = beta_kl * T + beta_lk * T.T               otherwise
    u   = 2*T - T.T                                 (always)
    UK  = u @ K.T                                   if same
        = u @ K.T + (2*T.T - T) @ K                 otherwise
    B_contrib = S @ TB @ S.T                        (n_ij, n_ij)
    E_contrib = S @ UK @ S.T                        (n_ij, n_ij)
    out_B[idx] += B_contrib
    out_E[idx] += E_contrib

Shapes::
    S        : (N, n_ij, n_kl)
    T, K     : (N, n_kl, n_kl)
    beta_kl, beta_lk : (N,)
    same     : (N,) uint8  (1 if k == l else 0)
    idx      : (N,) int64
    out_B    : (n_slots, n_ij, n_ij)
    out_E    : (n_slots, n_ij, n_ij)

Compute structure is the same two-stage pattern as the C_tilde kernels:
parallel ``prange`` produces per-item contributions into N-sized scratch
(Bc, Ec); a short serial phase scatter-adds them into the shared output
buffers.  TB / UK are never materialised — STB (= S @ TB) and SUK (=
S @ UK) go directly from T and K via hand-rolled triple loops, so the
per-item footprint is only the two (n_ij, n_ij) contrib tiles.
"""
from cython.parallel cimport prange

import numpy as np


def be_kernel(double[:, :, ::1] S,
              double[:, :, ::1] T,
              double[:, :, ::1] K,
              double[::1] beta_kl,
              double[::1] beta_lk,
              unsigned char[::1] same,
              long[::1] idx,
              double[:, :, ::1] out_B,
              double[:, :, ::1] out_E):
    """Scatter-accumulate B and E contributions for a bucket.

    Output shape bucket (n_ij, n_kl) is uniform across items in this
    call; the caller buckets by (n_ij, n_kl) and invokes once per shape.
    """
    cdef Py_ssize_t N = S.shape[0]
    cdef Py_ssize_t n_ij = S.shape[1]
    cdef Py_ssize_t n_kl = S.shape[2]
    cdef Py_ssize_t n, a, b, c, d
    cdef double bkl, blk
    cdef double s_stb, s_suk, s_bc, s_ec, t_bc, t_cb, t_bd, t_db, k_cd, k_dc
    cdef double uk_bc
    cdef unsigned char sm
    cdef long target

    if N == 0:
        return

    # Per-item (n_ij, n_kl) intermediates (used within one prange iter
    # only) and (n_ij, n_ij) output tiles (consumed by the scatter phase).
    cdef double[:, :, ::1] STB = np.empty((N, n_ij, n_kl))
    cdef double[:, :, ::1] SUK = np.empty((N, n_ij, n_kl))
    cdef double[:, :, ::1] UK  = np.empty((N, n_kl, n_kl))
    cdef double[:, :, ::1] Bc  = np.empty((N, n_ij, n_ij))
    cdef double[:, :, ::1] Ec  = np.empty((N, n_ij, n_ij))

    # --- Stage 1: parallel per-item compute ---
    with nogil:
        for n in prange(N, schedule='dynamic'):
            bkl = beta_kl[n]
            blk = beta_lk[n]
            sm = same[n]

            # UK[b, c] = sum_d (2*T[b,d] - T[d,b]) * K[c, d]
            # If !same:   += sum_d (2*T[d,b] - T[b,d]) * K[d, c]
            if sm:
                for b in range(n_kl):
                    for c in range(n_kl):
                        uk_bc = 0.0
                        for d in range(n_kl):
                            uk_bc = uk_bc + (2.0 * T[n, b, d] - T[n, d, b]) * K[n, c, d]
                        UK[n, b, c] = uk_bc
            else:
                for b in range(n_kl):
                    for c in range(n_kl):
                        uk_bc = 0.0
                        for d in range(n_kl):
                            uk_bc = uk_bc + (2.0 * T[n, b, d] - T[n, d, b]) * K[n, c, d]
                            uk_bc = uk_bc + (2.0 * T[n, d, b] - T[n, b, d]) * K[n, d, c]
                        UK[n, b, c] = uk_bc

            # STB[a, c] = sum_b S[a, b] * TB[b, c]
            # TB[b, c] = bkl * T[b, c]                       (same)
            #          = bkl * T[b, c] + blk * T[c, b]        (!same)
            if sm:
                for a in range(n_ij):
                    for c in range(n_kl):
                        s_stb = 0.0
                        for b in range(n_kl):
                            s_stb = s_stb + S[n, a, b] * (bkl * T[n, b, c])
                        STB[n, a, c] = s_stb
            else:
                for a in range(n_ij):
                    for c in range(n_kl):
                        s_stb = 0.0
                        for b in range(n_kl):
                            s_stb = s_stb + S[n, a, b] * (bkl * T[n, b, c]
                                                          + blk * T[n, c, b])
                        STB[n, a, c] = s_stb

            # SUK[a, c] = sum_b S[a, b] * UK[b, c]
            for a in range(n_ij):
                for c in range(n_kl):
                    s_suk = 0.0
                    for b in range(n_kl):
                        s_suk = s_suk + S[n, a, b] * UK[n, b, c]
                    SUK[n, a, c] = s_suk

            # Bc[a, d] = sum_c STB[a, c] * S[d, c]
            # Ec[a, d] = sum_c SUK[a, c] * S[d, c]
            for a in range(n_ij):
                for d in range(n_ij):
                    s_bc = 0.0
                    s_ec = 0.0
                    for c in range(n_kl):
                        s_bc = s_bc + STB[n, a, c] * S[n, d, c]
                        s_ec = s_ec + SUK[n, a, c] * S[n, d, c]
                    Bc[n, a, d] = s_bc
                    Ec[n, a, d] = s_ec

    # --- Stage 2: sequential scatter-add into flat outputs ---
    for n in range(N):
        target = idx[n]
        for a in range(n_ij):
            for d in range(n_ij):
                out_B[target, a, d] += Bc[n, a, d]
                out_E[target, a, d] += Ec[n, a, d]
