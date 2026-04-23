# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Cython kernels for the dressed C and D contractions inside
compute_residual_v2 (Phase 5e).

Per (ij, k, side) item the C and D contractions have matmul chains
that are uniform in shape across items of a bucket and completely
branch-free once the caller has gathered the per-cycle inputs.  The
two kernels below mirror the arithmetic of the reference path in
residual.py lines 2200-2334.

C term, per item (side = ij or ji, mirrored):
    gamma[a, d]   = J_bold[a, d]
                  + sum_{b,c} S_big[a, b] * ct[b, c] * S_mid[c, d]
    out[idx[n]]  -= sum_{d,e,f} gamma[a, d] * t2[e, d] * S_outer[f, e]
                 (= -(gamma @ t2.T @ S_outer.T))

D term, per item:
    U_proj[a, c] = sum_{b,d} S_a[a, b] * u[b, d] * S_b[d, c]       (Part A)
    D_A[a, f]    = sum_{c}   S_c[a, c] * dt[c, ?] * U_proj[f, ?]
                 (= S_c @ dt @ U_proj.T)
    D_B[a, f]    = sum_{b,d} KJ[a, b] * u[d, b] * S_a[f, d]          (Part B)
                 (= KJ @ u.T @ S_a.T)
    out[idx[n]] += scale * (D_A + D_B)

Missing contributions (absent J_bold, absent ct, absent dt, absent
KJ) are zero-filled by the caller's plan builder so the inner loops
stay branch-free.  The zero-filled flops per missing item are
O(n_pno x n_mid x n_outer) ~= 25^3 = 15k — negligible next to the
BLAS-dispatch wins from eliminating the per-item Python layer.
"""
from cython.parallel cimport prange

import numpy as np


def c_kernel(double[:, :, ::1] S_big,
             double[:, :, ::1] ct,
             double[:, :, ::1] S_mid,
             double[:, :, ::1] J_bold,
             double[:, :, ::1] t2,
             double[:, :, ::1] S_outer,
             long[::1] idx,
             double[:, :, ::1] out):
    """Accumulate the C-term contribution for one (n_pno, n_ct, n_other) bucket.

    Shapes::
        S_big   : (N, n_pno, n_ct)
        ct      : (N, n_ct,  n_ct)     per-cycle
        S_mid   : (N, n_ct,  n_other)
        J_bold  : (N, n_pno, n_other)  zero-filled where absent
        t2      : (N, n_other, n_other) per-cycle
        S_outer : (N, n_pno, n_other)
        idx     : (N,)                  output slot per item
        out     : (n_slots, n_pno, n_pno)  accumulated via -= sign
    """
    cdef Py_ssize_t N = S_big.shape[0]
    cdef Py_ssize_t n_pno = S_big.shape[1]
    cdef Py_ssize_t n_ct = S_big.shape[2]
    cdef Py_ssize_t n_other = S_mid.shape[2]
    cdef Py_ssize_t n, a, b, c, d, e, f
    cdef double s_stb, s_gamma, s_gt, s_cc
    cdef long target

    if N == 0:
        return

    # Per-item (n_pno, n_ct) and (n_pno, n_other) intermediates, plus
    # (n_pno, n_pno) output tile consumed by the scatter phase.
    cdef double[:, :, ::1] STB = np.empty((N, n_pno, n_ct))
    cdef double[:, :, ::1] GAMMA = np.empty((N, n_pno, n_other))
    cdef double[:, :, ::1] GT = np.empty((N, n_pno, n_other))
    cdef double[:, :, ::1] Cc = np.empty((N, n_pno, n_pno))

    with nogil:
        for n in prange(N, schedule='dynamic'):
            # STB = S_big @ ct
            for a in range(n_pno):
                for c in range(n_ct):
                    s_stb = 0.0
                    for b in range(n_ct):
                        s_stb = s_stb + S_big[n, a, b] * ct[n, b, c]
                    STB[n, a, c] = s_stb

            # gamma = STB @ S_mid + J_bold
            for a in range(n_pno):
                for d in range(n_other):
                    s_gamma = J_bold[n, a, d]
                    for c in range(n_ct):
                        s_gamma = s_gamma + STB[n, a, c] * S_mid[n, c, d]
                    GAMMA[n, a, d] = s_gamma

            # GT = gamma @ t2.T  (GT[a, e] = sum_d gamma[a, d] * t2[e, d])
            for a in range(n_pno):
                for e in range(n_other):
                    s_gt = 0.0
                    for d in range(n_other):
                        s_gt = s_gt + GAMMA[n, a, d] * t2[n, e, d]
                    GT[n, a, e] = s_gt

            # Cc = GT @ S_outer.T  (Cc[a, f] = sum_e GT[a, e] * S_outer[f, e])
            for a in range(n_pno):
                for f in range(n_pno):
                    s_cc = 0.0
                    for e in range(n_other):
                        s_cc = s_cc + GT[n, a, e] * S_outer[n, f, e]
                    Cc[n, a, f] = s_cc

    # Sequential scatter-subtract (C is accumulated with -= per item).
    for n in range(N):
        target = idx[n]
        for a in range(n_pno):
            for f in range(n_pno):
                out[target, a, f] -= Cc[n, a, f]


def d_kernel(double[:, :, ::1] S_a,
             double[:, :, ::1] u,
             double[:, :, ::1] S_b,
             double[:, :, ::1] S_c,
             double[:, :, ::1] dt,
             double[:, :, ::1] KJ,
             long[::1] idx,
             double[:, :, ::1] out,
             double scale):
    """Accumulate the D-term contribution for one (n_pno, n_A, n_B) bucket.

    Shapes::
        S_a    : (N, n_pno, n_A)
        u      : (N, n_A,   n_A)   per-cycle
        S_b    : (N, n_A,   n_B)
        S_c    : (N, n_pno, n_B)
        dt     : (N, n_B,   n_B)   per-cycle (zero-filled where absent)
        KJ     : (N, n_pno, n_A)   = 2*K_bold - J_bold  (zero-filled where absent)
        idx    : (N,)
        out    : (n_slots, n_pno, n_pno)
        scale  : +0.5 matches the reference (see residual.py line 2303)
    """
    cdef Py_ssize_t N = S_a.shape[0]
    cdef Py_ssize_t n_pno = S_a.shape[1]
    cdef Py_ssize_t n_A = S_a.shape[2]
    cdef Py_ssize_t n_B = S_b.shape[2]
    cdef Py_ssize_t n, a, b, c, d, f
    cdef double s_su, s_up, s_scd, s_Atile, s_b_int, s_Btile
    cdef long target

    if N == 0:
        return

    cdef double[:, :, ::1] SU    = np.empty((N, n_pno, n_A))    # S_a @ u
    cdef double[:, :, ::1] UP    = np.empty((N, n_pno, n_B))    # SU @ S_b
    cdef double[:, :, ::1] SCD   = np.empty((N, n_pno, n_B))    # S_c @ dt
    cdef double[:, :, ::1] Bint  = np.empty((N, n_pno, n_A))    # KJ @ u.T
    cdef double[:, :, ::1] Dtile = np.empty((N, n_pno, n_pno))

    with nogil:
        for n in prange(N, schedule='dynamic'):
            # SU = S_a @ u
            for a in range(n_pno):
                for d in range(n_A):
                    s_su = 0.0
                    for b in range(n_A):
                        s_su = s_su + S_a[n, a, b] * u[n, b, d]
                    SU[n, a, d] = s_su

            # U_proj = SU @ S_b
            for a in range(n_pno):
                for c in range(n_B):
                    s_up = 0.0
                    for d in range(n_A):
                        s_up = s_up + SU[n, a, d] * S_b[n, d, c]
                    UP[n, a, c] = s_up

            # SCD = S_c @ dt
            for a in range(n_pno):
                for c in range(n_B):
                    s_scd = 0.0
                    for b in range(n_B):
                        s_scd = s_scd + S_c[n, a, b] * dt[n, b, c]
                    SCD[n, a, c] = s_scd

            # Bint = KJ @ u.T   (Bint[a, d] = sum_b KJ[a, b] * u[d, b])
            for a in range(n_pno):
                for d in range(n_A):
                    s_b_int = 0.0
                    for b in range(n_A):
                        s_b_int = s_b_int + KJ[n, a, b] * u[n, d, b]
                    Bint[n, a, d] = s_b_int

            # Dtile[a, f] = Part A + Part B
            #   Part A: sum_c SCD[a, c] * UP[f, c]
            #   Part B: sum_d Bint[a, d] * S_a[f, d]
            for a in range(n_pno):
                for f in range(n_pno):
                    s_Atile = 0.0
                    for c in range(n_B):
                        s_Atile = s_Atile + SCD[n, a, c] * UP[n, f, c]
                    s_Btile = 0.0
                    for d in range(n_A):
                        s_Btile = s_Btile + Bint[n, a, d] * S_a[n, f, d]
                    Dtile[n, a, f] = s_Atile + s_Btile

    # Sequential scatter-add (D is accumulated with += scale * tile).
    for n in range(N):
        target = idx[n]
        for a in range(n_pno):
            for f in range(n_pno):
                out[target, a, f] += scale * Dtile[n, a, f]
