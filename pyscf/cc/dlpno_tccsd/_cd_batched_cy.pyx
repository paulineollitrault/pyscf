# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Single-call batched Cython kernels for ``compute_CD_terms_batched``.

Replaces the 4224 per-bucket ``c_kernel`` / ``d_kernel`` calls / cycle
with two single nogil ``prange`` calls (one for C, one for D) that
process all ~11500 items at once. Per-item shape variability handled
via flat offset arrays. Static metadata (S_big, S_mid, J_bold, S_outer
for C; S_a, S_b, S_c, KJ for D) is concatenated into one big buffer
per field at plan-build time. Per-cycle dynamic data (ct, t2, u, dt)
filled into pre-allocated flat buffers each iteration.

Per-item math matches _cd_cy.pyx exactly:

  C term (per item):
    STB[a, c]   = sum_b S_big[a, b] * ct[b, c]
    gamma[a, d] = J_bold[a, d] + sum_c STB[a, c] * S_mid[c, d]
    GT[a, e]    = sum_d gamma[a, d] * t2[e, d]
    Cc[a, f]    = sum_e GT[a, e] * S_outer[f, e]
    out[target, a, f] -= Cc[a, f]

  D term (per item):
    SU[a, d]    = sum_b S_a[a, b] * u[b, d]
    UP[a, c]    = sum_d SU[a, d] * S_b[d, c]
    SCD[a, c]   = sum_b S_c[a, b] * dt[b, c]
    Bint[a, d]  = sum_b KJ[a, b] * u[d, b]
    Dtile[a, f] = sum_c SCD[a, c] * UP[f, c] + sum_d Bint[a, d] * S_a[f, d]
    out[target, a, f] += scale * Dtile[a, f]

Output ``out`` is a single flat buffer; per-item ``target_off`` is the
absolute offset where this item's tile starts. Multiple items writing
to the same target accumulate (-= for C, += for D). Caller does the
final serial scatter (race-free) AFTER the prange compute step writes
per-item tiles into a per-item scratch buffer.
"""
from cython.parallel cimport prange
cimport openmp

import numpy as np


def c_kernel_batched(
    int N,                              # total number of items
    int max_n_pno, int max_n_ct, int max_n_other,

    # Per-item metadata
    int[::1] n_pno_arr,                 # (N,)
    int[::1] n_ct_arr,
    int[::1] n_other_arr,
    long[::1] S_big_off,                # (N,)
    long[::1] ct_off,
    long[::1] S_mid_off,
    long[::1] J_bold_off,
    long[::1] t2_off,
    long[::1] S_outer_off,
    long[::1] tile_off,                 # (N+1,) per-item Cc tile offsets in tiles_flat

    # Static flat buffers
    double[::1] S_big_flat,
    double[::1] S_mid_flat,
    double[::1] J_bold_flat,
    double[::1] S_outer_flat,

    # Per-cycle dynamic flat buffers
    double[::1] ct_flat,
    double[::1] t2_flat,

    # Per-thread scratch
    double[:, ::1] STB_scratch,         # (num_threads, max_n_pno*max_n_ct)
    double[:, ::1] GAMMA_scratch,       # (num_threads, max_n_pno*max_n_other)
    double[:, ::1] GT_scratch,          # (num_threads, max_n_pno*max_n_other)

    # Per-item Cc tiles (flat) — caller scatters into out
    double[::1] tiles_flat,

    int num_threads,
):
    cdef Py_ssize_t n
    cdef int tid
    cdef int n_pno, n_ct, n_other
    cdef Py_ssize_t a, b, c, d, e, f
    cdef double s
    cdef double *S_big
    cdef double *ct
    cdef double *S_mid
    cdef double *J_bold
    cdef double *t2
    cdef double *S_outer
    cdef double *STB
    cdef double *GAMMA
    cdef double *GT
    cdef double *Cc

    for n in prange(N, schedule='dynamic',
                    nogil=True, num_threads=num_threads):
        tid = openmp.omp_get_thread_num()
        n_pno = n_pno_arr[n]
        n_ct = n_ct_arr[n]
        n_other = n_other_arr[n]

        S_big   = &S_big_flat[S_big_off[n]]
        ct      = &ct_flat[ct_off[n]]
        S_mid   = &S_mid_flat[S_mid_off[n]]
        J_bold  = &J_bold_flat[J_bold_off[n]]
        t2      = &t2_flat[t2_off[n]]
        S_outer = &S_outer_flat[S_outer_off[n]]
        STB     = &STB_scratch[tid, 0]
        GAMMA   = &GAMMA_scratch[tid, 0]
        GT      = &GT_scratch[tid, 0]
        Cc      = &tiles_flat[tile_off[n]]

        # STB[a, c] = sum_b S_big[a, b] * ct[b, c]
        for a in range(n_pno):
            for c in range(n_ct):
                s = 0.0
                for b in range(n_ct):
                    s = s + S_big[a * n_ct + b] * ct[b * n_ct + c]
                STB[a * n_ct + c] = s

        # gamma[a, d] = J_bold[a, d] + sum_c STB[a, c] * S_mid[c, d]
        for a in range(n_pno):
            for d in range(n_other):
                s = J_bold[a * n_other + d]
                for c in range(n_ct):
                    s = s + STB[a * n_ct + c] * S_mid[c * n_other + d]
                GAMMA[a * n_other + d] = s

        # GT[a, e] = sum_d gamma[a, d] * t2[e, d]
        for a in range(n_pno):
            for e in range(n_other):
                s = 0.0
                for d in range(n_other):
                    s = s + GAMMA[a * n_other + d] * t2[e * n_other + d]
                GT[a * n_other + e] = s

        # Cc[a, f] = sum_e GT[a, e] * S_outer[f, e]
        # (Writes per-item Cc tile into tiles_flat at tile_off[n].
        # Caller does serial scatter into out_flat — multiple items
        # may share a target slot, so the scatter must be race-free.)
        for a in range(n_pno):
            for f in range(n_pno):
                s = 0.0
                for e in range(n_other):
                    s = s + GT[a * n_other + e] * S_outer[f * n_other + e]
                Cc[a * n_pno + f] = s


def d_kernel_batched(
    int N,
    int max_n_pno, int max_n_A, int max_n_B,

    # Per-item metadata
    int[::1] n_pno_arr,
    int[::1] n_A_arr,
    int[::1] n_B_arr,
    long[::1] S_a_off,
    long[::1] u_off,
    long[::1] S_b_off,
    long[::1] S_c_off,
    long[::1] dt_off,
    long[::1] KJ_off,
    long[::1] tile_off,

    double[::1] S_a_flat,
    double[::1] S_b_flat,
    double[::1] S_c_flat,
    double[::1] KJ_flat,

    double[::1] u_flat,
    double[::1] dt_flat,

    double[:, ::1] SU_scratch,          # (num_threads, max_n_pno*max_n_A)
    double[:, ::1] UP_scratch,          # (num_threads, max_n_pno*max_n_B)
    double[:, ::1] SCD_scratch,         # (num_threads, max_n_pno*max_n_B)
    double[:, ::1] Bint_scratch,        # (num_threads, max_n_pno*max_n_A)

    double[::1] tiles_flat,             # per-item Dtile output

    int num_threads,
):
    cdef Py_ssize_t n
    cdef int tid
    cdef int n_pno, n_A, n_B
    cdef Py_ssize_t a, b, c, d, f
    cdef double s, sA, sB
    cdef double *S_a
    cdef double *u
    cdef double *S_b
    cdef double *S_c
    cdef double *dt
    cdef double *KJ
    cdef double *SU
    cdef double *UP
    cdef double *SCD
    cdef double *Bint
    cdef double *Dtile

    for n in prange(N, schedule='dynamic',
                    nogil=True, num_threads=num_threads):
        tid = openmp.omp_get_thread_num()
        n_pno = n_pno_arr[n]
        n_A = n_A_arr[n]
        n_B = n_B_arr[n]

        S_a   = &S_a_flat[S_a_off[n]]
        u     = &u_flat[u_off[n]]
        S_b   = &S_b_flat[S_b_off[n]]
        S_c   = &S_c_flat[S_c_off[n]]
        dt    = &dt_flat[dt_off[n]]
        KJ    = &KJ_flat[KJ_off[n]]
        SU    = &SU_scratch[tid, 0]
        UP    = &UP_scratch[tid, 0]
        SCD   = &SCD_scratch[tid, 0]
        Bint  = &Bint_scratch[tid, 0]
        Dtile = &tiles_flat[tile_off[n]]

        # SU = S_a @ u  → (n_pno, n_A)
        for a in range(n_pno):
            for d in range(n_A):
                s = 0.0
                for b in range(n_A):
                    s = s + S_a[a * n_A + b] * u[b * n_A + d]
                SU[a * n_A + d] = s

        # UP = SU @ S_b  → (n_pno, n_B)
        for a in range(n_pno):
            for c in range(n_B):
                s = 0.0
                for d in range(n_A):
                    s = s + SU[a * n_A + d] * S_b[d * n_B + c]
                UP[a * n_B + c] = s

        # SCD = S_c @ dt  → (n_pno, n_B)
        for a in range(n_pno):
            for c in range(n_B):
                s = 0.0
                for b in range(n_B):
                    s = s + S_c[a * n_B + b] * dt[b * n_B + c]
                SCD[a * n_B + c] = s

        # Bint = KJ @ u.T → (n_pno, n_A)  (Bint[a, d] = sum_b KJ[a, b] * u[d, b])
        for a in range(n_pno):
            for d in range(n_A):
                s = 0.0
                for b in range(n_A):
                    s = s + KJ[a * n_A + b] * u[d * n_A + b]
                Bint[a * n_A + d] = s

        # Dtile[a, f] = Part A + Part B
        # (Caller does serial scatter — see c_kernel_batched note above.)
        for a in range(n_pno):
            for f in range(n_pno):
                sA = 0.0
                for c in range(n_B):
                    sA = sA + SCD[a * n_B + c] * UP[f * n_B + c]
                sB = 0.0
                for d in range(n_A):
                    sB = sB + Bint[a * n_A + d] * S_a[f * n_A + d]
                Dtile[a * n_pno + f] = sA + sB
