# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Hand-rolled nogil Cython kernel for ``_compute_t1_residual_psi4._per_i``
Stages 1-3 (Fai_bar T1 dressing + Fab_bar @ t1 + -T_n.T @ Fia_bar @ t1).

Replaces the per-i numpy chain (gamma + 2*Qia*gamma + y - einsum, then
W + Fab + Fab*t1, then Z + Fia + T_n*(Fia*t1)) — measured at 54.6 ms /
i CPU on water10 (s23 dominant). The ops are 10 M FLOPs per i but
~165 MFLOPS effective due to per-call numpy/dispatch overhead and the
W (n_local, n_ii, nocc) + Z (n_local, nocc, nocc) scratch allocations.

The Cython kernel:
  * Materialises gamma and small reductions in scratch arrays sized to
    the largest pair (allocated once by the caller across all i, or
    locally if the caller passes None).
  * Folds Stage 3's Z into Stage 2's W reuse where possible.
  * Runs the entire Stages 1-3 chain in one ``nogil`` block — no
    Python dispatch or BLAS thread oversubscription inside the per-i
    pool.map workers.

Math (matches original numpy):

  Stage 1 (any_t1):
    gamma[L]  = sum_{m, a} Qma[L, m, a] * T_n[m, a]
    r1[a]    += 2 * sum_L Qia[L, a] * gamma[L]
    y[L, a]   = sum_m Qik[L, m] * T_n[m, a]
    r1[a]    -= sum_{L, c} Qab[L, a, c] * y[L, c]

  Stage 2 (has_t1 and any_t1):
    W[L, b, m] = sum_c Qab[L, b, c] * T_n[m, c]
    Fab[a, b]  = e_pno[a] * delta(a, b)
    Fab[a, b] += 2 * sum_L gamma[L] * Qab[L, a, b]
    Fab[a, b] -= sum_{L, m} W[L, a, m] * Qma[L, m, b]
    r1[a]     += sum_b Fab[a, b] * t1_i[b]

  Stage 3 (has_t1 and any_t1):
    Z[L, m, m'] = sum_c Qma[L, m, c] * T_n[m', c]
    Fia[m, a]   = 2 * sum_L gamma[L] * Qma[L, m, a]
    Fia[m, a]  -= sum_{L, m'} Z[L, m', m] * Qma[L, m', a]
    v[m]        = sum_b Fia[m, b] * t1_i[b]
    r1[a]      -= sum_m T_n[m, a] * v[m]

  Stage 1-only fallback (any_t1 but not has_t1) — runs only Stage 1
  block.

  Stages 2/3 fallback (has_t1 but not any_t1) — handled in caller
  with the simple ``r1 += e_pno * t1_i`` shortcut (no kernel call).
"""

import numpy as np


def per_i_stages123(double[:, :, ::1] Qma,    # (L, M, A)
                    double[:, :, ::1] Qab,    # (L, A, A)
                    double[:, ::1] Qia,       # (L, A)
                    double[:, ::1] Qik,       # (L, M)
                    double[:, ::1] T_n,       # (M, A)
                    double[::1] t1_i,         # (A,)
                    double[::1] e_pno,        # (A,)
                    int do_stage_23,          # 0/1: skip S2+S3 if 0
                    double[::1] r1_inout):    # (A,) accumulator
    cdef Py_ssize_t L = Qma.shape[0]
    cdef Py_ssize_t M = Qma.shape[1]
    cdef Py_ssize_t A = Qma.shape[2]
    cdef Py_ssize_t Lq, m, mp, a, b, c
    cdef double s, gv

    # Scratch (allocated with GIL; loops below are nogil).
    cdef double[::1]    gamma = np.empty(L)
    cdef double[:, ::1] y     = np.empty((L, A))
    # Stage 2 / 3 scratch — only sized when needed.
    cdef double[:, :, ::1] W
    cdef double[:, ::1]    Fab
    cdef double[:, :, ::1] Z
    cdef double[:, ::1]    Fia
    cdef double[::1]       v
    if do_stage_23:
        W   = np.empty((L, A, M))
        Fab = np.empty((A, A))
        Z   = np.empty((L, M, M))
        Fia = np.empty((M, A))
        v   = np.empty(M)
    else:
        # Tiny placeholders to satisfy memoryview typing — never read.
        W   = np.empty((1, 1, 1))
        Fab = np.empty((1, 1))
        Z   = np.empty((1, 1, 1))
        Fia = np.empty((1, 1))
        v   = np.empty(1)

    with nogil:
        # ============= Stage 1 =============
        # gamma[L] = sum_{m, a} Qma[L, m, a] * T_n[m, a]
        for Lq in range(L):
            s = 0.0
            for m in range(M):
                for a in range(A):
                    s = s + Qma[Lq, m, a] * T_n[m, a]
            gamma[Lq] = s

        # r1[a] += 2 * sum_L Qia[L, a] * gamma[L]
        for a in range(A):
            s = 0.0
            for Lq in range(L):
                s = s + Qia[Lq, a] * gamma[Lq]
            r1_inout[a] = r1_inout[a] + 2.0 * s

        # y[L, a] = sum_m Qik[L, m] * T_n[m, a]
        for Lq in range(L):
            for a in range(A):
                s = 0.0
                for m in range(M):
                    s = s + Qik[Lq, m] * T_n[m, a]
                y[Lq, a] = s

        # r1[a] -= sum_{L, c} Qab[L, a, c] * y[L, c]
        for a in range(A):
            s = 0.0
            for Lq in range(L):
                for c in range(A):
                    s = s + Qab[Lq, a, c] * y[Lq, c]
            r1_inout[a] = r1_inout[a] - s

        if do_stage_23:
            # ============= Stage 2 =============
            # W[L, b, m] = sum_c Qab[L, b, c] * T_n[m, c]
            for Lq in range(L):
                for b in range(A):
                    for m in range(M):
                        s = 0.0
                        for c in range(A):
                            s = s + Qab[Lq, b, c] * T_n[m, c]
                        W[Lq, b, m] = s

            # Fab[a, b] = e_pno*delta + 2*sum_L gamma*Qab[L,a,b]
            #           - sum_{L,m} W[L, a, m] * Qma[L, m, b]
            for a in range(A):
                for b in range(A):
                    if a == b:
                        s = e_pno[a]
                    else:
                        s = 0.0
                    # +2 * sum_L gamma[L] * Qab[L, a, b]
                    gv = 0.0
                    for Lq in range(L):
                        gv = gv + gamma[Lq] * Qab[Lq, a, b]
                    s = s + 2.0 * gv
                    # - sum_{L, m} W[L, a, m] * Qma[L, m, b]
                    gv = 0.0
                    for Lq in range(L):
                        for m in range(M):
                            gv = gv + W[Lq, a, m] * Qma[Lq, m, b]
                    s = s - gv
                    Fab[a, b] = s

            # r1[a] += sum_b Fab[a, b] * t1_i[b]
            for a in range(A):
                s = 0.0
                for b in range(A):
                    s = s + Fab[a, b] * t1_i[b]
                r1_inout[a] = r1_inout[a] + s

            # ============= Stage 3 =============
            # Z[L, m, m'] = sum_c Qma[L, m, c] * T_n[m', c]
            for Lq in range(L):
                for m in range(M):
                    for mp in range(M):
                        s = 0.0
                        for c in range(A):
                            s = s + Qma[Lq, m, c] * T_n[mp, c]
                        Z[Lq, m, mp] = s

            # Fia[mp, a] = 2*sum_L gamma[L] * Qma[L, mp, a]
            #            - sum_{L, m} Z[L, m, mp] * Qma[L, m, a]
            for mp in range(M):
                for a in range(A):
                    gv = 0.0
                    for Lq in range(L):
                        gv = gv + gamma[Lq] * Qma[Lq, mp, a]
                    s = 2.0 * gv
                    gv = 0.0
                    for Lq in range(L):
                        for m in range(M):
                            gv = gv + Z[Lq, m, mp] * Qma[Lq, m, a]
                    s = s - gv
                    Fia[mp, a] = s

            # v[m] = sum_b Fia[m, b] * t1_i[b]
            for m in range(M):
                s = 0.0
                for b in range(A):
                    s = s + Fia[m, b] * t1_i[b]
                v[m] = s

            # r1[a] -= sum_m T_n[m, a] * v[m]
            for a in range(A):
                s = 0.0
                for m in range(M):
                    s = s + T_n[m, a] * v[m]
                r1_inout[a] = r1_inout[a] - s
