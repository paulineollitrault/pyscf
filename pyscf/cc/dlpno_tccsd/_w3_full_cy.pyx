# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Combined (T) per-triple kernel for DLPNO-(T) (Step B).

Wraps the hot phases of _w3_intermediate into a single nogil Cython
function, eliminating ~40 numpy call dispatches per triple (×1600 triples
×3 loops ≈ hundreds of ms on water8).  BLAS throughput is preserved via
``scipy.linalg.cython_blas.dgemm`` for the matmul ops — hand-rolled
matmuls lose to BLAS at n_tno ≈ 25.

Phases done inside the kernel:

* Phase 1b: six ``base[pidx] = K_ab[ip] @ t2_T[ir, iq]`` dgemms.
* Phase 1c: fused permuted accumulate ``W = Σ_pidx base[pidx][Π(pidx)]``.
* Phase 2:  vooo per-m subtract (inline Step A logic — eliminates the
            separate ``_vooo_cy`` Python call).
* Phase 3:  ``T = -W / D`` elementwise.
* Phase 4:  ``V = W + T1 disconnected`` with three (n,n) K-integrals.
* Phase 5:  antisymmetrized energy sum — six permuted sums of ``V*T``
            in one abc triple loop.

Inputs precomputed in numpy on the Python side:

* K_ab_cache ``(3, n, n, n)`` — three tensordots of ``ovL_sc @ vvL_sc``.
* t2_T_all   ``(3, 3, n, n)`` — ``t2_sc.transpose(0, 1, 3, 2).copy()``.
* K_jk, K_ik, K_ij ``(n, n)`` — ``ovL_sc[p] @ ovL_sc[q].T`` for pair (j,k),
  (i,k), (i,j).  Only used if ``has_t1 != 0``.

The t1 arg is always required (zero-filled caller-side when not used) so
the kernel signature stays simple.
"""
from scipy.linalg.cython_blas cimport dgemm

import numpy as np


def w3_full_kernel(
    double[:, :, :, ::1] K_ab_cache,       # (3, n, n, n) = (ip, a, b, f)
    double[:, :, :, ::1] t2_T_all,         # (3, 3, n, n) = (ir, iq, c, f)
    double[:, ::1] K_jk,                   # (n, n)
    double[:, ::1] K_ik,                   # (n, n)
    double[:, ::1] K_ij,                   # (n, n)
    double[:, :, :, ::1] K_ooov,           # (3, 3, n, m_dom)
    double[::1] U_flat,
    long[::1] U_offsets,
    long[::1] n_pno_arr,
    double[::1] T2_flat,
    long[::1] T2_offsets,
    signed char[::1] transpose_flags,
    double[::1] eps_occ,                   # (3,)
    double[::1] eps_vir,                   # (n,)
    double[:, ::1] t1_sc,                  # (3, n) — zeros if not used
    int has_t1,
    int occ_denom,
    int n, int m_dom, int n_pno_max,
):
    """All-in-one (T) per-triple kernel. Returns ``et / occ_denom`` as float."""
    cdef:
        Py_ssize_t a, b, c, l_ijk, r
        Py_ssize_t n_pno, flat_idx, u_off, t2_off
        double acc, tmp
        double W_abc, T_abc, V_abc, D_abc, K_val
        double S0, S1, S2, S3, S4, S5
        double K01a, K02a, K10b, K12b, K20c, K21c
        double T1ab_vooo, T0ba_vooo
        double t1_0_a, t1_1_b, t1_2_c
        double D_occ, eps_a, eps_ab
        double et
        char N_ = b'N', T_ = b'T'
        double one = 1.0, zero = 0.0
        int int_n = n
        int int_nn = n * n
        int int_n_pno, ld_U, ld_T2, ld_T2U, ld_T_il

    if n == 0 or occ_denom == 0:
        return 0.0

    # --- Scratch (outside nogil so np.empty can acquire GIL) ---
    cdef double[:, :, ::1] W = np.empty((n, n, n))
    cdef double[:, :, ::1] V = np.empty((n, n, n))
    cdef double[:, :, ::1] T_ten = np.empty((n, n, n))
    cdef double[:, :, ::1] base_buf = np.empty((n, n, n))
    cdef double[:, :, ::1] T_il = np.zeros((3, n, n))
    cdef Py_ssize_t _scratch_rows = n_pno_max if n_pno_max > 0 else 1
    cdef double[:, ::1] T2U = np.empty((_scratch_rows, n))

    D_occ = eps_occ[0] + eps_occ[1] + eps_occ[2]

    with nogil:
        # ================================================================
        # Phase 1: W = Σ_pidx trans[pidx](K_ab[ip] @ t2_T[ir, iq])
        # ================================================================
        # Zero W first (Phase 1 accumulates into it).
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    W[a, b, c] = 0.0

        # pidx=0: ip=0, iq=1, ir=2, trans = identity
        # base[a, b, c] = Σ_f K_ab[0, a, b, f] * t2_T[2, 1, c, f]
        # dgemm: C_C (n*n, n) = A_C (n*n, n) @ B_C (n, n)  with B_C = t2_T_all[2, 1].
        dgemm(&N_, &N_,
              &int_n, &int_nn, &int_n,
              &one,
              &t2_T_all[2, 1, 0, 0], &int_n,
              &K_ab_cache[0, 0, 0, 0], &int_n,
              &zero,
              &base_buf[0, 0, 0], &int_n)
        # W[a, b, c] += base[a, b, c]
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    W[a, b, c] += base_buf[a, b, c]

        # pidx=1: ip=0, iq=2, ir=1, trans = (0, 2, 1), W[a,b,c] += base[a, c, b]
        dgemm(&N_, &N_,
              &int_n, &int_nn, &int_n,
              &one,
              &t2_T_all[1, 2, 0, 0], &int_n,
              &K_ab_cache[0, 0, 0, 0], &int_n,
              &zero,
              &base_buf[0, 0, 0], &int_n)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    W[a, b, c] += base_buf[a, c, b]

        # pidx=2: ip=1, iq=0, ir=2, trans = (1, 0, 2), W[a,b,c] += base[b, a, c]
        dgemm(&N_, &N_,
              &int_n, &int_nn, &int_n,
              &one,
              &t2_T_all[2, 0, 0, 0], &int_n,
              &K_ab_cache[1, 0, 0, 0], &int_n,
              &zero,
              &base_buf[0, 0, 0], &int_n)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    W[a, b, c] += base_buf[b, a, c]

        # pidx=3: ip=1, iq=2, ir=0, trans = (2, 0, 1), W[a,b,c] += base[b, c, a]
        dgemm(&N_, &N_,
              &int_n, &int_nn, &int_n,
              &one,
              &t2_T_all[0, 2, 0, 0], &int_n,
              &K_ab_cache[1, 0, 0, 0], &int_n,
              &zero,
              &base_buf[0, 0, 0], &int_n)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    W[a, b, c] += base_buf[b, c, a]

        # pidx=4: ip=2, iq=0, ir=1, trans = (1, 2, 0), W[a,b,c] += base[c, a, b]
        dgemm(&N_, &N_,
              &int_n, &int_nn, &int_n,
              &one,
              &t2_T_all[1, 0, 0, 0], &int_n,
              &K_ab_cache[2, 0, 0, 0], &int_n,
              &zero,
              &base_buf[0, 0, 0], &int_n)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    W[a, b, c] += base_buf[c, a, b]

        # pidx=5: ip=2, iq=1, ir=0, trans = (2, 1, 0), W[a,b,c] += base[c, b, a]
        dgemm(&N_, &N_,
              &int_n, &int_nn, &int_n,
              &one,
              &t2_T_all[0, 1, 0, 0], &int_n,
              &K_ab_cache[2, 0, 0, 0], &int_n,
              &zero,
              &base_buf[0, 0, 0], &int_n)
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    W[a, b, c] += base_buf[c, b, a]

        # ================================================================
        # Phase 2: vooo per-m subtract (Step A logic inlined)
        # ================================================================
        for l_ijk in range(m_dom):
            for r in range(3):
                flat_idx = r * m_dom + l_ijk
                n_pno = n_pno_arr[flat_idx]
                if n_pno == 0:
                    for a in range(n):
                        for b in range(n):
                            T_il[r, a, b] = 0.0
                    continue
                int_n_pno = <int>n_pno
                u_off = U_offsets[flat_idx]
                t2_off = T2_offsets[flat_idx]
                ld_U = int_n
                ld_T2 = int_n_pno
                ld_T2U = int_n
                ld_T_il = int_n
                # T2U = T2 @ U : (n_pno, n_pno) @ (n_pno, n_tno) -> (n_pno, n_tno)
                dgemm(&N_, &N_,
                      &int_n, &int_n_pno, &int_n_pno,
                      &one,
                      &U_flat[u_off], &ld_U,
                      &T2_flat[t2_off], &ld_T2,
                      &zero,
                      &T2U[0, 0], &ld_T2U)
                # T_il[r] = U.T @ T2U : (n_tno, n_tno)
                dgemm(&N_, &T_,
                      &int_n, &int_n, &int_n_pno,
                      &one,
                      &T2U[0, 0], &ld_T2U,
                      &U_flat[u_off], &ld_U,
                      &zero,
                      &T_il[r, 0, 0], &ld_T_il)
                if transpose_flags[flat_idx] != 0:
                    for a in range(n):
                        for b in range(a + 1, n):
                            tmp = T_il[r, a, b]
                            T_il[r, a, b] = T_il[r, b, a]
                            T_il[r, b, a] = tmp

            # Six rank-1 subtracts — indexing derived from Step A.
            for a in range(n):
                K01a = K_ooov[0, 1, a, l_ijk]
                K02a = K_ooov[0, 2, a, l_ijk]
                for b in range(n):
                    K10b = K_ooov[1, 0, b, l_ijk]
                    K12b = K_ooov[1, 2, b, l_ijk]
                    T1ab_vooo = T_il[1, a, b]
                    T0ba_vooo = T_il[0, b, a]
                    for c in range(n):
                        K20c = K_ooov[2, 0, c, l_ijk]
                        K21c = K_ooov[2, 1, c, l_ijk]
                        acc = (K01a * T_il[2, b, c]
                               + K02a * T_il[1, c, b]
                               + K10b * T_il[2, a, c]
                               + K12b * T_il[0, c, a]
                               + K20c * T1ab_vooo
                               + K21c * T0ba_vooo)
                        W[a, b, c] -= acc

        # ================================================================
        # Phase 3+4: T = -W / D ; V = W + T1 disconnected.
        # ================================================================
        for a in range(n):
            eps_a = eps_vir[a]
            if has_t1:
                t1_0_a = t1_sc[0, a]
            else:
                t1_0_a = 0.0
            for b in range(n):
                eps_ab = eps_a + eps_vir[b]
                if has_t1:
                    t1_1_b = t1_sc[1, b]
                else:
                    t1_1_b = 0.0
                for c in range(n):
                    D_abc = eps_ab + eps_vir[c] - D_occ
                    W_abc = W[a, b, c]
                    T_ten[a, b, c] = -W_abc / D_abc
                    if has_t1:
                        t1_2_c = t1_sc[2, c]
                        V[a, b, c] = W_abc + (t1_0_a * K_jk[b, c]
                                              + t1_1_b * K_ik[a, c]
                                              + t1_2_c * K_ij[a, b])
                    else:
                        V[a, b, c] = W_abc

        # ================================================================
        # Phase 5: Antisymmetrized energy — six permuted sums, one triple loop.
        #   et = 8*Σ V⋅T  - 4*Σ V(210)⋅T  - 4*Σ V(021)⋅T  - 4*Σ V(102)⋅T
        #                 + 2*Σ V(120)⋅T  + 2*Σ V(201)⋅T
        # where V(perm)[a,b,c] corresponds to V at the permuted indices:
        #   V(210)[a,b,c] = V[c, b, a]
        #   V(021)[a,b,c] = V[a, c, b]
        #   V(102)[a,b,c] = V[b, a, c]
        #   V(120)[a,b,c] = V[c, a, b]  (perm (1,2,0): y[a,b,c] = x[c,a,b])
        #   V(201)[a,b,c] = V[b, c, a]  (perm (2,0,1): y[a,b,c] = x[b,c,a])
        # ================================================================
        S0 = 0.0; S1 = 0.0; S2 = 0.0
        S3 = 0.0; S4 = 0.0; S5 = 0.0
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    T_abc = T_ten[a, b, c]
                    S0 += V[a, b, c] * T_abc
                    S1 += V[c, b, a] * T_abc
                    S2 += V[a, c, b] * T_abc
                    S3 += V[b, a, c] * T_abc
                    S4 += V[c, a, b] * T_abc
                    S5 += V[b, c, a] * T_abc

        et = 8.0 * S0 - 4.0 * S1 - 4.0 * S2 - 4.0 * S3 + 2.0 * S4 + 2.0 * S5

    return et / occ_denom
