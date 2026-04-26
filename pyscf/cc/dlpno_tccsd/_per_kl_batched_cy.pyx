# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Single-call batched Cython kernel for ``_per_kl``.

Replaces the per-task pool.map dispatch (~1600 tasks / cycle on water10)
with one nogil prange call. All static metadata (which i's per task,
S-matrix offsets, has_A2 flags, etc.) is pre-built once on the Python
side; per-iter inputs (T2 amplitudes, T1 projections) are passed as
references to the underlying FlatTensorStore / FlatPairPairStore
``_buffer`` arrays — no per-task Python wrapper is needed.

Per-task math (matches the reference _per_kl in lccsd.py, which itself
is the Psi4 ccsd.cc:2055-2160 reference):

  T_n_kl = t1_cache[key_kl]            (M, n_kl)
  Tt_kl  = 2*T2[key_kl] - T2[key_kl].T (n_kl, n_kl) — with optional swap
  K_kilc = K_bar_kl + T_n_kl @ K_iajb_kl
  B_ia   = Tt_kl @ K_kilc.T            (n_kl, M)

  For each inner i (in pair_lmo_idx[key_kl]):
    n_pno_ii = pno_spaces[(i,i)]['C_pno'].shape[1]
    contrib[ti] = 0

    # B contribution
    if is_diag_kl_ii[ti]:
        contrib[ti, a] = -B_ia[a, i]
    elif has_S_ii_kl[ti]:
        contrib[ti, a] = -sum_c S_ii_kl[a, c] * B_ia[c, i]

    # A2 contribution
    if has_A2[ti]:
        Tt_ki  = 2*T2[key_ki] - T2[key_ki].T (with optional swap)
        T_n_l_ii = t1_cache[(i,i)][l]
        if is_diag_kl_ki[ti]:                  # key_kl == key_ki
            scalar = sum(K_iajb_kl * Tt_ki)
        else:
            X[c, d] = sum_a S_kl_ki[a, c] * K_iajb_kl[a, d]
            Z[c, b] = sum_d X[c, d] * S_ki_kl[b, d]
            scalar  = sum_{c, b} Tt_ki[c, b] * Z[c, b]
        contrib[ti, a] -= scalar * T_n_l_ii[a]
"""
from cython.parallel cimport prange
cimport openmp

import numpy as np


def per_kl_batched(
    # Per-task metadata
    int n_tasks,
    int M,                           # nocc; same for all tasks on this run
    int[::1] n_kl_arr,
    int[::1] t2_swap_kl,
    long[::1] K_iajb_kl_off,         # offset into K_iajb_buffer per task
    long[::1] K_bar_kl_off,          # offset into K_bar_kl_static_flat per task
    long[::1] t2_kl_canon_off,       # offset into t2_buffer (canonical pair)
    long[::1] T_n_kl_off,            # offset into t1_cache_buffer (canonical pair)
    long[::1] inner_off,             # (n_tasks+1,) cumulative count of inner i's

    # Per-(task, inner_i) metadata
    int[::1] i_arr,
    int[::1] n_pno_ii_arr,
    int[::1] is_diag_kl_ii,
    int[::1] has_S_ii_kl,
    long[::1] S_ii_kl_off,           # offset into S_pno_buffer
    int[::1] has_A2,
    int[::1] is_diag_kl_ki,
    int[::1] n_ki_arr,
    int[::1] t2_swap_ki,
    long[::1] t2_ki_canon_off,
    long[::1] S_kl_ki_off,
    long[::1] S_ki_kl_off,
    long[::1] T_n_l_ii_off,          # absolute offset into t1_cache_buffer (incl. l offset)
    long[::1] contrib_off,           # (n_total_inner+1,) offsets into contrib_flat

    # Static flat buffers (built once)
    double[::1] K_iajb_buffer,       # FlatTensorStore K_iajb._buffer
    double[::1] K_bar_kl_static,     # per-task K_bar_kl concatenated
    double[::1] S_pno_buffer,        # FlatPairPairStore S_pno_cache._buffer

    # Dynamic flat buffers (refreshed each cycle)
    double[::1] t2_buffer,           # FlatTensorStore t2_pno_all._buffer
    double[::1] t1_cache_buffer,     # FlatTensorStore _t1_cache._buffer

    # Per-thread scratch (preallocated for max sizes)
    double[:, ::1] Tt_kl_scratch,    # (num_threads, max_n_kl * max_n_kl)
    double[:, ::1] K_kilc_scratch,   # (num_threads, max_M * max_n_kl)
    double[:, ::1] B_ia_scratch,     # (num_threads, max_n_kl * max_M)
    double[:, ::1] Tt_ki_scratch,    # (num_threads, max_n_ki * max_n_ki)
    double[:, ::1] X_scratch,        # (num_threads, max_n_ki * max_n_kl)
    double[:, ::1] Z_scratch,        # (num_threads, max_n_ki * max_n_ki)

    # Output
    double[::1] contrib_flat,

    int num_threads,
):
    cdef Py_ssize_t t, ti, ti_start, ti_end
    cdef int tid
    cdef int n_kl, n_pno_ii, n_ki
    cdef int i_val
    cdef Py_ssize_t a, b, c, d, m
    cdef double s, scalar
    cdef double *T_n_kl
    cdef double *t2_kl
    cdef double *Tt_kl
    cdef double *K_iajb
    cdef double *K_bar_kl
    cdef double *K_kilc
    cdef double *B_ia
    cdef double *S_ii_kl
    cdef double *Tt_ki
    cdef double *t2_ki
    cdef double *S_kl_ki
    cdef double *S_ki_kl
    cdef double *T_n_l_ii
    cdef double *contrib
    cdef double *X
    cdef double *Z

    for t in prange(n_tasks, schedule='dynamic',
                    nogil=True, num_threads=num_threads):
        tid = openmp.omp_get_thread_num()
        n_kl = n_kl_arr[t]

        # ---- Build Tt_kl in scratch ----
        # t2_kl = t2_buffer[t2_kl_canon_off[t] : ...] reshape (n_kl, n_kl)
        # If swap: Tt_kl[a, b] = 2 * t2[b, a] - t2[a, b]
        # Else:    Tt_kl[a, b] = 2 * t2[a, b] - t2[b, a]
        t2_kl = &t2_buffer[t2_kl_canon_off[t]]
        Tt_kl = &Tt_kl_scratch[tid, 0]
        if t2_swap_kl[t]:
            for a in range(n_kl):
                for b in range(n_kl):
                    Tt_kl[a * n_kl + b] = (
                        2.0 * t2_kl[b * n_kl + a] - t2_kl[a * n_kl + b])
        else:
            for a in range(n_kl):
                for b in range(n_kl):
                    Tt_kl[a * n_kl + b] = (
                        2.0 * t2_kl[a * n_kl + b] - t2_kl[b * n_kl + a])

        # ---- Build K_kilc[m, c] = K_bar_kl[m, c] + sum_d T_n_kl[m, d] * K_iajb_kl[d, c] ----
        T_n_kl = &t1_cache_buffer[T_n_kl_off[t]]
        K_iajb = &K_iajb_buffer[K_iajb_kl_off[t]]
        K_bar_kl = &K_bar_kl_static[K_bar_kl_off[t]]
        K_kilc = &K_kilc_scratch[tid, 0]
        for m in range(M):
            for c in range(n_kl):
                s = K_bar_kl[m * n_kl + c]
                for d in range(n_kl):
                    s = s + T_n_kl[m * n_kl + d] * K_iajb[d * n_kl + c]
                K_kilc[m * n_kl + c] = s

        # ---- B_ia[c, m] = sum_d Tt_kl[c, d] * K_kilc[m, d] ----
        B_ia = &B_ia_scratch[tid, 0]
        for c in range(n_kl):
            for m in range(M):
                s = 0.0
                for d in range(n_kl):
                    s = s + Tt_kl[c * n_kl + d] * K_kilc[m * n_kl + d]
                B_ia[c * M + m] = s

        # ---- Inner per-i loop ----
        ti_start = inner_off[t]
        ti_end = inner_off[t + 1]
        for ti in range(ti_start, ti_end):
            i_val = i_arr[ti]
            n_pno_ii = n_pno_ii_arr[ti]
            contrib = &contrib_flat[contrib_off[ti]]

            # Zero-init contrib
            for a in range(n_pno_ii):
                contrib[a] = 0.0

            # ===== B contribution =====
            if is_diag_kl_ii[ti]:
                # n_pno_ii == n_kl in this branch
                for a in range(n_pno_ii):
                    contrib[a] = -B_ia[a * M + i_val]
            elif has_S_ii_kl[ti]:
                S_ii_kl = &S_pno_buffer[S_ii_kl_off[ti]]
                for a in range(n_pno_ii):
                    s = 0.0
                    for c in range(n_kl):
                        s = s + S_ii_kl[a * n_kl + c] * B_ia[c * M + i_val]
                    contrib[a] = -s

            # ===== A2 contribution =====
            if not has_A2[ti]:
                continue

            n_ki = n_ki_arr[ti]
            T_n_l_ii = &t1_cache_buffer[T_n_l_ii_off[ti]]

            # Build Tt_ki in scratch from t2_buffer
            t2_ki = &t2_buffer[t2_ki_canon_off[ti]]
            Tt_ki = &Tt_ki_scratch[tid, 0]
            if t2_swap_ki[ti]:
                for a in range(n_ki):
                    for b in range(n_ki):
                        Tt_ki[a * n_ki + b] = (
                            2.0 * t2_ki[b * n_ki + a]
                            - t2_ki[a * n_ki + b])
            else:
                for a in range(n_ki):
                    for b in range(n_ki):
                        Tt_ki[a * n_ki + b] = (
                            2.0 * t2_ki[a * n_ki + b]
                            - t2_ki[b * n_ki + a])

            if is_diag_kl_ki[ti]:
                # U_ki = Tt_ki, n_ki == n_kl
                # scalar = sum_{a, b} K_iajb_kl[a, b] * Tt_ki[a, b]
                scalar = 0.0
                for a in range(n_kl):
                    for b in range(n_kl):
                        scalar = scalar + (
                            K_iajb[a * n_kl + b] * Tt_ki[a * n_kl + b])
            else:
                S_kl_ki = &S_pno_buffer[S_kl_ki_off[ti]]   # (n_kl, n_ki)
                S_ki_kl = &S_pno_buffer[S_ki_kl_off[ti]]   # (n_ki, n_kl)
                X = &X_scratch[tid, 0]
                Z = &Z_scratch[tid, 0]

                # X[c, d] = sum_a S_kl_ki[a, c] * K_iajb_kl[a, d]
                # Shape (n_ki, n_kl)
                for c in range(n_ki):
                    for d in range(n_kl):
                        s = 0.0
                        for a in range(n_kl):
                            s = s + (
                                S_kl_ki[a * n_ki + c] * K_iajb[a * n_kl + d])
                        X[c * n_kl + d] = s

                # Z[c, b] = sum_d X[c, d] * S_ki_kl[b, d]
                # Shape (n_ki, n_ki)
                for c in range(n_ki):
                    for b in range(n_ki):
                        s = 0.0
                        for d in range(n_kl):
                            s = s + X[c * n_kl + d] * S_ki_kl[b * n_kl + d]
                        Z[c * n_ki + b] = s

                # scalar = sum_{c, b} Tt_ki[c, b] * Z[c, b]
                scalar = 0.0
                for c in range(n_ki):
                    for b in range(n_ki):
                        scalar = scalar + (
                            Tt_ki[c * n_ki + b] * Z[c * n_ki + b])

            # contrib[a] -= scalar * T_n_l_ii[a]
            for a in range(n_pno_ii):
                contrib[a] = contrib[a] - scalar * T_n_l_ii[a]
