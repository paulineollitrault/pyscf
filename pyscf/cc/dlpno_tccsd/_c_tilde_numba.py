"""Numba kernels for compute_C_tilde_batched Terms 3 and 4.

Each kernel consumes pre-stacked bucket tensors (constant across cycles,
built once by the plan-cache) plus per-cycle T1/T2 gathers, and
accumulates into a pre-zeroed per-n_ki flat output buffer.

Two-stage structure:
  1. parallel prange over items; small matmuls via ``np.dot`` (dispatched
     to BLAS from inside @njit — faster than hand-rolled loops at n=25).
  2. sequential scatter-add into the flat output (race-free).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def t3_kernel(K, S, t1i, T1l, idx, out):
    """Accumulate Term 3 contributions into ``out``.

    For each item n:
        Kt1    = K[n].T @ t1i[n]            (n_kl,)
        Kt1_ki = S[n]   @ Kt1               (n_ki,)
        out[idx[n], a, c] -= T1l[n, a] * Kt1_ki[c]

    K (N, n_kl, n_kl), S (N, n_ki, n_kl), t1i (N, n_kl), T1l (N, n_ki),
    idx (N,), out (n_slots, n_ki, n_ki).
    """
    N = K.shape[0]
    n_ki = S.shape[1]

    contrib = np.empty((N, n_ki, n_ki))
    for n in prange(N):
        # Kt1 = K[n].T @ t1i[n]  → (n_kl,).  Transpose via dot on np.ascontig.
        Kt1 = np.dot(K[n].T, t1i[n])
        # Kt1_ki = S[n] @ Kt1  → (n_ki,)
        Kt1_ki = np.dot(S[n], Kt1)
        # contrib[n] = -outer(T1l[n], Kt1_ki)
        for a in range(n_ki):
            v = -T1l[n, a]
            for c in range(n_ki):
                contrib[n, a, c] = v * Kt1_ki[c]

    # Sequential scatter-add.
    for n in range(N):
        t = idx[n]
        for a in range(n_ki):
            for c in range(n_ki):
                out[t, a, c] += contrib[n, a, c]


@njit(parallel=True, cache=True, fastmath=True, boundscheck=False)
def t4_kernel(S_ki_li, t2, S_li_kl, K, S_kl_ki, idx, out):
    """Accumulate Term 4 contributions into ``out``.

    For each item n:
        C[a, c] = -0.5 * (S_ki_li[n] @ t2[n] @ S_li_kl[n] @ K[n] @ S_kl_ki[n])
        out[idx[n]] += C

    S_ki_li (N, n_ki, n_li), t2 (N, n_li, n_li), S_li_kl (N, n_li, n_kl),
    K (N, n_kl, n_kl), S_kl_ki (N, n_kl, n_ki), idx (N,),
    out (n_slots, n_ki, n_ki).
    """
    N = S_ki_li.shape[0]
    n_ki = S_ki_li.shape[1]

    contrib = np.empty((N, n_ki, n_ki))
    for n in prange(N):
        tmp = np.dot(S_ki_li[n], t2[n])     # (n_ki, n_li)
        tmp = np.dot(tmp,        S_li_kl[n])  # (n_ki, n_kl)
        tmp = np.dot(tmp,        K[n])        # (n_ki, n_kl)
        C   = np.dot(tmp,        S_kl_ki[n])  # (n_ki, n_ki)
        for i in range(n_ki):
            for j in range(n_ki):
                contrib[n, i, j] = -0.5 * C[i, j]

    for n in range(N):
        t = idx[n]
        for i in range(n_ki):
            for j in range(n_ki):
                out[t, i, j] += contrib[n, i, j]
