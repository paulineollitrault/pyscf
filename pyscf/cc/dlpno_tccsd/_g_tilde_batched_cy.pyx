# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Batched Cython kernel for ``build_G_tilde._per_i``.

Path-(b) port matching Psi4 compute_G_tilde (ccsd.cc:1943): outer
prange over `(i, j)` slots in [0, naocc²), inner serial accumulation
over l. Eliminates the SERIAL G_tilde bottleneck (was 0.5 s/cycle).

Per-triple math (Psi4 form):
    U_lj_proj  = S_PNO(il, lj) @ Tt[lj] @ S_PNO(lj, il)
    contrib    = K_iajb[il].vector_dot(U_lj_proj.T)
            = sum_{a, b} K_il[a, b] * U_lj_proj[b, a]

Algebraic factoring (matches Psi4 parity insight from commit 9d6ccc7f8):
    K_proj   = S_il_lj.T @ K_il @ S_il_lj           (n_lj, n_lj) STATIC
    Tt_lj_oriented = 2*T2[c, d] - T2[d, c]   if l <= j
                  = 2*T2[d, c] - T2[c, d]   if l >  j
    contrib  = sum_{c, d} K_proj[d, c] * Tt_lj_oriented[c, d]

Folding the orientation into a per-triple `effective` static tensor:
    effective_case0 = 2*K_proj.T - K_proj    (when l <= j)
    effective_case1 = 2*K_proj   - K_proj.T  (when l >  j)
    contrib  = sum_{c, d} effective[c, d] * T2_canonical[c, d]
            = ddot(effective_flat, T2_canonical_flat)

For is_same triples (key_il == key_lj): K_il is symmetric (self-pair),
S_il_lj = I, so effective = K_il in both cases.

This single-ddot per triple replaces the per-iter triplet matmul
(S @ Tt @ S) + dot in our prior Python code.

Plan inputs (built once per CCSD run, see residual.py wrapper):
  effective_flat:    sum-of (n_lj²) per triple — STATIC
  effective_offset[t]
  T2_offset[pair_idx] — per canonical pair of n_lj²
  T2_pair_idx[t]     — which canonical T2 to read for triple t
  n_lj_arr[t]
  ij_triple_starts[slot+1] — slice of triples for each (i, j) slot
  ij_i_arr[slot], ij_j_arr[slot]

Per-iter:
  T2_flat — concatenated T2[key] for every canonical pair touched.
  G_addition (naocc, naocc) zero-init, kernel writes per (i, j) slot.
"""
from cython.parallel cimport prange
cimport openmp

import numpy as np


def g_tilde_batched(
    # Per-triple static
    long[::1] triple_eff_offset,       # offset into effective_flat
    long[::1] triple_T2_pair_idx,      # which canonical T2 buffer to read
    int[::1]  triple_n_lj,             # n_pno for the lj pair (== K_proj side)

    # Per-(i, j) outer slot bookkeeping (for prange)
    long[::1] ij_triple_starts,        # (n_ij_slots + 1,)
    int[::1]  ij_i_arr,                # (n_ij_slots,)
    int[::1]  ij_j_arr,                # (n_ij_slots,)

    # Static flat buffer: effective[d, c] precomputed per triple
    double[::1] effective_flat,

    # Per-iter flat T2 buffer (concatenated canonical T2[key])
    double[::1] T2_flat,
    long[::1]   T2_offsets,            # per canonical pair

    # Output: addition into G (caller initializes G to F̃kj)
    double[:, ::1] G_addition,         # (naocc, naocc)

    int num_threads,
):
    cdef Py_ssize_t n_ij_slots = ij_i_arr.shape[0]
    cdef Py_ssize_t slot
    cdef int i, j
    cdef Py_ssize_t t_start, t_end, t
    cdef Py_ssize_t n_lj, n_lj2, k
    cdef Py_ssize_t pair_idx
    cdef double *eff_ptr
    cdef double *T2_ptr
    cdef double sum_ij
    cdef double contribution

    for slot in prange(n_ij_slots, schedule='dynamic',
                       nogil=True, num_threads=num_threads):
        i = ij_i_arr[slot]
        j = ij_j_arr[slot]
        t_start = ij_triple_starts[slot]
        t_end = ij_triple_starts[slot + 1]

        sum_ij = 0.0
        for t in range(t_start, t_end):
            n_lj = triple_n_lj[t]
            n_lj2 = n_lj * n_lj
            eff_ptr = &effective_flat[triple_eff_offset[t]]
            pair_idx = triple_T2_pair_idx[t]
            T2_ptr = &T2_flat[T2_offsets[pair_idx]]

            # Hand-rolled ddot — small n_lj² (~625 typical), gcc -O3 vectorizes.
            contribution = 0.0
            for k in range(n_lj2):
                contribution = contribution + eff_ptr[k] * T2_ptr[k]
            sum_ij = sum_ij + contribution

        G_addition[i, j] = G_addition[i, j] + sum_ij
