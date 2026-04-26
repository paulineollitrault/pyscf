# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""nogil Cython kernel for one partner contribution inside
``compute_cc_integrals_sparse`` centerQ_loop.

Replaces the per-partner Python ops in [local_df.py:767-797]:

    sub = proj_ij[:, :, kj_u_in_Q]              # forced copy (fancy on last axis)
    raw_cross_kj[k][local_Q] = sub @ X_kj_slice
    raw_kv_kj[k][local_Q]    = qia_b[:, k_s, kj_u_in_Q] @ X_kj_slice

The fused kernel reads ``proj_ij`` / ``qia_b`` directly with index arrays,
avoiding the materialised ``sub`` copy, and runs all the inner sums in
one nogil block with single-precision dispatch overhead. Same code path
serves both kj_partners (writes to raw_cross_kj / raw_kv_kj) and
ki_partners (writes to raw_cross_ji / raw_kv_ki) — the caller just
passes the right output buffers.

Math:
    raw_cross[local_Q[Q], b, m] = sum_u proj_ij[Q, b, idx[u]] * X[u, m]
    raw_kv   [local_Q[Q], m]    = sum_u qia_b[Q, k_s, idx[u]] * X[u, m]
        (only computed when k_s >= 0)

Per partner FLOPs (water10 typical: nQp~10, npno~23, n_kj~25, npp~25):
    raw_cross: nQp * npno * n_kj * npp ≈ 144 k flops
    raw_kv:    nQp * n_kj * npp        ≈   6 k flops
"""

import numpy as np


def partner_apply(
    double[:, :, ::1] proj_ij,          # (nQp, npno, np_full)
    double[:, :, ::1] qia_b,            # (nQp, nl, np_full) — only k_s row read
    int k_s,                            # k LMO position in this centerQ (-1 if missing)
    long[::1]   local_Q,                # (nQp,) — rows of raw_cross/raw_kv to write
    long[::1]   idx,                    # (npp,) — np_full columns to gather
    double[:, ::1] X,                   # (npp, n_kj)
    double[:, :, ::1] raw_cross_out,    # (n_local_total, npno, n_kj)
    double[:, ::1]    raw_kv_out,       # (n_local_total, n_kj)
    int do_proj,                        # 1 if proj_ij contribution wanted
):
    cdef Py_ssize_t nQp  = local_Q.shape[0]
    cdef Py_ssize_t npp  = idx.shape[0]
    cdef Py_ssize_t n_kj = X.shape[1]
    cdef Py_ssize_t npno = proj_ij.shape[1]
    cdef Py_ssize_t Q, b, m, u
    cdef Py_ssize_t row, col
    cdef double s

    with nogil:
        if do_proj:
            for Q in range(nQp):
                row = local_Q[Q]
                for b in range(npno):
                    for m in range(n_kj):
                        s = 0.0
                        for u in range(npp):
                            col = idx[u]
                            s = s + proj_ij[Q, b, col] * X[u, m]
                        raw_cross_out[row, b, m] = s

        if k_s >= 0:
            for Q in range(nQp):
                row = local_Q[Q]
                for m in range(n_kj):
                    s = 0.0
                    for u in range(npp):
                        col = idx[u]
                        s = s + qia_b[Q, k_s, col] * X[u, m]
                    raw_kv_out[row, m] = s
