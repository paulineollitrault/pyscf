# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Cython kernel for ``_compute_foo_dressed_local._per_pair``.

Replaces the per-pair matmul + tensordot chain in ``lccsd.py:318`` with
one nogil function.  Correctness parity (bit-exact up to FP reordering)
with the reference NumPy path.

Intended use:

  out_q = np.zeros(nocc)
  out_m = np.zeros(nocc) if m != q else out_q  # m-path skipped if m==q
  foo_dressed_one(Qma, t2_mq_raw, m, q, out_q, out_m)

Shapes:
  Qma:    (n_local, nocc, n_pno)  — C-contiguous, from cc_ints[key]['Qma']
  t2:     (n_pno,  n_pno)         — C-contiguous, from t2_pno_all[key]
  out_q:  (nocc,)                 — C-contiguous output buffer
  out_m:  (nocc,)                 — C-contiguous output buffer
                                    (unused when m == q; pass a
                                    throwaway or the same array)

Math (matching the post-tensordot commit 111d811d7 Python reference):

  X_mq[a, L] = sum_b (2*t2[b, a] - t2[a, b]) * Qma[L, m, b]
  out_q[p]   = sum_{L, a} Qma[L, p, a] * X_mq[a, L]

  If m != q:
    X_qm[a, L] = sum_b (2*t2[a, b] - t2[b, a]) * Qma[L, q, b]
    out_m[p]   = sum_{L, a} Qma[L, p, a] * X_qm[a, L]
"""

import numpy as np


def foo_dressed_one(double[:, :, ::1] Qma,
                    double[:, ::1] t2,
                    int m,
                    int q,
                    double[::1] out_q,
                    double[::1] out_m):
    cdef Py_ssize_t n_local = Qma.shape[0]
    cdef Py_ssize_t nocc    = Qma.shape[1]
    cdef Py_ssize_t n_pno   = Qma.shape[2]
    cdef Py_ssize_t L, a, b, p
    cdef double s
    cdef bint need_m = (m != q)

    # Scratch for X[a, L].  Allocated with the GIL; loops below run nogil.
    cdef double[:, ::1] X_mq = np.empty((n_pno, n_local))
    cdef double[:, ::1] X_qm
    if need_m:
        X_qm = np.empty((n_pno, n_local))
    else:
        X_qm = X_mq  # alias, unused

    with nogil:
        # X_mq[a, L] = sum_b (2*t2[b, a] - t2[a, b]) * Qma[L, m, b]
        for a in range(n_pno):
            for L in range(n_local):
                s = 0.0
                for b in range(n_pno):
                    s = s + (2.0 * t2[b, a] - t2[a, b]) * Qma[L, m, b]
                X_mq[a, L] = s

        # out_q[p] = sum_{L, a} Qma[L, p, a] * X_mq[a, L]
        for p in range(nocc):
            s = 0.0
            for L in range(n_local):
                for a in range(n_pno):
                    s = s + Qma[L, p, a] * X_mq[a, L]
            out_q[p] = s

        if need_m:
            # X_qm[a, L] = sum_b (2*t2[a, b] - t2[b, a]) * Qma[L, q, b]
            for a in range(n_pno):
                for L in range(n_local):
                    s = 0.0
                    for b in range(n_pno):
                        s = s + (2.0 * t2[a, b] - t2[b, a]) * Qma[L, q, b]
                    X_qm[a, L] = s

            # out_m[p] = sum_{L, a} Qma[L, p, a] * X_qm[a, L]
            for p in range(nocc):
                s = 0.0
                for L in range(n_local):
                    for a in range(n_pno):
                        s = s + Qma[L, p, a] * X_qm[a, L]
                out_m[p] = s
