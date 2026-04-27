# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""Per-m vooo (K_ooov) subtract kernel for DLPNO-(T) (Step A).

Mirrors Psi4 `compute_lccsd_t0`'s per-m loop
(triples.cc L832-844): for each lmo ``l_ijk`` in the triple's occupied
domain, project the pair-PNO T2 of pair ``(outer_lmo, l)`` into the
triple-TNO basis (two dgemm calls) and apply six rank-1 subtracts into
``W`` — one per S_3 permutation.

Previously this was a batched numpy matmul that streamed the full
(m_dom, 3, n_tno, n_tno) ``t2_sc_full`` tile per triple.  The per-m form
keeps a single ``(n_tno, n_tno)`` projected tile in L1 across the six
rank-1 subtracts, which matches Psi4's bandwidth profile.

BLAS for the projection uses ``scipy.linalg.cython_blas.dgemm`` called
from a nogil region (same pattern as ``_cython_probe.blas_dgemm``).
The rank-1 subtracts are hand-rolled — they are ~n^3 reads/writes with
no matmul shape, so a tight C loop beats dispatching to BLAS at
n_tno ~= 25.
"""
from scipy.linalg.cython_blas cimport dgemm

import numpy as np


def vooo_m_subtract(
    double[:, :, :, ::1] K_ooov,           # (3, 3, n_tno, m_dom)
    double[::1] U_flat,                    # concatenated U matrices
    long[::1] U_offsets,                   # (3*m_dom + 1,) cumulative offsets
    long[::1] n_pno_arr,                   # (3*m_dom,) per-(r, l) n_pno; 0 = skip
    double[::1] T2_flat,                   # concatenated pair-PNO T2 matrices
    long[::1] T2_offsets,                  # (3*m_dom + 1,) cumulative offsets
    signed char[::1] transpose_flags,      # (3*m_dom,) 1 if project needs T.T
    double[:, :, ::1] W,                   # (n_tno, n_tno, n_tno) in-place
    int m_dom,
    int n_tno,
    int n_pno_max,
):
    """Accumulate ``W -= Σ_pidx trans[pidx](sum_m K_ooov[ip,iq][:, m] ⊗ T_il[ir, m])``.

    For each ``l_ijk`` in ``range(m_dom)`` and each ``r`` in ``(0, 1, 2)``:

        idx        = r * m_dom + l_ijk
        U_rl       = U_flat[U_offsets[idx]: U_offsets[idx+1]]
                     reshape (n_pno_arr[idx], n_tno)
        T2_rl      = T2_flat[T2_offsets[idx]: T2_offsets[idx+1]]
                     reshape (n_pno_arr[idx], n_pno_arr[idx])
        T_il[r]    = U_rl.T @ T2_rl @ U_rl     # (n_tno, n_tno)
        if transpose_flags[idx]:
            T_il[r] = T_il[r].T

    After projecting all three ``r`` slots, the six rank-1 subtracts
    apply (pidx: (ip, iq, ir) and virtual-index permutation Π):

        pidx=0 (ip=0, iq=1, ir=2): W[a,b,c] -= K[0,1,a,m] * T[2, b, c]
        pidx=1 (ip=0, iq=2, ir=1): W[a,b,c] -= K[0,2,a,m] * T[1, c, b]
        pidx=2 (ip=1, iq=0, ir=2): W[a,b,c] -= K[1,0,b,m] * T[2, a, c]
        pidx=3 (ip=1, iq=2, ir=0): W[a,b,c] -= K[1,2,b,m] * T[0, c, a]
        pidx=4 (ip=2, iq=0, ir=1): W[a,b,c] -= K[2,0,c,m] * T[1, a, b]
        pidx=5 (ip=2, iq=1, ir=0): W[a,b,c] -= K[2,1,c,m] * T[0, b, a]

    Called from a Python worker thread inside the outer triples pool;
    stays sequential (no prange) to avoid nested OMP with the pool.
    """
    cdef Py_ssize_t l_ijk, r, a, b, c
    cdef Py_ssize_t n_pno
    cdef Py_ssize_t flat_idx
    cdef Py_ssize_t u_off, t2_off
    cdef double K01a, K02a, K10b, K12b, K20c, K21c
    cdef double T2ab, T0ba
    cdef double acc, tmp

    cdef char N_ = b'N'
    cdef char T_ = b'T'
    cdef double one = 1.0
    cdef double zero = 0.0
    cdef int int_n_tno = n_tno
    cdef int int_n_pno
    cdef int ld_U, ld_T2, ld_T2U, ld_T_il

    if n_tno == 0 or m_dom == 0:
        return

    # Scratch for per-m projections.
    cdef double[:, :, ::1] T_il = np.zeros((3, n_tno, n_tno))
    # T2U stride is n_tno, rows up to n_pno_max.
    cdef Py_ssize_t _scratch_rows = n_pno_max if n_pno_max > 0 else 1
    cdef double[:, ::1] T2U = np.empty((_scratch_rows, n_tno))

    with nogil:
        for l_ijk in range(m_dom):
            # --- Project T_il[r] = U.T @ T2_pair @ U for r = 0, 1, 2 ---
            for r in range(3):
                flat_idx = r * m_dom + l_ijk
                n_pno = n_pno_arr[flat_idx]
                if n_pno == 0:
                    for a in range(n_tno):
                        for b in range(n_tno):
                            T_il[r, a, b] = 0.0
                    continue
                int_n_pno = <int>n_pno
                u_off = U_offsets[flat_idx]
                t2_off = T2_offsets[flat_idx]
                ld_U = int_n_tno      # U is row-major (n_pno, n_tno)
                ld_T2 = int_n_pno     # T2 is row-major (n_pno, n_pno)
                ld_T2U = int_n_tno    # T2U row-major (n_pno, n_tno)
                ld_T_il = int_n_tno   # T_il row-major (n_tno, n_tno)

                # Step 1: T2U = T2 @ U, shape (n_pno, n_tno).
                # A = T2 (n_pno, n_pno) C-order, B = U (n_pno, n_tno) C-order,
                # C = T2U (n_pno, n_tno) C-order. _cython_probe.blas_dgemm recipe.
                dgemm(&N_, &N_,
                      &int_n_tno, &int_n_pno, &int_n_pno,
                      &one,
                      &U_flat[u_off], &ld_U,
                      &T2_flat[t2_off], &ld_T2,
                      &zero,
                      &T2U[0, 0], &ld_T2U)

                # Step 2: T_il[r] = U.T @ T2U, shape (n_tno, n_tno).
                # For C = A.T @ B with A = U (K=n_pno, M=n_tno) and
                # B = T2U (K=n_pno, N=n_tno) both row-major, the
                # transposed Fortran call is dgemm('N', 'T', N, M, K, ...)
                # with dgemm's A-slot = B (T2U) and dgemm's B-slot = A (U).
                dgemm(&N_, &T_,
                      &int_n_tno, &int_n_tno, &int_n_pno,
                      &one,
                      &T2U[0, 0], &ld_T2U,
                      &U_flat[u_off], &ld_U,
                      &zero,
                      &T_il[r, 0, 0], &ld_T_il)

                # Swap axes if pair sort inverts the (outer, l) order.
                if transpose_flags[flat_idx] != 0:
                    for a in range(n_tno):
                        for b in range(a + 1, n_tno):
                            tmp = T_il[r, a, b]
                            T_il[r, a, b] = T_il[r, b, a]
                            T_il[r, b, a] = tmp

            # --- Six rank-1 subtracts fused into one abc triple loop. ---
            # T_il[1, a, b] and T_il[0, b, a] are c-invariant; hoist them.
            for a in range(n_tno):
                K01a = K_ooov[0, 1, a, l_ijk]
                K02a = K_ooov[0, 2, a, l_ijk]
                for b in range(n_tno):
                    K10b = K_ooov[1, 0, b, l_ijk]
                    K12b = K_ooov[1, 2, b, l_ijk]
                    T1ab = T_il[1, a, b]
                    T0ba = T_il[0, b, a]
                    for c in range(n_tno):
                        K20c = K_ooov[2, 0, c, l_ijk]
                        K21c = K_ooov[2, 1, c, l_ijk]
                        acc = (K01a * T_il[2, b, c]
                               + K02a * T_il[1, c, b]
                               + K10b * T_il[2, a, c]
                               + K12b * T_il[0, c, a]
                               + K20c * T1ab
                               + K21c * T0ba)
                        W[a, b, c] -= acc
