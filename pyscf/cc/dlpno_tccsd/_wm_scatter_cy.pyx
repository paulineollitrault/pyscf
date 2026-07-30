# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""nogil gather/scatter kernels for the AUX-FIRST W/M replay in
compute_cc_integrals_sparse (local_df.py).

The replay's dgemms release the GIL (BLAS); the measured Python cost is the
fancy-index traffic around them:
  - W[:, :, cols] += contrib                (scatter-add, last axis indexed)
  - sub = page[np.ix_(apos, krows, pcols)]  (3-axis gather)
  - M[np.ix_(rows, :, cols)] += contrib.transpose(1,0,2)
Each is a tiny loop nest here, run without the GIL so all pool threads
proceed concurrently.
"""
import numpy as np
cimport numpy as cnp
cnp.import_array()


def scatter_add_lastaxis(double[:, :, ::1] W, double[:, :, ::1] contrib,
                         long[::1] cols):
    """W[:, :, cols[c]] += contrib[:, :, c]   (shapes (A,B,Ng), (A,B,Nc))."""
    cdef Py_ssize_t A = contrib.shape[0]
    cdef Py_ssize_t B = contrib.shape[1]
    cdef Py_ssize_t Nc = contrib.shape[2]
    cdef Py_ssize_t a, b, c
    with nogil:
        for a in range(A):
            for b in range(B):
                for c in range(Nc):
                    W[a, b, cols[c]] += contrib[a, b, c]


def gather_qia(double[:, :, ::1] page, long[::1] qrows, long[::1] krows,
               long[::1] pcols, double[:, :, ::1] out):
    """out[q, k, c] = page[qrows[q], krows[k], pcols[c]]."""
    cdef Py_ssize_t Q = qrows.shape[0]
    cdef Py_ssize_t K = krows.shape[0]
    cdef Py_ssize_t C = pcols.shape[0]
    cdef Py_ssize_t q, k, c
    cdef Py_ssize_t qr, kr
    with nogil:
        for q in range(Q):
            qr = qrows[q]
            for k in range(K):
                kr = krows[k]
                for c in range(C):
                    out[q, k, c] = page[qr, kr, pcols[c]]


def scatter_add_rows_lastaxis(double[:, :, ::1] M, double[:, :, ::1] contrib,
                              long[::1] rows, long[::1] cols):
    """M[rows[k], :, cols[c]] += contrib[:, k, c]   (contrib (npno, K, Nc))."""
    cdef Py_ssize_t P = contrib.shape[0]
    cdef Py_ssize_t K = contrib.shape[1]
    cdef Py_ssize_t Nc = contrib.shape[2]
    cdef Py_ssize_t p, k, c
    cdef Py_ssize_t rk
    with nogil:
        for k in range(K):
            rk = rows[k]
            for p in range(P):
                for c in range(Nc):
                    M[rk, p, cols[c]] += contrib[p, k, c]


from cython.parallel cimport prange


def norms_by_offsets(double[::1] flat, long[::1] off, int[::1] n_arr,
                     double[::1] out):
    """out[i] = Frobenius norm of the (n_arr[i] x n_arr[i]) tile at
    flat[off[i]:].  Used for per-item magnitude screening bounds."""
    cdef Py_ssize_t N = out.shape[0]
    cdef Py_ssize_t i, e, sz
    cdef long o
    cdef double acc
    for i in prange(N, nogil=True, schedule='static'):
        o = off[i]
        sz = <Py_ssize_t>n_arr[i] * <Py_ssize_t>n_arr[i]
        acc = 0.0
        for e in range(sz):
            acc = acc + flat[o + e] * flat[o + e]
        out[i] = acc ** 0.5


cimport openmp
from scipy.linalg.cython_blas cimport dgemm


def build_uk_master(double[::1] t2_flat, long[::1] t2_off,
                    double[::1] k_flat, long[::1] k_off,
                    int[::1] n_arr, unsigned char[::1] same_arr,
                    double[:, ::1] scratch, double[::1] uk_out,
                    int num_threads):
    """UK hoist for the BE kernel: per canonical pair p (n = n_arr[p])
        TT_minus = 2*T - T^T ;  UK = dgemm('T','N', K, TT_minus)
        if k != l:  TT_plus = 2*T^T - T ;  UK += dgemm('N','N', K, TT_plus)
    — the EXACT per-item computation from DLPNObe_kernel_gathered
    (dlpno_be.c), hoisted to once per pair per cycle (it depends only on
    the pair, not the (ij,kl) item; neither we nor Psi4 exploited this).
    UK is written at t2_off[p] (T2's canonical layout), so per-item UK
    offsets equal the existing per-item T offsets.
    scratch: (num_threads, 2*max_n*max_n)."""
    cdef Py_ssize_t P = n_arr.shape[0]
    cdef Py_ssize_t p, b, d
    cdef int n, tid
    cdef double *T
    cdef double *K
    cdef double *UK
    cdef double *ttm
    cdef double *ttp
    cdef char TA = b'T'
    cdef char NA = b'N'
    cdef double one = 1.0, zero = 0.0
    for p in prange(P, nogil=True, schedule='dynamic',
                    num_threads=num_threads):
        n = n_arr[p]
        if n == 0:
            continue
        tid = openmp.omp_get_thread_num()
        T = &t2_flat[t2_off[p]]
        K = &k_flat[k_off[p]]
        UK = &uk_out[t2_off[p]]
        ttm = &scratch[tid, 0]
        ttp = ttm + <Py_ssize_t>n * n
        for b in range(n):
            for d in range(n):
                ttm[b * n + d] = 2.0 * T[b * n + d] - T[d * n + b]
        dgemm(&TA, &NA, &n, &n, &n, &one, K, &n, ttm, &n, &zero, UK, &n)
        if same_arr[p] == 0:
            for b in range(n):
                for d in range(n):
                    ttp[b * n + d] = 2.0 * T[d * n + b] - T[b * n + d]
            dgemm(&NA, &NA, &n, &n, &n, &one, K, &n, ttp, &n, &one, UK, &n)


def norms_by_offsets_rect(double[::1] flat, long[::1] off,
                          int[::1] n_rows, int[::1] n_cols,
                          double[::1] out):
    """out[i] = Frobenius norm of the (n_rows[i] x n_cols[i]) tile at
    flat[off[i]:] — rectangular variant for per-item S-overlap norms."""
    cdef Py_ssize_t N = out.shape[0]
    cdef Py_ssize_t i, e, sz
    cdef long o
    cdef double acc
    for i in prange(N, nogil=True, schedule='static'):
        o = off[i]
        sz = <Py_ssize_t>n_rows[i] * <Py_ssize_t>n_cols[i]
        acc = 0.0
        for e in range(sz):
            acc = acc + flat[o + e] * flat[o + e]
        out[i] = acc ** 0.5


from libc.string cimport memcpy

def build_scat_master(double[::1] S_master, long[::1] S_off,
                      int[::1] item_nij, int[::1] item_nkl,
                      long[::1] scat_base, long[::1] item_k0,
                      long[::1] item_KT, double[::1] S_cat):
    """BE slot-cat: copy each item's S block (row-major (n_ij, n_kl) at
    S_master[S_off[n]]) into the slot's col-major (KT, n_ij) panel:
      S_cat[scat_base[n] + j*KT[n] + item_k0[n] + k] = Sn[j, k]
    Row j is contiguous in both layouts -> one memcpy per (item, row)."""
    cdef Py_ssize_t n, j, N = S_off.shape[0]
    cdef long base, o, kt, k0
    cdef int nj, nk
    for n in prange(N, nogil=True, schedule='static'):
        nj = item_nij[n]; nk = item_nkl[n]
        base = scat_base[n]; k0 = item_k0[n]
        kt = item_KT[n]; o = S_off[n]
        for j in range(nj):
            memcpy(&S_cat[base + j * kt + k0],
                   &S_master[o + j * nk], nk * sizeof(double))
