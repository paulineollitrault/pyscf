# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
"""Cython build probe for the DLPNO restructure.

Three smoke tests that together exercise everything Phase 4's real
kernels will need:

- ``hello`` : plain Cython def — proves the build system produces a
  loadable ``.so``.
- ``prange_sum`` : ``cython.parallel.prange`` reduction with
  ``nogil`` — proves OpenMP linkage.
- ``blas_dgemm`` : direct ``scipy.linalg.cython_blas.dgemm`` call from
  a typed memoryview — proves we can call BLAS from a nogil region
  without going through numpy.

When Phase 4 starts, the real kernel inherits the same build recipe
(same ``setup.py``, same compile flags); it just adds more functions
next to these.
"""
from scipy.linalg.cython_blas cimport dgemm
from cython.parallel cimport prange


def hello():
    """Trivial ``def`` exposing a Python-level string.

    Returns
    -------
    str
    """
    return "cython_probe: hello from the .so"


def prange_sum(const double[::1] a, int num_threads):
    """Parallel reduction over ``a`` using OpenMP via ``prange``.

    Parameters
    ----------
    a : 1-D contiguous double array
    num_threads : int
        OpenMP thread count (``schedule='static'``).

    Returns
    -------
    float
        ``sum(a)``, bit-compatible with ``a.sum()`` up to FP reordering.
    """
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i
    cdef double total = 0.0
    if n == 0:
        return 0.0
    with nogil:
        for i in prange(n, schedule='static', num_threads=num_threads):
            total += a[i]
    return total


def blas_dgemm(double[:, ::1] A,
               double[:, ::1] B,
               double[:, ::1] C):
    """C = A @ B via direct ``scipy.linalg.cython_blas.dgemm``.

    A, B, C must all be C-contiguous float64 2-D arrays with matching
    inner dims (A: m×k, B: k×n, C: m×n).

    BLAS is Fortran-ordered: to compute ``C = A @ B`` with C-order
    matrices, we call ``dgemm`` on the transposed problem:

        C_F = B_F^T @ A_F^T  (seen as Fortran arrays)

    which evaluates to ``A_C @ B_C`` when viewed in C layout.
    """
    cdef int m = A.shape[0]
    cdef int k = A.shape[1]
    cdef int n = B.shape[1]
    if B.shape[0] != k or C.shape[0] != m or C.shape[1] != n:
        raise ValueError(
            "blas_dgemm: inconsistent shapes "
            f"A={A.shape} B={B.shape} C={C.shape}"
        )
    cdef int lda = k
    cdef int ldb = n
    cdef int ldc = n
    cdef double alpha = 1.0
    cdef double beta = 0.0
    cdef char transN = b'N'
    with nogil:
        dgemm(&transN, &transN, &n, &m, &k,
              &alpha,
              &B[0, 0], &ldb,
              &A[0, 0], &lda,
              &beta,
              &C[0, 0], &ldc)
