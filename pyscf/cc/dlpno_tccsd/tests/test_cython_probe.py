"""Smoke tests for the Phase 3 Cython build.

Run::

    /environments/miniconda3/envs/tmc/bin/python -m pytest \
        pyscf/cc/dlpno_tccsd/tests/test_cython_probe.py -v

All three tests exercise a distinct part of the build/runtime stack
that the Phase 4 real kernels depend on:

  - ``test_hello_loads``    : module imports + Python ``def`` works.
  - ``test_prange_sum``     : ``cython.parallel.prange`` + OpenMP linkage.
  - ``test_blas_dgemm``     : ``scipy.linalg.cython_blas`` inside nogil.

If the Phase 3 build is broken (wrong compiler flags, missing OpenMP,
scipy cython_blas not available), exactly one of these fails — that's
the signal to fix ``setup.py`` before moving on.
"""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def probe():
    """Import the compiled .so once."""
    from pyscf.cc.dlpno_tccsd import _cython_probe
    return _cython_probe


def test_hello_loads(probe):
    assert probe.hello() == "cython_probe: hello from the .so"


def test_prange_sum_matches_numpy(probe):
    """Parallel reduction should match ``np.sum`` to tight FP tol."""
    rng = np.random.default_rng(2026)
    a = rng.standard_normal(100_000)
    expected = float(a.sum())
    for n_threads in (1, 2, 8):
        got = probe.prange_sum(a, n_threads)
        assert abs(got - expected) < 1e-9, (
            f"threads={n_threads}: got={got} expected={expected}"
        )


def test_prange_sum_empty_array(probe):
    assert probe.prange_sum(np.zeros(0), 4) == 0.0


def test_blas_dgemm_matches_numpy(probe):
    """C = A @ B via cython_blas.dgemm must match numpy bit-for-bit."""
    rng = np.random.default_rng(7)
    for m, k, n in ((1, 1, 1), (4, 5, 6), (17, 23, 31), (64, 48, 56)):
        A = np.ascontiguousarray(rng.standard_normal((m, k)))
        B = np.ascontiguousarray(rng.standard_normal((k, n)))
        C = np.zeros((m, n))
        probe.blas_dgemm(A, B, C)
        expected = A @ B
        assert np.allclose(C, expected, atol=1e-12), (
            f"shape m={m} k={k} n={n}: max_err={np.max(np.abs(C - expected)):.2e}"
        )


def test_blas_dgemm_shape_mismatch_raises(probe):
    A = np.ascontiguousarray(np.zeros((4, 5)))
    B = np.ascontiguousarray(np.zeros((6, 7)))  # inner dim mismatch
    C = np.zeros((4, 7))
    with pytest.raises(ValueError, match="inconsistent shapes"):
        probe.blas_dgemm(A, B, C)


def test_blas_dgemm_rejects_non_contiguous(probe):
    """C-contiguous contract: non-contiguous input should be rejected.

    Cython's ``double[:, ::1]`` typed memoryview is strict about the
    C-contiguous stride requirement — this test confirms we can't
    accidentally feed a view.
    """
    rng = np.random.default_rng(11)
    A_big = rng.standard_normal((8, 10))
    A = A_big[::2]  # row-strided, not C-contiguous
    B = np.ascontiguousarray(rng.standard_normal((10, 6)))
    C = np.zeros((4, 6))
    with pytest.raises((ValueError, TypeError)):
        probe.blas_dgemm(A, B, C)
