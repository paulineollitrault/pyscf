# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: language_level=3
"""nogil Cython gather kernels for the CD batched path.

Replaces the per-item Python loop building ``t2_flat`` (C side) and
``u_flat`` (D side, with u = 2*t2 - t2.T) before the c_kernel_batched
/ d_kernel_batched calls. The per-item work is just memory copy +
transpose, but doing it in Python paid ~10 us/item × 11.5 k items per
cycle = 115 ms wall — bigger than the actual kernel compute. These
nogil kernels do the same work via prange across items, reading
``t2_pno_all._buffer`` directly with per-item absolute offsets, and
write into the pre-allocated flat output buffer.
"""
from cython.parallel cimport prange
cimport openmp

import numpy as np


def gather_t2_with_transpose(
    int N,
    int[::1] n_arr,                     # (N,) per-item dimension (n_other)
    long[::1] src_off,                  # (N,) absolute offset into t2_buffer
    int[::1] transpose,                 # (N,) 0/1
    long[::1] dst_off,                  # (N+1,) offset into dst_flat
    double[::1] t2_buffer,              # FlatTensorStore _buffer
    double[::1] dst_flat,               # output: per-item t2 (with optional transpose)
    int num_threads,
):
    """Build t2_flat[dst_off[n]:dst_off[n+1]] = t2 (or t2.T) per item."""
    cdef Py_ssize_t n
    cdef Py_ssize_t n_dim, a, b
    cdef double *src
    cdef double *dst

    for n in prange(N, schedule='static',
                    nogil=True, num_threads=num_threads):
        n_dim = n_arr[n]
        src = &t2_buffer[src_off[n]]
        dst = &dst_flat[dst_off[n]]
        if transpose[n]:
            for a in range(n_dim):
                for b in range(n_dim):
                    dst[a * n_dim + b] = src[b * n_dim + a]
        else:
            for a in range(n_dim):
                for b in range(n_dim):
                    dst[a * n_dim + b] = src[a * n_dim + b]


def gather_u_from_t2(
    int N,
    int[::1] n_arr,
    long[::1] src_off,
    int[::1] transpose,
    long[::1] dst_off,
    double[::1] t2_buffer,
    double[::1] dst_flat,
    int num_threads,
):
    """Build u_flat[dst_off[n]:dst_off[n+1]] = 2*t2 - t2.T (anti-symmetrized).

    With transpose flag: first transposes the canonical t2, then applies
    the (2*x - x.T) transform. Equivalent to:
        t2_d = t2.T if transpose else t2
        u    = 2 * t2_d - t2_d.T
    """
    cdef Py_ssize_t n
    cdef Py_ssize_t n_dim, a, b
    cdef double *src
    cdef double *dst
    cdef double v_ab, v_ba

    for n in prange(N, schedule='static',
                    nogil=True, num_threads=num_threads):
        n_dim = n_arr[n]
        src = &t2_buffer[src_off[n]]
        dst = &dst_flat[dst_off[n]]
        if transpose[n]:
            # t2_d = src.T  →  t2_d[a, b] = src[b, a]
            # u[a, b] = 2*t2_d[a, b] - t2_d[b, a] = 2*src[b, a] - src[a, b]
            for a in range(n_dim):
                for b in range(n_dim):
                    dst[a * n_dim + b] = (
                        2.0 * src[b * n_dim + a] - src[a * n_dim + b])
        else:
            # u[a, b] = 2*src[a, b] - src[b, a]
            for a in range(n_dim):
                for b in range(n_dim):
                    dst[a * n_dim + b] = (
                        2.0 * src[a * n_dim + b] - src[b * n_dim + a])
