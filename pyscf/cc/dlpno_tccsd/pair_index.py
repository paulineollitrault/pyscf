"""Pair-indexed data vocabulary for DLPNO-CCSD restructure.

Replaces the ad-hoc dict-of-pairs pattern used throughout the DLPNO code
with:

- ``PairIndex`` — canonical bijection between pair keys ``(i, j)`` (with
  ``i <= j``) and dense integer ``pair_idx``, plus per-pair metadata
  (``n_pno``, ``domain_lmos``, ordered-pair tables).
- ``TensorStore`` — list-backed pair-indexed container with both
  dict-compatible key access (for gradual migration from existing
  pair-keyed dicts) and fast integer indexed access (``.at(idx)``).

Target for the ongoing restructure: every pair-indexed tensor in the
DLPNO code (t2_pno_all, S_pno_cache, ovL_pno_cache, cc_ints, …) ends up
backed by one of these stores, with later phases upgrading the backend
to flat CSR-style buffers for Cython-nogil access.

See ``HANDOFF`` for the phase plan.  Phase 0: types exist, unit tests
pass, lccsd.py builds a PairIndex and asserts it matches the existing
dict metadata (no behaviour change).
"""

from __future__ import annotations

from typing import Iterable, Tuple

import os
import tempfile
import numpy as np


class PairIndex:
    """Canonical pair index + per-pair metadata.

    Attributes
    ----------
    canonical_keys : list[tuple[int, int]]
        Sorted list of ``(i, j)`` with ``i <= j``.  Position in the list is
        the canonical ``pair_idx``.
    canonical_to_idx : dict[tuple[int, int], int]
        Reverse map ``(i, j) -> pair_idx``.
    n_pairs : int
    nocc : int
    n_pno : np.ndarray(int32, shape=(n_pairs,))
        ``n_pno[p]`` = PNO-space dimension of canonical pair ``p``.
    domain_lmos : list[np.ndarray(int32)]
        Ragged: ``domain_lmos[p]`` = LMO indices in pair ``p``'s local
        domain (Psi4 ``lmopair_to_lmos_[ij]``).  Full ``arange(nocc)`` if
        the caller's ``pair_lmo_idx`` does not cover the pair.
    domain_size : np.ndarray(int32, shape=(n_pairs,))
    ordered_keys : list[tuple[int, int]]
        ``(i, j)`` and ``(j, i)`` for off-diagonal pairs (diagonals once).
    ordered_to_idx : dict[tuple[int, int], int]
    ordered_to_canonical : np.ndarray(int32, shape=(n_ordered,))
        Map ``ordered_idx -> canonical pair_idx``.
    n_ordered : int
    """

    __slots__ = (
        "canonical_keys", "canonical_to_idx", "n_pairs", "nocc",
        "n_pno", "domain_lmos", "domain_size",
        "ordered_keys", "ordered_to_idx", "ordered_to_canonical", "n_ordered",
    )

    def __init__(
        self,
        canonical_keys: Iterable[Tuple[int, int]],
        pno_spaces: dict,
        pair_lmo_idx: dict | None,
        nocc: int,
    ):
        # Normalise: ensure (min, max) ordering, drop duplicates, sort.
        keys_set = set((min(k), max(k)) for k in canonical_keys)
        self.canonical_keys = sorted(keys_set)
        self.canonical_to_idx = {
            k: i for i, k in enumerate(self.canonical_keys)
        }
        self.n_pairs = len(self.canonical_keys)
        self.nocc = int(nocc)

        # n_pno[p] from pno_spaces[k]['n_pno'].
        self.n_pno = np.empty(self.n_pairs, dtype=np.int32)
        for p, k in enumerate(self.canonical_keys):
            space = pno_spaces.get(k)
            if space is None:
                raise KeyError(f"pno_spaces missing key {k}")
            self.n_pno[p] = int(space["n_pno"])

        # Domain LMOs per pair (ragged).  Falls back to full nocc when
        # pair_lmo_idx is None or lacks the key — matches the convention
        # used throughout lccsd.py / residual.py.
        self.domain_lmos = []
        for k in self.canonical_keys:
            if pair_lmo_idx is not None and k in pair_lmo_idx:
                dl = np.asarray(pair_lmo_idx[k], dtype=np.int32)
            else:
                dl = np.arange(self.nocc, dtype=np.int32)
            self.domain_lmos.append(dl)
        self.domain_size = np.array(
            [len(d) for d in self.domain_lmos], dtype=np.int32
        )

        # Ordered-pair table: both directions for off-diagonal pairs,
        # single slot for diagonals.  Consumed by C_tilde / D_tilde style
        # workers that operate on ``(k, i)`` and ``(i, k)`` independently.
        self.ordered_keys = []
        ordered_canonical = []
        for p, (i, j) in enumerate(self.canonical_keys):
            self.ordered_keys.append((i, j))
            ordered_canonical.append(p)
            if i != j:
                self.ordered_keys.append((j, i))
                ordered_canonical.append(p)
        self.ordered_to_idx = {
            k: i for i, k in enumerate(self.ordered_keys)
        }
        self.ordered_to_canonical = np.asarray(
            ordered_canonical, dtype=np.int32
        )
        self.n_ordered = len(self.ordered_keys)

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------
    def idx_of(self, key: Tuple[int, int]) -> int:
        """Return canonical ``pair_idx`` for ``key=(i, j)`` (either order)."""
        return self.canonical_to_idx[(min(key), max(key))]

    def ordered_idx_of(self, key: Tuple[int, int]) -> int:
        """Return ``ordered_idx`` for the *specific* ordering in ``key``."""
        return self.ordered_to_idx[key]

    def canonical_of_ordered(self, ordered_idx: int) -> int:
        """Return canonical ``pair_idx`` for an ``ordered_idx``."""
        return int(self.ordered_to_canonical[ordered_idx])

    def __repr__(self) -> str:
        if self.n_pairs == 0:
            return "PairIndex(empty)"
        return (
            f"PairIndex(n_pairs={self.n_pairs} n_ordered={self.n_ordered} "
            f"nocc={self.nocc} "
            f"n_pno[{int(self.n_pno.min())}..{int(self.n_pno.max())}] "
            f"domain[{int(self.domain_size.min())}..{int(self.domain_size.max())}])"
        )


class TensorStore:
    """List-backed pair-indexed container with dict-compatible API.

    Used as a drop-in replacement for pair-keyed dicts during migration.
    Each slot holds one ndarray (or ``None`` if unset).  Access supports
    both ``store[key]`` (tuple) for dict compatibility and ``store.at(p)``
    for fast integer indexing inside hot loops.

    Parameters
    ----------
    pair_index : PairIndex
    shape_fn : callable(pair_idx) -> tuple, optional
        Shape builder used when ``init`` is ``None`` and ``fill_zero`` is
        ``True``.  Default: ``(n_pno[p], n_pno[p])`` (square PNO tensor).
    dtype : numpy dtype
    init : dict[(i, j), ndarray], optional
        Seed data.  Keys are canonicalised (``min``, ``max``).
    fill_zero : bool
        If ``True`` (default) and ``init`` is ``None``, pre-allocate
        zero tensors via ``shape_fn``.  If ``False``, slots are ``None``
        until set.
    """

    __slots__ = ("_pi", "_data")

    def __init__(
        self,
        pair_index: PairIndex,
        shape_fn=None,
        dtype=np.float64,
        init: dict | None = None,
        fill_zero: bool = True,
    ):
        self._pi = pair_index
        if shape_fn is None:
            def shape_fn(p):
                n = int(pair_index.n_pno[p])
                return (n, n)
        self._data = [None] * pair_index.n_pairs
        if init is not None:
            for key, arr in init.items():
                canonical = (min(key), max(key))
                idx = pair_index.canonical_to_idx.get(canonical)
                if idx is not None:
                    self._data[idx] = arr
        elif fill_zero:
            for p in range(pair_index.n_pairs):
                self._data[p] = np.zeros(shape_fn(p), dtype=dtype)

    # ------------------------------------------------------------------
    # Fast indexed access (preferred API for new code)
    # ------------------------------------------------------------------
    def at(self, pair_idx: int):
        return self._data[pair_idx]

    def set_at(self, pair_idx: int, value) -> None:
        self._data[pair_idx] = value

    # ------------------------------------------------------------------
    # dict-compatible API (for gradual migration)
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self._data[self._pi.canonical_to_idx[(min(key), max(key))]]
        return self._data[key]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            self._data[self._pi.canonical_to_idx[(min(key), max(key))]] = value
        else:
            self._data[key] = value

    def __contains__(self, key):
        if isinstance(key, tuple):
            canonical = (min(key), max(key))
            idx = self._pi.canonical_to_idx.get(canonical)
            if idx is None:
                return False
            return self._data[idx] is not None
        return 0 <= key < self._pi.n_pairs and self._data[key] is not None

    def get(self, key, default=None):
        if isinstance(key, tuple):
            canonical = (min(key), max(key))
            idx = self._pi.canonical_to_idx.get(canonical)
            if idx is None:
                return default
            v = self._data[idx]
            return v if v is not None else default
        if 0 <= key < self._pi.n_pairs:
            v = self._data[key]
            return v if v is not None else default
        return default

    def keys(self):
        return [
            self._pi.canonical_keys[p]
            for p in range(self._pi.n_pairs)
            if self._data[p] is not None
        ]

    def values(self):
        return [v for v in self._data if v is not None]

    def items(self):
        return [
            (self._pi.canonical_keys[p], self._data[p])
            for p in range(self._pi.n_pairs)
            if self._data[p] is not None
        ]

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return sum(1 for v in self._data if v is not None)

    def __repr__(self):
        return (
            f"TensorStore(n_slots={self._pi.n_pairs} n_present={len(self)})"
        )


class FlatTensorStore:
    """Pair-indexed tensor store backed by a single flat ndarray.

    Storage layout (CSR-like):

      - ``_buffer``  : 1D ndarray, total size ``sum_p prod(shape[p])``
      - ``_offsets`` : int64[n_pairs + 1] slicing the buffer per pair
      - ``_shapes``  : int32[n_pairs, max_ndim] per-pair shapes
                       (trailing zeros indicate unused dims; ``_ndim``
                       stores the canonical rank so dim=0 and unused
                       are distinguishable)
      - ``_ndim``    : int, common rank of all pair tensors

    Key differences vs ``TensorStore``:

      - Zero-copy ``at(p)`` returns an ``ndarray`` *view* into
        ``_buffer`` with the correct shape — no per-pair allocation.
      - A single contiguous buffer + offset table is what Cython-nogil
        kernels need: the kernel can take ``_buffer`` as a typed
        memoryview, ``_offsets`` / ``_shapes`` as int arrays, and
        navigate without Python.
      - ``__setitem__`` copies the incoming array into the pair's
        reserved slot (shape must match the allocated slot).

    Shapes may be ragged across pairs (e.g. ``n_pno[p]`` varies), but
    within a pair the tensor is contiguous in C order.  All pairs must
    have the same rank (enforced at construction).
    """

    __slots__ = ("_pi", "_buffer", "_offsets", "_shapes", "_ndim", "dtype",
                 "_views", "_canon_to_idx", "_mmap_path")

    def __init__(self, pair_index, shape_fn, dtype=np.float64):
        self._pi = pair_index
        self.dtype = np.dtype(dtype)
        self._mmap_path = None
        n_pairs = pair_index.n_pairs

        # First pass: determine per-pair shapes + common rank + sizes.
        shape_tuples = [tuple(shape_fn(p)) for p in range(n_pairs)]
        if n_pairs == 0:
            self._ndim = 0
            self._shapes = np.zeros((0, 0), dtype=np.int32)
            self._offsets = np.zeros(1, dtype=np.int64)
            self._buffer = np.zeros(0, dtype=self.dtype)
            self._views = []
            self._canon_to_idx = pair_index.canonical_to_idx
            return

        ranks = {len(s) for s in shape_tuples}
        if len(ranks) != 1:
            raise ValueError(
                f"FlatTensorStore requires uniform rank across pairs; "
                f"got {ranks}"
            )
        self._ndim = next(iter(ranks))

        self._shapes = np.zeros((n_pairs, self._ndim), dtype=np.int32)
        sizes = np.zeros(n_pairs, dtype=np.int64)
        for p, shape in enumerate(shape_tuples):
            for d, n in enumerate(shape):
                self._shapes[p, d] = n
            sizes[p] = int(np.prod(shape)) if shape else 1

        self._offsets = np.zeros(n_pairs + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum(sizes)
        _total = int(self._offsets[-1])
        # Low-memory mode: back the buffer with a memmap on PYSCF_TMPDIR (a
        # real NVMe disk).  cc_ints is the dominant Stage-5 base allocation
        # (~79 GiB on a TM complex); paging it out keeps it off the resident
        # set.  Crucially the buffer is written once during flatten then read
        # only — so its file-backed pages are CLEAN and the kernel can evict
        # them under memory pressure (e.g. while the C++-class pack builds its
        # own copy), giving automatic "use-and-free" without a code rewrite.
        # The Cython/C kernels read it through the same raw pointer; the OS
        # pages it transparently.  Enable with DLPNO_CCINTS_MMAP=1.
        if os.environ.get('DLPNO_CCINTS_MMAP') and _total > 0:
            _tmpdir = os.environ.get('PYSCF_TMPDIR') or tempfile.gettempdir()
            _fd, self._mmap_path = tempfile.mkstemp(
                suffix='.ccflat', prefix='dlpno_', dir=_tmpdir)
            os.close(_fd)
            self._buffer = np.memmap(self._mmap_path, dtype=self.dtype,
                                     mode='w+', shape=(_total,))
            _register_plan_tmpfile(self._mmap_path)   # remove at exit
        else:
            self._buffer = np.zeros(_total, dtype=self.dtype)

        # Pre-build per-pair views.  The buffer is allocated once and
        # never resized; mutations go through ``set_at`` (view[:] = ...)
        # so these views always reflect current buffer contents.  This
        # collapses ``at()`` from 2–3µs of Python overhead (tuple parse
        # + int casts + shape tuple build + reshape) down to a list
        # index.  Zero-sized slots get a fresh zeros() on each call —
        # rare in practice.
        self._views = []
        for p, shape in enumerate(shape_tuples):
            start = int(self._offsets[p])
            end = int(self._offsets[p + 1])
            if end == start:
                self._views.append(None)   # sentinel — fallback in at()
            else:
                self._views.append(
                    self._buffer[start:end].reshape(shape)
                )
        # Local reference to the (tuple → int) dict avoids attribute
        # chase on every tuple lookup.
        self._canon_to_idx = pair_index.canonical_to_idx

    # ------------------------------------------------------------------
    # Zero-copy access
    # ------------------------------------------------------------------
    def at(self, pair_idx):
        """Return a view of pair ``pair_idx``'s tensor (zero-copy).

        Hot path: pre-built views are cached in ``self._views``.  Tuple
        keys incur one dict lookup on top of that.  A zero-sized slot
        (None sentinel) falls through to a fresh np.zeros — the declared
        shape comes from ``self._shapes`` in that cold path.
        """
        if isinstance(pair_idx, tuple):
            if pair_idx[0] <= pair_idx[1]:
                pair_idx = self._canon_to_idx[pair_idx]
            else:
                pair_idx = self._canon_to_idx[(pair_idx[1], pair_idx[0])]
        v = self._views[pair_idx]
        if v is not None:
            return v
        shape = tuple(int(n) for n in self._shapes[pair_idx])
        return np.zeros(shape, dtype=self.dtype)

    def set_at(self, pair_idx, value):
        view = self.at(pair_idx)
        if view.size == 0:
            return
        view[:] = value

    # ------------------------------------------------------------------
    # dict-compatible API (mirror of TensorStore)
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        return self.at(key)

    def __setitem__(self, key, value):
        self.set_at(key, value)

    def __contains__(self, key):
        if isinstance(key, tuple):
            canonical = (min(key), max(key))
            return canonical in self._pi.canonical_to_idx
        return 0 <= key < self._pi.n_pairs

    def get(self, key, default=None):
        if isinstance(key, tuple):
            canonical = (min(key), max(key))
            idx = self._pi.canonical_to_idx.get(canonical)
            if idx is None:
                return default
            return self.at(idx)
        if 0 <= key < self._pi.n_pairs:
            return self.at(key)
        return default

    def keys(self):
        return list(self._pi.canonical_keys)

    def values(self):
        return [self.at(p) for p in range(self._pi.n_pairs)]

    def items(self):
        return [(self._pi.canonical_keys[p], self.at(p))
                for p in range(self._pi.n_pairs)]

    def __iter__(self):
        return iter(self._pi.canonical_keys)

    def __len__(self):
        return self._pi.n_pairs

    def __repr__(self):
        return (
            f"FlatTensorStore(n_pairs={self._pi.n_pairs} "
            f"buffer={self._buffer.size} ndim={self._ndim} "
            f"dtype={self.dtype})"
        )

    # ------------------------------------------------------------------
    # Low-level accessors for Cython / nogil consumers.
    # The kernel takes these three arrays as typed memoryviews.
    # ------------------------------------------------------------------
    @property
    def buffer(self):
        """Flat backing buffer.  Cython: ``double[::1] buf``."""
        return self._buffer

    @property
    def offsets(self):
        """CSR-style offsets (int64).  Cython: ``long[::1] off``."""
        return self._offsets

    @property
    def shapes(self):
        """Per-pair shape matrix (int32, shape ``(n_pairs, ndim)``)."""
        return self._shapes

    @classmethod
    def from_dict(cls, pair_index, source_dict, shape_fn=None,
                  dtype=np.float64):
        """Build a FlatTensorStore seeded from a pair-keyed dict.

        ``shape_fn`` defaults to ``source_dict[key].shape`` for each pair.
        Missing entries get zero-filled.
        """
        if shape_fn is None:
            def shape_fn(p):
                key = pair_index.canonical_keys[p]
                arr = source_dict.get(key)
                return tuple(arr.shape) if arr is not None else (0,)
        store = cls(pair_index, shape_fn, dtype=dtype)
        for key, arr in source_dict.items():
            canonical = (min(key), max(key))
            if canonical in pair_index.canonical_to_idx:
                store[key] = arr
        return store


def _madvise_dontneed(buf):
    """Drop a (just-flushed) memmap's pages from this process's RAM.

    For a MAP_SHARED file-backed mapping, MADV_DONTNEED frees the resident
    pages after their dirty data has been written back (we call .flush()
    first); subsequent access re-faults the data from /scratch.  This keeps
    the S_pno buffer's resident footprint near zero while it is being filled.
    Best-effort: any failure (non-Linux, no libc) is silently ignored.
    """
    try:
        import ctypes
        base = buf.ctypes.data            # mmap base is page-aligned
        nbytes = buf.nbytes
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        MADV_DONTNEED = 4
        libc.madvise(ctypes.c_void_p(base), ctypes.c_size_t(nbytes),
                     ctypes.c_int(MADV_DONTNEED))
    except Exception:
        pass


def _plan_stream_on():
    """True if CCSD-cycle plan buffers should be NVMe-backed (Stage 1).

    Gated by DLPNO_STREAM_PLANS (or the umbrella DLPNO_CCINTS_MMAP, so the
    existing 'turn streaming on' flag also streams the plan caches).
    """
    return bool(os.environ.get('DLPNO_STREAM_PLANS')
                or os.environ.get('DLPNO_CCINTS_MMAP'))


def stream_empty(shape, dtype=np.float64, tag='plan'):
    """np.empty, but NVMe-memmap-backed when plan-streaming is on and the
    buffer is large (Stage 1).

    CCSD-cycle plan buffers (gathered K / S stacks etc.) are write-once,
    read-every-cycle — the same profile as cc_ints.  Backing them with a
    file-mapped buffer turns them from anon RAM (which the OOM killer counts)
    into clean file pages the kernel can EVICT under memory pressure, then
    re-fault on the next cycle's read.  Numerically identical: same bytes, the
    C kernels read through the same raw pointer.

    Threshold DLPNO_STREAM_PLAN_MIN_MB (default 128 MiB) keeps tiny buffers in
    RAM (not worth a file).  Returns a normal np.empty otherwise.
    """
    shape = tuple(int(d) for d in (shape if isinstance(shape, (tuple, list))
                                   else (shape,)))
    nelem = 1
    for d in shape:
        nelem *= d
    itemsize = np.dtype(dtype).itemsize
    _min = float(os.environ.get('DLPNO_STREAM_PLAN_MIN_MB', '128')) * (1 << 20)
    if _plan_stream_on() and nelem * itemsize > _min:
        _td = os.environ.get('PYSCF_TMPDIR') or tempfile.gettempdir()
        _fd, _path = tempfile.mkstemp(suffix='.' + tag, prefix='dlpno_plan_',
                                      dir=_td)
        os.close(_fd)
        buf = np.memmap(_path, dtype=dtype, mode='w+', shape=shape)
        # Do NOT unlink here: an unlinked inode + madvise(DONTNEED) re-fault
        # corrupts data (unlike a live file). Match the cc_ints/S_pno stores
        # (which keep the path) and clean the files at interpreter exit.
        _register_plan_tmpfile(_path)
        return buf
    return np.empty(shape, dtype=dtype)


_PLAN_TMPFILES = []


def _register_plan_tmpfile(path):
    """Track a plan-stream backing file for best-effort removal at exit."""
    if not _PLAN_TMPFILES:
        import atexit

        def _cleanup():
            for _p in _PLAN_TMPFILES:
                try:
                    os.unlink(_p)
                except OSError:
                    pass
        atexit.register(_cleanup)
    _PLAN_TMPFILES.append(path)


def stream_settle(buf):
    """After a stream_empty() buffer is fully written, flush its dirty pages to
    disk and drop them from RAM (they re-fault on read).  No-op for a normal
    ndarray.  Call once the plan buffer's fill loop is complete."""
    if isinstance(buf, np.memmap):
        buf.flush()
        if not os.environ.get('DLPNO_STREAM_NO_MADVISE'):
            _madvise_dontneed(buf)


def stream_plan_cache(plan, tag='plancache'):
    """Consolidate a built plan cache's large OWNED float64 arrays into one
    NVMe-backed buffer (Stage 1), replacing them with views.

    Why not stream_empty per array: the BE/CD/C~/D~/G-term plans are built as
    MANY small per-shape bucket arrays, each below the per-buffer threshold but
    summing to tens of GiB of anon RAM held read-only across ALL CCSD cycles.
    This walks the finished plan, packs every large *owned* (base is None),
    non-memmap, C-contiguous float64 array into a single memmap, and points the
    plan at views into it — so the whole plan becomes OS-EVICTABLE file pages
    instead of anon (which the OOM killer counts).  Arrays that are already
    views/aliases (e.g. into the mmap'd cc_ints / S_pno masters) have base != None
    and are LEFT UNTOUCHED, preserving the existing zero-copy dedup.

    Incremental copy+free keeps the transient ~= plan size (no 2x spike).
    Numerically identical (same bytes; C kernels read the same layout through
    the view's raw pointer).  No-op unless plan-streaming is on.
    """
    if plan is None or not _plan_stream_on():
        return plan
    _min_arr = float(os.environ.get('DLPNO_STREAM_PLAN_ARR_MIN_MB', '1')) \
        * (1 << 20)
    _min_total = float(os.environ.get('DLPNO_STREAM_PLAN_MIN_MB', '128')) \
        * (1 << 20)

    def _eligible(a):
        return (isinstance(a, np.ndarray) and not isinstance(a, np.memmap)
                and a.dtype == np.float64 and a.base is None
                and a.flags['C_CONTIGUOUS'] and a.nbytes >= _min_arr)

    _dbg = {} if os.environ.get('DLPNO_STREAM_PLAN_DEBUG') else None
    _dbg_masters = {}   # id(root) -> nbytes, for arrays reached only via views

    def _root_base(a):
        b = a
        while isinstance(getattr(b, 'base', None), np.ndarray):
            b = b.base
        return b

    def _categorize(a):
        if isinstance(a, np.memmap):
            k = 'memmap'
        elif a.dtype != np.float64:
            k = 'nonf64'
        elif a.base is not None:
            k = 'view'
            r = _root_base(a)
            if not isinstance(r, np.memmap):
                _dbg_masters[id(r)] = int(r.nbytes)
        elif not a.flags['C_CONTIGUOUS']:
            k = 'noncontig'
        elif a.nbytes < _min_arr:
            k = 'small'
        else:
            k = 'caught'
        e = _dbg.setdefault(k, [0, 0])
        e[0] += 1
        e[1] += int(a.nbytes)

    # Pass 1: collect unique eligible arrays by identity (mirror driver's
    # _walk_bytes: recurse dict / list / tuple / set, capped depth).
    order = []          # unique eligible arrays, in first-seen order
    ids = set()
    seen = set()

    def _collect(obj, depth=0):
        if depth > 8 or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, np.ndarray):
            if _dbg is not None:
                _categorize(obj)
            if _eligible(obj) and id(obj) not in ids:
                ids.add(id(obj))
                order.append(obj)
            return
        if isinstance(obj, dict):
            for v in obj.values():
                _collect(v, depth + 1)
        elif isinstance(obj, (list, tuple, set)):
            for v in obj:
                _collect(v, depth + 1)

    _collect(plan)
    if _dbg is not None:
        _mtot = sum(_dbg_masters.values())
        _s = '  '.join(f'{k}={v[1]/2**30:.2f}G/{v[0]}' for k, v in _dbg.items())
        print(f'  [PLANSTREAM/{tag}] {_s}  view_masters={_mtot/2**30:.2f}G/'
              f'{len(_dbg_masters)}', flush=True)
    if not order:
        return plan
    total = sum(a.size for a in order)
    if total * 8 < _min_total:
        return plan
    # Pack into one memmap; build id(arr) -> view map.
    buf = stream_empty((total,), tag=tag)
    views = {}
    off = 0
    for a in order:
        n = a.size
        buf[off:off + n] = a.ravel()
        views[id(a)] = buf[off:off + n].reshape(a.shape)  # base -> buf (kept)
        off += n
    stream_settle(buf)
    del order          # drop our refs; plan still holds originals until rebuild

    # Pass 2: rebuild the structure, swapping each collected array for its view.
    # dict/list mutated in place; tuples rebuilt (immutable). As each parent's
    # last reference to an original is replaced, that anon array is freed.
    rebuilt = {}

    def _rebuild(obj, depth=0):
        vid = views.get(id(obj))
        if vid is not None:
            return vid
        if depth > 8:
            return obj
        oid = id(obj)
        if oid in rebuilt:
            return rebuilt[oid]
        if isinstance(obj, dict):
            rebuilt[oid] = obj
            for k in list(obj.keys()):
                obj[k] = _rebuild(obj[k], depth + 1)
            return obj
        if isinstance(obj, list):
            rebuilt[oid] = obj
            for i in range(len(obj)):
                obj[i] = _rebuild(obj[i], depth + 1)
            return obj
        if isinstance(obj, tuple):
            new = tuple(_rebuild(x, depth + 1) for x in obj)
            rebuilt[oid] = new
            return new
        return obj

    return _rebuild(plan)


class FlatPairPairStore:
    """Sparse ``(pair_a, pair_b) -> ndarray`` store with flat backing.

    Handles pair-of-pair-indexed tensors like ``S_pno_cache`` where
    ``S[(pair_a, pair_b)]`` is the PNO-overlap matrix between two
    distinct pair PNO spaces.  Not every (pair_a, pair_b) has an entry
    — only pairs whose PNO spaces actually overlap — hence the sparse
    layout.

    Two storage tiers:

    1. **Flat tier** — ``(n_flat_entries,)`` concatenated into a single
       ndarray ``_buffer`` with CSR-style ``_offsets`` (int64) and
       per-entry ``_shapes`` (int32, shape ``(N, 2)``).  Entry k's view
       is ``_buffer[_offsets[k]:_offsets[k+1]].reshape(_shapes[k])``.
       Indexed by ``(pair_a_idx, pair_b_idx) -> k`` via ``_key_to_idx``.

    2. **Overflow dict** — catches lazy insertions after construction
       (``_s_pno_getter`` can compute new entries on demand).  Keyed by
       ``(pair_a_idx, pair_b_idx)`` and holds the raw ndarray.

    A Cython kernel consumes only the flat tier via ``buffer``,
    ``offsets``, ``shapes``, and ``index_matrix`` (a dense
    ``(n_pairs, n_pairs)`` int32 table giving ``k`` or ``-1``).  The
    overflow dict is Python-only.
    """

    __slots__ = (
        "_pi", "dtype", "_buffer", "_offsets", "_shapes",
        "_idx_matrix", "_n_flat", "_overflow",
        "_views", "_canon_to_idx", "_mmap_path",
    )

    def __init__(self, pair_index, initial=None, dtype=np.float64):
        """
        Parameters
        ----------
        pair_index : PairIndex
        initial : dict[(pair_a, pair_b), ndarray] | None
            Entries seeded into the flat tier.  Missing pairs (not in
            ``pair_index.canonical_keys``) are skipped silently.
        dtype : numpy dtype
        """
        self._pi = pair_index
        self.dtype = np.dtype(dtype)
        self._mmap_path = None

        # Low-memory mode: back the flat buffer with a memmap on PYSCF_TMPDIR
        # (a real disk such as /scratch) instead of RAM.  The S_pno_cache
        # buffer is the dominant Stage-5 allocation on large/TM systems
        # (tens of GiB); paging it to NVMe keeps it out of the resident set.
        # The C++ solver / Cython kernels read it through the same raw
        # pointer, so the memmap is transparent (the OS pages it in/out).
        # When enabled we also POP each source array out of ``initial`` as it
        # is written so the source dict and the buffer never both stay
        # resident — this is what reduces the *construction* peak.
        _mmap = bool(os.environ.get('DLPNO_SPNO_MMAP'))

        # Pass 1: resolve (idx_a, idx_b) + shapes WITHOUT retaining the arrays
        # (so memmap mode can free them incrementally in pass 2).
        meta = []  # (ia, ib, orig_key, shape)
        if initial is not None:
            for (pa, pb), arr in initial.items():
                ia = pair_index.canonical_to_idx.get((min(pa), max(pa)))
                ib = pair_index.canonical_to_idx.get((min(pb), max(pb)))
                if ia is None or ib is None:
                    continue
                shape = arr.shape
                if len(shape) != 2:
                    raise ValueError(
                        f"FlatPairPairStore expects rank-2 entries; "
                        f"got shape {shape} for ({ia},{ib})")
                meta.append((ia, ib, (pa, pb), shape))

        n = len(meta)
        self._n_flat = n
        self._shapes = np.zeros((n, 2), dtype=np.int32)
        sizes = np.zeros(n, dtype=np.int64)
        for k, (ia, ib, _ok, shape) in enumerate(meta):
            self._shapes[k, 0] = shape[0]
            self._shapes[k, 1] = shape[1]
            sizes[k] = shape[0] * shape[1]

        self._offsets = np.zeros(n + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum(sizes)
        _total = int(self._offsets[-1])
        if _mmap and _total > 0:
            _tmpdir = os.environ.get('PYSCF_TMPDIR') or tempfile.gettempdir()
            _fd, self._mmap_path = tempfile.mkstemp(
                suffix='.spno', prefix='dlpno_', dir=_tmpdir)
            os.close(_fd)
            self._buffer = np.memmap(self._mmap_path, dtype=self.dtype,
                                     mode='w+', shape=(_total,))
            _register_plan_tmpfile(self._mmap_path)   # remove at exit
        else:
            self._buffer = np.empty(_total, dtype=self.dtype)

        # (n_pairs, n_pairs) int32 dense lookup; ``-1`` == "not in flat tier".
        n_pairs = pair_index.n_pairs
        self._idx_matrix = np.full((n_pairs, n_pairs), -1, dtype=np.int32)
        # Flush the memmap to disk every ~4 GiB of writes so the dirty pages
        # don't accumulate in RAM (which would defeat the point of paging the
        # buffer out).  After flush+madvise the pages are clean/file-backed
        # and the kernel can evict them under memory pressure.
        _flush_every = (4 << 30) // self.dtype.itemsize if _mmap else 0
        _since_flush = 0
        for k, (ia, ib, okey, _shape) in enumerate(meta):
            start = int(self._offsets[k])
            end = int(self._offsets[k + 1])
            if end > start:
                arr = (initial.pop(okey) if _mmap else initial[okey])
                self._buffer[start:end] = arr.ravel()
                del arr
                _since_flush += (end - start)
                if _flush_every and _since_flush >= _flush_every:
                    self._buffer.flush()
                    _madvise_dontneed(self._buffer)
                    _since_flush = 0
            self._idx_matrix[ia, ib] = k
        if _mmap and _total > 0:
            self._buffer.flush()
            _madvise_dontneed(self._buffer)

        self._overflow = {}

        # Pre-build views for every flat entry (buffer is stable; writes
        # go through view[:] = ... in __setitem__).  Cold path for zero-
        # sized entries returns a fresh zeros().
        self._views = []
        for k in range(n):
            start = int(self._offsets[k])
            end = int(self._offsets[k + 1])
            shape = (int(self._shapes[k, 0]), int(self._shapes[k, 1]))
            if end == start:
                self._views.append(None)
            else:
                self._views.append(
                    self._buffer[start:end].reshape(shape)
                )
        # Local ref avoids the attribute chase on every lookup.
        self._canon_to_idx = pair_index.canonical_to_idx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _canonical_idx(self, key):
        """(pair_a, pair_b) tuple of pairs → (idx_a, idx_b) ints."""
        pair_a, pair_b = key
        if pair_a[0] <= pair_a[1]:
            ia = self._canon_to_idx[pair_a]
        else:
            ia = self._canon_to_idx[(pair_a[1], pair_a[0])]
        if pair_b[0] <= pair_b[1]:
            ib = self._canon_to_idx[pair_b]
        else:
            ib = self._canon_to_idx[(pair_b[1], pair_b[0])]
        return ia, ib

    def _view_at(self, k):
        v = self._views[k]
        if v is not None:
            return v
        shape = (int(self._shapes[k, 0]), int(self._shapes[k, 1]))
        return np.zeros(shape, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Integer-indexed fast path (Cython-friendly)
    # ------------------------------------------------------------------
    def at_idx(self, ia, ib):
        """Return ndarray for ``(idx_a, idx_b)`` or ``None``."""
        k = self._idx_matrix[ia, ib]
        if k >= 0:
            v = self._views[k]
            if v is not None:
                return v
            shape = (int(self._shapes[k, 0]), int(self._shapes[k, 1]))
            return np.zeros(shape, dtype=self.dtype)
        return self._overflow.get((ia, ib))

    # ------------------------------------------------------------------
    # Dict-compat API (for gradual migration)
    # ------------------------------------------------------------------
    def __getitem__(self, key):
        ia, ib = self._canonical_idx(key)
        v = self.at_idx(ia, ib)
        if v is None:
            raise KeyError(key)
        return v

    def __setitem__(self, key, value):
        ia, ib = self._canonical_idx(key)
        k = int(self._idx_matrix[ia, ib])
        if k >= 0:
            shape = (int(self._shapes[k, 0]), int(self._shapes[k, 1]))
            if tuple(value.shape) == shape:
                # In-place update into the flat buffer.
                start = int(self._offsets[k])
                end = int(self._offsets[k + 1])
                if end > start:
                    self._buffer[start:end] = np.asarray(value).ravel()
                return
            # Shape changed (unexpected) — fall through to overflow.
        self._overflow[(ia, ib)] = value

    def __contains__(self, key):
        try:
            ia, ib = self._canonical_idx(key)
        except KeyError:
            return False
        return (
            self._idx_matrix[ia, ib] >= 0
            or (ia, ib) in self._overflow
        )

    def get(self, key, default=None):
        try:
            ia, ib = self._canonical_idx(key)
        except KeyError:
            return default
        v = self.at_idx(ia, ib)
        return v if v is not None else default

    def keys(self):
        ckeys = self._pi.canonical_keys
        flat = np.argwhere(self._idx_matrix >= 0)
        out = [(ckeys[int(ia)], ckeys[int(ib)]) for ia, ib in flat]
        out.extend((ckeys[ia], ckeys[ib]) for (ia, ib) in self._overflow)
        return out

    def values(self):
        flat = np.argwhere(self._idx_matrix >= 0)
        out = [self._view_at(int(self._idx_matrix[ia, ib]))
               for ia, ib in flat]
        out.extend(self._overflow.values())
        return out

    def items(self):
        return list(zip(self.keys(), self.values()))

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return self._n_flat + len(self._overflow)

    def __repr__(self):
        return (
            f"FlatPairPairStore(n_flat={self._n_flat} "
            f"n_overflow={len(self._overflow)} "
            f"buffer={self._buffer.size} dtype={self.dtype})"
        )

    # ------------------------------------------------------------------
    # Cython / nogil accessors
    # ------------------------------------------------------------------
    @property
    def buffer(self):
        return self._buffer

    @property
    def offsets(self):
        return self._offsets

    @property
    def shapes(self):
        return self._shapes

    @property
    def n_flat_entries(self):
        return self._n_flat

    @property
    def n_overflow_entries(self):
        return len(self._overflow)

    def index_matrix(self):
        """``(n_pairs, n_pairs)`` int32: flat index ``k`` or ``-1``.

        Use this as a dense sparse table inside Cython kernels.
        ``-1`` means the pair-of-pair is not in the flat tier (might be
        in overflow, but Cython should not see those — the caller must
        guarantee completeness before entering a nogil region).
        """
        return self._idx_matrix


# ----------------------------------------------------------------------
# T1 projection cache — Phase 1 of the restructure.
#
# Matches Psi4's `T_n_ij_[ij]` pattern: every CCSD cycle, project t1_k
# into every pair's PNO basis once, so downstream kernels can index
# into a pre-built matrix instead of each site calling
# `_project_t1_to_pair` lazily (~1.5 M calls/run at water8 in the old
# path).  Stored as a TensorStore keyed by canonical pair with entries
# of shape ``(nocc, n_pno[pair])``.
# ----------------------------------------------------------------------
def build_t1_cache(
    t1_pno: dict,
    pair_index: PairIndex,
    S_pno_cache: dict,
    pno_spaces: dict,
) -> "FlatTensorStore":
    """Pre-project t1 into every pair's PNO basis, one row per LMO.

    Semantics match ``_project_t1_to_pair``:
      - If ``t1_pno[k]`` is absent / empty → row is zero.
      - If ``pair == (k, k)`` → row is ``t1_pno[k]`` (no projection).
      - Else if ``S_pno_cache[(pair, (k,k))]`` is present → row is
        ``S @ t1_pno[k]``.
      - Else (S missing) → row is zero (fallback).

    Returns
    -------
    FlatTensorStore
        Indexed by canonical pair key.  Each slot is a ``(nocc, n_pno[p])``
        view into a single flat float64 buffer.  The same ``store[key][k]``
        indexing pattern works as with the legacy ``TensorStore``, but
        all per-pair rows share one contiguous allocation — exactly the
        layout Phase 4's Cython kernels can take as typed memoryviews
        without Python dispatch.
    """
    nocc = pair_index.nocc
    cache = FlatTensorStore(
        pair_index,
        shape_fn=lambda p: (nocc, int(pair_index.n_pno[p])),
    )
    for p, pair_key in enumerate(pair_index.canonical_keys):
        n_pno_p = int(pair_index.n_pno[p])
        if n_pno_p == 0:
            continue
        out = cache.at(p)  # (nocc, n_pno_p) view into flat buffer (zeros).
        for k in range(nocc):
            t1_k = t1_pno.get(k)
            if t1_k is None or t1_k.size == 0:
                continue  # already zeros
            key_kk = (k, k)
            if pair_key == key_kk:
                out[k] = t1_k
                continue
            S = S_pno_cache.get((pair_key, key_kk))
            if S is not None:
                out[k] = S @ t1_k
            # else: leave row as zeros — matches _project_t1_to_pair's
            # final ``return np.zeros(...)`` fallback.
    return cache


# ----------------------------------------------------------------------
# cc_ints flattening — Phase 2e of the restructure.
# ----------------------------------------------------------------------
# Per-pair DF / exchange integrals start life as nested dicts of
# ndarrays (see ``local_df.compute_cc_integrals_sparse``).  Phase 2e
# replaces each tensor field's per-pair ndarray with a view into one
# shared ``FlatTensorStore`` backing buffer.  The dict structure of
# ``cc_ints`` is preserved so existing Python callers work unchanged,
# but the underlying memory is now contiguous across pairs — which is
# what Phase 4's Cython kernels need.
#
# Only the 12 fields with uniform rank across pairs are flattened:
#
#   Rank 3: Qab (n_local, n_pno, n_pno), Qma (n_local, nocc, n_pno)
#   Rank 2: i_Qa, j_Qa (n_local, n_pno);  i_Qk, j_Qk (n_local, nocc);
#           K_iajb, J_ijab (n_pno, n_pno); K_mnij (nocc, nocc);
#           K_bar_chem, K_bar_ij, K_bar_ji (nocc, n_pno).
#
# Left untouched (kept as normal dict entries):
#
#   - Scalars / small metadata: n_local, aux_idx, p_lmos, p_lmos_dense.
#   - Nested dicts with composite (pair, k) keys: J_ij_kj, K_ij_kj,
#     J_ji_ki, K_ji_ki.  Flattening them requires per-(pair, k)
#     indexing; deferred — they're small and accessed in few sites.
# ----------------------------------------------------------------------
_CC_INTS_FLAT_FIELDS_3D = ("Qab", "Qma")
_CC_INTS_FLAT_FIELDS_2D = (
    "i_Qa", "j_Qa", "i_Qk", "j_Qk",
    "K_iajb", "K_bar_ij", "K_bar_ji", "K_bar_chem", "J_ijab",
)


def flatten_cc_ints_fields(cc_ints, pair_index, _pool=None):
    """Replace per-pair tensor fields in ``cc_ints`` with flat-buffer views.

    For each of the 12 tensor fields listed above, build one
    ``FlatTensorStore`` holding that field's data for every pair in a
    single contiguous buffer.  Each ``cc_ints[pair][field]`` is then
    overwritten with ``store.at(pair_idx)`` — a view into the shared
    buffer — so the original per-pair ndarray can be garbage-collected.

    The ``cc_ints`` dict is mutated in place.  Nested dicts, scalars,
    and metadata fields are left untouched.

    If ``_pool`` is provided, the 12 fields are flattened in parallel —
    each field is independent (different sub-dict keys) and the inner
    numpy copy releases the GIL.

    Parameters
    ----------
    cc_ints : dict
        ``{pair_key: per_pair_dict | None}`` as produced by
        ``compute_cc_integrals_sparse``.
    pair_index : PairIndex
    _pool : concurrent.futures.Executor, optional
        Thread pool for parallel per-field flatten.

    Returns
    -------
    dict[field_name, FlatTensorStore]
        For each flattened field, the shared flat store exposing
        ``.buffer`` / ``.offsets`` / ``.shapes`` for Cython consumers.
    """
    canonical_keys = pair_index.canonical_keys

    def _flatten_one_field(field, rank):
        zero_shape = (0,) * rank

        def shape_fn(p, field=field, zero_shape=zero_shape):
            key = canonical_keys[p]
            entry = cc_ints.get(key)
            if entry is None:
                return zero_shape
            arr = entry.get(field)
            if arr is None:
                return zero_shape
            return tuple(arr.shape)

        store = FlatTensorStore(pair_index, shape_fn=shape_fn)

        # Seed the buffer + replace each entry's ndarray with a view.
        # Processed pair-by-pair so the old ndarray's refcount drops
        # to zero as soon as we overwrite the dict slot.
        for p, key in enumerate(canonical_keys):
            entry = cc_ints.get(key)
            if entry is None:
                continue
            arr = entry.get(field)
            if arr is None:
                continue
            store[key] = arr             # copy into flat buffer
            entry[field] = store.at(p)   # replace with view
        return field, store

    tasks = (
        [(f, 3) for f in _CC_INTS_FLAT_FIELDS_3D]
        + [(f, 2) for f in _CC_INTS_FLAT_FIELDS_2D]
    )

    if _pool is not None and len(tasks) > 1:
        results = list(_pool.map(
            lambda ft: _flatten_one_field(ft[0], ft[1]), tasks))
    else:
        results = [_flatten_one_field(f, r) for f, r in tasks]

    # If any field is memmap-backed (DLPNO_CCINTS_MMAP), flush its writes to
    # disk now so the pages become CLEAN/file-backed — only then can the OS
    # evict them under memory pressure (the whole point of the mmap).
    for _f, _store in results:
        _buf = getattr(_store, '_buffer', None)
        if getattr(_store, '_mmap_path', None) is not None and _buf is not None:
            _buf.flush()

    return dict(results)


def build_combined_ktilde_store(cc_ints, pair_index):
    """Pack every pair's ``K_tilde_chem_i`` / ``K_tilde_chem_j`` into one
    contiguous buffer and replace the per-pair entries with views into it.

    ``K_tilde_chem_{i,j}`` are the (n_pno, n_pno²) — i.e. n_pno³ — Term-2
    intermediates.  ``compute_C_tilde_batched`` and ``build_D_tilde_batched``
    each used to gather the per-pair (i-or-j) selection into a *retained*
    flat plan buffer, so on a TM complex K_tilde_chem was held ~4× (cc_ints
    i+j originals, plus a C_tilde copy, plus a D_tilde copy).  Packing it once
    here and pointing both builders at this single buffer (via the returned
    element offsets) collapses that to 1×.

    Returns ``{'buf', 'off_i', 'off_j'}`` where ``off_{i,j}`` are dicts
    ``{canonical_key: element_start_offset}`` into ``buf``, or ``None`` if no
    pair carries K_tilde_chem.
    """
    canon = pair_index.canonical_keys
    off_i = {}
    off_j = {}

    total = 0
    plan = []  # (key, which, size, shape)
    for key in canon:
        entry = cc_ints.get(key)
        if entry is None:
            continue
        for which, off_d in (('K_tilde_chem_i', off_i),
                             ('K_tilde_chem_j', off_j)):
            arr = entry.get(which)
            if arr is None:
                continue
            off_d[key] = total
            plan.append((key, which, arr.size, arr.shape))
            total += arr.size

    if total == 0:
        return None

    # NVMe-back the combined K_tilde_chem buffer when streaming is on: it is
    # n_pno^3 per pair, written once here and only READ by the C~/D~ builders
    # every cycle, so as file-backed pages the OS can evict it under pressure
    # (it was plain anon RAM before — a large slice of the cc_ints-era floor).
    buf = stream_empty((total,), tag='ktilde')
    # Copy each source array into the buffer, then replace the dict slot with
    # a view so the original (held only by that slot) is freed.
    for (key, which, size, shape) in plan:
        start = int(off_i[key] if which == 'K_tilde_chem_i' else off_j[key])
        entry = cc_ints[key]
        buf[start:start + size] = np.ascontiguousarray(entry[which]).ravel()
        entry[which] = buf[start:start + size].reshape(shape)
    stream_settle(buf)

    return {'buf': buf, 'off_i': off_i, 'off_j': off_j}


# ----------------------------------------------------------------------
# Consistency checker used by the lccsd.py assertion in Phase 0.  Keeps
# assertion wiring concise and off the hot path.
# ----------------------------------------------------------------------
def assert_consistent_with_dicts(
    pair_index: PairIndex,
    pno_spaces: dict,
    pair_lmo_idx: dict | None,
) -> None:
    """Raise AssertionError if ``pair_index`` disagrees with existing dicts.

    Verifies:
      * canonical_keys is exactly the sorted set of normalised keys in
        pno_spaces (with n_pno > 0).  A subset is OK — we only require
        that every canonical_key is present in pno_spaces.
      * n_pno[p] matches pno_spaces[k]['n_pno'].
      * domain_lmos[p] matches pair_lmo_idx[k] when present.
    """
    for p, k in enumerate(pair_index.canonical_keys):
        assert k in pno_spaces, f"canonical_keys has {k} absent from pno_spaces"
        expected = int(pno_spaces[k]["n_pno"])
        got = int(pair_index.n_pno[p])
        assert got == expected, (
            f"n_pno mismatch at {k}: PairIndex={got} pno_spaces={expected}"
        )
        if pair_lmo_idx is not None and k in pair_lmo_idx:
            expect_dom = np.asarray(pair_lmo_idx[k], dtype=np.int32)
            got_dom = pair_index.domain_lmos[p]
            assert np.array_equal(got_dom, expect_dom), (
                f"domain_lmos mismatch at {k}: "
                f"PairIndex={got_dom} pair_lmo_idx={expect_dom}"
            )
