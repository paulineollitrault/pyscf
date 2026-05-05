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

        # n_pno[p] from pno_spaces[k]['C_pno'].shape[1].
        self.n_pno = np.empty(self.n_pairs, dtype=np.int32)
        for p, k in enumerate(self.canonical_keys):
            space = pno_spaces.get(k)
            if space is None:
                raise KeyError(f"pno_spaces missing key {k}")
            self.n_pno[p] = int(space["C_pno"].shape[1])

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
                 "_views", "_canon_to_idx")

    def __init__(self, pair_index, shape_fn, dtype=np.float64):
        self._pi = pair_index
        self.dtype = np.dtype(dtype)
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
        self._buffer = np.zeros(int(self._offsets[-1]), dtype=self.dtype)

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
        "_views", "_canon_to_idx",
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

        # Resolve initial entries to (idx_a, idx_b) preserving order.
        entries = []
        if initial is not None:
            for (pa, pb), arr in initial.items():
                pa_norm = (min(pa), max(pa))
                pb_norm = (min(pb), max(pb))
                ia = pair_index.canonical_to_idx.get(pa_norm)
                ib = pair_index.canonical_to_idx.get(pb_norm)
                if ia is None or ib is None:
                    continue
                entries.append(((ia, ib), arr))

        n = len(entries)
        self._n_flat = n
        self._shapes = np.zeros((n, 2), dtype=np.int32)
        sizes = np.zeros(n, dtype=np.int64)
        for k, ((ia, ib), arr) in enumerate(entries):
            shape = arr.shape
            if len(shape) != 2:
                raise ValueError(
                    f"FlatPairPairStore expects rank-2 entries; "
                    f"got shape {shape} for ({ia},{ib})"
                )
            self._shapes[k, 0] = shape[0]
            self._shapes[k, 1] = shape[1]
            sizes[k] = shape[0] * shape[1]

        self._offsets = np.zeros(n + 1, dtype=np.int64)
        self._offsets[1:] = np.cumsum(sizes)
        self._buffer = np.empty(int(self._offsets[-1]), dtype=self.dtype)

        # (n_pairs, n_pairs) int32 dense lookup replaces the slow tuple
        # dict.  ``-1`` means "not in flat tier".  Memory cost is
        # ``4 * n_pairs**2`` bytes (~1 MB at n_pairs = 528) — cheap.
        n_pairs = pair_index.n_pairs
        self._idx_matrix = np.full((n_pairs, n_pairs), -1, dtype=np.int32)
        for k, ((ia, ib), arr) in enumerate(entries):
            start = int(self._offsets[k])
            end = int(self._offsets[k + 1])
            if end > start:
                self._buffer[start:end] = arr.ravel()
            self._idx_matrix[ia, ib] = k

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

    return dict(results)


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
      * n_pno[p] matches pno_spaces[k]['C_pno'].shape[1].
      * domain_lmos[p] matches pair_lmo_idx[k] when present.
    """
    for p, k in enumerate(pair_index.canonical_keys):
        assert k in pno_spaces, f"canonical_keys has {k} absent from pno_spaces"
        expected = int(pno_spaces[k]["C_pno"].shape[1])
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
