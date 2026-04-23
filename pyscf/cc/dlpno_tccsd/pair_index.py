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
) -> TensorStore:
    """Pre-project t1 into every pair's PNO basis, one row per LMO.

    Semantics match ``_project_t1_to_pair``:
      - If ``t1_pno[k]`` is absent / empty → row is zero.
      - If ``pair == (k, k)`` → row is ``t1_pno[k]`` (no projection).
      - Else if ``S_pno_cache[(pair, (k,k))]`` is present → row is
        ``S @ t1_pno[k]``.
      - Else (S missing) → row is zero (fallback).

    Returns
    -------
    TensorStore
        Indexed by canonical pair key.  Slot ``p`` holds a contiguous
        ``(nocc, n_pno[p])`` float64 array.
    """
    nocc = pair_index.nocc
    cache = TensorStore(
        pair_index,
        shape_fn=lambda p: (nocc, int(pair_index.n_pno[p])),
    )
    for p, pair_key in enumerate(pair_index.canonical_keys):
        n_pno_p = int(pair_index.n_pno[p])
        if n_pno_p == 0:
            continue
        out = cache.at(p)  # (nocc, n_pno_p), zero-initialised
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
