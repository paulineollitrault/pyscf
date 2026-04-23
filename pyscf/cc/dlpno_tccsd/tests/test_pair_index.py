"""Unit tests for PairIndex + TensorStore (Phase 0 of DLPNO restructure).

Run with::

    /environments/miniconda3/envs/tmc/bin/python -m pytest \
        pyscf/cc/dlpno_tccsd/tests/test_pair_index.py -v
"""

import numpy as np
import pytest

from pyscf.cc.dlpno_tccsd.pair_index import (
    PairIndex, TensorStore, FlatTensorStore,
    build_t1_cache, assert_consistent_with_dicts,
)
from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair


# ----------------------------------------------------------------------
# Fixture — small synthetic pair set
# ----------------------------------------------------------------------
def _fixture():
    """Synthetic pno_spaces + pair_lmo_idx with nocc=3."""
    pno_spaces = {
        (0, 0): {"C_pno": np.zeros((10, 3))},
        (0, 1): {"C_pno": np.zeros((10, 4))},
        (1, 1): {"C_pno": np.zeros((10, 2))},
        (0, 2): {"C_pno": np.zeros((10, 5))},
        (1, 2): {"C_pno": np.zeros((10, 3))},
        (2, 2): {"C_pno": np.zeros((10, 4))},
    }
    pair_lmo_idx = {
        (0, 0): np.array([0, 1]),
        (0, 1): np.array([0, 1, 2]),
        (1, 1): np.array([0, 1, 2]),
        (0, 2): np.array([0, 2]),
        (1, 2): np.array([1, 2]),
        (2, 2): np.array([1, 2]),
    }
    return pno_spaces, pair_lmo_idx


# ----------------------------------------------------------------------
# PairIndex
# ----------------------------------------------------------------------
def test_pair_index_round_trip():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    assert pi.n_pairs == 6
    assert pi.canonical_keys == [(0, 0), (0, 1), (0, 2),
                                 (1, 1), (1, 2), (2, 2)]
    for i, k in enumerate(pi.canonical_keys):
        assert pi.idx_of(k) == i
        assert pi.canonical_to_idx[k] == i


def test_pair_index_canonicalises_flipped_keys():
    pno_spaces, pair_lmo_idx = _fixture()
    # Feed in unsorted / partly flipped keys — should canonicalise.
    keys = [(2, 0), (1, 0), (0, 0), (1, 1), (2, 1), (2, 2)]
    pi = PairIndex(keys, pno_spaces, pair_lmo_idx, nocc=3)
    assert pi.canonical_keys == sorted({(min(k), max(k)) for k in keys})
    assert pi.idx_of((2, 0)) == pi.idx_of((0, 2))


def test_pair_index_metadata():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    # n_pno matches pno_spaces
    for p, k in enumerate(pi.canonical_keys):
        assert pi.n_pno[p] == pno_spaces[k]["C_pno"].shape[1]
    # domain_lmos matches pair_lmo_idx
    for p, k in enumerate(pi.canonical_keys):
        assert np.array_equal(pi.domain_lmos[p], pair_lmo_idx[k])
    # domain_size consistent
    assert list(pi.domain_size) == [len(d) for d in pi.domain_lmos]


def test_pair_index_falls_back_to_full_nocc():
    pno_spaces, _ = _fixture()
    # pair_lmo_idx=None → every pair's domain is arange(nocc)
    pi = PairIndex(pno_spaces.keys(), pno_spaces, None, nocc=3)
    for dom in pi.domain_lmos:
        assert np.array_equal(dom, np.arange(3, dtype=np.int32))


def test_pair_index_ordered_pairs():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    # 3 diagonals (single) + 3 off-diagonals (doubled) = 9
    assert pi.n_ordered == 9
    assert (0, 1) in pi.ordered_to_idx
    assert (1, 0) in pi.ordered_to_idx
    assert (0, 0) in pi.ordered_to_idx
    # ordered -> canonical consistency
    for o_key in pi.ordered_keys:
        canonical_expected = pi.canonical_to_idx[(min(o_key), max(o_key))]
        ordered_idx = pi.ordered_idx_of(o_key)
        assert pi.canonical_of_ordered(ordered_idx) == canonical_expected


def test_pair_index_missing_pno_raises():
    pno_spaces, _ = _fixture()
    pno_spaces_partial = {k: v for k, v in pno_spaces.items() if k != (0, 2)}
    with pytest.raises(KeyError):
        PairIndex(pno_spaces.keys(), pno_spaces_partial, None, nocc=3)


# ----------------------------------------------------------------------
# TensorStore
# ----------------------------------------------------------------------
def test_tensor_store_default_zero_fill():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    ts = TensorStore(pi)
    # Default shape: (n_pno, n_pno)
    for p, k in enumerate(pi.canonical_keys):
        n = int(pi.n_pno[p])
        assert ts[k].shape == (n, n)
        assert ts.at(p).shape == (n, n)
        assert np.all(ts[k] == 0.0)


def test_tensor_store_set_by_key_and_idx():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    ts = TensorStore(pi)
    # Set via tuple key
    ts[(0, 1)] = np.eye(4)
    assert np.allclose(ts[(0, 1)], np.eye(4))
    assert np.allclose(ts.at(pi.idx_of((0, 1))), np.eye(4))
    # Key-flipping round trip
    assert np.allclose(ts[(1, 0)], np.eye(4))
    # Set via integer idx
    ts.set_at(0, np.full((3, 3), 7.0))
    assert np.allclose(ts[(0, 0)], np.full((3, 3), 7.0))


def test_tensor_store_init_from_dict():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    seed = {(0, 0): np.ones((3, 3)), (1, 2): np.full((3, 3), 5.0)}
    ts = TensorStore(pi, init=seed)
    assert np.allclose(ts[(0, 0)], np.ones((3, 3)))
    assert np.allclose(ts[(2, 1)], np.full((3, 3), 5.0))
    # Unseeded slots absent
    assert (0, 1) not in ts
    assert (0, 2) not in ts
    # Iteration sees only present slots
    assert set(ts.keys()) == {(0, 0), (1, 2)}
    assert len(ts) == 2


def test_tensor_store_dict_compat_api():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    ts = TensorStore(pi)
    # __contains__, get with default
    assert (0, 1) in ts
    assert (9, 9) not in ts
    assert ts.get((9, 9)) is None
    assert ts.get((9, 9), "missing") == "missing"
    # items / values / keys / __iter__
    keys = list(ts.keys())
    assert len(keys) == 6
    for (k, v), kk in zip(ts.items(), ts):
        assert k == kk
        assert v is not None
    assert len(ts.values()) == 6


def test_tensor_store_custom_shape_fn():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    # Shape (domain_size, n_pno) — e.g. for T1 projections.
    ts = TensorStore(
        pi,
        shape_fn=lambda p: (int(pi.domain_size[p]), int(pi.n_pno[p])),
    )
    for p, k in enumerate(pi.canonical_keys):
        assert ts[k].shape == (int(pi.domain_size[p]), int(pi.n_pno[p]))


def test_tensor_store_no_zero_fill():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    ts = TensorStore(pi, fill_zero=False)
    assert len(ts) == 0
    for k in pi.canonical_keys:
        assert k not in ts
    # Set one
    ts[(0, 1)] = np.eye(4)
    assert len(ts) == 1
    assert (0, 1) in ts


# ----------------------------------------------------------------------
# assert_consistent_with_dicts
# ----------------------------------------------------------------------
def test_consistency_checker_passes():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    assert_consistent_with_dicts(pi, pno_spaces, pair_lmo_idx)


def test_consistency_checker_catches_n_pno_mismatch():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    # Mutate after construction
    bad = dict(pno_spaces)
    bad[(0, 1)] = {"C_pno": np.zeros((10, 99))}  # was 4
    with pytest.raises(AssertionError, match="n_pno mismatch"):
        assert_consistent_with_dicts(pi, bad, pair_lmo_idx)


def test_consistency_checker_catches_domain_mismatch():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    bad = dict(pair_lmo_idx)
    bad[(0, 1)] = np.array([0, 1])  # was [0, 1, 2]
    with pytest.raises(AssertionError, match="domain_lmos mismatch"):
        assert_consistent_with_dicts(pi, pno_spaces, bad)


# ----------------------------------------------------------------------
# build_t1_cache — must reproduce _project_t1_to_pair exactly.
# ----------------------------------------------------------------------
def _t1_cache_fixture(seed=0):
    """Fixture with realistic t1_pno and S_pno_cache populated."""
    rng = np.random.default_rng(seed)
    pno_spaces = {
        (0, 0): {"C_pno": np.zeros((10, 3))},
        (0, 1): {"C_pno": np.zeros((10, 4))},
        (1, 1): {"C_pno": np.zeros((10, 2))},
        (0, 2): {"C_pno": np.zeros((10, 5))},
        (1, 2): {"C_pno": np.zeros((10, 3))},
        (2, 2): {"C_pno": np.zeros((10, 4))},
    }
    pair_lmo_idx = {k: np.arange(3) for k in pno_spaces}
    nocc = 3

    # t1_pno keyed by LMO i; shape (n_pno[i,i],)
    t1_pno = {i: rng.standard_normal(pno_spaces[(i, i)]["C_pno"].shape[1])
              for i in range(nocc)}

    # S_pno_cache[(pair_a, pair_b)] = overlap (n_pno_a, n_pno_b).
    # Populate all (pair, (k,k)) needed for projections.
    S_pno_cache = {}
    for pair, space in pno_spaces.items():
        n_pair = space["C_pno"].shape[1]
        for k in range(nocc):
            key_kk = (k, k)
            if pair == key_kk:
                continue  # diagonal projection is identity-ish
            n_k = pno_spaces[key_kk]["C_pno"].shape[1]
            S_pno_cache[(pair, key_kk)] = rng.standard_normal((n_pair, n_k))

    return pno_spaces, pair_lmo_idx, t1_pno, S_pno_cache, nocc


def test_build_t1_cache_matches_lazy_projection():
    pno_spaces, pair_lmo_idx, t1_pno, S_pno_cache, nocc = _t1_cache_fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
    cache = build_t1_cache(t1_pno, pi, S_pno_cache, pno_spaces)

    # For every pair and every LMO, cached row must equal lazy projection.
    for pair in pno_spaces:
        n_pno = pno_spaces[pair]["C_pno"].shape[1]
        if n_pno == 0:
            continue
        for k in range(nocc):
            expected = _project_t1_to_pair(
                t1_pno, k, pair, S_pno_cache, pno_spaces)
            got = cache[pair][k]
            assert np.allclose(got, expected, atol=1e-15), (
                f"pair={pair} k={k}: "
                f"cache={got} lazy={expected}"
            )


def test_build_t1_cache_diagonal_no_copy_loop():
    """Row k of pair (k, k) should equal t1_pno[k] exactly (not a copy)."""
    pno_spaces, pair_lmo_idx, t1_pno, S_pno_cache, nocc = _t1_cache_fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
    cache = build_t1_cache(t1_pno, pi, S_pno_cache, pno_spaces)
    for k in range(nocc):
        key_kk = (k, k)
        row = cache[key_kk][k]
        assert np.array_equal(row, t1_pno[k])


def test_build_t1_cache_missing_t1_returns_zeros():
    """t1_pno[k] absent → row is zero."""
    pno_spaces, pair_lmo_idx, t1_pno, S_pno_cache, nocc = _t1_cache_fixture()
    # Drop LMO 1
    t1_pno = {0: t1_pno[0], 2: t1_pno[2]}
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
    cache = build_t1_cache(t1_pno, pi, S_pno_cache, pno_spaces)
    for pair in pno_spaces:
        if pno_spaces[pair]["C_pno"].shape[1] == 0:
            continue
        assert np.all(cache[pair][1] == 0.0), (
            f"pair={pair} LMO 1 should be zero (t1 absent)"
        )


def test_build_t1_cache_missing_overlap_returns_zeros():
    """S_pno_cache entry absent → row is zero (fallback path)."""
    pno_spaces, pair_lmo_idx, t1_pno, S_pno_cache, nocc = _t1_cache_fixture()
    pair = (0, 1)
    key_kk = (2, 2)
    S_pno_cache_bad = dict(S_pno_cache)
    del S_pno_cache_bad[(pair, key_kk)]
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
    cache = build_t1_cache(t1_pno, pi, S_pno_cache_bad, pno_spaces)
    assert np.all(cache[pair][2] == 0.0)
    # Other rows unaffected
    lazy_k0 = _project_t1_to_pair(t1_pno, 0, pair, S_pno_cache_bad, pno_spaces)
    assert np.allclose(cache[pair][0], lazy_k0)


# ----------------------------------------------------------------------
# FlatTensorStore (Phase 2a)
# ----------------------------------------------------------------------
def test_flat_tensor_store_construction():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    # Default square shape (n_pno, n_pno)
    fts = FlatTensorStore(
        pi, shape_fn=lambda p: (int(pi.n_pno[p]), int(pi.n_pno[p])),
    )
    # Buffer size matches sum of per-pair sizes
    expected_size = sum(int(n) ** 2 for n in pi.n_pno)
    assert fts.buffer.size == expected_size
    assert fts.offsets[0] == 0
    assert fts.offsets[-1] == expected_size
    assert fts._ndim == 2


def test_flat_tensor_store_at_returns_view():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    fts = FlatTensorStore(
        pi, shape_fn=lambda p: (int(pi.n_pno[p]), int(pi.n_pno[p])),
    )
    # Write through a view, verify buffer content changes.
    view = fts.at(pi.idx_of((0, 1)))
    assert view.shape == (4, 4)
    view[0, 0] = 42.0
    # Buffer should show it
    start = int(fts.offsets[pi.idx_of((0, 1))])
    assert fts.buffer[start] == 42.0
    # Read back via another at() call → same memory
    assert fts.at(pi.idx_of((0, 1)))[0, 0] == 42.0


def test_flat_tensor_store_setitem_copies():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    fts = FlatTensorStore(
        pi, shape_fn=lambda p: (int(pi.n_pno[p]), int(pi.n_pno[p])),
    )
    src = np.arange(16.0).reshape(4, 4)
    fts[(0, 1)] = src
    assert np.array_equal(fts[(0, 1)], src)
    # Mutating src does NOT affect the flat buffer
    src[0, 0] = -99.0
    assert fts[(0, 1)][0, 0] == 0.0


def test_flat_tensor_store_dict_compat():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    fts = FlatTensorStore(
        pi, shape_fn=lambda p: (int(pi.n_pno[p]), int(pi.n_pno[p])),
    )
    # keys/values/items/iter/len/contains
    assert len(fts) == 6
    assert (0, 1) in fts
    assert (2, 0) in fts  # canonicalised
    assert (9, 9) not in fts
    assert fts.get((9, 9)) is None
    assert fts.get((9, 9), "missing") == "missing"
    for k in fts:
        assert fts[k].shape == (int(pi.n_pno[pi.idx_of(k)]),) * 2
    assert len(fts.keys()) == 6
    assert len(fts.values()) == 6
    assert len(fts.items()) == 6


def test_flat_tensor_store_ragged_shapes():
    """Per-pair tensor shapes can vary across pairs as long as rank matches."""
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    # Shape (domain_size, n_pno) — different per pair
    fts = FlatTensorStore(
        pi,
        shape_fn=lambda p: (int(pi.domain_size[p]), int(pi.n_pno[p])),
    )
    for p, k in enumerate(pi.canonical_keys):
        assert fts[k].shape == (int(pi.domain_size[p]), int(pi.n_pno[p]))


def test_flat_tensor_store_uniform_rank_enforced():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    # Mix rank-1 and rank-2 shapes — should raise.
    def shape_fn(p):
        if p == 0:
            return (5,)  # rank 1
        return (3, 3)    # rank 2
    with pytest.raises(ValueError, match="uniform rank"):
        FlatTensorStore(pi, shape_fn=shape_fn)


def test_flat_tensor_store_empty_pair():
    """Zero-size pair slot returns a zero-shaped ndarray view."""
    pno_spaces = {
        (0, 0): {"C_pno": np.zeros((10, 3))},
        (0, 1): {"C_pno": np.zeros((10, 0))},  # empty
    }
    pair_lmo_idx = {(0, 0): np.array([0]), (0, 1): np.array([0, 1])}
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=2)
    fts = FlatTensorStore(
        pi, shape_fn=lambda p: (int(pi.n_pno[p]), int(pi.n_pno[p])),
    )
    assert fts[(0, 1)].shape == (0, 0)
    assert fts[(0, 0)].shape == (3, 3)


def test_flat_tensor_store_from_dict():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    src = {
        (0, 0): np.full((3, 3), 1.0),
        (1, 2): np.full((3, 3), 7.0),
    }
    fts = FlatTensorStore.from_dict(
        pi, src,
        shape_fn=lambda p: (int(pi.n_pno[p]), int(pi.n_pno[p])),
    )
    assert np.allclose(fts[(0, 0)], 1.0)
    assert np.allclose(fts[(2, 1)], 7.0)  # canonicalised
    # Unseeded slots are zeros (pre-allocated)
    assert np.all(fts[(0, 1)] == 0.0)


def test_flat_tensor_store_low_level_accessors():
    pno_spaces, pair_lmo_idx = _fixture()
    pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc=3)
    fts = FlatTensorStore(
        pi, shape_fn=lambda p: (int(pi.n_pno[p]), int(pi.n_pno[p])),
    )
    # buffer, offsets, shapes exposed and dtype-correct
    assert fts.buffer.dtype == np.float64
    assert fts.offsets.dtype == np.int64
    assert fts.offsets.shape == (pi.n_pairs + 1,)
    assert fts.shapes.shape == (pi.n_pairs, 2)
    # Offsets strictly non-decreasing
    assert np.all(np.diff(fts.offsets) >= 0)
    # Shape sum across dims × per-pair size = slot length
    for p in range(pi.n_pairs):
        slot_len = int(fts.offsets[p + 1] - fts.offsets[p])
        expected = int(np.prod(fts.shapes[p]))
        assert slot_len == expected
