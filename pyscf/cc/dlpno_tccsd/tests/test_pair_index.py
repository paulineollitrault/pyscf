"""Unit tests for PairIndex + TensorStore (Phase 0 of DLPNO restructure).

Run with::

    /environments/miniconda3/envs/tmc/bin/python -m pytest \
        pyscf/cc/dlpno_tccsd/tests/test_pair_index.py -v
"""

import numpy as np
import pytest

from pyscf.cc.dlpno_tccsd.pair_index import (
    PairIndex, TensorStore, assert_consistent_with_dicts,
)


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
