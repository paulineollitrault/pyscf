"""DLPNO-CCSD monolithic solver — real-data packer.

Converts the live PySCF DLPNO-CCSD state (`cc_ints` dict, `t1_cache` dict,
`pno_spaces`, `pair_lmo_idx`, `t1_pno`, etc.) into the flat-buffer
``PySolverInputs`` layout that ``phase_t1_ints`` (and eventually all
phases) consumes.

Step 2m / real-data integration: only the subset of SolverInputs needed
for ``phase_t1_ints`` is populated for now (Qma, Qab, i_Qa/j_Qa, i_Qk/j_Qk,
T1_in_pair, sparsity arrays, F_lmo / eps_lmo / fov_flat / e_pno_flat /
T1_flat / T2_flat / pno_offsets / t2_offsets / ij_to_i_j / i_j_to_ij).

Cross-canonical S_pno_cache, K_tilde_chem_*, K_bar variants, K_iajb,
J_ij_kj, K_ij_kj, L_*, ordered-pair sparsity — left null/empty for the
t1_ints-only path.  Subsequent steps extend the packer.
"""

import ctypes
import os
import tempfile
import numpy as np

from pyscf.cc.dlpno_tccsd._ccsd_solver import (
    PySolverInputs, PyFlatPairStore,
)


def _ptr(arr):
    if arr is None:
        return None
    return arr.ctypes.data


def _build_flat_from_per_pair(per_pair_arrays, ownership):
    """Coalesce list of per-pair numpy arrays into one contiguous flat
    buffer + length-(N+1) int64 offsets.  Returns (PyFlatPairStore, flat,
    offsets).  Ownership list extended with both numpy arrays.
    """
    n = len(per_pair_arrays)
    offsets = np.zeros(n + 1, dtype=np.int64)
    for p in range(n):
        offsets[p + 1] = offsets[p] + per_pair_arrays[p].size
    _total = int(offsets[-1])
    # Page LARGE packed buffers to NVMe (DLPNO_CCINTS_MMAP).  The class pack
    # is a second copy of cc_ints; building it in RAM while cc_ints is still
    # paged in needs ~2× cc_ints resident (the OOM wall).  As a memmap, this
    # copy's pages flush/evict under pressure during the build, and the C++
    # solver reads it through the same raw pointer.  >1 GiB fields only (tiny
    # fields aren't worth a file).
    _min_bytes = float(os.environ.get('DLPNO_PACK_MMAP_MIN_MB', '1024')) * (1 << 20)
    _mmap = bool(os.environ.get('DLPNO_CCINTS_MMAP')) and _total * 8 > _min_bytes
    if _mmap:
        _tmpdir = os.environ.get('PYSCF_TMPDIR') or tempfile.gettempdir()
        _fd, _path = tempfile.mkstemp(
            suffix='.packflat', prefix='dlpno_', dir=_tmpdir)
        os.close(_fd)
        flat = np.memmap(_path, dtype=np.float64, mode='w+', shape=(_total,))
    else:
        flat = np.empty(_total, dtype=np.float64)
    for p in range(n):
        flat[offsets[p]:offsets[p + 1]] = (
            np.ascontiguousarray(per_pair_arrays[p]).ravel())
    if _mmap:
        flat.flush()   # dirty -> clean/file-backed so pages can evict
    ownership.append(flat)
    ownership.append(offsets)
    fps = PyFlatPairStore()
    fps.data = _ptr(flat)
    fps.offsets = _ptr(offsets)
    return fps, flat, offsets


def _reorder_diag_first(keys_sorted_real, nocc):
    """Return a reordered key list: (0,0), (1,1), ..., (nocc-1, nocc-1)
    first (placeholders if absent in real keys), then off-diagonals in
    natural lex order.  Required for the per-occupied / per-canonical
    aliased pno_offsets convention (see SolverInputs schema in cpp).
    """
    real_set = set(keys_sorted_real)
    diag_keys = [(i, i) for i in range(nocc)]
    offdiag = sorted([k for k in keys_sorted_real if k[0] != k[1]])
    keys_reordered = diag_keys + offdiag
    return keys_reordered, real_set


def _alias_cc_ints_field(field, cc_ints_flat, keys_sorted, c2i, ownership):
    """Zero-copy FlatPairStore aliasing the shared cc_ints_flat[field] buffer.

    The solver packs pairs in diag-first order (keys_sorted); cc_ints_flat is
    in canonical order.  We hand the C++ side the SHARED buffer as ``data``,
    a per-pack-pair ``block_start`` giving each pair's canonical position, and
    ``offsets`` = cumulative sizes in pack order (for the size arithmetic).
    No per-pair copy — eliminates the ~cc_ints-sized pack duplication.

    Returns a PyFlatPairStore, or None if the field is absent or any pair has
    no cc_ints entry (e.g. a diag placeholder) — caller then falls back to the
    copying path for correctness.
    """
    store = cc_ints_flat.get(field) if cc_ints_flat is not None else None
    if store is None:
        if os.environ.get('DLPNO_ALIAS_DEBUG'):
            print(f'  [alias {field}] FALLBACK: no store', flush=True)
        return None
    buf = store.buffer
    soff = store.offsets
    n = len(keys_sorted)
    block_start = np.empty(n, dtype=np.int64)
    sizes = np.empty(n, dtype=np.int64)
    for p, key in enumerate(keys_sorted):
        ci = c2i.get((min(key), max(key)))
        if ci is None:
            if os.environ.get('DLPNO_ALIAS_DEBUG'):
                print(f'  [alias {field}] FALLBACK: pair {key} not in c2i '
                      f'(p={p}/{n})', flush=True)
            return None
        block_start[p] = soff[ci]
        sizes[p] = soff[ci + 1] - soff[ci]
    offsets = np.zeros(n + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    ownership.append(buf)
    ownership.append(offsets)
    ownership.append(block_start)
    fps = PyFlatPairStore()
    fps.data = _ptr(buf)
    fps.offsets = _ptr(offsets)
    fps.block_start = _ptr(block_start)
    return fps


def pack_for_t1_ints(cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
                     F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
                     t2_pno_all=None, S_pno_cache=None, _pool=None,
                     cc_ints_flat=None, pair_index=None):
    """Build a PySolverInputs covering t1_ints-phase requirements only.

    Returns (inputs, ownership, key_to_p, aux) where ``ownership`` is a
    list of numpy arrays the caller MUST keep alive past the C call.
    ``key_to_p`` maps a canonical-pair key (i, j) to its index p in the
    flat buffers (matches keys_sorted iteration order).  ``aux`` holds
    references to per-pair arrays for downstream parity-comparison code.
    """
    # Reorder so diagonals (i, i) come first at indices 0..nocc-1.  This
    # is the diag-first convention assumed by SolverInputs: pno_offsets[i]
    # for occupied i must match pno_offsets[p_ii] for canonical pair (i, i).
    keys_sorted, _real_set = _reorder_diag_first(list(keys_sorted), nocc)
    n_canon = len(keys_sorted)
    key_to_p = {key: p for p, key in enumerate(keys_sorted)}

    # ------------------------------------------------------------------
    # Sparsity: pair_lmo_idx, n_pno_per_pair, ij_to_i_j, i_j_to_ij,
    # pno_offsets (per-occupied for T1/fov AND per-canon for e_pno),
    # t2_offsets.
    # ------------------------------------------------------------------
    n_pno_per_pair = np.empty(n_canon, dtype=np.int32)
    ij_to_i_j      = np.empty(2 * n_canon, dtype=np.int32)
    pair_lmo_lens  = np.empty(n_canon, dtype=np.int64)

    pair_lmo_lists = [None] * n_canon
    for p, key in enumerate(keys_sorted):
        i, j = key
        ij_to_i_j[2 * p]     = i
        ij_to_i_j[2 * p + 1] = j
        n_pno_per_pair[p]    = int(pno_spaces[key]['n_pno'])
        if pair_lmo_idx is not None and key in pair_lmo_idx:
            ll = np.asarray(pair_lmo_idx[key], dtype=np.int32)
        else:
            ll = np.arange(nocc, dtype=np.int32)
        pair_lmo_lists[p] = ll
        pair_lmo_lens[p]  = ll.size

    pair_lmo_idx_offsets = np.zeros(n_canon + 1, dtype=np.int64)
    pair_lmo_idx_offsets[1:] = np.cumsum(pair_lmo_lens)
    pair_lmo_idx_flat = np.empty(int(pair_lmo_idx_offsets[-1]), dtype=np.int32)
    for p in range(n_canon):
        pair_lmo_idx_flat[
            pair_lmo_idx_offsets[p]:pair_lmo_idx_offsets[p + 1]] = pair_lmo_lists[p]

    # i_j_to_ij: nocc x nocc, -1 if not significant.
    i_j_to_ij_2d = -np.ones((nocc, nocc), dtype=np.int32)
    for p, key in enumerate(keys_sorted):
        i, j = key
        i_j_to_ij_2d[i, j] = p
        if i != j:
            i_j_to_ij_2d[j, i] = p
    i_j_to_ij = i_j_to_ij_2d.ravel().copy()

    # ij_to_ji: each canonical pair maps to itself in our (canonical-only)
    # convention (we don't store ordered swaps as separate canonical pairs).
    ij_to_ji = np.arange(n_canon, dtype=np.int32)

    # pno_offsets (length n_canon+1; serves both per-occupied and
    # per-canon-pair indexing — see SolverInputs schema in cpp file).
    # Convention: for occupied i in [0, nocc), pno_offsets[i] is the
    # start of T1/fov for occupied i.  This works ONLY if the first nocc
    # canonical-pair indices ARE the diagonals (i, i) in order
    # i = 0, 1, ..., nocc-1.  If keys_sorted doesn't start with diagonals,
    # the per-occupied indexing breaks for downstream phases.
    #
    # For the t1_ints validation specifically, pno_offsets is only used
    # for per-canon-pair e_pno_flat offsets; per-occupied T1/fov are not
    # consumed by phase_t1_ints (T1_in_pair carries the per-pair t1).
    # So we relax the diag-first requirement here and compute
    # pno_offsets from the natural keys_sorted order.
    pno_offsets = np.zeros(n_canon + 1, dtype=np.int64)
    for p in range(n_canon):
        pno_offsets[p + 1] = pno_offsets[p] + int(n_pno_per_pair[p])

    t2_offsets = np.zeros(n_canon + 1, dtype=np.int64)
    for p in range(n_canon):
        t2_offsets[p + 1] = t2_offsets[p] + int(n_pno_per_pair[p]) ** 2

    # ------------------------------------------------------------------
    # Per-pair tensors from cc_ints: Qma, Qab, i_Qa, j_Qa, i_Qk, j_Qk.
    # Already in pair_lmo_idx-axis layout (Phase III invariant).
    # ------------------------------------------------------------------
    Qma_list  = [None] * n_canon
    Qab_list  = [None] * n_canon
    i_Qa_list = [None] * n_canon
    j_Qa_list = [None] * n_canon
    i_Qk_list = [None] * n_canon
    j_Qk_list = [None] * n_canon
    T1_in_pair_list = [None] * n_canon
    T1_in_pair_full_list = [None] * n_canon
    K_iajb_list         = [None] * n_canon
    K_bar_chem_list     = [None] * n_canon
    K_bar_ij_list       = [None] * n_canon
    K_bar_ji_list       = [None] * n_canon
    K_tilde_chem_i_list = [None] * n_canon
    K_tilde_chem_j_list = [None] * n_canon

    # Per-pair packing.  Each pair's work is independent (read-only access
    # to shared cc_ints / t1_cache / pno_spaces; writes go to per-pair list
    # slots assigned by index).  Dispatched via shared _pool when given —
    # eliminates ~1.0s serial Python on water-15 (`pack_for_t1_ints`
    # was previously the largest single-CPU stretch in Stage 5 setup).
    def _pack_one_pair(p):
        key = keys_sorted[p]
        ci = cc_ints.get(key)
        nlmo_p = int(pair_lmo_lens[p])
        npno_p = int(n_pno_per_pair[p])
        if ci is None:
            return (
                np.zeros((1, nlmo_p, npno_p), dtype=np.float64),  # Qma
                np.zeros((1, npno_p, npno_p), dtype=np.float64),  # Qab
                np.zeros((1, npno_p),       dtype=np.float64),    # i_Qa
                np.zeros((1, npno_p),       dtype=np.float64),    # j_Qa
                np.zeros((1, nlmo_p),       dtype=np.float64),    # i_Qk
                np.zeros((1, nlmo_p),       dtype=np.float64),    # j_Qk
                np.zeros((nlmo_p, npno_p),  dtype=np.float64),    # T1_in_pair
                np.zeros((nocc,  npno_p),   dtype=np.float64),    # T1_in_pair_full
                np.zeros((npno_p, npno_p),  dtype=np.float64),    # K_iajb
                np.zeros((nlmo_p, npno_p),  dtype=np.float64),    # K_bar_chem
                np.zeros((nlmo_p, npno_p),  dtype=np.float64),    # K_bar_ij
                np.zeros((nlmo_p, npno_p),  dtype=np.float64),    # K_bar_ji
                np.zeros((npno_p, npno_p * npno_p), dtype=np.float64),
                np.zeros((npno_p, npno_p * npno_p), dtype=np.float64),
            )
        ll = pair_lmo_lists[p]
        Qma = np.ascontiguousarray(ci['Qma'])
        Qab = np.ascontiguousarray(ci['Qab'])
        i_Qa = np.ascontiguousarray(ci['i_Qa'])
        j_Qa = np.ascontiguousarray(ci['j_Qa'])
        i_Qk = np.ascontiguousarray(ci['i_Qk'])
        j_Qk = np.ascontiguousarray(ci['j_Qk'])
        T1_in_pair = np.ascontiguousarray(
            t1_cache[key][np.asarray(ll, dtype=np.intp)])
        T1_in_pair_full = np.ascontiguousarray(t1_cache[key])
        K_iajb = (np.ascontiguousarray(ci['K_iajb'])
                  if 'K_iajb' in ci
                  else np.zeros((npno_p, npno_p), dtype=np.float64))
        K_bar_chem = (np.ascontiguousarray(ci['K_bar_chem'])
                      if 'K_bar_chem' in ci
                      else np.zeros((nlmo_p, npno_p), dtype=np.float64))
        K_bar_ij = (np.ascontiguousarray(ci['K_bar_ij'])
                    if 'K_bar_ij' in ci
                    else np.zeros((nlmo_p, npno_p), dtype=np.float64))
        K_bar_ji = (np.ascontiguousarray(ci['K_bar_ji'])
                    if 'K_bar_ji' in ci
                    else np.zeros((nlmo_p, npno_p), dtype=np.float64))
        K_tilde_chem_i = (np.ascontiguousarray(ci['K_tilde_chem_i'])
                          if 'K_tilde_chem_i' in ci
                          else np.zeros((npno_p, npno_p * npno_p),
                                         dtype=np.float64))
        K_tilde_chem_j = (np.ascontiguousarray(ci['K_tilde_chem_j'])
                          if 'K_tilde_chem_j' in ci
                          else np.zeros((npno_p, npno_p * npno_p),
                                         dtype=np.float64))
        return (Qma, Qab, i_Qa, j_Qa, i_Qk, j_Qk,
                T1_in_pair, T1_in_pair_full, K_iajb,
                K_bar_chem, K_bar_ij, K_bar_ji,
                K_tilde_chem_i, K_tilde_chem_j)

    # Per-pair packing: serial. The work is mostly numpy view checks via
    # ascontiguousarray (~4 ms total at water-15) — pool dispatch overhead
    # dominates parallel attempts, so we keep this straight Python.
    _results = [_pack_one_pair(p) for p in range(n_canon)]
    for p, r in enumerate(_results):
        (Qma_list[p], Qab_list[p], i_Qa_list[p], j_Qa_list[p],
         i_Qk_list[p], j_Qk_list[p],
         T1_in_pair_list[p], T1_in_pair_full_list[p],
         K_iajb_list[p], K_bar_chem_list[p],
         K_bar_ij_list[p], K_bar_ji_list[p],
         K_tilde_chem_i_list[p], K_tilde_chem_j_list[p]) = r

    # ------------------------------------------------------------------
    # Scalar / per-occupied buffers: F_lmo, eps_lmo, foo (zero), fov_flat,
    # e_pno_flat, T1_flat, T2_flat.
    # ------------------------------------------------------------------
    F_lmo_arr   = np.ascontiguousarray(F_lmo, dtype=np.float64)
    eps_lmo_arr = np.ascontiguousarray(eps_lmo, dtype=np.float64)
    foo_arr     = np.zeros((nocc, nocc), dtype=np.float64)  # not used by t1_ints

    # fov_flat / T1_flat: per-occupied i, length npno_ii doubles.  These
    # depend on pno_offsets[i] being the right start for occupied i —
    # which requires diag-first ordering.  Here we punt: build them
    # respecting whatever pno_offsets[i] returns (i.e., the i-th canonical
    # pair's npno).  For phase_t1_ints validation, fov_flat / T1_flat are
    # not read.
    e_pno_flat_total = int(pno_offsets[-1])
    e_pno_flat = np.zeros(e_pno_flat_total, dtype=np.float64)
    for p, key in enumerate(keys_sorted):
        e_pno_flat[pno_offsets[p]:pno_offsets[p + 1]] = (
            np.asarray(pno_spaces[key]['e_pno'], dtype=np.float64))

    # fov_flat / T1_flat are per-occupied i ∈ [0, nocc) under diag-first
    # convention.  pno_offsets[i] is the start of occupied i's block.
    fov_flat = np.zeros(e_pno_flat_total, dtype=np.float64)
    T1_flat  = np.zeros(e_pno_flat_total, dtype=np.float64)
    for i in range(nocc):
        npno_ii = int(n_pno_per_pair[i])    # diag-first ⇒ index i = pair (i, i)
        if npno_ii == 0:
            continue
        i_off = int(pno_offsets[i])
        if i in t1_pno and t1_pno[i].size == npno_ii:
            T1_flat[i_off:i_off + npno_ii] = np.asarray(t1_pno[i],
                                                          dtype=np.float64)
        if i in fov_pno and fov_pno[i].size == npno_ii:
            fov_flat[i_off:i_off + npno_ii] = np.asarray(fov_pno[i],
                                                          dtype=np.float64)

    t2_total = int(t2_offsets[-1])
    T2_flat  = np.zeros(t2_total, dtype=np.float64)
    if t2_pno_all is not None:
        for p, key in enumerate(keys_sorted):
            t2_p = t2_pno_all.get(key)
            if t2_p is None:
                continue
            T2_flat[t2_offsets[p]:t2_offsets[p + 1]] = (
                np.ascontiguousarray(t2_p, dtype=np.float64).ravel())

    # Stub i_j_to_ij arrays etc. into ownership.
    ownership = [
        n_pno_per_pair, ij_to_i_j, ij_to_ji,
        pair_lmo_idx_flat, pair_lmo_idx_offsets,
        pno_offsets, t2_offsets, i_j_to_ij,
        F_lmo_arr, eps_lmo_arr, foo_arr,
        fov_flat, e_pno_flat, T1_flat, T2_flat,
    ]

    inputs = PySolverInputs()
    inputs.nocc           = int(nocc)
    inputs.nlmo           = int(nocc)
    inputs.n_canon_pairs  = int(n_canon)
    inputs.n_strong_pairs = int(n_canon)
    inputs.diis_max_vecs  = 0
    inputs.max_cycle      = 0
    inputs.e_conv         = 0.0
    inputs.r_conv         = 0.0

    inputs.i_j_to_ij            = _ptr(i_j_to_ij)
    inputs.ij_to_i_j            = _ptr(ij_to_i_j)
    inputs.ij_to_ji             = _ptr(ij_to_ji)
    inputs.pair_lmo_idx_flat    = _ptr(pair_lmo_idx_flat)
    inputs.pair_lmo_idx_offsets = _ptr(pair_lmo_idx_offsets)
    inputs.n_pno_per_pair       = _ptr(n_pno_per_pair)
    inputs.pno_offsets          = _ptr(pno_offsets)
    inputs.t2_offsets           = _ptr(t2_offsets)

    inputs.F_lmo      = _ptr(F_lmo_arr)
    inputs.eps_lmo    = _ptr(eps_lmo_arr)
    inputs.foo        = _ptr(foo_arr)
    inputs.fov_flat   = _ptr(fov_flat)
    inputs.e_pno_flat = _ptr(e_pno_flat)

    inputs.T1_flat = _ptr(T1_flat)
    inputs.T2_flat = _ptr(T2_flat)

    # Per-pair FlatPairStores.  cc_ints-backed fields are ALIASED zero-copy
    # from the shared cc_ints_flat buffer (no second copy); others copy.
    _c2i = pair_index.canonical_to_idx if pair_index is not None else None

    _alias_allow = os.environ.get('DLPNO_ALIAS_FIELDS')
    _alias_set = (set(_alias_allow.split(',')) if _alias_allow else None)

    def _F(field, lst):
        _ok = (_c2i is not None
               and (_alias_set is None or field in _alias_set))
        fps = (_alias_cc_ints_field(field, cc_ints_flat, keys_sorted,
                                    _c2i, ownership) if _ok else None)
        if fps is not None:
            return fps
        return _build_flat_from_per_pair(lst, ownership)[0]

    Qma_fps  = _F('Qma',  Qma_list)
    Qab_fps  = _F('Qab',  Qab_list)
    i_Qk_fps = _F('i_Qk', i_Qk_list)
    j_Qk_fps = _F('j_Qk', j_Qk_list)
    i_Qa_fps = _F('i_Qa', i_Qa_list)
    j_Qa_fps = _F('j_Qa', j_Qa_list)
    T1_in_pair_fps, T1_in_pair_flat_arr, T1_in_pair_offs = (
        _build_flat_from_per_pair(T1_in_pair_list, ownership))
    K_iajb_fps     = _F('K_iajb',     K_iajb_list)
    K_bar_chem_fps = _F('K_bar_chem', K_bar_chem_list)
    K_bar_ij_fps   = _F('K_bar_ij',   K_bar_ij_list)
    K_bar_ji_fps   = _F('K_bar_ji',   K_bar_ji_list)
    K_tilde_chem_i_fps, _, _ = _build_flat_from_per_pair(
        K_tilde_chem_i_list, ownership)
    K_tilde_chem_j_fps, _, _ = _build_flat_from_per_pair(
        K_tilde_chem_j_list, ownership)

    inputs.Qma    = Qma_fps
    inputs.Qab    = Qab_fps
    inputs.i_Qk   = i_Qk_fps
    inputs.j_Qk   = j_Qk_fps
    inputs.i_Qa   = i_Qa_fps
    inputs.j_Qa   = j_Qa_fps
    inputs.T1_in_pair = T1_in_pair_fps
    T1_in_pair_full_fps, T1_in_pair_full_flat_arr, T1_in_pair_full_offs = (
        _build_flat_from_per_pair(T1_in_pair_full_list, ownership))
    inputs.T1_in_pair_full = T1_in_pair_full_fps
    inputs.K_iajb = K_iajb_fps
    inputs.K_bar_chem     = K_bar_chem_fps
    inputs.K_bar_ij       = K_bar_ij_fps
    inputs.K_bar_ji       = K_bar_ji_fps
    inputs.K_tilde_chem_i = K_tilde_chem_i_fps
    inputs.K_tilde_chem_j = K_tilde_chem_j_fps

    # Fields not yet needed — leave as null pointers.
    for fname in ('J_ij_kj', 'K_ij_kj', 'L_iajb', 'L_bar'):
        empty_fps = PyFlatPairStore()
        empty_fps.data = None
        empty_fps.offsets = None
        setattr(inputs, fname, empty_fps)

    # Cross-canonical S_pno_cache (full-table layout, n_canon² blocks).
    # For (p_a, p_b) pairs not in the cache, store zero-size entries
    # (zero-magnitude data; offsets equal => kernel skips contribution).
    if S_pno_cache is not None:
        n_blocks = n_canon * n_canon
        S_pno_offsets = np.zeros(n_blocks + 1, dtype=np.int64)
        S_blocks = []
        # Sparse layout: only iterate the keys actually present in
        # S_pno_cache. The dense (n_canon × n_canon) loop visited 854² =
        # 729k pairs on water-15, of which only ~184k are populated; the
        # rest were dict-misses with serial Python overhead, dominating
        # PACK-ONCE setup. We now batch the entries in cache-key order
        # and let the outer flat-offsets fill from the sparse hits only.
        if hasattr(S_pno_cache, 'iter_keys'):
            cache_iter = S_pno_cache.iter_keys()
        else:
            cache_iter = list(S_pno_cache.keys()) if hasattr(
                S_pno_cache, 'keys') else []
        # SPARSE S_PNO (dedup): alias the S_pno_cache's own buffer (slot order)
        # and hand the C++ a dense (p_a*N+p_b -> slot) index, instead of
        # building a SECOND dense copy of all overlaps (~73 GiB on a TM
        # complex).  per_kl already shares this cache buffer; this makes the
        # main residual share it too -> one S_PNO copy total -> fits in RAM ->
        # no per-cycle paging.  C++ reads via s_pno_lookup(S_pno_index).
        _sparse_ok = (hasattr(S_pno_cache, '_buffer')
                      and hasattr(S_pno_cache, '_idx_matrix')
                      and hasattr(S_pno_cache, '_offsets')
                      and hasattr(S_pno_cache, '_pi'))
        if _sparse_ok:
            _cache_c2i = S_pno_cache._pi.canonical_to_idx
            _cache_idxm = S_pno_cache._idx_matrix
            S_pno_index = np.full(n_blocks, -1, dtype=np.int32)
            for cache_key in cache_iter:
                key_a, key_b = cache_key
                p_a = key_to_p.get(key_a, -1)
                p_b = key_to_p.get(key_b, -1)
                if p_a < 0 or p_b < 0:
                    continue
                ia = _cache_c2i.get(key_a)
                ib = _cache_c2i.get(key_b)
                if ia is None or ib is None:
                    continue
                slot = int(_cache_idxm[ia, ib])
                if slot < 0:
                    continue
                S_pno_index[p_a * n_canon + p_b] = slot
            _cache_buf = S_pno_cache._buffer
            _cache_off = np.ascontiguousarray(S_pno_cache._offsets,
                                              dtype=np.int64)
            if os.environ.get('DLPNO_PACK_PROBE'):
                print(f'  [pack_probe] S_pno SPARSE alias '
                      f'(cache buffer {_cache_buf.nbytes/2**30:.1f} GiB, '
                      f'n_slots={int(np.sum(S_pno_index >= 0))}) '
                      f'— no dense copy', flush=True)
            ownership.append(_cache_buf)
            ownership.append(_cache_off)
            ownership.append(S_pno_index)
            inputs.S_pno_data    = _ptr(_cache_buf)
            inputs.S_pno_offsets = _ptr(_cache_off)
            inputs.S_pno_index   = _ptr(S_pno_index)
        else:
            # Legacy dense build (cache without flat buffer / idx_matrix).
            sizes = np.zeros(n_blocks, dtype=np.int64)
            kv_pairs = []
            for cache_key in cache_iter:
                key_a, key_b = cache_key
                p_a = key_to_p.get(key_a, -1)
                p_b = key_to_p.get(key_b, -1)
                if p_a < 0 or p_b < 0:
                    continue
                S_ab = S_pno_cache.get(cache_key)
                if S_ab is None:
                    continue
                expected = int(n_pno_per_pair[p_a]) * int(n_pno_per_pair[p_b])
                if int(S_ab.size) != expected:
                    continue
                idx = p_a * n_canon + p_b
                sizes[idx] = expected
                kv_pairs.append((idx, S_ab))
            S_pno_offsets[1:] = np.cumsum(sizes)
            _total = int(S_pno_offsets[-1])
            S_pno_data = np.empty(_total, dtype=np.float64)
            for idx, S_ab in kv_pairs:
                o = int(S_pno_offsets[idx])
                blk = np.ascontiguousarray(S_ab, dtype=np.float64).ravel()
                S_pno_data[o:o + blk.size] = blk
            ownership.append(S_pno_data)
            ownership.append(S_pno_offsets)
            inputs.S_pno_data    = _ptr(S_pno_data)
            inputs.S_pno_offsets = _ptr(S_pno_offsets)
            inputs.S_pno_index   = None
    else:
        inputs.S_pno_data    = None
        inputs.S_pno_offsets = None
        inputs.S_pno_index   = None

    # Ordered pairs: (a, b) and (b, a) for each canonical off-diagonal,
    # (i, i) once for diagonals.  Mirrors Psi4 all_pairs.
    ordered_i_list = []
    ordered_k_list = []
    for (a, b) in keys_sorted:
        ordered_i_list.append(a)
        ordered_k_list.append(b)
        if a != b:
            ordered_i_list.append(b)
            ordered_k_list.append(a)
    ordered_pair_i_idx = np.array(ordered_i_list, dtype=np.int32)
    ordered_pair_k_idx = np.array(ordered_k_list, dtype=np.int32)
    ownership.append(ordered_pair_i_idx)
    ownership.append(ordered_pair_k_idx)
    inputs.n_ordered_pairs    = int(ordered_pair_i_idx.size)
    inputs.ordered_pair_i_idx = _ptr(ordered_pair_i_idx)
    inputs.ordered_pair_k_idx = _ptr(ordered_pair_k_idx)

    inputs.n_cas_blocks      = 0
    inputs.cas_block_pair    = None
    inputs.cas_block_offsets = None
    inputs.cas_block_data    = None
    inputs.cas_block_slice   = None

    # Optional Psi4-faithful overrides (full strong+weak scope).
    # Default null; populate via separate helpers when needed.
    inputs.Fij_bar_full = None
    fkc_fps = PyFlatPairStore()
    fkc_fps.data = None
    fkc_fps.offsets = None
    inputs.Fkc_per_ordered = fkc_fps
    inputs.R2_external = None
    inputs.is_strong_pair = None

    # NOTE: the per-pair ndarray lists (Qma_list, i_Qa_list, ..., T1_in_pair_*
    # _list) are intentionally NOT stored in aux.  The class cycle loop reads
    # only the flat/aliased stores + metadata below — never the per-pair lists
    # (verified in _ccsd_solver.py:run_remaining_cycles_via_class).  The
    # aliased lists were views into cc_ints_flat (free), but T1_in_pair_list /
    # T1_in_pair_full_list were real copies; dropping all of them lets them GC
    # as soon as pack_for_t1_ints returns instead of being pinned for the whole
    # class phase.
    aux = {
        'keys_sorted': list(keys_sorted),
        'key_to_p': key_to_p,
        'pair_lmo_lists': pair_lmo_lists,
        'n_pno_per_pair': n_pno_per_pair,
        'T1_in_pair_flat': T1_in_pair_flat_arr,
        'T1_in_pair_offs': T1_in_pair_offs,
        'T1_in_pair_full_flat': T1_in_pair_full_flat_arr,
        'T1_in_pair_full_offs': T1_in_pair_full_offs,
        't2_offsets': t2_offsets,
        'T2_flat': T2_flat,
        'pno_offsets': pno_offsets,
        'fov_flat': fov_flat,
        'T1_flat': T1_flat,
        'ordered_pair_i_idx': ordered_pair_i_idx,
        'ordered_pair_k_idx': ordered_pair_k_idx,
        'i_j_to_ij': i_j_to_ij_2d.copy(),  # (nocc, nocc)
    }
    return inputs, ownership, key_to_p, aux
