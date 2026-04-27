"""DLPNO-CCSD residual and T1-dressed intermediates.

Implements the DLPNO-CCSD T2 residual (Jiang et al. JCP 2024, Eqs 75-86)
as a Psi4-compatible two-buffer formulation (R + P̂(Rn)) with all T1
dressing via explicit intermediates (B_tilde, C_tilde, D_tilde, G_tilde,
Fab, Fkj). All integrals are BARE (computed from undressed C_lmo); T1
enters exclusively through the dressed intermediates built here.
"""
import contextlib
import os
import numpy as np
from pyscf.ao2mo import _ao2mo
from pyscf.cc.dlpno_tccsd.lccsd import (
    _project_t1_to_pair, _compute_ladder, _project_t2_full,
)
from pyscf.cc.dlpno_tccsd.local_df import (
    get_local_K, get_local_ovL, get_local_ooL_vec,
)


def _omp_threads_ctx(n_threads):
    """Context manager that boosts OpenMP to ``n_threads`` for its scope.

    The CCSD driver imports numpy / pyscf under ``OMP_NUM_THREADS=1`` so
    that Python-pool workers each hold a single BLAS/OMP thread.  Our
    nogil-prange Cython kernels run from the main thread (outside the
    pool) and want every core; scoping ``user_api='openmp'`` lifts the
    cap for the kernel call only, leaving pool workers unaffected.

    ``n_threads=None`` or threadpoolctl-missing returns a no-op CM so
    callers don't need ``if``-ladders.
    """
    if n_threads is None:
        return contextlib.nullcontext()
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return contextlib.nullcontext()
    return threadpool_limits(limits=int(n_threads), user_api='openmp')


def _chunked_map(pool, fn, items, chunks_per_worker=4):
    """Dispatch many small tasks to a ThreadPoolExecutor in chunks.

    ThreadPoolExecutor.map has ~5-50 us of dispatch overhead per task,
    which dominates when tasks are <1 ms. Chunking into N_workers*4 groups
    collapses overhead into ~N_workers dispatches while keeping load
    balancing healthy (4 chunks per worker).

    Yields results in submission order, matching pool.map semantics.
    """
    if pool is None:
        for it in items:
            yield fn(it)
        return
    items = list(items)
    n = len(items)
    if n == 0:
        return
    workers = getattr(pool, '_max_workers', 1)
    chunk_size = max(1, (n + workers * chunks_per_worker - 1)
                        // (workers * chunks_per_worker))

    def _process_chunk(chunk):
        return [fn(x) for x in chunk]

    chunks = [items[i:i + chunk_size] for i in range(0, n, chunk_size)]
    for chunk_result in pool.map(_process_chunk, chunks):
        yield from chunk_result


# =========================================================================
# T1-dressed intermediates: G_tilde, D_tilde, Fab, Fkj, Fia_bar (Eqs 82-86)
# =========================================================================



def build_G_tilde(t2_pno_all, t1_pno, pno_spaces, nocc,
                  ovL_bare, ooL_bare, S_pno_cache,
                  Fkj, foo_t1, cc_ints=None,
                  S_pao_full=None, s1e=None, _pool=None):
    """Build G_tilde (Eq 86): double-dressed Fock oo.

    G_tilde[k,j] = F̃_{kj} + Σ_l u_lj × K_il (bare exchange)

    Psi4 ccsd.cc lines 1820-1844.

    Args:
        Fkj: (nocc, nocc) = F̃_{kj} = dressed Fock oo (from build_Fkj)
        foo_t1: (nocc, nocc) = T1 correction to Fock (from _compute_foo_t1)
        _pool: optional ThreadPoolExecutor. The outer-i loop is
            dispatched across it with disjoint row writes — `row_contrib`
            is returned per-i and scattered into G in the main thread.
    """
    G = Fkj.copy()
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    # ------------------------------------------------------------------
    # Path-(b) batched G_tilde: precompute per-triple `effective` tensor
    # so each iteration is one ddot per triple. Plan structure mirrors
    # Psi4 compute_G_tilde (ccsd.cc:1943): outer prange over (i, j) slots
    # in [0, naocc²), inner serial accumulation over l. The prior SERIAL
    # path with j-batched tensordot stays as a fallback for diagnostics.
    # ------------------------------------------------------------------
    from pyscf.cc.dlpno_tccsd._g_tilde_batched_cy import g_tilde_batched

    plan_key = (id(cc_ints), id(t2_pno_all),
                id(S_pno_cache), id(pno_spaces))
    plan = getattr(build_G_tilde, '_batched_plan', None)
    if plan is None or plan.get('key') != plan_key:
        # Enumerate canonical pairs touched by t2_pno_all and assign
        # contiguous indices for T2 buffer access.
        canonical_pair_idx = {}
        canonical_pairs = []
        for key in t2_pno_all:
            if key not in canonical_pair_idx:
                t2 = t2_pno_all[key]
                if t2 is None or t2.shape[0] == 0:
                    continue
                canonical_pair_idx[key] = len(canonical_pairs)
                canonical_pairs.append(key)

        # Per-canonical-pair n_pno + offsets for T2 flat buffer.
        n_canon = len(canonical_pairs)
        T2_sizes = np.empty(n_canon, dtype=np.int64)
        canon_n_pno = np.empty(n_canon, dtype=np.int32)
        for p, key in enumerate(canonical_pairs):
            n_pno = pno_spaces[key]['C_pno'].shape[1]
            canon_n_pno[p] = n_pno
            T2_sizes[p] = n_pno * n_pno
        T2_offsets = np.empty(n_canon + 1, dtype=np.int64)
        T2_offsets[0] = 0
        T2_offsets[1:] = np.cumsum(T2_sizes)

        # Cache K_proj_static per UNIQUE (key_il, key_lj, i, l) tuple so
        # different (i, j) outer slots that share these inputs reuse the
        # static computation. K_proj depends on (key_il, key_lj) pair-pair
        # and (i, l) — the LMO indices used to slice K_il = (Qma_i.T @ Qma_l).
        # Two of the four orientation/transpose variants are needed:
        #   case_le: 2*K_proj_T - K_proj   (used when l <= j)
        #   case_gt: 2*K_proj   - K_proj_T (used when l >  j)
        kproj_cache = {}  # (key_il, key_lj, i, l) -> (case_le_flat, case_gt_flat)

        # Enumerate triples grouped by outer (i, j) slot.
        ij_slots = []          # list of (i, j)
        ij_triple_offsets = [0]
        triple_eff_list = []
        triple_n_lj = []
        triple_T2_pair_idx = []

        for i in range(nocc):
            for j in range(nocc):
                for l in range(nocc):
                    key_il = (min(i, l), max(i, l))
                    if key_il not in t2_pno_all:
                        continue
                    t2_il = t2_pno_all[key_il]
                    if t2_il is None or t2_il.shape[0] == 0:
                        continue
                    key_lj = (min(l, j), max(l, j))
                    if key_lj not in canonical_pair_idx:
                        continue
                    n_lj = pno_spaces[key_lj]['C_pno'].shape[1]

                    cache_key = (key_il, key_lj, i, l)
                    pair_kproj = kproj_cache.get(cache_key)
                    if pair_kproj is None:
                        K_il = get_local_K(cc_ints, key_il, i, l)
                        if K_il is None:
                            continue
                        if key_il == key_lj:
                            K_proj = K_il   # self-pair: S_il_lj = I
                        else:
                            S_il_lj = _s_pno_get(key_il, key_lj)
                            if S_il_lj is None:
                                continue
                            K_proj = S_il_lj.T @ K_il @ S_il_lj
                        K_proj_T = K_proj.T
                        case_le = np.ascontiguousarray(
                            2.0 * K_proj_T - K_proj)
                        case_gt = np.ascontiguousarray(
                            2.0 * K_proj - K_proj_T)
                        pair_kproj = (case_le, case_gt)
                        kproj_cache[cache_key] = pair_kproj

                    case_le, case_gt = pair_kproj
                    eff = case_le if l <= j else case_gt
                    triple_eff_list.append(eff)
                    triple_n_lj.append(n_lj)
                    triple_T2_pair_idx.append(canonical_pair_idx[key_lj])

                ij_slots.append((i, j))
                ij_triple_offsets.append(len(triple_eff_list))

        if not ij_slots:
            # Nothing to do — Fkj copy is the answer.
            build_G_tilde._batched_plan = {'key': plan_key, 'empty': True}
            return G

        # Flatten effective per-triple buffers
        eff_sizes = np.array([e.size for e in triple_eff_list], dtype=np.int64)
        eff_offsets = np.empty(len(triple_eff_list) + 1, dtype=np.int64)
        eff_offsets[0] = 0
        eff_offsets[1:] = np.cumsum(eff_sizes)
        effective_flat = np.empty(int(eff_offsets[-1]))
        for k, e in enumerate(triple_eff_list):
            effective_flat[eff_offsets[k]:eff_offsets[k + 1]] = e.ravel()

        ij_i_arr = np.array([s[0] for s in ij_slots], dtype=np.int32)
        ij_j_arr = np.array([s[1] for s in ij_slots], dtype=np.int32)
        ij_triple_starts = np.array(ij_triple_offsets, dtype=np.int64)
        triple_eff_off_arr = eff_offsets[:-1].astype(np.int64)
        triple_n_lj_arr = np.array(triple_n_lj, dtype=np.int32)
        triple_T2_pair_idx_arr = np.array(triple_T2_pair_idx, dtype=np.int64)

        plan = {
            'key': plan_key,
            'empty': False,
            'canonical_pairs': canonical_pairs,
            'canonical_pair_idx': canonical_pair_idx,
            'canon_n_pno': canon_n_pno,
            'T2_offsets': T2_offsets,
            'T2_total': int(T2_offsets[-1]),
            'ij_i_arr': ij_i_arr,
            'ij_j_arr': ij_j_arr,
            'ij_triple_starts': ij_triple_starts,
            'effective_flat': effective_flat,
            'triple_eff_offset': triple_eff_off_arr,
            'triple_n_lj': triple_n_lj_arr,
            'triple_T2_pair_idx': triple_T2_pair_idx_arr,
            'num_threads': min(32, len(ij_slots)) if ij_slots else 1,
        }
        build_G_tilde._batched_plan = plan

    if plan.get('empty'):
        return G

    # Per-iter: build T2 flat buffer in canonical pair order.
    T2_flat = np.empty(plan['T2_total'])
    T2_offsets = plan['T2_offsets']
    for p, key in enumerate(plan['canonical_pairs']):
        t2 = t2_pno_all[key]
        T2_flat[T2_offsets[p]:T2_offsets[p + 1]] = t2.ravel()

    # G_addition is what the kernel adds onto G. Pass G itself; kernel does
    # G[i, j] += sum_ij in place. (G already initialized to Fkj copy above.)
    g_tilde_batched(
        plan['triple_eff_offset'], plan['triple_T2_pair_idx'],
        plan['triple_n_lj'],
        plan['ij_triple_starts'], plan['ij_i_arr'], plan['ij_j_arr'],
        plan['effective_flat'],
        T2_flat, plan['T2_offsets'],
        G,
        plan['num_threads'],
    )

    # GTILDE_DUMP: parity dump vs Psi4 ccsd.cc:1943 compute_G_tilde.
    # G is the full (nocc, nocc) double-dressed Fock oo. Track per-call
    # iteration count to dump only the first 3 iterations.
    if int(os.environ.get('DLPNO_DUMP_GTILDE', '0')):
        _it = getattr(build_G_tilde, '_iter', 0)
        build_G_tilde._iter = _it + 1
        if _it <= 2:
            rms = float(np.sqrt((G ** 2).mean()))
            sm = float(G.sum())
            tr = float(np.trace(G))
            fro = float(np.linalg.norm(G, 'fro'))
            # Off-diag mass (i != j)
            off = G - np.diag(np.diag(G))
            off_fro = float(np.linalg.norm(off, 'fro'))
            print(f"GTILDE_DUMP iter={_it} nocc={G.shape[0]} "
                  f"rms={rms:.12e} sum={sm:.12e} tr={tr:.12e} "
                  f"fro={fro:.12e} off_fro={off_fro:.12e}",
                  flush=True)

    return G


# =========================================================================
# G-term (Eq 81 Fock-oo coupling in T2 residual) — batched across pairs
# =========================================================================

def _build_g_term_plan(strong_keys, pno_spaces, pair_lmo_idx,
                      t2_pno_all, S_pno_cache, nocc):
    """One-time plan for compute_G_term_batched.

    Enumerates every (ij, k) item — ij ∈ strong_keys, k ∈ ij's LMO
    domain, both (i,k) and (j,k) partner pairs existing in t2_pno_all.
    Buckets items by ``(n_ij, n_ik)`` so each bucket is a uniform
    batched matmul.

    Plan is iteration-independent: t2 values change each cycle, but S,
    bucket assignments, and scatter indices are static.  Built once per
    CCSD run, reused every cycle.
    """
    # Output slots: one per strong ij, grouped by n_ij.
    pairs_by_n_ij = {}
    pair_to_slot = {}
    for key_ij in strong_keys:
        n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_ij == 0:
            continue
        pairs_by_n_ij.setdefault(n_ij, []).append(key_ij)
        pair_to_slot[key_ij] = len(pairs_by_n_ij[n_ij]) - 1

    # Collect items per side.  Each item: (key_ij, key_pk, k, side,
    # scalar_other_lmo, transpose_flag, n_ij, n_pk, S, pair_slot).
    # side = 0 -> ik direction (G_ij), side = 1 -> jk direction (G_ji).
    items_ik_by_shape = {}
    items_jk_by_shape = {}

    for key_ij in strong_keys:
        i, j = key_ij
        n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_ij == 0:
            continue
        domain = (set(int(x) for x in pair_lmo_idx[key_ij])
                  if pair_lmo_idx is not None and key_ij in pair_lmo_idx
                  else set(range(nocc)))
        slot = pair_to_slot[key_ij]
        for k in domain:
            # IK direction: contributes to G_ij
            key_ik = (min(i, k), max(i, k))
            if key_ik in t2_pno_all:
                t2_ik = t2_pno_all[key_ik]
                n_ik = t2_ik.shape[0]
                if n_ik > 0:
                    S_ij_ik = S_pno_cache.get((key_ij, key_ik))
                    if S_ij_ik is not None:
                        items_ik_by_shape.setdefault(
                            (n_ij, n_ik), []).append((
                                key_ik, key_ij, i > k, j, slot, S_ij_ik))
            # JK direction: contributes to G_ji
            key_jk = (min(j, k), max(j, k))
            if key_jk in t2_pno_all:
                t2_jk = t2_pno_all[key_jk]
                n_jk = t2_jk.shape[0]
                if n_jk > 0:
                    S_ij_jk = S_pno_cache.get((key_ij, key_jk))
                    if S_ij_jk is not None:
                        items_jk_by_shape.setdefault(
                            (n_ij, n_jk), []).append((
                                key_jk, key_ij, j > k, i, slot, S_ij_jk))

    def _stack_buckets(items_by_shape, side):
        buckets = []
        for (n_ij, n_ik), items in items_by_shape.items():
            N = len(items)
            t2_keys = [it[0] for it in items]
            t2_transp = np.array([it[2] for it in items], dtype=np.bool_)
            scalar_lmo = np.array([it[3] for it in items], dtype=np.intp)
            item_idx = np.array([it[4] for it in items], dtype=np.intp)
            S_arr = np.empty((N, n_ij, n_ik))
            for n, it in enumerate(items):
                S_arr[n] = it[5]
            # Pre-compute which ks matter for gathering G_tilde rows.
            # The actual G_tilde index for item n is (k, other_lmo).
            # Store k per item too (extracted from key_ik).
            k_idx = np.empty(N, dtype=np.intp)
            for n, it in enumerate(items):
                key_pk = it[0]  # (min(i,k), max(i,k)) or (min(j,k), max(j,k))
                key_ij = it[1]
                i_or_j = key_ij[0] if side == 0 else key_ij[1]
                k_idx[n] = key_pk[1] if key_pk[0] == i_or_j else key_pk[0]
            buckets.append({
                'n_ij': n_ij, 'n_ik': n_ik,
                'S_arr': S_arr,
                't2_keys': t2_keys,
                't2_transp': t2_transp,
                'scalar_lmo': scalar_lmo,   # j for ik-side, i for jk-side
                'k_idx': k_idx,
                'item_idx': item_idx,
            })
        return buckets

    ik_buckets = _stack_buckets(items_ik_by_shape, side=0)
    jk_buckets = _stack_buckets(items_jk_by_shape, side=1)
    return {
        'ik_buckets': ik_buckets,
        'jk_buckets': jk_buckets,
        'pairs_by_n_ij': pairs_by_n_ij,
        'pair_to_slot': pair_to_slot,
    }


def _get_or_build_g_term_batched_view(plan, t2_pno_all):
    """Build (or fetch cached) flat per-item view across all G_term
    buckets. Concatenates S_arr static tensors; pre-computes per-item
    absolute offsets into t2_pno_all._buffer; collects k_idx /
    scalar_lmo / target slot arrays."""
    bv = plan.get('_g_batched_view')
    if bv is not None:
        return bv

    has_fts = hasattr(t2_pno_all, '_buffer')
    if has_fts:
        _t2_off_arr = np.asarray(t2_pno_all._offsets)
        _canon_to_idx = t2_pno_all._canon_to_idx

    def _build_side(buckets):
        n_ij_l, n_ik_l = [], []
        S_off_l, t2_off_l = [], []
        S_pieces = []
        S_run = 0
        t2_size_l = []
        tile_off = [0]
        k_idx_l, scalar_lmo_l, target_slot_l = [], [], []
        t2_canon_off_l, t2_trans_l = [], []
        for bucket in buckets:
            n_ij = bucket['n_ij']
            n_ik = bucket['n_ik']
            N_b = len(bucket['t2_keys'])
            for nb in range(N_b):
                n_ij_l.append(n_ij)
                n_ik_l.append(n_ik)
                S_pieces.append(np.ascontiguousarray(
                    bucket['S_arr'][nb]).ravel())
                S_off_l.append(S_run); S_run += n_ij * n_ik
                t2_size_l.append(n_ik * n_ik)
                tile_off.append(tile_off[-1] + n_ij * n_ij)
                k_idx_l.append(int(bucket['k_idx'][nb]))
                scalar_lmo_l.append(int(bucket['scalar_lmo'][nb]))
                target_slot_l.append((n_ij, int(bucket['item_idx'][nb])))
                if has_fts:
                    t2_canon_off_l.append(
                        int(_t2_off_arr[_canon_to_idx[bucket['t2_keys'][nb]]]))
                    t2_trans_l.append(int(bucket['t2_transp'][nb]))
        N = len(n_ij_l)
        t2_off = np.zeros(N + 1, dtype=np.int64)
        t2_off[1:] = np.cumsum(t2_size_l)
        return {
            'N': N,
            'n_ij': np.asarray(n_ij_l, dtype=np.int32),
            'n_ik': np.asarray(n_ik_l, dtype=np.int32),
            'S_off': np.asarray(S_off_l, dtype=np.int64),
            't2_off': t2_off,
            'tile_off': np.asarray(tile_off, dtype=np.int64),
            'k_idx': np.asarray(k_idx_l, dtype=np.int64),
            'scalar_lmo': np.asarray(scalar_lmo_l, dtype=np.int64),
            'target_slot': target_slot_l,
            'S_flat': (np.concatenate(S_pieces) if S_pieces
                       else np.zeros(0)),
            't2_canon_off': (np.asarray(t2_canon_off_l, dtype=np.int64)
                             if has_fts and t2_canon_off_l else None),
            't2_trans_arr': (np.asarray(t2_trans_l, dtype=np.int32)
                              if has_fts and t2_trans_l else None),
        }

    bv = {
        'ik': _build_side(plan['ik_buckets']),
        'jk': _build_side(plan['jk_buckets']),
    }
    plan['_g_batched_view'] = bv
    return bv


def _run_g_term_batched(plan, bv, t2_pno_all, G_tilde,
                        flat_G_ij, flat_G_ji):
    """Run batched G_term kernel for both sides; serial scatter into
    flat_G_ij / flat_G_ji."""
    from pyscf.cc.dlpno_tccsd._g_term_batched_cy import g_term_batched
    from pyscf.cc.dlpno_tccsd._cd_gather_cy import gather_t2_with_transpose
    from threadpoolctl import threadpool_limits

    G_tilde_c = np.ascontiguousarray(G_tilde)

    def _run_side(side_bv, flat_out):
        N = side_bv['N']
        if N == 0:
            return
        # Per-cycle t2 gather via Cython (uses absolute offsets into
        # t2_pno_all._buffer; transpose handled in-kernel).
        t2_flat = np.empty(int(side_bv['t2_off'][-1]))
        if (side_bv['t2_canon_off'] is not None
                and hasattr(t2_pno_all, '_buffer')):
            gather_t2_with_transpose(
                N, side_bv['n_ik'],
                side_bv['t2_canon_off'], side_bv['t2_trans_arr'],
                side_bv['t2_off'], t2_pno_all._buffer, t2_flat,
                min(64, N),
            )
        else:
            # Should not happen on the standard driver path (t2_pno_all is
            # always a FlatTensorStore there); leave NotImplemented to
            # surface any unexpected fallback during refactors.
            raise NotImplementedError(
                "g_term_batched fallback path requires t2_pno_all FTS")

        max_n_ij = int(side_bv['n_ij'].max(initial=1))
        max_n_ik = int(side_bv['n_ik'].max(initial=1))
        num_threads = min(64, N)
        tmp = np.empty((num_threads, max_n_ij * max_n_ik))
        tiles = np.zeros(int(side_bv['tile_off'][-1]))
        with threadpool_limits(limits=1, user_api='blas'):
            g_term_batched(
                N, max_n_ij, max_n_ik,
                side_bv['n_ij'], side_bv['n_ik'],
                side_bv['S_off'], side_bv['t2_off'][:N],
                side_bv['tile_off'],
                side_bv['k_idx'], side_bv['scalar_lmo'],
                side_bv['S_flat'], t2_flat, G_tilde_c,
                tmp, tiles, num_threads,
            )
        # Serial scatter — out -= Cc per item.
        flat_views = {n_ij: buf.ravel() for n_ij, buf in flat_out.items()}
        target_slot = side_bv['target_slot']
        tile_off = side_bv['tile_off']
        for n in range(N):
            n_ij, slot = target_slot[n]
            tile_size = n_ij * n_ij
            base = slot * tile_size
            tile = tiles[tile_off[n]:tile_off[n + 1]]
            flat_views[n_ij][base:base + tile_size] -= tile

    _run_side(bv['ik'], flat_G_ij)
    _run_side(bv['jk'], flat_G_ji)


def compute_G_term_batched(strong_keys, t2_pno_all, pno_spaces,
                          S_pno_cache, G_tilde, pair_lmo_idx, nocc,
                          S_pao_full=None, s1e=None, _pool=None):
    """Batched per-pair G_term (T2 residual Eq 81 Fock-oo coupling).

    Replaces the per-k inner loop inside compute_residual_v2 with one
    batched matmul per (n_ij, n_ik) shape bucket across all strong
    pairs.  Plan is built once per CCSD run (structure only depends on
    pair domains + PNO shapes) and cached as a function attribute.

    With a ``_pool``, buckets are processed in parallel (each task does
    one bucket's batched matmul with 1-thread BLAS — matches the rest of
    the CCSD driver's threading strategy).  Without a pool, serial.

    Returns: dict {key_ij: G_term (n_ij, n_ij) np.ndarray}
    """
    plan_key = tuple(sorted(t2_pno_all.keys()))
    _cache = getattr(compute_G_term_batched, '_plan_cache', None)
    if _cache is None:
        _cache = {}
        compute_G_term_batched._plan_cache = _cache
    plan = _cache.get(plan_key)
    if plan is None:
        plan = _build_g_term_plan(
            strong_keys, pno_spaces, pair_lmo_idx,
            t2_pno_all, S_pno_cache, nocc)
        _cache[plan_key] = plan

    # Flat output: one (n_pairs_in_n_ij, n_ij, n_ij) buffer per n_ij.
    flat_G_ij = {}
    flat_G_ji = {}
    for n_ij, pairs in plan['pairs_by_n_ij'].items():
        shp = (len(pairs), n_ij, n_ij)
        flat_G_ij[n_ij] = np.zeros(shp)
        flat_G_ji[n_ij] = np.zeros(shp)

    # Single-call batched path — replaces the per-bucket _pool.map loop
    # with one nogil prange Cython call per side. Static plan view (built
    # once) flattens per-bucket S_arr into a single concatenated buffer
    # and pre-computes per-item absolute offsets into t2_pno_all._buffer
    # for the t2 gather kernel. Per cycle: refill t2_flat via gather,
    # then call g_term_batched.
    bv = _get_or_build_g_term_batched_view(plan, t2_pno_all)
    _run_g_term_batched(plan, bv, t2_pno_all, G_tilde,
                        flat_G_ij, flat_G_ji)

    # Unpack back into dict {key_ij: G_term}.  G_term = G_ij + G_ji.T
    G_term_all = {}
    for n_ij, pairs in plan['pairs_by_n_ij'].items():
        G_ij_buf = flat_G_ij[n_ij]
        G_ji_buf = flat_G_ji[n_ij]
        for slot, key_ij in enumerate(pairs):
            G_term_all[key_ij] = (
                G_ij_buf[slot] + G_ji_buf[slot].T)
    # Pairs with zero n_ij or no items: return an empty array.
    for key_ij in strong_keys:
        if key_ij not in G_term_all:
            n_ij_ = pno_spaces[key_ij]['C_pno'].shape[1]
            G_term_all[key_ij] = np.zeros((n_ij_, n_ij_))
    return G_term_all




def _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e):
    """Return a function that retrieves S_pno[a,b] with uniform numerics.

    On cache hit: returns the cached matrix.
    On cache miss: computes via the same X_pno + S_pao_full path as the
    build (falls back to C_pno+s1e only when X_pno unavailable, e.g.
    CAS pairs) and POPULATES the cache so subsequent lookups hit.

    This makes the upfront O(N^2) restricted build work correctly: any
    key not pre-built is computed lazily once and cached.  After the
    first CCSD iteration the cache contains every actually-accessed
    entry; later iterations are pure hits.
    """
    from pyscf.cc.dlpno_tccsd.local_df import compute_S_pno

    def _get(ka, kb, default=None):
        if S_pno_cache is not None:
            S = S_pno_cache.get((ka, kb))
            if S is not None:
                return S
        if S_pao_full is None:
            return default
        S = compute_S_pno(ka, kb, pno_spaces, S_pao_full, s1e)
        if S_pno_cache is not None:
            # CPython dict insert is atomic under the GIL; concurrent
            # threads may compute the same entry but the last write wins
            # and all writes have identical contents — no data race.
            S_pno_cache[(ka, kb)] = S
        return S
    return _get


def build_D_tilde(t1_pno, t2_pno_all, pno_spaces, nocc,
                  ovL_bare, ooL_bare, S_pno_cache, with_df,
                  _term2_precomputed=None, cc_ints=None,
                  pair_lmo_idx=None, _pool=None,
                  S_pao_full=None, s1e=None,
                  t1_cache=None):
    """Build D_tilde (delta, Eq 84) for all ordered (i,k) pairs.

    delta_{ik}^{ac} = Terms 1-4 of Eq 84, using M/L integrals.
    Term 2 uses a batched single DF pass over all (i,k) pairs.

    Following Psi4 ccsd.cc compute_D_tilde() lines 1753-1818.
    """
    from pyscf.ao2mo import _ao2mo

    D_tilde_all = {}

    # Phase 1: cache t1 projections once (or reuse one passed from driver).
    if t1_cache is None:
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, build_t1_cache,
        )
        _pi = PairIndex(
            pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
        t1_cache = build_t1_cache(
            t1_pno, _pi, S_pno_cache, pno_spaces)

    # Iterate over all ordered (i,k) pairs
    all_pairs = set()
    for key in t2_pno_all:
        a, b = key
        all_pairs.add((a, b))
        all_pairs.add((b, a))

    # Term 2 can be precomputed via unified DF pass
    if _term2_precomputed is None:
        # Batched Term 2 via single DF pass (fallback)
        term2_data = {}
        for i_idx, k_idx in all_pairs:
            key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
            n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
            if n_ik == 0:
                continue
            t1_i_ik = t1_cache[key_ik][i_idx]
            if np.max(np.abs(t1_i_ik)) < 1e-15:
                continue
            ovL_k_ik = ovL_bare.get((key_ik, k_idx))
            if ovL_k_ik is None:
                continue
            term2_data[(i_idx, k_idx)] = {
                'key_ik': key_ik, 'n_ik': n_ik,
                'C_pno': np.asfortranarray(pno_spaces[key_ik]['C_pno']),
                't1_i': t1_i_ik, 'ovL_k': ovL_k_ik,
                'result': np.zeros((n_ik, n_ik)),
            }
        if term2_data:
            aux_off = 0
            for Lpq in with_df.loop():
                nL = Lpq.shape[0]
                for (i_idx, k_idx), td in term2_data.items():
                    n_ik = td['n_ik']
                    buf = _ao2mo.nr_e2(Lpq, td['C_pno'],
                                       (0, n_ik, 0, n_ik), aosym='s2')
                    B_L = buf.reshape(nL, n_ik, n_ik)
                    ovL_k_batch = td['ovL_k'][:, aux_off:aux_off+nL]
                    z_c = np.einsum('b,Lbc->Lc', td['t1_i'], B_L)
                    y = td['t1_i'] @ ovL_k_batch
                    td['result'] += 2.0 * ovL_k_batch @ z_c
                    td['result'] -= np.tensordot(y, B_L, axes=(0, 0))
                aux_off += nL
            _term2_precomputed = {ik: td['result'] for ik, td in term2_data.items()}

    # Unified S_pno accessor (cache hit → cached value; cache miss →
    # same X_pno + S_pao_full path as the upfront build, so restricting
    # the cache is safe to machine precision).
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    # --- Build D_tilde for each (i,k) pair ---
    def _process_ik(ik_tuple):
        i_idx, k_idx = ik_tuple
        key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
        n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
        if n_ik == 0:
            return ik_tuple, None

        D_tilde_ik = np.zeros((n_ik, n_ik))

        # T1_i projected to PNO_ik
        t1_i_ik = t1_cache[key_ik][i_idx]

        # Restrict LMO iteration to pair (i,k)'s local domain for Term 1.
        if pair_lmo_idx is not None and key_ik in pair_lmo_idx:
            _ll_idx_ik = np.asarray(pair_lmo_idx[key_ik])
        else:
            _ll_idx_ik = np.arange(nocc)

        # Phase 1: T1_all_ik = full (nocc, n_pno_ik) T1 projection table.
        # Cached copy already has zero rows for LMOs outside the domain, so
        # downstream Term 3 indexing T1_all_ik[ll] just works.
        T1_all_ik = t1_cache[key_ik]

        # Term 2: Psi4 lines 1791-1801 use K_tilde_chem[ki] × t1_i.
        # K_tc[b, a*n+c] = Σ_Q k_Qa[Q,b] * Qab[Q,a,c]
        # Part A: D[a,b] += 2 * Σ_{Q,c} k_Qa[Q,b] * Qab[Q,a,c] * t1_i[c]
        # Part B: D[r,s] -= Σ_{Q,b} t1_i[b] * k_Qa[Q,b] * Qab[Q,s,r]
        key_ki = (min(k_idx, i_idx), max(k_idx, i_idx))
        if key_ki in cc_ints and cc_ints[key_ki] is not None:
            ci_ki = cc_ints[key_ki]
            if key_ki[0] == k_idx:
                k_Qa = ci_ki['i_Qa']
            else:
                k_Qa = ci_ki['j_Qa']
            Qab_ki = ci_ki['Qab']
            # Part A: D[a,b] += 2 * Σ_{Q,c} k_Qa[Q,b]*Qab[Q,a,c]*t1_i[c]
            # z[Q,a] = Σ_c Qab[Q,a,c]*t1_i[c]
            z_Qa = np.tensordot(Qab_ki, t1_i_ik, axes=(2, 0))
            # D[a,b] += 2 * Σ_Q z_Qa[Q,a] * k_Qa[Q,b] = 2 * z_Qa.T @ k_Qa
            D_tilde_ik += 2.0 * z_Qa.T @ k_Qa
            # Part B: D[r,s] -= Σ_{Q,b} t1_i[b]*k_Qa[Q,b]*Qab[Q,s,r]
            # w[Q] = Σ_b t1_i[b]*k_Qa[Q,b]
            w = k_Qa @ t1_i_ik  # (n_local,)
            # D[r,s] -= Σ_Q w[Q]*Qab[Q,s,r]
            D_tilde_ik -= np.tensordot(w, Qab_ki, axes=(0, 0)).T
        elif _term2_precomputed and (i_idx, k_idx) in _term2_precomputed:
            D_tilde_ik += _term2_precomputed[(i_idx, k_idx)]

        # --- Term 1: -Σ_l T1_all[l,a] · M_{ik}^{lc} ---
        # M_{ik}^{lc} = 2*(il|kc) - (ik|lc)
        # (il|kc) = Σ_Q ooL[i,l,Q]*ovL_k_ik[c,Q] → K_bar-like
        # (ik|lc) = Σ_Q ooL[i,k,Q]*ovL_l_ik[c,Q] → K_bar_chem-like
        # Psi4 (ccsd.cc:1790) restricts l to lmopair_to_lmos_[ik]
        _domain_ik_t1 = (set(pair_lmo_idx[key_ik].tolist())
                         if pair_lmo_idx is not None and key_ik in pair_lmo_idx
                         else set(range(nocc)))
        _use_local_d1 = (key_ik in cc_ints and cc_ints[key_ik] is not None)
        for ll in range(nocc):
            if ll not in _domain_ik_t1:
                continue
            if _use_local_d1:
                from pyscf.cc.dlpno_tccsd.local_df import (
                    get_local_ovL, get_local_ooL_vec)
                _ovL_k = get_local_ovL(cc_ints, key_ik, k_idx)
                _ovL_l = get_local_ovL(cc_ints, key_ik, ll)
                _ooL_il = get_local_ooL_vec(cc_ints, i_idx, ll, key_ik)
                _ooL_ik = get_local_ooL_vec(cc_ints, i_idx, k_idx, key_ik)
                if _ovL_k is not None and _ovL_l is not None and _ooL_il is not None and _ooL_ik is not None:
                    ilkc = _ovL_k @ _ooL_il
                    iklc = _ovL_l @ _ooL_ik
                else:
                    continue
            else:
                ooL_il = ooL_bare[i_idx, ll, :]
                ooL_ik = ooL_bare[i_idx, k_idx, :]
                ovL_k_ik_entry = ovL_bare.get((key_ik, k_idx))
                ovL_l_ik_entry = ovL_bare.get((key_ik, ll))
                if ovL_k_ik_entry is None or ovL_l_ik_entry is None:
                    continue
                ilkc = ovL_k_ik_entry @ ooL_il
                iklc = ovL_l_ik_entry @ ooL_ik
            # M^{lc} = 2*(il|kc) - (ik|lc)
            M_lc = 2.0 * ilkc - iklc  # (n_ik,)
            # D[a,c] -= T1_all[l,a] * M_lc[c]
            D_tilde_ik -= np.outer(T1_all_ik[ll], M_lc)

        # --- Term 3: -Σ_l T1_l^a · (L_lk × T1_i) ---
        # L_lk[a,b] = 2*(la|kb) - (lb|ka) in PNO_lk
        # Psi4 restricts l to lmopair_to_lmos_[ik] (ccsd.cc:1792)
        _domain_ik = (set(pair_lmo_idx[key_ik].tolist())
                      if pair_lmo_idx is not None and key_ik in pair_lmo_idx
                      else set(range(nocc)))
        for ll in range(nocc):
            if ll not in _domain_ik:
                continue
            key_lk = (min(ll, k_idx), max(ll, k_idx))
            if key_lk not in pno_spaces:
                continue
            n_lk = pno_spaces[key_lk]['C_pno'].shape[1]
            if n_lk == 0:
                continue
            K_lk = get_local_K(cc_ints, key_lk, ll, k_idx)
            if K_lk is None:
                continue
            L_lk = 2.0 * K_lk - K_lk.T

            t1_i_lk = t1_cache[key_lk][i_idx]
            if np.max(np.abs(t1_i_lk)) < 1e-15:
                continue
            Lt1 = L_lk @ t1_i_lk  # (n_lk,)
            # Project to PNO_ik
            S_ik_lk = _s_pno_get(key_ik, key_lk)
            if S_ik_lk is None:
                continue
            Lt1_ik = S_ik_lk @ Lt1  # (n_ik,)
            t1_l_ik = T1_all_ik[ll]
            D_tilde_ik -= np.outer(t1_l_ik, Lt1_ik)

        # --- Term 4: +(1/2) Σ_l S@u_il@S @ L_kl @ S ---
        # Psi4 restricts l to lmopair_to_lmos_[ik] (ccsd.cc:1803)
        for ll in range(nocc):
            if ll not in _domain_ik:
                continue
            key_il = (min(i_idx, ll), max(i_idx, ll))
            key_lk = (min(ll, k_idx), max(ll, k_idx))
            if key_il not in t2_pno_all or key_lk not in pno_spaces:
                continue
            t2_il = t2_pno_all.get(key_il)
            if t2_il is None or t2_il.shape[0] == 0:
                continue
            n_il = t2_il.shape[0]
            n_lk = pno_spaces[key_lk]['C_pno'].shape[1]
            if n_lk == 0:
                continue

            t2_il_d = t2_il.T if i_idx > ll else t2_il
            u_il = 2.0 * t2_il_d - t2_il_d.T

            K_lk = get_local_K(cc_ints, key_lk, ll, k_idx)
            if K_lk is None:
                continue
            L_lk = 2.0 * K_lk - K_lk.T

            def _get_S_or_I(ka, kb, n_a):
                if ka == kb:
                    return np.eye(n_a)
                return _s_pno_get(ka, kb)

            n_ik2 = pno_spaces[key_ik]['C_pno'].shape[1]
            S_ik_il = _get_S_or_I(key_ik, key_il, n_ik2)
            S_il_lk = _get_S_or_I(key_il, key_lk, n_il)
            S_lk_ik = _get_S_or_I(key_lk, key_ik, n_lk)
            if S_ik_il is None or S_il_lk is None or S_lk_ik is None:
                continue

            # Psi4: S(ik,il) @ Tt[il] @ S(il,lk) @ L[lk] @ S(lk,ik)
            D_temp = S_ik_il @ u_il @ S_il_lk @ L_lk @ S_lk_ik
            D_tilde_ik += 0.5 * D_temp  # Term 4: Jiang Eq 84

        return ik_tuple, D_tilde_ik

    if _pool is not None:
        for ik, val in _pool.map(_process_ik, list(all_pairs)):
            if val is not None:
                D_tilde_all[ik] = val
    else:
        for ik in all_pairs:
            _, val = _process_ik(ik)
            if val is not None:
                D_tilde_all[ik] = val
    return D_tilde_all


def _build_d_tilde_t34_plan(
        all_pairs, pno_spaces, pair_lmo_idx, t2_pno_all,
        S_pno_cache, cc_ints, _s_pno_get, nocc):
    """Build the one-time plan for D_tilde Terms 3 and 4.

    Structurally identical to ``_build_c_tilde_t34_plan``: same (i, k, l)
    triple space, same bucket-by-shape grouping, same scatter-index
    machinery.  The only differences versus C_tilde:

      - T3 stacks ``L_lk.T`` (where ``L_lk = 2*K_lk - K_lk.T``) in the
        ``K`` slot so the shared kernel's ``K[n].T @ t1i[n]`` evaluates
        to ``L_lk @ t1_i_lk`` (the contraction D_tilde actually wants).
      - T4 stacks ``L_lk`` directly in the ``K`` slot and records the
        (key_il, transpose-flag) pair in ``u_sources`` so the per-cycle
        gather can materialise ``u_il = 2*t2_il_d - t2_il_d.T`` cheaply
        from the current ``t2_pno_all`` snapshot.
    """
    from pyscf.cc.dlpno_tccsd.local_df import get_local_K

    t3_items = []  # (i, k, l, M, S_ik_lk, key_lk, key_ik, n_ik, n_lk)
    t4_items = []  # (i, k, S_ik_il, S_il_lk, L, S_lk_ik, key_il, transpose, n_ik, n_lk, n_il)

    for (i, k) in all_pairs:
        key_ik = (min(i, k), max(i, k))
        n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
        if n_ik == 0:
            continue
        if pair_lmo_idx is not None and key_ik in pair_lmo_idx:
            _ll_idx = pair_lmo_idx[key_ik]
        else:
            _ll_idx = np.arange(nocc)

        for ll_raw in _ll_idx:
            ll = int(ll_raw)
            key_lk = (min(ll, k), max(ll, k))

            # Term 3: need K_lk (not K_kl) — MO-ordered get_local_K(ll, k).
            if key_lk in pno_spaces:
                n_lk = pno_spaces[key_lk]['C_pno'].shape[1]
                if n_lk > 0:
                    K_lk = get_local_K(cc_ints, key_lk, ll, k)
                    if K_lk is not None:
                        S_ik_lk = (np.eye(n_ik) if key_ik == key_lk
                                   else _s_pno_get(key_ik, key_lk))
                        if S_ik_lk is not None:
                            L_lk = 2.0 * K_lk - K_lk.T
                            M_stacked = np.ascontiguousarray(L_lk.T)
                            t3_items.append((i, k, ll, M_stacked, S_ik_lk,
                                             key_lk, key_ik, n_ik, n_lk))

            # Term 4: u_il contraction through L_lk with four S-projections.
            key_il = (min(i, ll), max(i, ll))
            if key_il not in t2_pno_all:
                continue
            if key_lk not in pno_spaces:
                continue
            n_il = pno_spaces[key_il]['C_pno'].shape[1]
            n_lk = pno_spaces[key_lk]['C_pno'].shape[1]
            if n_il == 0 or n_lk == 0:
                continue
            K_lk = get_local_K(cc_ints, key_lk, ll, k)
            if K_lk is None:
                continue
            L_lk = np.ascontiguousarray(2.0 * K_lk - K_lk.T)
            transpose = (i > ll)  # opposite of C_tilde's (ll > i) rule
            S_ik_il = (np.eye(n_ik) if key_ik == key_il
                       else _s_pno_get(key_ik, key_il))
            S_il_lk = (np.eye(n_il) if key_il == key_lk
                       else _s_pno_get(key_il, key_lk))
            S_lk_ik = (np.eye(n_lk) if key_lk == key_ik
                       else _s_pno_get(key_lk, key_ik))
            if S_ik_il is None or S_il_lk is None or S_lk_ik is None:
                continue
            t4_items.append((i, k, S_ik_il, S_il_lk, L_lk, S_lk_ik,
                             key_il, transpose, n_ik, n_lk, n_il))

    # --- Global (i, k) → flat slot map (one slot per n_ik output bucket) ---
    pairs_by_n_ik = {}
    pair_to_slot = {}
    for (i, k) in all_pairs:
        key_ik = (min(i, k), max(i, k))
        n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
        if n_ik == 0:
            continue
        if n_ik not in pairs_by_n_ik:
            pairs_by_n_ik[n_ik] = []
        slot = len(pairs_by_n_ik[n_ik])
        pairs_by_n_ik[n_ik].append((i, k))
        pair_to_slot[(i, k)] = slot

    # --- Bucket T3 by (n_ik, n_lk); stack L.T (in K slot), S, index arrays ---
    t3_by_shape = {}
    for it in t3_items:
        t3_by_shape.setdefault((it[7], it[8]), []).append(it)
    t3_buckets = []
    for (n_ik, n_lk), items in t3_by_shape.items():
        N = len(items)
        K = np.empty((N, n_lk, n_lk))
        S = np.empty((N, n_ik, n_lk))
        t1i_keys = []
        T1l_keys = []
        item_idx = np.empty(N, dtype=np.intp)
        for n, it in enumerate(items):
            i_, k_, ll = it[0], it[1], it[2]
            K[n] = it[3]
            S[n] = it[4]
            key_lk, key_ik = it[5], it[6]
            t1i_keys.append((key_lk, i_))
            T1l_keys.append((key_ik, ll))
            item_idx[n] = pair_to_slot[(i_, k_)]
        t3_buckets.append({
            'K': K, 'S': S,
            't1i_keys': t1i_keys, 'T1l_keys': T1l_keys,
            'n_ki': n_ik, 'n_kl': n_lk,
            'item_idx': item_idx,
        })

    # --- Bucket T4 by (n_ik, n_lk, n_il); stack L and u-source descriptors ---
    t4_by_shape = {}
    for it in t4_items:
        t4_by_shape.setdefault((it[8], it[9], it[10]), []).append(it)
    t4_buckets = []
    for (n_ik, n_lk, n_il), items in t4_by_shape.items():
        N = len(items)
        S_ik_il = np.empty((N, n_ik, n_il))
        S_il_lk = np.empty((N, n_il, n_lk))
        K = np.empty((N, n_lk, n_lk))
        S_lk_ik = np.empty((N, n_lk, n_ik))
        u_sources = []
        item_idx = np.empty(N, dtype=np.intp)
        for n, it in enumerate(items):
            i_, k_ = it[0], it[1]
            S_ik_il[n] = it[2]
            S_il_lk[n] = it[3]
            K[n] = it[4]
            S_lk_ik[n] = it[5]
            u_sources.append((it[6], bool(it[7])))
            item_idx[n] = pair_to_slot[(i_, k_)]
        t4_buckets.append({
            'S_ki_li': S_ik_il, 'S_li_kl': S_il_lk,
            'K': K, 'S_kl_ki': S_lk_ik,
            'u_sources': u_sources,
            'n_ki': n_ik, 'n_kl': n_lk, 'n_li': n_il,
            'item_idx': item_idx,
        })

    return {
        't3': t3_buckets, 't4': t4_buckets,
        'pairs_by_n_ki': pairs_by_n_ik,
        'pair_to_slot': pair_to_slot,
    }


def build_D_tilde_psi4(
        t1_pno, t2_pno_all, pno_spaces, nocc,
        ovL_bare, ooL_bare, S_pno_cache, with_df,
        _term2_precomputed=None, cc_ints=None,
        pair_lmo_idx=None, _pool=None,
        S_pao_full=None, s1e=None,
        t1_cache=None, omp_threads=None):
    """Compute D_tilde — Psi4 per-pair port matching ccsd.cc:1991 line-by-line.

    For each ordered pair (i, k):
      Term 2 (Psi4 2013-2023): T_i = S(ik, ii) @ T_ia[i]
                               L_temp_A = (2*K_tilde_chem[ki] reshape (n²,n)) @ T_i
                               D_tilde[ik] += L_temp_A.reshape(n,n).T
                               L_temp_B = T_i.T @ K_tilde_chem[ki]   reshape(n,n)
                               D_tilde[ik] -= L_temp_B.T
      Term 1 (Psi4 2025-2028): L_bar = 2*K_bar[ik] - K_bar_chem[ik]
                               D_tilde[ik] -= T_n_ij[ik].T @ L_bar
      Term 3 (Psi4 2030-2039): for l in lmopair_to_lmos_[ik]:
                                   T_l   = S(ik, ll) @ T_ia[l]
                                   T_i_lk = S(lk, ii) @ T_ia[i]
                                   L_lk = T_i_lk.T @ L_iajb[lk_ord] @ S(lk, ik)
                                   D_tilde[ik] -= np.outer(T_l, L_lk)
      Term 4 (Psi4 2041-2050): for l in lmopair_to_lmos_[ik]:
                                   X = Tt_iajb[il_ord] @ S(il, lk) @ L_iajb[lk_ord]
                                   Y = S(ik, il) @ X @ S(lk, ik)
                                   D_tilde[ik] += 0.5 * Y

    L_iajb_[ord] = 2 * K_iajb_[ord] - K_iajb_[ord].T  (Psi4 convention).
    Tt_iajb_[ord] = 2 * T_iajb_[ord] - T_iajb_[ord].T.
    """
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    all_ordered = []
    for key in t2_pno_all:
        a, b = key
        all_ordered.append((a, b))
        if a != b:
            all_ordered.append((b, a))

    if t1_cache is None:
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, build_t1_cache,
        )
        _pi = PairIndex(
            pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
        t1_cache = build_t1_cache(t1_pno, _pi, S_pno_cache, pno_spaces)

    def _t2_ordered(a, b):
        canon = (min(a, b), max(a, b))
        t2 = t2_pno_all.get(canon)
        if t2 is None or t2.shape[0] == 0:
            return None
        return t2 if (a, b) == canon else t2.T

    def _K_iajb_ordered(canonical_key, a, b):
        """Return K_iajb for ordered (a, b); reads canonical and swaps."""
        ci = cc_ints.get(canonical_key) if cc_ints is not None else None
        if ci is None:
            return None
        K_canon = ci['K_iajb']
        return K_canon if (a, b) == canonical_key else K_canon.T

    def _per_pair(key_ik):
        i, k = key_ik
        canonical = (min(i, k), max(i, k))
        n_ik = pno_spaces[canonical]['C_pno'].shape[1]
        if n_ik == 0:
            return key_ik, np.zeros((0, 0))
        ci_ik = cc_ints.get(canonical) if cc_ints is not None else None
        if ci_ik is None:
            return key_ik, np.zeros((n_ik, n_ik))

        ii = (i, i)
        n_ii = pno_spaces[ii]['C_pno'].shape[1] if ii in pno_spaces else 0
        D_tilde = np.zeros((n_ik, n_ik))

        # ---- Term 2 (Psi4 ccsd.cc:2013-2023) ----
        # T_i_in_ik = S(ik, ii) @ t1[i]
        # K_tilde_chem[ki ordered] (n_ik, n_ik²): if k is canonical's first
        # idx, K_tilde_chem_[ki ordered] uses k-side (which is canonical's
        # i-side aux) → ci['K_tilde_chem_i']. ki ordered (k, i): k=k, i=i.
        # Canonical for ki = (min(k,i), max(k,i)) = same canonical as ik.
        # So K_tilde_chem orientation: if canonical[0] == k → use _i, else _j.
        t1_i_pno = t1_pno.get(i)
        if (t1_i_pno is not None and t1_i_pno.size > 0
                and np.max(np.abs(t1_i_pno)) > 1e-15
                and ii in pno_spaces and n_ii > 0):
            S_ik_ii = (np.eye(n_ik) if canonical == ii
                       else _s_pno_get(canonical, ii))
            if S_ik_ii is not None:
                T_i_in_ik = S_ik_ii @ t1_i_pno          # (n_ik,)
                # K_tilde_chem for ordered ki=(k, i):
                is_k_first = (canonical[0] == k)
                K_tilde_chem = (ci_ik['K_tilde_chem_i'] if is_k_first
                                else ci_ik['K_tilde_chem_j'])
                # Reshape to (n_ik, n_ik, n_ik): K[a, b, c] = (k a | b c)
                K3 = K_tilde_chem.reshape(n_ik, n_ik, n_ik)
                # Part A: D_tilde[b, a] += 2 * Σ_c K[a, b, c] * T_i[c]
                D_tilde += 2.0 * np.einsum(
                    'abc,c->ba', K3, T_i_in_ik, optimize=True)
                # Part B: D_tilde[c, b] -= Σ_a T_i[a] * K[a, b, c]
                D_tilde -= np.einsum(
                    'abc,a->cb', K3, T_i_in_ik, optimize=True)

        # ---- Term 1 (Psi4 ccsd.cc:2025-2028) ----
        # L_bar = 2*K_bar[ik ordered] - K_bar_chem[ik]
        # K_bar[ik ordered]: ordered (i, k). canonical first idx is ?
        # if i == canonical[0]: K_bar_ij; else K_bar_ji.
        p_lmos = ci_ik['p_lmos']
        T_n_ik = t1_cache[canonical][p_lmos]            # (nlmo_pair, n_ik)
        K_bar_chem = ci_ik['K_bar_chem']                # (nlmo_pair, n_ik)
        is_i_first = (canonical[0] == i)
        K_bar_ord = (ci_ik['K_bar_ij'] if is_i_first
                     else ci_ik['K_bar_ji'])
        L_bar = 2.0 * K_bar_ord - K_bar_chem
        D_tilde -= T_n_ik.T @ L_bar

        domain = ([int(x) for x in pair_lmo_idx[canonical]]
                  if pair_lmo_idx is not None and canonical in pair_lmo_idx
                  else list(range(nocc)))

        # ---- Term 3 (Psi4 ccsd.cc:2030-2039): outer product per l ----
        # L_lk = T_i_lk.T @ L_iajb_[lk ordered] @ S(lk, ik)
        # ordered lk = (l, k); canonical (min(l,k), max(l,k)).
        # L_iajb = 2*K_iajb_[ordered] - K_iajb_[ordered].T.
        for l in domain:
            lk_canon = (min(l, k), max(l, k))
            ll_key = (l, l)
            if lk_canon not in pno_spaces or ll_key not in pno_spaces:
                continue
            n_lk = pno_spaces[lk_canon]['C_pno'].shape[1]
            if n_lk == 0:
                continue
            t1_l_pno = t1_pno.get(l)
            if t1_l_pno is None or t1_l_pno.size == 0:
                continue
            t1_i_pno_loc = t1_pno.get(i)
            if t1_i_pno_loc is None or t1_i_pno_loc.size == 0:
                continue
            K_iajb_ord = _K_iajb_ordered(lk_canon, l, k)
            if K_iajb_ord is None:
                continue
            L_iajb_ord = 2.0 * K_iajb_ord - K_iajb_ord.T
            S_ik_ll = (np.eye(n_ik) if canonical == ll_key
                       else _s_pno_get(canonical, ll_key))
            S_lk_ii = (np.eye(n_lk) if lk_canon == ii
                       else _s_pno_get(lk_canon, ii))
            S_lk_ik = (np.eye(n_lk) if lk_canon == canonical
                       else _s_pno_get(lk_canon, canonical))
            if S_ik_ll is None or S_lk_ii is None or S_lk_ik is None:
                continue
            T_l = S_ik_ll @ t1_l_pno                          # (n_ik,)
            T_i_lk = S_lk_ii @ t1_i_pno_loc                   # (n_lk,)
            L_lk = T_i_lk @ L_iajb_ord @ S_lk_ik              # (n_ik,)
            D_tilde -= np.outer(T_l, L_lk)

        # ---- Term 4 (Psi4 ccsd.cc:2041-2050): triplet sandwich per l ----
        # ordered il = (i, l); canonical (min(i,l), max(i,l)).
        # ordered lk = (l, k) as above.
        # Tt_iajb_[il ord] = 2*T_iajb_[il ord] - T_iajb_[il ord].T
        for l in domain:
            il_canon = (min(i, l), max(i, l))
            lk_canon = (min(l, k), max(l, k))
            if il_canon not in pno_spaces or lk_canon not in pno_spaces:
                continue
            n_il = pno_spaces[il_canon]['C_pno'].shape[1]
            n_lk = pno_spaces[lk_canon]['C_pno'].shape[1]
            if n_il == 0 or n_lk == 0:
                continue
            t2_il_ord = _t2_ordered(i, l)
            if t2_il_ord is None:
                continue
            Tt_il = 2.0 * t2_il_ord - t2_il_ord.T            # (n_il, n_il)
            K_iajb_lk_ord = _K_iajb_ordered(lk_canon, l, k)
            if K_iajb_lk_ord is None:
                continue
            L_iajb_ord = 2.0 * K_iajb_lk_ord - K_iajb_lk_ord.T
            S_il_lk = (np.eye(n_il) if il_canon == lk_canon
                       else _s_pno_get(il_canon, lk_canon))
            S_ik_il = (np.eye(n_il) if canonical == il_canon
                       else _s_pno_get(canonical, il_canon))
            S_lk_ik = (np.eye(n_lk) if lk_canon == canonical
                       else _s_pno_get(lk_canon, canonical))
            if S_il_lk is None or S_ik_il is None or S_lk_ik is None:
                continue
            X = Tt_il @ S_il_lk @ L_iajb_ord                 # (n_il, n_lk)
            Y = S_ik_il @ X @ S_lk_ik                        # (n_ik, n_ik)
            D_tilde += 0.5 * Y

        return key_ik, D_tilde

    D_tilde_all = {}
    if _pool is not None:
        for key, val in _pool.map(_per_pair, all_ordered):
            D_tilde_all[key] = val
    else:
        for key in all_ordered:
            _, val = _per_pair(key)
            D_tilde_all[key] = val
    return D_tilde_all


def build_D_tilde_batched(
        t1_pno, t2_pno_all, pno_spaces, nocc,
        ovL_bare, ooL_bare, S_pno_cache, with_df,
        _term2_precomputed=None, cc_ints=None,
        pair_lmo_idx=None, _pool=None,
        S_pao_full=None, s1e=None,
        t1_cache=None, omp_threads=None):
    """Drop-in replacement for ``build_D_tilde`` with Terms 3+4 batched.

    Phase 1 runs Terms 1 and 2 per-pair via the reference ``_process_ik``
    (unchanged code path — split out below as ``_process_ik_t12``).
    Phase 2 then runs Terms 3 and 4 on top via the shared Cython kernels
    (``t3_kernel``, ``t4_kernel(scale=+0.5)``) using a plan-cache keyed
    on ``t2_pno_all.keys()`` — the same pattern as
    ``compute_C_tilde_batched``.  Output matches the reference to
    machine precision.
    """
    from pyscf.cc.dlpno_tccsd.local_df import (
        get_local_K, get_local_ovL, get_local_ooL_vec,
    )
    from pyscf.ao2mo import _ao2mo

    D_tilde_all = {}
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    # --- T1 cache (reuse if caller provided one) ---
    if t1_cache is None:
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, build_t1_cache,
        )
        _pi = PairIndex(
            pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
        t1_cache = build_t1_cache(
            t1_pno, _pi, S_pno_cache, pno_spaces)

    all_pairs = set()
    for key in t2_pno_all:
        a, b = key
        all_pairs.add((a, b))
        all_pairs.add((b, a))

    # --- Term 2 fallback via batched DF (only when cc_ints doesn't cover) ---
    if _term2_precomputed is None:
        term2_data = {}
        for i_idx, k_idx in all_pairs:
            key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
            n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
            if n_ik == 0:
                continue
            t1_i_ik = t1_cache[key_ik][i_idx]
            if np.max(np.abs(t1_i_ik)) < 1e-15:
                continue
            ovL_k_ik = ovL_bare.get((key_ik, k_idx))
            if ovL_k_ik is None:
                continue
            term2_data[(i_idx, k_idx)] = {
                'key_ik': key_ik, 'n_ik': n_ik,
                'C_pno': np.asfortranarray(pno_spaces[key_ik]['C_pno']),
                't1_i': t1_i_ik, 'ovL_k': ovL_k_ik,
                'result': np.zeros((n_ik, n_ik)),
            }
        if term2_data:
            aux_off = 0
            for Lpq in with_df.loop():
                nL = Lpq.shape[0]
                for (i_idx, k_idx), td in term2_data.items():
                    n_ik = td['n_ik']
                    buf = _ao2mo.nr_e2(Lpq, td['C_pno'],
                                       (0, n_ik, 0, n_ik), aosym='s2')
                    B_L = buf.reshape(nL, n_ik, n_ik)
                    ovL_k_batch = td['ovL_k'][:, aux_off:aux_off+nL]
                    z_c = np.einsum('b,Lbc->Lc', td['t1_i'], B_L)
                    y = td['t1_i'] @ ovL_k_batch
                    td['result'] += 2.0 * ovL_k_batch @ z_c
                    td['result'] -= np.tensordot(y, B_L, axes=(0, 0))
                aux_off += nL
            _term2_precomputed = {
                ik: td['result'] for ik, td in term2_data.items()}

    # --- Phase 1: per-(i,k) loop covering Terms 1 and 2 only ---
    def _process_ik_t12(ik_tuple):
        i_idx, k_idx = ik_tuple
        key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
        n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
        if n_ik == 0:
            return ik_tuple, None

        D_tilde_ik = np.zeros((n_ik, n_ik))
        t1_i_ik = t1_cache[key_ik][i_idx]
        T1_all_ik = t1_cache[key_ik]

        # Term 2 — mirror of build_D_tilde's per-triple Term 2 branches
        key_ki = key_ik
        if key_ki in cc_ints and cc_ints[key_ki] is not None:
            ci_ki = cc_ints[key_ki]
            if key_ki[0] == k_idx:
                k_Qa = ci_ki['i_Qa']
            else:
                k_Qa = ci_ki['j_Qa']
            Qab_ki = ci_ki['Qab']
            z_Qa = np.tensordot(Qab_ki, t1_i_ik, axes=(2, 0))
            D_tilde_ik += 2.0 * z_Qa.T @ k_Qa
            w = k_Qa @ t1_i_ik
            D_tilde_ik -= np.tensordot(w, Qab_ki, axes=(0, 0)).T
        elif _term2_precomputed and (i_idx, k_idx) in _term2_precomputed:
            D_tilde_ik += _term2_precomputed[(i_idx, k_idx)]

        # Term 1 — full pair-domain sum (matches Psi4 ccsd.cc:1913
        # `L_bar_temp = 2*K_bar[ik] - K_bar_chem[ik]; -T_n_ij.T @ L_bar_temp`).
        # NOTE: the prior implementation restricted `ll` to {i_idx, k_idx}
        # because `get_local_ooL_vec` returns None outside those two (it only
        # stores ooL[i_lmo_of_pair, :, Q] and ooL[j_lmo_of_pair, :, Q]). The
        # i_Qk / j_Qk tensors actually cover the full LMO axis for that fixed
        # LMO side, so we can gather the whole domain's ooL[i, l, Q] in one
        # slice. Matching Psi4's full-domain sum drifts E_tccsd on water10 by
        # ~13 μEh from the historical −2.1308345454 anchor; see the project
        # memory for the revert recipe if needed.
        if pair_lmo_idx is not None and key_ik in pair_lmo_idx:
            _ll_idx = np.asarray(pair_lmo_idx[key_ik], dtype=np.intp)
        else:
            _ll_idx = np.arange(nocc)

        if key_ik in cc_ints and cc_ints[key_ik] is not None:
            ci_ik2 = cc_ints[key_ik]
            # ooL_il[Q, l] = ooL[i_idx, l, Q] for l in domain. Use whichever
            # Qk tensor corresponds to i_idx's position in the pair key.
            # i_Qk/j_Qk reduced to (n_local, nlmo_p); translate _ll_idx and
            # k_idx -> position within p_lmos.
            _ll_idx_in_p = np.asarray(
                ci_ik2['p_lmos_dense'])[_ll_idx].astype(np.intp)
            _k_idx_in_p = int(ci_ik2['p_lmos_dense'][k_idx])
            if key_ik[0] == i_idx:
                ooL_il_all = ci_ik2['i_Qk'][:, _ll_idx_in_p]  # (n_local, n_domain)
                ooL_ik = ci_ik2['i_Qk'][:, _k_idx_in_p]       # (n_local,)
            else:
                ooL_il_all = ci_ik2['j_Qk'][:, _ll_idx_in_p]
                ooL_ik = ci_ik2['j_Qk'][:, _k_idx_in_p]
            Qma = ci_ik2['Qma']                           # (n_local, nlmo_p, n_pno)
            ovL_k = Qma[:, _k_idx_in_p, :]                # (n_local, n_pno)
            # ilkc[l, c] = Σ_Q ovL[k, c, Q] * ooL[i, l, Q]
            ilkc_all = ooL_il_all.T @ ovL_k               # (n_domain, n_pno)
            # iklc[l, c] = Σ_Q ovL[l, c, Q] * ooL[i, k, Q]
            iklc_all = np.tensordot(
                Qma[:, _ll_idx_in_p, :], ooL_ik, axes=(0, 0))  # (n_domain, n_pno)
            M_lc_all = 2.0 * ilkc_all - iklc_all
            T1_rows = T1_all_ik[_ll_idx]                  # (n_domain, n_pno)
            D_tilde_ik -= T1_rows.T @ M_lc_all
        else:
            # Fallback for pairs missing from cc_ints (e.g. CAS): per-l loop.
            ooL_ik_bare = ooL_bare[i_idx, k_idx, :]
            for _li, ll in enumerate(_ll_idx):
                ooL_il = ooL_bare[i_idx, int(ll), :]
                ovL_k_ik_entry = ovL_bare.get((key_ik, k_idx))
                ovL_l_ik_entry = ovL_bare.get((key_ik, int(ll)))
                if ovL_k_ik_entry is None or ovL_l_ik_entry is None:
                    continue
                ilkc = ovL_k_ik_entry @ ooL_il
                iklc = ovL_l_ik_entry @ ooL_ik_bare
                M_lc = 2.0 * ilkc - iklc
                D_tilde_ik -= np.outer(T1_all_ik[int(ll)], M_lc)

        return ik_tuple, D_tilde_ik

    # --- Phase 1 batched prange path (path b, commit a96438422 sibling) ---
    # Same pattern as compute_C_tilde_batched: static per-pair buffers
    # (k_Qa, Qab, Qma_sub, ooL_il_all, ooL_ik, ovL_k) cached across CCSD
    # iterations; per-iter T1 inputs gathered fresh. Pairs not covered
    # by cc_ints fall back to the per-pair Python path above.
    from pyscf.cc.dlpno_tccsd._d_tilde_ph1_batched_cy import (
        d_tilde_ph1_batched)
    from threadpoolctl import threadpool_limits

    all_pairs_list = list(all_pairs)
    covered_pairs = []
    fallback_pairs = []
    for ik in all_pairs_list:
        i_idx, k_idx = ik
        key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
        n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
        if n_ik == 0:
            continue
        if key_ik in cc_ints and cc_ints[key_ik] is not None:
            covered_pairs.append(ik)
        else:
            fallback_pairs.append(ik)

    if covered_pairs:
        N = len(covered_pairs)
        plan_key = (id(cc_ints), tuple(covered_pairs),
                    id(pair_lmo_idx) if pair_lmo_idx is not None else 0)
        plan = getattr(build_D_tilde_batched, '_ph1_plan', None)
        if plan is None or plan.get('key') != plan_key:
            # Fixes #3 + #1 + #2 algorithmic parity with Psi4:
            # Precompute K_tilde_chem_k (L pre-summed k_Qa×Qab) and
            # M_static = 2*K_bar_ij_or_ji[ll_idx] - K_bar_chem[ll_idx].
            # Per-iter kernel is 2 dgemv + 1 dgemm on precomputed tensors.
            n_pno_arr = np.zeros(N, dtype=np.int32)
            n_domain_arr = np.zeros(N, dtype=np.int32)
            K_tilde_chem_list = [None] * N
            M_static_list = [None] * N
            ll_idx_list = [None] * N
            key_ik_list = [None] * N
            i_idx_list = [None] * N

            for p, ik in enumerate(covered_pairs):
                i_idx, k_idx = ik
                key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
                ci = cc_ints[key_ik]
                n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
                if pair_lmo_idx is not None and key_ik in pair_lmo_idx:
                    ll_idx = np.asarray(pair_lmo_idx[key_ik], dtype=np.intp)
                else:
                    ll_idx = np.arange(nocc, dtype=np.intp)
                n_domain = ll_idx.size
                n_local = ci['Qma'].shape[0]

                is_k_first = (key_ik[0] == k_idx)
                is_i_first = (key_ik[0] == i_idx)
                # K_tilde_chem precomputed in cc_ints — shared across
                # compute_C_tilde / build_D_tilde / T1 residual.
                K_tilde_chem_list[p] = (ci['K_tilde_chem_i'] if is_k_first
                                        else ci['K_tilde_chem_j'])

                # M_static = 2 * K_bar_ij_or_ji[ll_idx] - K_bar_chem[ll_idx]
                # All three K_bar_* are reduced to (nlmo_p, npno); translate
                # ll_idx -> p_lmos-domain position before indexing.
                _ll_idx_in_p = np.asarray(
                    ci['p_lmos_dense'])[ll_idx].astype(np.intp)
                if is_i_first:
                    K_bar_slice = ci['K_bar_ij'][_ll_idx_in_p]
                else:
                    K_bar_slice = ci['K_bar_ji'][_ll_idx_in_p]
                M_static_list[p] = np.ascontiguousarray(
                    2.0 * K_bar_slice - ci['K_bar_chem'][_ll_idx_in_p])

                n_pno_arr[p] = n_ik
                n_domain_arr[p] = n_domain
                ll_idx_list[p] = ll_idx
                key_ik_list[p] = key_ik
                i_idx_list[p] = i_idx

            def _flat(arrs):
                sizes = np.array([a.size for a in arrs], dtype=np.int64)
                offsets = np.empty(len(arrs) + 1, dtype=np.int64)
                offsets[0] = 0
                offsets[1:] = np.cumsum(sizes)
                buf = np.empty(int(offsets[-1]))
                for idx, a in enumerate(arrs):
                    buf[offsets[idx]:offsets[idx + 1]] = a.ravel()
                return buf, offsets

            K_tilde_chem_flat, K_tilde_chem_off = _flat(K_tilde_chem_list)
            M_static_flat, M_static_off = _flat(M_static_list)

            t1_sizes = n_pno_arr.astype(np.int64)
            t1_off = np.empty(N + 1, dtype=np.int64)
            t1_off[0] = 0
            t1_off[1:] = np.cumsum(t1_sizes)

            T1_rows_sizes = (n_domain_arr.astype(np.int64)
                             * n_pno_arr.astype(np.int64))
            T1_rows_off = np.empty(N + 1, dtype=np.int64)
            T1_rows_off[0] = 0
            T1_rows_off[1:] = np.cumsum(T1_rows_sizes)

            D_sizes = n_pno_arr.astype(np.int64) ** 2
            D_off = np.empty(N + 1, dtype=np.int64)
            D_off[0] = 0
            D_off[1:] = np.cumsum(D_sizes)

            max_n_pno = int(n_pno_arr.max())
            num_threads = min(32, N)

            scratch = {
                'part1': np.empty((num_threads, max_n_pno * max_n_pno)),
                'part2': np.empty((num_threads, max_n_pno * max_n_pno)),
            }
            for buf in scratch.values():
                buf.fill(0.0)

            plan = {
                'key': plan_key,
                'covered_pairs': covered_pairs,
                'key_ik_list': key_ik_list,
                'i_idx_list': i_idx_list,
                'll_idx_list': ll_idx_list,
                'n_pno_arr': n_pno_arr,
                'n_domain_arr': n_domain_arr,
                'K_tilde_chem_flat': K_tilde_chem_flat,
                'K_tilde_chem_off': K_tilde_chem_off,
                'M_static_flat': M_static_flat,
                'M_static_off': M_static_off,
                't1_off': t1_off, 't1_total': int(t1_off[-1]),
                'T1_rows_off': T1_rows_off,
                'T1_rows_total': int(T1_rows_off[-1]),
                'D_off': D_off, 'D_total': int(D_off[-1]),
                'scratch': scratch, 'num_threads': num_threads,
            }
            build_D_tilde_batched._ph1_plan = plan

        t1_flat = np.empty(plan['t1_total'])
        T1_rows_flat = np.empty(plan['T1_rows_total'])
        t1_off_plan = plan['t1_off']
        T1_rows_off_plan = plan['T1_rows_off']
        for p in range(N):
            key_ik = plan['key_ik_list'][p]
            i_idx = plan['i_idx_list'][p]
            n_pno = int(plan['n_pno_arr'][p])
            ll_idx = plan['ll_idx_list'][p]
            n_dom = ll_idx.size
            T1_all_ik = t1_cache[key_ik]
            t1_flat[t1_off_plan[p]:t1_off_plan[p + 1]] = T1_all_ik[i_idx]
            rows_buf = T1_rows_flat[
                T1_rows_off_plan[p]:T1_rows_off_plan[p + 1]
            ].reshape(n_dom, n_pno)
            rows_buf[:] = T1_all_ik[ll_idx]

        D_flat = np.zeros(plan['D_total'])
        sc = plan['scratch']

        with threadpool_limits(limits=1, user_api='blas'):
            d_tilde_ph1_batched(
                plan['K_tilde_chem_flat'], plan['K_tilde_chem_off'],
                plan['M_static_flat'], plan['M_static_off'],
                t1_flat, t1_off_plan,
                T1_rows_flat, T1_rows_off_plan,
                plan['n_pno_arr'], plan['n_domain_arr'],
                sc['part1'], sc['part2'],
                D_flat, plan['D_off'],
                plan['num_threads'],
            )

        D_off_plan = plan['D_off']
        for p, ik in enumerate(plan['covered_pairs']):
            n_pno = int(plan['n_pno_arr'][p])
            D_tilde_all[ik] = (
                D_flat[D_off_plan[p]:D_off_plan[p + 1]]
                .reshape(n_pno, n_pno).copy())

    # Fallback Python path for pairs not covered by cc_ints
    for ik in fallback_pairs:
        _, val = _process_ik_t12(ik)
        if val is not None:
            D_tilde_all[ik] = val

    # --- Phase 2: Terms 3 + 4 via plan-cached Cython kernels ---
    plan_key = tuple(sorted(t2_pno_all.keys()))
    _cache_attr = getattr(build_D_tilde_batched, '_plan_cache', None)
    if _cache_attr is None:
        _cache_attr = {}
        build_D_tilde_batched._plan_cache = _cache_attr
    plan = _cache_attr.get(plan_key)
    if plan is None:
        plan = _build_d_tilde_t34_plan(
            all_pairs, pno_spaces, pair_lmo_idx, t2_pno_all,
            S_pno_cache, cc_ints, _s_pno_get, nocc)
        _cache_attr[plan_key] = plan

    from pyscf.cc.dlpno_tccsd._c_tilde_cy import (
        t3_kernel as _t3_kern, t4_kernel as _t4_kern,
    )

    # Per-n_ik flat buffers carrying Phase 1 (Terms 1 + 2) results; the
    # Cython kernels accumulate Terms 3 + 4 on top, then we unpack back
    # into D_tilde_all.
    flat_out = {}
    for n_ik, pairs in plan['pairs_by_n_ki'].items():
        buf = np.zeros((len(pairs), n_ik, n_ik))
        for slot, pair in enumerate(pairs):
            v = D_tilde_all.get(pair)
            if v is not None:
                buf[slot] = v
        flat_out[n_ik] = buf

    # Single-call batched path: ONE t3_kernel_batched + ONE t4_kernel_batched
    # per cycle replaces the per-bucket loop. Plan view (built once) flattens
    # all per-bucket K/S static tensors into single concatenated buffers and
    # pre-computes per-item t1 absolute offsets into t1_cache._buffer.
    bv = _get_or_build_t34_batched_view(plan, t1_cache, t2_pno_all)
    _run_t34_batched(plan, bv, t1_cache, t2_pno_all, flat_out,
                     t4_use_u=True, t4_scale=0.5)

    for n_ik, pairs in plan['pairs_by_n_ki'].items():
        buf = flat_out[n_ik]
        for slot, pair in enumerate(pairs):
            D_tilde_all[pair] = buf[slot]

    # DTILDE_DUMP: parity dump vs Psi4 ccsd.cc:1964 compute_D_tilde output.
    if int(os.environ.get('DLPNO_DUMP_DTILDE', '0')):
        _it = getattr(build_D_tilde_batched, '_iter', 0)
        build_D_tilde_batched._iter = _it + 1
        if _it <= 2:
            tot_fro2 = 0.0
            tot_tr = 0.0
            tot_sum = 0.0
            n_pairs = 0
            for key, M in D_tilde_all.items():
                if M is None or M.size == 0:
                    continue
                n_pairs += 1
                tot_fro2 += float(np.sum(M ** 2))
                tot_tr += float(np.trace(M))
                tot_sum += float(M.sum())
            print(f"DTILDE_DUMP iter={_it} n_pairs={n_pairs} "
                  f"fro2={tot_fro2:.12e} tr={tot_tr:.12e} "
                  f"sum={tot_sum:.12e}", flush=True)

    return D_tilde_all


def build_mixed_domain_integrals(t2_pno_all, pno_spaces, nocc,
                                 ovL_bare, ooL_bare, S_pno_cache,
                                 cc_ints=None):
    """Build J_ij_kj and K_ij_kj: mixed-domain bare integrals for C/D terms.

    J_ij_kj[(key_ij, k)] = (ik|a_ij c_kj) = Coulomb with mixed PNO domains
    K_ij_kj[(key_ij, k)] = (ia_ij|kc_kj) = exchange with mixed PNO domains

    These are the "bold" terms in Eqs 78-79.
    """
    J_cache = {}
    K_cache = {}

    keys_sorted = sorted(t2_pno_all.keys())
    for key_ij in keys_sorted:
        i, j = key_ij
        n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_ij == 0:
            continue

        for k in range(nocc):
            key_kj = (min(k, j), max(k, j))
            if key_kj not in pno_spaces:
                continue
            n_kj = pno_spaces[key_kj]['C_pno'].shape[1]
            if n_kj == 0:
                continue

            # J(ik|a_ij c_kj) = Σ_Q ooL[i,k,Q] * vvL_mixed[a_ij, c_kj, Q]
            # But vvL_mixed requires mixed PNO basis transform.
            # For the Coulomb integral: (ik|ac) = ooL[i,k,Q] * (ac|Q)
            # where a is in PNO_ij and c is in PNO_kj.
            # (ac|Q) = C_pno_ij[:,a].T @ Lpq @ C_pno_kj[:,c] → mixed DF
            # This is exactly what K_coul_cache stores!
            # K_coul_cache[(key_ij, key_kj, i, k)] = (ik|a_ij c_kj)
            # So J_ij_kj = K_coul_cache[(key_ij, key_kj, i, k)]
            # Already computed! Just reference it.
            # We pass K_coul_cache directly instead.

            # K(ia_ij|kc_kj) = ovL_i_ij[a,Q] * ovL_k_kj[c,Q]
            if key_ij in cc_ints and cc_ints[key_ij] is not None:
                _K_local = cc_ints[key_ij].get('K_ij_kj', {}).get((key_ij, k))
                if _K_local is not None:
                    K_cache[(key_ij, k)] = _K_local
                else:
                    ovL_i_ij = ovL_bare.get((key_ij, i))
                    ovL_k_kj = ovL_bare.get((key_kj, k))
                    if ovL_i_ij is not None and ovL_k_kj is not None:
                        K_cache[(key_ij, k)] = ovL_i_ij @ ovL_k_kj.T
            else:
                ovL_i_ij = ovL_bare.get((key_ij, i))
                ovL_k_kj = ovL_bare.get((key_kj, k))
                if ovL_i_ij is not None and ovL_k_kj is not None:
                    K_cache[(key_ij, k)] = ovL_i_ij @ ovL_k_kj.T

    return K_cache

# =========================================================================
# Dressed DF integrals (ooL/ovL) and C_tilde (Eqs 83, 91-93)
# =========================================================================
# ---------------------------------------------------------------------------
# Dressed DF integral builders (Eqs 91-93)
# ---------------------------------------------------------------------------



def compute_C_tilde(t1_pno, t2_pno_all, pno_spaces, nocc,
                    ovL_pno_bare, ooL_bare, S_pno_cache, with_df,
                    _term2_precomputed=None, cc_ints=None,
                    pair_lmo_idx=None, _pool=None,
                    S_pao_full=None, s1e=None,
                    t1_cache=None):
    """Compute C_tilde (gamma intermediate, Eq 83) for all pairs.

    C_tilde[ki][a_ki, c_ki] = gamma_{ki}^{ac} =
        + Σ_b t̃_i^b · (kb|ac)              [Term 2: vovv × T1_i]
        - Σ_l t̃_l^a · (ki|lc)              [Term 1: ooov × T1_all]
        - Σ_l t̃_l^a · (K_kl × T1_i)        [Term 3: T1² × K]
        - (1/2) Σ_l S @ t2_li @ S @ K_kl @ S [Term 4: T2 × K]

    Following Psi4 ccsd.cc compute_C_tilde() lines 1694-1751.

    Returns dict: pair_key → (n_pno_ki, n_pno_ki) matrix.
    """
    C_tilde_all = {}
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    # Phase 1: t1 projections are cached once per CCSD cycle.  If caller
    # passed ``t1_cache``, use it; otherwise build a local one (back-compat
    # for callers that predate the cache-based migration).
    if t1_cache is None:
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, build_t1_cache,
        )
        _pi = PairIndex(
            pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
        t1_cache = build_t1_cache(
            t1_pno, _pi, S_pno_cache, pno_spaces)

    # Iterate over ALL ordered (k,i) pairs, not just min/max.
    # gamma_{ki} depends on which LMO is "i" (the T1 source).
    all_pairs = set()
    for key in t2_pno_all:
        k, i = key  # stored as (min, max)
        all_pairs.add((k, i))
        all_pairs.add((i, k))

    # Term 2 can be precomputed via unified DF pass
    if _term2_precomputed is None:
        # Batched Term 2 via single DF pass (fallback)
        term2_data = {}
        for k, i in all_pairs:
            key_ki = (min(k, i), max(k, i))
            n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
            if n_ki == 0:
                continue
            t1_i_ki = t1_cache[key_ki][i]
            if np.max(np.abs(t1_i_ki)) < 1e-15:
                continue
            ovL_i_ki = ovL_pno_bare.get((key_ki, i))
            if ovL_i_ki is None:
                continue
            term2_data[(k, i)] = {
                'key_ki': key_ki, 'n_ki': n_ki,
                'C_pno': np.asfortranarray(pno_spaces[key_ki]['C_pno']),
                't1_i': t1_i_ki, 'ovL_i': ovL_i_ki,
                'result': np.zeros((n_ki, n_ki)),
            }
        if term2_data:
            aux_off = 0
            for Lpq in with_df.loop():
                nL = Lpq.shape[0]
                for (k, i), td in term2_data.items():
                    n_ki = td['n_ki']
                    buf = _ao2mo.nr_e2(Lpq, td['C_pno'],
                                       (0, n_ki, 0, n_ki), aosym='s2')
                    B_L = buf.reshape(nL, n_ki, n_ki)
                    z_i = td['t1_i'] @ td['ovL_i'][:, aux_off:aux_off+nL]
                    td['result'] += np.tensordot(z_i, B_L, axes=(0, 0))
                aux_off += nL
        _term2_precomputed = {ki: td['result'] for ki, td in term2_data.items()}

    def _process_ki(ki_tuple):
        k, i = ki_tuple
        key_ki = (min(k, i), max(k, i))  # storage key
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        if n_ki == 0:
            return ki_tuple, None

        C_tilde_ki = np.zeros((n_ki, n_ki))
        _dump_terms = ((k, i) == (10, 10)
                       and getattr(compute_C_tilde, '_dump_iter', -1) == 1)
        _term_log = {} if _dump_terms else None

        # Term 2: Σ_b t1_i^b · (kb|ac) = Σ_{Q,b} t1_i[b]*k_Qa[Q,b]*Qab[Q,a,c]
        # Use local DF when cc_ints available, otherwise precomputed (global DF)
        if key_ki in cc_ints and cc_ints[key_ki] is not None:
            ci_ki = cc_ints[key_ki]
            if key_ki[0] == k:
                k_Qa = ci_ki['i_Qa']
            else:
                k_Qa = ci_ki['j_Qa']
            Qab_ki = ci_ki['Qab']
            t1_i_ki = t1_cache[key_ki][i]
            z = k_Qa @ t1_i_ki
            T2_contrib = np.tensordot(z, Qab_ki, axes=(0, 0))
            C_tilde_ki += T2_contrib
            if _dump_terms:
                ev = np.sort(np.linalg.eigvalsh(0.5*(T2_contrib+T2_contrib.T)))[::-1]
                _term_log['T2'] = (float(np.sqrt(np.mean(T2_contrib**2))), ev[0], ev[-1])
        elif (k, i) in _term2_precomputed:
            C_tilde_ki += _term2_precomputed[(k, i)]

        # --- Term 1: -Σ_l T1_all[l,a] · (ki|lc) ---
        # Restricted to l ∈ lmopair_to_lmos_[ki] (Psi4 line 1722).
        if pair_lmo_idx is not None and key_ki in pair_lmo_idx:
            _ll_idx = np.asarray(pair_lmo_idx[key_ki])
        else:
            _ll_idx = np.arange(nocc)
        _domain_ki = set(int(x) for x in _ll_idx)

        # Phase 1: fancy-index into cached (nocc, n_pno) matrix replaces
        # the per-l projection loop.
        T1_local_ki = np.ascontiguousarray(
            t1_cache[key_ki][np.asarray(_ll_idx, dtype=np.intp)])

        K_bar_chem_local = np.zeros((len(_ll_idx), n_ki))
        if key_ki in cc_ints and cc_ints[key_ki] is not None:
            from pyscf.cc.dlpno_tccsd.local_df import get_local_ovL, get_local_ooL_vec
            _ooL_ki = get_local_ooL_vec(cc_ints, k, i, key_ki)
            if _ooL_ki is not None:
                for _li, _ll in enumerate(_ll_idx):
                    _ovL_l = get_local_ovL(cc_ints, key_ki, int(_ll))
                    if _ovL_l is not None:
                        K_bar_chem_local[_li] = _ovL_l @ _ooL_ki
            else:
                ooL_ki = ooL_bare[k, i, :]
                for _li, _ll in enumerate(_ll_idx):
                    ovL_l_ki = ovL_pno_bare.get((key_ki, int(_ll)))
                    if ovL_l_ki is not None:
                        K_bar_chem_local[_li] = ovL_l_ki @ ooL_ki
        else:
            ooL_ki = ooL_bare[k, i, :]
            for _li, _ll in enumerate(_ll_idx):
                ovL_l_ki = ovL_pno_bare.get((key_ki, int(_ll)))
                if ovL_l_ki is not None:
                    K_bar_chem_local[_li] = ovL_l_ki @ ooL_ki

        T1_contrib = -T1_local_ki.T @ K_bar_chem_local
        C_tilde_ki += T1_contrib  # (n_ki, n_ki)

        # Scatter T1_local_ki into T1_all_ki for downstream Term 3/Term 4
        # code that still uses full-nocc indexing (T1_all_ki[ll]).
        T1_all_ki = np.zeros((nocc, n_ki))
        for _li, _ll in enumerate(_ll_idx):
            T1_all_ki[int(_ll)] = T1_local_ki[_li]
        if _dump_terms:
            ev = np.sort(np.linalg.eigvalsh(0.5*(T1_contrib+T1_contrib.T)))[::-1]
            _term_log['T1'] = (float(np.sqrt(np.mean(T1_contrib**2))), ev[0], ev[-1])

        # --- Term 3: -Σ_l T1_l^a · (S @ K_kl @ T1_i_kl) ---
        # Psi4 restricts l to lmopair_to_lmos_[ki] (ccsd.cc:1724)
        T3_contrib = np.zeros_like(C_tilde_ki) if _dump_terms else None
        for ll in range(nocc):
            if ll not in _domain_ki:
                continue
            key_kl = (min(k, ll), max(k, ll))
            if key_kl not in pno_spaces:
                continue
            n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
            if n_kl == 0:
                continue
            # K_kl[a_kl, b_kl] = (ka|lb) = exchange in PNO_kl
            K_kl = get_local_K(cc_ints, key_kl, k, ll)
            if K_kl is None:
                continue

            # T1_i projected to PNO_kl
            t1_i_kl = t1_cache[key_kl][i]
            if np.max(np.abs(t1_i_kl)) < 1e-15:
                continue

            # Per Eq 83: t1_i contracts with b (first index of K_kl^{bc} where
            # K_iajb[kl][a,b]=(ka|lb) has "a" at k-partner position).
            Kt1 = K_kl.T @ t1_i_kl  # (n_kl,)
            S_ki_kl = _s_pno_get(key_ki, key_kl)
            if S_ki_kl is None:
                continue
            Kt1_ki = S_ki_kl @ Kt1  # (n_ki,)
            t1_l_ki = T1_all_ki[ll]
            contrib3 = -np.outer(t1_l_ki, Kt1_ki)
            C_tilde_ki += contrib3
            if _dump_terms:
                T3_contrib += contrib3
        if _dump_terms:
            ev = np.sort(np.linalg.eigvalsh(0.5*(T3_contrib+T3_contrib.T)))[::-1]
            _term_log['T3'] = (float(np.sqrt(np.mean(T3_contrib**2))), ev[0], ev[-1])

        # --- Term 4: -(1/2) Σ_l S @ t2_li @ S @ K_kl @ S ---
        # Psi4 restricts l to lmopair_to_lmos_[ki] (ccsd.cc:1736)
        T4_contrib = np.zeros_like(C_tilde_ki)
        for ll in range(nocc):
            if ll not in _domain_ki:
                continue
            key_li = (min(ll, i), max(ll, i))
            key_kl = (min(k, ll), max(k, ll))
            if key_li not in t2_pno_all or key_kl not in pno_spaces:
                continue
            t2_li_raw = t2_pno_all.get(key_li)
            if t2_li_raw is None or t2_li_raw.shape[0] == 0:
                continue
            n_li = pno_spaces[key_li]['C_pno'].shape[1]
            n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
            if n_li == 0 or n_kl == 0:
                continue

            t2_li = t2_li_raw.T if ll > i else t2_li_raw

            K_kl = get_local_K(cc_ints, key_kl, k, ll)
            if K_kl is None:
                continue

            def _get_S_or_I(ka, kb, n_a):
                if ka == kb:
                    return np.eye(n_a)
                return _s_pno_get(ka, kb)

            n_ki2 = pno_spaces[key_ki]['C_pno'].shape[1]
            S_ki_li = _get_S_or_I(key_ki, key_li, n_ki2)
            S_li_kl = _get_S_or_I(key_li, key_kl, n_li)
            S_kl_ki = _get_S_or_I(key_kl, key_ki, n_kl)
            if S_ki_li is None or S_li_kl is None or S_kl_ki is None:
                continue

            # Psi4: S(ki,li) @ T[li] @ S(li,kl) @ K[kl] @ S(kl,ki)
            C_temp = S_ki_li @ t2_li @ S_li_kl @ K_kl @ S_kl_ki
            contrib4 = -0.5 * C_temp
            C_tilde_ki += contrib4  # Term 4: Jiang Eq 83
            T4_contrib += contrib4
        if _dump_terms:
            ev = np.sort(np.linalg.eigvalsh(0.5*(T4_contrib+T4_contrib.T)))[::-1]
            _term_log['T4'] = (float(np.sqrt(np.mean(T4_contrib**2))), ev[0], ev[-1])
            for _tname, (_trms, _t0, _tl) in _term_log.items():
                print(f"  CTILDE_TERMS_US iter 1 ki=(10,10) {_tname}: "
                      f"rms={_trms:.6e} ev_top={_t0:.6e} ev_bot={_tl:.6e}",
                      flush=True)

        return ki_tuple, C_tilde_ki  # store with ORDERED (k,i) key

    if _pool is not None:
        for ki, val in _pool.map(_process_ki, list(all_pairs)):
            if val is not None:
                C_tilde_all[ki] = val
    else:
        for ki in all_pairs:
            _, val = _process_ki(ki)
            if val is not None:
                C_tilde_all[ki] = val
    return C_tilde_all


# =========================================================================
# Prototype: compute_C_tilde_batched (plan-cached)
# =========================================================================
# Architecture test for "no per-pair Python loops; batched matmul with
# BLAS threading."  Terms 1 and 2 run per-pair (same code path as the
# reference).  Terms 3 and 4 — the per-l inner Python loops inside the
# reference — are replaced with a gather/bucket/batched-matmul/scatter
# pipeline across ALL (k, i, l) triples, with BLAS threading enabled for
# the matmul phase via threadpool_limits.
#
# The "plan" (bucket shape grouping, pre-stacked constant tensors K/S,
# scatter indices) is built once per run and cached on the function
# object.  Per-cycle work is only: gather T1/T2 values (which change),
# batched matmul, scatter.
# =========================================================================


def _build_c_tilde_t34_plan(
        all_pairs, pno_spaces, pair_lmo_idx, t2_pno_all,
        S_pno_cache, cc_ints, _s_pno_get, nocc):
    """Build the one-time plan for Terms 3 and 4.

    Returns dict with keys ``'t3'`` and ``'t4'``, each a list of per-shape
    bucket dicts carrying pre-stacked constant tensors (K, S) and the
    index arrays for T1/T2 gather + output scatter.  Per-cycle execution
    only needs to gather T1/T2 values and run batched matmuls; no Python
    loop over triples.
    """
    from pyscf.cc.dlpno_tccsd.local_df import get_local_K

    # --- Collect all valid (k, i, l) triples for Terms 3 and 4 ---
    t3_items = []  # (k, i, l, K_kl, S_ki_kl, key_kl, key_ki, n_ki, n_kl)
    t4_items = []  # (k, i, S_ki_li, S_li_kl, K_kl, S_kl_ki, key_li, transpose, n_ki, n_kl, n_li)

    for (k, i) in all_pairs:
        key_ki = (min(k, i), max(k, i))
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        if n_ki == 0:
            continue
        if pair_lmo_idx is not None and key_ki in pair_lmo_idx:
            _ll_idx = pair_lmo_idx[key_ki]
        else:
            _ll_idx = np.arange(nocc)

        for ll_raw in _ll_idx:
            ll = int(ll_raw)
            key_kl = (min(k, ll), max(k, ll))

            # Term 3 gather
            if key_kl in pno_spaces:
                n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
                if n_kl > 0:
                    K_kl = get_local_K(cc_ints, key_kl, k, ll)
                    if K_kl is not None:
                        S_ki_kl = _s_pno_get(key_ki, key_kl)
                        if S_ki_kl is not None:
                            t3_items.append((k, i, ll, K_kl, S_ki_kl,
                                             key_kl, key_ki, n_ki, n_kl))

            # Term 4 gather
            key_li = (min(ll, i), max(ll, i))
            if key_li not in t2_pno_all:
                continue
            if key_kl not in pno_spaces:
                continue
            n_li = pno_spaces[key_li]['C_pno'].shape[1]
            n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
            if n_li == 0 or n_kl == 0:
                continue
            K_kl = get_local_K(cc_ints, key_kl, k, ll)
            if K_kl is None:
                continue
            transpose = (ll > i)
            S_ki_li = (np.eye(n_ki) if key_ki == key_li
                       else _s_pno_get(key_ki, key_li))
            S_li_kl = (np.eye(n_li) if key_li == key_kl
                       else _s_pno_get(key_li, key_kl))
            S_kl_ki = (np.eye(n_kl) if key_kl == key_ki
                       else _s_pno_get(key_kl, key_ki))
            if S_ki_li is None or S_li_kl is None or S_kl_ki is None:
                continue
            t4_items.append((k, i, S_ki_li, S_li_kl, K_kl, S_kl_ki,
                             key_li, transpose, n_ki, n_kl, n_li))

    # --- Global pair→(n_ki, flat_slot) map: covers Phase 1 + Phase 2 outputs ---
    # Every (k, i) with n_ki > 0 gets a slot in flat_out[n_ki].  Phase 1
    # results are copied into flat_out before Numba kernels accumulate
    # Terms 3 + 4 on top; at function exit flat_out is unpacked into
    # C_tilde_all dict.
    pairs_by_n_ki = {}
    pair_to_slot = {}
    for (k, i) in all_pairs:
        key_ki = (min(k, i), max(k, i))
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        if n_ki == 0:
            continue
        if n_ki not in pairs_by_n_ki:
            pairs_by_n_ki[n_ki] = []
        slot = len(pairs_by_n_ki[n_ki])
        pairs_by_n_ki[n_ki].append((k, i))
        pair_to_slot[(k, i)] = slot

    # --- Bucket Term 3 by (n_ki, n_kl); pre-stack K, S, gather index arrays ---
    t3_by_shape = {}
    for it in t3_items:
        t3_by_shape.setdefault((it[7], it[8]), []).append(it)

    t3_buckets = []
    for (n_ki, n_kl), items in t3_by_shape.items():
        N = len(items)
        K = np.empty((N, n_kl, n_kl))
        S = np.empty((N, n_ki, n_kl))
        t1i_keys = []      # list of (key_kl, i)  → T1_cache lookup for t1_i in PNO_kl
        T1l_keys = []      # list of (key_ki, l)  → T1_cache lookup for t1_l in PNO_ki
        # item_idx points into flat_out[n_ki] directly (GLOBAL slot),
        # so the Numba kernel can scatter without per-bucket unpacking.
        item_idx = np.empty(N, dtype=np.intp)
        for n, it in enumerate(items):
            k, i, ll = it[0], it[1], it[2]
            K[n] = it[3]
            S[n] = it[4]
            key_kl, key_ki = it[5], it[6]
            t1i_keys.append((key_kl, i))
            T1l_keys.append((key_ki, ll))
            item_idx[n] = pair_to_slot[(k, i)]
        t3_buckets.append({
            'K': K, 'S': S,
            't1i_keys': t1i_keys, 'T1l_keys': T1l_keys,
            'n_ki': n_ki, 'n_kl': n_kl,
            'item_idx': item_idx,
        })

    # --- Bucket Term 4 by (n_ki, n_kl, n_li); pre-stack 4 S matrices + K ---
    t4_by_shape = {}
    for it in t4_items:
        t4_by_shape.setdefault((it[8], it[9], it[10]), []).append(it)

    t4_buckets = []
    for (n_ki, n_kl, n_li), items in t4_by_shape.items():
        N = len(items)
        S_ki_li = np.empty((N, n_ki, n_li))
        S_li_kl = np.empty((N, n_li, n_kl))
        K = np.empty((N, n_kl, n_kl))
        S_kl_ki = np.empty((N, n_kl, n_ki))
        t2_sources = []    # list of (key_li, transpose)
        item_idx = np.empty(N, dtype=np.intp)
        for n, it in enumerate(items):
            k, i = it[0], it[1]
            S_ki_li[n] = it[2]
            S_li_kl[n] = it[3]
            K[n] = it[4]
            S_kl_ki[n] = it[5]
            t2_sources.append((it[6], bool(it[7])))
            item_idx[n] = pair_to_slot[(k, i)]
        t4_buckets.append({
            'S_ki_li': S_ki_li, 'S_li_kl': S_li_kl,
            'K': K, 'S_kl_ki': S_kl_ki,
            't2_sources': t2_sources,
            'n_ki': n_ki, 'n_kl': n_kl, 'n_li': n_li,
            'item_idx': item_idx,
        })

    return {
        't3': t3_buckets, 't4': t4_buckets,
        'pairs_by_n_ki': pairs_by_n_ki,
        'pair_to_slot': pair_to_slot,
    }


def compute_C_tilde_psi4(
        t1_pno, t2_pno_all, pno_spaces, nocc,
        ovL_pno_bare, ooL_bare, S_pno_cache, with_df,
        _term2_precomputed=None, cc_ints=None,
        pair_lmo_idx=None, _pool=None,
        S_pao_full=None, s1e=None,
        blas_threads=32, omp_threads=None,
        t1_cache=None):
    """Compute C_tilde — Psi4 per-pair port matching ccsd.cc:1809 line-by-line.

    For each ordered pair (k, i):
      Term 2 (Psi4 1841): T_i = S(ki, ii) @ T_ia[i]
                          C_tilde[ki] += (T_i @ K_tilde_chem[ki]).reshape(n_ki, n_ki)
      Term 1 (Psi4 1847): C_tilde[ki] -= T_n_ij[ki].T @ K_bar_chem[ki]
      Term 3 (Psi4 1855): for l in lmopair_to_lmos_[ki]:
                              T_l    = S(ki, ll) @ T_ia[l]
                              T_i_kl = S(kl, ii) @ T_ia[i]
                              K_kl   = S(ki, kl) @ K_iajb[kl] @ T_i_kl
                              C_tilde[ki] -= np.outer(T_l, K_kl)
      Term 4 (Psi4 1896): for l in lmopair_to_lmos_[ki]:
                              X = T_iajb[li] @ S(li, kl) @ K_iajb[kl]
                              Y = S(ki, li) @ X @ S(kl, ki)
                              C_tilde[ki] -= 0.5 * Y

    Drop-in replacement for compute_C_tilde_batched. Hot inner loops
    (Terms 3+4) will move into a per-pair C kernel in
    pyscf/lib/cc/dlpno_c_tilde.c (TODO).
    """
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    # All ordered pairs (k, i) — Psi4 iterates BOTH (k, i) and (i, k).
    all_ordered = []
    for key in t2_pno_all:
        a, b = key
        all_ordered.append((a, b))
        if a != b:
            all_ordered.append((b, a))

    # Build t1_cache locally if caller didn't pass it (back-compat).
    if t1_cache is None:
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, build_t1_cache,
        )
        _pi = PairIndex(
            pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
        t1_cache = build_t1_cache(t1_pno, _pi, S_pno_cache, pno_spaces)

    def _t2_ordered(l, i):
        """Return T_iajb[(l, i)] for ordered pair (l, i)."""
        canon = (min(l, i), max(l, i))
        t2 = t2_pno_all.get(canon)
        if t2 is None or t2.shape[0] == 0:
            return None
        return t2 if (l, i) == canon else t2.T

    def _per_pair(key_ki):
        k, i = key_ki
        canonical = (min(k, i), max(k, i))
        n_ki = pno_spaces[canonical]['C_pno'].shape[1]
        if n_ki == 0:
            return key_ki, np.zeros((0, 0))
        ci_ki = cc_ints.get(canonical) if cc_ints is not None else None
        if ci_ki is None:
            # Fallback to legacy DF-rebuild path (rarely hit when cc_ints
            # covers all strong+weak pairs as in normal runs).
            return key_ki, np.zeros((n_ki, n_ki))

        ii = (i, i)
        n_ii = pno_spaces[ii]['C_pno'].shape[1] if ii in pno_spaces else 0
        C_tilde = np.zeros((n_ki, n_ki))

        # ---- Term 2 (Psi4 ccsd.cc:1841-1844) ----
        # T_i_in_ki = S(ki, ii) @ T_ia[i]
        # K_temp    = T_i.T @ K_tilde_chem[ki]    shape (n_ki**2,)
        # C_tilde += K_temp.reshape(n_ki, n_ki)
        t1_i_pno = t1_pno.get(i)
        if (t1_i_pno is not None and t1_i_pno.size > 0
                and np.max(np.abs(t1_i_pno)) > 1e-15
                and ii in pno_spaces and n_ii > 0):
            S_ki_ii = (np.eye(n_ki) if canonical == ii
                       else _s_pno_get(canonical, ii))
            if S_ki_ii is not None:
                T_i_in_ki = S_ki_ii @ t1_i_pno              # (n_ki,)
                # K_tilde_chem orientation: if k is canonical's first idx,
                # use the i-side K_tilde_chem; else j-side.
                is_k_first = (canonical[0] == k)
                K_tilde_chem = (ci_ki['K_tilde_chem_i'] if is_k_first
                                else ci_ki['K_tilde_chem_j'])
                # K_tilde_chem is (n_ki, n_ki**2).
                K_temp = T_i_in_ki @ K_tilde_chem            # (n_ki**2,)
                C_tilde += K_temp.reshape(n_ki, n_ki)

        # ---- Term 1 (Psi4 ccsd.cc:1847) ----
        # C_tilde[ki] -= T_n_ij[ki].T @ K_bar_chem[ki]
        # t1_cache[canonical] is (nocc, n_ki); gather to pair domain to
        # match K_bar_chem (nlmo_pair, n_ki) shape post-Phase III.
        p_lmos = ci_ki['p_lmos']
        T_n_ki = t1_cache[canonical][p_lmos]          # (nlmo_pair, n_ki)
        K_bar_chem = ci_ki['K_bar_chem']              # (nlmo_pair, n_ki)
        C_tilde -= T_n_ki.T @ K_bar_chem

        # Determine pair-domain LMO list (Psi4 lmopair_to_lmos_[ki]).
        if pair_lmo_idx is not None and canonical in pair_lmo_idx:
            domain = [int(x) for x in pair_lmo_idx[canonical]]
        else:
            domain = list(range(nocc))

        # ---- Term 3 (Psi4 ccsd.cc:1855-1887): outer product per l ----
        # Psi4: K_kl = S(ki, kl) @ K_iajb_[kl_ordered].T @ T_i_kl
        # For ordered (k, l): K_iajb_[ordered] = canonical if (k,l) == canon
        # else canonical.T. So K_iajb_[ordered].T = canonical.T or canonical.
        for l in domain:
            kl_canon = (min(k, l), max(k, l))
            ll_key = (l, l)
            if kl_canon not in pno_spaces or ll_key not in pno_spaces:
                continue
            n_kl = pno_spaces[kl_canon]['C_pno'].shape[1]
            if n_kl == 0:
                continue
            t1_l_pno = t1_pno.get(l)
            if t1_l_pno is None or t1_l_pno.size == 0:
                continue
            t1_i_pno_loc = t1_pno.get(i)
            if t1_i_pno_loc is None or t1_i_pno_loc.size == 0:
                continue
            ci_kl = cc_ints.get(kl_canon)
            if ci_kl is None:
                continue
            # K_iajb (canonical) — pick orientation matching Psi4.
            kl_swap = (k > l)  # ordered (k, l) != canonical when k > l
            K_iajb_canon = ci_kl['K_iajb']
            K_iajb_eff = K_iajb_canon if kl_swap else K_iajb_canon.T
            S_ki_ll = (np.eye(n_ki) if canonical == ll_key
                       else _s_pno_get(canonical, ll_key))
            S_kl_ii = (np.eye(n_kl) if kl_canon == ii
                       else _s_pno_get(kl_canon, ii))
            S_ki_kl = (np.eye(n_kl) if canonical == kl_canon
                       else _s_pno_get(canonical, kl_canon))
            if S_ki_ll is None or S_kl_ii is None or S_ki_kl is None:
                continue
            T_l = S_ki_ll @ t1_l_pno                          # (n_ki,)
            T_i_kl = S_kl_ii @ t1_i_pno_loc                   # (n_kl,)
            K_kl_vec = S_ki_kl @ (K_iajb_eff @ T_i_kl)        # (n_ki,)
            C_tilde -= np.outer(T_l, K_kl_vec)

        # ---- Term 4 (Psi4 ccsd.cc:1896-1910): triplet sandwich per l ----
        # Psi4: triplet(T_iajb_[li_ordered], S(li, kl), K_iajb_[kl_ordered])
        # T_iajb ordered (l, i): canonical t2 if (l,i) == canon, else canon.T
        # K_iajb ordered (k, l): canonical if (k,l) == canon, else canon.T
        for l in domain:
            kl_canon = (min(k, l), max(k, l))
            li_canon = (min(l, i), max(l, i))
            if kl_canon not in pno_spaces or li_canon not in pno_spaces:
                continue
            n_kl = pno_spaces[kl_canon]['C_pno'].shape[1]
            n_li = pno_spaces[li_canon]['C_pno'].shape[1]
            if n_kl == 0 or n_li == 0:
                continue
            t2_li_ord = _t2_ordered(l, i)
            if t2_li_ord is None:
                continue
            ci_kl = cc_ints.get(kl_canon)
            if ci_kl is None:
                continue
            kl_swap = (k > l)  # ordered (k, l) != canonical
            K_iajb_canon = ci_kl['K_iajb']
            K_iajb_eff = K_iajb_canon.T if kl_swap else K_iajb_canon
            S_li_kl = (np.eye(n_li) if li_canon == kl_canon
                       else _s_pno_get(li_canon, kl_canon))
            S_ki_li = (np.eye(n_li) if canonical == li_canon
                       else _s_pno_get(canonical, li_canon))
            S_kl_ki = (np.eye(n_kl) if kl_canon == canonical
                       else _s_pno_get(kl_canon, canonical))
            if S_li_kl is None or S_ki_li is None or S_kl_ki is None:
                continue
            X = t2_li_ord @ S_li_kl @ K_iajb_eff             # (n_li, n_kl)
            Y = S_ki_li @ X @ S_kl_ki                        # (n_ki, n_ki)
            C_tilde -= 0.5 * Y

        return key_ki, C_tilde

    C_tilde_all = {}
    if _pool is not None:
        for key, val in _pool.map(_per_pair, all_ordered):
            C_tilde_all[key] = val
    else:
        for key in all_ordered:
            _, val = _per_pair(key)
            C_tilde_all[key] = val
    return C_tilde_all


def compute_C_tilde_batched(
        t1_pno, t2_pno_all, pno_spaces, nocc,
        ovL_pno_bare, ooL_bare, S_pno_cache, with_df,
        _term2_precomputed=None, cc_ints=None,
        pair_lmo_idx=None, _pool=None,
        S_pao_full=None, s1e=None,
        blas_threads=32, omp_threads=None,
        t1_cache=None):
    """Drop-in replacement for compute_C_tilde with Terms 3+4 batched.

    Signature matches compute_C_tilde exactly plus one kwarg
    ``blas_threads`` controlling the threadpool_limits scope used during
    the batched phase (set to None to keep the caller's BLAS thread
    count).  Output dict format and values match the reference to
    machine precision.
    """
    import time as _time_dbg
    _dbg = getattr(compute_C_tilde_batched, '_dump_timing', False)
    _pt = {}
    _t0 = _time_dbg.perf_counter()

    from pyscf.cc.dlpno_tccsd.local_df import (
        get_local_K, get_local_ovL, get_local_ooL_vec,
    )

    C_tilde_all = {}
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    # All ordered (k, i) pairs gamma_{ki} is defined over.
    all_pairs = set()
    canonical_keys = set()
    for key in t2_pno_all:
        a, b = key
        all_pairs.add((a, b))
        all_pairs.add((b, a))
        canonical_keys.add(key)

    # ------------------------------------------------------------------
    # T1 projection cache. When the driver passes its global ``t1_cache``
    # (FlatTensorStore built once per CCSD iteration), reuse it directly:
    # ``t1_cache[pk][l]`` already gives the same projection
    # ``_project_t1_to_pair`` would build, with zeros for missing rows —
    # so we just skip the per-(pair, l) rebuild and adapt downstream
    # lookups via ``_t1_get`` (always returns an ndarray, never None).
    # Fallback path builds the local dict for callers that don't pass
    # ``t1_cache=`` (back-compat).
    # ------------------------------------------------------------------
    if t1_cache is not None:
        T1_cache = None
        def _t1_get(pk, l):
            return t1_cache[pk][l]
    else:
        T1_cache = {}
        for pk in canonical_keys:
            if pno_spaces[pk]['C_pno'].shape[1] == 0:
                continue
            for l in range(nocc):
                T1_cache[(pk, l)] = _project_t1_to_pair(
                    t1_pno, l, pk, S_pno_cache, pno_spaces)
        def _t1_get(pk, l):
            return T1_cache.get((pk, l))
    _pt['T1_cache'] = _time_dbg.perf_counter() - _t0; _t0 = _time_dbg.perf_counter()

    # ------------------------------------------------------------------
    # Term 2 DF-pass fallback (only when cc_ints doesn't cover a pair).
    # This is the same code as the reference's fallback branch.
    # ------------------------------------------------------------------
    if _term2_precomputed is None:
        term2_data = {}
        for k, i in all_pairs:
            key_ki = (min(k, i), max(k, i))
            n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
            if n_ki == 0:
                continue
            t1_i_ki = _t1_get(key_ki, i)
            if t1_i_ki is None or np.max(np.abs(t1_i_ki)) < 1e-15:
                continue
            ovL_i_ki = ovL_pno_bare.get((key_ki, i))
            if ovL_i_ki is None:
                continue
            term2_data[(k, i)] = {
                'key_ki': key_ki, 'n_ki': n_ki,
                'C_pno': np.asfortranarray(pno_spaces[key_ki]['C_pno']),
                't1_i': t1_i_ki, 'ovL_i': ovL_i_ki,
                'result': np.zeros((n_ki, n_ki)),
            }
        if term2_data:
            aux_off = 0
            for Lpq in with_df.loop():
                nL = Lpq.shape[0]
                for (k, i), td in term2_data.items():
                    n_ki = td['n_ki']
                    buf = _ao2mo.nr_e2(Lpq, td['C_pno'],
                                       (0, n_ki, 0, n_ki), aosym='s2')
                    B_L = buf.reshape(nL, n_ki, n_ki)
                    z_i = td['t1_i'] @ td['ovL_i'][:, aux_off:aux_off+nL]
                    td['result'] += np.tensordot(z_i, B_L, axes=(0, 0))
                aux_off += nL
        _term2_precomputed = {ki: td['result'] for ki, td in term2_data.items()}

    # ------------------------------------------------------------------
    # Phase 1: Terms 1 + 2 per-pair.
    # Batched prange kernel with isolated OpenMP team. Static per-pair
    # data (k_Qa, Qab, Qma_sub, ooL_ki, shapes, offsets, scratch) cached
    # across CCSD iterations. Per-iter we rebuild T1 inputs and outputs.
    # Pairs not covered by cc_ints fall back to the original Python
    # per-pair path (rare for typical molecules).
    # ------------------------------------------------------------------
    from pyscf.cc.dlpno_tccsd._c_tilde_ph1_batched_cy import (
        c_tilde_ph1_batched)
    from threadpoolctl import threadpool_limits

    all_pairs_list = list(all_pairs)
    covered_pairs = []
    fallback_pairs = []
    for ki in all_pairs_list:
        k, i = ki
        key_ki = (min(k, i), max(k, i))
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        if n_ki == 0:
            continue
        if key_ki in cc_ints and cc_ints[key_ki] is not None:
            covered_pairs.append(ki)
        else:
            fallback_pairs.append(ki)

    # --- Batched path for cc_ints-covered pairs ---
    if covered_pairs:
        N = len(covered_pairs)
        plan_key = (id(cc_ints), tuple(covered_pairs),
                    id(pair_lmo_idx) if pair_lmo_idx is not None else 0)
        plan = getattr(compute_C_tilde_batched, '_ph1_plan', None)
        if plan is None or plan.get('key') != plan_key:
            # Fix #3 + #1 algorithmic parity with Psi4:
            # Precompute K_tilde_chem_k (L pre-summed (k_Qa.T @ Qab)) and
            # K_bar_chem_slice (row-slice of cc_ints['K_bar_chem']). Per-iter
            # kernel is then 1 dgemv + 1 dgemm on small precomputed tensors —
            # no n_local-axis sums in the hot path.
            n_pno_arr = np.zeros(N, dtype=np.int32)
            n_domain_arr = np.zeros(N, dtype=np.int32)
            K_tilde_chem_list = [None] * N
            K_bar_chem_slice_list = [None] * N
            ll_idx_list = [None] * N
            key_ki_list = [None] * N
            i_idx_list = [None] * N

            for p, ki in enumerate(covered_pairs):
                k, i = ki
                key_ki = (min(k, i), max(k, i))
                ci_ki = cc_ints[key_ki]
                n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
                if pair_lmo_idx is not None and key_ki in pair_lmo_idx:
                    ll_idx = np.asarray(pair_lmo_idx[key_ki], dtype=np.intp)
                else:
                    ll_idx = np.arange(nocc, dtype=np.intp)
                n_domain = ll_idx.size
                n_local = ci_ki['Qma'].shape[0]

                is_k_first = (key_ki[0] == k)
                # K_tilde_chem precomputed in cc_ints (local_df.py
                # compute_cc_integrals_sparse) — same tensor as Psi4
                # K_tilde_chem_[ki] (ccsd.cc:1402). Shared with build_D_tilde
                # and the T1 residual.
                K_tilde_chem_list[p] = (ci_ki['K_tilde_chem_i'] if is_k_first
                                        else ci_ki['K_tilde_chem_j'])
                # K_bar_chem[l, c] = sum_L q_pair[L] * Qma[L, l, c];
                # reduced to (nlmo_p, npno) in cc_ints. Translate global
                # ll_idx -> position within p_lmos before indexing.
                _ll_idx_in_p = np.asarray(
                    ci_ki['p_lmos_dense'])[ll_idx].astype(np.intp)
                K_bar_chem_slice_list[p] = np.ascontiguousarray(
                    ci_ki['K_bar_chem'][_ll_idx_in_p])

                n_pno_arr[p] = n_ki
                n_domain_arr[p] = n_domain
                ll_idx_list[p] = ll_idx
                key_ki_list[p] = key_ki
                i_idx_list[p] = i

            def _flat(arrs):
                sizes = np.array([a.size for a in arrs], dtype=np.int64)
                offsets = np.empty(len(arrs) + 1, dtype=np.int64)
                offsets[0] = 0
                offsets[1:] = np.cumsum(sizes)
                buf = np.empty(int(offsets[-1]))
                for idx, a in enumerate(arrs):
                    buf[offsets[idx]:offsets[idx + 1]] = a.ravel()
                return buf, offsets

            K_tilde_chem_flat, K_tilde_chem_off = _flat(K_tilde_chem_list)
            K_bar_chem_slice_flat, K_bar_chem_slice_off = _flat(
                K_bar_chem_slice_list)

            t1_sizes = n_pno_arr.astype(np.int64)
            t1_off = np.empty(N + 1, dtype=np.int64)
            t1_off[0] = 0
            t1_off[1:] = np.cumsum(t1_sizes)

            T1_local_sizes = (n_domain_arr.astype(np.int64)
                              * n_pno_arr.astype(np.int64))
            T1_local_off = np.empty(N + 1, dtype=np.int64)
            T1_local_off[0] = 0
            T1_local_off[1:] = np.cumsum(T1_local_sizes)

            C_sizes = n_pno_arr.astype(np.int64) ** 2
            C_off = np.empty(N + 1, dtype=np.int64)
            C_off[0] = 0
            C_off[1:] = np.cumsum(C_sizes)

            num_threads = min(32, N)

            plan = {
                'key': plan_key,
                'covered_pairs': covered_pairs,
                'key_ki_list': key_ki_list,
                'i_idx_list': i_idx_list,
                'll_idx_list': ll_idx_list,
                'n_pno_arr': n_pno_arr,
                'n_domain_arr': n_domain_arr,
                'K_tilde_chem_flat': K_tilde_chem_flat,
                'K_tilde_chem_off': K_tilde_chem_off,
                'K_bar_chem_slice_flat': K_bar_chem_slice_flat,
                'K_bar_chem_slice_off': K_bar_chem_slice_off,
                't1_off': t1_off, 't1_total': int(t1_off[-1]),
                'T1_local_off': T1_local_off,
                'T1_local_total': int(T1_local_off[-1]),
                'C_off': C_off, 'C_total': int(C_off[-1]),
                'num_threads': num_threads,
            }
            compute_C_tilde_batched._ph1_plan = plan

        t1_flat = np.empty(plan['t1_total'])
        T1_local_flat = np.empty(plan['T1_local_total'])
        t1_off_plan = plan['t1_off']
        T1_local_off_plan = plan['T1_local_off']
        zero_pno_cache = {}
        for p in range(N):
            key_ki = plan['key_ki_list'][p]
            i = plan['i_idx_list'][p]
            n_pno = int(plan['n_pno_arr'][p])
            t1_i_ki = _t1_get(key_ki, i)
            if t1_i_ki is None:
                zv = zero_pno_cache.get(n_pno)
                if zv is None:
                    zv = np.zeros(n_pno)
                    zero_pno_cache[n_pno] = zv
                t1_i_ki = zv
            t1_flat[t1_off_plan[p]:t1_off_plan[p + 1]] = t1_i_ki

            ll_idx = plan['ll_idx_list'][p]
            n_dom = ll_idx.size
            rows_buf = T1_local_flat[
                T1_local_off_plan[p]:T1_local_off_plan[p + 1]
            ].reshape(n_dom, n_pno)
            if t1_cache is not None:
                # Single fancy-index gather from the (nocc, n_pno) view
                rows_buf[:] = t1_cache[key_ki][np.asarray(ll_idx, dtype=np.intp)]
            else:
                for li, ll in enumerate(ll_idx):
                    r = T1_cache.get((key_ki, int(ll)))
                    if r is None:
                        rows_buf[li].fill(0.0)
                    else:
                        rows_buf[li] = r

        C_flat = np.zeros(plan['C_total'])

        with threadpool_limits(limits=1, user_api='blas'):
            c_tilde_ph1_batched(
                plan['K_tilde_chem_flat'], plan['K_tilde_chem_off'],
                plan['K_bar_chem_slice_flat'], plan['K_bar_chem_slice_off'],
                t1_flat, t1_off_plan,
                T1_local_flat, T1_local_off_plan,
                plan['n_pno_arr'], plan['n_domain_arr'],
                C_flat, plan['C_off'],
                plan['num_threads'],
            )

        C_off_plan = plan['C_off']
        for p, ki in enumerate(plan['covered_pairs']):
            n_pno = int(plan['n_pno_arr'][p])
            C_tilde_all[ki] = (
                C_flat[C_off_plan[p]:C_off_plan[p + 1]]
                .reshape(n_pno, n_pno).copy())

    # --- Fallback Python path for pairs not covered by cc_ints ---
    for ki in fallback_pairs:
        k, i = ki
        key_ki = (min(k, i), max(k, i))
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        C_tilde_ki = np.zeros((n_ki, n_ki))
        if _term2_precomputed is not None and (k, i) in _term2_precomputed:
            C_tilde_ki += _term2_precomputed[(k, i)]
        if pair_lmo_idx is not None and key_ki in pair_lmo_idx:
            _ll_idx = np.asarray(pair_lmo_idx[key_ki], dtype=np.intp)
        else:
            _ll_idx = np.arange(nocc)
        T1_rows = [_t1_get(key_ki, int(_ll)) for _ll in _ll_idx]
        T1_local_ki = np.ascontiguousarray(
            np.array([r if r is not None else np.zeros(n_ki)
                      for r in T1_rows]))
        ooL_ki = ooL_bare[k, i, :]
        K_bar_chem_local = np.zeros((len(_ll_idx), n_ki))
        for _li, _ll in enumerate(_ll_idx):
            ovL_l_ki = ovL_pno_bare.get((key_ki, int(_ll)))
            if ovL_l_ki is not None:
                K_bar_chem_local[_li] = ovL_l_ki @ ooL_ki
        C_tilde_ki += -T1_local_ki.T @ K_bar_chem_local
        C_tilde_all[ki] = C_tilde_ki
    _pt['phase1_t12'] = _time_dbg.perf_counter() - _t0; _t0 = _time_dbg.perf_counter()

    # ------------------------------------------------------------------
    # Phase 2: Terms 3 + 4 via plan cache.
    #
    # Plan depends only on the pair structure (pno_spaces, pair_lmo_idx,
    # S_pno_cache contents, cc_ints integrals) — all fixed after Stage 3.
    # Built once per CCSD run and reused every cycle; per-cycle cost is
    # just T1/T2 gather + batched matmul + scatter.
    # ------------------------------------------------------------------
    plan_key = tuple(sorted(t2_pno_all.keys()))
    _cache_attr = getattr(compute_C_tilde_batched, '_plan_cache', None)
    if _cache_attr is None:
        _cache_attr = {}
        compute_C_tilde_batched._plan_cache = _cache_attr
    plan = _cache_attr.get(plan_key)
    if plan is None:
        plan = _build_c_tilde_t34_plan(
            all_pairs, pno_spaces, pair_lmo_idx, t2_pno_all,
            S_pno_cache, cc_ints, _s_pno_get, nocc)
        _cache_attr[plan_key] = plan

    # ------------------------------------------------------------------
    # Migrate Phase 1 results into flat per-n_ki buffers; nogil Cython
    # kernels accumulate Terms 3 + 4 on top.  Two-stage structure:
    # parallel prange compute → sequential scatter-add (race-free).
    # ------------------------------------------------------------------
    from pyscf.cc.dlpno_tccsd._c_tilde_cy import (
        t3_kernel as _t3_kern, t4_kernel as _t4_kern,
    )

    _t_alloc = _time_dbg.perf_counter()
    flat_out = {}
    for n_ki, pairs in plan['pairs_by_n_ki'].items():
        buf = np.zeros((len(pairs), n_ki, n_ki))
        for slot, pair in enumerate(pairs):
            v = C_tilde_all.get(pair)
            if v is not None:
                buf[slot] = v
        flat_out[n_ki] = buf
    _pt['flat_alloc'] = _time_dbg.perf_counter() - _t_alloc

    _pt['t3_gather'] = 0.0; _pt['t3_kern'] = 0.0
    _pt['t4_gather'] = 0.0; _pt['t4_kern'] = 0.0
    # Single-call batched path: ONE t3+t4 prange call replaces the per-bucket
    # loop. compute_C_tilde uses scale=-0.5 and t2 directly (not 2*t2-t2.T).
    _tn_batch = _time_dbg.perf_counter()
    bv = _get_or_build_t34_batched_view(plan, t1_cache, t2_pno_all)
    _run_t34_batched(plan, bv, t1_cache, t2_pno_all, flat_out,
                     t4_use_u=False, t4_scale=-0.5)
    _pt['t4_kern'] = _time_dbg.perf_counter() - _tn_batch

    # Unpack flat_out back into C_tilde_all dict.
    _t_unpack = _time_dbg.perf_counter()
    for n_ki, pairs in plan['pairs_by_n_ki'].items():
        buf = flat_out[n_ki]
        for slot, pair in enumerate(pairs):
            C_tilde_all[pair] = buf[slot]
    _pt['unpack'] = _time_dbg.perf_counter() - _t_unpack

    if _dbg:
        _per = ' '.join(f'{k}={v*1e3:.0f}ms' for k, v in _pt.items())
        _tot = sum(_pt.values())
        print(f'  [c_tilde_batched_timing] tot={_tot*1e3:.0f}ms '
              f'{_per}', flush=True)

    # CTILDE_DUMP: parity dump vs Psi4 ccsd.cc:1809 compute_C_tilde output.
    if int(os.environ.get('DLPNO_DUMP_CTILDE', '0')):
        _it = getattr(compute_C_tilde_batched, '_iter', 0)
        compute_C_tilde_batched._iter = _it + 1
        if _it <= 2:
            tot_fro2 = 0.0
            tot_tr = 0.0
            tot_sum = 0.0
            n_pairs = 0
            for key, M in C_tilde_all.items():
                if M is None or M.size == 0:
                    continue
                n_pairs += 1
                tot_fro2 += float(np.sum(M ** 2))
                tot_tr += float(np.trace(M))
                tot_sum += float(M.sum())
            print(f"CTILDE_DUMP iter={_it} n_pairs={n_pairs} "
                  f"fro2={tot_fro2:.12e} tr={tot_tr:.12e} "
                  f"sum={tot_sum:.12e}", flush=True)

    return C_tilde_all


# ---------------------------------------------------------------------------
# T1-dressed Fock intermediates (Eqs 94-101, 85-86)
# ---------------------------------------------------------------------------


def compute_all_df_terms_local(t1_pno, fov_pno, t2_pno_all, pno_spaces,
                                nocc, cc_ints, S_pno_cache, pair_keys,
                                _pool=None, pair_lmo_idx=None,
                                t1_cache=None):
    """Per-pair, local-aux-per-pair version of compute_all_df_terms.

    Drop-in replacement that reads from ``cc_ints[pk]['Qab' / 'Qma']``
    (already fitted, local-aux per pair) instead of iterating
    ``with_df.loop()`` over global aux blocks. Each pair's contractions
    sum over the pair's local aux only — Psi4-equivalent and avoids the
    per-iteration full-naux ovL rebuild (12s of `_ao2mo.nr_e2` on str 010).

    Same return signature as ``compute_all_df_terms``:
        fvv_t1_all, c_term2_all, d_term2_all, ladder_all

    Args:
        t1_pno: dict i -> (n_pno_ii,)
        fov_pno: unused (kept for signature compat)
        t2_pno_all: dict pk -> (n_pno, n_pno)
        pno_spaces: dict from make_pnos
        nocc: int
        cc_ints: dict pk -> {'Qab': (n_local, n, n), 'Qma': (n_local, nocc, n), ...}
        S_pno_cache: dict (key1, key2) -> (n1, n2) overlap
        pair_keys: list of pair keys for which fvv_t1 and ladder are returned.
    """
    # Phase 1: cache t1 projections once (or reuse one from driver).
    if t1_cache is None:
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, build_t1_cache,
        )
        _pi = PairIndex(
            pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
        t1_cache = build_t1_cache(
            t1_pno, _pi, S_pno_cache, pno_spaces)

    # All ordered (k, i) pairs that appear in t2 (both directions).
    all_ordered = set()
    for key in t2_pno_all:
        a, b = key
        all_ordered.add((a, b))
        all_ordered.add((b, a))

    # NOTE: fvv_t1_all and ladder_all are kept in the return signature for
    # backward compat but are no longer computed here — both were dead:
    #   * fvv_t1 was returned but never consumed anywhere.
    #   * ladder_all ended up in _jiang_cache['ladder_all'] as a fallback,
    #     but _update_pair always rebuilds its own _ladder_local via
    #     compute_ladder() because every pair has cc_ints on our target
    #     geometries.  Skipping the ~15 GFLOPs/iter these two produced.
    fvv_t1_all = {}
    c_term2_all = {}
    d_term2_all = {}
    ladder_all = {}

    # ============================================================
    # C_tilde Term 2: per (k, i) ordered pair
    # ============================================================
    def _c_term2(ki_tuple):
        k, i = ki_tuple
        key_ki = (min(k, i), max(k, i))
        n = pno_spaces[key_ki]['C_pno'].shape[1]
        if n == 0:
            return ki_tuple, None
        ci = cc_ints.get(key_ki)
        if ci is None:
            return ki_tuple, None
        t1_i = t1_cache[key_ki][i]
        if np.max(np.abs(t1_i)) < 1e-15:
            return ki_tuple, None
        Qma = ci['Qma']
        Qab = ci['Qab']
        # Qma reduced; translate global LMO i -> p_lmos position.
        i_red = int(ci['p_lmos_dense'][i])
        if i_red < 0:
            return ki_tuple, None
        z_i = Qma[:, i_red, :] @ t1_i
        return ki_tuple, np.einsum('L,Lab->ab', z_i, Qab, optimize=True)

    # ============================================================
    # D_tilde Term 2: per (i, k) ordered pair
    # ============================================================
    def _d_term2(ik_tuple):
        i_idx, k_idx = ik_tuple
        key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
        n = pno_spaces[key_ik]['C_pno'].shape[1]
        if n == 0:
            return ik_tuple, None
        ci = cc_ints.get(key_ik)
        if ci is None:
            return ik_tuple, None
        t1_i = t1_cache[key_ik][i_idx]
        if np.max(np.abs(t1_i)) < 1e-15:
            return ik_tuple, None
        Qma = ci['Qma']
        Qab = ci['Qab']
        # Qma reduced; translate global LMO k_idx -> p_lmos position.
        k_red = int(ci['p_lmos_dense'][k_idx])
        if k_red < 0:
            return ik_tuple, None
        ovL_k = Qma[:, k_red, :].T
        z_c = Qab @ t1_i
        y = Qma[:, k_red, :] @ t1_i
        result = 2.0 * (ovL_k @ z_c)
        result -= np.einsum('L,Lab->ab', y, Qab, optimize=True)
        return ik_tuple, result

    # Unified dispatch: chunked pool.map over c_term2 + d_term2 items only.
    # (fvv_ladder removed — see note above.)
    ordered_list = list(all_ordered)
    work = ([('c_term2', ki) for ki in ordered_list]
            + [('d_term2', ik) for ik in ordered_list])

    def _dispatch(item):
        kind, key = item
        if kind == 'c_term2':
            return kind, _c_term2(key)
        else:  # d_term2
            return kind, _d_term2(key)

    for kind, payload in _chunked_map(_pool, _dispatch, work):
        if kind == 'c_term2':
            ki, val = payload
            if val is not None:
                c_term2_all[ki] = val
        else:  # d_term2
            ik, val = payload
            if val is not None:
                d_term2_all[ik] = val

    return fvv_t1_all, c_term2_all, d_term2_all, ladder_all


# =========================================================================
# Batched B and E terms — moved out of per-pair compute_residual_v2 so
# that the many small S @ t2 @ S.T ops can be vectorised into batched BLAS.
# =========================================================================


def compute_B_E_batched(strong_keys, t2_pno_all, pno_spaces, S_pno_cache,
                        cc_ints, B_tilde_per_ij, pair_lmo_idx, nocc, _pool=None,
                        S_pao_full=None, s1e=None):
    """Compute B and E-contribution dicts for all strong pairs, batched.

    B_ij = Σ_{kl ∈ domain_ij²} β[k,l] · S_{kl→ij} @ t2_kl @ S_{kl→ij}.T
           + (l≠k:) β[l,k] · S_{kl→ij} @ t2_kl.T @ S_{kl→ij}.T

    E_contrib_ij = Σ_{kl ∈ domain_ij²} S_{kl→ij} @ (u_kl @ K_kl.T
                                       + (k≠l: (2t2_kl.T - t2_kl) @ K_kl))
                                       @ S_{kl→ij}.T

    where u_kl = 2 t2_kl - t2_kl.T.  Both terms share identical projection
    structure; batching them together halves the BLAS overhead relative
    to two separate loops.

    ``B_tilde_per_ij`` is a dict keyed by strong pair: each entry is the
    pair's (nocc, nocc) B_tilde matrix.

    Returns:
        B_all: dict key_ij → (n_ij, n_ij) B term
        E_contrib_all: dict key_ij → (n_ij, n_ij) E subtracted contribution
    """
    B_all = {}
    E_all = {}
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    def _per_ij(key_ij):
        i, j = key_ij
        n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_ij == 0:
            return key_ij, np.zeros((n_ij, n_ij)), np.zeros((n_ij, n_ij))

        # Psi4-layout B_tilde: tuple (B_local (nlmo,nlmo), p_dense (nocc,) -> k_ij).
        B_tilde_entry = B_tilde_per_ij[key_ij]
        if isinstance(B_tilde_entry, tuple):
            B_local, p_dense = B_tilde_entry
            _btilde_lookup = lambda k, l: B_local[p_dense[k], p_dense[l]]
        else:
            _btilde_lookup = lambda k, l: B_tilde_entry[k, l]
        domain = (set(int(x) for x in pair_lmo_idx[key_ij])
                  if pair_lmo_idx is not None and key_ij in pair_lmo_idx
                  else set(range(nocc)))

        # --- Gather phase: collect pointers only (no per-kl math). ---
        # Bucket by n_kl so each bucket is a uniform-shape batched contraction.
        # Each entry carries (S, t2, K, beta_kl, beta_lk, same) where same = (k==l).
        buckets = {}  # n_kl -> list of (S, t2, K, beta_kl, beta_lk, same)
        for key_kl, t2_kl in t2_pno_all.items():
            if t2_kl is None or t2_kl.shape[0] == 0:
                continue
            k, l = key_kl
            if k not in domain or l not in domain:
                continue
            S = _s_pno_get(key_ij, key_kl)
            if S is None:
                continue
            K_kl = get_local_K(cc_ints, key_kl, k, l)
            if K_kl is None:
                continue
            same = (k == l)
            beta_kl = _btilde_lookup(k, l)
            beta_lk = 0.0 if same else _btilde_lookup(l, k)
            n_kl = t2_kl.shape[0]
            buckets.setdefault(n_kl, []).append(
                (S, t2_kl, K_kl, beta_kl, beta_lk, same))

        if not buckets:
            return key_ij, np.zeros((n_ij, n_ij)), np.zeros((n_ij, n_ij))

        # --- Batched phase: stack, then do per-bucket batched operations. ---
        # This replaces ~n_kept per-element numpy calls (which dominated at
        # small PNO size via BLAS-dispatch overhead) with O(1) big BLAS calls.
        B_sum = np.zeros((n_ij, n_ij))
        E_sum = np.zeros((n_ij, n_ij))
        for n_kl, items in buckets.items():
            N = len(items)
            S_arr = np.empty((N, n_ij, n_kl))
            T_arr = np.empty((N, n_kl, n_kl))
            K_arr = np.empty((N, n_kl, n_kl))
            b_kl_arr = np.empty(N)
            b_lk_arr = np.empty(N)
            same_arr = np.zeros(N, dtype=bool)
            for n, (S, t2, K, bkl, blk, same) in enumerate(items):
                S_arr[n] = S
                T_arr[n] = t2
                K_arr[n] = K
                b_kl_arr[n] = bkl
                b_lk_arr[n] = blk
                same_arr[n] = same

            T_T = T_arr.transpose(0, 2, 1)

            # TB[n] = β_kl·t2_n (for k==l)  OR  β_kl·t2_n + β_lk·t2_n.T (k!=l)
            TB = b_kl_arr[:, None, None] * T_arr
            if (~same_arr).any():
                TB = TB + b_lk_arr[:, None, None] * T_T

            # UK[n] = u·K.T (for k==l)  OR  u·K.T + (2 t2.T - t2)·K (k!=l)
            u = 2.0 * T_arr - T_T                        # (N, n_kl, n_kl)
            UK = np.matmul(u, K_arr.transpose(0, 2, 1))  # (N, n_kl, n_kl)
            if (~same_arr).any():
                v = 2.0 * T_T - T_arr
                vk = np.matmul(v, K_arr)
                vk[same_arr] = 0.0
                UK = UK + vk

            # B_sum += Σ_n S_n @ TB_n @ S_n.T  (batched)
            S_T = S_arr.transpose(0, 2, 1)
            B_sum += np.matmul(np.matmul(S_arr, TB), S_T).sum(axis=0)
            E_sum += np.matmul(np.matmul(S_arr, UK), S_T).sum(axis=0)

        return key_ij, B_sum, E_sum

    if _pool is not None:
        for key, B, E in _pool.map(_per_ij, list(strong_keys)):
            B_all[key] = B
            E_all[key] = E
    else:
        for key in strong_keys:
            key, B, E = _per_ij(key)
            B_all[key] = B
            E_all[key] = E

    return B_all, E_all


def _build_be_plan(strong_keys, t2_pno_all, pno_spaces, pair_lmo_idx,
                   cc_ints, _s_pno_get, nocc):
    """One-time plan for compute_B_E_batched_v2.

    Enumerates every (ij, kl) item — ij ∈ strong_keys, kl with k,l in
    ij's LMO domain and t2_kl present — and pre-stacks the cycle-
    invariant constants (S, K, same-flag) into per-(n_ij, n_kl) buckets.
    Each item also carries the flat output slot for its ij and the keys
    needed to gather T / β_kl / β_lk per cycle.
    """
    from pyscf.cc.dlpno_tccsd.local_df import get_local_K

    # Output slots: one per strong ij, grouped by n_ij.
    pairs_by_n_ij = {}
    pair_to_slot = {}
    for key_ij in strong_keys:
        n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_ij == 0:
            continue
        pairs_by_n_ij.setdefault(n_ij, []).append(key_ij)
        pair_to_slot[key_ij] = len(pairs_by_n_ij[n_ij]) - 1

    # Per-item gather: (key_ij, key_kl, k, l, same, n_ij, n_kl, S, K).
    items_by_shape = {}
    for key_ij in strong_keys:
        n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_ij == 0:
            continue
        domain = (set(int(x) for x in pair_lmo_idx[key_ij])
                  if pair_lmo_idx is not None and key_ij in pair_lmo_idx
                  else set(range(nocc)))
        for key_kl, t2_kl in t2_pno_all.items():
            if t2_kl is None or t2_kl.shape[0] == 0:
                continue
            k, l = key_kl
            if k not in domain or l not in domain:
                continue
            S = _s_pno_get(key_ij, key_kl)
            if S is None:
                continue
            K_kl = get_local_K(cc_ints, key_kl, k, l)
            if K_kl is None:
                continue
            n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
            items_by_shape.setdefault((n_ij, n_kl), []).append(
                (key_ij, key_kl, k, l, k == l, S, K_kl))

    buckets = []
    for (n_ij, n_kl), items in items_by_shape.items():
        N = len(items)
        S_arr = np.empty((N, n_ij, n_kl))
        K_arr = np.empty((N, n_kl, n_kl))
        same_arr = np.empty(N, dtype=np.uint8)
        item_idx = np.empty(N, dtype=np.intp)
        kl_keys = []       # per-item key_kl for T lookup
        beta_coords = []   # per-item (key_ij, k, l) for β lookup
        for n, (key_ij, key_kl, k, l, same, S, K_kl) in enumerate(items):
            S_arr[n] = S
            K_arr[n] = K_kl
            same_arr[n] = 1 if same else 0
            item_idx[n] = pair_to_slot[key_ij]
            kl_keys.append(key_kl)
            beta_coords.append((key_ij, k, l))
        buckets.append({
            'n_ij': n_ij, 'n_kl': n_kl,
            'S': S_arr, 'K': K_arr,
            'same': same_arr, 'item_idx': item_idx,
            'kl_keys': kl_keys, 'beta_coords': beta_coords,
        })

    return {
        'buckets': buckets,
        'pairs_by_n_ij': pairs_by_n_ij,
        'pair_to_slot': pair_to_slot,
    }


def compute_B_E_batched_v2(
        strong_keys, t2_pno_all, pno_spaces, S_pno_cache,
        cc_ints, B_tilde_per_ij, pair_lmo_idx, nocc, _pool=None,
        S_pao_full=None, s1e=None, omp_threads=None):
    """Plan-cached + Cython-kernel rewrite of ``compute_B_E_batched``.

    Same signature, same output semantics (B_all, E_all dicts keyed by
    strong pairs).  Gains versus the reference come from:

      - The (ij, kl) item structure, per-bucket stacks of S and K, the
        same-flag and the output slot map are all constant across CCSD
        cycles, so they are cached on the function object and the
        per-cycle work reduces to gathering T (from t2_pno_all) and
        β_kl / β_lk (from B_tilde_per_ij).
      - Per-item B and E tiles come from a single nogil ``prange``
        Cython kernel (``be_kernel``) that fuses the TB/UK build with
        the S @ · @ S.T contraction into three hand-rolled triple
        loops and one quadruple loop — no intermediate (N, n_kl, n_kl)
        TB/UK allocation, no BLAS dispatch overhead at n ≈ 25.
    """
    from pyscf.cc.dlpno_tccsd._be_cy import be_kernel

    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    plan_key = (tuple(sorted(strong_keys)), tuple(sorted(t2_pno_all.keys())))
    _cache_attr = getattr(compute_B_E_batched_v2, '_plan_cache', None)
    if _cache_attr is None:
        _cache_attr = {}
        compute_B_E_batched_v2._plan_cache = _cache_attr
    plan = _cache_attr.get(plan_key)
    if plan is None:
        plan = _build_be_plan(
            strong_keys, t2_pno_all, pno_spaces, pair_lmo_idx,
            cc_ints, _s_pno_get, nocc)
        _cache_attr[plan_key] = plan

    # Flat output buffers, one per n_ij bucket.
    flat_B = {}
    flat_E = {}
    for n_ij, pairs in plan['pairs_by_n_ij'].items():
        flat_B[n_ij] = np.zeros((len(pairs), n_ij, n_ij))
        flat_E[n_ij] = np.zeros((len(pairs), n_ij, n_ij))

    # Per-cycle gather + kernel dispatch, one bucket at a time.
    # Scope OpenMP to ``omp_threads`` so the nogil prange inside be_kernel
    # actually runs in parallel (the CCSD driver caps OMP=1 at import time
    # so that Python-pool workers stay on 1 BLAS thread each).
    with _omp_threads_ctx(omp_threads):
        for bucket in plan['buckets']:
            n_ij = bucket['n_ij']
            n_kl = bucket['n_kl']
            N = len(bucket['kl_keys'])
            T_arr = np.empty((N, n_kl, n_kl))
            beta_kl_arr = np.empty(N)
            beta_lk_arr = np.empty(N)
            for n in range(N):
                T_arr[n] = t2_pno_all[bucket['kl_keys'][n]]
                key_ij, k, l = bucket['beta_coords'][n]
                B_tilde = B_tilde_per_ij[key_ij]
                if isinstance(B_tilde, tuple):
                    B_local, p_dense = B_tilde
                    beta_kl_arr[n] = B_local[p_dense[k], p_dense[l]]
                    beta_lk_arr[n] = (
                        0.0 if k == l
                        else B_local[p_dense[l], p_dense[k]])
                else:
                    beta_kl_arr[n] = B_tilde[k, l]
                    beta_lk_arr[n] = 0.0 if k == l else B_tilde[l, k]
            be_kernel(bucket['S'], T_arr, bucket['K'],
                      beta_kl_arr, beta_lk_arr, bucket['same'],
                      bucket['item_idx'],
                      flat_B[n_ij], flat_E[n_ij])

    # Unpack flat outputs into dicts keyed by strong pair.
    B_all = {}
    E_all = {}
    for n_ij, pairs in plan['pairs_by_n_ij'].items():
        bb = flat_B[n_ij]
        ee = flat_E[n_ij]
        for slot, key_ij in enumerate(pairs):
            B_all[key_ij] = bb[slot]
            E_all[key_ij] = ee[slot]

    # Keys that the reference returns as zero-sized outputs (n_ij == 0)
    # still need entries for downstream lookups.
    for key_ij in strong_keys:
        if key_ij not in B_all:
            n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
            B_all[key_ij] = np.zeros((n_ij, n_ij))
            E_all[key_ij] = np.zeros((n_ij, n_ij))

    return B_all, E_all


# =========================================================================
# C and D dressed contractions: plan-cached + Cython kernels (Phase 5e).
#
# These are the two biggest remaining CPU bins inside compute_residual_v2
# (C ~11s CPU/iter, D ~16s CPU/iter at water8).  The per-(ij, k, side)
# items are structurally uniform once bucketed by shape, so the gather/
# scatter plan is cycle-invariant and can be cached; only ct/dt and t2
# values change per cycle.
# =========================================================================


def _build_cd_plan(strong_keys, t2_pno_all, pno_spaces, pair_lmo_idx,
                   cc_ints, K_ij_kj_all, K_coul_cache,
                   _s_pno_get, nocc):
    """Plan for compute_CD_terms_batched.

    Enumerates every (key_ij, k, side) item that the reference C/D
    blocks of compute_residual_v2 iterate over (residual.py lines
    2200-2334), groups them by shape, and pre-stacks all constant
    tensors (S projections, J_bold for C, 2*K-J for D).

    Returns a dict with two buckets lists — ``c_buckets`` and
    ``d_buckets`` — each bucketed by shape; plus the output slot map
    ``pair_to_slot`` and ``pairs_by_n_pno``.  Missing entries (absent
    ct, absent bold integrals) are zero-filled so the Cython kernels
    stay branch-free.
    """
    # --- Output slot map (one slot per strong pair, grouped by n_pno) ---
    pairs_by_n_pno = {}
    pair_to_slot = {}
    for key_ij in strong_keys:
        n_pno = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_pno == 0:
            continue
        pairs_by_n_pno.setdefault(n_pno, []).append(key_ij)
        pair_to_slot[key_ij] = len(pairs_by_n_pno[n_pno]) - 1

    # --- Enumerate items, bucketed by shape ---
    c_items_by_shape = {}   # (n_pno, n_ct, n_other) -> list
    d_items_by_shape = {}   # (n_pno, n_A,  n_B)     -> list

    for key_ij in strong_keys:
        i, j = key_ij
        n_pno = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_pno == 0:
            continue
        domain = (sorted(int(x) for x in pair_lmo_idx[key_ij])
                  if pair_lmo_idx is not None and key_ij in pair_lmo_idx
                  else list(range(nocc)))
        ci_ij = cc_ints.get(key_ij)
        J_ij_kj_local = ci_ij.get('J_ij_kj', {}) if ci_ij is not None else {}
        J_ji_ki = ci_ij.get('J_ji_ki', {}) if ci_ij is not None else {}
        K_ji_ki = ci_ij.get('K_ji_ki', {}) if ci_ij is not None else {}

        for k in domain:
            key_ik = (min(i, k), max(i, k))
            key_kj = (min(k, j), max(k, j))
            key_jk = (min(j, k), max(j, k))
            key_ki = (min(k, i), max(k, i))

            # ========= C_ij side: t2(key_kj), ct=(k, i) =========
            if key_kj in t2_pno_all and t2_pno_all[key_kj] is not None \
                    and t2_pno_all[key_kj].shape[0] > 0:
                n_other = t2_pno_all[key_kj].shape[0]
                n_ct = pno_spaces[key_ik]['C_pno'].shape[1]
                if n_ct > 0:
                    S_big = _s_pno_get(key_ij, key_ik)     # (n_pno, n_ct)
                    S_mid = _s_pno_get(key_ik, key_kj)     # (n_ct, n_other)
                    S_outer = _s_pno_get(key_ij, key_kj)   # (n_pno, n_other)
                    if (S_big is not None and S_mid is not None
                            and S_outer is not None):
                        # J_ij_kj: local cc_ints preferred; K_coul_cache
                        # fallback for entries missing locally (matches
                        # _update_pair's augment logic at lccsd.py:1730).
                        J_b = J_ij_kj_local.get((key_ij, k))
                        if J_b is None and K_coul_cache is not None:
                            J_b = K_coul_cache.get(
                                (key_ij, key_kj, i, k))
                        J_bold = (np.ascontiguousarray(J_b)
                                  if J_b is not None
                                  else np.zeros((n_pno, n_other)))
                        ct_key = (k, i)
                        c_items_by_shape.setdefault(
                            (n_pno, n_ct, n_other), []).append({
                                'side': 'ij',
                                'key_ij': key_ij,
                                'S_big': np.ascontiguousarray(S_big),
                                'S_mid': np.ascontiguousarray(S_mid),
                                'S_outer': np.ascontiguousarray(S_outer),
                                'J_bold': J_bold,
                                'ct_key': ct_key,
                                't2_key': key_kj,
                                't2_transpose': (k > j),
                            })

            # ========= C_ji side: t2(key_ki), ct_j=(k, j) =========
            if key_ki in t2_pno_all and t2_pno_all[key_ki] is not None \
                    and t2_pno_all[key_ki].shape[0] > 0:
                n_other = t2_pno_all[key_ki].shape[0]
                n_ct = pno_spaces[key_jk]['C_pno'].shape[1]
                if n_ct > 0:
                    S_big = _s_pno_get(key_ij, key_jk)
                    S_mid = _s_pno_get(key_jk, key_ki)
                    S_outer = _s_pno_get(key_ij, key_ki)
                    if (S_big is not None and S_mid is not None
                            and S_outer is not None):
                        # J_bold: local J_ji_ki preferred; fall back to
                        # K_coul_cache (global DF) when local absent.
                        J_b = J_ji_ki.get((key_ij, k)) if J_ji_ki else None
                        if J_b is None and K_coul_cache is not None:
                            J_b = K_coul_cache.get((key_ij, key_ki, j, k))
                        J_bold = (np.ascontiguousarray(J_b)
                                  if J_b is not None
                                  else np.zeros((n_pno, n_other)))
                        ct_key = (k, j)
                        c_items_by_shape.setdefault(
                            (n_pno, n_ct, n_other), []).append({
                                'side': 'ji',
                                'key_ij': key_ij,
                                'S_big': np.ascontiguousarray(S_big),
                                'S_mid': np.ascontiguousarray(S_mid),
                                'S_outer': np.ascontiguousarray(S_outer),
                                'J_bold': J_bold,
                                'ct_key': ct_key,
                                't2_key': key_ki,
                                't2_transpose': (k > i),
                            })

            # ========= D_ij side: t2(key_jk), dt=(i, k) =========
            if key_jk in t2_pno_all and t2_pno_all[key_jk] is not None \
                    and t2_pno_all[key_jk].shape[0] > 0:
                n_A = t2_pno_all[key_jk].shape[0]
                n_B = pno_spaces[key_ik]['C_pno'].shape[1]
                if n_A > 0:
                    S_a = _s_pno_get(key_ij, key_jk)   # (n_pno, n_A)
                    S_b = (_s_pno_get(key_jk, key_ik)  # (n_A, n_B)
                           if n_B > 0 else None)
                    S_c = (_s_pno_get(key_ij, key_ik)  # (n_pno, n_B)
                           if n_B > 0 else None)
                    if S_a is not None:
                        # Part A requires S_b, S_c, and non-null n_B.
                        has_A = (n_B > 0 and S_b is not None
                                 and S_c is not None)
                        if not has_A:
                            n_B = 1   # placeholder dimension; Part A
                                      # is zero-filled
                        K_b = (K_ij_kj_all.get((key_ij, k))
                               if K_ij_kj_all is not None else None)
                        # D uses J_ij_kj from local cc_ints (reference
                        # residual.py:2800 — no K_coul_cache fallback
                        # for the D bold term).
                        J_b = J_ij_kj_local.get((key_ij, k))
                        if K_b is not None and J_b is not None:
                            KJ = np.ascontiguousarray(2.0 * K_b - J_b)
                        else:
                            KJ = np.zeros((n_pno, n_A))
                        d_items_by_shape.setdefault(
                            (n_pno, n_A, n_B), []).append({
                                'side': 'ij',
                                'key_ij': key_ij,
                                'S_a': np.ascontiguousarray(S_a),
                                'S_b': (np.ascontiguousarray(S_b) if has_A
                                        else np.zeros((n_A, n_B))),
                                'S_c': (np.ascontiguousarray(S_c) if has_A
                                        else np.zeros((n_pno, n_B))),
                                'KJ': KJ,
                                'dt_key': ((i, k) if has_A else None),
                                'dt_dim': n_B if has_A else n_B,
                                't2_key': key_jk,
                                't2_transpose': (j > k),
                            })

            # ========= D_ji side: t2(key_ik), dt_j=(j, k) =========
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None \
                    and t2_pno_all[key_ik].shape[0] > 0:
                n_A = t2_pno_all[key_ik].shape[0]
                n_B = pno_spaces[key_jk]['C_pno'].shape[1]
                if n_A > 0:
                    S_a = _s_pno_get(key_ij, key_ik)
                    S_b = (_s_pno_get(key_ik, key_jk)
                           if n_B > 0 else None)
                    S_c = (_s_pno_get(key_ij, key_jk)
                           if n_B > 0 else None)
                    if S_a is not None:
                        has_A = (n_B > 0 and S_b is not None
                                 and S_c is not None)
                        if not has_A:
                            n_B = 1
                        K_b = K_ji_ki.get((key_ij, k)) if K_ji_ki else None
                        J_b = J_ji_ki.get((key_ij, k)) if J_ji_ki else None
                        if K_b is not None and J_b is not None:
                            KJ = np.ascontiguousarray(2.0 * K_b - J_b)
                        else:
                            KJ = np.zeros((n_pno, n_A))
                        d_items_by_shape.setdefault(
                            (n_pno, n_A, n_B), []).append({
                                'side': 'ji',
                                'key_ij': key_ij,
                                'S_a': np.ascontiguousarray(S_a),
                                'S_b': (np.ascontiguousarray(S_b) if has_A
                                        else np.zeros((n_A, n_B))),
                                'S_c': (np.ascontiguousarray(S_c) if has_A
                                        else np.zeros((n_pno, n_B))),
                                'KJ': KJ,
                                'dt_key': ((j, k) if has_A else None),
                                'dt_dim': n_B if has_A else n_B,
                                't2_key': key_ik,
                                't2_transpose': (i > k),
                            })

    # --- Stack each bucket's constants into contiguous 3D arrays ---
    # Bucket key is (shape, side) so that at runtime every bucket is
    # homogeneous and we can hand it straight to the Cython kernel
    # without the old mask + ascontiguousarray(sel) split pass.  Items
    # are partitioned by ``it['side']`` (0 = ij, 1 = ji) before packing.
    def _pack_c(items, n_pno, n_ct, n_other, side):
        N = len(items)
        S_big = np.empty((N, n_pno, n_ct))
        S_mid = np.empty((N, n_ct, n_other))
        S_outer = np.empty((N, n_pno, n_other))
        J_bold = np.empty((N, n_pno, n_other))
        ct_keys = [None] * N
        t2_keys = [None] * N
        t2_trans = np.empty(N, dtype=np.uint8)
        item_idx = np.empty(N, dtype=np.intp)
        for n, it in enumerate(items):
            S_big[n] = it['S_big']
            S_mid[n] = it['S_mid']
            S_outer[n] = it['S_outer']
            J_bold[n] = it['J_bold']
            ct_keys[n] = it['ct_key']
            t2_keys[n] = it['t2_key']
            t2_trans[n] = 1 if it['t2_transpose'] else 0
            item_idx[n] = pair_to_slot[it['key_ij']]
        return {
            'n_pno': n_pno, 'n_ct': n_ct, 'n_other': n_other,
            'side': side,
            'S_big': S_big, 'S_mid': S_mid, 'S_outer': S_outer,
            'J_bold': J_bold,
            'ct_keys': ct_keys, 't2_keys': t2_keys, 't2_trans': t2_trans,
            'item_idx': item_idx,
        }

    def _pack_d(items, n_pno, n_A, n_B, side):
        N = len(items)
        S_a = np.empty((N, n_pno, n_A))
        S_b = np.empty((N, n_A, n_B))
        S_c = np.empty((N, n_pno, n_B))
        KJ = np.empty((N, n_pno, n_A))
        dt_keys = [None] * N
        t2_keys = [None] * N
        t2_trans = np.empty(N, dtype=np.uint8)
        item_idx = np.empty(N, dtype=np.intp)
        for n, it in enumerate(items):
            S_a[n] = it['S_a']
            S_b[n] = it['S_b']
            S_c[n] = it['S_c']
            KJ[n] = it['KJ']
            dt_keys[n] = it['dt_key']
            t2_keys[n] = it['t2_key']
            t2_trans[n] = 1 if it['t2_transpose'] else 0
            item_idx[n] = pair_to_slot[it['key_ij']]
        return {
            'n_pno': n_pno, 'n_A': n_A, 'n_B': n_B,
            'side': side,
            'S_a': S_a, 'S_b': S_b, 'S_c': S_c, 'KJ': KJ,
            'dt_keys': dt_keys, 't2_keys': t2_keys, 't2_trans': t2_trans,
            'item_idx': item_idx,
        }

    def _split_by_side(items):
        ij = [it for it in items if it['side'] == 'ij']
        ji = [it for it in items if it['side'] == 'ji']
        return ij, ji

    c_buckets = []
    for shape, items in c_items_by_shape.items():
        ij_items, ji_items = _split_by_side(items)
        if ij_items:
            c_buckets.append(_pack_c(ij_items, *shape, side=0))
        if ji_items:
            c_buckets.append(_pack_c(ji_items, *shape, side=1))

    d_buckets = []
    for shape, items in d_items_by_shape.items():
        ij_items, ji_items = _split_by_side(items)
        if ij_items:
            d_buckets.append(_pack_d(ij_items, *shape, side=0))
        if ji_items:
            d_buckets.append(_pack_d(ji_items, *shape, side=1))

    return {
        'c_buckets': c_buckets, 'd_buckets': d_buckets,
        'pairs_by_n_pno': pairs_by_n_pno,
        'pair_to_slot': pair_to_slot,
    }


def _get_or_build_t34_batched_view(plan, t1_cache, t2_pno_all=None):
    """Build (or fetch cached) flat per-item view across all t3/t4 buckets
    in a compute_C_tilde / build_D_tilde Phase 2 plan.

    Concatenates the per-bucket K/S static tensors into single flat
    buffers; pre-computes per-item t1 absolute offsets into the
    t1_cache._buffer (stable across CCSD iterations because pno_spaces
    shapes don't change). Per-iter only ``t2`` and ``t1_cache._buffer``
    pointers need refresh.
    """
    bv = plan.get('_t34_batched_view')
    if bv is not None:
        return bv

    t1c_offsets = np.asarray(t1_cache._offsets)

    def _t1_abs(key, l):
        # Absolute offset of t1_cache[key][l] within t1_cache._buffer
        # (each per-pair view is (nocc, n_pno) C-contig; row l starts at
        # offset + l * n_pno).
        canon = (min(key), max(key))
        pidx = t1_cache._canon_to_idx[canon]
        n_pno = int(t1_cache._shapes[pidx, 1])
        return int(t1c_offsets[pidx]) + l * n_pno

    # ---- t3 side ----
    t3_n_kl_l, t3_n_ki_l = [], []
    t3_K_off_l, t3_S_off_l = [], []
    t3_t1i_off_l, t3_T1l_off_l = [], []
    t3_target_slot_l = []   # (n_ki_bucket_key, slot)
    t3_K_pieces, t3_S_pieces = [], []
    t3_K_run = t3_S_run = 0
    t3_tile_off = [0]

    for bucket in plan['t3']:
        n_ki = bucket['n_ki']
        n_kl = bucket['n_kl']
        N_b = len(bucket['t1i_keys'])
        for nb in range(N_b):
            t3_n_kl_l.append(n_kl)
            t3_n_ki_l.append(n_ki)
            t3_K_pieces.append(np.ascontiguousarray(
                bucket['K'][nb]).ravel())
            t3_S_pieces.append(np.ascontiguousarray(
                bucket['S'][nb]).ravel())
            t3_K_off_l.append(t3_K_run); t3_K_run += n_kl * n_kl
            t3_S_off_l.append(t3_S_run); t3_S_run += n_ki * n_kl
            key_t1i, l_t1i = bucket['t1i_keys'][nb]
            key_T1l, l_T1l = bucket['T1l_keys'][nb]
            t3_t1i_off_l.append(_t1_abs(key_t1i, l_t1i))
            t3_T1l_off_l.append(_t1_abs(key_T1l, l_T1l))
            t3_target_slot_l.append((n_ki, int(bucket['item_idx'][nb])))
            t3_tile_off.append(t3_tile_off[-1] + n_ki * n_ki)

    t3_N = len(t3_n_kl_l)

    # ---- t4 side ----
    t4_n_ki_l, t4_n_li_l, t4_n_kl_l = [], [], []
    t4_S_ki_li_off_l, t4_S_li_kl_off_l = [], []
    t4_K_off_l, t4_S_kl_ki_off_l = [], []
    t4_t2_keys_l, t4_t2_trans_l = [], []
    t4_target_slot_l = []
    t4_S_ki_li_pieces, t4_S_li_kl_pieces = [], []
    t4_K_pieces, t4_S_kl_ki_pieces = [], []
    (t4_S_ki_li_run, t4_S_li_kl_run, t4_K_run,
     t4_S_kl_ki_run) = 0, 0, 0, 0
    t4_t2_size_l = []
    t4_tile_off = [0]

    for bucket in plan['t4']:
        n_ki = bucket['n_ki']
        n_li = bucket['n_li']
        n_kl = bucket['n_kl']
        # Source list of (canonical_t2_key, transpose_flag); compute_C uses
        # 't2_sources', build_D uses 'u_sources' — same shape, same role.
        sources = bucket.get('t2_sources') or bucket['u_sources']
        N_b = len(sources)
        for nb in range(N_b):
            t4_n_ki_l.append(n_ki)
            t4_n_li_l.append(n_li)
            t4_n_kl_l.append(n_kl)
            t4_S_ki_li_pieces.append(np.ascontiguousarray(
                bucket['S_ki_li'][nb]).ravel())
            t4_S_li_kl_pieces.append(np.ascontiguousarray(
                bucket['S_li_kl'][nb]).ravel())
            t4_K_pieces.append(np.ascontiguousarray(bucket['K'][nb]).ravel())
            t4_S_kl_ki_pieces.append(np.ascontiguousarray(
                bucket['S_kl_ki'][nb]).ravel())
            t4_S_ki_li_off_l.append(t4_S_ki_li_run)
            t4_S_ki_li_run += n_ki * n_li
            t4_S_li_kl_off_l.append(t4_S_li_kl_run)
            t4_S_li_kl_run += n_li * n_kl
            t4_K_off_l.append(t4_K_run); t4_K_run += n_kl * n_kl
            t4_S_kl_ki_off_l.append(t4_S_kl_ki_run)
            t4_S_kl_ki_run += n_kl * n_ki
            key, tr = sources[nb]
            t4_t2_keys_l.append(key)
            t4_t2_trans_l.append(bool(tr))
            t4_t2_size_l.append(n_li * n_li)
            t4_target_slot_l.append((n_ki, int(bucket['item_idx'][nb])))
            t4_tile_off.append(t4_tile_off[-1] + n_ki * n_ki)

    t4_N = len(t4_n_ki_l)

    t4_t2_off = np.zeros(t4_N + 1, dtype=np.int64)
    t4_t2_off[1:] = np.cumsum(t4_t2_size_l)
    # Per-item absolute offsets into t2_pno_all._buffer (FlatTensorStore)
    # so the per-cycle t2/u gather can run as a nogil prange Cython kernel
    # instead of a Python loop. Same trick as the cd batched gather.
    if t2_pno_all is not None and hasattr(t2_pno_all, '_offsets'):
        _t2_off_arr = np.asarray(t2_pno_all._offsets)
        _canon_to_idx = t2_pno_all._canon_to_idx
        t4_t2_canon_off = np.array(
            [_t2_off_arr[_canon_to_idx[k]] for k in t4_t2_keys_l],
            dtype=np.int64)
        t4_t2_trans_arr = np.asarray(t4_t2_trans_l, dtype=np.int32)
    else:
        t4_t2_canon_off = None
        t4_t2_trans_arr = None

    bv = {
        # ---- t3 ----
        't3_N': t3_N,
        't3_n_kl': np.asarray(t3_n_kl_l, dtype=np.int32),
        't3_n_ki': np.asarray(t3_n_ki_l, dtype=np.int32),
        't3_K_off': np.asarray(t3_K_off_l, dtype=np.int64),
        't3_S_off': np.asarray(t3_S_off_l, dtype=np.int64),
        't3_t1i_off': np.asarray(t3_t1i_off_l, dtype=np.int64),
        't3_T1l_off': np.asarray(t3_T1l_off_l, dtype=np.int64),
        't3_K_flat': (np.concatenate(t3_K_pieces)
                      if t3_K_pieces else np.zeros(0)),
        't3_S_flat': (np.concatenate(t3_S_pieces)
                      if t3_S_pieces else np.zeros(0)),
        't3_target_slot': t3_target_slot_l,
        't3_tile_off': np.asarray(t3_tile_off, dtype=np.int64),
        # ---- t4 ----
        't4_N': t4_N,
        't4_n_ki': np.asarray(t4_n_ki_l, dtype=np.int32),
        't4_n_li': np.asarray(t4_n_li_l, dtype=np.int32),
        't4_n_kl': np.asarray(t4_n_kl_l, dtype=np.int32),
        't4_S_ki_li_off': np.asarray(t4_S_ki_li_off_l, dtype=np.int64),
        't4_S_li_kl_off': np.asarray(t4_S_li_kl_off_l, dtype=np.int64),
        't4_K_off': np.asarray(t4_K_off_l, dtype=np.int64),
        't4_S_kl_ki_off': np.asarray(t4_S_kl_ki_off_l, dtype=np.int64),
        't4_S_ki_li_flat': (np.concatenate(t4_S_ki_li_pieces)
                             if t4_S_ki_li_pieces else np.zeros(0)),
        't4_S_li_kl_flat': (np.concatenate(t4_S_li_kl_pieces)
                             if t4_S_li_kl_pieces else np.zeros(0)),
        't4_K_flat': (np.concatenate(t4_K_pieces)
                      if t4_K_pieces else np.zeros(0)),
        't4_S_kl_ki_flat': (np.concatenate(t4_S_kl_ki_pieces)
                             if t4_S_kl_ki_pieces else np.zeros(0)),
        't4_t2_keys': t4_t2_keys_l,
        't4_t2_trans': np.asarray(t4_t2_trans_l, dtype=bool),
        't4_t2_canon_off': t4_t2_canon_off,
        't4_t2_trans_arr': t4_t2_trans_arr,
        't4_t2_off': t4_t2_off,
        't4_target_slot': t4_target_slot_l,
        't4_tile_off': np.asarray(t4_tile_off, dtype=np.int64),
    }
    plan['_t34_batched_view'] = bv
    return bv


def _run_t34_batched(plan, bv, t1_cache, t2_pno_all, flat_out,
                     t4_use_u, t4_scale):
    """Run batched t3 + t4 kernels and scatter into flat_out (per n_ki).

    flat_out: dict {n_ki -> ndarray (n_pairs_n_ki, n_ki, n_ki)}.
    t4_use_u: True for build_D_tilde (input is u = 2*t2 - t2.T),
              False for compute_C_tilde (input is t2 directly).
    t4_scale: -0.5 (C_tilde) or +0.5 (D_tilde).
    """
    from pyscf.cc.dlpno_tccsd._t34_batched_cy import (
        t3_kernel_batched, t4_kernel_batched,
    )
    from threadpoolctl import threadpool_limits

    # Flat views over per-n_ki output buffers (we update in-place via .ravel).
    flat_views = {n_ki: buf.ravel() for n_ki, buf in flat_out.items()}

    # ---- t3 ----
    t3_N = bv['t3_N']
    if t3_N > 0:
        max_n_kl = int(bv['t3_n_kl'].max(initial=1))
        max_n_ki = int(bv['t3_n_ki'].max(initial=1))
        num_threads = min(64, t3_N)
        Kt1 = np.empty((num_threads, max_n_kl))
        Kt1_ki = np.empty((num_threads, max_n_ki))
        t3_tiles = np.zeros(int(bv['t3_tile_off'][-1]))
        with threadpool_limits(limits=1, user_api='blas'):
            t3_kernel_batched(
                t3_N, max_n_ki, max_n_kl,
                bv['t3_n_kl'], bv['t3_n_ki'],
                bv['t3_K_off'], bv['t3_S_off'],
                bv['t3_t1i_off'], bv['t3_T1l_off'],
                bv['t3_tile_off'],
                bv['t3_K_flat'], bv['t3_S_flat'],
                t1_cache._buffer,
                Kt1, Kt1_ki,
                t3_tiles, num_threads,
            )
        # Scatter contrib tiles into flat_out (-= for both C and D —
        # the sign is absorbed into the kernel via the negative T1l).
        # Reference scatters with `out[idx] += contrib`, kernel stores
        # contrib = -T1l * Kt1_ki, so we use += here too.
        t3_tile_off = bv['t3_tile_off']
        t3_target = bv['t3_target_slot']
        t3_n_ki = bv['t3_n_ki']
        for n in range(t3_N):
            n_ki, slot = t3_target[n]
            tile_size = n_ki * n_ki
            base = slot * tile_size
            tile = t3_tiles[t3_tile_off[n]:t3_tile_off[n + 1]]
            flat_views[n_ki][base:base + tile_size] += tile

    # ---- t4 ----
    t4_N = bv['t4_N']
    if t4_N > 0:
        # Per-cycle t2 flat buffer — built via nogil prange Cython kernel
        # using absolute offsets into t2_pno_all._buffer (FlatTensorStore).
        # Same gather pattern as the cd batched path.
        t4_t2_off = bv['t4_t2_off']
        t4_n_li = bv['t4_n_li']
        t2_flat = np.empty(int(t4_t2_off[-1]))
        if (bv['t4_t2_canon_off'] is not None
                and hasattr(t2_pno_all, '_buffer')):
            from pyscf.cc.dlpno_tccsd._cd_gather_cy import (
                gather_t2_with_transpose, gather_u_from_t2,
            )
            if t4_use_u:
                gather_u_from_t2(
                    t4_N, t4_n_li,
                    bv['t4_t2_canon_off'], bv['t4_t2_trans_arr'],
                    t4_t2_off, t2_pno_all._buffer, t2_flat,
                    min(64, t4_N),
                )
            else:
                gather_t2_with_transpose(
                    t4_N, t4_n_li,
                    bv['t4_t2_canon_off'], bv['t4_t2_trans_arr'],
                    t4_t2_off, t2_pno_all._buffer, t2_flat,
                    min(64, t4_N),
                )
        else:
            # Fallback: per-item Python loop
            t4_t2_keys = bv['t4_t2_keys']
            t4_t2_trans = bv['t4_t2_trans']
            for n in range(t4_N):
                t2 = t2_pno_all[t4_t2_keys[n]]
                t2_d = t2.T if t4_t2_trans[n] else t2
                if t4_use_u:
                    t2_flat[t4_t2_off[n]:t4_t2_off[n + 1]] = (
                        2.0 * t2_d - t2_d.T).ravel()
                else:
                    t2_flat[t4_t2_off[n]:t4_t2_off[n + 1]] = t2_d.ravel()

        max_n_ki = int(bv['t4_n_ki'].max(initial=1))
        max_n_li = int(bv['t4_n_li'].max(initial=1))
        max_n_kl = int(bv['t4_n_kl'].max(initial=1))
        num_threads = min(64, t4_N)
        tmp1 = np.empty((num_threads, max_n_ki * max_n_li))
        tmp2 = np.empty((num_threads, max_n_ki * max_n_kl))
        tmp3 = np.empty((num_threads, max_n_ki * max_n_kl))
        t4_tiles = np.zeros(int(bv['t4_tile_off'][-1]))
        with threadpool_limits(limits=1, user_api='blas'):
            t4_kernel_batched(
                t4_N, max_n_ki, max_n_li, max_n_kl,
                bv['t4_n_ki'], bv['t4_n_li'], bv['t4_n_kl'],
                bv['t4_S_ki_li_off'], bv['t4_t2_off'][:t4_N],
                bv['t4_S_li_kl_off'], bv['t4_K_off'],
                bv['t4_S_kl_ki_off'], bv['t4_tile_off'],
                bv['t4_S_ki_li_flat'], bv['t4_S_li_kl_flat'],
                bv['t4_K_flat'], bv['t4_S_kl_ki_flat'],
                t2_flat,
                tmp1, tmp2, tmp3,
                t4_tiles, t4_scale, num_threads,
            )
        t4_tile_off = bv['t4_tile_off']
        t4_target = bv['t4_target_slot']
        t4_n_ki = bv['t4_n_ki']
        for n in range(t4_N):
            n_ki, slot = t4_target[n]
            tile_size = n_ki * n_ki
            base = slot * tile_size
            tile = t4_tiles[t4_tile_off[n]:t4_tile_off[n + 1]]
            flat_views[n_ki][base:base + tile_size] += tile


def _get_or_build_cd_batched_view(plan, pno_spaces, t2_pno_all=None):
    """Build (or fetch cached) flat per-item view across all c_/d_ buckets.

    Concatenates the per-bucket S/J_bold static tensors into single flat
    buffers, and pre-computes per-item offset arrays so a single kernel
    call can process all items via prange. Cached on the plan dict.
    """
    bv = plan.get('_batched_view')
    if bv is not None:
        return bv

    # ---- Output slot tables (per n_pno → list of pair keys) ----
    pairs_by_n_pno = plan['pairs_by_n_pno']
    # Flat output target offsets: each (n_pno, slot) maps to offset
    # in a per-side flat buffer. We use one flat buffer per
    # (side ∈ {ij, ji}) per term {C, D}; per item knows its absolute
    # offset = (cumulative pair offset in flat) + 0 for the (n_pno, n_pno)
    # tile starting position.
    n_pno_offsets = {}     # n_pno -> starting offset in per-side flat buffer
    flat_total = 0
    for n_pno in sorted(pairs_by_n_pno):
        n_pno_offsets[n_pno] = flat_total
        flat_total += len(pairs_by_n_pno[n_pno]) * n_pno * n_pno

    def _slot_to_off(n_pno, slot):
        return n_pno_offsets[n_pno] + slot * n_pno * n_pno

    # ---- C side: walk c_buckets, flatten per-item ----
    c_n_pno_l, c_n_ct_l, c_n_other_l = [], [], []
    c_S_big_off_l, c_S_mid_off_l, c_J_bold_off_l, c_S_outer_off_l = (
        [], [], [], [])
    c_ct_keys_l, c_t2_keys_l, c_t2_trans_l = [], [], []
    c_target_off_ij_l, c_target_off_ji_l = [], []
    c_S_big_pieces, c_S_mid_pieces, c_J_bold_pieces, c_S_outer_pieces = (
        [], [], [], [])
    c_S_big_run = c_S_mid_run = c_J_bold_run = c_S_outer_run = 0
    c_ct_size_l, c_t2_size_l = [], []
    c_tile_off = [0]
    c_side_l = []
    for bucket in plan['c_buckets']:
        n_pno = bucket['n_pno']
        n_ct = bucket['n_ct']
        n_other = bucket['n_other']
        side = bucket['side']
        N_b = len(bucket['ct_keys'])
        for nb in range(N_b):
            c_n_pno_l.append(n_pno)
            c_n_ct_l.append(n_ct)
            c_n_other_l.append(n_other)
            c_S_big_pieces.append(np.ascontiguousarray(
                bucket['S_big'][nb]).ravel())
            c_S_mid_pieces.append(np.ascontiguousarray(
                bucket['S_mid'][nb]).ravel())
            c_J_bold_pieces.append(np.ascontiguousarray(
                bucket['J_bold'][nb]).ravel())
            c_S_outer_pieces.append(np.ascontiguousarray(
                bucket['S_outer'][nb]).ravel())
            c_S_big_off_l.append(c_S_big_run);   c_S_big_run   += n_pno * n_ct
            c_S_mid_off_l.append(c_S_mid_run);   c_S_mid_run   += n_ct * n_other
            c_J_bold_off_l.append(c_J_bold_run); c_J_bold_run += n_pno * n_other
            c_S_outer_off_l.append(c_S_outer_run); c_S_outer_run += n_pno * n_other
            c_ct_keys_l.append(bucket['ct_keys'][nb])
            c_t2_keys_l.append(bucket['t2_keys'][nb])
            c_t2_trans_l.append(bool(bucket['t2_trans'][nb]))
            c_ct_size_l.append(n_ct * n_ct)
            c_t2_size_l.append(n_other * n_other)
            c_side_l.append(side)
            slot = int(bucket['item_idx'][nb])
            c_tile_off.append(c_tile_off[-1] + n_pno * n_pno)
            if side == 0:
                c_target_off_ij_l.append(_slot_to_off(n_pno, slot))
                c_target_off_ji_l.append(-1)
            else:
                c_target_off_ij_l.append(-1)
                c_target_off_ji_l.append(_slot_to_off(n_pno, slot))

    c_N = len(c_n_pno_l)
    c_ct_off = np.zeros(c_N + 1, dtype=np.int64)
    c_ct_off[1:] = np.cumsum(c_ct_size_l)
    c_t2_off = np.zeros(c_N + 1, dtype=np.int64)
    c_t2_off[1:] = np.cumsum(c_t2_size_l)
    # Per-item absolute offset into t2_pno_all._buffer for the canonical
    # t2 tile (transpose handled by the gather kernel). Used to build
    # t2_flat per cycle in nogil prange instead of a Python loop.
    if t2_pno_all is not None and hasattr(t2_pno_all, '_offsets'):
        _t2_off_arr = np.asarray(t2_pno_all._offsets)
        _canon_to_idx = t2_pno_all._canon_to_idx
        c_t2_canon_off = np.array(
            [_t2_off_arr[_canon_to_idx[k]] for k in c_t2_keys_l],
            dtype=np.int64)
        c_t2_trans_arr = np.asarray(c_t2_trans_l, dtype=np.int32)
    else:
        c_t2_canon_off = None
        c_t2_trans_arr = None

    # ---- D side: walk d_buckets, flatten per-item ----
    d_n_pno_l, d_n_A_l, d_n_B_l = [], [], []
    d_S_a_off_l, d_S_b_off_l, d_S_c_off_l, d_KJ_off_l = [], [], [], []
    d_t2_keys_l, d_t2_trans_l, d_dt_keys_l = [], [], []
    d_target_off_ij_l, d_target_off_ji_l = [], []
    d_S_a_pieces, d_S_b_pieces, d_S_c_pieces, d_KJ_pieces = [], [], [], []
    d_S_a_run = d_S_b_run = d_S_c_run = d_KJ_run = 0
    d_u_size_l, d_dt_size_l = [], []
    d_tile_off = [0]
    d_side_l = []
    for bucket in plan['d_buckets']:
        n_pno = bucket['n_pno']
        n_A = bucket['n_A']
        n_B = bucket['n_B']
        side = bucket['side']
        N_b = len(bucket['dt_keys'])
        for nb in range(N_b):
            d_n_pno_l.append(n_pno)
            d_n_A_l.append(n_A)
            d_n_B_l.append(n_B)
            d_S_a_pieces.append(np.ascontiguousarray(bucket['S_a'][nb]).ravel())
            d_S_b_pieces.append(np.ascontiguousarray(bucket['S_b'][nb]).ravel())
            d_S_c_pieces.append(np.ascontiguousarray(bucket['S_c'][nb]).ravel())
            d_KJ_pieces.append(np.ascontiguousarray(bucket['KJ'][nb]).ravel())
            d_S_a_off_l.append(d_S_a_run); d_S_a_run += n_pno * n_A
            d_S_b_off_l.append(d_S_b_run); d_S_b_run += n_A * n_B
            d_S_c_off_l.append(d_S_c_run); d_S_c_run += n_pno * n_B
            d_KJ_off_l.append(d_KJ_run);   d_KJ_run  += n_pno * n_A
            d_t2_keys_l.append(bucket['t2_keys'][nb])
            d_t2_trans_l.append(bool(bucket['t2_trans'][nb]))
            d_dt_keys_l.append(bucket['dt_keys'][nb])
            d_u_size_l.append(n_A * n_A)
            d_dt_size_l.append(n_B * n_B)
            d_side_l.append(side)
            slot = int(bucket['item_idx'][nb])
            d_tile_off.append(d_tile_off[-1] + n_pno * n_pno)
            if side == 0:
                d_target_off_ij_l.append(_slot_to_off(n_pno, slot))
                d_target_off_ji_l.append(-1)
            else:
                d_target_off_ij_l.append(-1)
                d_target_off_ji_l.append(_slot_to_off(n_pno, slot))

    d_N = len(d_n_pno_l)
    d_u_off = np.zeros(d_N + 1, dtype=np.int64)
    d_u_off[1:] = np.cumsum(d_u_size_l)
    d_dt_off = np.zeros(d_N + 1, dtype=np.int64)
    d_dt_off[1:] = np.cumsum(d_dt_size_l)
    if t2_pno_all is not None and hasattr(t2_pno_all, '_offsets'):
        d_t2_canon_off = np.array(
            [_t2_off_arr[_canon_to_idx[k]] for k in d_t2_keys_l],
            dtype=np.int64)
        d_t2_trans_arr = np.asarray(d_t2_trans_l, dtype=np.int32)
    else:
        d_t2_canon_off = None
        d_t2_trans_arr = None

    bv = {
        # C side
        'c_N': c_N,
        'c_n_pno': np.asarray(c_n_pno_l, dtype=np.int32),
        'c_n_ct': np.asarray(c_n_ct_l, dtype=np.int32),
        'c_n_other': np.asarray(c_n_other_l, dtype=np.int32),
        'c_S_big_off': np.asarray(c_S_big_off_l, dtype=np.int64),
        'c_S_mid_off': np.asarray(c_S_mid_off_l, dtype=np.int64),
        'c_J_bold_off': np.asarray(c_J_bold_off_l, dtype=np.int64),
        'c_S_outer_off': np.asarray(c_S_outer_off_l, dtype=np.int64),
        'c_S_big_flat': (np.concatenate(c_S_big_pieces)
                         if c_S_big_pieces else np.zeros(0)),
        'c_S_mid_flat': (np.concatenate(c_S_mid_pieces)
                         if c_S_mid_pieces else np.zeros(0)),
        'c_J_bold_flat': (np.concatenate(c_J_bold_pieces)
                          if c_J_bold_pieces else np.zeros(0)),
        'c_S_outer_flat': (np.concatenate(c_S_outer_pieces)
                           if c_S_outer_pieces else np.zeros(0)),
        'c_ct_keys': c_ct_keys_l,
        'c_t2_keys': c_t2_keys_l,
        'c_t2_trans': np.asarray(c_t2_trans_l, dtype=bool),
        'c_t2_canon_off': c_t2_canon_off,
        'c_t2_trans_arr': c_t2_trans_arr,
        'c_ct_off': c_ct_off,
        'c_t2_off': c_t2_off,
        'c_target_off_ij': np.asarray(c_target_off_ij_l, dtype=np.int64),
        'c_target_off_ji': np.asarray(c_target_off_ji_l, dtype=np.int64),
        'c_tile_off': np.asarray(c_tile_off, dtype=np.int64),
        'c_side': np.asarray(c_side_l, dtype=np.int32),
        # D side
        'd_N': d_N,
        'd_n_pno': np.asarray(d_n_pno_l, dtype=np.int32),
        'd_n_A': np.asarray(d_n_A_l, dtype=np.int32),
        'd_n_B': np.asarray(d_n_B_l, dtype=np.int32),
        'd_S_a_off': np.asarray(d_S_a_off_l, dtype=np.int64),
        'd_S_b_off': np.asarray(d_S_b_off_l, dtype=np.int64),
        'd_S_c_off': np.asarray(d_S_c_off_l, dtype=np.int64),
        'd_KJ_off': np.asarray(d_KJ_off_l, dtype=np.int64),
        'd_S_a_flat': (np.concatenate(d_S_a_pieces)
                       if d_S_a_pieces else np.zeros(0)),
        'd_S_b_flat': (np.concatenate(d_S_b_pieces)
                       if d_S_b_pieces else np.zeros(0)),
        'd_S_c_flat': (np.concatenate(d_S_c_pieces)
                       if d_S_c_pieces else np.zeros(0)),
        'd_KJ_flat': (np.concatenate(d_KJ_pieces)
                      if d_KJ_pieces else np.zeros(0)),
        'd_t2_keys': d_t2_keys_l,
        'd_t2_trans': np.asarray(d_t2_trans_l, dtype=bool),
        'd_t2_canon_off': d_t2_canon_off,
        'd_t2_trans_arr': d_t2_trans_arr,
        'd_dt_keys': d_dt_keys_l,
        'd_u_off': d_u_off,
        'd_dt_off': d_dt_off,
        'd_target_off_ij': np.asarray(d_target_off_ij_l, dtype=np.int64),
        'd_target_off_ji': np.asarray(d_target_off_ji_l, dtype=np.int64),
        'd_tile_off': np.asarray(d_tile_off, dtype=np.int64),
        'd_side': np.asarray(d_side_l, dtype=np.int32),
        # Output flat layout
        'pairs_by_n_pno_keys': sorted(pairs_by_n_pno),
        'n_pno_offsets': n_pno_offsets,
        'flat_total': flat_total,
    }
    plan['_batched_view'] = bv
    return bv


def _run_cd_batched(plan, bv, t2_pno_all, C_tilde_cache, D_tilde_cache,
                    flat_C_ij, flat_C_ji, flat_D_ij, flat_D_ji,
                    omp_threads):
    """Run the batched C and D kernels and scatter their per-item tiles
    into the per-(n_pno, side) output buffers. Single Cython call per
    side; serial scatter handles target-slot collisions."""
    from pyscf.cc.dlpno_tccsd._cd_batched_cy import (
        c_kernel_batched, d_kernel_batched,
    )
    from threadpoolctl import threadpool_limits

    import time as _cd_time
    _cd_dump = getattr(_run_cd_batched, '_dump_timing', False)
    _cd_t = {'c_gather': 0, 'c_kern': 0, 'c_scatter': 0,
             'd_gather': 0, 'd_kern': 0, 'd_scatter': 0}

    # ---- C side ----
    c_N = bv['c_N']
    if c_N > 0:
        _t0 = _cd_time.perf_counter() if _cd_dump else 0.0
        # Per-cycle gather: ct (from C_tilde_cache, dict) Python loop —
        # C_tilde_cache is a regular dict so per-item lookup is unavoidable
        # without a wider restructure. t2 is gathered via the nogil prange
        # kernel below using absolute offsets into t2_pno_all._buffer (which
        # is a FlatTensorStore — same `_buffer` pointer across cycles, just
        # values updated in-place).
        ct_flat = np.zeros(int(bv['c_ct_off'][-1]))
        c_n_ct = bv['c_n_ct']
        c_n_other = bv['c_n_other']
        c_ct_keys = bv['c_ct_keys']
        c_ct_off = bv['c_ct_off']
        for n in range(c_N):
            ct_val = (C_tilde_cache.get(c_ct_keys[n])
                      if C_tilde_cache is not None else None)
            if ct_val is not None and ct_val.shape[0] == int(c_n_ct[n]):
                ct_flat[c_ct_off[n]:c_ct_off[n + 1]] = ct_val.ravel()

        t2_flat = np.empty(int(bv['c_t2_off'][-1]))
        if (bv['c_t2_canon_off'] is not None
                and hasattr(t2_pno_all, '_buffer')):
            from pyscf.cc.dlpno_tccsd._cd_gather_cy import (
                gather_t2_with_transpose,
            )
            gather_t2_with_transpose(
                c_N, c_n_other,
                bv['c_t2_canon_off'], bv['c_t2_trans_arr'],
                bv['c_t2_off'], t2_pno_all._buffer, t2_flat,
                min(64, c_N),
            )
        else:
            # Fallback: per-item Python loop (back-compat for non-FTS t2)
            c_t2_keys = bv['c_t2_keys']
            c_t2_trans = bv['c_t2_trans']
            for n in range(c_N):
                t2 = t2_pno_all[c_t2_keys[n]]
                if c_t2_trans[n]:
                    t2_flat[bv['c_t2_off'][n]:bv['c_t2_off'][n + 1]] = (
                        t2.T.ravel())
                else:
                    t2_flat[bv['c_t2_off'][n]:bv['c_t2_off'][n + 1]] = (
                        t2.ravel())

        # Per-thread scratch
        max_n_pno = int(bv['c_n_pno'].max(initial=1))
        max_n_ct = int(bv['c_n_ct'].max(initial=1))
        max_n_other = int(bv['c_n_other'].max(initial=1))
        num_threads = min(64, c_N)

        STB = np.empty((num_threads, max_n_pno * max_n_ct))
        GAMMA = np.empty((num_threads, max_n_pno * max_n_other))
        GT = np.empty((num_threads, max_n_pno * max_n_other))
        c_tiles = np.zeros(int(bv['c_tile_off'][-1]))
        if _cd_dump:
            _cd_t['c_gather'] = _cd_time.perf_counter() - _t0
            _t0 = _cd_time.perf_counter()

        with threadpool_limits(limits=1, user_api='blas'):
            c_kernel_batched(
                c_N, max_n_pno, max_n_ct, max_n_other,
                bv['c_n_pno'], bv['c_n_ct'], bv['c_n_other'],
                bv['c_S_big_off'], bv['c_ct_off'][:c_N],
                bv['c_S_mid_off'], bv['c_J_bold_off'],
                bv['c_t2_off'][:c_N], bv['c_S_outer_off'],
                bv['c_tile_off'],
                bv['c_S_big_flat'], bv['c_S_mid_flat'],
                bv['c_J_bold_flat'], bv['c_S_outer_flat'],
                ct_flat, t2_flat,
                STB, GAMMA, GT,
                c_tiles, num_threads,
            )
        if _cd_dump:
            _cd_t['c_kern'] = _cd_time.perf_counter() - _t0
            _t0 = _cd_time.perf_counter()

        # Serial scatter — flatten output buffers per n_pno per side
        n_pno_offsets = bv['n_pno_offsets']
        c_target_ij = bv['c_target_off_ij']
        c_target_ji = bv['c_target_off_ji']
        c_tile_off = bv['c_tile_off']
        c_n_pno = bv['c_n_pno']
        # Build flat views over flat_C_ij / flat_C_ji
        flat_C_ij_views = {n_pno: flat_C_ij[n_pno].ravel()
                            for n_pno in n_pno_offsets}
        flat_C_ji_views = {n_pno: flat_C_ji[n_pno].ravel()
                            for n_pno in n_pno_offsets}
        for n in range(c_N):
            n_pno = int(c_n_pno[n])
            tile_size = n_pno * n_pno
            tile = c_tiles[c_tile_off[n]:c_tile_off[n + 1]]
            if c_target_ij[n] >= 0:
                base = c_target_ij[n] - n_pno_offsets[n_pno]
                flat_C_ij_views[n_pno][base:base + tile_size] -= tile
            else:
                base = c_target_ji[n] - n_pno_offsets[n_pno]
                flat_C_ji_views[n_pno][base:base + tile_size] -= tile
        if _cd_dump:
            _cd_t['c_scatter'] = _cd_time.perf_counter() - _t0

    # ---- D side ----
    d_N = bv['d_N']
    if d_N > 0:
        _t0 = _cd_time.perf_counter() if _cd_dump else 0.0
        d_n_A = bv['d_n_A']
        d_n_B = bv['d_n_B']
        d_dt_keys = bv['d_dt_keys']
        d_u_off = bv['d_u_off']
        d_dt_off = bv['d_dt_off']

        # u = 2*t2 - t2.T (anti-symmetrized); built via nogil prange kernel.
        u_flat = np.empty(int(d_u_off[-1]))
        if (bv['d_t2_canon_off'] is not None
                and hasattr(t2_pno_all, '_buffer')):
            from pyscf.cc.dlpno_tccsd._cd_gather_cy import gather_u_from_t2
            gather_u_from_t2(
                d_N, d_n_A,
                bv['d_t2_canon_off'], bv['d_t2_trans_arr'],
                d_u_off, t2_pno_all._buffer, u_flat,
                min(64, d_N),
            )
        else:
            d_t2_keys = bv['d_t2_keys']
            d_t2_trans = bv['d_t2_trans']
            for n in range(d_N):
                t2 = t2_pno_all[d_t2_keys[n]]
                t2_d = t2.T if d_t2_trans[n] else t2
                u_flat[d_u_off[n]:d_u_off[n + 1]] = (
                    2.0 * t2_d - t2_d.T).ravel()

        # dt (D_tilde_cache, dict) — Python loop unavoidable.
        dt_flat = np.zeros(int(d_dt_off[-1]))
        for n in range(d_N):
            dk = d_dt_keys[n]
            n_B = int(d_n_B[n])
            if dk is not None and D_tilde_cache is not None:
                dt_val = D_tilde_cache.get(dk)
                if dt_val is not None and dt_val.shape[0] == n_B:
                    dt_flat[d_dt_off[n]:d_dt_off[n + 1]] = dt_val.ravel()

        max_n_pno = int(bv['d_n_pno'].max(initial=1))
        max_n_A = int(bv['d_n_A'].max(initial=1))
        max_n_B = int(bv['d_n_B'].max(initial=1))
        num_threads = min(64, d_N)

        SU = np.empty((num_threads, max_n_pno * max_n_A))
        UP = np.empty((num_threads, max_n_pno * max_n_B))
        SCD = np.empty((num_threads, max_n_pno * max_n_B))
        Bint = np.empty((num_threads, max_n_pno * max_n_A))
        d_tiles = np.zeros(int(bv['d_tile_off'][-1]))
        if _cd_dump:
            _cd_t['d_gather'] = _cd_time.perf_counter() - _t0
            _t0 = _cd_time.perf_counter()

        with threadpool_limits(limits=1, user_api='blas'):
            d_kernel_batched(
                d_N, max_n_pno, max_n_A, max_n_B,
                bv['d_n_pno'], bv['d_n_A'], bv['d_n_B'],
                bv['d_S_a_off'], bv['d_u_off'][:d_N],
                bv['d_S_b_off'], bv['d_S_c_off'],
                bv['d_dt_off'][:d_N], bv['d_KJ_off'],
                bv['d_tile_off'],
                bv['d_S_a_flat'], bv['d_S_b_flat'],
                bv['d_S_c_flat'], bv['d_KJ_flat'],
                u_flat, dt_flat,
                SU, UP, SCD, Bint,
                d_tiles, num_threads,
            )
        if _cd_dump:
            _cd_t['d_kern'] = _cd_time.perf_counter() - _t0
            _t0 = _cd_time.perf_counter()

        n_pno_offsets = bv['n_pno_offsets']
        d_target_ij = bv['d_target_off_ij']
        d_target_ji = bv['d_target_off_ji']
        d_tile_off = bv['d_tile_off']
        d_n_pno = bv['d_n_pno']
        flat_D_ij_views = {n_pno: flat_D_ij[n_pno].ravel()
                            for n_pno in n_pno_offsets}
        flat_D_ji_views = {n_pno: flat_D_ji[n_pno].ravel()
                            for n_pno in n_pno_offsets}
        for n in range(d_N):
            n_pno = int(d_n_pno[n])
            tile_size = n_pno * n_pno
            tile = d_tiles[d_tile_off[n]:d_tile_off[n + 1]]
            if d_target_ij[n] >= 0:
                base = d_target_ij[n] - n_pno_offsets[n_pno]
                flat_D_ij_views[n_pno][base:base + tile_size] += 0.5 * tile
            else:
                base = d_target_ji[n] - n_pno_offsets[n_pno]
                flat_D_ji_views[n_pno][base:base + tile_size] += 0.5 * tile
        if _cd_dump:
            _cd_t['d_scatter'] = _cd_time.perf_counter() - _t0

    if _cd_dump:
        print(f'  [CD_DBG] c_N={c_N} '
              f'gather={_cd_t["c_gather"]*1e3:.1f}ms '
              f'kern={_cd_t["c_kern"]*1e3:.1f}ms '
              f'scatter={_cd_t["c_scatter"]*1e3:.1f}ms  '
              f'd_N={d_N} '
              f'gather={_cd_t["d_gather"]*1e3:.1f}ms '
              f'kern={_cd_t["d_kern"]*1e3:.1f}ms '
              f'scatter={_cd_t["d_scatter"]*1e3:.1f}ms', flush=True)


def compute_CD_terms_batched(
        strong_keys, t2_pno_all, pno_spaces, S_pno_cache,
        cc_ints, C_tilde_cache, D_tilde_cache,
        K_ij_kj_all, K_coul_cache,
        pair_lmo_idx, nocc,
        S_pao_full=None, s1e=None, omp_threads=None):
    """Plan-cached batched build of the compute_residual_v2 C and D terms.

    For each strong pair key_ij, returns two (n_pno, n_pno) tiles —
    ``C_term[key_ij]`` and ``D_term[key_ij]`` — that a caller can hand
    to ``compute_residual_v2`` via ``C_term_override`` and
    ``D_term_override``, bypassing the per-pair Python k-loop.

    The C-term symmetrization is:
        C_term = 0.5*C_ij + C_ij.T + 0.5*C_ji.T + C_ji
    The D-term combination is:
        D_term = D_ij + D_ji.T
    """
    from pyscf.cc.dlpno_tccsd._cd_cy import c_kernel, d_kernel

    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    # Plan structure is cycle-invariant once t2_pno_all's keys are fixed.
    plan_key = (tuple(sorted(strong_keys)),
                tuple(sorted(t2_pno_all.keys())))
    _cache_attr = getattr(compute_CD_terms_batched, '_plan_cache', None)
    if _cache_attr is None:
        _cache_attr = {}
        compute_CD_terms_batched._plan_cache = _cache_attr
    plan = _cache_attr.get(plan_key)
    if plan is None:
        plan = _build_cd_plan(
            strong_keys, t2_pno_all, pno_spaces, pair_lmo_idx,
            cc_ints, K_ij_kj_all, K_coul_cache,
            _s_pno_get, nocc)
        _cache_attr[plan_key] = plan

    # Flat output buffers per n_pno — one for each of C_ij, C_ji, D_ij, D_ji.
    flat_C_ij = {}
    flat_C_ji = {}
    flat_D_ij = {}
    flat_D_ji = {}
    for n_pno, pairs in plan['pairs_by_n_pno'].items():
        shp = (len(pairs), n_pno, n_pno)
        flat_C_ij[n_pno] = np.zeros(shp)
        flat_C_ji[n_pno] = np.zeros(shp)
        flat_D_ij[n_pno] = np.zeros(shp)
        flat_D_ji[n_pno] = np.zeros(shp)

    # Single-call batched path: ONE c_kernel_batched + ONE d_kernel_batched
    # per cycle replaces 4224 per-bucket kernel calls. Plan view (built
    # once) flattens all items into per-item offset arrays; per-cycle
    # gather populates ct/t2/u/dt flat buffers, kernel does prange
    # compute, caller does serial scatter.
    bv = _get_or_build_cd_batched_view(plan, pno_spaces, t2_pno_all)
    _run_cd_batched(
        plan, bv, t2_pno_all, C_tilde_cache, D_tilde_cache,
        flat_C_ij, flat_C_ji, flat_D_ij, flat_D_ji,
        omp_threads,
    )

    # --- Assemble final C_term and D_term dicts, keyed by strong pair ---
    C_term = {}
    D_term = {}
    for n_pno, pairs in plan['pairs_by_n_pno'].items():
        Cij = flat_C_ij[n_pno]
        Cji = flat_C_ji[n_pno]
        Dij = flat_D_ij[n_pno]
        Dji = flat_D_ji[n_pno]
        for slot, key_ij in enumerate(pairs):
            C_term[key_ij] = (0.5 * Cij[slot] + Cij[slot].T
                              + 0.5 * Cji[slot].T + Cji[slot])
            D_term[key_ij] = Dij[slot] + Dji[slot].T

    # Keys with n_pno == 0 need zero tiles (for downstream lookup).
    for key_ij in strong_keys:
        if key_ij not in C_term:
            n_pno = pno_spaces[key_ij]['C_pno'].shape[1]
            C_term[key_ij] = np.zeros((n_pno, n_pno))
            D_term[key_ij] = np.zeros((n_pno, n_pno))

    return C_term, D_term


# =========================================================================
# T2 residual: Psi4-compatible two-buffer formulation
# =========================================================================


def compute_residual_v2(
        i, j, t2_pno_all, pno_spaces, nocc,
        F_lmo, s1e, with_df, eps_lmo,
        # Pre-built intermediates (built once per iteration):
        ovL_bare,       # dict: (pair_key, m) -> (n_pno, naux)
        ooL_bare,       # (nocc, nocc, naux)
        S_pno_cache,    # dict: (key1, key2) -> overlap matrix
        K_coul_cache,   # dict: (key_ij, key_pk, occ1, occ2) -> (n_ij, n_pk)
        K_pno_bare,     # dict: pair_key -> (n_pno, n_pno) bare exchange
        # T1-dressed intermediates:
        ovL_dressed,    # dict: (pair_key, m) -> dressed ovL (Eq 92)
        ooL_dressed,    # (nocc, nocc, naux) asymmetric (Eq 91)
        B_tilde,        # (nocc, nocc) dressed beta for Woooo
        Fab,            # dict: pair_key -> (n_pno, n_pno) double-dressed Fock vv
        G_tilde,        # (nocc, nocc) double-dressed Fock oo
        C_tilde_cache,  # dict: (k,i) -> (n_ki, n_ki) gamma intermediate
        D_tilde_cache,  # dict: (i,k) -> (n_ik, n_ik) delta intermediate
        # Mixed-domain integrals:
        J_ij_kj,        # dict: (ij, k) -> (n_ij, n_kj) bare Coulomb (ik|a_ij c_kj)
        K_ij_kj,        # dict: (ij, k) -> (n_ij, n_kj) bare exchange (ia_ij|kc_kj)
        t1_pno=None,
        ladder_precomputed=None,
        K_dressed_override=None,
        cc_ints=None,
        pair_domain=None,
        B_term_override=None,
        E_contrib_override=None,
        C_term_override=None,
        D_term_override=None,
        G_term_override=None,
        S_pao_full=None,
        t1_cache=None,
):
    """Compute T2 residual following Psi4 ccsd.cc lines 2052-2221 exactly.

    Returns P̂-symmetrized R_ij = R_sym + Rn[ij] + Rn[ji].T
    """
    key = (min(i, j), max(i, j))
    data = pno_spaces[key]
    C_pno_ij = data['C_pno']
    n_pno = C_pno_ij.shape[1]

    if n_pno == 0:
        return np.zeros((0, 0))

    from pyscf.cc.dlpno_tccsd.local_df import compute_S_pno as _compute_S_pno
    _getS_misses = [0, 0]  # [get_S, get_S2]

    def _get_S(key_other):
        if S_pno_cache is not None:
            S = S_pno_cache.get((key, key_other))
            if S is not None:
                return S
        _getS_misses[0] += 1
        if S_pao_full is not None:
            S = _compute_S_pno(key, key_other, pno_spaces, S_pao_full, s1e)
        else:
            S = C_pno_ij.T @ (s1e @ pno_spaces[key_other]['C_pno'])
        if S_pno_cache is not None:
            S_pno_cache[(key, key_other)] = S
        return S

    def _get_S2(key_a, key_b):
        """Get S overlap between any two pair keys, with identity fallback."""
        if key_a == key_b:
            return np.eye(pno_spaces[key_a]['C_pno'].shape[1])
        if S_pno_cache is not None:
            S = S_pno_cache.get((key_a, key_b))
            if S is not None:
                return S
        _getS_misses[1] += 1
        if S_pao_full is not None:
            S = _compute_S_pno(key_a, key_b, pno_spaces, S_pao_full, s1e)
        else:
            S = pno_spaces[key_a]['C_pno'].T @ (s1e @ pno_spaces[key_b]['C_pno'])
        if S_pno_cache is not None:
            S_pno_cache[(key_a, key_b)] = S
        return S

    t2_ij = t2_pno_all[key]

    # Pair LMO domain: Psi4 restricts k/l loops to lmopair_to_lmos_[ij]
    if pair_domain is not None:
        _domain_set = set(pair_domain)
    else:
        _domain_set = set(range(nocc))

    # === Symmetric buffer (K̃, A, B, E) ===
    R_sym = np.zeros((n_pno, n_pno))
    _DEBUG_PTERM = getattr(compute_residual_v2, '_debug_pterm', False) or \
                   getattr(compute_residual_v2, '_debug_pterm_all', False)
    _ITER = getattr(compute_residual_v2, '_iter', 0)

    def _rms(M):
        return float(np.sqrt(np.mean(M*M)))

    # --- K̃ (Eq 75): dressed exchange ---
    K_term = np.zeros((n_pno, n_pno))
    if K_dressed_override is not None:
        K_term += K_dressed_override
    else:
        ovL_i_d = ovL_dressed.get((key, i))
        ovL_j_d = ovL_dressed.get((key, j))
        if ovL_i_d is not None and ovL_j_d is not None:
            K_term += ovL_i_d @ ovL_j_d.T
    R_sym += K_term
    if _DEBUG_PTERM:
        print(f"  PTERM iter {_ITER} pair({i},{j}): K_rms={_rms(K_term):.12f} R_K={_rms(R_sym):.12f}")

    # --- A (Eq 76): dressed ladder ---
    # B̃_{ab} = B_{ab} - Σ_k T1_all[k,a]*ovL_k[b,Q] (Eq 93)
    # Phase 1: use pre-built T1 cache; fall back to lazy build otherwise.
    if t1_cache is not None:
        T1_all_ij = t1_cache[key]
    elif t1_pno is not None:
        T1_all_ij = np.zeros((nocc, n_pno))
        for kk in range(nocc):
            T1_all_ij[kk] = _project_t1_to_pair(
                t1_pno, kk, key, S_pno_cache, pno_spaces)
    else:
        T1_all_ij = np.zeros((nocc, n_pno))

    t1_i_pno = T1_all_ij[i] if t1_pno else np.zeros(n_pno)
    t1_j_pno = T1_all_ij[j] if t1_pno else np.zeros(n_pno)
    tau_ij = t2_ij + np.outer(t1_i_pno, t1_j_pno)

    A_term = ladder_precomputed[key] if (
        ladder_precomputed is not None and key in ladder_precomputed
    ) else np.zeros((n_pno, n_pno))
    R_sym += A_term
    if _DEBUG_PTERM:
        print(f"  PTERM iter {_ITER} pair({i},{j}): A_rms={_rms(A_term):.12f} R_KA={_rms(R_sym):.12f}")

    import time as _time
    _pt = {}
    _t0 = _time.perf_counter()
    # --- B (Eq 77/82): Woooo with dressed β ---
    # Use batched override if provided (compute_B_E_batched); else in-place.
    if B_term_override is not None:
        B_term = B_term_override
    else:
        # Psi4-layout B_tilde: tuple (B_local, p_dense) for the per-pair port.
        if isinstance(B_tilde, tuple):
            _Bt_local, _Bt_pdense = B_tilde
            _btilde_lookup = lambda k, l: _Bt_local[_Bt_pdense[k], _Bt_pdense[l]]
        else:
            _btilde_lookup = lambda k, l: B_tilde[k, l]
        B_term = np.zeros((n_pno, n_pno))
        for key_kl, t2_kl in t2_pno_all.items():
            if t2_kl is None or t2_kl.shape[0] == 0:
                continue
            k, l = key_kl
            if k not in _domain_set or l not in _domain_set:
                continue
            S_proj = _get_S(key_kl)
            t2_kl_proj = S_proj @ t2_kl @ S_proj.T
            beta_kl = _btilde_lookup(k, l)
            if k != l:
                beta_lk = _btilde_lookup(l, k)
                B_term += beta_kl * t2_kl_proj + beta_lk * t2_kl_proj.T
            else:
                B_term += beta_kl * t2_kl_proj
    R_sym += B_term
    _pt['B'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
    if _DEBUG_PTERM:
        print(f"  PTERM iter {_ITER} pair({i},{j}): B_rms={_rms(B_term):.12f} R_KAB={_rms(R_sym):.12f}")

    # --- E (Eq 80): t2 × F̃̃_{ab} ---
    # E_tilde = Fab_[ij] - Σ_kl S @ u_kl × K_kl @ S
    # Psi4: starts with Fab_[ij], subtracts u×K (bare K_iajb)
    # Fab includes diag(e_pno) per Eq 101. But our code puts e_pno in the
    # denominator D, not the numerator. Subtract the diagonal to avoid
    # double-counting. (Psi4 uses T -= R/D iterative update which handles
    # this differently.)
    Fab_ij = Fab.get(key, np.zeros((n_pno, n_pno))).copy()
    e_pno = data['e_pno']
    # Match Psi4 exactly: keep full Fab including diag(e_pno).
    # The (e_a+e_b)*T term enters R; balanced by -(F_ii+F_jj)*T from G.
    # Update is T -= R/D_psi4, so R = 0 at convergence (full Psi4 residual).
    E_tilde = Fab_ij.copy()
    # Psi4 E term: subtract u×K from E_tilde (Eq 85: F̃̃ = F̃ - u×K)
    if E_contrib_override is not None:
        E_tilde -= E_contrib_override
    else:
        for key_kl, t2_kl in t2_pno_all.items():
            if t2_kl is None or t2_kl.shape[0] == 0:
                continue
            k, l = key_kl
            if k not in _domain_set or l not in _domain_set:
                continue
            u_kl = 2.0 * t2_kl - t2_kl.T
            _K = get_local_K(cc_ints, key_kl, k, l)
            if _K is None:
                print(f"WARN: No local DF K for pair {key_kl}, skipping E contribution")
                continue
            K_kl = _K
            S_kl_ij = _get_S(key_kl)
            E_tilde -= S_kl_ij @ (u_kl @ K_kl.T) @ S_kl_ij.T
            if k != l:
                E_tilde -= S_kl_ij @ ((2.0*t2_kl.T - t2_kl) @ K_kl) @ S_kl_ij.T

    if _DEBUG_PTERM:
        print(f"  PTERM iter {_ITER} pair({i},{j}): Etilde_rms={_rms(E_tilde):.12f}")
        # Decompose: Fab and the subtraction
        E_tilde_diag = np.diag(E_tilde)
        Fab_diag = np.diag(Fab_ij)
        sub = Fab_ij - E_tilde
        print(f"  PTERM_DECOMP pair({i},{j}): Fab_rms={_rms(Fab_ij):.12f} "
              f"Fab_diag_first={Fab_diag[0]:.10f} sub_rms={_rms(sub):.12f}")
    # Apply: R += t2 @ E_tilde.T + E_tilde @ t2 (Psi4 lines 2146-2147)
    E_term = t2_ij @ E_tilde.T + E_tilde @ t2_ij
    R_sym += E_term
    _pt['E'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
    if _DEBUG_PTERM:
        print(f"  PTERM iter {_ITER} pair({i},{j}): E_rms={_rms(E_term):.12f} R_KABE={_rms(R_sym):.12f}")

    # === Non-symmetric terms (C, D, G) ===
    # Compute C_ij and C_ji in one pass, then form the full P̂ result.
    # P̂(0.5*C + C_ji) = 0.5*(C_ij + C_ij.T) + (C_ji + C_ji.T) for i≠j
    #                  = 1.5*(C + C.T) for i=j (since C_ji = C_ij)
    Rn_ij = np.zeros((n_pno, n_pno))

    # --- C (Eq 78) ---
    if C_term_override is not None:
        Rn_ij += C_term_override
        _pt['C'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
    elif C_tilde_cache is not None:
        C_ij = np.zeros((n_pno, n_pno))
        C_ji = np.zeros((n_pno, n_pno))
        for k in sorted(_domain_set):
            # --- C_ij: gamma(ki) × t2(kj) ---
            key_ik = (min(i, k), max(i, k))
            key_kj = (min(k, j), max(k, j))
            if key_kj in t2_pno_all and t2_pno_all[key_kj] is not None:
                t2_kj_raw = t2_pno_all[key_kj]
                if t2_kj_raw.shape[0] > 0:
                    t2_kj = t2_kj_raw.T if k > j else t2_kj_raw
                    n_kj = t2_kj.shape[0]
                    gamma_ij = np.zeros((n_pno, n_kj))
                    # Bold: J(ik|a_ij c_kj)
                    if J_ij_kj:
                        J_b = J_ij_kj.get((key, k))
                        if J_b is not None:
                            gamma_ij += J_b
                    # C_tilde(ki)
                    ct = C_tilde_cache.get((k, i))
                    if ct is not None:
                        S_ij_ik = _get_S(key_ik)
                        S_ik_kj = _get_S2(key_ik, key_kj)
                        if S_ij_ik is not None and S_ik_kj is not None:
                            gamma_ij += S_ij_ik @ ct @ S_ik_kj
                    S_kj_ij = _get_S(key_kj)
                    C_ij -= gamma_ij @ t2_kj.T @ S_kj_ij.T

            # --- C_ji: gamma(kj) × t2(ki) ---
            key_jk = (min(j, k), max(j, k))
            key_ki = (min(k, i), max(k, i))
            if key_ki in t2_pno_all and t2_pno_all[key_ki] is not None:
                t2_ki_raw = t2_pno_all[key_ki]
                if t2_ki_raw.shape[0] > 0:
                    t2_ki = t2_ki_raw.T if k > i else t2_ki_raw
                    n_ki = t2_ki.shape[0]
                    gamma_ji = np.zeros((n_pno, n_ki))
                    # Bold: J(jk|a_ij c_ki) — use LOCAL DF (J_ji_ki) when
                    # available so it matches Psi4's J_ij_kj_[(j,i)][k_ij]
                    # (fitted with pair ij's local aux metric).  Falling back
                    # to global DF (K_coul_cache) introduces small systematic
                    # differences vs the other bold terms which use local DF.
                    _ci_ij_c = cc_ints.get(key)
                    J_ji_local_c = (_ci_ij_c.get('J_ji_ki', {}).get((key, k))
                                    if _ci_ij_c else None)
                    if J_ji_local_c is not None:
                        gamma_ji += J_ji_local_c
                    elif K_coul_cache:
                        J_b_ji = K_coul_cache.get((key, key_ki, j, k))
                        if J_b_ji is not None:
                            gamma_ji += J_b_ji
                    # C_tilde(kj)
                    ct_j = C_tilde_cache.get((k, j))
                    if ct_j is not None:
                        S_ij_jk = _get_S(key_jk)
                        S_jk_ki = _get_S2(key_jk, key_ki)
                        if S_ij_jk is not None and S_jk_ki is not None:
                            gamma_ji += S_ij_jk @ ct_j @ S_jk_ki
                    S_ki_ij = _get_S(key_ki)
                    C_ji -= gamma_ji @ t2_ki.T @ S_ki_ij.T

        # P̂: Rn[ij] + Rn[ji].T where Rn[ij] = 0.5*C_ij + C_ij.T
        Rn_C_ij = 0.5 * C_ij + C_ij.T
        Rn_C_ji = 0.5 * C_ji + C_ji.T
        C_term = Rn_C_ij + Rn_C_ji.T
        Rn_ij += C_term
        _pt['C'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
        if _DEBUG_PTERM:
            # Print C_ij and C_ji separately to compare with Psi4 (which prints
            # the unsymmetrized C_ij per ordered pair).
            print(f"  PTERM iter {_ITER} pair({i},{j}): C_ij_unsym_rms={_rms(C_ij):.12f} C_ji_unsym_rms={_rms(C_ji):.12f}")
    else:
        C_term = np.zeros((n_pno, n_pno))

    # --- D (Eq 79): antisymmetric ring with delta ---
    if D_term_override is not None:
        Rn_ij += D_term_override
        D_term = D_term_override
    elif D_tilde_cache is not None:
        D_ij = np.zeros((n_pno, n_pno))
        D_ji = np.zeros((n_pno, n_pno))
        for k in sorted(_domain_set):
            key_ik = (min(i, k), max(i, k))
            key_jk = (min(j, k), max(j, k))

            # D_ij: delta(ik) × u(jk)
            if key_jk in t2_pno_all and t2_pno_all[key_jk] is not None:
                t2_jk_raw = t2_pno_all[key_jk]
                if t2_jk_raw.shape[0] > 0:
                    t2_jk = t2_jk_raw.T if j > k else t2_jk_raw
                    u_jk = 2.0 * t2_jk - t2_jk.T
                    S_ij_jk = _get_S(key_jk)
                    S_jk_ik = _get_S2(key_jk, key_ik)
                    if S_ij_jk is not None and S_jk_ik is not None:
                        U_jk_proj = S_ij_jk @ u_jk @ S_jk_ik
                        D_temp = np.zeros((n_pno, n_pno))
                        dt = D_tilde_cache.get((i, k))
                        if dt is not None:
                            S_ij_ik = _get_S(key_ik)
                            D_temp += S_ij_ik @ dt @ U_jk_proj.T
                        # Bold: L = 2K-J with mixed domains
                        K_b = K_ij_kj.get((key, k)) if K_ij_kj else None
                        J_b = J_ij_kj.get((key, k)) if J_ij_kj else None
                        if K_b is not None and J_b is not None:
                            D_temp += (2.0*K_b - J_b) @ u_jk.T @ S_ij_jk.T
                        D_ij += 0.5 * D_temp

            # D_ji: delta(jk) × u(ik)
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
                t2_ik_raw = t2_pno_all[key_ik]
                if t2_ik_raw.shape[0] > 0:
                    t2_ik = t2_ik_raw.T if i > k else t2_ik_raw
                    u_ik = 2.0 * t2_ik - t2_ik.T
                    S_ij_ik2 = _get_S(key_ik)
                    S_ik_jk = _get_S2(key_ik, key_jk)
                    if S_ij_ik2 is not None and S_ik_jk is not None:
                        U_ik_proj = S_ij_ik2 @ u_ik @ S_ik_jk
                        D_temp_j = np.zeros((n_pno, n_pno))
                        dt_j = D_tilde_cache.get((j, k))
                        if dt_j is not None:
                            S_ij_jk2 = _get_S(key_jk)
                            D_temp_j += S_ij_jk2 @ dt_j @ U_ik_proj.T
                        # Bold for D_ji: L(jk|a_ij c_ik) = 2K-J cross-domain
                        # Use ONLY local DF cross-integrals (J_ji_ki, K_ji_ki)
                        _ci_ij = cc_ints.get(key)
                        K_ji_local = _ci_ij.get('K_ji_ki', {}).get((key, k)) if _ci_ij else None
                        J_ji_local = _ci_ij.get('J_ji_ki', {}).get((key, k)) if _ci_ij else None
                        if K_ji_local is not None and J_ji_local is not None:
                            D_temp_j += (2.0*K_ji_local - J_ji_local) @ u_ik.T @ S_ij_ik2.T
                        D_ji += 0.5 * D_temp_j

        D_term = D_ij + D_ji.T
        Rn_ij += D_term
        if _DEBUG_PTERM:
            print(f"  PTERM iter {_ITER} pair({i},{j}): D_ij_unsym_rms={_rms(D_ij):.12f} D_ji_unsym_rms={_rms(D_ji):.12f}")
    else:
        D_term = np.zeros((n_pno, n_pno))
    _pt['D'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()

    # --- G (Eq 81): Fock oo coupling ---
    # Psi4 restricts the k sum to lmopair_to_lmos_[ij]; doing so here
    # makes the S_pno_cache restriction self-consistent (no lookups for
    # k outside domain) and matches Psi4's formula.
    if G_term_override is not None:
        G_term = G_term_override
    else:
        G_ij = np.zeros((n_pno, n_pno))
        G_ji = np.zeros((n_pno, n_pno))
        for k in sorted(_domain_set):
            key_ik = (min(i, k), max(i, k))
            key_jk = (min(j, k), max(j, k))
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
                t2_ik_raw = t2_pno_all[key_ik]
                if t2_ik_raw.shape[0] > 0:
                    t2_ik = t2_ik_raw.T if i > k else t2_ik_raw
                    S_ij_ik = _get_S(key_ik)
                    G_ij -= (S_ij_ik @ t2_ik @ S_ij_ik.T) * G_tilde[k, j]
            if key_jk in t2_pno_all and t2_pno_all[key_jk] is not None:
                t2_jk_raw = t2_pno_all[key_jk]
                if t2_jk_raw.shape[0] > 0:
                    t2_jk = t2_jk_raw.T if j > k else t2_jk_raw
                    S_ij_jk = _get_S(key_jk)
                    G_ji -= (S_ij_jk @ t2_jk @ S_ij_jk.T) * G_tilde[k, i]
        G_term = G_ij + G_ji.T
    Rn_ij += G_term
    _pt['G'] = _time.perf_counter() - _t0

    # Dump per-term timings into a module-level aggregator (optional)
    _accum = getattr(compute_residual_v2, '_term_times', None)
    if _accum is not None:
        for k_, v_ in _pt.items():
            _accum[k_] = _accum.get(k_, 0.0) + v_
        _accum['_n'] = _accum.get('_n', 0) + 1
        _accum['miss_S'] = _accum.get('miss_S', 0) + _getS_misses[0]
        _accum['miss_S2'] = _accum.get('miss_S2', 0) + _getS_misses[1]

    if _DEBUG_PTERM:
        print(f"  PTERM iter {_ITER} pair({i},{j}): G_ij_unsym_rms={_rms(G_ij):.12f} G_ji_unsym_rms={_rms(G_ji):.12f}")
        R_total = R_sym + Rn_ij
        print(f"  PTERM iter {_ITER} pair({i},{j}): Rn={_rms(Rn_ij):.12f} R_total={_rms(R_total):.12f}")
    if _DEBUG_PTERM and not getattr(compute_residual_v2, '_dump_done', False) and \
       not getattr(compute_residual_v2, '_debug_pterm_all', False):
        if i == 0 and j == 0:
            print(f"  DUMP_R2 pair(0,0) npno={n_pno}")
            for a in range(min(n_pno, 5)):
                for b in range(min(n_pno, 5)):
                    print(f"  DUMP_R2[{a},{b}]={R_total[a,b]:.15e}")
            print(f"  DUMP_T2 pair(0,0)")
            for a in range(min(n_pno, 5)):
                for b in range(min(n_pno, 5)):
                    print(f"  DUMP_T2[{a},{b}]={t2_ij[a,b]:.15e}")
            print(f"  DUMP_K pair(0,0)")
            K_dump = K_dressed_override if K_dressed_override is not None else K_term
            for a in range(min(n_pno, 5)):
                for b in range(min(n_pno, 5)):
                    print(f"  DUMP_K[{a},{b}]={K_dump[a,b]:.15e}")
            print(f"  DUMP_EPNO pair(0,0)")
            for a in range(min(n_pno, 10)):
                print(f"  DUMP_EPNO[{a}]={e_pno[a]:.15e}")
            compute_residual_v2._dump_done = True

    # === R_final = R_sym + Rn (already fully P̂-symmetrized) ===
    return R_sym + Rn_ij
