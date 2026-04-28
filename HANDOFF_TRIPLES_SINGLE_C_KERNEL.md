# DLPNO-(T) single-C-kernel plan — Session 3 Phase 3c handoff

## Goal

One C function `DLPNOcompute_E_T0` that takes the full per-CCSD-run
arena + the triples list, runs `#pragma omp parallel for
schedule(dynamic) reduction(+:E_T)` over all triples, and returns the
total E(T). Eliminates ALL per-triple Python orchestration, ctypes
round-trips, and ThreadPoolExecutor GIL contention.

## Current state (committed)

water-10 / 64 cores / TightPNO baseline:
| Phase | (T) wall | Cumulative |
|---|---|---|
| Python baseline | 29.75s | — |
| Session 1 (DF kernel) | 24.09s | -19% |
| Session 2 (TNO body BLAS) | 23.29s | -22% |
| Session 3a (LAPACK in C) | 15.13s | **-49%** |
| Session 3b (W3 in C) | 15.65s | -47% (noise) |

Anchor: E(T) = -0.0323130689 (water-10), -0.0129209861 (water-4).
Bit-perfect through all phases.

## Why Phase 3c is needed

Phase 3b (W3 → C) gave no measurable gain because the Cython kernel was
already nogil + cython_blas. The remaining ~5s of (T) wall is in
`_process_one_triple`'s Python orchestration:
1. Set up u_pks list (60+ pair keys) from per-triple lookups
2. Marshal U cache flat buffers, call C
3. Build K_ab_cache (3 numpy tensordots)
4. Build K_ooov (1 numpy matmul)
5. Build K_jk/K_ik/K_ij (3 numpy matmuls) for V intermediate
6. Marshal U_flat / T2_flat / transpose_flags for W3 ctypes call
7. Marshal eps/t1/etc, call W3 ctypes

That's ~5-7 ctypes round-trips and ~10 small numpy ops per triple.
With 2455 triples, the cumulative Python overhead is the remaining
gap. It only goes away when the entire post-TNO body becomes ONE C
function.

## Required arena (passed once per CCSD run)

### Per-pair flat data (over ALL pairs)
```
n_pairs:           int
pair_keys:         (n_pairs, 2) int64 — sorted (i, j)
pair_lookup:       (nocc, nocc) int64 — pair_idx or -1
pair_paos_n:       (n_pairs,) int32
pair_paos_off:     (n_pairs+1,) int64
pair_paos_flat:    int64
n_pno_arr:         (n_pairs,) int32
X_pno_off:         (n_pairs+1,) int64
X_pno_flat:        float64
T2_off:            (n_pairs+1,) int64
T2_flat:           float64
```

### Per-LMO flat data
```
t1_off:            (nocc+1,) int64
t1_flat:           float64    — concat of t1_pno[i] over diagonal pairs
pao_domains_off:   (nocc+1,) int64
pao_domains_flat:  int64      — concat of pao_domains_triple[i]
diag_pair_idx:     (nocc,) int64 — pair_idx of (i,i) or -1
```

### PAO/aux globals (already partly cached from Sessions 1+)
```
F_pao_full, S_pao_full, F_lmo, j2c_full, lmo_aux_mask
qij_atom_flat/qia_atom_flat/qab_atom_flat + offsets + n_aux/n_lmo/n_pao
aux_atom_ids, aux_pos_in_atom
riatom_to_lmos_ext_dense, riatom_to_paos_ext_dense
nonneg_pair: (nocc, nocc) bool→long — for triple_domain build
```

### Triples list
```
n_triples:         int
ijk_list:          (n_triples, 3) int64
weak_pair_count:   (n_triples,) int8 — for SC-MP2 prescreen routing
```

### Config
```
T_CutTNO, S_cut_domain, has_t1
```

### Output
```
et_per_triple:     (n_triples,) float64
return value:      double E_T (sum)
```

## Per-triple body (single C function)

```c
double compute_one_triple(int i, int j, int k, /* arena */) {
    // 1. Build triple_paos union (sorted) from pao_domains[i,j,k]
    //    Stack-allocated int array, in-place sort.
    
    // 2. Build triple_domain: m s.t. nonneg[i,m] && nonneg[j,m] && nonneg[k,m]
    //    Linear scan over nocc.
    
    // 3. Look up pair_idx_3 = [pair_lookup[ij], pair_lookup[jk], pair_lookup[ik]]
    //    Skip triple if any is -1.
    
    // 4. TNO transform — reuse Phase 3a kernel.
    //    Returns X_tno_ijk, eps_tno, n_tno, n_pao_can.
    
    // 5. Build aux_idx (lmo_aux_mask[i] | [j] | [k]) and jhi via in-C eigh.
    
    // 6. Build local DF (Session 1 kernel) → ovL, vvL, ooL.
    
    // 7. Enumerate u_pks: for r in {i,j,k}:
    //      for m in triple_domain: add (min(r,m), max(r,m))
    //      add (r, r)  // for t1
    //    for r1, r2 in {i,j,k}^2: add (min, max)  // for t2_block
    //    Look up pair_idx for each via pair_lookup.
    //    Build U cache for these via DLPNObuild_U_for_triple.
    
    // 8. Build t2_block (3, 3, n_tno, n_tno):
    //    For each (p, q) in 3x3, locate U_p, U_q, T2_{pq}, transpose flag.
    //    Use 2 dgemms: tmp = T2 @ U_q; t2_block = U_p.T @ tmp.
    
    // 9. K_ab_cache (3, n, n, n): for ip in {0,1,2},
    //    K_ab[ip, a, b, f] = sum_L ovL[ip, a, L] * vvL[b, f, L]
    //    One dgemm per ip with reshape/transpose.
    
    // 10. K_ooov (3, 3, n, m_dom): one batched dgemm
    //     K_ooov[p, q, a, m] = sum_L ovL[p, a, L] * ooL[q, m, L]
    
    // 11. K_jk/K_ik/K_ij (n, n) each — for V intermediate (T1 path).
    //     Three small dgemms.
    
    // 12. Project t1: t1_lmo (3, n_tno) from per-LMO t1[r] @ U_diag.
    //     Diagonal pairs (r, r) lookup.
    
    // 13. fvo (n_tno, 3): C_tno_sc.T @ fock_ao @ C_lmo[:, [i,j,k]]
    //     For PAO basis: fvo[a, p] = sum_u X_tno_ijk[u, a] * F_pao[trip[u], lmo_p_in_pao]?
    //     Actually fvo enters W3 via t1_sc; check derivation.
    //     ACTUALLY fvo isn't passed to W3 — t1_sc comes from t1_pno proj.
    
    // 14. W3 + energy contraction — call DLPNOcompute_w3_energy.
    
    // 15. Free per-triple buffers; return et_ijk.
}
```

## Session-by-session plan

### Session 3c-1 (next): C orchestrator skeleton + arena builder
- Write `dlpno_triples_orch.c` with `DLPNOprocess_one_triple` that does
  steps 1-7 (TNO, DF, jhi, U cache build).
- Returns intermediate buffers via output args (ovL, vvL, ooL, X_tno,
  eps_tno, U_flat, et_per_triple TBD).
- Python builds the arena, calls per triple, then does steps 8-14 in
  Python for now. Validate water-4 anchor.
- Estimate: 600 lines C + 300 lines Python.

### Session 3c-2: t2_block + K_ab + K_ooov + K_*_for_V in C
- Add steps 8-11 to the per-triple C function.
- Validate water-4 anchor.
- Estimate: 300 lines C.

### Session 3c-3: W3 connection + energy contraction in C
- Add steps 12-14: t1 projection + W3 + energy.
- Returns et_ijk directly.
- Replace per-triple Python body with one ctypes call.
- Validate water-4 + water-10 anchors. Measure speedup.
- Estimate: 200 lines C + cleanup of `_process_one_triple` Python.

### Session 3c-4: OMP parallel-for over triples
- Add public `DLPNOcompute_E_T0_full` that wraps the per-triple
  function in `#pragma omp parallel for schedule(dynamic)
  reduction(+:E_T)`.
- Replace ThreadPoolExecutor in `run_lccsd_t_ext` with one ctypes call.
- Each thread allocates its own scratch (no shared state).
- Validate anchors. Measure additional speedup.
- Estimate: 100 lines C + 200 lines Python rework.

## Estimated total

- ~1500 lines new C (orchestrator + helpers)
- ~600 lines Python rework (arena builder + simplified
  `run_lccsd_t_ext`)
- 4 sessions of careful piecewise validation

## Expected outcome

- water-10 (T) wall: 15.13s → 10-12s (Psi4 territory)
- Cleaner architecture: per-triple work is one C function, no Python
  during compute.
- Foundation for the parallel CCSD-as-single-kernel arc (separate
  multi-session effort).

## ACTUAL OUTCOME (Phases 3c-1 through 3c-3)

Phase 3c-1 + 3c-2/3 landed.  Validated correct (water-4
E(T) = -0.0129209861 bit-perfect; water-10 E(T) = -0.0323130688
within FP-summation noise).  But (T) wall on water-10 is **23s vs
Phase 3a's 15.13s** — **slower, not faster**.

Optimizations attempted:
1. Cached globals (F_pao, S_pao, j2c, lmo_aux_mask) on the function
   to avoid per-triple ascontiguousarray + bool→int64 copy.  No win.
2. W3 task offsets aliased into U_flat_cache + u_T2_flat (no data
   copy from cache to per-task arena).  Saved 2.77s (26 → 23).
3. __thread-backed scratch arena: ~25 per-triple mallocs replaced
   with grow-only persistent buffers.  No measurable win.
4. Global per-CCSD pair arena (X_pno + T2 flat, indexed by pair_idx);
   per-triple just looks up offsets.  No win.

Cumulative: malloc churn, marshalling, global-array copies — none
were the dominant cost.  The 8s gap to Phase 3a is somewhere else,
likely:
- Cache thrashing from oversized persistent scratch buffers
  (vvL_sc grows to max naux_ijk ~250, kept across all triples even
  when current triple uses ~175).
- Different BLAS dispatch patterns vs numpy/Cython
- Possibly GIL contention through ctypes (need to verify CDLL
  releases GIL during the call).

## Recommendation

Phase 3c-2/3 (one C function per triple) is architecturally COMPLETE
and CORRECT but does NOT outperform Phase 3a on water-10.  The end
state the user requested (single C kernel mirroring Psi4) is in
place; what remains for the perf win is **Phase 3c-4: OMP-over-
triples in C with thread-pinned scratch**, which eliminates the
ThreadPoolExecutor + GIL roundtrips entirely.  Estimated +500 lines
C, 1 session.

For now, default `DLPNO_TRIPLE_ORCH_FULL=0` (keep Phase 3a as the
fast path).  The orch infrastructure stays in tree for Phase 3c-4 to
build on.

## Known risks / pitfalls

- **Per-thread BLAS oversubscription** when OMP loops over triples and
  each thread also runs OpenBLAS multi-threaded. Need to set
  `openblas_set_num_threads(1)` from inside the OMP region or rely on
  `OPENBLAS_NUM_THREADS=1` env var (already used in benchmarks).
- **Memory**: each triple allocates ~5 MB of scratch (ovL, vvL, ooL,
  K_ab, K_ooov, W, V, T). With 32-64 OMP threads that's 160-320 MB
  concurrent. Per-thread pre-allocated scratch from the orchestrator
  outer level avoids per-triple malloc/free churn.
- **Correctness of fancy indexing** in C: the
  `riatom_to_lmos_ext_dense[centerQ, dom_arr]` etc. expressions need
  careful translation. Validated already in Sessions 1-2.
- **Different LAPACK reduction order across threads** can give E(T)
  differences at the 1e-9 level. Acceptable; mention in commit.
