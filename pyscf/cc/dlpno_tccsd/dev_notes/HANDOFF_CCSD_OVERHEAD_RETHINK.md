# Handoff: architecture-level rethink of CCSD Python overhead

## One-paragraph context

Session 2026-04-22 pulled ~15% off CCSD wall time and dropped the scaling exponent from α = 2.10 → 1.96 via four targeted micro-refactors (`compute_B_E_batched` per-kl vectorization, LT1 hoist, `build_G_tilde` loop reorder, Stage 4 matvec). Those were all local "hoist redundant work" or "batch when bucket N is large" patterns. The low-hanging fruit is picked. **The user now wants a session focused on architecture: what would it take to step-change the Python overhead further?** Not more micro-hoists. The question is whether the current data layout, cache/projection discipline, and pool structure are the right shape for the problem — and if not, what a redesigned layer would look like.

See `/home/ec2-user/.claude/projects/-home-ec2-user/memory/project_dlpno_ccsd_overhead.md` for the cumulative findings + per-cycle wall-time map.

## Current state of the world

**Benchmark (water4/8/10, cc-pVDZ, 64 cores, pool=32 / OMP=1 / SCF BLAS=16):**

| chain | baseline t_CCSD | current t_CCSD | Δ |
|---|---:|---:|---:|
| water4  | 24.06s  | 22.80s  | −5.3% |
| water8  | 99.44s  | 86.85s  | −12.7% |
| water10 | 167.09s | 138.31s | −17.2% |

**Scaling exponent:** CCSD α = 1.959, (T) α = 2.032, total α = 1.972. All energies match baseline exactly.

**Per-cycle wall-time map at water8 (post-landings):** cycle ≈ 5.2s.
- setup (ovL/Kcoul/foo): ~0.6s
- jiang block (C_tilde, D_tilde, t1_fock, G_tilde, K_mixed): ~2.4s
- pairs block (B_E_batched + _update_pair loop incl. residual): ~2.0s
- T1 + misc: ~0.2s

**cProfile self-time top hits (whole w8 run, ~157s):**
- `_process_ki` (C_tilde worker): 4.32s self / 14336 calls = 300μs/call
- `_process_ik` (D_tilde worker): 4.22s self / 14336 calls = 290μs/call
- `_compute_t1_residual_psi4`: 4.14s self / 14 calls = 296ms/call (**serial, main thread only**)
- `build_G_tilde`: 2.56s self / 14 calls = 183ms/call (serial, main thread only)
- `compute_S_pno`: 5.07s / 89k calls = 57μs/call (cache fill, not in hot path)
- `_project_t1_to_pair`: 2.87s / 1.52M calls = 1.9μs/call — **these add up across all callers**

## Why "more of the same" won't work

### Rule of thumb we learned

Batched-numpy vectorization (gather-then-stack-then-one-gemm) wins when **per-call bucket N ≥ 50**. Below that, np.stack setup + batched matmul overhead exceeds the per-item dispatch savings. This is why `compute_B_E_batched` worked (its inner loop scans ~214 strong pairs → N~100+) but the same refactor regressed in `compute_C_tilde` (inner loop over pair-local domain ~20 items → N~10).

**Most remaining per-kl / per-l loops in the code are pair-local.** Bucket N in `compute_C_tilde`, `build_D_tilde`, `compute_residual_v2`'s C/D/G k-loops, `compute_ladder`, etc. is ~5–20. Applying the BE pattern directly there either makes things slightly worse or is a wash.

### Where we're actually stuck

Two structural ceilings:

1. **Per-pair pool tasks each do their own small-N loops.** ThreadPoolExecutor dispatches 400-1000 per-pair tasks per cycle. Each task does ~20 Python iterations with small (n_pno, n_pno) matmuls at n_pno ≈ 25. Per-iter numpy dispatch is ~5-50μs. Under the GIL, the 32 workers serialize on Python sections, so the effective parallelism for pure-Python work collapses toward 1. The fact that 64 cores aren't fully utilized on Python-heavy phases is evidence.

2. **Redundant projection work across callers.** `_project_t1_to_pair` is called 1.52M times per full run. `compute_C_tilde`, `build_D_tilde`, `t1_fock`, `_compute_t1_residual_psi4`, `compute_ladder`, `t1_ints`, `compute_B_tilde` each rebuild per-pair T1 projections from scratch, with no shared cache across functions.

## The architecture question for next session

The user wants to step back and think about **data-layout and dataflow redesign** that would unlock either:

1. **Bigger batched operations that span pairs**, not just items within a pair. E.g., "for every strong pair, compute T1 projected into its PNO basis" → currently a per-caller loop of 214 small matmuls; could be one padded 3D tensor + one GEMM if we accept the padding cost.
2. **Moving per-pair Python loops into C-level kernels** (Cython .pyx like the existing `_ladder_cy.pyx`, or Numba — but Numba isn't installed and adding it is a dep decision).
3. **Dataflow restructuring**: a single cycle-start "build all shared projections" pre-pass that every downstream kernel queries from a flat array indexed by canonical pair-index, avoiding the dict-lookup overhead that currently dominates `_project_t1_to_pair`.
4. **Threading model**: is the ThreadPoolExecutor of 32 workers the right choice when Python/GIL serializes the small-matmul dispatch? Would a `multiprocessing`-based pool (separate interpreter per worker, no GIL contention) be better for the pair-residual phase specifically, accepting the data-copy cost via shared memory?

## Concrete starting points for next session

- **Read the cProfile output at `/tmp/profile_out3.txt`** for the current state (post all landings). The top self-time lines still include `_process_ki`, `_process_ik` (each ~4.2s self, ~300μs/call × 14k calls).
- **Profile with `py-spy` or a sampling profiler** to see GIL contention during the pairs phase. If workers spend significant time waiting on the GIL, that's the architecture signal for multiprocessing.
- **Measure CPU utilization** during a cycle (`top -p <pid>` or `perf stat`): if 64 cores show 30-40% average, Python/GIL is the ceiling. If 80%+, we're compute-bound and more architecture work has diminishing returns.
- **Consider a "projection batch builder"**: at cycle start, build one flat `(n_pair_pair, n_pno²)` tensor of all (ij → kl) T2-projections. Everything downstream indexes it. The initial build is one big batched gemm (leveraging S_pno_cache which is already a flat dict). Eliminates the 1.5M `_project_t1_to_pair` calls.
- **Consider Cython for `_process_ki` / `_process_ik`**: these are per-pair workers called hundreds of times per cycle, each with a per-l inner loop of 20+ small matmuls. There's an existing `_ladder_cy.pyx` precedent in this module. A Cython version of the per-l inner loop could match the BE win without needing big-N buckets.
- **Fij_bar batching in `_compute_t1_residual_psi4`** is a remaining micro-opportunity (L640-668), but sub-1% expected. Probably not worth it as an isolated fix.

## What's been ruled out

- **ThreadPoolExecutor dispatch overhead is NOT the bottleneck**. Testing showed that the wrapper `_be_one` → batched-call API refactor (step 1 of the session plan) was a wash perf-wise. Pool dispatch adds ~50-200μs per task but the real work per task is much bigger.
- **Pool contention between concurrent `compute_C_tilde` / `build_D_tilde` / `t1_fock` driver threads is NOT the bottleneck**. Running them sequentially (step 2 of the session plan) was also a wash. The pool schedules tasks from all three drivers interleaved and the aggregate CPU is the same.
- **Numba is not installed** and was checked. Adding it requires a dep decision.

## Files and landmarks (current installed state)

All changes synced to installed pkg at `/environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/`. Source of truth:

- `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/residual.py`:
  - `compute_B_E_batched` at L900 — **BE-vectorized gather-then-batched-matmul**, signature takes `B_tilde_per_ij` dict.
  - `build_G_tilde` at L55 — **loop reordered to (i, l, j); u_lj cache**.
  - `compute_C_tilde` at L468 — per-ki _process_ki worker; per-l loops at Term 1/3/4 unchanged (tried batching, reverted).
  - `build_D_tilde` at L149 — per-ik _process_ik worker; per-l loops unchanged.
- `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/lccsd.py`:
  - `_compute_t1_residual_psi4` at L537 — **LT1 cache + Stage 4 matvec vectorization**.
  - `_run_dlpno_lccsd` at L1003 — main CCSD driver.
  - C/D/F jiang block at L1499+ — **run sequentially now (no threading)**.
  - `_update_pair` at L1610 — per-pair worker (t1_ints, compute_ladder, Fab, residual).
  - BE batched call site at L1591 — **single call over all keys with `_pool`**.
- `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/local_df.py`:
  - `t1_ints` at L904, `t1_fock` at L959, `compute_ladder` at L1149, `compute_S_pno` at L355, `_project_t1_to_pair` consumers throughout.

## Validation commands

```bash
# CCSD + (T) full scaling run:
cd /home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling
/environments/miniconda3/envs/tmc/bin/python -u run_water_scaling.py \
    --basis cc-pvdz --ncores 64 --scf-blas 16 --chains 4,8,10 \
    --out results/water_scaling_<tag>.json

# Compare against current state:
/environments/miniconda3/envs/tmc/bin/python -c "
import json, numpy as np
base = '/home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling/results/'
cur  = {**json.load(open(base+'water_scaling_stage4_w4.json')),
        **json.load(open(base+'water_scaling_stage4_w8.json')),
        **json.load(open(base+'water_scaling_stage4_w10.json'))}
new  = json.load(open(base+'<your new>.json'))
for k in new:
    dt = new[k]['t_ccsd'] - cur[k]['t_ccsd']
    de = (new[k]['e_ccsd'] - cur[k]['e_ccsd'])*1e6
    print(f'{k}: Δt_CCSD={dt:+.2f}s  Δe_CCSD={de:+.4f} μEh')
"
```

## Git status

Branch `dlpno_tccsd` — uncommitted edits from this session across `lccsd.py`, `residual.py` on top of the already-uncommitted (T) edits from earlier sessions. **Do NOT commit without user approval.**
