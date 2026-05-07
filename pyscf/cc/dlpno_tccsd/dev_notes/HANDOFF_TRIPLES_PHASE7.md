# Handoff — DLPNO-(T) arithmetic / locality port

## Context

This handoff continues the `dlpno_restructure` branch after Phase 5e
(commit `5821c2012`).  The CCSD side is in good shape; the remaining
performance target is the **(T) correction**.

## Where we are (water8, cc-pVDZ, 64 cores, TightPNO)

| phase | CCSD wall | (T) wall | total | delta vs prev |
|---|---|---|---|---|
| pre-Phase-4 baseline (Numba + dual path) | 104.5 s | 20 s  | 125 s | — |
| Phase 4  (Cython C_tilde 3+4)            |  93.5 s | 20 s  | 114 s | −9% |
| Phase 5a (+ Cython D_tilde 3+4)          |  88.6 s | 20 s  | 109 s | −4% |
| Phase 5c (+ Cython be_kernel + plan)     |  82.0 s | 20 s  | 102 s | −6% |
| Phase 6  (+ OMP scope on Cython kernels) |  75.0 s | 20 s  |  95 s | −7% |
| **Phase 5e (+ Cython C/D contractions)** | **69.0 s** | **20 s** | **89 s** | −6% |

CCSD is now ~34% faster than baseline.  (T) has not been touched — it's
still at ~20 s on water8 and is the dominant remaining cost.

## (T) scaling gap (from project memory)

Jiang target exponent vs chain length: **1.78**.  Our current: **2.20**
after Step 1 of HANDOFF_TRIPLES_FLOPS_RESTRUCTURE.md.  Closing the 0.4
exponent gap needs the arithmetic / locality work below, **not**
Python-overhead removal.

## Stage 6 profile (water8, all three per-triple loops use `_process_one_triple`)

```
Stage 6 wall time:          20.57 s
  prescreen (1593 triples):  8.75 s   42%   (loose thresholds)
  tight     ( 670 triples):  5.23 s   25%
  degenerate(318 pairs×2):   4.33 s   21%
  DF infra + post:           2.26 s   11%
```

Per-triple work (prior profiling, directionally still valid):

| fn                         | w4 / w10 (ms) | per-call exp |
|---|---|---|
| `_triple_pno_union_psi4`   | 27  → 41      | 0.46         |
| `_build_triple_local_DF`   | 84  → 173     | 0.79         |
| `_w3_intermediate`         | 37  → 45      | 0.22         |
| remaining (`t2_mr` etc.)   | 58  → 90      | 0.48         |
| total `_process_one_triple`| 206 → 349     | 0.58         |

## Where the (T) gap actually lives — three arithmetic / locality issues

Not Python dispatch overhead.  Porting `_w3_intermediate`'s body to
hand-rolled Cython was attempted this session and **regressed** the
(T) wall (5.0 s → 8.65 s on water4) because hand-rolled loops at
n_tno ≈ 25 cannot match BLAS throughput.  The real gaps:

### 1. DF integral assembly per triple (`_build_triple_local_DF`)

File: [`pyscf/cc/dlpno_tccsd/lccsd_t.py`](pyscf/cc/dlpno_tccsd/lccsd_t.py)
lines ~L542-684.

- **Current**: Python loop over aux centers per triple; per-center
  `qij`/`qia`/`qab` slice + batched matmul to produce `ovL_sc`,
  `vvL_sc`, `ooL_sc` in the triple-local TNO basis.  Good Psi4-style
  center grouping, but still per-triple Python outer loop.
- **Gap vs Psi4**: Psi4's
  [`compute_lccsd_t0`](/environments/psi4_jiang/psi4/src/psi4/dlpno/triples.cc)
  L670+ has the same algorithmic structure but in C++.
  The sparsity masks (`lmo_aux_mask`, `riatom_to_paos_ext_dense`,
  `riatom_to_lmos_ext_dense`) already exist and match Psi4.
- **Locality issue**: the per-triple `qab_A_uu` slice of the full-aux
  `qab_atom` stack touches memory that has no triple-local reuse.
  At water10, `naux_ijk` is saturated (~176) but `n_domain` is still
  growing (17.6 / 40), so the memory traffic per triple scales worse
  than the compute.

### 2. vooo materialize-vs-stream (HANDOFF_TRIPLES_FLOPS_RESTRUCTURE.md Step 2)

File: [`pyscf/cc/dlpno_tccsd/lccsd_t.py`](pyscf/cc/dlpno_tccsd/lccsd_t.py)
lines L826-829 (`t2_mr` build) and L442-455 (`K_ooov` subtract).

- **Current**: `t2_mr = np.zeros((m_dom, 3, n_tno, n_tno))` built
  per-triple with `_proj_t2` (O(1) per pair via `_U_for` cache).
  Then vectorised across the 6 S_3 permutations via one batched
  matmul `np.matmul(K_batch, t2_batch.reshape(6, m_dom, n*n))`.
- **Psi4's form** ([triples.cc L832-844](
  /environments/psi4_jiang/psi4/src/psi4/dlpno/triples.cc)):
  per-m loop `for l_ijk in nlmo_ijk`, build `T_il` on the fly from
  `T_iajb_[il]` (pair-PNO T2, small), subtract a rank-1 outer-product
  from `W`.
- **Locality issue**: our batched form touches
  `(m_dom × 3 × n_tno²)` memory per triple.  At water10 with
  m_dom=17.6, that's ~27 KB per perm-slot per triple; the second
  matmul (`A_al @ t2_mbc`) streams this in.  Psi4's per-m loop
  keeps the working set at one `(n_tno, n_tno)` tile and reuses it
  across the rank-1 subtract, staying in L1.

### 3. Tensor-contraction memory bandwidth at large n_tno

File: `_w3_intermediate` (lccsd_t.py L347-491), specifically the
`K_ab_cache` build at L414-419 and the K_ovvv perm loop at L430-435.

- **Compute breakdown per triple** at n_tno=25, naux=180:
  - `K_ab_cache`: 3 × tensordot (n, naux) × (n, n, naux) = ~9M flops
  - K_ovvv perm loop: 6 × matmul (n², n) × (n, n) = ~4.7M flops
  - K_ooov subtract: 1 batched matmul (6, n, m_dom) × (6, m_dom, n²) = ~1.5M flops
- **Memory traffic**: `K_ab_cache` writes 3 × n³ × 8 B = 375 KB per
  triple; the perm loop reads it back, producing `W` (125 KB).  At
  1600 triples this is ~0.8 GB of cache traffic per CCSD iteration.
- **Bandwidth dominates**: at n_tno=25 the compute is small enough that
  BLAS dispatch + cache misses account for most of the wall time.
  Porting to hand-rolled Cython **does not help** because it can't beat
  BLAS throughput.  Porting to Cython **with scipy cython_blas.dgemm**
  could shave numpy dispatch overhead but the BW bottleneck remains.

## Recommended refactor strategy

Attack the three gaps in order of risk/payoff:

### Step A: per-m streaming restructure (addresses Gap 2)

Highest payoff and the one most likely to move the scaling exponent.

1. In `_process_one_triple` (lccsd_t.py L826-829), **drop the
   `t2_mr` pre-allocation**.  Instead, extend the `_U_for` /
   `_proj_t2` caches to be callable per-m.
2. In `_w3_intermediate` L437-455 (the K_ooov fast path), replace the
   batched matmul with a per-m loop that:
   - For each `l_ijk` in 0..m_dom:
     - Project `T_il` (the pair-PNO T2 for pair `(outer_lmo, l)`)
       to the triple-TNO basis — small 2-matmul projection.
     - For each of the 6 permutations, do rank-1 update:
       `W[..., c] -= T_il[..., :] * K_ooov[pidx, :, l_ijk]`
   - This keeps `T_il` in registers/L1 instead of streaming a large
     `t2_sc_full`.
3. The per-m loop is a prime target for a nogil Cython kernel with
   prange over `l_ijk`.  Per-m work is small (2 matmuls + 1 rank-1
   update), and prange gives cheap parallelism even at
   m_dom ≈ 16–20.
4. **Validation**: energy identical to ~30 μEh (FP reordering only);
   (T) exponent on water4/8/10 should drop from 2.20 toward 1.9–2.0.

### Step B: Cython kernel for W-build + energy using `scipy.linalg.cython_blas` (addresses Gap 3)

After Step A works:

1. Wrap the entire `_w3_intermediate` body (K_ab_cache, W build, T, V,
   energy) in a single nogil Cython function.
2. Call BLAS `dgemm`/`dgemv` from within nogil via
   `scipy.linalg.cython_blas.dgemm` (same pattern as `_cython_probe`
   already uses).  This preserves BLAS throughput.
3. The win here comes from **eliminating numpy's Python/dispatch
   overhead** (~15 numpy calls × 5 μs × 1600 triples × 3 loops ≈ 360 ms
   wall savings on water8).  Modest but clean.
4. **Do not hand-roll the matmuls** — that path was tested this
   session and is 3× slower than BLAS at n_tno=25.

### Step C: DF infra locality (addresses Gap 1)

Lowest-risk, highest-effort, and probably the biggest single
contributor at water10 (`_build_triple_local_DF` is 84–173 ms).

Options:
- **Option C1**: restructure the per-center loop so the outer iteration
  is over (center, triple) jointly, amortising center metadata across
  triples.  Requires a scheduler that groups triples by which aux
  centers they touch.
- **Option C2**: port the per-center inner matmul to a Cython kernel
  that uses `cython_blas.dgemm` for the (nQ_c, nu, nu) × (nu, n_tno)
  step.  Eliminates numpy overhead, preserves BLAS throughput.
- **Option C3**: (ambitious) batch triples whose aux masks overlap
  heavily, sharing the DF fetch across them.  Big structural change.

Start with C2 if A and B land clean.  C1 / C3 are dedicated
multi-session projects.

## Files and kernels already in place

Cython infra:
- [`pyscf/cc/dlpno_tccsd/setup.py`](pyscf/cc/dlpno_tccsd/setup.py) —
  builds `.so` in-place, picks up any new `.pyx` automatically.
- [`_c_tilde_cy.pyx`](pyscf/cc/dlpno_tccsd/_c_tilde_cy.pyx) —
  t3/t4 kernels used by compute_C_tilde_batched and build_D_tilde_batched.
- [`_cd_cy.pyx`](pyscf/cc/dlpno_tccsd/_cd_cy.pyx) —
  c_kernel, d_kernel for the residual C/D contractions (Phase 5e).
- [`_be_cy.pyx`](pyscf/cc/dlpno_tccsd/_be_cy.pyx) —
  be_kernel for compute_B_E_batched_v2 (Phase 5c).
- [`_cython_probe.pyx`](pyscf/cc/dlpno_tccsd/_cython_probe.pyx) —
  has a working `blas_dgemm` example of `scipy.linalg.cython_blas`
  called from nogil.

Existing (T) code (to refactor):
- [`pyscf/cc/dlpno_tccsd/lccsd_t.py`](pyscf/cc/dlpno_tccsd/lccsd_t.py) — 1917 lines
  - `_triple_pno_union_psi4`  at L140
  - `_w3_intermediate`        at L347
  - `_build_triple_local_DF`  at L542
  - `_process_one_triple`     at L687
  - `_process_degenerate_pair` at L885
  - `run_lccsd_t_ext`         at L1070

Psi4 reference (read-only):
- `/environments/psi4_jiang/psi4/src/psi4/dlpno/triples.cc`
  - `compute_lccsd_t0`        at L609 — main per-triple body
  - K_ooov pre-build block    at L775-809 (our Step 1, already done)
  - vooo per-m loop           at L832-844 (Step A of this handoff)

Memory notes (in `/home/ec2-user/.claude/projects/-home-ec2-user/memory/`):
- `project_triples_psi4_port.md` — cumulative history, rule-out list
- `project_dlpno_accuracy_gaps.md` — overall PySCF-vs-Psi4 gaps

Benchmark harness:
- `/home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling/run_water_scaling.py`
- `/home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling/results/*.json` —
  prior timing artefacts; comparable baselines are `water_scaling_psi4match.json`
  for (T) and `/tmp/phase5f_w8*.json` for the Phase 5e CCSD state.

## Sync reminder

Dev code is at `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/`.
The running pyscf installation is at
`/environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/`.
Every edit + `.so` rebuild needs to be copied to the install location
before benchmarks see the change.  `setup.py build_ext --inplace`
builds into the dev tree only.

## Run / validate commands

```bash
# Build Cython
cd /home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd
/environments/miniconda3/envs/tmc/bin/python setup.py build_ext --inplace

# Sync to installed pkg
cp *.so *.pyx \
   /environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/

# Water scaling (the exponent is what matters for gap closure)
cd /home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling
DLPNO_NCORES=64 /environments/miniconda3/envs/tmc/bin/python -u \
    run_water_scaling.py --chains 4,8,10 \
    --out results/phase7_stepA.json --force
```

Energy gate after Step A: each chain must match the Phase 5e energies
below to ≤ 30 μEh:

| chain  | e_ccsd_t          |
|--------|-------------------|
| water4 | -304.98977525 Eh  |
| water8 | -609.98093506 Eh  |
