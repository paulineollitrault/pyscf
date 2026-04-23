# DLPNO-CCSD Algorithmic Parity with Psi4 — Handoff Prompt

## Goal

Close the ~9× wall-time gap between our PySCF DLPNO-CCSD and Psi4's
DLPNO-CCSD on water10 / cc-pVDZ. Target: bring our 138s CCSD down to
~15-30s, matching Psi4.

## Measured gap (source of truth)

Psi4 timings: `/home/ec2-user/Work/3d_tmcs/orca_benchmarks/t1_dlpno_ccsd_t_si/watercluster_timings.csv`

| System | Psi4 CCSD | Our CCSD | Gap | Our (T) | Psi4 (T) | (T) gap |
|---|---|---|---|---|---|---|
| Water-4 | 3.4s | 23s | 6.8× | — | — | — |
| **Water-10** | **15.1s** | **138s** | **9.1×** | 35s | 17.6s | 2.0× |

**Key signal: the (T) gap is 2× but CCSD is 9×.** (T) received the
algorithmic port work (see `HANDOFF_TRIPLES_FLOPS_RESTRUCTURE.md`);
CCSD hasn't. The same kind of work should close the CCSD gap.

## Why this is an algorithmic problem, not overhead

Phase 7 (overhead reduction) was exhausted in the previous session:

- **7a** (commit `661262a95`): parallel S_pno prebuild + cache miss
  writeback + FlatTensorStore view cache → saved **27s on water10 CCSD**.
- **7b**: G-term batched across pairs → saved **2.6s**.
- **7c**: tried Cython kernel for G (both hand-rolled and direct dgemm).
  Both regressed vs numpy batched. Cython can't beat BLAS on pure
  dgemm chains at n_pno ~25.

Instrumented `compute_CD_terms_batched` + `compute_C_tilde_batched`
after 7b shows **only ~10s of Python-side overhead remains across all
jiang-phase functions**. Cython-porting those would save 7-8% max.
The remaining ~95s gap on water10 is real algorithmic work — things
we're computing that Psi4 isn't, or computing more expensively.

## First concrete step: term-by-term comparison

Profile Psi4's DLPNO-CCSD on water10 / cc-pVDZ and compare
phase-by-phase with our code. Questions to answer:

1. **Per-pair work**: what's Psi4's average n_pno vs ours? If they're
   smaller, why? (PNO truncation, T_CUT_PNO calibration)
2. **Pair count**: how many strong pairs does Psi4 keep vs our 269?
   Tighter prescreening (T_CUT_PAIRS, T_CUT_DO) could be the win.
3. **Integral path**: does Psi4 stay fully atom-local for every
   residual term? Our `local_df.py` is partially ported — profile
   which terms still hit global-DF paths.
4. **Per-cycle overhead**: does Psi4 recompute any dressed integrals
   per cycle? We rebuild C_tilde/D_tilde/G_tilde each cycle. If Psi4
   reuses some across cycles, that's a major structural diff.
5. **Residual formulation**: is Psi4 using a different form of the
   residual equations that sums fewer terms?

Psi4 source is at `/environments/psi4_jiang/psi4/src/psi4/dlpno/`:
- `dlpnobase.cc` — orbital setup, DF integrals, PNO generation
- `ccsd.cc` — main CCSD iteration, dressed intermediates, residual

Ours:
- `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/lccsd.py` — driver
- `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/residual.py` — residual + dressed ints
- `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/local_df.py` — DF integrals

## In-progress work to leverage

Per auto-memory, two active algorithmic efforts should feed directly
into this:

1. **Psi4 X_pno migration** (`project_dlpno_xpno_migration.md`):
   4-phase plan to switch from `C_pno (nao, npno)` to Psi4's
   `X_pno (pair_paos, npno)` layout. Reduces memory and matches Psi4
   exactly. Sparse qij/qia/qab builders already validated at machine
   precision.

2. **Local DF refactor** (`project_local_df_refactor.md`): per-pair
   local DF to match Psi4; `K̃ + B_tilde` done; `vvL (Qab)` per pair
   still needed.

Finishing these is likely the first concrete chunk of algorithmic
parity. Don't duplicate — continue them.

## Test setup

**Benchmark system**: water10 / cc-pVDZ with Jiang TightPNO thresholds.

**Profile script** (ready to go): `/tmp/profile_dlpno.py` runs one
water10 DLPNO-CCSD(T) with cProfile. Expected runtime ~5 min.

**Thread config** (already in the script and driver): `OMP=OPENBLAS=MKL=1`
at env level, `ThreadPoolExecutor` with 64 workers, 1-thread BLAS per
worker. On 64-physical-core EC2.

**Current baseline** (post-Phase 7b, committed): **235s total,
138s CCSD, 35s (T)**. `E_tccsd = -2.13083454540527` is the correctness
reference (matches prior 13 digits).

## Build sync caveat (critical)

Dev source is at `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/`.
Installed package is at
`/environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/`.
**Edits must be `cp`-ed to the installed location** before running the
profile script. `.so` files live at both locations and are separate
copies, not symlinks.

Cython extension rebuild:
```bash
cd /home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd
python setup.py build_ext --inplace
cp _g_term_cy.cpython-312-x86_64-linux-gnu.so \
   /environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/
```

## Correctness anchor

Every change must preserve `E_tccsd` to at least 10 digits on water10.
Reference: **-2.1308345454** (full: -2.13083454540527).

## What NOT to spend time on

- More Python overhead reduction / view caching — exhausted.
- Cython kernel ports of pure dgemm chains — proved unproductive in 7c.
- Micro-optimizing `np.add.at` or `np.ascontiguousarray` — small gains.

The remaining gap is **algorithmic, not implementational**. Spend time
comparing to Psi4 and matching its algorithmic choices, not micro-
tuning ours.

## First action when you pick this up

1. Read `HANDOFF_TRIPLES_FLOPS_RESTRUCTURE.md` in the same directory —
   analogous work already done for (T). Same pattern will apply here.
2. Run the profile script on water10 to confirm current baseline
   (~138s CCSD, E = -2.1308345454).
3. Run Psi4 on the same water10 geometry with matched thresholds to
   get an apples-to-apples profile to compare against.
4. Pick ONE concrete algorithmic difference (likely from questions 1-5
   above) and port it. Measure. Repeat.

Work in small commits keyed to specific algorithmic changes, each
with a before/after CCSD wall number on water10 and E_tccsd
correctness check.
