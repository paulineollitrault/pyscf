# DLPNO-CCSD Psi4 Parity — Handoff v2

Follow-up to `HANDOFF_CCSD_ALGORITHMIC_PARITY.md` (the v1 handoff that
started this session). That session landed four commits on the
`dlpno_restructure` branch; this v2 handoff is where the next session
picks up.

## Where we are now

- Branch: `dlpno_restructure`, last 4 commits:
  - `7ca6b192a` build_G_tilde: _pool kwarg + DLPNO_FINE_POOL_SIZE env var
  - `418993a9d` Batch G_tilde inner j-loop + add sub-phase timers
  - `a4c74919a` T1 residual B+A2 parallelized with per-worker reduction
  - `040878ed3` DLPNO-CCSD water10 perf bundle (T1 parallel + fine_pool +
    dead-code strips + D_tilde Term 1 Psi4-semantics fix)

- **water10 / cc-pVDZ / Jiang TightPNO baseline → now: 140.75s → 126.12s
  CCSD (−10.4%).** E_tccsd = **−2.1308829900** (new reference anchor;
  was −2.1308345454 — drifted −48 μEh to match Psi4's D_tilde full-domain
  sum; revert recipe in the memory note).

- Full change log, per-cycle timing breakdown, CPU-utilization map, and
  revert recipes in `memory/project_dlpno_ccsd_perf_session.md`.

## Target

Jiang et al. paper reports water10 / cc-pVDZ DLPNO-CCSD at **15.1s** on
16 cores Intel Xeon 6136 (TightPNO).
Our 126s on 64 cores is still **~8× gap in total compute**.

## First actions when you pick this up

1. Read `memory/project_dlpno_ccsd_perf_session.md` for context,
   landed changes, and (critically) the "what did NOT work" list — save
   yourself time by not reproducing the 5 regressions already tried.
2. Run the profile script:
   ```bash
   /environments/miniconda3/envs/tmc/bin/python -u /tmp/profile_dlpno.py
   ```
   Confirm baseline: **CCSD = 126s, E_tccsd = −2.1308829900**.
3. Look at the per-cycle timing breakdown in the output:
   ```
   foo=0.29 jiang=X.XX(C=X.XX D=X.XX Fock=X.XX Km=X.XX G=X.XX)
   pairs=X.XX(bt=X.XX be=X.XX cd=X.XX gterm=X.XX upd=X.XX)
   ```
   These are the sub-phase wall timers I added — they tell you which
   phase each lever should target.

## Dominant remaining hot phases

Per-iteration (steady-state, 14 iters total):
- `D = 1.05s` (fine_pool=8 Phase 1 + Cython Phase 2)
- `C = 0.90s` (same)
- `G = 0.53s` SERIAL — see below
- `cd = 0.51s` (Cython OMP kernel; already 64-thread)
- `foo = 0.29s` (fine_pool)
- `Fock = 0.26s` (fine_pool)
- `be = 0.24s` (Cython OMP)
- `upd = 0.22s` (full pool over 269 strong pairs; residual is cheap now)
- `gterm = 0.19s` (full pool + batched matmul)
- `bt = 0.09s` (fine_pool)

## The remaining ~8× gap is CPU utilization, not FLOPs

Our CPU time for CCSD is ~7000 s on 64 cores (~55 cores avg, ~85% eff).
Psi4 on 16 cores at 15s × ~80% eff ≈ 190 CPU-s. Ratio ≈ 35×. We do ~35×
more *total compute* — it's Python dispatch overhead around many small
numpy/Cython calls, not bigger FLOPs. `cProfile` tottime confirms:
`tensordot` 24s, `reshape` 22s, `zeros` 14s, all distributed over CCSD.

Phase CPU breakdown:
- SERIAL (1 CPU): G + Km ≈ 0.6s/iter → 8s pure single-thread
- fine_pool=8: C + D + Fock + foo + bt + be ≈ 2.8s/iter → ~40s at ≤8 CPUs
- 64-CPU capable: cd + gterm + upd ≈ 1s/iter → ~14s at full utilization

So ~48s of the 126s CCSD is running at ≤8 CPUs out of 64. **That is what
htop shows as low-CPU stretches.**

## Biggest lever: Cython nogil port of the Phase 1 per-pair bodies

The fine_pool=8 limit isn't arbitrary: benchmarking showed 8 is the
sweet spot for our current Python bodies — more workers cause GIL
contention that outweighs the parallelism gain. To break past this
needs Python-bound inner code to release the GIL.

Specific functions (in priority order by cum time):
- `residual.py:build_D_tilde_batched._process_ik_t12` (D_tilde Phase 1,
  ~15s cum) — now Psi4-full-domain; the tensordot chain is clean.
- `residual.py:compute_C_tilde_batched._process_ki_terms12` (C Phase 1,
  ~15s cum).
- `residual.py:build_G_tilde` inner (i, l) loop (~11s cum, SERIAL).
- `local_df.py:t1_fock._per_pair`, `lccsd.py:_compute_foo_dressed_local._per_pair`
  (~4s cum each).

Port each to a Cython `nogil prange` over pairs. Pattern: existing
`_be_cy.pyx`, `_cd_cy.pyx`, `_c_tilde_cy.pyx` in the package already
do this for Phase 2 of C/D; use them as templates. These already win;
the Phase 1 bodies are just Python.

Expected win: if we can release the GIL on Phase 1 and let 64-way
parallelism land, each of C/D/Fock/foo drops by roughly 8×/16×. Savings:
- C: 0.90 → 0.1s  ⇒ −11s/run
- D: 1.05 → 0.15s ⇒ −13s/run
- Fock+foo+bt: −5s/run
- Total: potentially **−25-30s** (CCSD 126 → ~100s) — brings us under 4×
  from Psi4.

G_tilde is trickier (shape-ragged inner) but could get a similar
nogil/prange treatment over outer `i`.

## What NOT to retry (already failed this session)

- **Overlap `build_G_tilde` with bt/be/cd via `_pool.submit`** — G holds
  the GIL, starves the fine_pool workers (regressed +4.6s wall).
- **Parallelize `build_G_tilde` outer-i via `_pool.map(..., fine_pool)`**
  — even with the lighter inner batched einsum, Python dict.get +
  np.stack hold the GIL enough to serialize across 8 workers (+2.0s).
- **Inline `compute_CD_terms_batched` back into `_update_pair`'s
  per-pair loop (Lever C)** — Cython `c_kernel/d_kernel` is faster
  per-pair than the Python k-loop it replaced; inlining regressed +11s.
- **Shared T1_cache via dict-view wrapper between driver and
  `compute_C_tilde_batched`** — lookup overhead ate the build-cost
  savings, net zero.
- **Fine_pool size sweep 4/8/16/32** — a naïve sweep has state-leakage
  bugs; with fresh interpreters, 8 is still the best (4 regresses to
  135s, 16 was same as 8, 32 worse).

## Second lever: Psi4 comparison artifact

The v1 handoff asked us to compare term-by-term to Psi4 with matching
thresholds. Psi4 (built at `/environments/psi4_jiang/install/bin/psi4`)
can be launched with the input at `/tmp/psi4_water10.in`. In this
session the Psi4 run got stuck at the SC-LMP2 stage — didn't confirm
whether it's hangs or just slow for the dev build (it's ~1 core in htop,
so it's likely a serial-build issue). Running Psi4 fresh and grabbing
the per-timer output (`set print_timings true`) would let the next
session map our `jiang=2.73 pairs=1.25` to Psi4's equivalent phases.

## Useful artifacts

- `/tmp/profile_dlpno.py` — profile runner; prints per-cycle timers and
  cProfile top.
- `/tmp/profile_step14.log` — last clean step-14 run log (CCSD 126.12s,
  E −2.1308829900, detailed per-phase timings).
- `/tmp/pool_overhead.py`, `/tmp/pool_overhead2.py` — microbench that
  established fine_pool=8 sweet spot. Rerun with different task sizes
  if considering re-tuning.
- `memory/project_dlpno_ccsd_perf_session.md` — full change log,
  per-phase breakdown, D_tilde revert recipe.
- `HANDOFF_CCSD_ALGORITHMIC_PARITY.md` — the v1 handoff (still useful
  for the bigger-picture framing and Psi4 code pointers).

## Build sync caveat (from v1, still applies)

Dev edits in `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/` must be
`cp`'d to the installed package at
`/environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/`.
Cython `.so`s built with:
```
cd /home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd
python setup.py build_ext --inplace
cp _*.cpython-312-x86_64-linux-gnu.so \
   /environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/
```

## Correctness anchor

E_tccsd = **−2.1308829900** on water10 / cc-pVDZ / TightPNO. Preserve to
10+ digits unless the change is an intentional semantic match with Psi4
(document in a commit message + memory note with revert recipe, like
the D_tilde Term 1 change in `040878ed3`).
