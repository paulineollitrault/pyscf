# DLPNO-CCSD single-C-kernel arc — close the gap to Psi4-OMP

## Target

| Phase | Current | Psi4-OMP target | Gap |
|---|---|---|---|
| CCSD wall (water-10) | 38s | **15.1s** | 2.5× slower |
| (T) wall (water-10)  | 15.6s | 17.6s | at parity |
| Total | ~53s | 32.7s | -20s |

Per-cycle PySCF breakdown (water-10, steady state, 32 pool workers):
- pairs: 0.46s (be 0.13, cd 0.12, upd 0.14, bt 0.06, gterm 0.02)
- jiang: 0.36s (C 0.13, D 0.14, Fock 0.05, Km 0.02, G 0.02)
- t1r: 0.23s
- foo: 0.04s
- TOTAL: 1.1s/cycle

Stage 5 = 37.32s breakdown:
- Local DF integrals (cc_ints): 11.9s
- S_pno_cache + post-cc_ints setup: 3.7s
- Cycle 1 (cache fills): 8.1s
- Cycles 2-14 (steady): 14.3s

## Strategy

Mirror the (T) Phase 3c-4 architecture:
- ONE public C entry point `DLPNOcompute_lccsd_omp` replaces the
  Python `while` loop driver
- Per-cycle phases each become OMP-over-pairs INSIDE the C entry
- Existing per-pair C kernels (`DLPNObe_kernel`, `DLPNOcompute_C_tilde`,
  etc.) get called inside the OMP region — no Python in the inner loop
- Single global per-CCSD-run arena (pair_to_idx + flat data) built
  once before the C call

## Multi-session plan

### Session A (this one — PILOT): OMP-over-i for t1_residual
**Goal**: Prove the approach gives measurable speedup on a single
phase before porting all phases.

**Why t1r**: Biggest single per-cycle phase (0.23s × 13 cycles = 3s).
Existing `_compute_t1_residual_psi4` uses `pool.map(_per_i, range(nocc))`.
Has the most measurable Python overhead (per profile: ~99ms wall is
Python orchestration on 192ms total).

**Deliverable**:
- New C kernel `dlpno_t1_residual_omp.c::DLPNOcompute_t1_residual_omp`
  that wraps existing `DLPNOper_i_stages123` and `DLPNOcompute_t1_residual`
  in `#pragma omp parallel for` over `i in range(nocc)`
- Python side: build per-pair arena once (or reuse existing),
  call new kernel, replace pool.map(_per_i)
- Expected: t1r 0.23s → 0.13s/cycle (eliminate ~50% Python overhead)
- Stage 5 wall: 38s → 36s (modest but proves approach)

### Session B: pairs phases (be, cd, upd, bt, gterm)
Each becomes OMP-over-pairs in C. Largest phase per cycle = 0.46s.
Expected: pairs 0.46 → 0.30s/cycle, save 2s wall.

### Session C: jiang phases (C, D, Fock, Km, G)
Same pattern. Largest = jiang.D at 0.14s.
Expected: jiang 0.36 → 0.20s/cycle, save 2s.

### Session D: cycle driver + DIIS + foo + t1_ints + t1_fock
Move cycle driver itself into C. DIIS becomes a C function. Now ONE
ctypes call per CCSD run.
Expected: cycle 1 cache fill 8.1s → 2s, save 6s.

### Session E: cc_ints DF (pre-iter, 11.9s)
Optimize the per-pair DF integral build. Possibly batch across pairs
sharing aux atoms. Expected: 11.9 → 6s, save 6s.

### Session F: validate + tune
End state targets:
- Stage 5: 38s → 18-22s (3-4× the cycle work, half the setup)
- Total CCSD+(T): 53s → 35-40s (within ~10% of Psi4-OMP)

## Risks / lessons from (T) arc

1. **Per-pair internal mallocs in sub-kernels** — sub-kernels (be,
   cd, etc.) malloc/free internally per call. With 32 OMP threads
   doing this concurrently, allocator contention can hurt scaling.
   May need __thread scratch in each sub-kernel, OR refactor to
   take scratch args. (For triples this only became an issue with
   OMP=32+; we capped at OMP=16 to dodge it.)

2. **OMP thread count tuning** — 64-physical-core / 128-logical box
   has memory-wall + NUMA effects. Best (T) thread count was 8-16.
   For CCSD per-pair work (~210 pairs), 16 threads probably also
   the sweet spot. Use the same `omp_set_num_threads(min(omp_max, 16))`
   pattern from `DLPNOcompute_E_T0_omp`.

3. **Cycle driver**: moving the while loop into C requires DIIS
   in C. Pulay extrapolation is straightforward (~200 lines C with
   LAPACK lstsq), but adds complexity. Could alternatively keep
   cycle driver in Python with one C call per phase.

4. **Validation**: water-10 E_TCCSD anchor = -2.13083960807753 (current
   converged value). Bit-perfect (within FP noise) at every step.

## Out of scope (for now)

- Algorithmic changes (different intermediates, different DLPNO
  formulation). Psi4 may have different choices that give them
  inherent advantage. Stick to architectural optimization.
- Full GPU port. Different effort entirely.
