# Handoff: DLPNO-CCSD class BE-plan native B_tilde refresh

**Date:** 2026-04-30
**Goal:** Fix water-10 1.6 mEh drift in C++ DLPNOCCSDSolver class drop-in mode (`DLPNO_CCSD_MONO_DROPIN_CYCLE=1`)
**Target perf:** Psi4 reference (water-10 CCSD 15.1s on 16-core Xeon 6136); current class 89s, current Python+Cython baseline 63s

## Current state at checkpoint

| | Baseline (PySCF+Cython) | Drop-in (C++ class) | Psi4 |
|---|---|---|---|
| water-4 E_corr | -0.85650131 | -0.85650718 (✓ 5.7e-6 drift) | — |
| water-4 CCSD | 9.0s | 22.5s | 3.4s |
| water-10 E_corr | -2.14141403 | **-2.13977987 (✗ 1.6 mEh drift)** | — |
| water-10 CCSD | 63.4s | 89.2s | 15.1s |

## Root cause analysis (confirmed)

The drop-in driver `run_remaining_cycles_via_class` in [_ccsd_solver.py:5533+](Work/pyscf/pyscf/cc/dlpno_tccsd/_ccsd_solver.py#L5533) reuses cycle-0 PySCF plan caches across all cycles (pack-once architecture).

**The CD plan correctly handles per-cycle refresh** via the native gather mechanism — `c_term_ct_ord_pair_idx` / `d_term_dt_ord_pair_idx` (cpp:805-810) tells the class to rebuild `ct_flat` / `dt_flat` from its own `C_tilde_flat` / `D_tilde_flat` (built fresh in Phase 6 from current T1).

**The BE plan does NOT have an analogous mechanism.** Each BE bucket stores a flat `beta_kl` / `beta_lk` array of doubles (lengths N_b each). At pack time, these are populated from `b_tilde_per_ij[(i,j)][p_dense[k], p_dense[l]]` — the cycle-0 B_tilde dict from PySCF.

The class DOES build its own `B_tilde_flat` natively each cycle in Phase 5 ([dlpno_ccsd_solver.cpp:1217-1227](Work/pyscf/pyscf/lib/cc/dlpno_ccsd_solver.cpp#L1217)) but never uses it to refresh the BE bucket beta_kl/lk arrays — those stay frozen at cycle-0 values.

Why water-4 (5.7e-6) works but water-10 (1.6 mEh) breaks:
- water-4 has 76 weak pairs and 72 strong pairs; small staleness effect
- water-10 has 377 weak pairs and 175 strong pairs; staleness effect ~280× larger
- The BE residual contribution sums over `kl` neighbors of each strong `ij`, with `kl` spanning all pairs

Confirming evidence:
- Disabling skip-weak guards diverges (cycles to -2.22 by cycle 14, dE=3.3e-3 still climbing) — confirms skip-weak is NOT the bug
- Python refresh of `b_tilde_per_ij` dict (so beta_kl/lk re-extracted each cycle from fresh B_tilde) made water-4 WORSE (drift 5.7e-6 → 6.6e-4), suggesting the Python `compute_B_tilde` and class's `run_phase_b_tilde_into` produce subtly different values; only-class-native is consistent

## Fix design (recommended)

Mirror the CD plan's native-gather pattern for BE:

### Python side ([_ccsd_solver.py:5267+](Work/pyscf/pyscf/cc/dlpno_tccsd/_ccsd_solver.py#L5267))
For each bucket entry `n`, store three additional int32 arrays:
```python
bucket_p_ij[n]       = key_to_p[key_ij_n]
bucket_dense_k[n]    = p_dense[k_n]   # B_tilde row index for pair p_ij
bucket_dense_l[n]    = p_dense[l_n]   # B_tilde col index
```
Wire pointers into `BEInputs` struct: `p_ij_arr`, `dense_k_arr`, `dense_l_arr`.

### C++ side ([dlpno_ccsd_solver.cpp:472-498](Work/pyscf/pyscf/lib/cc/dlpno_ccsd_solver.cpp#L472))
1. Extend `BEInputs`:
   - Change `const double *beta_kl, *beta_lk` to `double *beta_kl_mut, *beta_lk_mut` (or add aliases) so Phase 5 can write to them
   - Add `const int *p_ij_arr, *dense_k_arr, *dense_l_arr`
2. In `run_one_cycle` after Phase 5 (line 1227), before BE step (line 1514):
   ```cpp
   // Refresh BE buckets' beta_kl/lk from native B_tilde_flat.
   for (int b = 0; b < plans->be_n_buckets; ++b) {
       BEInputs *bucket = ...;  // (cast away const for refresh)
       for (int n = 0; n < bucket->N; ++n) {
           const int p = bucket->p_ij_arr[n];
           const int dk = bucket->dense_k_arr[n];
           const int dl = bucket->dense_l_arr[n];
           const int npno = npno_arr[p];
           const int64_t off = b_tilde_off[p];
           bucket->beta_kl_mut[n] = B_tilde_flat[off + dk*npno + dl];
           bucket->beta_lk_mut[n] = (dk == dl) ? 0.0 :
                                    B_tilde_flat[off + dl*npno + dk];
       }
   }
   ```

### Validation
1. After implementation, run [_test_water10_perf.py](Work/pyscf/pyscf/cc/dlpno_tccsd/_test_water10_perf.py) with `DLPNO_CCSD_MONO_DROPIN_CYCLE=1`. Target E_corr = -2.14141403 ± 1e-5.
2. water-4 should still match.
3. If energy matches, the 1.6 mEh drift is closed and the perf gap (3.5× per cycle) becomes the next focus.

## After accuracy: perf gap

Class run_cyc=4.5s/cycle, Python+Cython baseline=1.3s/cycle on water-10.
Per-phase breakdown not yet available for class (no per-phase timers inside `run_one_cycle`). Next step: instrument C++ phases.

The Python+Cython baseline got faster via:
- 12 hot-path C kernels (B_tilde, G_tilde, t1_ints, C_tilde-ph1+ph2, D_tilde-ph1+ph2, t1_fock, t1_residual, be_kernel, g_term_batched, c+d term batched, t3+t4 batched)
- Cython per-pair foo_dressed (3.3× on that phase)
- cc_ints C ports (3.3× over Cython baseline cumulative)

The class re-uses many of these same kernels. The 3.5× gap suggests the class's orchestration overhead dominates: scratch allocation, per-phase dispatch, OMP setup. Profile carefully before optimizing.

## Files

- [_ccsd_solver.py](Work/pyscf/pyscf/cc/dlpno_tccsd/_ccsd_solver.py) — drop-in driver + plan extractors
- [dlpno_ccsd_solver.cpp](Work/pyscf/pyscf/lib/cc/dlpno_ccsd_solver.cpp) — DLPNOCCSDSolver class
- [_test_water10_perf.py](Work/pyscf/pyscf/cc/dlpno_tccsd/_test_water10_perf.py) — perf test driver
- [_test_water4_perf.py](Work/pyscf/pyscf/cc/dlpno_tccsd/_test_water4_perf.py) — faster smoke test

Run with `DLPNO_CCSD_MONO_DROPIN_CYCLE=1` to enable class drop-in.
