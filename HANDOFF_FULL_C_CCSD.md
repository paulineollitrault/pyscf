# DLPNO-CCSD: full-C cycle port (Psi4-equivalence target)

**Goal:** match Psi4's DLPNO-CCSD performance and code structure by porting the entire CCSD iteration body into native C in `pyscf/lib/cc/dlpno_*.c`. Python becomes a thin driver: builds `cc_ints`, calls `DLPNOccsd_cycle()` per iteration, handles DIIS + convergence + energy print.

**Why:** the Psi4-style Python ports we've already done (`compute_C_tilde_psi4`, `build_D_tilde_psi4`, `build_G_tilde_psi4`) are 2× slower than the batched plan-cached implementations and 3-5× slower than Psi4. The numpy per-call overhead (~1-3 μs/op × thousands of ops/iteration) is unrecoverable in pure Python. The natural end state of "match Psi4 layout AND speed AND structure" is "be Psi4 in C with PySCF I/O conventions". Phase III storage layout (cc_ints on `pair_lmo_idx`-axis) and Phases I+II (cross-code dumps + first 3 C kernels) are the prerequisites — they're done.

**Scope:** ~1500-3000 lines of new C in `pyscf/lib/cc/`. 13-18 sessions of careful work. High-risk for silent FP bugs, mitigated by extensive per-pair dump validation.

This doc supersedes `HANDOFF_PSI4_PORT_ARC.md` Phase II/III sections. The arc plan was the right intermediate step but the per-pair-Python middle ground turned out too slow. We're committing to native C for the residual hot path.

## Architecture

### Data layout

All per-pair tensors are stored as **flat double arrays + offset tables**, exactly matching the existing `FlatTensorStore` pattern in `pair_index.py`. Python builds these once during cc_ints construction; they're handed to C as raw pointers.

Per-pair static (built once, lifetime = full CCSD run):

| C-side name | Python source | Shape per pair | Notes |
|---|---|---|---|
| `Qab_flat`, `Qab_off` | `cc_ints[k]['Qab']` | `(n_local, n_pno, n_pno)` | local-aux × PNO × PNO |
| `Qma_flat`, `Qma_off` | `cc_ints[k]['Qma']` | `(n_local, nlmo_pair, n_pno)` | post-Phase III pair_lmo_idx |
| `i_Qa_flat`, `i_Qa_off` | `cc_ints[k]['i_Qa']` | `(n_local, n_pno)` | i-side virtual |
| `j_Qa_flat`, `j_Qa_off` | `cc_ints[k]['j_Qa']` | same | j-side virtual |
| `i_Qk_flat`, `i_Qk_off` | `cc_ints[k]['i_Qk']` | `(n_local, nlmo_pair)` | post-Phase III |
| `j_Qk_flat`, `j_Qk_off` | `cc_ints[k]['j_Qk']` | same | |
| `K_iajb_flat`, `K_iajb_off` | `cc_ints[k]['K_iajb']` | `(n_pno, n_pno)` | bare exchange |
| `K_bar_chem_flat`, ... | `cc_ints[k]['K_bar_chem']` | `(nlmo_pair, n_pno)` | post-Phase III |
| `K_bar_ij_flat`, `K_bar_ji_flat` | `cc_ints[k]['K_bar_ij/ji']` | same | |
| `K_tilde_chem_i/j_flat` | `cc_ints[k]['K_tilde_chem_i/j']` | `(n_pno, n_pno²)` | |
| `J_ijab_flat`, `J_ijab_off` | `cc_ints[k]['J_ijab']` | `(n_pno, n_pno)` | bare Coulomb |
| `e_pno_flat`, `e_pno_off` | `pno_spaces[k]['e_pno']` | `(n_pno,)` | semicanonical eigvals |

Per-pair-pair (S_PNO):
- `S_pno_flat` — concatenated `(n_pno_a × n_pno_b)` blocks, ordered by `(pair_idx_a × n_pairs + pair_idx_b)`.
- `S_pno_off[n_pairs² + 1]` — offset table; -1 for missing entries.

Per-pair amplitudes (in/out, lifetime = one cycle):
- `t1_flat[]`, `t1_off[n_pairs+1]` — at canonical pair (i, i): t1[i] in PNO_ii basis.
- `t2_flat[]`, `t2_off[n_pairs+1]` — t2 at canonical key.

Domain metadata:
- `n_pno_arr[n_pairs]`, `nlmo_pair_arr[n_pairs]`, `naux_pair_arr[n_pairs]`
- `pair_lmos_flat[]`, `pair_lmos_off[n_pairs+1]` — global LMO indices per pair
- `pair_lmos_dense_flat[n_pairs * nocc]` — inverse map per pair
- `i_j_to_pair_idx[nocc * nocc]` — Psi4's `i_j_to_ij_`; -1 for missing pairs
- `pair_to_canonical_idx[n_pairs]` — for ordered → canonical lookup
- `pair_swap[n_pairs]` — 1 if ordered != canonical, 0 otherwise

### Function decomposition

One C source file per Psi4 function, matching ccsd.cc structure:

| Psi4 function | New C file | Lines (est) | Status |
|---|---|---|---|
| `t1_ints` (ccsd.cc:1491) | `dlpno_t1_ints.c` | 100 | DONE (session 3, 2026-04-27) |
| `t1_fock` (ccsd.cc:1540) | `dlpno_t1_fock.c` | 200 | TODO |
| `compute_B_tilde` (ccsd.cc:1688) | `dlpno_b_tilde.c` | 50 | DONE (session 1, 2026-04-27) |
| `compute_C_tilde` (ccsd.cc:1809) | `dlpno_c_tilde.c` | 100 | TODO |
| `compute_D_tilde` (ccsd.cc:1991) | `dlpno_d_tilde.c` | 100 | TODO |
| `compute_G_tilde` (ccsd.cc:2085) | `dlpno_g_tilde.c` | 50 | DONE (session 2, 2026-04-27) |
| T1 residual (ccsd.cc:2073-2230) | `dlpno_t1_residual.c` | 200 | TODO |
| T2 residual (ccsd.cc:2240-2500) | `dlpno_t2_residual.c` | 400 | TODO |
| Cycle integration | `dlpno_ccsd_cycle.c` | 100 | TODO |
| **Total** | | **~1300 lines** | |

Existing Phase II C kernels (`dlpno_foo_dressed.c`, `dlpno_per_i.c`, `dlpno_partner.c`) are templates / examples of the pattern.

### BLAS in C

Each C file uses BLAS via `cblas` or `extern dgemm_` (see `pyscf/lib/cc/ccsd_t.c` for the pattern in pyscf). Small matrices (~25×25) — BLAS dispatch overhead per call ~1-2 μs in C (vs ~3-5 μs from Python).

### Threading

Outer parallelism: `#pragma omp parallel for schedule(dynamic, 1)` over pair index `ij` in `[0, n_pairs)`. Same as Psi4. NO Python-level pool — the C function is called once per CCSD iteration.

BLAS threads: set to 1 inside the parallel region (each OMP thread gets a serial BLAS). Same pattern as Psi4.

### Call from Python

```python
from pyscf import lib as _pyscflib
import ctypes

_libcc = _pyscflib.load_library('libcc')
_libcc.DLPNOccsd_cycle.argtypes = [
    ctypes.c_int,                              # n_pairs
    ctypes.c_int,                              # nocc
    # ... (many flat-buffer ptrs and offsets)
]
_libcc.DLPNOccsd_cycle.restype = ctypes.c_double  # E_corr

def lccsd_iterate(...):
    # ... build cc_ints, flat buffers, S_pno table
    while not converged:
        e_corr = _libcc.DLPNOccsd_cycle(
            n_pairs, nocc,
            t1_flat.ctypes.data_as(ctypes.c_void_p),
            t2_flat.ctypes.data_as(ctypes.c_void_p),
            r1_flat.ctypes.data_as(ctypes.c_void_p),
            r2_flat.ctypes.data_as(ctypes.c_void_p),
            # ... many static buffer ptrs
        )
        # Python handles DIIS:
        t1_flat, t2_flat = diis.update(t1_flat - r1_flat / d1,
                                       t2_flat - r2_flat / d2)
```

## Validation plan

The Phase I dump infrastructure is **load-bearing** for this port. Every C function ported gets validated by:

1. **Per-pair dump diff vs Psi4** using the existing `*_DUMP` env toggles. For each major intermediate (B_tilde, C_tilde, D_tilde, G_tilde, Fkj, Fab, t1_ints output, T1/T2 residuals): emit per-pair Frobenius/trace/sum + matrix data, diff against Psi4's matching dumps.

2. **Per-pair dump diff vs the existing Psi4-style Python** (`compute_C_tilde_psi4` etc.) — validates that our C port matches our Python interpretation of the same algorithm.

3. **Energy anchor** at every step: `water-4 E_TCCSD(T) = -304.98979787`, `water-10 E_TCCSD = -2.13088299002 (11 digits)`. Must hold (or drift only by FP-reorder noise) after each function port.

4. **Dual-build runs** during the transition: env `DLPNO_C_CYCLE=1` activates the C path; default stays Python (batched or Psi4-style via the existing toggles). Diff energies between the two paths.

## Session-by-session roadmap

Each session: one C function, pattern matches the foo_dressed / per_i / partner template established in Phase II.

| Session | Function | Validation |
|---|---|---|
| 1 (DONE 2026-04-27) | `dlpno_b_tilde.c` (~100 lines incl. comments) | water-4 −304.98979787 ✓, water-10 −2.13088299002 ✓, BTILDE_DUMP py-vs-C max abs 1.1e-11 over 306 dumps |
| 2 (DONE 2026-04-27) | `dlpno_g_tilde.c` (~75 lines incl. comments) | water-4 −304.98979787 ✓, water-10 −2.13088299002 ✓, GTILDE_DUMP py-vs-C diff: last-digit FP-reorder noise across 3 iters |
| 3 (DONE 2026-04-27) | `dlpno_t1_ints.c` (~85 lines incl. comments) | water-4 −304.98979787 ✓, water-10 −2.13088299002 ✓, T1INTS_DUMP py-vs-C aggregates match to last 1-2 digits |
| 4 | `dlpno_c_tilde.c` (~100 lines) | water-4/10 anchor, CTILDE_DUMP diff |
| 5 | `dlpno_d_tilde.c` (~100 lines) | water-4/10 anchor, DTILDE_DUMP diff |
| 6-7 | `dlpno_t1_fock.c` (~200 lines) | water-4/10 anchor, FKJ_DUMP diff |
| 8-9 | `dlpno_t1_residual.c` (~200 lines) | water-4/10 anchor, R1 dumps |
| 10-12 | `dlpno_t2_residual.c` (~400 lines, the big one) | water-4/10 anchor, R2 dumps |
| 13-14 | `dlpno_ccsd_cycle.c` integration: glue function calling all the above; replaces the Python while-loop body | water-4/10/22 anchors, perf vs Psi4 |
| 15+ | Cleanup: delete legacy batched Python paths, delete Cython kernels superseded by C, finalize Python driver to ~200 lines | water-22 perf measurement |

## Key code references

Same algorithm location in our codebase + Psi4 (memorize these):

| Function | Psi4 (read-only ref) | Our Python | New C target |
|---|---|---|---|
| compute_B_tilde | `ccsd.cc:1688` | `local_df.py:1273` | `dlpno_b_tilde.c` |
| compute_C_tilde | `ccsd.cc:1809` | `residual.py:compute_C_tilde_psi4` (already ported) | `dlpno_c_tilde.c` |
| compute_D_tilde | `ccsd.cc:1991` | `residual.py:build_D_tilde_psi4` (already ported) | `dlpno_d_tilde.c` |
| compute_G_tilde | `ccsd.cc:2085` | `residual.py:build_G_tilde_psi4` (Psi4-faithful) / `residual.py:build_G_tilde` (plan-cached, default; native-C path now lives here under DLPNO_C_CYCLE) | `dlpno_g_tilde.c` |
| t1_ints | `ccsd.cc:1491` | `local_df.py:t1_ints` | `dlpno_t1_ints.c` |
| t1_fock | `ccsd.cc:1540` | `local_df.py:t1_fock` | `dlpno_t1_fock.c` |
| T1 residual | `ccsd.cc:2073` | `lccsd.py:_compute_t1_residual_psi4` | `dlpno_t1_residual.c` |
| T2 residual | `ccsd.cc:2240` | `lccsd.py:` (compute_residual_v2 driven from here) | `dlpno_t2_residual.c` |

## What we DON'T port to C

- **`cc_ints` build (`compute_cc_integrals_sparse`)** — already runs once per CCSD run, not in the hot iteration loop. Stays Python.
- **DIIS** — small CPU footprint, stays Python.
- **Energy convergence check** — small, stays Python.
- **PNO truncation, MP2 init, etc.** — pre-CCSD work, stays Python.
- **(T) triples** — separate handoff (`HANDOFF_TRIPLES_FLOPS_RESTRUCTURE.md`).
- **Localization** — pre-CCSD, stays Python (calls PySCF localizers).

## Anchors

- water-4 cc-pVDZ TightPNO PM: `E_TCCSD(T) = -304.98979787`
- water-10 cc-pVDZ TightPNO PM: `E_TCCSD = -2.13088299002` (11 digits)

These hold to FP-reorder noise (~10 μEh / system) at every checkpoint.

## What's already in the tree (helpful to the port)

- **Phase II C kernel pattern**: `pyscf/lib/cc/dlpno_foo_dressed.c`, `dlpno_per_i.c`, `dlpno_partner.c` (commits `95534950e`, `55686530b`, `6eecd5bad`).
- **Phase III storage**: cc_ints on pair_lmo_idx-axis (commit `87cf87c22`).
- **Phase I dumps**: env toggles `DLPNO_DUMP_BTILDE/LADDER/FKJ/GTILDE/T1INTS/CTILDE/DTILDE` on both sides.
- **Psi4-style Python references**: `compute_C_tilde_psi4`, `build_D_tilde_psi4`, `build_G_tilde_psi4` in `residual.py`. These are **DELETED** in the cleanup phase (session 15+) once the C versions land — until then they're useful as line-by-line cross-checks for the C ports.
- **CMake build**: `pyscf/lib/cc/CMakeLists.txt` already has the Phase II C kernels. Each new file goes in the `add_library(cc SHARED ...)` line. Build with `make cc -j8` from `pyscf/lib/build`.
- **Build flag note**: cmake auto-detects the wrong libopenblas. Use:
  ```
  cmake .. -DBLAS_LIBRARIES=/environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/lib/libopenblas-r0-f650aae0.3.3.so
  ```

## Stretch / out-of-scope

- **Match Psi4's algorithm exactly** including LMO localization and PNO basis. We currently use PySCF localizers; Psi4 uses its own. The 0.6 mEh CCSD gap to Psi4 is from this, NOT from algorithm differences (Phase I confirmed). Closing it is a separate handoff (`project_dlpno_xpno_migration.md`).
- **Replace `cc_ints` build with C** — possible later for memory wins, not on the perf-critical path.
- **GPU port** — far future.
