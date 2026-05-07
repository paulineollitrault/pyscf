# DLPNO-CCSD: monolithic C++ class to match Psi4

## Mission

Restructure DLPNO-CCSD to mirror Psi4's `compute_lccsd` architecture: **one C++ class owns all CCSD state, runs the entire cycle loop in C++, exposes a single Python entry point**. Eliminate the Python/C ping-pong that adds latency to every cycle phase.

This sets up a longer-term goal: closing the gap to Psi4's 15.1s on water-10 (we're at 38s).

## Why piecemeal kernel ports plateaued

Across 13+ sessions we ported every hot-path DLPNO-CCSD per-pair kernel to C (B_tilde, G_tilde, C_tilde, D_tilde, t1_ints, t1_fock, t1_residual, be, cd, gterm, etc.) under `DLPNO_C_CYCLE=1`. Water-10 dropped from 60s → 38s. Then it stalled.

Per-cycle steady-state (water-10, post-port):
- pairs: 0.46s
- jiang: 0.36s (C 0.13, D 0.14, Fock 0.05, Km 0.02, G 0.02)
- t1r: 0.23s
- foo: 0.04s
- **total: 1.1s/cycle**

Psi4 estimated steady cycle: ~0.7s. Our gap per cycle: 0.4s × 13 cycles = **5s** of the remaining 23s gap.

The OTHER 18s gap lives in **pre-iter setup (16s) + cycle 1 plan-build inflation (8s above steady)**. Our cycle 1 = 8.1s vs estimated Psi4 ~1s.

### Last attempt (Session B, 2026-04-28) — what we tried, what we learned

**Goal:** port G_tilde plan-build kproj math to C (`pyscf/lib/cc/dlpno_g_tilde_plan.c::DLPNOcompute_kproj_batched`).

**Profile breakdown:** cycle 1 G_tilde was 2.61s. Of that:
- Per-triple `S^T @ K_il @ S` numpy/BLAS dispatch: 0.3s (the kernel-portable matmul)
- **Lazy `compute_S_pno` for 44922 (canon_il, canon_lj) pairs not in upfront flat tier: ~2.3s**

**Result:** C kernel works. Cycle 1 G dropped from 2.61s → 1.9s. But the savings are bounded by `compute_S_pno`, not the matmul. Net cycle 1 saving: ~0.7s. Far short of the 5-8s target.

**Why it plateaus:** every Python wrapper for a C kernel (`build_G_tilde`, `compute_C_tilde_batched`, etc.) does its own setup → ctypes call → unpack. Each one is fast individually, but the Python orchestration between phases adds up. AND the cycle driver itself (DIIS, residual orchestration, T1/T2 update) lives in Python.

**Files left in place:** `pyscf/lib/cc/dlpno_g_tilde_plan.c`, modified `pyscf/cc/dlpno_tccsd/residual.py::build_G_tilde` plan-build path. Energy converges (within 1e-5 Eh of anchor — small drift due to summation ordering differences in side-S compute, NOT a bug). Can revert via `git checkout` if undesired for the new approach.

## The architecture this session should build

Mirror `pyscf/lib/cc/dlpno_triples_orch.c::DLPNOcompute_E_T0_omp` (the (T) port that matched Psi4 perf). For (T) we have ONE C entry point that owns all per-triple state and orchestrates Phase A precompute + Phase B per-triple work.

For CCSD, we need:

```cpp
// pyscf/lib/cc/dlpno_ccsd_solver.cpp (NEW)
class DLPNOCCSDSolver {
    // Owned state — populated by Python before .solve()
    int nocc, nlmo, n_canon_pairs;
    double *F_lmo, *foo, *fov, *fvv;
    FlatTensorStore cc_ints;       // K_iajb, Qma, Qab, K_bar, ...
    FlatPairPairStore S_pno_cache; // pre-built upfront
    double *T1, *T2;               // current amplitudes (in-place updates)
    DIISState diis;                // C++ DIIS
    PlanCache plans;               // per-phase plan-build state, built once

    // Per-cycle phases (called internally; no Python on inner path)
    void phase_t1_ints();
    void phase_jiang_C_tilde();
    void phase_jiang_D_tilde();
    void phase_jiang_G_tilde();
    void phase_jiang_Fock();
    void phase_pairs_be();
    void phase_pairs_cd();
    void phase_pairs_gterm();
    void phase_pairs_update();
    void phase_t1_residual();
    void apply_diis();

public:
    void solve(double e_conv, double r_conv, int max_cycle);
    double get_energy() const;
    // ...
};

// Single C entry point
extern "C" void DLPNOcompute_lccsd_omp(/* state pointers */);
```

Python side becomes ~50 lines:
1. Build initial T1/T2 (same as today).
2. Pack all state into the C-callable layout (already mostly flat).
3. ONE ctypes call to `DLPNOcompute_lccsd_omp`.
4. Read converged T1/T2/energy back.

## Concrete first steps

1. **Read Psi4's `compute_lccsd`** in `psi4_jiang/install/...` — study exactly what fields the C++ object holds and the order of operations per cycle. Mirror that structure.

2. **Inventory existing state**: `pyscf/cc/dlpno_tccsd/lccsd.py:1500-2700` is the cycle driver. Catalogue every variable it touches (T1, T2, foo, fov, F_lmo, eps_lmo, cc_ints, S_pno_cache, K_pno_cache, t2_pno_all, pno_spaces, pair_lmo_idx, T1_cache, ...). This becomes the C++ class's member set.

3. **Plan caches**: each `compute_*_batched` builds a plan once and caches it. List every plan, its inputs, when it gets invalidated. Move plan-build into the C++ constructor (one-time pre-iter cost).

4. **DIIS in C++**: ~200 lines of LAPACK `dgelsy` lstsq. Already proved feasible in `dlpno_triples_full.c` lapack_helper pattern.

5. **Phase ordering**: every cycle does (1) t1_ints → (2) foo_dressed → (3) jiang.{C,D,G,Fock,Km} → (4) pairs.{be,cd,upd,bt,gterm} → (5) t1_residual → (6) DIIS → (7) energy. Each becomes a method on the class.

6. **Lazy S_pno_cache pre-warming** (the Session B finding): the upfront flat tier covers ~122K of ~600K possible (canon × canon) pairs. The remaining 44K-78K pairs get lazy-computed on first cycle. **For G_tilde specifically, ~70% of the (i, j, l) triples need lazy S.** Move this lazy compute into the upfront S build. Use `DLPNObuild_S_pno_for_pair` on canon × canon to bring the missing entries into flat tier. Memory cost: ~30-50 MB extra. Wall savings: ~2.4s in cycle 1.

7. **First milestone**: a working `DLPNOcompute_lccsd_omp` that runs cycles 2+ in C (matches steady-state perf). Cycle 1 plan-build can stay in Python initially — port last.

## Key validation

- Water-10 anchor: `E_TCCSD = -2.13083960807753` (10 digits). Bit-perfect within FP noise.
- Test set: water-4 (small, fast iteration) → water-10 (full target) → S22-{water cluster, methane dimer} (validate scaling).
- Per-phase cross-validation: dump intermediates from old Python path vs new C++ path, compare numerically.

## What NOT to do

- Don't optimize the per-pair inner C kernels further. They're already at 1-2s/cycle for water-10 (FLOPS-bound). The win is **architectural**: removing Python orchestration overhead.
- Don't try to "match Psi4 algorithm" — focus on architecture. We previously validated all per-pair intermediates are structurally Psi4-faithful (Phase I done, see memory).
- Don't add Cython. The user explicitly wants C/C++ only.

## Existing memory worth reading

- `project_triples_phase3c.md` — the (T) port that DID work. Same pattern.
- `project_full_c_ccsd_session1.md` — 12 per-pair C kernels already exist; the new class delegates to them.
- `project_psi4_port_phase3_done.md` — cc_ints storage axis already Psi4-faithful.
- `project_ccsd_g_tilde_session_b.md` — what failed and why.

## Open question to discuss with the user before writing code

The user's exact framing: "simplify the C++ kernels, minimize the hybrid python/C++ overhead and match closer psi4's implementation."

Confirm with them:
1. Are existing per-pair kernels (`dlpno_be.c`, `dlpno_g_tilde.c`, etc.) intended to remain as-is and be CALLED FROM the new monolithic class? Or rewritten/folded into class methods?
2. Is `DLPNO_C_CYCLE=1` intended to stay as the default, or is the new architecture replacing it?
3. Memory layout: should the new class own its own copies of T1/T2/cc_ints, or take pointers into Python-allocated buffers?
