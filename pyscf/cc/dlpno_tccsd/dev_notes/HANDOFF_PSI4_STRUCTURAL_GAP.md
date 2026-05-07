# Handoff: Closing the per-cycle gap to Psi4 (structural finding)

**Date:** 2026-04-30
**Current state:** water-10 total wall 91s (3.0× session improvement); per-cycle 1.32s matches PySCF baseline 1.3s. Psi4 reference 32.7s on 16-core Xeon 6136 (parallel build, not the local serial-only build). Remaining gap: 2.78× to Psi4.

## Key structural finding from Psi4 source code

Read `/environments/psi4_jiang/psi4/src/psi4/dlpno/ccsd.cc` and identified the major architectural difference:

### Psi4's R2 residual: ONE monolithic per-pair loop

[ccsd.cc:2417-2606](file:///environments/psi4_jiang/psi4/src/psi4/dlpno/ccsd.cc#L2417):

```cpp
#pragma omp parallel for schedule(dynamic, 1)
for (int ij = 0; ij < n_lmo_pairs; ++ij) {
    if (is_weak_pair || npno_ij == 0) continue;
    
    // ALL THESE BUILT IN ONE LOOP BODY PER ij:
    // - K_ij (Term 1): line 2436   linalg::doublet
    // - A_ij ladder (Term 11): lines 2440-2451   per-Q triplet
    // - B_ij + E_tilde (BE step): lines 2502-2533   per-(k,l) triplet
    // - C_ij (Term 13): lines 2545-2553   per-k triplet
    // - D_ij (Term 14): lines 2566-2584   per-k triplet
    
    R_iajb[ij]->add(K_ij);     R_iajb[ij]->add(A_ij);
    R_iajb[ij]->add(B_ij);     R_iajb[ij]->add(E_ij);
    Rn_iajb[ij]->add(C_ij);    Rn_iajb[ij]->add(D_ij);
}
```

Each pair's `Qma`, `Qab`, `T_iajb`, `S_PNO`, `K_iajb` are loaded ONCE and reused across all residual contributions. Cache locality is excellent.

### Our class: 4+ separate kernel calls, each doing a parallel-for over pairs

```cpp
run_phase_k_ladder_into(...);      // parallel-for over pairs
// ... (BE plan kernel runs over buckets, then per-pair scatter)
run_phase_be_into(...);            // per bucket
// (per-pair scatter for BE)         // separate parallel-for
run_phase_c_term_into(...);        // batched kernel
run_phase_d_term_into(...);        // batched kernel
// (per-pair CD scatter)             // separate parallel-for
run_phase_g_term_into(...);        // batched kernel (× 2 for ik/jk)
// (per-pair G_term scatter)         // separate parallel-for
```

Each phase: separate omp-parallel-for spawn (~25-50us each), reload Qma/Qab/T2 from RAM, scratch alloc. 4-6 separate sweeps per cycle.

## What closing this gap looks like

**Refactor target:** combine the per-pair scatters of K+ladder, r2_BE, r2_CD, r2_G into ONE outer parallel-for over canonical pairs. The kernels (DLPNObe_kernel, c_term/d_term/g_term batched) still produce their flat output buffers in separate calls (their bucketed structure is independent of canonical pair ordering); the per-pair COMBINE step (R2[p] += K + A + B + E_tilde*T2 + C + D + G) becomes one fused loop.

Expected wins:
- Saves 3-4 thread-spawn overheads (~150us total)
- Cache locality on R2_buf, Fab_flat, T2_flat: each pair touched once, not 4-5 times. For ~50MB R2 working set across 16 threads, this matters.
- Estimated saving: 0.05-0.10s/cycle. NOT enough to fully close the 0.24s/cycle gap to Psi4.

**For full closure**, would need to also:
1. **Eliminate cycle 1 (PySCF baseline at 7.5s)**: currently builds plan caches (G_tilde, BE buckets, CD batched view, t3+t4 plans). Class consumes these. Moving plan-build into class C++ side eliminates this 7.5s one-time cost and the 1.81s pack-once.
2. **Reduce convergence cycles 16 → 14**: numerical drift (1.9e-5 Eh) causes class to converge one cycle later than baseline. Would need exact-numerical-match across kernels to fix. Worth ~3s/run.

## Other findings during diagnosis

1. **Psi4 BE step is also `~0.4s`-ish per-cycle in CPU time** (estimated from dgemm count: 175 ij × ~30 (k,l) × 4 BLAS calls × 6us = ~0.5s). Their per-cycle parity isn't from a faster BE — it's from cache locality and reduced spawn overhead.

2. **Psi4 also has separate `compute_B_tilde()`, `t1_fock()`, etc.** functions called from the cycle. Same general structure as ours. The big difference is at the R2 ASSEMBLY step.

3. **Psi4 build at `/environments/psi4_jiang/install/bin/psi4` is serial-only** — direct wall comparison is unfair. The reference 15.1s came from the user's parallel-build benchmark on Xeon 6136. Don't use the local Psi4 wall.

## Files to modify for the fusion refactor

- [`pyscf/lib/cc/dlpno_ccsd_solver.cpp`](pyscf/lib/cc/dlpno_ccsd_solver.cpp) — `run_one_cycle` lines 1448-1789 hold all 4 R2 phases. Fuse the per-pair scatter sections into one outer parallel-for.
- The kernels (`run_phase_be_into`, `run_phase_c_term_into`, etc.) keep their bucketed/batched structure — they fill flat output buffers. Only the per-pair gather/combine step is fused.

Sketch of fused loop:
```cpp
#pragma omp parallel for schedule(dynamic, 1)
for (int p = 0; p < N; ++p) {
    if (is_strong_pair[p] == 0) continue;
    const int npno = npno_arr[p];
    if (npno == 0) continue;
    const int64_t r2_off = in_.t2_offsets[p];
    
    // Step 1: K + A_ladder (in-place into R2)  — currently done inside run_phase_k_ladder_into
    // Step 2: R += B[p]                         — from BE plan flat_B
    // Step 3: E_tilde + T2*E_tilde + E_tilde*T2 — from BE plan flat_E + Fab_flat
    // Step 4: C_term: 0.5*Cij + Cij.T + 0.5*Cji.T + Cji
    // Step 5: D_term: Dij + Dji.T
    // Step 6: G_term: flat_G_ij + flat_G_ji.T
}
```

Total per-pair work is small (a few BLAS-equivalent ops × npno²) — should fit in L1/L2 cache.
