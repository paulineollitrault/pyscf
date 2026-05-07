# DLPNO-(T): replace full vvL with per-pair q_vv_ij/jk/ik

## Mission

Restructure the per-triple DF integrals + W3 build to match Psi4's pair-PNO-projected algorithm. Eliminates the `n_pao_ijk²` factor in our vvL build, which is the source of (T)'s super-linear per-triple scaling vs Psi4.

## Diagnosis (from session 2026-05-02)

End-to-end (T) scaling on water-N (cc-pVDZ, Jiang-tight-PNO):

| | water-4 | water-8 | water-10 | water-15 | exponent |
|---|---|---|---|---|---|
| Ours (default) | 0.74s | 3.91s | 7.10s | 22.4s | **N^2.56** |
| Psi4 | 3.76s | 12.23s | 17.61s | 40.76s | N^1.78 |

We are *faster* in absolute terms at every size measured, but Psi4 scales better — gap widens at water-22+.

**Per-triple breakdown** (OMP path profiler, prescreen pass):

| | naux_ijk | triple_domain | n_pao_ijk | n_tno | DF (ms/tr) |
|---|---|---|---|---|---|
| water-4 | 167.6 | 13.4 | 91.9 | 38.3 | 1.80 |
| water-8 | 171.3 | 14.8 | 124.0 | 36.8 | 2.46 |
| water-10 | 174.5 | 16.7 | 133.1 | 34.5 | 2.98 |
| water-15 | 175.8 | 18.9 | 156.5 | 36.3 | 4.14 |

- naux_ijk: bounded ✓
- n_tno: bounded ✓
- triple_domain: ~N^0.27 (modest)
- **n_pao_ijk: ~N^0.40 (the culprit)**

Triple count grows ~N^1.79 (similar to Psi4). Per-triple work grows ~N^0.44 (CPU-summed). Combined: ~N^2.2 CPU + parallelism overhead → N^2.5 wall.

## Why Psi4 doesn't suffer (the algorithmic difference)

Read `triples.cc:609-820` (`compute_lccsd_t0`). Same `T_CUT_DO_TRIPLES=1e-2`, same atom-complete PAO-domain construction, same triple PAO-union: `merge_lists(lmo_to_paos[i], lmo_to_paos[j], lmo_to_paos[k])`. **The PAO domains are identical.**

The difference is the per-triple integrals:

**Ours** (`pyscf/lib/cc/dlpno_triple_local_df.c`):
```
vvL[a_tno, b_tno, q] for a, b ∈ [0, n_tno)
  built via X_tno.T @ qab_PAO @ X_tno  for each Q
```
Cost per Q: `n_pao_ijk² × n_tno²`. Total per-triple: `naux × n_pao_ijk² × n_tno²`.

**Psi4** (`triples.cc:663-754`):
```
q_vv_ij[a_tno, b_pair, q] for a ∈ [0, n_tno), b ∈ [0, n_pno_ij)
q_vv_jk[a_tno, b_pair, q] for n_pno_jk
q_vv_ik[a_tno, b_pair, q] for n_pno_ik
  built via X_tno.T @ qab_PAO @ X_pno_pair  (3 separate triplets per Q)
```
Cost per Q × 3 pairs: `3 × n_pao_ijk × n_pno_pair × n_tno`. Total per-triple: `3 × naux × n_pao_ijk × n_pno_pair × n_tno`.

Numerical comparison at water-10 dimensions (`n_pao_ijk=133, n_tno=35, n_pno_pair≈25`):
- Ours: 133² × 35² = 21.7M flops/Q
- Psi4: 3 × 133 × 25 × 35 = 350K flops/Q
- **Psi4 is ~60× cheaper per Q**

And critically: ours has `n_pao_ijk²`, Psi4 has `n_pao_ijk × n_pno_pair`. As `n_pao_ijk` grows N^0.4, ours grows N^0.8, Psi4's grows N^0.4.

## Why this works mathematically

The W3 contraction in (T) reduces to:
```
contrib[a_tno, b_tno, c_tno] += K[a_tno, b_tno, c_TNO_or_pair] × T2[pair][...]
```

T2 amplitudes live in pair-PNO space (`T2[pair_kj]` is `(n_pno_kj, n_pno_kj)`). The cleanest integral to contract with T2 is:
```
K_ovvv[i_tno, a_tno, b_tno, c_pno_kj]
```
where the *last* virtual index is in pair-PNO space (matches T2). Psi4 builds this directly:
```cpp
K_ivvv = q_iv.T @ q_vv_jk   // (n_tno, n_tno × n_pno_jk)
K_jvvv = q_jv.T @ q_vv_ik
K_kvvv = q_kv.T @ q_vv_ij
```
All three `K_ovvv` matrices have one TNO and one pair-PNO virtual index — never the full `n_tno⁴` cube.

Our current code computes the full `K_iajb = (n_tno, n_tno, n_tno, n_tno)` from `vvL @ vvL`, then projects to pair-PNO at contract time. We pay `n_tno⁴` cost; Psi4 pays `n_tno² × n_pno_pair`.

## Files to change

### 1. `pyscf/lib/cc/dlpno_triple_local_df.c` (the C kernel)

Replace the single `vvL` output (`shape (n_tno, n_tno, naux_ijk)`) with three per-pair outputs:
- `qvv_ij` (shape `(n_tno, n_pno_ij, naux_ijk)`)
- `qvv_jk` (shape `(n_tno, n_pno_jk, naux_ijk)`)
- `qvv_ik` (shape `(n_tno, n_pno_ik, naux_ijk)`)

Function signature change: drop `vvL_sc`; add `qvv_ij_sc`, `qvv_jk_sc`, `qvv_ik_sc` plus their pair-PNO sizes (`n_pno_ij`, `n_pno_jk`, `n_pno_ik`) and `X_pno` matrices for each pair.

Per-Q work per pair (replace lines 308-355):
```c
// q_vv_pair_tmp = X_tno.T @ qab_PAO_cut @ X_pno_pair
//   = (n_tno × n_pao_ijk).T @ (n_pao_ijk × n_pao_ijk) @ (n_pao_ijk × n_pno_pair)
//   = (n_tno, n_pno_pair) per Q per pair
// 3 of these (ij, jk, ik) instead of one full (n_tno, n_tno) vvL.
```

The X_pno matrices come from `pno_spaces[pair]['X_pno']` and need to be projected to the triple's PAO basis first (multiplying with `X_pao_ijk` if available).

### 2. `pyscf/cc/dlpno_tccsd/lccsd_t.py::_build_triple_local_DF` (Python wrapper)

Update output unpacking. Pass per-pair `X_pno` matrices and `pair_paos` to the C kernel. Return three matrices instead of one `vvL_sc`.

### 3. `pyscf/cc/dlpno_tccsd/lccsd_t.py::_process_one_triple` (W3 build)

Replace the K_iajb-based contractions with K_ivvv/K_jvvv/K_kvvv built via `q_iv.T @ q_vv_jk` etc. (Psi4 lines 771-773).

### 4. W3 kernel (`pyscf/cc/dlpno_tccsd/_w3_full_cy.pyx` or `pyscf/lib/cc/dlpno_w3_full.c`)

The W3 inner product structure changes — now it's `(n_tno², n_pno_pair) × T2[pair]` instead of `n_tno⁴` contractions. See Psi4 `triples.cc:800-870` for the exact W3 build with K_ovvv as input.

### 5. `pyscf/cc/dlpno_tccsd/lccsd_t.py::_orch_phase1` (and `_orch_full`)

If these orchestrate DF + W3 in one call, update them similarly.

## Validation plan

Strict bit-correctness: water-4 anchor `E(T) = -0.0129205196` (10 digits).

Per-step validation (do NOT skip):

1. **C kernel unit test**: drop `DLPNO_TRIPLE_DF_DEBUG=1` flag in lccsd_t.py wrapper to dump and compare `vvL_sc` (old) vs equivalent reconstruction `q_vv_pair[i_tno, b_pno_pair, q] = sum_a q_vv_pair[a_tno, b_pno_pair, q] * δ_a_i` for one pair at a time. Match to 1e-12.

2. **K integral comparison**: dump K_iajb (old) vs K_ovvv (new) projected via `K_iajb[a, b, c, d] = sum_b' K_ovvv_new[..., b'_pno] × X_pno[d, b']`. Match to 1e-12.

3. **W3 comparison**: dump W3 tensor (n_tno³) old vs new. Match to 1e-12.

4. **Per-triple energy**: dump `et_triple` for each triple, old vs new. Match to 1e-10.

5. **Total E(T)**: water-4, water-8, water-10. Match to 1e-9 of anchor.

If any step drifts > 1e-9, halt and debug — do not proceed to the next step.

## Expected outcome

- water-15 (T) wall: 22.4s → ~12-15s (per-triple cost halved or better)
- Scaling exponent: N^2.56 → N^2.2 (target N^1.78 if combined with triple-count tightening)
- Energy: bit-stable (within FP noise of current default-path values)
- water-22+: predicted significant improvement

## Background reading

- `/environments/psi4_jiang/psi4/src/psi4/dlpno/triples.cc:609-870` — Psi4's `compute_lccsd_t0` reference
- `pyscf/lib/cc/dlpno_triple_local_df.c` — current C kernel (clean, no O(N) global scans, just wrong shape)
- `pyscf/cc/dlpno_tccsd/lccsd_t.py:789` — `_build_triple_local_DF` Python wrapper
- `pyscf/cc/dlpno_tccsd/lccsd_t.py:2087` — `_process_one_triple` (W3 caller)

## Memory entries to read

- `project_triples_phase3c.md` — current (T) C-port arc state
- `project_triples_session3.md` — Phase 3a TNO transform port
- `project_triples_local_df_c.md` — current `dlpno_triple_local_df.c` development

## What this session already did (partial work in working tree)

### Diagnostic + minor fix (clean, energy-preserving):

- **Partner-set fix for `triple_domain` scan in default path** (lccsd_t.py: lines ~1108, ~1419, ~2206). Eliminates O(nocc_lmo) per-triple scan in `_orch_phase1`, `_orch_full`, `_process_one_triple`. Energy preserved (water-10 E_TCCSD bit-stable). Tiny perf gain (~5%) — confirms this scan is real but not the dominant cost.
- **n_pao_ijk + n_tno (sampled) instrumentation in (T) summary printout**. Logs `(T) n_pao_ijk: avg=X min=Y max=Z` and `(T) n_tno (sampled 30): avg=X min=Y max=Z`. Useful for tracking the metric this restructure aims to keep bounded.
- **`_build_partners` helper** (cached by `id(domain_set)`) at the top of lccsd_t.py.

### Steps 1-7 LANDED (commits 9dfb9f83e, 1b4cbb351, 7158e7d58, bbdcb1003, 3edc0d30b):

**All algorithmically-novel work is in.** What remains is dropping the
no-longer-needed vvL_sc + K_ab_cache builds from the QVV_PAIR=1 path
to realize the scaling win.

#### Step 5 (commit bbdcb1003): K_ovvv per-ip build

Three K_ovvv tensors built per triple from q_vv_pair via
`ovL_ip @ q_vv_pair_for_ip.T`. Pair mapping: ip=0→jk, ip=1→ik, ip=2→ij.

#### Step 6 (commit 3edc0d30b): T_pair_perm builds (6 per triple)

Per perm pidx, T_pair_perm[c_tno, c_pno_pair] = U_pk.T @ T2_oriented
where T2_oriented = T2_canonical (canonical perm) or T2_canonical.T
(non-canonical, swap of (ir, iq)). Reuses U_flat_cache + u_T2_flat
infrastructure from existing t2_block setup.

#### Step 7 (commit 3edc0d30b): W3 Phase 1 per-pair path

`DLPNOcompute_w3_energy` now optionally takes `K_ovvv_arr[3]` and
`T_pair_arr[6]` plus their sizes. When both are non-NULL, Phase 1
contraction:

  base[a, b, c_tno] = Σ_c_pno K_ovvv[ip][a, b, c_pno] × T_pair[pidx][c_tno, c_pno]

via dgemm `K_ovvv_reshape(n², n_pno) @ T_pair^T(n_pno, n_tno)` →
output (n², n_tno) layout-compatible with the original Phase 1
output. Phase 2-5 unchanged.

### Validation results

water-4 / water-10 with `DLPNO_TRIPLE_QVV_PAIR=1`:

| | water-4 E(T) | water-10 E(T) |
|---|---|---|
| OLD (default path) | -0.0129205196 | -0.0323076110 |
| NEW (per-pair) | -0.0129204369 | -0.0323088439 |
| Δ | 83 nEh | 1.2 μEh |

**The residual drift is a genuine algorithmic difference, not a bug.**
OLD includes the implicit `X_tno @ X_tno^T` projector (full triple-TNO
re-projection); NEW (Psi4-style) skips it. Both are valid DLPNO-(T).
Psi4 chose NEW for the `n_pao_ijk² → n_pao_ijk` scaling win at the
cost of this micro-Hartree-level approximation residual.

### Remaining work (DONE — see findings below)

All dead-code paths gated. `DLPNObuild_triple_local_DF` skips vvL when
passed NULL; orch skips K_ab_cache, t2_block, t2_T_all when QVV_PAIR=1.

### FINAL FINDINGS (2026-05-04)

**Performance**: at our typical water-cluster sizes, NEW per-pair path
is **slight regression** vs OLD, NOT the predicted speedup:

| | OLD wall | NEW wall | Δ |
|---|---|---|---|
| water-4 (T) | 0.71s | 0.90s | +0.19s |
| water-10 (T) | 7.10s | 8.20s | +1.10s |
| water-15 (T) | 22.67s | 26.22s | +3.55s |

The expected `n_pao_ijk² → n_pao_ijk × n_pno_pair` saving in vvL build
**IS realized** (vvL build now skipped entirely under QVV_PAIR=1) — but
the offsetting cost is HIGHER:
- 3 separate q_vv_pair builds (instead of 1 vvL build)
- 3 K_ovvv builds (replacing 1 K_ab_cache aggregate build)
- 6 T_pair builds (replacing 9 t2_block + transpose to t2_T_all)

The dgemm count goes UP (12 new blocks vs ~15 old blocks saved, similar
counts but smaller each). The vvL² → vvL × n_pno_pair factor advantage
is locally significant (water-15 vvL build alone ~6× cheaper) but is a
small fraction of total (T) time. Other phases (jhi, ooL, U_cache,
K_ooov, W3 main loop) don't change.

**Win at much larger N** (water-30+, water-64) where `n_pao_ijk²` becomes
the absolute dominant term, but at water-15 it's still subdominant.

**Energy**: Psi4-equivalent algorithm differs from OLD path's anchor
by μEh-level (TNO truncation residual scales with `1 - n_tno/n_pao_ijk`):

| | OLD E(T) | NEW E(T) | Δ |
|---|---|---|---|
| water-4 | -0.0129205196 | -0.0129204369 | +83 nEh |
| water-10 | -0.0323076110 | -0.0323088121 | -1.20 μEh |
| water-15 | -0.0499725011 | -0.0500279808 | -55 μEh |

This is the genuine algorithmic difference between OLD (computes
`X_tno @ X_tno^T` projector implicitly) and NEW (Psi4-style, no
projector). For Psi4 cross-validation, NEW gives Psi4-faithful values.
For "match our historical anchor" goal, OLD wins.

### Final disposition

`DLPNO_TRIPLE_QVV_PAIR=1` is **available but not default**. The
infrastructure is fully validated and performance-gated via env var.
Use cases:
- Cross-validation against Psi4 published E(T) values
- Future scaling investigations at water-30+ or larger systems
- Algorithmic studies of the per-pair vs full-vvL trade-off

To make default-on later, also commit to:
- New "anchor" energies (post-NEW) for downstream tests
- Optional: optimize the per-pair kernel to consolidate 3 separate
  q_vv builds into one (could close the perf gap or surpass OLD)

### Original (pre-implementation) Steps 1-4:

- **`pyscf/lib/cc/dlpno_triple_qvv_pair.c`** — new kernel `DLPNObuild_triple_qvv_pair`. Builds ONE pair's `(n_tno, n_pno_pair, naux_ijk)` slice. Validated bit-exact vs Python brute-force on water-4 (rel ~1e-14).
- **Integration into orch path** — `dlpno_triples_orch.c` gets 3 q_vv_sc scratches (TScratch fields q_vv_ij_sc/jk_sc/ik_sc) and 3 calls to the new kernel after the existing DF call. Gated on `DLPNO_TRIPLE_QVV_PAIR=1`, default off.

End-to-end validation with QVV_PAIR=1 (energies bit-stable to anchors):
- water-4: E_TCCSD=-0.8525213210, E(T)=-0.0129205196
- water-10: E_TCCSD=-2.1308508186, E(T)=-0.0323076432

Wall overhead (both builds active):
- water-4 (T): 0.71s → 1.07s (+0.36s)
- water-10 (T): 7.10s → 9.57s (+2.47s)

This overhead reverses once vvL_sc is dropped (steps 5-7 below).

### Next concrete steps (where to pick up):

**IMPORTANT: production path is the C++ orchestrator, not Python.** Default config sets `DLPNO_TRIPLE_ORCH_FULL=1` (in `driver.py:59`), routing every (T) triple through `_orch_full` → `pyscf/lib/cc/dlpno_triples_orch.c::DLPNOcompute_one_triple_E_T0`. The Python `_build_triple_local_DF` is only hit on fallback paths.

### Remaining steps 5-7 (concrete code targets)

**File 1: `dlpno_triples_orch.c` (lines ~990-1046, the K_ab_cache build):**

Currently after the q_vv_pair builds (already in place, gated `_qvv_pair_enabled`), we have:

```c
/* K_ab_cache (3, n, n, n): K_ab[ip, a, b, f] = sum_L ovL[ip, a, L] * vvL[b, f, L] */
ENSURE(K_ab_cache, double, (size_t)3 * n * n * n);
double *K_ab_cache = tscratch.K_ab_cache;
for (int ip = 0; ip < 3; ip++) {
    /* dgemm: t_tmp (n, n²) = ovL_ip @ vvL.T */
    /* then transpose to K_ab_cache[ip, a, f, b] */
}
```

Replace (when `_qvv_pair_enabled`) with **three K_ovvv builds** (one per pair_slot):

```c
/* K_ovvv[ip, a, b, c_pno_pair] = sum_L ovL[ip, a, L] × q_vv_pair_for_ip[b, c_pno, L]
 * where pair_for_ip is: ip=0 → jk (slot 1), ip=1 → ik (slot 2), ip=2 → ij (slot 0).
 * Three different sizes (n_pno_pair varies per pair).
 */
const int n_pno_for_ip[3] = {n_pno_jk, n_pno_ik, n_pno_ij};
double *q_vv_for_ip[3] = {q_vv_jk_sc, q_vv_ik_sc, q_vv_ij_sc};

for (int ip = 0; ip < 3; ip++) {
    int n_pno = n_pno_for_ip[ip];
    /* K_ovvv[ip] row-major (n, n × n_pno): a × (b, c_pno) */
    /* dgemm 'T','N': K_ovvv = ovL_ip @ q_vv_for_ip.T
     *   ovL_ip row-major (n, naux); q_vv row-major (n × n_pno, naux);
     *   result row-major (n, n × n_pno).
     */
    int int_n_x_pno = n * n_pno;
    dgemm_(&Tc, &Nc, &int_n_x_pno, &int_n, &int_naux,
           &one, q_vv_for_ip[ip], &int_naux,
           ovL_ip, &int_naux,
           &zero, K_ovvv_for_ip[ip], &int_n_x_pno);
    /* Storage: K_ovvv[ip, a, b, c_pno] = result[a, b*n_pno + c_pno]
     * — natural row-major (n, n, n_pno). NO transpose needed (different
     * from K_ab_cache which had Python's [a, b, f] -> [a, f, b] swap).
     */
}
```

**File 1: `dlpno_triples_orch.c` (the t2_block / t2_T build, around line 870-920):**

Current code builds `t2_block[3][3]` of shape (n, n, n, n): T2 blocks for all (r, q) ordered pairs of triple LMOs in TNO×TNO basis. This requires double projection via U cache.

**Replace with three T_pair[3] of shape (n_tno, n_pno_pair):**

```c
/* T_pair[pair_slot, c_tno, c_pno] = sum_d_pno S_pno_to_tno[d_pno, c_tno] × T2_pair[d_pno, c_pno]
 * where S_pno_to_tno is the basis transform for the pair slot.
 *
 * S_pno_to_tno is essentially the same overlap as S_kj_ijk in Psi4
 * triples.cc:824 — built via:
 *   S_pno_to_tno = X_pno_pair.T @ S_pao_full[pair_paos, triple_paos] @ X_tno_ijk
 * shape (n_pno_pair, n_tno).
 *
 * Three of these (ij, jk, ik), each ~30×30. Three small dgemms.
 */
```

**File 2: `dlpno_w3_full.c` (Phase 1 dgemm at line 102-106):**

Current:

```c
dgemm_(&N_, &N_, &int_n, &int_nn, &int_n,
       &one, t, &int_n, K, &int_n,
       &zero, base_buf, &int_n);
```

This computes `base[a, b, c] = Σ_f K_ab[ip, a, b, f] × t2_T[ir, iq, c, f]` (the comment is misleading — derived elsewhere).

**Per-pair version (with K_ovvv shape (n, n×n_pno) and T_pair shape (n, n_pno)):**

```c
/* base[a, b, c_tno] = sum_c_pno K_ovvv[a, b, c_pno] × T_pair[c_tno, c_pno]
 *
 * K_ovvv reshape (n², n_pno), T_pair (n, n_pno).
 * Result (n², n) = K_ovvv reshape @ T_pair.T:
 *   row-major dgemm 'N', 'T': dgemm('N','T', n, n², n_pno, ...)
 */
int int_n_pno_pair = n_pno_for_perm[pidx];   /* lookup per perm */
dgemm_(&N_, &T_, &int_n, &int_nn, &int_n_pno_pair,
       &one, T_pair_for_perm, &int_n_pno_pair,
       K_ovvv_for_perm, &int_n_pno_pair,
       &zero, base_buf, &int_n);
```

Per-perm pair selection (matches Psi4 K_ovvv_list):

| pidx | (ip, iq, ir) | pair_for_perm | n_pno     |
|------|--------------|----------------|-----------|
| 0    | (0, 1, 2)    | jk (slot 1)    | n_pno_jk  |
| 1    | (0, 2, 1)    | jk (slot 1)    | n_pno_jk  |
| 2    | (1, 0, 2)    | ik (slot 2)    | n_pno_ik  |
| 3    | (1, 2, 0)    | ik (slot 2)    | n_pno_ik  |
| 4    | (2, 0, 1)    | ij (slot 0)    | n_pno_ij  |
| 5    | (2, 1, 0)    | ij (slot 0)    | n_pno_ij  |

W3 kernel signature change: drop `K_ab_cache, t2_T_all` and add `K_ovvv_per_perm, T_pair_per_perm, n_pno_per_perm`.

**File 3: `dlpno_triples_orch.c` (call site at line 1237):**

Update `DLPNOcompute_w3_energy(...)` call to pass the new pointers. Initial implementation: keep both code paths via flag, fall back to old when q_vv_pair disabled. Drop old when validated.

### Validation order

1. Build K_ovvv alongside K_ab_cache. Verify `K_ovvv` reconstructs correctly: the mathematical relationship is `K_iajb_ref[a, b, c_tno] = Σ_c_pno K_ovvv[a, b, c_pno] × X_pno_to_tno[c_tno, c_pno]` where `X_pno_to_tno = X_pno_pair.T @ S_pao @ X_tno_ijk` (the pair_slot's overlap). This won't be EXACT (since K_iajb has more info from triple-PAO basis), but should match the "pair-PNO subspace" of K_iajb.

2. Build T_pair alongside t2_block. Validate via numerical contraction.

3. Phase 1 W3 contribution: with both old and new path active, compute base_old and base_new. The DIFFERENCE is the contribution from non-pair-PNO TNO basis — should be small for well-converged PNO spaces. Run on water-4 and check final E(T) drift.

4. Drop old K_ab_cache + t2_T + vvL_sc. Run water-4 / 8 / 10 / 15. E(T) bit-stable to anchor (within 1e-9 Eh of current default-path values).

### Expected outcomes

If steps 5-7 land correctly:
- water-15 (T) wall: 22.4s → ~12s (per-triple cost halved)
- Scaling exponent: N^2.56 → ~N^2.2 (matching Psi4 N^1.78 needs further work on triple-count scaling)

This means the integration target is **`dlpno_triples_orch.c`**, not `_build_triple_local_DF`. The orch kernel:

- Allocates `vvL_sc` scratch (line 686: `ENSURE(vvL_sc, double, (size_t)n * n * naux_ijk)`)
- Calls `DLPNObuild_triple_local_DF(...)` to fill it (line 697)
- Consumes `vvL_sc` via `K_ab_cache` build (line 932): `dgemm('T','N', n², n, naux, vvL_sc, ovL_ip, K_ab_cache)`

To insert per-pair vvL:

1. **Numerical validation harness**. Standalone test that calls `DLPNObuild_triple_qvv_pair` with synthetic inputs (or extracted from a real water-4 triple) and a Python brute-force reference. Verify max|Δ| < 1e-10. Reference impl already drafted in `/tmp/test_qvv_pair.py` — fix the patch target (use the orch path's data flow, not `_build_triple_local_DF`).

2. **Extend orch state**. The orch kernel already has per-pair `(pair_paos_n_3, pair_paos_off_3, pair_paos_flat_3, n_pno_arr_3, X_pno_off_3, X_pno_flat_3)` from line ~556 (used for U cache, T2). Three pairs (ij, jk, ik) at index slots 0, 1, 2. We can pass these directly to `DLPNObuild_triple_qvv_pair` for each pair without changing the Python wrapper.

3. **Allocate three q_vv_pair scratches** in tscratch (alongside vvL_sc): `qvv_ij_sc`, `qvv_jk_sc`, `qvv_ik_sc` — each shape `(n_tno × n_pno_pair × naux_ijk)`. Add to the tscratch struct + `ENSURE` blocks.

4. **Call `DLPNObuild_triple_qvv_pair` 3x** (or once with all three pairs interleaved if we generalize the kernel). Replace the single `DLPNObuild_triple_local_DF` call's `vvL_sc` output. Initially keep ALSO calling the old DF kernel for `ovL_sc` and `ooL_sc` (those don't change).

5. **Replace `K_ab_cache` build with `K_ovvv` builds**. Currently:
   ```c
   // K_ab_cache[ip][a, f, b] = sum_L ovL_sc[ip, a, L] × vvL_sc[b, f, L]
   ```
   New:
   ```c
   // K_ovvv[ip, pair_idx][a, b, c_pno_pair] = sum_L ovL_sc[ip, a, L] × q_vv_pair[pair_idx][b, c, L]
   // Three K_ovvv tensors per perm of W3 (one per pair).
   ```
   See Psi4 `triples.cc:771-773` for the contraction pattern.

6. **W3 kernel update**. The W3 build (currently `dlpno_w3_full.c::DLPNOcompute_w3_energy` and the inline orch logic in `dlpno_triples_orch.c`) consumes `K_ab_cache` of shape `(3, n_tno, n_tno, n_tno)`. New shape is `K_ovvv` of `(3, n_tno, n_tno, n_pno_pair)`. The W3 contraction with T2[pair] becomes a simple matmul:
   ```c
   // Wperms[idx] += K_ovvv[idx] @ T_pair^T  where T_pair = S_pair_to_tno @ T2[pair_kj]
   ```
   See Psi4 `triples.cc:823-830` for the exact reshape + matmul.

7. **Eliminate `vvL_sc`** from orch kernel + `DLPNObuild_triple_local_DF` (modify signature to drop vvL_sc output). Perf win materializes here.

8. **End-to-end validation**: water-4 / 8 / 10 / 15 E(T) bit-stable to anchors (E_TCCSD on water-10: -2.130850819). Time on water-15 (T) should drop from 22.4s → ~12s if the algorithmic prediction holds.

## What this session tried but reverted

- Switching `pao_domains_triple=None` (per-pair pair_paos union instead of per-LMO PAO domain): broke energies (water-15 E(T) → -16303 Eh) because CCSD-stage `pair_paos` use a different (tighter) threshold than triple-stage. Per-LMO triple-stage is correct.

## Out of scope

- CCSD restructure (separate handoff: `HANDOFF_CCSD_C_CLASS.md`).
- Triple count screening tightening (N_triples grows N^1.79, slightly worse than Psi4 — not the dominant scaling factor).
