# Handoff: per-triple vooo/vvov FLOPS restructure to match Psi4

## One-paragraph context

Water-chain (T) exponent is currently 2.197 after landing `T_CUT_DO_TRIPLES=1e-2` + the Psi4-matched prescreen (separate PRE sparse-DF infrastructure with `T_CUT_MKN_TRIPLES_PRE=0.1`, `T_CUT_DO_TRIPLES_PRE=2e-2`, `T_CUT_TNO_PRE=1e-7`). Target is Jiang's 1.78 — still a ≈0.4 exponent gap. Diagnostic shows `naux_ijk` is already saturated at ~176 (O(1) per triple), so aux locality is not the issue. **The remaining N-scaling lives in the per-triple vooo/vvov contraction pattern in `_w3_intermediate`.** Threshold tweaks won't close this gap; only a structural restructure of the per-triple contractions will.

See `/home/ec2-user/.claude/projects/-home-ec2-user/memory/project_triples_psi4_port.md` for the full cumulative history (thresholds audited, what was ruled out, benchmark artifacts).

## The diagnosis

Diagnostic added 2026-04-22 at `run_lccsd_t_ext` (prints avg `naux_ijk` and `triple_domain` over all valid triples):

| chain | nocc | naux_ijk | triple_domain |
|---|---|---|---|
| water4  | 16 | avg 169 / 336  | avg 13.7 / 16  (max 16) |
| water8  | 32 | avg 175 / 672  | avg 16.0 / 32  (max 32) |
| water10 | 40 | avg 176 / 840  | avg 17.6 / 40  (max 38) |

`naux_ijk` is **saturated** (flat across w4→w10 as total naux grows 2.5×). `triple_domain` is **still growing** (13.7→17.6), with max approaching full nocc. The vooo/vvov contractions use `triple_domain` as an inner index — that's our remaining O(N) per triple.

Per-triple profile from prior session (still directionally valid):

```
fn                         w4 / w10 (ms)  per-call exp
_triple_pno_union_psi4     27  → 41       0.46
_build_triple_local_DF     84  → 173      0.79
_w3_intermediate           37  → 45       0.22   ← looks fine solo
remaining (t2_mr etc.)     58  → 90       0.48   ← but t2_mr is the n_domain axis!
total _process_one_triple  206 → 349      0.58
```

The `remaining (t2_mr etc.)` bucket is the building of `t2_mr[m, r, a, b]` and its usage — that's where `n_domain` scaling hides.

## The plan

Restructure the per-triple flow in `_process_one_triple` and `_w3_intermediate` to mirror Psi4 `triples.cc:609+`.

### Step 1 — pre-build all 6 K_ooov permutation matrices once per triple

File: `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/lccsd_t.py`, in `_process_one_triple` around L654 (after building `ovL_ijk`, `ooL_lmo_full`, `vvL_sc`).

Psi4 at [triples.cc:775-809](/environments/psi4_jiang/psi4/src/psi4/dlpno/triples.cc#L775-L809) builds six matrices:
```cpp
K_iojv = ovL[i] @ ooL[j]^T    // shape (n_tno, nlmo_ijk)
K_iokv = ovL[i] @ ooL[k]^T
K_jokv = ovL[j] @ ooL[k]^T
K_joiv = ovL[j] @ ooL[i]^T
K_koiv = ovL[k] @ ooL[i]^T
K_kojv = ovL[k] @ ooL[j]^T
```
All 6 once. Shape `(n_tno, nlmo_ijk)` each. Used in the vooo subtraction loop.

Our current code (in `_w3_intermediate` L432-437) RECOMPUTES `A_al = ovL_sc[ip] @ ooL_sc_full[iq].T` *inside* the 6-permutation loop. That's 6 matmuls of shape `(n_tno, naux_ijk) @ (naux_ijk, n_domain) → (n_tno, n_domain)` — redundant with the pre-build.

**Action**: lift this out of the per-permutation loop. Build all 6 K_ooov once in `_process_one_triple`, pass into `_w3_intermediate` via a new kwarg.

### Step 2 — restructure the vooo subtraction as a per-m loop

Psi4 [triples.cc:832-844](/environments/psi4_jiang/psi4/src/psi4/dlpno/triples.cc#L832-L844):
```cpp
for (int l_ijk = 0; l_ijk < nlmo_ijk; ++l_ijk) {
    int l = lmotriplet_to_lmos_[ijk][l_ijk];
    int il = i_j_to_ij_[i][l];  // pair (i,l)
    // Build per-pair S overlap: shape (n_pno_il, n_tno)
    auto S_il_ijk = linalg::doublet(X_pno_[il], submatrix_rows(*S_ijk, il_idx_list), true, false);
    // T_il: shape (n_tno, n_tno)
    auto T_il = linalg::triplet(S_il_ijk, T_iajb_[il], S_il_ijk, true, false, false);
    for (int a=0; a<n_tno; a++)
        for (int b=0; b<n_tno; b++)
            for (int c=0; c<n_tno; c++)
                W[idx](a, b*n_tno+c) -= T_il[a,b] * K_ooov[idx](l_ijk, c);
}
```

Key points:
- **Per-m pair lookup**: `T_iajb_[il]` is the pair-PNO T2 amplitude (small: `n_pno × n_pno`), not a dense `(n_domain, ...)` tensor.
- **Per-pair S projection**: `S_il_ijk` is built fresh per m. Shape `(n_pno_il, n_tno)` — small.
- **T_il**: `(n_tno, n_tno)` — result of triple product.
- Outer loop over `l_ijk` (= `m_local`).

Our current code (in `_w3_intermediate` L431-437):
```python
A_al = ovL_sc[ip] @ ooL_sc_full[iq].T          # (n_tno, n_domain)
t2_mbc = t2_sc_full[:, ir]                     # (n_domain, n_tno, n_tno)
base -= (A_al @ t2_mbc.reshape(m, n*n)).reshape(n, n, n)
```

The bulk matmul uses `t2_sc_full` pre-built at L802-805 of `_process_one_triple` — that's where `n_domain` is materialized:

```python
t2_mr = np.zeros((m_dom_size, 3, n_tno, n_tno))
for r_local, r_global in enumerate(triple_lmo):
    for m_local, m_global in enumerate(triple_domain):
        t2_mr[m_local, r_local] = _proj_t2(m_global, r_global)  # O(1) per pair via _U_for cache
```

The `_proj_t2` call is O(1) per pair (thanks to the W cache), so building `t2_mr` is O(n_domain × 3 × n_pno × n_tno²) — and it's the same asymptotic as Psi4's per-m loop *in FLOPS*, but with completely different memory access and cache-reuse patterns.

**Hypothesis**: the bulk path materializes a big (n_domain, 3, n_tno, n_tno) tensor and then contracts it. Cache behavior on this tensor (in the second matmul `A_al @ t2_mbc`) may incur O(n_domain²) in practice due to memory traffic scaling with both dimensions. Psi4's per-m loop streams one m at a time and keeps the working set small.

**Action**: restructure `_w3_intermediate` to loop per-m, compute T_il on the fly, subtract into W with a rank-1-ish outer product. Avoid materializing `t2_mr` at all if possible.

### Step 3 — validate incrementally

Each step should be validated in isolation:
1. After step 1 (K_ooov pre-build): energy identical to machine precision, wall time maybe 5-10% faster on w10.
2. After step 2 (per-m loop): energy identical, exponent target 1.9-2.0.

Validation commands:

```bash
# Sync
cp /home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/lccsd_t.py \
   /environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/lccsd_t.py

# Water scaling — the exponent is what matters
cd /home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling
/environments/miniconda3/envs/tmc/bin/python -u run_water_scaling.py \
    --basis cc-pvdz --ncores 64 --scf-blas 16 --chains 4,8,10 \
    --out results/water_scaling_flops_step1.json   # or _step2

# S22-1/2/3/8 accuracy gate
cd /home/ec2-user/Work/3d_tmcs/orca_benchmarks/s22
/environments/miniconda3/envs/tmc/bin/python -u run_s22_pyscf.py \
    --basis cc-pvdz --ncores 64 --scf-blas 16 --dimers 1,2,3,8 \
    --out results/pyscf_flops_step1.json
```

Compare to baselines listed below.

## Current state of the world (before this handoff)

Water-chain (T) scaling, cc-pVDZ TightPNO:

| checkpoint | t_(T) w4/w8/w10 (s) | exponent | Δe_t vs full (w10) |
|---|---|---|---|
| full-naux | 4.5 / 34.9 / 47.8 | 2.679 | — |
| Psi4 port + batched sparse DF | 4.0 / 23.6 / 45.5 | 2.398 | — |
| + MKN_TRIPLES=1e-2 | 4.0 / 19.4 / 37.1 | 2.386 | −36 μEh |
| + T_CUT_DO_TRIPLES=1e-2 | 4.2 / 17.9 / 33.3 | 2.207 | −33 μEh |
| + (T0) prescreen, full Psi4 match | 4.6 / 19.9 / 35.6 | **2.197** | **−21 μEh** |
| **Jiang target** | — | **1.78** | — |

All changes synced to installed pkg at `/environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/`.

## Files and landmarks

Source of truth (edit here, then sync to installed):
- `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/lccsd_t.py` — 1917 lines.
  - `_triple_pno_union_psi4` at L140 (has `pao_domains_triple` kwarg)
  - `_w3_intermediate` at L338 (the hot path to restructure)
  - `_build_triple_local_DF` at L518
  - `_process_one_triple` at L654 (has `pao_domains_triple` kwarg)
  - `_process_degenerate_pair` at L848
  - `run_lccsd_t_ext` at L1033 (signature includes `doi_iu` and default `T_CutTriplesWeak=1e-7`)
  - `_build_triples_infrastructure` closure inside `run_lccsd_t_ext` (L1208-1300-ish) — builds tight + PRE infrastructures
  - Diagnostic block prints `naux_ijk` and `triple_domain` stats right before the main loop

Psi4 reference (read-only):
- `/environments/psi4_jiang/psi4/src/psi4/dlpno/triples.cc`
  - `compute_lccsd_t0` at L609 (the main per-triple kernel)
  - **K_ooov pre-build block** at L775-809 (step 1)
  - **vooo per-m loop** at L832-844 (step 2)
  - `triples_sparsity(prescreening)` at ~L260-390
  - `lmotriplet_to_lmos_[ijk]` at L354-357 (their triple_domain — uses strong AND weak pairs)
- `/environments/psi4_jiang/psi4/src/psi4/dlpno/dlpnobase.cc` L139-148 — TIGHT preset values
- `/environments/psi4_jiang/psi4/src/read_options.cc` L2513-2590 — full Psi4 threshold list

Benchmark data:
- `/home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling/results/`
  - `water_scaling_cc-pvdz.json` — full-naux baseline
  - `water_scaling_psi4match.json` — current state (this session's final result)
- `/home/ec2-user/Work/3d_tmcs/orca_benchmarks/s22/results/pyscf_do_triples.json` — S22 baseline for accuracy gate

## What to watch

- **Energy drift gate**: after step 1 (pure refactor, no semantic change), energies MUST match psi4match within 1 μEh. After step 2 (new contraction order), within ~30 μEh is acceptable (floating-point reordering) but not more.
- **Exponent target**: step 2 should bring us below 2.0. If it doesn't, the bottleneck is somewhere else and we need more profiling.
- **Water chain may not reach 1.78**: linear geometry is adversarial for locality. Jiang's 1.78 may be on 3D systems. Consider running a 3D benchmark (S66 or small protein fragment) to cross-check what exponent is actually achievable for our code on non-chain systems.

## Git status

Branch `dlpno_tccsd` — uncommitted edits across `lccsd_t.py`, `driver.py`, `local_orbs.py` plus two test files updated for the new `make_paos` return tuple (`test_pno_construction.py`, `test_dlpno_tccsd_vs_tccsd.py`). Do NOT commit without approval.
