# DLPNO-CCSD Psi4-port multi-session arc

**Goal:** port Psi4's DLPNO-CCSD to PySCF in PySCF style — pair-domain `(nlmo_ij, ...)` tensor layout throughout, kernels in `pyscf/lib/cc/dlpno_*.c` (not Cython), bit-for-bit Psi4 algorithmic parity.

This document supersedes [HANDOFF_CC_INTS_REDUCED_REFACTOR.md](HANDOFF_CC_INTS_REDUCED_REFACTOR.md) — that doc described Steps 1-10 of the storage refactor. We've completed Step 1 (p_lmos-axis); the remaining storage-axis tightening (Steps 7-9, p_lmos → pair_lmo_idx) is gated on the C-kernel rewrite, so the work needs a different sequencing than originally planned.

## Where we are (commit `8b108ec10`)

| | Layout axis | Shape | Anchor preserved? |
|---|---|---|---|
| Old (before today) | `nocc` | `(nocc, ...)` scattered | — |
| **Now** | **`p_lmos`** (riatom union) | `(nlmo_p, ...)` | ✓ water-4: -304.98979787; water-10: -2.13088299002 |
| Psi4-true target | `pair_lmo_idx` | `(nlmo_ij, ...)` | gated (see below) |

`p_lmos ⊃ pair_lmo_idx`. On water-4, p_lmos has ~25% more LMOs than pair_lmo_idx. Going from `nocc → p_lmos` was the big memory win (3-4× on Qma at water-22). Going from `p_lmos → pair_lmo_idx` is a smaller follow-on win.

Algorithmic parity status:
- `compute_B_tilde`: ✓ Psi4 layout port done; consumers translate via tuple `(B_local, p_dense)`.
- `compute_ladder`: ✓ Cross-code validated — diag pairs trace match 0.00-0.08% rel err.
- `t1_ints`, `t1_fock`, `compute_C_tilde`, `compute_D_tilde`: not validated cross-code.
- T1/T2 residuals: untouched.

## Why pair_lmo_idx-axis is blocked

Empirical findings (2026-04-27 session):

1. **Data in p_lmos\pair_lmo_idx rows is tiny** (verified by `DLPNO_ZERO_EXTRA_LMOS=all`):
   - water-4: 0.4 μEh drift
   - water-10: 8 μEh drift
   - Per-field bisection: only Qma rows have non-zero data. K_bar_*, i_Qk, j_Qk extras are essentially zero.

2. **Naive axis switch (storage at pair_lmo_idx) drifts much more** than data loss alone:
   - water-4: 30 μEh
   - water-10: 1.22 mEh (initial), -262 μEh (after construction-fix)
   - Direction can flip depending on construction details.

3. The 30/262 μEh excess drift is **not** from helper None paths (verified — `DLPNO_DEBUG_OOD` counter goes to 0 after construction fix), and **not** from the construction fix in isolation (~0.2 μEh drift on top of p_lmos storage).

4. Where the excess drift comes from (root cause unconfirmed, hypothesis): the 3 Cython scatter-back sites (foo_dressed, per_i_stages, T1 residual K_bar) place fewer rows into nocc-shape buffers when ci['p_lmos'] is the smaller pair-domain set. Mathematically the kernel sums should be equivalent (zero rows contribute zero), but empirically there's a discrepancy. **Did not finish localizing this** before stopping the session.

## The Task A / Task C entanglement

The handoff doc Steps 7-9 say "pass reduced T_n / nlmo_pair to the Cython kernels". But the kernel signatures (`per_i_stages123`, `foo_dressed_one`, `t1_fock_batched`, `_per_kl_batched`) hard-code `M=nocc` as the LMO loop bound and accept `T_n[m, a]` indexed by global LMOs. To restrict to pair_lmo_idx-domain we either:

- **(a)** Rewrite all kernel signatures to take `M=nlmo_pair` + a pair-position list. Touches 4-5 `.pyx` files, breaks every caller, must coordinate.
- **(b)** Rewrite kernels in `pyscf/lib/cc/dlpno_*.c` with pair-domain signature from day 1. Bigger upfront, but matches PySCF style (the goal anyway).
- **(c)** Keep the scatter-back compromise and live with the empirical discrepancy.

Decision: **(b)**. Task A (storage axis to pair_lmo_idx) is gated on Task C (C-kernel rewrite). Doing them together makes sense because the kernel signatures are where the axis is encoded.

## Multi-session arc

### Phase I — Cross-code algorithm parity (low-risk, high-value validation)

**Why first:** confirms which functions are already structurally Psi4-faithful before we invest weeks rewriting kernels. Catches silent algorithmic divergence that would otherwise be masked by anchor-preserving refactors.

Estimated: **2-4 sessions**.

**Per-function (template established by `compute_B_tilde` / `compute_ladder`):**

1. Add a per-pair printf dump in Psi4 ccsd.cc next to the function's output.
2. Mirror an env-gated dump in our function.
3. Run Psi4 + ours on water-4 with PM localizer.
4. Diff per-pair invariants (Frobenius norm, trace, sorted eigenvalues on diagonal pairs).

**Functions to validate:**

| Function | Psi4 location | Our location | Output shape |
|---|---|---|---|
| `t1_ints` (i_Qk_t1, i_Qa_t1) | [ccsd.cc:1491-1538](environments/psi4_jiang/psi4/src/psi4/dlpno/ccsd.cc) | [local_df.py:t1_ints](pyscf/cc/dlpno_tccsd/local_df.py) | per-pair (n_local, ...) |
| `t1_fock` (Fkj, Fab, Fai) | [ccsd.cc:1540-1686](environments/psi4_jiang/psi4/src/psi4/dlpno/ccsd.cc) | [local_df.py:t1_fock](pyscf/cc/dlpno_tccsd/local_df.py) | scalar Fock + per-pair Fab |
| `compute_C_tilde` | [ccsd.cc:1721-1873](environments/psi4_jiang/psi4/src/psi4/dlpno/ccsd.cc) | [residual.py:compute_C_tilde_batched](pyscf/cc/dlpno_tccsd/residual.py) | per-pair (npno, npno) |
| `compute_D_tilde` | [ccsd.cc:1876-1941](environments/psi4_jiang/psi4/src/psi4/dlpno/ccsd.cc) | [residual.py:build_D_tilde_batched](pyscf/cc/dlpno_tccsd/residual.py) | per-pair (npno, npno) |
| `compute_G_tilde` | [ccsd.cc:1943-1967](environments/psi4_jiang/psi4/src/psi4/dlpno/ccsd.cc) | [local_df.py:build_G_tilde](pyscf/cc/dlpno_tccsd/local_df.py) | (nocc, nocc) |

After Phase I, we'll know:
- Which functions match structurally (no port needed, like `compute_ladder`).
- Which have algorithmic divergence to fix.
- The size of remaining algorithmic gap to Psi4 separate from LMO/PNO basis differences.

### Phase II — C-kernel rewrites in `pyscf/lib/cc/dlpno_*.c`

**Why next:** PySCF style alignment. Removes the 3 scatter-back compromises. Sets up the kernel signatures that allow Phase III's pair_lmo_idx-axis switch to be a no-op.

Estimated: **6-12 sessions** (one kernel per session, with build/test/validate).

**Migration order** (smallest first to establish pattern):

1. **`_foo_dressed_cy.pyx`** → `pyscf/lib/cc/dlpno_foo_dressed.c`
   - Smallest kernel. Single-pair input (Qma, t2_mq).
   - New signature: takes `nlmo_pair` parameter, `T_n[nlmo_pair, npno]`.
   - Caller drops the scatter-back at [lccsd.py:328](pyscf/cc/dlpno_tccsd/lccsd.py#L328).
   - Validate water-10 anchor.
2. **`_per_i_stages_cy.pyx`** → `pyscf/lib/cc/dlpno_per_i.c`
   - Stages 1-3 for T1 residual. Takes Qik, Qia, Qab, T_n.
   - Drops scatter-back at [lccsd.py:743](pyscf/cc/dlpno_tccsd/lccsd.py#L743).
3. **`_per_kl_batched_cy.pyx`** + **`_t1_fock_batched_cy.pyx`** → `dlpno_t1_residual.c` + `dlpno_t1_fock.c`
   - Drops 2 scatter-backs at [lccsd.py:933, 1063](pyscf/cc/dlpno_tccsd/lccsd.py#L933).
4. The remaining 8+ `_cy.pyx` files (be, c_tilde, cd_batched, ...) — rewrite as part of natural maintenance, not blocking Phase III.

**PySCF C-kernel template** (study `pyscf/lib/cc/ccsd_pack.c` and `ccsd_t.c`):
- `void function_name(double *out, double *in1, ...)` C signature.
- OpenMP `#pragma omp parallel for` for parallelism.
- BLAS via `dgemm_` extern.
- `pyscf/lib/cc/CMakeLists.txt` adds the source to the `cc` library target.
- Caller: `_ccsd.libcc.function_name(out.ctypes.data_as(...), ...)` via `ctypes`.

### Phase III — pair_lmo_idx-axis storage flip

**Why last:** with C kernels accepting `nlmo_pair` directly, the storage flip becomes a no-op. We'll have already audited consumer behavior against Psi4 in Phase I, so any remaining drift is a fixable algorithmic divergence (not a tools-mismatch).

Estimated: **1-2 sessions** after Phase II is done.

Steps:
1. Change `compute_cc_integrals_sparse` to project to pair_lmo_idx-axis (the same code path I tried in this session, lines around local_df.py:840 — the diff is in `git stash` or commit history if needed).
2. Update consumers to use `pair_lmo_idx[key]` directly (no translation needed).
3. Add the construction fix: `pair_lmo_idx[(i,j)]` always contains i, j (even for negligible pairs).
4. Validate water-4 + water-10 anchors.
5. Measure water-22 memory savings.

## Side track — accuracy investigation

The 0.6 mEh CCSD gap to Psi4 (water-4: -304.989798 ours vs -304.990365 Psi4-PM) is **not** caused by the storage axis or pair_lmo_idx-vs-p_lmos question. It's a separate algorithmic divergence (LMO/PNO basis, possibly the `X_pno` migration tracked in `project_dlpno_xpno_migration.md`). Phase I cross-code validation should localize which function diverges.

## What lives in the tree as debug aids

`compute_cc_integrals_sparse` end ([local_df.py](pyscf/cc/dlpno_tccsd/local_df.py)):
- `DLPNO_ZERO_EXTRA_LMOS=Qma,K_bar_chem,...` — zero p_lmos\pair_lmo_idx rows in selected fields. Useful for bisecting "where does data loss come from".
- `DLPNO_DEBUG_OOD=1` — counts None firings in `get_local_K`/`get_local_ovL`/`get_local_ooL_vec`. Useful for spotting helper bugs in axis-switch attempts.

`compute_B_tilde` and `compute_ladder` ([local_df.py](pyscf/cc/dlpno_tccsd/local_df.py)):
- `DLPNO_DUMP_BTILDE=1`, `DLPNO_DUMP_LADDER=1` — env-gated per-pair printf dumps for cross-code parity checks.

Diff scripts in `/tmp`:
- `/tmp/diff_btilde.py`, `/tmp/diff_btilde_self.py`, `/tmp/btilde_invariants.py`, `/tmp/diff_ladder.py`.

## Out of scope

- The 0.6 mEh accuracy gap (separate, see `project_dlpno_xpno_migration.md`).
- (T) triples — separate handoff doc `HANDOFF_TRIPLES_FLOPS_RESTRUCTURE.md`.
- Localization differences (Psi4-Boys vs ours-PM) — pick one and stick with it for cross-code validation runs.

## Anchor

`E_TCCSD = -2.13088299002` water-10 cc-pVDZ TightPNO PySCF-PM. Must hold to 11 digits at every checkpoint.
