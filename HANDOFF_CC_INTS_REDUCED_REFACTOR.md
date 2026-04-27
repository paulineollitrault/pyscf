# DLPNO-CCSD: cc_ints reduced-axis refactor (linear-scaling memory)

**Goal:** Switch `compute_cc_integrals_sparse` to store occupied-axis tensors
(`Qma`, `i_Qk`, `j_Qk`, `K_mnij`, `K_bar_ij`, `K_bar_ji`, `K_bar_chem`) in their
**reduced** `(*, nlmo_p, *)` shape — only the pair's own LMO domain — instead
of scattering to full `(*, nocc, *)`. Matches Psi4 storage. Restores DLPNO
linear scaling on the per-pair tensors (currently O(N) per pair × O(N) pairs
= O(N²) total; after refactor O(1) per pair × O(N) pairs = O(N)).

**Memory impact (water-22 ≈ 66 atoms, nocc=80, mean nlmo_p≈25, ~1500 pairs):**
- Per-pair `Qma`: `(n_local=200, nocc=80, npno=25)` × 8 B = 3.2 MB → `(200, 25, 25)` = 1 MB. **Saves ≈ 3.3 GB on Qma alone.**
- Same ~3× factor on `i_Qk`, `j_Qk`, `K_mnij`, `K_bar_*`.
- Total savings: **≈ 15-25 GB at water-22**, restoring linear scaling.

**⚠️ CRITICAL CORRECTION (verified empirically 2026-04-27):**
`pair_lmo_idx[key] ⊂ p_lmos[key]` — NOT equal. Diagnostic on water-4:
```
key=(0, 9) pair_lmo_idx=[0,1,2,3,4,5,6,8,9,10,11,12,14]
          p_lmos     =[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
```
`p_lmos` is the union of riatom_to_lmos_ext over the pair's aux centers
(line 644-650 of local_df.py); `pair_lmo_idx` is Psi4's
`lmopair_to_lmos_[ij]` — the pair's domain. p_lmos is typically a
SUPERSET (~25% larger on water-4). Consumers that previously did
`tensor[pair_lmo_idx]` on a full-nocc tensor cannot just "drop the
slicing" with reduced storage — they must translate via
`tensor[ci['p_lmos_dense'][pair_lmo_idx]]`. **First attempt at this
refactor (2026-04-27) failed validation** by 7.4 mEh on water-4
because of this confusion (assumed equality, dropped slicing); revert
hash is in `git reflog`.

**⚠️ FUNDAMENTAL REVISION (2026-04-27 after studying Psi4 ccsd.cc:1183-1230):**
**The reduction target axis should be `pair_lmo_idx` (Psi4's
`lmopair_to_lmos_[ij]`), NOT our extended union `p_lmos`.** Psi4
literally allocates `q_io = Matrix(naux_ij, nlmo_ij)` where
`nlmo_ij = lmopair_to_lmos_[ij].size()`. They use a per-pair mapping
table `lmopair_lmo_to_riatom_lmo_[ij][q_ij]` to resolve sparse
per-aux indices to the pair-domain LMO positions when filling.

| Storage axis | water-4 size | water-22 size |
|---|---|---|
| nocc (current) | 16 | 80 |
| p_lmos (= riatom union) | 16 | ~30 |
| **pair_lmo_idx (Psi4 target)** | **13** | **~20** |

Per-pair `Qma` memory at water-22: `200×80×25×8 = 3.2 MB` (current) →
`200×20×25×8 = 0.8 MB` (target). **4× smaller AND bounded as system
grows** — pair domain stays ~constant in DLPNO; nocc grows linearly.

This is the real linear-scaling story.

**⚠️ EMPIRICAL FINDING (2026-04-27, second attempt):**
Just changing the storage to use `pair_lmo_idx` semantics WITHOUT a
coordinated algorithmic restructure breaks correctness by **137 mEh on
water-4** (E_corr -0.7153 vs anchor -0.8525). Our current consumers
sum over the wider extended-union LMO set; restricting the storage to
pair_lmo_idx-only drops contributions they expected to find.

**Conclusion: the storage refactor is NOT a self-contained change.**
To match Psi4's per-pair memory layout, every consumer's loop bounds
must ALSO restrict to `pair_lmo_idx`. This is an *algorithmic* port
of the residual + dressed-Fock builds, not just a memory layout fix.
Concretely, look at Psi4 ccsd.cc:
- `compute_F_pair_pair` (lines 86-260) — every k loop uses
  `lmopair_to_lmos_dense_[ij][k]` to skip out-of-domain
- `compute_F_pno` (lines 1180-1430) — per-pair tensors built only on
  `nlmo_ij` axis from the start; consumers index by `k_ij` (pair-domain
  position), never by global `k`
- The whole CCSD residual (lines 1500-2500) loops `for k_ij in
  range(nlmo_ij)` and reads `Qma_ij_[ij][q_ij](k_ij, ...)` — all
  pair-domain indexed throughout

To get linear scaling, **port the residual algorithm to pair_lmo_idx
loop bounds first**, THEN the storage change becomes a no-op (consumers
already won't ask for out-of-domain rows). This is a multi-week
algorithmic rewrite, not a memory refactor.

**Scope:** ~30 consumer sites + 4 Cython kernel signature changes. Not big-bang
risky if done in the order below — each step is independently verifiable.

**Anchor energy:** water-10/cc-pVDZ/TightPNO `E_TCCSD = -2.13088299002...`
(must hold to 10 digits at every checkpoint). Validation via:
```bash
cd /home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling
/environments/miniconda3/envs/tmc/bin/python run_water_scaling.py --chains 4 --force      # ~10s, fast iteration
/environments/miniconda3/envs/tmc/bin/python run_water_scaling.py --chains 10 --force     # ~3 min, anchor check
```

---

## Current state (HEAD = `f48e9d4` or wherever the gather/G_term commits land)

`compute_cc_integrals_sparse` ([local_df.py:572-908](pyscf/cc/dlpno_tccsd/local_df.py#L572)) builds reduced
`q_io_red`, `q_jo_red`, `Qma_red` then immediately scatters them to full-nocc
`q_io`, `q_jo`, `Qma` ([line 828-839](pyscf/cc/dlpno_tccsd/local_df.py#L828))
"so downstream consumers continue to work unchanged". `K_mnij`/`K_bar_*`/
`K_bar_chem` are then computed from the scattered full versions.

The dict returned at [line 879-895](pyscf/cc/dlpno_tccsd/local_df.py#L879) has
**inaccurate comments** that say `(nlmo_p, ...) reduced` but the actual
storage is `(nocc, ...) full`. Fix the storage, the comments will be right.

---

## Refactor plan (10 steps, in order)

### Step 1 — Storage change in compute_cc_integrals_sparse

**File:** [pyscf/cc/dlpno_tccsd/local_df.py:828-845](pyscf/cc/dlpno_tccsd/local_df.py#L828)

Replace:
```python
q_io = np.zeros((n_local, nocc))
q_jo = np.zeros((n_local, nocc))
Qma = np.zeros((n_local, nocc, npno))
q_io[:, p_lmos] = q_io_red
q_jo[:, p_lmos] = q_jo_red
Qma[:, p_lmos, :] = Qma_red

K_iajb = q_iv.T @ q_jv
K_mnij = q_io.T @ q_jo                    # (nocc, nocc) full
K_bar_ij = q_io.T @ q_jv
K_bar_ji = q_jo.T @ q_iv
K_bar_chem = np.tensordot(q_pair, Qma, axes=(0, 0))
```

With:
```python
q_io = q_io_red                           # (n_local, nlmo_p)
q_jo = q_jo_red                           # (n_local, nlmo_p)
Qma = Qma_red                             # (n_local, nlmo_p, npno)

K_iajb = q_iv.T @ q_jv
K_mnij = q_io.T @ q_jo                    # (nlmo_p, nlmo_p) reduced
K_bar_ij = q_io.T @ q_jv                  # (nlmo_p, npno) reduced
K_bar_ji = q_jo.T @ q_iv                  # (nlmo_p, npno) reduced
K_bar_chem = np.tensordot(q_pair, Qma, axes=(0, 0))  # (nlmo_p, npno) reduced
```

The cross-pair partner loop ([line 856-877](pyscf/cc/dlpno_tccsd/local_df.py#L856)) already does
`k_loc = int(p_lmos_dense[k]); q_ik = q_io[:, k_loc]` — this NOW correctly
indexes the reduced axis (it was a latent bug for full-nocc storage; the
indexing only worked because columns outside p_lmos were zero so the
"wrong" lookup happened to return zero from a zero column).

**After this step, every consumer reading any of `Qma`, `i_Qk`, `j_Qk`,
`K_mnij`, `K_bar_ij`, `K_bar_ji`, `K_bar_chem` from cc_ints will return
reduced-shape arrays.** Steps 2-9 fix each consumer.

## Translation pattern (memorize this)

For every consumer that did `tensor[lmo_idx]` on a full-nocc tensor where
`lmo_idx = pair_lmo_idx[key]`, the equivalent on the reduced (nlmo_p)
tensor is:
```python
ll_red = np.asarray(ci['p_lmos_dense'])[lmo_idx]
# ll_red has -1 entries if lmo_idx contains LMOs not in p_lmos.
# In practice pair_lmo_idx ⊂ p_lmos so ll_red is all >= 0; assert this
# in dev mode for sanity:
#   assert (ll_red >= 0).all(), f"pair_lmo_idx \ p_lmos = {lmo_idx[ll_red<0]}"
out = tensor[ll_red]
```

For SINGLE global LMO indexing `tensor[:, k, :]`:
```python
k_red = int(ci['p_lmos_dense'][k])
if k_red < 0:
    # k not in p_lmos — old form gave zeros, return zero/skip
    return np.zeros(...)
out = tensor[:, k_red, :]
```

**Validation harness diagnostic** (run at the start of any session
attempting this refactor — make sure it doesn't fire):
```python
# in compute_cc_integrals_sparse, right after p_lmos is built:
if pair_lmo_idx is not None and key in pair_lmo_idx:
    pli = set(int(x) for x in pair_lmo_idx[key])
    pl = set(int(x) for x in p_lmos)
    assert pli.issubset(pl), f"pair_lmo_idx not in p_lmos: {pli - pl}"
```

## Refactor steps (in order)

### Step 2 — `_foo_dressed_cy` Cython kernel

**File:** [pyscf/cc/dlpno_tccsd/_foo_dressed_cy.pyx](pyscf/cc/dlpno_tccsd/_foo_dressed_cy.pyx) — kernel takes `Qma` of shape `(n_local, nocc, n_pno)` and
indices `m`, `q` as global LMOs.

**Caller:** [lccsd.py:308-353](pyscf/cc/dlpno_tccsd/lccsd.py#L308) `_compute_foo_dressed_local`

After Step 1, `Qma.shape[1]` is `nlmo_p` not `nocc`. The caller's `m`, `q` are
global pair LMOs (key_mq = (m, q)).

**Fix the caller:**
```python
def _per_pair(key_mq):
    ...
    ci = cc_ints.get(key_mq)
    if ci is None: return key_mq, None
    m, q = key_mq
    Qma = ci['Qma']                          # (n_local, nlmo_p, npno)
    p_lmos = ci['p_lmos']                    # array of global LMO indices
    p_lmos_dense = ci['p_lmos_dense']        # nocc-sized inverse map
    nlmo_p = Qma.shape[1]
    m_red = int(p_lmos_dense[m])
    q_red = int(p_lmos_dense[q])
    if m_red < 0 or q_red < 0:
        return key_mq, None  # shouldn't happen; pair LMOs are in domain

    contrib_q_red = np.zeros(nlmo_p)
    if m != q:
        contrib_m_red = np.zeros(nlmo_p)
        foo_dressed_one(Qma, t2_mq_raw, m_red, q_red,
                        contrib_q_red, contrib_m_red)
    else:
        contrib_m_red = None
        foo_dressed_one(Qma, t2_mq_raw, m_red, q_red,
                        contrib_q_red, contrib_q_red)
    # Caller now scatters reduced contrib back to full-nocc foo:
    return key_mq, (contrib_q_red, contrib_m_red, p_lmos)

# In the aggregator:
for key_mq, payload in results:
    if payload is None: continue
    m, q = key_mq
    contrib_q_red, contrib_m_red, p_lmos = payload
    foo[p_lmos, q] += contrib_q_red
    if contrib_m_red is not None:
        foo[p_lmos, m] += contrib_m_red
```

The Cython kernel does NOT need signature change — its `nocc` parameter is
just a loop bound, and the caller now passes the reduced size implicitly via
`Qma.shape[1]`. The kernel writes to `out_q` of size `nlmo_p` instead of
`nocc`. **Verify** by running water-4: `E_TCCSD` must match anchor.

### Step 3 — `t1_fock_batched` Cython + Python wrapper

**File:** [local_df.py:t1_fock](pyscf/cc/dlpno_tccsd/local_df.py#L1014) +
[_t1_fock_batched_cy.pyx](pyscf/cc/dlpno_tccsd/_t1_fock_batched_cy.pyx)

The wrapper at [line 1095](pyscf/cc/dlpno_tccsd/local_df.py#L1095) does:
```python
Qma_list[p] = np.ascontiguousarray(ci['Qma'][:, lmo_idx, :])
```
where `lmo_idx = pair_lmo_idx[key]`. After Step 1, `ci['Qma']` is already
reduced and `pair_lmo_idx[key] == p_lmos` by construction (verified in
[line 644-649](pyscf/cc/dlpno_tccsd/local_df.py#L644)). So:

```python
Qma_list[p] = np.ascontiguousarray(ci['Qma'])  # already reduced; no slicing
```

Kernel signature unchanged. Same for K_chem/K_ji/K_ij at [lines 1083-1085](pyscf/cc/dlpno_tccsd/local_df.py#L1083): drop the `[lmo_idx]` slicing.

### Step 4 — `_t1_fock` Eq 94 inner loop ([local_df.py:1235](pyscf/cc/dlpno_tccsd/local_df.py#L1235))

```python
Qma_jj = ci['Qma'][:, lmo_idx, :]
```
becomes
```python
Qma_jj = ci['Qma']  # already reduced
lmo_idx = ci['p_lmos']  # for the Fkj scatter at the end
```

The Fkj scatter `Fkj[lmo_idx, j_idx] += ...` already uses lmo_idx == p_lmos.

### Step 5 — `compute_B_tilde` ([local_df.py:1283-1346](pyscf/cc/dlpno_tccsd/local_df.py#L1283))

```python
lmo_idx = pair_lmo_idx[key]   # global LMO indices
nlmo = len(lmo_idx)
Qma = ci['Qma'][:, lmo_idx, :]           # (n_local, nlmo, npno)
i_Qk = ci['i_Qk'][:, lmo_idx]            # (n_local, nlmo)
j_Qk = ci['j_Qk'][:, lmo_idx]
```
becomes
```python
lmo_idx = ci['p_lmos']
Qma = ci['Qma']
i_Qk = ci['i_Qk']
j_Qk = ci['j_Qk']
nlmo = Qma.shape[1]
```

The final scatter `B_tilde[np.ix_(lmo_idx, lmo_idx)] = B_local` works
unchanged because lmo_idx == p_lmos.

### Step 6 — `compute_ladder` (similar; same file)

Same pattern — drop `[lmo_idx]` slicing on Qma. Currently
[local_df.py:1397-1410](pyscf/cc/dlpno_tccsd/local_df.py#L1397).

### Step 7 — Fij_bar precompute in T1 residual ([lccsd.py:670-681](pyscf/cc/dlpno_tccsd/lccsd.py#L670))

```python
Fij_bar[i0, j0] += (2.0 * np.sum(T_n_ij_mat * ci_ij['K_bar_chem'])
                    - np.sum(T_n_ij_mat * ci_ij['K_bar_ji']))
```

`T_n_ij_mat` is `(nocc, n_pno)` (from t1_cache). `K_bar_chem` and
`K_bar_ji` are NOW `(nlmo_p, n_pno)` reduced (vs previously `(nocc, n_pno)`
with zeros). The shapes don't match, so this errors out.

**Fix:**
```python
p_lmos_ij = ci_ij['p_lmos']
T_n_red = T_n_ij_mat[p_lmos_ij]      # gather only the rows in pair domain
Fij_bar[i0, j0] += (2.0 * np.sum(T_n_red * ci_ij['K_bar_chem'])
                    - np.sum(T_n_red * ci_ij['K_bar_ji']))
```

Same for the j0,i0 block (uses K_bar_ij). Sum is mathematically identical
because rows outside p_lmos contributed 0 in the old form.

### Step 8 — `_per_i` Stage 1 in T1 residual ([lccsd.py:744-790](pyscf/cc/dlpno_tccsd/lccsd.py#L744))

```python
Qma_ii = ci_ii['Qma']      # (n_local, nocc, n_ii)
Qab_ii = ci_ii['Qab']
Qia_ii = ci_ii['i_Qa']
Qik_ii = ci_ii['i_Qk']     # (n_local, nocc)
T_n_ii = t1_cache[key_ii]  # (nocc, n_ii)
```

After Step 1: `Qma_ii.shape[1] == nlmo_p` (not nocc). `T_n_ii` is still
nocc. The reductions in Stage 1 (gamma = Qma . T_n_ii.ravel()) require
matching shapes.

**Fix:** reduce T_n_ii to pair domain:
```python
p_lmos_ii = ci_ii['p_lmos']
T_n_ii_red = T_n_ii[p_lmos_ii]   # (nlmo_p, n_ii)
# Pass T_n_ii_red to the Cython kernel and to all subsequent matmuls
```

The Cython kernel `_per_i_stages_cy.per_i_stages123` takes `T_n: (M, A)` —
M is just a loop bound; pass nlmo_p instead of nocc, and the kernel writes
`Fia[mp, a]` of size nlmo_p instead of nocc. Stage 4 does
`r1_i -= Fij_bar[:, i] @ T_n_ii` — needs the full T_n still:
```python
r1_i -= Fij_bar[:, i] @ T_n_ii   # T_n_ii here is the FULL nocc-shaped t1_cache row
                                  # (not the reduced T_n_ii_red passed to the kernel)
```

### Step 9 — `_per_kl` in T1 residual ([lccsd.py:933-934, 1068-1069](pyscf/cc/dlpno_tccsd/lccsd.py#L933))

```python
K_bar_kl = ci_kl['K_bar_ij'] if key_kl[0] == k else ci_kl['K_bar_ji']
```

Now reduced `(nlmo_p, n_pno)`. Consumer math: `K_kilc = K_bar_kl + T_n_kl @ K_iajb_kl` — T_n_kl is nocc-shaped from t1_cache. Need:
```python
T_n_kl_red = t1_cache[key_kl][ci_kl['p_lmos']]
K_kilc_red = K_bar_kl + T_n_kl_red @ K_iajb_kl  # shape (nlmo_p, n_pno)
B_ia_red = Tt_kl @ K_kilc_red.T                  # (n_pno, nlmo_p)
# Inner per-i loop indexes B_ia[:, i] — i is a global LMO. Translate:
i_red = ci_kl['p_lmos_dense'][i]
B_contrib = -B_ia_red[:, i_red]   # ... etc
```

### Step 10 — Batched view caches I added (residual.py)

Each plan-cache builder concatenates per-item `S_big`/`S_mid`/`J_bold`/etc.
sourced from `S_pno_cache.get((key_a, key_b))`. Those S matrices are NOT in
cc_ints, so this step is *unaffected* by the reduced cc_ints refactor.

But the per-item gather of `K_bar_chem_slice` in
[compute_C_tilde_batched](pyscf/cc/dlpno_tccsd/residual.py#L1942) and
[build_D_tilde_batched](pyscf/cc/dlpno_tccsd/residual.py) does
`ci_ki['K_bar_chem'][ll_idx]` where ll_idx == pair_lmo_idx. After Step 1,
that's reading from a reduced (nlmo_p, n_pno) buffer. Indexing must change:

```python
# Before:
K_bar_chem_slice = ci_ki['K_bar_chem'][ll_idx]   # ll_idx is global LMOs
# After:
K_bar_chem_slice = ci_ki['K_bar_chem']           # already reduced; ll_idx == p_lmos
```

Verify: `pair_lmo_idx[key_ki] == ci_ki['p_lmos']` — they should be equal by
construction (the pair_lmo_idx feeds into the cc_ints build at
[local_df.py:644-649](pyscf/cc/dlpno_tccsd/local_df.py#L644)).

---

## Validation strategy

### Per-step verification (recommended order)

After each Step (1 thru 9), run:
```bash
cp /home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/<modified_files> \
   /environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/
cd /home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling
/environments/miniconda3/envs/tmc/bin/python run_water_scaling.py --chains 4 --force
```

Water-4 takes ~15s and exercises all the per-pair code paths. Watch for:
- `E_TCCSD = -304.98979787...` (or whatever the exact anchor is from a known-good baseline)
- Convergence in ≤16 cycles
- No `IndexError`, `ValueError` from shape mismatches

After ALL steps complete, validate full anchor:
```bash
/environments/miniconda3/envs/tmc/bin/python run_water_scaling.py --chains 10 --force
# Expect E_TCCSD = -2.13088299002... (10 digits)
```

Then run scaling scan to verify memory + scaling improvement:
```bash
/environments/miniconda3/envs/tmc/bin/python run_water_scaling.py --chains 4,8,10,15 --force
# Watch RAM with `top` in another terminal
```

### Rollback

```bash
git stash       # if local changes
git revert HEAD # if committed
```

The refactor is **all in compute_cc_integrals_sparse + ~10 consumer files**.
No Cython kernel signatures change (they're shape-agnostic — the storage
shape just becomes smaller). Pure Python wrappers + scatter/gather logic.

---

## Suggested implementation order (revised after failed attempt)

The 2026-04-27 attempt tried to do all 10 steps in one pass and revert
because of confusing `pair_lmo_idx vs p_lmos` semantics. Better strategy:

1. Add `assert ll_red.min() >= 0` everywhere translation is needed
   so any `pair_lmo_idx \ p_lmos` mismatch fails loudly.
2. Do Step 1 (storage change) AND Step 2 (foo_dressed) AND Step 5
   (compute_B_tilde) together — these are tightly coupled and the
   foo_dressed math is independently testable.
3. Validate water-4 — should match `-304.98979787` to 1e-7 Eh.
4. Then add Steps 3, 4, 6, 7, 8, 9 one at a time, validating after each.
5. Step 10 (residual.py batched paths) last — most complex, most
   surfaces to check.
6. Run scaling scan (water-4..15) at the end — should see ~3× memory
   savings on cc_ints, scaling exponent moves toward Psi4's 1.81 from
   our current 2.32.

## Out of scope

- (T) triples (`lccsd_t.py`) — does not directly access `ci['Qma']` etc.
  per the grep above. Safe.
- `compute_B_E_batched_v2` — uses `S_pno_cache` and pre-built per-pair
  intermediates. Unaffected.
- `compute_G_term_batched` — same, uses S_pno_cache + G_tilde + t2.

## Memory savings I added that should ALSO be addressed (separate, smaller fixes)

1. **`_per_kl_batched._S_consolidated`** ([lccsd.py:~1090-1110](pyscf/cc/dlpno_tccsd/lccsd.py)) — duplicates `S_pno_cache._buffer`. Saves 5-10 GB at water-22. Fix: pass two buffers (main + overflow) + per-item buffer-selector flag instead of concatenating.
2. **Batched view static-tensor concatenation** in `_cd_batched_view`,
   `_t34_batched_view`, `_g_term_batched_view` — concatenate per-item S/K/J matrices into one big buffer. Each ~GBs at water-22. Fix: index into source buffers (S_pno_cache._buffer, cc_ints flat stores) directly via offset arrays instead of duplicating.

These are independent of the cc_ints reduced refactor and can be done in
parallel.

---

## Expected outcome

- water-22 RAM: 67 GB → ~25-35 GB (and falling per pair as size grows)
- water-22 wall: should drop ~30% from less memory pressure (cache misses)
- Scaling exponent: ~2.32 (current) → closer to 1.8 (Psi4's), maybe 2.0
- Bigger systems become tractable (water-49, water-64 with our current
  storage probably OOMs, freeing memory should let them run)

True linear scaling (~N^1.0) requires more than this refactor — it also
needs aggressive pair screening + true sparse local-aux storage. But this
refactor closes the biggest gap to Psi4.
