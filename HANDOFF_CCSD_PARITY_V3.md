# DLPNO-CCSD Psi4 Parity — Handoff v3

Follow-up to `HANDOFF_CCSD_PARITY_V2.md` (v2 handoff). This session
landed one small tensordot-swap commit; the next session needs to take
on the Cython nogil port of the Phase-1 per-pair bodies (the v2 "biggest
lever").

## Where we are now

- Branch: `dlpno_restructure`, HEAD = `8e7f4bca9` "DLPNO-CCSD: Cython
  nogil port of foo_dressed_local._per_pair" (the v3 pilot).
- Prior relevant commits: `111d811d7` (einsum→tensordot swap),
  `80c74f8f2` (this handoff doc — now updated with lessons learned).
- New correctness anchor (bit-equivalent to v2's): **E_tccsd =
  −2.13088299002238** on water10 / cc-pVDZ / Jiang TightPNO.
- Wall time on this box: **CCSD = 128–130s** (two samples: 128.54s,
  129.22s). v2 handoff reported 126.12s — 3–4s higher here, well
  within machine noise. v2 baseline was 138.61s on my first run, so
  the tensordot change is worth ~2s of attributable win plus sub-timer
  evidence: foo=0.29→0.20 and G=0.52→0.48 per cycle (verified across
  steady-state cycles 4–14).

## Pilot port validated (commit 8e7f4bca9)

**The simple "per-pair Cython nogil kernel behind pool.map" pattern
works.** foo_dressed was ported as the pilot:

- Wrote `_foo_dressed_cy.pyx` with one function that takes typed
  memoryviews for one pair's inputs, releases the GIL, and does the
  per-pair math as hand-rolled loops. No plan builder, no prange
  across pairs — the Python ThreadPoolExecutor.pool.map provides
  cross-pair parallelism exactly as before.
- Microbench per-pair showed only 1.2–1.75× speedup vs NumPy, but
  **end-to-end foo sub-timer collapsed 0.20 → 0.06 s/cycle (3.3×)**
  because releasing the GIL let the pool actually scale instead of
  serializing on numpy internals. CCSD wall 129 → 124.7 s (−4.5s
  real signal, not noise).

**Key lesson:** don't over-engineer the first port. Plan builder +
bucketed prange is not required for a meaningful win. A plain
`with nogil` block around hand-rolled loops in a per-pair function is
enough — the win is mostly from GIL release, not cache locality or
FLOP reduction.

**Correctness anchor (new):** `-2.13088299002219`; preserved to 10+
digits vs prior anchors. 6 synthetic-shape tests in `/tmp/test_foo_cy.py`
pass at 3e-12 max diff (FP reordering noise).

## First actions when you pick this up

1. Read `memory/project_dlpno_ccsd_perf_session.md` for the running
   change log (row 10 is this session's commit).
2. Run the profile to confirm baseline:
   ```bash
   /environments/miniconda3/envs/tmc/bin/python -u /tmp/profile_dlpno.py
   ```
   Expect: **CCSD ≈ 128–130s, E_tccsd = −2.13088299…**
3. Confirm these sub-timers in the cycle lines:
   ```
   foo=0.18–0.22 jiang=2.6–3.0(C=0.8–1.0 D=0.9–1.1 ... G=0.48) pairs=1.2–1.3
   ```
   If `foo ≥ 0.27` or `G ≥ 0.51`, the tensordot commit got reverted —
   check `git log --oneline -5`.

## Next lever: Cython nogil port of Phase 1 per-pair bodies

### Priority order (by cum time in cProfile)

Top targets:
1. `residual.py:879 _process_ik_t12` (D_tilde Phase 1) —
   `build_D_tilde_batched` tottime = 10.9s, cum ≈ 15s.
2. `residual.py:1607 _process_ki_terms12` (C_tilde Phase 1) —
   `compute_C_tilde_batched` tottime = 10.7s, cum ≈ 15s.
3. `local_df.py:993 t1_fock._per_pair` — ~4s cum.
4. `lccsd.py:318 _compute_foo_dressed_local._per_pair` — ~4s cum.
   Note: foo's einsum was swapped this session; remaining cost is
   matmul + tensordot dispatch.

### Design for `_process_ik_t12` (biggest target)

Per-pair work when `cc_ints` covers:

```
Term 2 (3 ops on (n_pno, n_pno)):
  z_Qa = tensordot(Qab_ki, t1_i_ik, axes=(2, 0))       # (n_local, n_pno)
  D   += 2 * z_Qa.T @ k_Qa                              # (n_pno, n_pno)
  w    = k_Qa @ t1_i_ik                                 # (n_local,)
  D   -= tensordot(w, Qab_ki, axes=(0, 0)).T            # (n_pno, n_pno)

Term 1 (full pair-domain sum, 3 ops on (n_domain, n_pno)):
  ooL_il = i_Qk[:, ll_idx]       or j_Qk[:, ll_idx]     # (n_local, n_domain)
  ooL_ik = i_Qk[:,  k_idx]       or j_Qk[:,  k_idx]     # (n_local,)
  ovL_k  = Qma[:, k_idx, :]                              # (n_local, n_pno)
  ilkc   = ooL_il.T @ ovL_k                              # (n_domain, n_pno)
  iklc   = tensordot(Qma[:, ll_idx, :], ooL_ik, (0, 0)) # (n_domain, n_pno)
  M      = 2 * ilkc - iklc
  D     -= T1_all[ll_idx].T @ M                          # (n_pno, n_pno)
```

For water10: n_pno ≈ 25, n_local ≈ 168, n_domain ≈ 15. Per-pair total
~450k FLOPs × 820 pairs × 14 iters = 5.2 Gflops. Currently ~1.05 s/iter
on 8 workers — ≈15s/run in 30× slowdown from Python/numpy dispatch.

### Shapes are heterogeneous — two port strategies

From `/tmp/shape_scan.py` (ran this session):
- 820 active pairs, 149 distinct `(n_local, nocc, n_pno)` tuples for Qma.
- `n_local` has only 18 distinct values in [84, 406].
- `n_pno` has 33 distinct values in [6, 46].
- Top 15 buckets have 16–34 pairs each (uneven tail).

**Strategy A (bucketed prange)** — follow `_c_tilde_cy.pyx` template.
Build plan once per run grouping pairs by `(n_pno, n_local, n_domain)`;
allocate uniform `(N, ...)` stacked arrays per bucket; prange over
bucket's N items with hand-rolled matmul loops. **Risk**: many small
buckets (149+), each below 64-way parallel threshold — may not saturate
the cores.

**Strategy B (heterogeneous prange over all pairs)** — single big
kernel with per-pair size/offset arrays. Each thread reads its pair's
shapes and pointers and does the work at those sizes. **Risk**:
more complex; requires raw-pointer arithmetic under `nogil`; harder to
debug.

Recommend starting with Strategy A for the first pair body to prove
the pattern matches existing infrastructure, then revisit B if bucket
fragmentation kills parallelism.

### Infrastructure already in place

- `pyscf/cc/dlpno_tccsd/pair_index.py:269 FlatTensorStore` — per-pair
  tensors in a single flat buffer with `(n_pairs, rank)` shape table
  and `(n_pairs+1)` offsets. The `buffer`, `offsets`, `shapes`
  attributes are exactly what a Cython kernel needs.
- `pyscf/cc/dlpno_tccsd/lccsd.py:1338 flatten_cc_ints_fields(...)`
  builds the flat store for all per-pair DF fields. `_cc_ints_flat`
  is built once at CCSD entry and survives every cycle.
- cc_ints fields relevant here, all full-nocc padded (verified via
  shape scan; the "reduced" comments at local_df.py:874-876 are stale):
  - `'Qab'`: (n_local, n_pno, n_pno)
  - `'Qma'`: (n_local, nocc, n_pno)
  - `'i_Qk'`, `'j_Qk'`: (n_local, nocc)
  - `'i_Qa'`, `'j_Qa'`: (n_local, n_pno)
- Cython .pyx templates: `_c_tilde_cy.pyx`, `_cd_cy.pyx`, `_be_cy.pyx`.
  These all use the bucket-with-prange pattern.

### What NOT to retry (already failed in v2)

See `HANDOFF_CCSD_PARITY_V2.md` § "What NOT to retry". Short version:
overlap G with bt/be/cd, parallelize build_G_tilde outer-i, inline
compute_CD_terms_batched, shared T1_cache wrapper, fine_pool sweep
4/8/16/32.

### Suggested session scope (updated after foo pilot)

The pilot proved the simple serial-per-pair nogil pattern wins (3.3×
on the ported phase). Next session should follow the same pattern for
the remaining bodies — **skip the batched/bucketed design unless the
simple pattern stalls**.

1. Pick next target. Ordered by expected win (sub-timer × ~3×):
   - `t1_fock._per_pair` (Fock=0.26/cycle → ~0.08, ~2.5s/run).
     More math than foo: `gamma`, `Y`, `Fia_bar`, `Fab` chain — write
     carefully with standalone bit-exact test first.
   - `compute_C_tilde_batched._process_ki_terms12`
     (C=0.85-1.05/cycle → ~0.30, ~8-10s/run). Biggest target, but
     Phase 1 is only ~40% of total C — see commit 8e7f4bca9 memory
     note for the `phase1_t12=370ms / tot=955ms` breakdown. Port just
     the Phase-1 body, not the whole thing.
   - `build_D_tilde_batched._process_ik_t12` (D=0.94-1.29/cycle →
     Phase 1 portion similar to C). Has the most complex math of the
     four (two terms, fancy-index gathers); do it last.

2. Use foo as the template: new `_<name>_cy.pyx`, single function taking
   typed memoryviews for one pair. Release GIL via `with nogil`. No
   plan builder.

3. Build pipeline:
   ```bash
   cd /home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd
   /environments/miniconda3/envs/tmc/bin/python setup.py build_ext --inplace
   cp _<name>_cy.cpython-312-x86_64-linux-gnu.so \
      /environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/
   cp lccsd.py (or residual.py) to the installed path too.
   ```

4. Validate before running CCSD: `/tmp/test_<name>_cy.py` on 4-6
   synthetic shapes, asserting `max(|ref - cy|) < 1e-11`. Only when
   green, run `/tmp/profile_dlpno.py` and check E_tccsd.

5. Correctness anchor: **-2.13088299…** to 10+ digits.

Total projected savings (across 3 remaining targets): ~12-15 s —
brings CCSD to ~110s, under 8× Psi4. The v2 projection of 25–30s was
optimistic; the per-pair-serial approach misses some parallelism that
a true batched prange kernel would capture, but the engineering cost
is much lower.

## Build sync caveat (from v2, still applies)

```
cd /home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd
python setup.py build_ext --inplace
cp _*.cpython-312-x86_64-linux-gnu.so \
   /environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/
```
Plus `cp lccsd.py` and `cp residual.py` to the installed package path
when editing their dev sources.

## Correctness anchor (unchanged from v2)

**E_tccsd = −2.13088299…** on water10 / cc-pVDZ / TightPNO. Preserve
to 10+ digits unless the change is an intentional semantic match with
Psi4 (document in a commit message + memory note with revert recipe).

## Useful artifacts

- `/tmp/profile_dlpno.py` — profile runner (unchanged from v2).
- `/tmp/profile_v2_baseline.log` — baseline this session (CCSD 138.61s).
- `/tmp/profile_v2_step2_verify.log` — after tensordot commit
  (CCSD 129.22s).
- `/tmp/shape_scan.py` — shape distribution scan (reusable).
- `memory/project_dlpno_ccsd_perf_session.md` — full change log.
- `HANDOFF_CCSD_PARITY_V2.md` — v2 handoff (bigger-picture framing).
