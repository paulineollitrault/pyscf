# Handoff: implement `T_CUT_DO_TRIPLES = 1e-2` for DLPNO (T)

## One-paragraph context

We've been bringing the PySCF DLPNO-CCSD(T) (T) kernel (`pyscf/cc/dlpno_tccsd/lccsd_t.py`) structurally in line with Psi4/Jiang's `dlpno_jiang/triples.cc::compute_lccsd_t0` so that (T) scaling approaches Jiang's N_basis^1.78 on water chains. Current state: baseline was b=2.68, we're at b=2.39 (water4→water10, cc-pVDZ, TightPNO). We've verified the last threshold mismatch vs Psi4's defaults (`read_options.cc`):

- `T_CUT_MKN_TRIPLES`: **DONE** — changed 1e-3 → 1e-2 at lccsd_t.py:1131. Energies match full-naux baseline within ~15 μEh on S22-1/2/3/8.
- `T_CUT_DO_TRIPLES = 1e-2`: **TODO (this handoff)** — Psi4 rebuilds per-LMO PAO domains at the triples stage using a fresh DOI cutoff, tighter than the TightPNO CCSD stage (5e-3). This directly shrinks `riatom_to_paos_ext` and the `nu²` FLOPS bottleneck in `_build_triple_local_DF`.

## The plan

Goal: at the triples stage, replace `_pao_domains` (built from TightPNO `T_CutDO=5e-3` in `make_paos`) with tighter per-LMO PAO domains using `T_CUT_DO_TRIPLES = 1e-2` on the DOI matrix that `make_paos` already computes.

### Step 1 — return the DOI matrix from `make_paos`

File: `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/local_orbs.py`

Currently `make_paos` computes `doi_iu` (shape `(nocc_lmo, nao)`) in the `doi_method='grid'` branch (local_orbs.py:235-255) and uses it locally. Change the function signature:

```python
# line 331 — currently:
return C_pao, pao_domains, S_pao, F_pao
# change to:
return C_pao, pao_domains, S_pao, F_pao, doi_iu
```

Then in each domain branch, make sure `doi_iu` is defined (even if `None`):
- `doi_method='mulliken'` (line 197): set `doi_iu = None` — no DOI computed.
- `doi_method='grid'` (line 220): already computed. Keep as-is.
- `doi_method='pao'` (line 272): computed inside the branch. Stash it before the `del ovL_lmo_pao`.
- Löwdin fallback (line 293): set `doi_iu = None`.

### Step 2 — propagate `doi_iu` through the driver

File: `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/driver.py`

Update the call at line 366-368:

```python
C_pao, pao_domains, S_pao, F_pao, doi_iu = make_paos(
    mf_or_mc, C_lmo, T_CutDO=T_CutDO, s1e=s1e, with_df=_with_df,
    doi_method='grid')
```

Then pass `doi_iu=doi_iu` to `run_lccsd_t_ext(...)` (call at ~line 486). You'll need to check the exact line — use `grep -n "run_lccsd_t_ext" driver.py`.

### Step 3 — rebuild PAO domains in `run_lccsd_t_ext` at triples stage

File: `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/lccsd_t.py`

Add `doi_iu=None` to the signature around line 977-990 (`def run_lccsd_t_ext(...)`).

Then around line 1161 (where `_pao_domains` is built from diagonal `pair_paos`), add a Psi4-style DOI-based rebuild. The current code is:

```python
# Per-LMO PAO-domain atom set (needed by build_screening_maps).
# Derive from diagonal pairs' 'pair_paos' array (in AO index space).
_pao_domains = []
for ii in range(nocc_lmo):
    key_ii = (ii, ii)
    if (key_ii in pno_spaces
            and pno_spaces[key_ii].get('pair_paos') is not None):
        _pao_domains.append(
            np.asarray(pno_spaces[key_ii]['pair_paos']))
    else:
        _pao_domains.append(np.zeros(0, dtype=int))
```

Replace with:

```python
_T_CUT_DO_TRIPLES = 1e-2   # Psi4 read_options.cc:2575

# Psi4-style: rebuild lmo_to_paos at the triples stage using DOI cutoff.
# For each LMO i: keep PAOs u with DOI[i, u] > T_CUT_DO_TRIPLES, then
# atom-complete (if any PAO on atom A passes, include ALL PAOs on A).
# Mirrors triples.cc:316-329.
_ao_labels_tr = mf.mol.ao_labels(fmt=False)
_atom_ids_tr  = np.array([lbl[0] for lbl in _ao_labels_tr])
_atom_to_ao = [np.where(_atom_ids_tr == a)[0] for a in range(_natm)]

_pao_domains = []
if doi_iu is not None:
    for ii in range(nocc_lmo):
        doi = doi_iu[ii]                              # (nao,)
        pao_inds = np.where(doi > _T_CUT_DO_TRIPLES)[0]
        if pao_inds.size == 0:
            pao_inds = np.array([int(np.argmax(doi))])
        # Atom completion
        atoms_in = np.unique(_atom_ids_tr[pao_inds])
        domain_i = np.concatenate([_atom_to_ao[a] for a in atoms_in])
        _pao_domains.append(np.sort(domain_i))
else:
    # Fallback: old behaviour (diagonal pair_paos from CCSD stage)
    for ii in range(nocc_lmo):
        key_ii = (ii, ii)
        if (key_ii in pno_spaces
                and pno_spaces[key_ii].get('pair_paos') is not None):
            _pao_domains.append(
                np.asarray(pno_spaces[key_ii]['pair_paos']))
        else:
            _pao_domains.append(np.zeros(0, dtype=int))
```

### Step 4 — `triple_paos` also needs to use the tighter PAO domain

Psi4 uses the triples-stage `lmo_to_paos` to build `lmotriplet_to_paos_[ijk]` (union over i, j, k). In our code, `_triple_pno_union_psi4` (line 140+) builds `triple_paos` from the **pair_paos** of (ij), (jk), (ik) — i.e., from `pno_spaces[pk]['pair_paos']`, which are CCSD-stage domains.

Two options:

**(a) Quick option**: intersect each pair's `pair_paos` with the triples-stage `_pao_domains[i] ∪ _pao_domains[j]` before computing the union. Pass `_pao_domains` as a new kwarg to `_triple_pno_union_psi4`.

**(b) Clean option**: compute `triple_paos` directly as `np.unique(np.concatenate([_pao_domains[i], _pao_domains[j], _pao_domains[k]]))`. This matches Psi4 exactly.

Go with **(b)** — simpler, matches Psi4.

In `_triple_pno_union_psi4` (line 140), replace the triple_paos construction (lines ~173-181):

```python
# Was: union of pair_paos for (ij, jk, ik)
pao_set = set()
for key in (ij, jk, ik):
    pp = pno_spaces[key].get('pair_paos')
    if pp is not None:
        pao_set.update(int(x) for x in np.asarray(pp).tolist())
if not pao_set:
    return np.zeros((nao, 0)), 0, None, None, None
triple_paos = np.array(sorted(pao_set), dtype=np.int64)
```

with:

```python
# Psi4 lmotriplet_to_paos_[ijk] = union of triples-stage lmo_to_paos[i, j, k]
if _pao_domains_triple is not None:
    pao_set = set()
    for lmo in (i, j, k):
        pao_set.update(int(x) for x in _pao_domains_triple[lmo].tolist())
else:
    pao_set = set()
    for key in (ij, jk, ik):
        pp = pno_spaces[key].get('pair_paos')
        if pp is not None:
            pao_set.update(int(x) for x in np.asarray(pp).tolist())
if not pao_set:
    return np.zeros((nao, 0)), 0, None, None, None
triple_paos = np.array(sorted(pao_set), dtype=np.int64)
```

And pass `_pao_domains_triple=_pao_domains` from `run_lccsd_t_ext` through `triple_kwargs` → `_process_one_triple` → `_triple_pno_union_psi4`. Each of those signatures needs the new kwarg added.

### Step 5 — validate

Sync, then smoke-test:

```bash
cp /home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/lccsd_t.py /environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/lccsd_t.py
cp /home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/driver.py /environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/driver.py
cp /home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/local_orbs.py /environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/local_orbs.py

# Smoke test: S22-1/2/3/8 — (T) interaction energies should stay within ~30 μEh
# of the full-naux baseline (pyscf_cc-pvdz.json). Compare against mkn1e2 result
# (pyscf_mkn_1e2.json — the current state before this change).
cd /home/ec2-user/Work/3d_tmcs/orca_benchmarks/s22
/environments/miniconda3/envs/tmc/bin/python -u run_s22_pyscf.py \
    --basis cc-pvdz --ncores 64 --scf-blas 16 --dimers 1,2,3,8 \
    --out results/pyscf_do_triples.json

# Water scaling — the exponent is what matters. Compare to mkn1e2 baseline.
cd /home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling
/environments/miniconda3/envs/tmc/bin/python -u run_water_scaling.py \
    --basis cc-pvdz --ncores 64 --scf-blas 16 --chains 4,8,10 \
    --out results/water_scaling_do_triples.json
```

Then compare the three checkpoints:
```python
import json, numpy as np
d_base = json.load(open('.../water_scaling_cc-pvdz.json'))        # full-naux, b=2.68
d_mkn  = json.load(open('.../water_scaling_mkn1e2.json'))         # current, b=2.39
d_do   = json.load(open('.../water_scaling_do_triples.json'))     # NEW
# Expect t_triples to drop further, exponent to drop too.
```

## Current state of the world (baseline before T_CUT_DO_TRIPLES)

Water chain (T) times, cc-pVDZ TightPNO:

| chain | nao | full-naux | psi4-port batched + Wcache + MKN=1e-2 |
|---|---|---|---|
| water4  |  96 |  4.5s |  4.0s |
| water8  | 192 | 34.9s | 19.4s |
| water10 | 240 | 47.8s | 37.1s |
| **exponent** | | **2.679** | **2.386** |

Target: Jiang (T) exponent 1.78.

## Files and landmarks

Source of truth (edit here, then sync to installed):
- `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/lccsd_t.py` — 1800 lines; key functions: `_triple_pno_union_psi4` (L140), `_build_triple_local_DF` (L509), `_process_one_triple` (L644), `run_lccsd_t_ext` (L977).
- `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/local_orbs.py` — `make_paos` at L104, DOI branches at L197/220/272/293, return at L331.
- `/home/ec2-user/Work/pyscf/pyscf/cc/dlpno_tccsd/driver.py` — `make_paos` call at L366.

Installed copy (must be kept in sync — dev edits to `/home/ec2-user/Work/pyscf` do NOT auto-install):
- `/environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/cc/dlpno_tccsd/`

Psi4 reference (read-only):
- `/environments/psi4_jiang/psi4/src/psi4/dlpno/triples.cc` — `tno_transform` at L431, `compute_lccsd_t0` at L609, PAO-domain rebuild at L316-329.
- `/environments/psi4_jiang/psi4/src/read_options.cc` — defaults at L2513-2575.

Benchmark data:
- `/home/ec2-user/Work/3d_tmcs/orca_benchmarks/s22/results/pyscf_cc-pvdz.json` — full-naux S22 baseline
- `/home/ec2-user/Work/3d_tmcs/orca_benchmarks/s22/results/pyscf_mkn_1e2.json` — current state S22
- `/home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling/results/water_scaling_cc-pvdz.json` — full-naux water baseline
- `/home/ec2-user/Work/3d_tmcs/orca_benchmarks/water_scaling/results/water_scaling_mkn1e2.json` — current state water scaling

## Git status

Branch `dlpno_tccsd` is ahead of origin by 1 commit. Uncommitted: `pyscf/cc/dlpno_tccsd/lccsd_t.py` (562 lines changed — the Psi4 port itself). The previous session's commit covers the pre-port state; this session's changes are unstaged. Do NOT commit without user approval.

## What to watch

- Energy drift: a looser per-LMO PAO domain at the triples stage is deliberately approximate. Psi4 absorbs ~20-30 μEh per monomer. If your drift is >100 μEh on a well-behaved water dimer, something is off (likely atom-completion logic or missing LMOs from the domain).
- The `_pao_domains` you build for triples should ONLY affect the triples stage — don't overwrite the CCSD-stage `pao_domains`. CCSD already ran and its `pair_paos` are stored in `pno_spaces[pk]['pair_paos']`.
- If `doi_iu` is `None` (e.g. user picked `doi_method='mulliken'`), fall back to the old behaviour (keep the else-branch intact). The driver currently always uses `doi_method='grid'`, so in practice `doi_iu` will always be available.
