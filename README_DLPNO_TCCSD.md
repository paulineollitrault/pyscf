# DMRG-DLPNO-TCCSD(T)

A PySCF implementation of the DMRG-based Domain-based Local Pair Natural Orbital Tailored Coupled Cluster Singles and Doubles with perturbative Triples [Lang et al., JCTC 2020, 16, 3028].

## Method Overview

The method combines three key ingredients:

1. **DMRG-CASSCF** — captures strong (static) correlation in an active space via the Density Matrix Renormalization Group (block2 backend).
2. **DLPNO** — recovers dynamic correlation outside the active space using the local pair natural orbital approximation, keeping computational cost near-linear with system size.
3. **Tailored Coupled Cluster (TCC)** — links the two by injecting the DMRG cluster amplitudes into the CC iterations, avoiding double-counting and ensuring size-consistency of the active-space contribution.

The energy is:

```
E_total = E_HF + E_TCCSD(strong pairs) + E_LMP2(weak pairs) + E_(T)(external triples)
```

Pure-CAS triples are excluded from the (T) correction because they are already described by the DMRG wavefunction.

### References

- Lang, Sivalingam, Neese, JCTC 2020, 16, 3028 — DLPNO-TCCSD
- Veis, Antalík, Brabec, Neese, Legeza, Pittner, JPCL 2016, 7, 4086 — oxo-Mn(Salen) benchmark
- Riplinger & Neese, JCP 2013, 138, 034106 — DLPNO-CCSD
- Ye & Berkelbach, JCTC 2024, 20, 8948 — LNO-CCSD in PySCF (infrastructure reference)
- Lee & Head-Gordon, JCTC 2019, 15, 4594 — TCCSD(T) CAS exclusion

## Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| PySCF   | >= 2.3  | Required |
| block2  | >= 0.5.3 | Required for DMRG; SU2 or non-SU2 build |
| pyscf-dmrgscf | any | Required; bundled with block2 extras |
| numpy   | >= 1.21 | Required |
| scipy   | >= 1.7  | Required |
| mpi4py  | any     | Optional; enables MPI-parallel DMRG sweeps |

## Installation

### 1. Install PySCF

```bash
pip install pyscf
# or, for the development branch:
git clone https://github.com/pyscf/pyscf
pip install -e pyscf/
```

### 2. Install block2 with SU2 symmetry

block2 must be compiled with SU2 spin-orbital symmetry support for the DMRG-CASSCF interface.

```bash
# Option A: pip (pre-built wheels, recommended)
pip install block2

# Option B: build from source with MPI support
git clone https://github.com/block-hczhai/block2-preview
cd block2-preview
pip install -e ".[mpi]"
```

Verify the installation:

```python
import block2
from pyblock2.driver.core import DMRGDriver
print(block2.__version__)
```

### 3. Install this module

This module lives in the `pyscf/cc/dlpno_tccsd/` subdirectory of the PySCF source tree:

```bash
# If using the PySCF development branch at /path/to/pyscf:
export PYTHONPATH=/path/to/pyscf:$PYTHONPATH
python -c "from pyscf.cc.dlpno_tccsd import run_dlpno_tccsd_t; print('OK')"
```

## Quick Start

### Minimal example: H₂O

```python
from pyscf import gto
from pyscf.cc.dlpno_tccsd import run_dlpno_tccsd_t

mol = gto.M(
    atom='O 0 0 0; H 0 0.96 -0.34; H 0 -0.96 -0.34',
    basis='cc-pvdz',
    verbose=4,
)

result = run_dlpno_tccsd_t(
    mol,
    ncas=4,           # 4 active orbitals (2 bonding + 2 antibonding)
    nelec_cas=4,      # 4 active electrons
    maxM=200,         # DMRG bond dimension
    T_CutPNO=1e-7,    # TightPNO threshold
)

print(f"E(HF)     = {result['e_hf']:.10f}")
print(f"E(CASSCF) = {result['e_casscf']:.10f}")
print(f"E(TCCSD)  = {result['e_tccsd']:.10f}  (correlation)")
print(f"E(T) ext  = {result['e_t']:.10f}")
print(f"E(total)  = {result['e_total']:.10f}")
```

### Transition metal example: oxo-Mn(Salen) model

This reproduces the benchmark from Veis et al. JPCL 2016 (Table S1) using a
model MnO unit with NH₃ trans-axial and NH₂ equatorial ligands.

```python
from pyscf import gto
from pyscf.cc.dlpno_tccsd import run_dlpno_tccsd_t

mol = gto.M(
    atom='''
    Mn  0.000  0.000  0.000
    O   0.000  0.000  1.589
    N   1.904  0.000  0.000
    C   2.465  1.178  0.000
    C   2.465 -1.178  0.000
    N  -1.904  0.000  0.000
    C  -2.465  1.178  0.000
    C  -2.465 -1.178  0.000
    ''',
    basis='cc-pvtz',      # Veis 2016 uses cc-pVTZ
    charge=1,
    spin=2,               # triplet Mn(III) d^4
    verbose=4,
)

result = run_dlpno_tccsd_t(
    mol,
    ncas=6,
    nelec_cas=(3, 3),     # d^3 manifold for Mn(IV)
    maxM=1000,            # Veis uses M=1000 for production
    T_CutPNO=1e-7,        # TightPNO
    T_CutPairs=1e-5,
    scratch='./dmrg_scratch',
)
```

### Starting from a pre-computed CASSCF object

If DMRG-CASSCF has already been run (e.g. in a previous calculation), use the
`run_dlpno_tccsd_t_from_mc` entry point to skip Stage 1–2:

```python
from pyscf.cc.dlpno_tccsd.driver import run_dlpno_tccsd_t_from_mc

# mc  — converged DMRGSCF/CASSCF object
# mf  — RHF object with density fitting (mf.with_df must be set)
result = run_dlpno_tccsd_t_from_mc(
    mc, mf,
    T_CutPNO=1e-7,
    verbose=4,
)
```

## Threshold Guide

The method supports three accuracy levels following the ORCA convention:

| Level       | T_CutPNO  | T_CutPairs | Expected error vs CCSD(T) |
|-------------|-----------|------------|--------------------------|
| LoosePNO    | 1e-6      | 1e-3       | ~5 mEh                  |
| NormalPNO   | 3.33e-7   | 1e-4       | ~1 mEh                  |
| TightPNO    | 1e-7      | 1e-5       | ~0.1 mEh                |

For benchmark comparisons to published DLPNO-TCCSD results, use TightPNO
settings and a bond dimension M ≥ 1000.

## Module Structure

```
pyscf/cc/dlpno_tccsd/
├── __init__.py          # Public API exports
├── dmrg_interface.py    # DMRG-CASSCF + cluster amplitude extraction
├── local_orbs.py        # LMO (Pipek-Mezey/Boys) + PAO construction
├── pno.py               # PNO generation, LMP2, DF integrals
├── screening.py         # Pair classification (CAS/strong/weak/negligible)
├── lccsd.py             # Fragment LCCSD with CAS amplitude injection
├── lccsd_t.py           # External-space (T) correction with CAS exclusion
└── driver.py            # Top-level pipeline driver

pyscf/cc/test/
├── test_cas_amps.py                # CAS amplitude extraction vs FCI RDMs
├── test_pno_construction.py        # PNO LMP2 vs canonical MP2
├── test_dlpno_tccsd_vs_tccsd.py    # End-to-end H2 validation
└── test_oxomn_salen_small.py       # Transition-metal benchmark (requires block2)
```

## Running the Tests

```bash
# Fast tests (no block2 required):
pytest pyscf/cc/test/test_cas_amps.py -v
pytest pyscf/cc/test/test_pno_construction.py -v
pytest pyscf/cc/test/test_dlpno_tccsd_vs_tccsd.py -v

# Slow benchmark test (requires block2):
pytest pyscf/cc/test/test_oxomn_salen_small.py -v -s
```

## Known Limitations

### Current implementation

1. **Restricted reference only.** Only closed-shell RHF references are supported.
   Open-shell (UHF/ROHF) references are not yet implemented.

2. **Pair CCSD approximation.** The fragment LCCSD uses a simplified "ring
   diagram" pair approximation rather than the full DLPNO residual with
   inter-pair PNO overlap coupling (Pinski/Riplinger). This means weak
   inter-pair coupling terms (responsible for ~0.3 mEh in typical systems)
   are currently neglected.

3. **No singles in PNO basis.** T1 amplitudes are set to zero in the pair
   iteration (single amplitudes are typically small in RHF-based DLPNO-CCSD
   and can be folded into a t1-transformed Hamiltonian as in Jiang's Psi4
   implementation). The CAS t1 contributions are injected but not
   self-consistently iterated for the external singles.

4. **Triple PNO intersection.** The TNO space uses the pair (ij) as reference
   and projects onto (ik)/(jk) by raw inner products without the AO overlap
   metric. This is the T0 semicanonical approximation and introduces a small
   error relative to the full DLPNO-(T) of Jiang (JCP 2024).

5. **No DIIS for pair CCSD.** The per-pair CCSD uses direct Jacobi iterations
   without DIIS acceleration. Convergence may be slow for strongly correlated
   pairs at tight thresholds.

6. **Memory:** All pair amplitudes are stored in memory as a dictionary of
   `(n_pno, n_pno)` arrays. For very large systems (> 100 occupied orbitals)
   this may require significant RAM; no disk-based storage is implemented.

### Physical limitations

7. **CAS-pair criterion.** Pairs are classified as CAS pairs only if both
   occupied indices are in the CAS space AND the pair PNOs project onto the
   CAS virtual subspace with threshold ≥ 0.99. Mixed core/CAS pairs always
   go through DLPNO-CCSD.

8. **Basis set.** PAOs are constructed from the AO basis of the RHF
   calculation. Diffuse functions can lead to linearly dependent PAO domains;
   if convergence issues arise, use a non-augmented basis for the PAO/PNO
   step.

## TODO / Roadmap

- [ ] Full inter-pair coupling in LCCSD residual (Pinski Eq. 14–16)
- [ ] DIIS for pair CCSD iterations
- [ ] Unrestricted (UHF) reference support
- [ ] T1-transformed Hamiltonian for singles
- [ ] Disk-based storage of pair amplitudes for large systems
- [ ] Parallelization over pairs (trivially parallel)
- [ ] Proper S-metric TNO construction (full AO-overlap-weighted projection)
- [ ] Restart capability (save/load converged pair amplitudes)
- [ ] Interface to NEVPT2 as an alternative to CASSCF for CAS energies
