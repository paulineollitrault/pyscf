"""
DMRG-based DLPNO-TCCSD(T) [Lang et al., JCTC 2020, 16, 3028]

This module implements the DLPNO-TCCSD(T) method, which combines:
  - Density Matrix Renormalization Group (DMRG) for the active CAS space
  - Density-fitted Local Pair Natural Orbital (DLPNO) approximation for
    the external correlation
  - Tailored Coupled Cluster (TCC) to link DMRG and CC amplitudes

The method provides accurate treatment of strongly correlated systems at
reduced computational cost by:
  1. Treating the CAS with DMRG (captures static correlation exactly)
  2. Using PNO-compressed pair spaces for external CCSD (dynamic correlation)
  3. Injecting DMRG amplitudes directly into the CC equations for CAS pairs
  4. Computing (T) corrections only over external-space triples

References:
    Lang et al., J. Chem. Theory Comput. 2020, 16, 3028
    Riplinger & Neese, J. Chem. Phys. 2013, 138, 034106
    Ye & Berkelbach, J. Chem. Theory Comput. 2024, 20, 8948

Dependencies:
    pyscf >= 2.3
    block2 >= 0.5.3  (for DMRG backend)
    mpi4py           (optional, for MPI parallelism over fragments)
"""

from pyscf.cc.dlpno_tccsd.driver import run_dlpno_tccsd_t
from pyscf.cc.dlpno_tccsd.dmrg_interface import run_dmrg_casscf, get_cas_amplitudes
from pyscf.cc.dlpno_tccsd.local_orbs import make_lmos, make_paos
from pyscf.cc.dlpno_tccsd.pno import make_pnos, classify_cas_pairs
from pyscf.cc.dlpno_tccsd.lccsd import run_lccsd
from pyscf.cc.dlpno_tccsd.lccsd_t import run_lccsd_t_ext

__all__ = [
    'run_dlpno_tccsd_t',
    'run_dmrg_casscf',
    'get_cas_amplitudes',
    'make_lmos',
    'make_paos',
    'make_pnos',
    'classify_cas_pairs',
    'run_lccsd',
    'run_lccsd_t_ext',
]
