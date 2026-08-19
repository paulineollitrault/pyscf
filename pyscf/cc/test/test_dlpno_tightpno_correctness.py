"""DLPNO-CCSD must be correct at the PUBLISHED TightPNO thresholds.

Guards the PAO index-space defect fixed in fa02a551d. DLPNOcompute_pair_used
returns POSITIONS within an atom's page; the aux-first replay consumed the
same array as GLOBAL PAO indices. The two agree only while an atom's PAO
list spans every PAO -- true at large domains, false at the published
T_CutDO of 5e-3, where 2.06% of column lookups went to the wrong PAO.

The damage was severe and unstable: (H2O)8 E_corr came out anywhere from
-4.19 to +4.08 Eh, varied with thread count, varied BETWEEN RUNS at a fixed
thread count, and often diverged. Against a canonical -1.70460 Eh.

Two properties are asserted, because either alone would have missed it:
  1. the energy is right at the published thresholds -- a test run at a
     tighter T_CutDO passes even with the bug, which is exactly how this
     survived a full S22 campaign;
  2. the answer does not depend on the thread count.

(H2O)8 / cc-pVDZ is the smallest system in the scaling series that actually
triggers absent columns, so it is the cheapest honest guard. ~1 min.
"""

import unittest

import numpy as np

from pyscf import gto, scf
from pyscf.cc.dlpno_tccsd import (run_dlpno_ccsd_t, count_frozen_core,
                                  presets)

# (H2O)8, from the published scaling series used by Jiang et al.
WATER8 = [
    ('O', (2.573, -1.034, -1.721)),
    ('H', (2.493, -1.949, -1.992)),
    ('H', (2.160, -0.537, -2.427)),
    ('O', (0.705, 0.744, 0.160)),
    ('H', (-0.071, 0.264, 0.450)),
    ('H', (1.356, 0.064, -0.014)),
    ('O', (0.146, 3.420, 0.167)),
    ('H', (0.077, 2.479, 0.006)),
    ('H', (0.908, 3.510, 0.740)),
    ('O', (-2.852, -2.432, 0.188)),
    ('H', (-2.344, -1.934, 0.829)),
    ('H', (-2.321, -3.209, 0.012)),
    ('O', (-3.211, -0.002, -1.696)),
    ('H', (-3.615, -0.714, -2.191)),
    ('H', (-3.124, -0.346, -0.807)),
    ('O', (2.169, -1.407, 1.640)),
    ('H', (1.438, -0.898, 1.989)),
    ('H', (1.772, -2.224, 1.339)),
    ('O', (-1.092, -0.672, 1.792)),
    ('H', (-0.725, -1.146, 2.538)),
    ('H', (-1.282, 0.203, 2.132)),
    ('O', (3.039, 2.832, -1.018)),
    ('H', (3.392, 2.234, -1.678)),
    ('H', (2.156, 2.506, -0.847)),
]

# Reference: this implementation at the published TightPNO, verified against
# canonical DF-CCSD(T)/cc-pVDZ (-1.704598 Eh) -- a 0.36 kcal/mol local
# approximation error -- and matching the June 2026 result to 6 decimals.
E_CORR_REF = -1.70517372
TOL = 1e-6


def _run(ncores):
    mol = gto.M(atom=WATER8, basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-ri')
    mf.conv_tol = 1e-9
    mf.verbose = 0
    mf.kernel()
    thr = presets.thresholds('TightPNO')
    res = run_dlpno_ccsd_t(mf, frozen=count_frozen_core(mol), ncores=ncores,
                           verbose=0, **thr)
    return res['e_tccsd'] + res.get('e_lmp2_weak', 0.0)


class TightPNOCorrectness(unittest.TestCase):

    def test_energy_at_published_tightpno(self):
        e = _run(4)
        self.assertAlmostEqual(
            e, E_CORR_REF, delta=TOL,
            msg=f'(H2O)8 E_corr {e:.8f} != {E_CORR_REF:.8f} at the published '
                f'TightPNO; PAO index handling is wrong again')

    def test_thread_count_does_not_change_the_answer(self):
        a, b = _run(1), _run(8)
        self.assertAlmostEqual(
            a, b, delta=TOL,
            msg=f'1 thread gives {a:.8f}, 8 threads {b:.8f}')

    def test_uses_the_published_thresholds(self):
        """The guard is void if run at a tighter T_CutDO -- it passes there
        even with the bug present."""
        self.assertEqual(presets.thresholds('TightPNO')['T_CutDO'], 5e-3)


if __name__ == '__main__':
    unittest.main()
