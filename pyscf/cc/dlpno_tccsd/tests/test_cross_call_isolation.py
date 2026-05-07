"""Regression test for cross-call DLPNO state pollution.

Several batched residual builders cache plan structures on the function
object (compute_G_term_batched._plan_cache, compute_C_tilde_batched._plan_cache,
etc.). The cache key was tuple(sorted(t2_pno_all.keys())); the cached plan
stored references to S matrices from the FIRST run. When two molecules
with the same pair structure ran in sequence, the second run reused the
first run's S references, corrupting the second result by tens of mEh.

run_lccsd._clear_dlpno_caches() clears these between runs. This test
verifies that running the same DLPNO-CCSD(T) calculation 3 times in the
same Python process gives the same correlation energy (within thread
noise).
"""
import os
import unittest
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyscf import gto, scf
from pyscf.cc.dlpno_tccsd import run_dlpno_ccsd_t, count_frozen_core


# Threshold: thread-noise envelope on 32-thread BLAS reductions.  Anything
# bigger means the cross-call cache is leaking again.
TOL_HARTREE = 1e-7


class CrossCallIsolation(unittest.TestCase):
    """Three sequential DLPNO calls on the same molecule must agree to
    within thread-noise."""

    def test_ammonia_3x(self):
        # Single ammonia in cc-pVDZ, frozen core, JIANG TightPNO settings.
        mol = gto.M(
            atom='''
              N  0.0  0.0  0.116
              H  0.0  0.939 -0.270
              H  0.813 -0.469 -0.270
              H -0.813 -0.469 -0.270
            ''',
            basis='cc-pvdz', verbose=0)
        jiang = dict(
            T_CutPNO=1e-7, T_CutEnergy=0.997, T_CutTrace=0.999,
            T_CutDO=5e-3, T_CutPairs=1e-5, T_CutPairs_MP2=1e-6,
            lmo_method='pipek-mezey')
        e_corr_runs = []
        for _ in range(3):
            mf = scf.RHF(mol).density_fit()
            mf.with_df.auxbasis = 'cc-pvdz-ri'
            mf.kernel()
            pool = ThreadPoolExecutor(max_workers=8)
            r = run_dlpno_ccsd_t(
                mf, frozen=count_frozen_core(mol),
                ccsd_conv_tol=1e-7, ccsd_max_cycle=200, verbose=0,
                _pool=pool, **jiang)
            pool.shutdown(wait=True)
            e_corr_runs.append(r['e_tccsd'] - mf.e_tot)

        # All runs identical to within thread-noise.
        spread = max(e_corr_runs) - min(e_corr_runs)
        self.assertLess(
            spread, TOL_HARTREE,
            f'cross-call energy drift {spread:.3e} Eh > {TOL_HARTREE:.0e}; '
            f'plan-cache pollution may have regressed. '
            f'Runs: {e_corr_runs}')


if __name__ == '__main__':
    unittest.main()
