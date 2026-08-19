"""Regression guard: the DLPNO energy must not depend on the thread count.

make_pnos once sized its auxiliary-shell blocking from the thread pool's
worker count. A 16-worker and a 32-worker run therefore partitioned the DF
int3c2e evaluation differently, handed BLAS different GEMM shapes, and
produced last-bit different integrals. Invisible on its own -- but it
propagates through the iterative LMP2 into the pair densities and flips
borderline PNO keep/drop decisions at T_CutPNO, which was worth 0.18
kcal/mol on an S22 dimer and made results non-reproducible across machines.

Fixed by deriving the partition from a fixed nominal width
(DLPNO_PNO_AUX_WORKERS) instead of the pool. This test fails if that
coupling comes back.

Each thread count runs in its OWN subprocess: state leaks between
calculations in one interpreter (see benchmarks/KNOWN_ISSUES.md), which
would confound the comparison.
"""
import json
import os
import subprocess
import sys
import textwrap
import unittest

# Energies must agree far below anything chemically meaningful. The floor is
# BLAS thread-scheduling noise, which is ~1e-9 Eh on a system this size; a
# returning partition dependence shows up orders of magnitude above it.
TOL = 1e-7

WORKER = textwrap.dedent("""
    import json, sys
    from pyscf import gto, scf
    from pyscf.cc.dlpno_tccsd import run_dlpno_ccsd_t
    ncores = int(sys.argv[1])
    # Water dimer: small enough to be a fast test, big enough to exercise
    # the blocked aux path and have a non-trivial pair list.
    mol = gto.M(atom='''
        O -1.551007 -0.114520 0.000000; H -1.934259 0.762503 0.000000
        H -0.599677  0.040712 0.000000; O  1.350625 0.111469 0.000000
        H  1.680398 -0.373741 -0.758561; H 1.680398 -0.373741 0.758561''',
        basis='cc-pvdz', verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    res = run_dlpno_ccsd_t(mf, frozen=2, ncores=ncores, verbose=0,
                           T_CutPNO=1e-7, T_CutEnergy=0.997,
                           T_CutTrace=0.999, T_CutDO=1e-3,
                           T_CutPairs=1e-5, T_CutPairs_MP2=1e-6)
    print('RESULT ' + json.dumps({'e': res['e_total'], 'hf': mf.e_tot}))
""")


def _run(ncores):
    env = dict(os.environ, PYSCF_TMPDIR=os.environ.get('PYSCF_TMPDIR', '/tmp'))
    out = subprocess.run([sys.executable, '-c', WORKER, str(ncores)],
                         capture_output=True, text=True, env=env, timeout=1800)
    for line in out.stdout.splitlines():
        if line.startswith('RESULT '):
            return json.loads(line[7:])
    raise RuntimeError(f'worker (ncores={ncores}) produced no result:\n'
                       f'{out.stdout[-2000:]}\n{out.stderr[-2000:]}')


class PartitionInvariance(unittest.TestCase):
    """Direct guard on the mechanism.

    The end-to-end energy check below only fails when a system happens to
    have a PNO sitting on the truncation boundary -- reintroducing the bug
    and running the water dimer does NOT move its energy. These assertions
    fail the moment the coupling comes back, regardless of the test system.
    """

    def test_partition_ignores_the_thread_pool(self):
        import ast
        import inspect
        from pyscf.cc.dlpno_tccsd import pno
        tree = ast.parse(inspect.getsource(pno._aux_target_q_per_block))
        fn = tree.body[0]
        if (fn.body and isinstance(fn.body[0], ast.Expr)
                and isinstance(fn.body[0].value, ast.Constant)):
            fn.body = fn.body[1:]          # the docstring explains the bug
        body = ast.dump(ast.Module(body=fn.body, type_ignores=[]))
        for bad in ('_max_workers', '_pool', 'cpu_count', 'nproc'):
            self.assertNotIn(
                bad, body,
                'the aux-shell partition must not be sized from the thread '
                'pool or the core count -- see the docstring of '
                '_aux_target_q_per_block')

    def test_partition_is_a_pure_function_of_size_and_env(self):
        from pyscf.cc.dlpno_tccsd import pno
        import concurrent.futures as cf
        w = pno._aux_target_q_per_block(200, 180)
        # merely having pools of different sizes alive must change nothing
        for n in (1, 4, 64):
            with cf.ThreadPoolExecutor(max_workers=n):
                self.assertEqual(pno._aux_target_q_per_block(200, 180), w)
        # the documented knob is the only thing that may change it
        os.environ['DLPNO_PNO_AUX_WORKERS'] = '8'
        try:
            self.assertNotEqual(pno._aux_target_q_per_block(200, 180), w)
        finally:
            del os.environ['DLPNO_PNO_AUX_WORKERS']


class KnownValues(unittest.TestCase):
    def test_energy_is_thread_count_invariant(self):
        a, b = _run(4), _run(16)
        self.assertAlmostEqual(a['hf'], b['hf'], delta=1e-9,
                               msg='SCF differs between thread counts')
        d = abs(a['e'] - b['e'])
        self.assertLess(
            d, TOL,
            msg=(f'DLPNO-CCSD(T) total energy depends on the thread count: '
                 f'{a["e"]:.10f} (4 threads) vs {b["e"]:.10f} (16), '
                 f'difference {d:.2e} Eh = {d * 627.5094740631:.4f} kcal/mol. '
                 f'Check that the aux-shell partition in make_pnos does not '
                 f'depend on _pool._max_workers.'))


if __name__ == '__main__':
    unittest.main()
