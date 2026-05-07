"""Standalone water-4 driver for the t1_ints real-data parity test.

Run with `DLPNO_CCSD_MONO_TEST_T1_INTS=1` to exit after the validation
hook fires (see lccsd.py).  Or run normally to skip the hook.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')

from pyscf import gto, scf, lib as pyscf_lib

pyscf_lib.num_threads(16)


# Water tetramer (typical DLPNO test geom).
WATER_4_XYZ = """
O   0.000000   0.000000   0.000000
H   0.757000   0.586000   0.000000
H  -0.757000   0.586000   0.000000
O   2.800000   0.000000   1.500000
H   3.557000   0.586000   1.500000
H   2.043000   0.586000   1.500000
O   0.000000   2.800000   1.500000
H   0.757000   3.386000   1.500000
H  -0.757000   3.386000   1.500000
O   2.800000   2.800000   0.000000
H   3.557000   3.386000   0.000000
H   2.043000   3.386000   0.000000
"""


def main():
    mol = gto.M(atom=WATER_4_XYZ, basis='cc-pvdz', charge=0, spin=0,
                symmetry=False, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    print(f'water-4 HF = {mf.e_tot:.10f}', flush=True)

    from pyscf.cc.dlpno_tccsd import run_dlpno_ccsd_t
    res = run_dlpno_ccsd_t(mf, ncores=16, ccsd_max_cycle=1)
    print(f'DLPNO-CCSD result: {res}', flush=True)


if __name__ == '__main__':
    main()
