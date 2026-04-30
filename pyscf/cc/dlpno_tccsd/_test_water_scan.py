"""Water-N scan to compare scaling vs Psi4 reference.

Runs water-{4,8,10,15} (skip larger as they take a long time) and
prints CCSD/(T)/total timings alongside Psi4 reference values from
the watercluster_timings.csv.
"""
import os
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')
os.environ.setdefault('DLPNO_C_CYCLE', '1')
os.environ.setdefault('DLPNO_CCSD_MONO_DROPIN_CYCLE', '1')

from pyscf import gto, scf, lib as pyscf_lib

pyscf_lib.num_threads(16)

GEOM_DIR = '/home/ec2-user/Work/molecular-structures/S22_structures/water_geoms'

# Psi4 reference values (cc-pVDZ, Xeon 6136 16 cores)
PSI4_REF = {
    'water4':  {'ccsd': 3.416,  '(t)': 3.764,  'total': 7.18},
    'water8':  {'ccsd': 9.852,  '(t)': 12.233, 'total': 22.085},
    'water10': {'ccsd': 15.095, '(t)': 17.606, 'total': 32.701},
    'water15': {'ccsd': 38.753, '(t)': 40.76,  'total': 79.513},
}


def run_one(name):
    geom = f'{GEOM_DIR}/{name}.xyz'
    print(f'\n=== {name} ===', flush=True)
    mol = gto.M(atom=geom, basis='cc-pvdz', charge=0, spin=0,
                symmetry=False, verbose=0)

    t_hf = time.perf_counter()
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    t_hf_wall = time.perf_counter() - t_hf

    from pyscf.cc.dlpno_tccsd import run_dlpno_ccsd_t

    t_ccsd = time.perf_counter()
    res = run_dlpno_ccsd_t(mf, ncores=16)
    t_total = time.perf_counter() - t_ccsd

    ref = PSI4_REF.get(name, {})
    print(f'  HF wall:  {t_hf_wall:.2f} s', flush=True)
    print(f'  Total:    {t_total:.2f} s   (Psi4 ref: {ref.get("total", "?")})',
          flush=True)
    if 'total' in ref:
        ratio = t_total / ref['total']
        print(f'  Ratio vs Psi4: {ratio:.2f}x', flush=True)


if __name__ == '__main__':
    targets = sys.argv[1:] if len(sys.argv) > 1 else ['water4', 'water8', 'water10']
    for name in targets:
        run_one(name)
