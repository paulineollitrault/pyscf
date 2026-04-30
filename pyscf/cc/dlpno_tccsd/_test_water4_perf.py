"""Water-4 DLPNO-CCSD perf test (faster than water-10 for iteration).

Psi4 reference (cc-pVDZ): CCSD=3.416s, (T)=3.764s, total=7.18s.

Set DLPNO_CCSD_MONO_DROPIN_CYCLE=1 to use the C++ class drop-in cycle
driver.  Without it, the baseline PySCF Python cycle loop runs.
"""
import os
import time
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')

from pyscf import gto, scf, lib as pyscf_lib

pyscf_lib.num_threads(16)

GEOM_PATH = ('/home/ec2-user/Work/molecular-structures/S22_structures/'
             'water_geoms/water4.xyz')


def main():
    mol = gto.M(atom=GEOM_PATH, basis='cc-pvdz', charge=0, spin=0,
                symmetry=False, verbose=0)

    mode = ('CLASS DROP-IN' if int(os.environ.get(
        'DLPNO_CCSD_MONO_DROPIN_CYCLE', '0')) else 'BASELINE PySCF')
    print('=' * 70, flush=True)
    print(f'water-4 / cc-pvdz DLPNO-CCSD perf test [{mode}]', flush=True)
    print(f'  Psi4 reference (cc-pVDZ): CCSD=3.416s, (T)=3.764s, total=7.18s',
          flush=True)
    print('=' * 70, flush=True)

    t_hf = time.perf_counter()
    mf = scf.RHF(mol).density_fit()
    mf.kernel()
    print(f'HF wall: {time.perf_counter() - t_hf:.2f} s '
          f'(HF energy = {mf.e_tot:.10f})', flush=True)

    from pyscf.cc.dlpno_tccsd import run_dlpno_ccsd_t

    t_ccsd = time.perf_counter()
    res = run_dlpno_ccsd_t(mf, ncores=16)
    t_total = time.perf_counter() - t_ccsd

    print('=' * 70, flush=True)
    print(f'water-4 DLPNO-CCSD + (T) total wall: {t_total:.2f} s', flush=True)
    if isinstance(res, tuple) and len(res) >= 1:
        e_total = res[0]
        print(f'  E_total (HF + CC + (T)) = {e_total}', flush=True)
    print('=' * 70, flush=True)


if __name__ == '__main__':
    main()
