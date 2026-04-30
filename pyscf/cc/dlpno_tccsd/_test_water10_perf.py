"""Water-10 DLPNO-CCSD perf test.

Compares wall time + energy vs published Psi4 timing (15.095s on cc-pVDZ
per /home/ec2-user/Work/3d_tmcs/orca_benchmarks/t1_dlpno_ccsd_t_si/watercluster_timings.csv).

Run with:
    /environments/miniconda3/envs/tmc/bin/python -m pyscf.cc.dlpno_tccsd._test_water10_perf
"""
import os
import time
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')

from pyscf import gto, scf, lib as pyscf_lib

pyscf_lib.num_threads(16)

GEOM_PATH = ('/home/ec2-user/Work/molecular-structures/S22_structures/'
             'water_geoms/water10.xyz')


def main():
    mol = gto.M(atom=GEOM_PATH, basis='cc-pvdz', charge=0, spin=0,
                symmetry=False, verbose=0)

    print('=' * 70, flush=True)
    print(f'water-10 / cc-pvdz baseline DLPNO-CCSD perf test', flush=True)
    print(f'  Psi4 reference (cc-pVDZ): 15.095 s', flush=True)
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
    print(f'water-10 DLPNO-CCSD + (T) total wall: {t_total:.2f} s', flush=True)
    # res is a tuple; pick the energy fields without dumping amplitudes.
    if isinstance(res, tuple) and len(res) >= 1:
        e_total = res[0]
        print(f'  E_total (HF + CC + (T)) = {e_total}', flush=True)
    print(f'  Psi4 ref (CCSD): 15.095s, (T): 17.606s, total: 32.701s',
          flush=True)
    print('=' * 70, flush=True)


if __name__ == '__main__':
    main()
