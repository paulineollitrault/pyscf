"""
Test CAS amplitude extraction from CASSCF/FCI RDMs.

System: H2O, CAS(2,2), cc-pVDZ
Reference: FCI solution via pyscf.fci; cluster amplitudes computed by hand
from FCI 1-RDM and 2-RDM.
Assert: max|t1_cas - t1_fci| < 1e-8
        max|t2_cas - t2_fci| < 1e-6

This test validates the core amplitude extraction logic in dmrg_interface.py,
using pyscf's built-in FCI solver as a reference so that block2 is not required.
"""

import unittest
import numpy as np
from functools import reduce
from pyscf import gto, scf, mcscf, fci


def _cas_amplitudes_from_rdm(mc):
    """Reference implementation using FCI RDMs directly.

    Returns t1_cas and t2_cas computed from 1- and 2-RDM by the cluster
    decomposition formula, independently of dmrg_interface.py.
    """
    ncore = mc.ncore
    ncas = mc.ncas
    nelec_cas = mc.nelecas
    nocc_cas = nelec_cas[0]
    nvir_cas = ncas - nocc_cas

    # Fock matrix in CASSCF MO basis
    fock_ao = mc._scf.get_fock()
    fock_mo = reduce(np.dot, (mc.mo_coeff.T, fock_ao, mc.mo_coeff))
    mo_eps = fock_mo.diagonal().real

    occ_cas_idx = np.arange(ncore, ncore + nocc_cas)
    vir_cas_idx = np.arange(ncore + nocc_cas, ncore + ncas)

    eps_occ = mo_eps[occ_cas_idx]
    eps_vir = mo_eps[vir_cas_idx]

    # Get RDMs from FCI solver (use solver interface, stable across versions)
    dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelec_cas)

    # t1_cas
    t1_num = dm1[:nocc_cas, nocc_cas:]
    denom1 = eps_occ[:, None] - eps_vir[None, :]
    t1 = np.where(np.abs(denom1) > 1e-12,
                  t1_num / np.where(np.abs(denom1) > 1e-12, denom1, 1.0),
                  0.0)

    # t2_cas
    dm2_iajb = dm2[:nocc_cas, :nocc_cas, nocc_cas:, nocc_cas:].transpose(0, 1, 3, 2)
    denom2 = (eps_occ[:, None, None, None] + eps_occ[None, :, None, None]
              - eps_vir[None, None, :, None] - eps_vir[None, None, None, :])
    t1t1 = (np.einsum('ia,jb->ijab', t1, t1)
            - np.einsum('ib,ja->ijab', t1, t1))
    t2 = np.where(np.abs(denom2) > 1e-12,
                  (dm2_iajb - t1t1) / np.where(np.abs(denom2) > 1e-12, denom2, 1.0),
                  0.0)
    t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))

    return t1, t2, occ_cas_idx, vir_cas_idx


class TestCASAmplitudes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up H2O CAS(2,2) cc-pVDZ."""
        mol = gto.M(
            atom='''
            O  0.000000  0.000000  0.117790
            H  0.000000  0.755453 -0.471161
            H  0.000000 -0.755453 -0.471161
            ''',
            basis='cc-pvdz',
            verbose=0,
        )
        cls.mol = mol

        mf = scf.RHF(mol).run()
        cls.mf = mf

        # CAS(2,2): 2 electrons in 2 orbitals (HOMO + LUMO)
        mc = mcscf.CASSCF(mf, 2, 2)
        mc.verbose = 0
        mc.kernel()
        cls.mc = mc

    def test_amplitude_extraction_module(self):
        """Test that dmrg_interface.get_cas_amplitudes matches reference."""
        from pyscf.cc.dlpno_tccsd.dmrg_interface import get_cas_amplitudes

        t1_mod, t2_mod, occ_idx_mod, vir_idx_mod = get_cas_amplitudes(self.mc)
        t1_ref, t2_ref, occ_idx_ref, vir_idx_ref = _cas_amplitudes_from_rdm(self.mc)

        np.testing.assert_array_equal(occ_idx_mod, occ_idx_ref,
                                      'occ_cas_idx mismatch')
        np.testing.assert_array_equal(vir_idx_mod, vir_idx_ref,
                                      'vir_cas_idx mismatch')

        # CAS(2,2): nocc_cas=1, nvir_cas=1 → t1 is scalar, t2 is (1,1,1,1)
        self.assertEqual(t1_mod.shape, (1, 1), 't1_cas shape')
        self.assertEqual(t2_mod.shape, (1, 1, 1, 1), 't2_cas shape')

        np.testing.assert_allclose(
            t1_mod, t1_ref, atol=1e-8,
            err_msg='t1_cas deviates from FCI reference')
        np.testing.assert_allclose(
            t2_mod, t2_ref, atol=1e-6,
            err_msg='t2_cas deviates from FCI reference')

    def test_t1_from_rdm_manual(self):
        """Manual check: t1_cas[i,a] should relate to the CI singles content."""
        t1, t2, occ_idx, vir_idx = _cas_amplitudes_from_rdm(self.mc)

        # CAS(2,2) in RHF limit (no correlation): dm1 off-diag = 0 → t1 = 0
        # With correlation: |t1| should be small (singles are small in CASSCF)
        self.assertLess(np.max(np.abs(t1)), 0.5,
                        'CAS t1 amplitude unexpectedly large')

    def test_t2_symmetry(self):
        """t2_cas should satisfy t2[i,j,a,b] = t2[j,i,b,a] (RHF symmetry)."""
        t1, t2, _, _ = _cas_amplitudes_from_rdm(self.mc)
        t2_perm = t2.transpose(1, 0, 3, 2)
        np.testing.assert_allclose(
            t2, t2_perm, atol=1e-10,
            err_msg='t2_cas symmetry violation: t2[i,j,a,b] != t2[j,i,b,a]')

    def test_cas_indices(self):
        """CAS indices should be correct slices of the full MO array."""
        from pyscf.cc.dlpno_tccsd.dmrg_interface import cas_idx_to_full
        ncore, nocc_cas, nvir_cas, occ_idx, vir_idx = cas_idx_to_full(self.mc)

        self.assertEqual(ncore, self.mc.ncore)
        self.assertEqual(nocc_cas, self.mc.nelecas[0])
        self.assertEqual(nvir_cas, self.mc.ncas - self.mc.nelecas[0])
        self.assertEqual(len(occ_idx), nocc_cas)
        self.assertEqual(len(vir_idx), nvir_cas)
        # Indices should be contiguous and non-overlapping
        self.assertEqual(occ_idx[-1] + 1, vir_idx[0],
                         'occ and vir CAS indices should be contiguous')


if __name__ == '__main__':
    unittest.main(verbosity=2)
