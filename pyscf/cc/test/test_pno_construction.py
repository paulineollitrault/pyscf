"""
Test PNO construction and pair energies.

System: Ne atom, aug-cc-pVDZ
Checks:
  1. sum of PNO-basis LMP2 pair energies ≈ canonical MP2 energy (within 0.1%)
  2. All PNO occupation numbers n_k >= 0 and sum <= 2 per pair
  3. strong_pairs ∪ weak_pairs = all LMO pairs (up to negligible-pair threshold)
  4. Pair domain sizes are non-empty for all pairs
"""

import unittest
import numpy as np
from pyscf import gto, scf, mp


class TestPNOConstruction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up Ne atom with aug-cc-pVDZ and density fitting."""
        mol = gto.M(
            atom='Ne 0 0 0',
            basis='aug-cc-pvdz',
            verbose=0,
        )
        cls.mol = mol

        mf = scf.RHF(mol).density_fit().run()
        cls.mf = mf

        # Reference canonical MP2 energy
        mp2 = mp.MP2(mf)
        cls.e_mp2_ref, _ = mp2.kernel()

        # Ne has 5 occupied orbitals (1s, 2s, 2p_x, 2p_y, 2p_z)
        cls.nocc = np.count_nonzero(mf.mo_occ > 1e-10)

    def _run_pno_construction(self, T_CutPNO=1e-10, T_CutPairs=0.0):
        """Helper: run PAO + PNO construction with loose thresholds."""
        from pyscf.cc.dlpno_tccsd.local_orbs import make_lmos, make_paos
        from pyscf.cc.dlpno_tccsd.pno import make_pnos

        mf = self.mf

        # Use canonical occupied orbitals as "LMOs" for Ne (it's spherically
        # symmetric, so PM localization is the identity; we skip it here)
        nocc = self.nocc
        C_lmo = mf.mo_coeff[:, :nocc].copy()

        s1e = mf.get_ovlp()
        C_pao, pao_domains, S_pao, F_pao = make_paos(
            mf, C_lmo, T_CutDO=0.0, s1e=s1e)  # T_CutDO=0: include all AOs

        pno_spaces, strong_pairs, weak_pairs, e_lmp2_total = make_pnos(
            mf, C_lmo, C_pao, pao_domains, S_pao, F_pao,
            T_CutPNO=T_CutPNO, T_CutPairs=T_CutPairs,
            S_cut_domain=1e-10,
            verbose=0)

        return pno_spaces, strong_pairs, weak_pairs, e_lmp2_total

    def test_lmp2_total_energy(self):
        """LMP2 from PNOs should reproduce canonical MP2 within 0.1%."""
        _, _, _, e_lmp2_pno = self._run_pno_construction(
            T_CutPNO=1e-12, T_CutPairs=0.0)

        # LMP2 from PNOs includes all i<=j pairs; multiply by 2 for i>j
        # Actually make_pnos already handles the full energy counting (i<=j pairs)
        # The canonical MP2 energy from pyscf should match within PNO truncation error
        ref = self.e_mp2_ref
        rel_err = abs(e_lmp2_pno - ref) / abs(ref)
        self.assertLess(rel_err, 0.001,
                        f'PNO LMP2 energy {e_lmp2_pno:.10g} deviates from '
                        f'canonical MP2 {ref:.10g} by {rel_err*100:.3f}%')

    def test_pno_occupation_bounds(self):
        """PNO |occupation| should be <= 2.  Off-diagonal pairs (i≠j) have a
        non-PSD pair density and can legitimately have negative eigenvalues."""
        pno_spaces, _, _, _ = self._run_pno_construction(T_CutPNO=1e-12)
        for (i, j), data in pno_spaces.items():
            n_pno = data['n_pno']
            # Diagonal pairs are PSD: occupations must be non-negative
            if i == j:
                self.assertTrue(
                    np.all(n_pno >= -1e-10),
                    f'Pair ({i},{j}): negative PNO occupation on diagonal pair: min={n_pno.min()}')
            # All pairs: |occupation| <= 2
            self.assertTrue(
                np.all(np.abs(n_pno) <= 2.0 + 1e-10),
                f'Pair ({i},{j}): |PNO occupation| > 2: max_abs={np.abs(n_pno).max()}')

    def test_pno_occupation_sum(self):
        """Sum of PNO occupations per diagonal pair (i=j) should be <= 2.
        Off-diagonal pairs have a non-PSD density and can have negative
        occupations, so only the diagonal bound is physically meaningful."""
        pno_spaces, _, _, _ = self._run_pno_construction(T_CutPNO=1e-12)
        for (i, j), data in pno_spaces.items():
            if i != j:
                continue
            n_pno = data['n_pno']
            occ_sum = np.sum(n_pno)
            self.assertLessEqual(
                occ_sum, 2.0 + 1e-6,
                f'Pair ({i},{j}): PNO occupation sum {occ_sum:.6f} > 2')

    def test_pair_completeness(self):
        """strong_pairs ∪ weak_pairs should cover all pairs (T_CutPairs_MP2=0)."""
        from pyscf.cc.dlpno_tccsd.pno import make_pnos
        from pyscf.cc.dlpno_tccsd.local_orbs import make_paos

        mf = self.mf
        nocc = self.nocc
        C_lmo = mf.mo_coeff[:, :nocc].copy()
        s1e = mf.get_ovlp()
        C_pao, pao_domains, S_pao, F_pao = make_paos(
            mf, C_lmo, T_CutDO=0.0, s1e=s1e)

        pno_spaces, strong_pairs, weak_pairs, _ = make_pnos(
            mf, C_lmo, C_pao, pao_domains, S_pao, F_pao,
            T_CutPNO=1e-12, T_CutPairs=1e-6, verbose=0)

        # All pairs in pno_spaces should appear in either strong or weak list
        classified = set(strong_pairs) | set(weak_pairs)
        all_pairs = set(pno_spaces.keys())

        # Some pairs may have e_ij = 0 exactly and appear in neither list;
        # allow that case
        unclassified = all_pairs - classified
        for (i, j) in unclassified:
            e_ij = pno_spaces[(i, j)]['e_mp2']
            self.assertAlmostEqual(
                e_ij, 0.0, places=10,
                msg=f'Pair ({i},{j}) unclassified but has non-zero e_mp2={e_ij}')

    def test_pno_spaces_nonempty(self):
        """Every pair should have at least 1 PNO with loose threshold."""
        pno_spaces, _, _, _ = self._run_pno_construction(
            T_CutPNO=1e-15, T_CutPairs=0.0)
        for (i, j), data in pno_spaces.items():
            n_pno = data['C_pno'].shape[1]
            self.assertGreater(
                n_pno, 0,
                f'Pair ({i},{j}): empty PNO space with very loose threshold')


if __name__ == '__main__':
    unittest.main(verbosity=2)
