"""The DLPNO threshold presets must match the published tables exactly.

Every number below is re-stated independently from Table I and Table II of

    Jiang, Turney, Schaefer et al., J. Chem. Phys. (2024),
    doi 10.1063/5.0219963

so that editing presets.py cannot quietly change what "TightPNO" means. This
exists because both of this project's benchmark campaigns had been run with
invented thresholds -- S22 with T_CutDO 1e-3, MOBH35 with T_CutDO 0.02,
T_CutEnergy 0.97 and T_CutTrace disabled -- while being reported as
comparable to ORCA TightPNO. They were not.
"""

import unittest

from pyscf.cc.dlpno_tccsd import presets


# Table I, transcribed from the paper, independently of presets.py.
PAPER_TABLE_I = {
    #                    TightPNO   NormalPNO
    'T_CutPNO':         (1e-7,      3.33e-7),
    'T_CutEnergy':      (0.997,     0.99),
    'T_CutTrace':       (0.999,     0.99),
    'T_CutPNO_MP2':     (1e-9,      3.33e-9),
    'T_CutEnergy_MP2':  (0.999,     0.997),
    'T_CutTrace_MP2':   (0.9999,    0.999),
    'T_DiagScale':      (1e-3,      1e-3),
    'T_CutDO':          (5e-3,      1e-2),
    'T_CutDO_ij':       (1e-5,      1e-5),
    'T_CutPre':         (1e-7,      1e-6),
    'T_CutPairs':       (1e-5,      1e-4),
    'T_CutPairs_MP2':   (1e-6,      1e-6),
    'T_CutMKN':         (1e-3,      1e-3),
}

# Table II, same values across PNO settings.
PAPER_TABLE_II = {
    'T_CutTNO':         1e-9,
    'T_CutTNO_Pre':     1e-7,
    'T_CutTriples_Pre': 1e-7,
    'T_CutDO_Triples':  1e-2,
    'T_CutMKN_Triples': 1e-2,
}


class PublishedThresholds(unittest.TestCase):

    def test_tightpno_matches_table_i(self):
        for name, (tight, _) in PAPER_TABLE_I.items():
            self.assertIn(name, presets.TIGHTPNO, f'{name} missing')
            self.assertEqual(
                presets.TIGHTPNO[name], tight,
                f'TightPNO {name} is {presets.TIGHTPNO[name]}, '
                f'the paper says {tight}')

    def test_normalpno_matches_table_i(self):
        for name, (_, normal) in PAPER_TABLE_I.items():
            self.assertIn(name, presets.NORMALPNO, f'{name} missing')
            self.assertEqual(
                presets.NORMALPNO[name], normal,
                f'NormalPNO {name} is {presets.NORMALPNO[name]}, '
                f'the paper says {normal}')

    def test_triples_match_table_ii(self):
        for name, val in PAPER_TABLE_II.items():
            self.assertIn(name, presets.TRIPLES, f'{name} missing')
            self.assertEqual(presets.TRIPLES[name], val,
                             f'{name} is {presets.TRIPLES[name]}, '
                             f'the paper says {val}')

    def test_no_extra_invented_parameters(self):
        """A preset may not carry a threshold the paper does not define."""
        for label, table in (('TightPNO', presets.TIGHTPNO),
                             ('NormalPNO', presets.NORMALPNO)):
            extra = set(table) - set(PAPER_TABLE_I)
            self.assertFalse(extra, f'{label} has unpublished keys: {extra}')

    def test_thresholds_helper_is_tight_by_default(self):
        d = presets.thresholds()
        self.assertEqual(d['T_CutDO'], 5e-3)
        self.assertEqual(d['T_CutEnergy'], 0.997)
        self.assertEqual(d['T_CutTrace'], 0.999)
        self.assertEqual(d['T_CutPairs'], 1e-5)

    def test_unknown_setting_is_rejected(self):
        with self.assertRaises(ValueError):
            presets.thresholds('LooseIshPNO')


if __name__ == '__main__':
    unittest.main()
