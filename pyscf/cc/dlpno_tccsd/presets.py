"""Published DLPNO threshold sets, transcribed from the reference paper.

    Jiang, Turney, Schaefer et al., "Accurate and efficient open-source
    implementation of domain-based local pair natural orbital (DLPNO)
    coupled-cluster theory", J. Chem. Phys. (2024), doi 10.1063/5.0219963
    -- Table I (PNO settings) and Table II (triples).

These values are DATA, not tuning knobs. Do not "improve" them for a
particular benchmark: a threshold that has to be changed to make one
application behave is masking a defect, and it makes results from different
applications incomparable. That happened here -- an invented T_CutDO of 1e-3
made S22 look healthy while hiding a DLPNO-CCSD instability that only shows
up at the published 5e-3 (see ../../../../3d_tmcs/benchmarks/KNOWN_ISSUES.md).

pyscf/cc/test/test_dlpno_presets.py re-states every number independently and
fails if these drift.
"""

# --- Table I: PNO settings ------------------------------------------------
# Parameter                     TightPNO     NormalPNO
TIGHTPNO = {
    'T_CutPNO':         1e-7,
    'T_CutEnergy':      0.997,
    'T_CutTrace':       0.999,
    'T_CutPNO_MP2':     1e-9,
    'T_CutEnergy_MP2':  0.999,
    'T_CutTrace_MP2':   0.9999,
    'T_DiagScale':      1e-3,
    'T_CutDO':          5e-3,
    'T_CutDO_ij':       1e-5,
    'T_CutPre':         1e-7,
    'T_CutPairs':       1e-5,
    'T_CutPairs_MP2':   1e-6,
    'T_CutMKN':         1e-3,
}

NORMALPNO = {
    'T_CutPNO':         3.33e-7,
    'T_CutEnergy':      0.99,
    'T_CutTrace':       0.99,
    'T_CutPNO_MP2':     3.33e-9,
    'T_CutEnergy_MP2':  0.997,
    'T_CutTrace_MP2':   0.999,
    'T_DiagScale':      1e-3,
    'T_CutDO':          1e-2,
    'T_CutDO_ij':       1e-5,
    'T_CutPre':         1e-6,
    'T_CutPairs':       1e-4,
    'T_CutPairs_MP2':   1e-6,
    'T_CutMKN':         1e-3,
}

# --- Table II: triples (same values for every PNO setting) ----------------
TRIPLES = {
    'T_CutTNO':          1e-9,
    'T_CutTNO_Pre':      1e-7,
    'T_CutTriples_Pre':  1e-7,
    'T_CutDO_Triples':   1e-2,
    'T_CutMKN_Triples':  1e-2,
}

PRESETS = {'TightPNO': TIGHTPNO, 'NormalPNO': NORMALPNO}

# Published parameters that this implementation does not yet expose as a
# separate knob. Listed explicitly so the gap is visible rather than implied
# by silence; each is either fixed internally at the published value or
# folded into another threshold.
NOT_YET_WIRED = ('T_DiagScale', 'T_CutDO_ij', 'T_CutPre', 'T_CutMKN')

# Keyword names accepted by run_dlpno_ccsd_t / make_pnos.
_DRIVER_KEYS = ('T_CutPNO', 'T_CutEnergy', 'T_CutTrace', 'T_CutDO',
                'T_CutPairs', 'T_CutPairs_MP2', 'T_CutPNO_MP2',
                'T_CutEnergy_MP2', 'T_CutTrace_MP2')


def thresholds(setting='TightPNO'):
    """Return the published thresholds for `setting` as driver kwargs.

    >>> thresholds('TightPNO')['T_CutDO']
    0.005
    """
    try:
        table = PRESETS[setting]
    except KeyError:
        raise ValueError(
            f'unknown PNO setting {setting!r}; expected one of '
            f'{sorted(PRESETS)}') from None
    return {k: table[k] for k in _DRIVER_KEYS if k in table}
