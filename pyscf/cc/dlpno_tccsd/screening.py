"""
Pair screening and classification for DLPNO-TCCSD(T).

Classifies LMO pairs (i,j) into four categories:
  1. CAS pairs    — both i,j ∈ CAS occupied AND PNOs contained in CAS vir
                    → inject DMRG amplitudes, skip LCCSD
  2. Strong pairs — |e_ij^LMP2| > T_CutPairs
                    → full LCCSD solve in PNO basis
  3. Weak pairs   — T_CutPairs_MP2 < |e_ij^LMP2| <= T_CutPairs
                    → keep at LMP2 level (use semicanonical T2 directly)
  4. Distant/negligible pairs  — |e_ij^LMP2| <= T_CutPairs_MP2
                    → energy estimated by dipole approximation or neglected

Reference thresholds (Lang et al. 2020 / Riplinger 2016):
  TightPNO:   T_CutPNO = 1e-7, T_CutPairs = 1e-5
  NormalPNO:  T_CutPNO = 3.33e-7, T_CutPairs = 1e-4
  LoosePNO:   T_CutPNO = 1e-6, T_CutPairs = 1e-3

References:
    Lang et al., JCTC 2020, 16, 3028
    Riplinger et al., JCP 2016, 144, 024109
    Pinski et al., JCP 2015, 143, 034108
"""

import numpy as np
from pyscf.lib import logger


def classify_pairs(pno_spaces, occ_cas_idx, vir_cas_idx,
                   mo_coeff, s1e,
                   T_CutPairs=1e-4, T_CutPairs_MP2=1e-6,
                   verbose=None):
    """Classify all LMO pairs into CAS / strong / weak / negligible.

    CAS pairs are identified purely by occupancy: both i,j must be in the
    CAS occupied space. CAS pairs are also added to strong_pairs so they
    go through tailored CCSD with extended PNOs.

    Args:
        pno_spaces (dict): Output of pno.make_pnos.
        occ_cas_idx (np.ndarray): CAS occupied indices in full MO list.
        vir_cas_idx (np.ndarray): CAS virtual indices in full MO list.
        mo_coeff (np.ndarray): Full MO coefficient matrix (from mc.mo_coeff).
        s1e (np.ndarray): AO overlap matrix.
        T_CutPairs (float): Strong/weak boundary threshold.
        T_CutPairs_MP2 (float): Weak/negligible boundary threshold.
        cas_pno_proj_thresh (float): Projection threshold for CAS-pair test.
        verbose: Verbosity.

    Returns:
        cas_pairs (set): (i,j) pairs with DMRG amplitude injection.
        strong_pairs (list): (i,j) pairs for LCCSD solve (excluding CAS).
        weak_pairs (list): (i,j) pairs at LMP2 level.
        negligible_pairs (list): (i,j) pairs that are dropped / dipole-estimated.
        e_lmp2_weak (float): LMP2 energy contribution from weak pairs.
        e_lmp2_strong (float): LMP2 energy from strong pairs (for delta-PT2).
    """
    log = logger.new_logger(None, verbose)

    occ_cas_set = set(occ_cas_idx.tolist())

    cas_pairs = set()
    strong_pairs = []
    weak_pairs = []
    negligible_pairs = []

    e_lmp2_weak = 0.0
    e_lmp2_strong = 0.0

    for (i, j), data in pno_spaces.items():
        e_ij = data['e_mp2']
        abs_e = abs(e_ij)

        if abs_e <= T_CutPairs_MP2:
            negligible_pairs.append((i, j))
            continue

        # CAS pair: both occupied indices in CAS occupied space
        is_cas_occ = (i in occ_cas_set) and (j in occ_cas_set)
        if is_cas_occ:
            cas_pairs.add((i, j))
            # CAS pairs are also strong pairs — they go through tailored CCSD
            strong_pairs.append((i, j))
            fac = 1.0 if i == j else 2.0
            e_lmp2_strong += fac * e_ij
            continue

        fac = 1.0 if i == j else 2.0  # factor 2 for i<j pairs
        if abs_e > T_CutPairs:
            strong_pairs.append((i, j))
            e_lmp2_strong += fac * e_ij
        else:
            weak_pairs.append((i, j))
            e_lmp2_weak += fac * e_ij

    log.info('Pair classification:')
    log.info('  CAS pairs: %d', len(cas_pairs))
    log.info('  Strong pairs: %d  (LMP2: %.15g)', len(strong_pairs), e_lmp2_strong)
    log.info('  Weak pairs: %d  (LMP2: %.15g)', len(weak_pairs), e_lmp2_weak)
    log.info('  Negligible pairs: %d', len(negligible_pairs))

    return (cas_pairs, strong_pairs, weak_pairs, negligible_pairs,
            e_lmp2_weak, e_lmp2_strong)


def get_lmo_frag_list(C_lmo):
    """Generate default fragment list: each LMO is its own fragment.

    This is the standard DLPNO fragment assignment where each LMO contributes
    one fragment. The fragment energy partitioning follows the population-based
    projector approach of Ye & Berkelbach (JCTC 2024).

    Args:
        C_lmo (np.ndarray): (nao, nocc_lmo). LMO coefficients.

    Returns:
        frag_lolist (list): [[0], [1], ..., [nocc_lmo-1]].
    """
    nocc_lmo = C_lmo.shape[1]
    return [[i] for i in range(nocc_lmo)]


def dipole_pair_energy_estimate(C_lmo, mol, F_lmo, eps_vir_ij, K_ij_dipole,
                                 R_ij_min=3.0):
    """Estimate energy of distant LMO pair using dipole interaction formula.

    Implements Eq. (17) of Pinski et al. JCP 2015 for pairs that are too far
    apart to be screened into the strong or weak pair lists.

    For practical use this requires dipole matrix elements and estimated
    virtual orbital energies — the function signature is provided for
    completeness but the implementation is left as a stub since distant pairs
    are typically negligible in the TCCSD context.

    Args:
        C_lmo (np.ndarray): LMO coefficients.
        mol: PySCF Mole object.
        F_lmo (np.ndarray): Fock matrix in LMO basis.
        eps_vir_ij (tuple): Approximate virtual orbital energies for this pair.
        K_ij_dipole (float): Dipole-approximated exchange integral.
        R_ij_min (float): Minimum LMO pair distance (Bohr).

    Returns:
        e_dipole (float): Estimated pair energy.
    """
    # Stub: distant pairs are added as a constant correction in the driver.
    # See Pinski 2015 Eq. 17 for the full expression.
    raise NotImplementedError(
        'Dipole pair energy estimate not implemented. '
        'Set T_CutPairs_MP2 = 0 to include all pairs at LMP2 level, '
        'or accept that distant pairs are neglected.'
    )
