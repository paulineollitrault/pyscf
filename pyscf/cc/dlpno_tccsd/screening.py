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


def compute_doi_ij(C_lmo, with_df):
    """Compute LMO-LMO differential overlap integrals via DF.

    DOI_ij = sqrt( (ii|jj) ) where (ii|jj) = sum_Q B^Q_ii * B^Q_jj.
    Approximates the grid-based DOI from Jiang/Psi4.

    Args:
        C_lmo: (nao, nocc) LMO coefficients.
        with_df: DF object with prebuilt 3-index integrals.

    Returns:
        doi_ij: (nocc, nocc) DOI matrix.
    """
    from pyscf.ao2mo import _ao2mo
    nocc = C_lmo.shape[1]
    naux = with_df.get_naoaux()
    mo = np.asfortranarray(C_lmo)

    # Build (i i | Q) = diagonal of ooL
    # Use nr_e2 to get (i j | Q) then extract diagonal
    chunks = []
    for Lpq in with_df.loop():
        chunks.append(Lpq.copy())
    Lpq_full = np.vstack(chunks)

    ijslice = (0, nocc, 0, nocc)
    buf = _ao2mo.nr_e2(Lpq_full, mo, ijslice, aosym='s2')
    ooL = buf.reshape(naux, nocc, nocc)  # (Q, i, j)

    # DOI_ij = sqrt( sum_Q ooL[Q,i,i] * ooL[Q,j,j] )
    iiL = np.array([ooL[:, i, i] for i in range(nocc)])  # (nocc, Q)
    doi_ij = np.sqrt(np.dot(iiL, iiL.T))
    return doi_ij


def compute_dipole_pair_energies(C_lmo, C_pao, mol, F_lmo, pao_domains,
                                  S_pao, F_pao, T_CUT_DO_PRE=3e-2,
                                  S_cut=1e-6, with_df=None):
    """Estimate MP2 pair energies using the dipole approximation.

    Follows Jiang/Psi4 compute_dipole_ints(): for each LMO i, build a
    semicanonical virtual space from PAOs with DOI > T_CUT_DO_PRE, compute
    LMO-PAO dipole integrals, then estimate pair energies via the dipole formula.

    Args:
        C_lmo: (nao, nocc) LMO coefficients.
        C_pao: (nao, nao) PAO coefficients.
        mol: PySCF Mole object.
        F_lmo: (nocc, nocc) Fock in LMO basis.
        pao_domains: list of arrays, PAO domains per LMO.
        S_pao: (nao, nao) PAO overlap.
        F_pao: (nao, nao) PAO Fock.
        T_CUT_DO_PRE: DOI threshold for dipole prescreening domain (default 3e-2).
        S_cut: Canonical orthogonalization cutoff.
        with_df: DF object (used for DOI_iu if available).

    Returns:
        dipole_e: (nocc, nocc) dipole-estimated MP2 pair energies.
        dipole_e_bound: (nocc, nocc) upper bound (parallel dipoles).
    """
    from functools import reduce
    nocc = C_lmo.shape[1]
    nao = C_lmo.shape[0]
    s1e = mol.intor_symmetric('int1e_ovlp')

    # AO dipole integrals
    with mol.with_common_orig((0, 0, 0)):
        dip_ao = mol.intor('int1e_r', comp=3)  # (3, nao, nao)

    # LMO-LMO dipoles: R_i = <i|r|i>
    R_i = np.zeros((nocc, 3))
    for x in range(3):
        dip_lmo = reduce(np.dot, (C_lmo.T, dip_ao[x], C_lmo))
        R_i[:, x] = np.diag(dip_lmo)

    # LMO-PAO dipoles
    dip_lmo_pao = np.zeros((3, nocc, nao))
    for x in range(3):
        dip_lmo_pao[x] = np.dot(C_lmo.T, np.dot(dip_ao[x], C_pao))

    # Per-LMO: build prescreened semicanonical virtual space and dipole vectors
    lmo_pao_dr = []   # list of (n_sc_i, 3) dipole vectors
    lmo_pao_eps = []   # list of (n_sc_i,) semicanonical energies

    for i in range(nocc):
        domain_i = pao_domains[i]
        if len(domain_i) == 0:
            lmo_pao_dr.append(np.zeros((0, 3)))
            lmo_pao_eps.append(np.zeros(0))
            continue

        # Orthogonalize PAOs in domain
        S_dom = S_pao[np.ix_(domain_i, domain_i)]
        eigvals, eigvecs = np.linalg.eigh(S_dom)
        keep = eigvals > S_cut
        if not np.any(keep):
            lmo_pao_dr.append(np.zeros((0, 3)))
            lmo_pao_eps.append(np.zeros(0))
            continue
        X = eigvecs[:, keep] / np.sqrt(eigvals[keep])

        # Semicanonicalize
        F_dom = F_pao[np.ix_(domain_i, domain_i)]
        F_orth = reduce(np.dot, (X.T, F_dom, X))
        eps_sc, U_sc = np.linalg.eigh(F_orth)
        XU = np.dot(X, U_sc)  # (n_dom, n_sc)

        # Dipole vectors: dr[u, x] = <i|r_x|u_sc> - R_i[x] * delta(overlap)
        # Following Psi4: lmo_pao_dr[i][u] = <i|r|u_sc> (in SC basis)
        dr = np.zeros((len(eps_sc), 3))
        for x in range(3):
            dip_i_dom = dip_lmo_pao[x, i, domain_i]  # (n_dom,)
            dr[:, x] = np.dot(dip_i_dom, XU)

        lmo_pao_dr.append(dr)
        lmo_pao_eps.append(eps_sc)

    # Dipole pair energy estimate
    dipole_e = np.zeros((nocc, nocc))
    dipole_e_bound = np.zeros((nocc, nocc))

    for i in range(nocc):
        for j in range(i + 1, nocc):
            R_ij = R_i[i] - R_i[j]
            R_norm = np.linalg.norm(R_ij)
            if R_norm < 1e-10:
                continue
            Rh = R_ij / R_norm

            dr_i = lmo_pao_dr[i]  # (n_i, 3)
            dr_j = lmo_pao_dr[j]  # (n_j, 3)
            eps_i = lmo_pao_eps[i]
            eps_j = lmo_pao_eps[j]

            if len(eps_i) == 0 or len(eps_j) == 0:
                continue

            e_actual = 0.0
            e_bound = 0.0
            for u in range(len(eps_i)):
                for v in range(len(eps_j)):
                    iu_dot_jv = np.dot(dr_i[u], dr_j[v])
                    iu_dot_R = np.dot(dr_i[u], Rh)
                    jv_dot_R = np.dot(dr_j[v], Rh)

                    num_actual = (iu_dot_jv - 3.0 * iu_dot_R * jv_dot_R) ** 2
                    num_linear = (-2.0 * iu_dot_jv) ** 2

                    denom = (eps_i[u] + eps_j[v]) - (F_lmo[i, i] + F_lmo[j, j])
                    if abs(denom) < 1e-12:
                        continue

                    e_actual += num_actual / denom
                    e_bound += num_linear / denom

            factor = -4.0 * R_norm ** (-6)
            dipole_e[i, j] = dipole_e[j, i] = e_actual * factor
            dipole_e_bound[i, j] = dipole_e_bound[j, i] = e_bound * factor

    return dipole_e, dipole_e_bound


def prescreen_pairs(C_lmo, mol, F_lmo, C_pao, pao_domains, S_pao, F_pao,
                    with_df=None, T_CUT_DO_ij=1e-5, T_CUT_PRE=1e-6,
                    verbose=None):
    """Prescreen LMO pairs using DOI_ij and dipole pair energy estimates.

    Follows Jiang/Psi4: a pair (i,j) is kept if DOI_ij > T_CUT_DO_ij OR
    |dipole_e_bound[i,j]| > T_CUT_PRE. Dropped pairs get their energy
    estimated via the dipole formula.

    Args:
        C_lmo: (nao, nocc) LMO coefficients.
        mol: PySCF Mole object.
        F_lmo: (nocc, nocc) Fock in LMO basis.
        C_pao, pao_domains, S_pao, F_pao: PAO data from make_paos.
        with_df: DF object.
        T_CUT_DO_ij: DOI_ij threshold for keeping a pair.
        T_CUT_PRE: Dipole energy bound threshold for keeping a pair.

    Returns:
        keep_pairs: set of (i,j) tuples (i<=j) that pass prescreening.
        e_dipole_dropped: float, estimated energy of dropped pairs.
        doi_ij: (nocc, nocc) DOI matrix.
    """
    log = logger.new_logger(None, verbose)
    nocc = C_lmo.shape[1]

    # Compute DOI_ij
    if with_df is not None:
        doi_ij = compute_doi_ij(C_lmo, with_df)
    else:
        doi_ij = np.ones((nocc, nocc))  # keep all if no DF

    # Compute dipole pair energy estimates
    dipole_e, dipole_e_bound = compute_dipole_pair_energies(
        C_lmo, C_pao, mol, F_lmo, pao_domains, S_pao, F_pao,
        with_df=with_df)

    # Prescreen
    keep_pairs = set()
    e_dipole_dropped = 0.0
    n_kept = 0
    n_dropped = 0

    for i in range(nocc):
        for j in range(i, nocc):
            overlap_big = doi_ij[i, j] > T_CUT_DO_ij
            energy_big = abs(dipole_e_bound[i, j]) > T_CUT_PRE

            if overlap_big or energy_big:
                keep_pairs.add((i, j))
                n_kept += 1
            else:
                fac = 1.0 if i == j else 2.0
                e_dipole_dropped += fac * dipole_e[i, j]
                n_dropped += 1

    log.info('Pair prescreening: %d kept, %d dropped (dipole E=%.6e)',
             n_kept, n_dropped, e_dipole_dropped)

    return keep_pairs, e_dipole_dropped, doi_ij
