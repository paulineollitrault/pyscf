"""
Local orbital construction for DLPNO-TCCSD(T).

Provides:
  - Pipek-Mezey LMO localization of occupied MOs (using mc.mo_coeff as input)
  - PAO construction by projecting occupied space from AO basis
    (Algorithm 1 of Jiang JCP 2024; Eq. 1-5 of Riplinger/Neese 2013)
  - PAO domain assignment based on LMO differential overlap thresholds
  - IAO construction as an alternative fragment basis

Primary template: Ye & Berkelbach lnocc/lno.py (pyscf-forge lnocc branch)
PAO construction translated from Jiang/psi4 dlpnobase.cc setup_orbitals()
"""

import numpy as np
from functools import reduce
from pyscf import lib, lo
from pyscf.lib import logger


def make_lmos(mf_or_mc, method='pipek-mezey', frozen=0):
    """Localize occupied MOs using Pipek-Mezey localization.

    Uses the MO coefficients from mc.mo_coeff (CASSCF-optimized) if a CASSCF
    object is passed, or mf.mo_coeff if an RHF object is passed.  The CASSCF
    MO basis must be used as the starting point throughout so that CAS and
    external-orbital index bookkeeping remains consistent.

    Args:
        mf_or_mc: RHF or CASSCF object. If CASSCF, uses mc.mo_coeff.
        method (str): 'pipek-mezey' (default) or 'boys'.
        frozen (int): Number of frozen core orbitals (excluded from localization).

    Returns:
        C_lmo (np.ndarray): Shape (nao, nocc - frozen). LMO coefficient matrix
            in AO basis. Columns are LMOs ordered to match the occupied block of
            mf_or_mc.mo_coeff (first `frozen` canonical MOs are left unchanged
            and are prepended).
        C_lmo_full (np.ndarray): Shape (nao, nocc). Full occupied block
            including frozen core (canonical) + localized active occupied.
    """
    # Resolve mo_coeff and mo_occ
    if hasattr(mf_or_mc, 'nelecas'):
        # CASSCF object
        mo_coeff = mf_or_mc.mo_coeff
        mol = mf_or_mc.mol
        nocc = mf_or_mc._scf.mol.nelectron // 2
    else:
        # RHF object
        mo_coeff = mf_or_mc.mo_coeff
        mol = mf_or_mc.mol
        nocc = np.count_nonzero(mf_or_mc.mo_occ > 1e-10)

    log = logger.new_logger(mol)

    # Active occupied block (after frozen core)
    orbocc_active = mo_coeff[:, frozen:nocc]   # (nao, nocc_active)
    orbocc_frozen = mo_coeff[:, :frozen]        # (nao, frozen)

    nocc_active = orbocc_active.shape[1]

    if nocc_active == 0:
        log.warn('No active occupied orbitals to localize.')
        return orbocc_active, mo_coeff[:, :nocc]

    # PySCF's localization routines suffer severe thread contention with
    # many BLAS threads (e.g. 0.6s → 6s going from 1 → 64 threads).
    # Temporarily reduce to 1 thread for this small, serial step.
    _saved_threads = lib.num_threads()
    lib.num_threads(1)
    try:
        if method.lower() in ('pipek-mezey', 'pm'):
            mlo = lo.PipekMezey(mol, orbocc_active)
            mlo.verbose = mol.verbose - 1
            C_lmo = mlo.kernel()
            # Stability: always do Jacobi sweep to escape local minima
            for _ in range(3):
                C_lmo1 = mlo.stability_jacobi()[1]
                if C_lmo1 is C_lmo:
                    break
                mlo = lo.PipekMezey(mol, C_lmo1)
                mlo.verbose = mol.verbose - 1
                mlo.init_guess = None
                C_lmo = mlo.kernel()
        elif method.lower() == 'boys':
            mlo = lo.Boys(mol, orbocc_active)
            mlo.verbose = mol.verbose - 1
            C_lmo = mlo.kernel()
        else:
            raise ValueError(f'Unknown localization method: {method}')
    finally:
        lib.num_threads(_saved_threads)

    # Assemble full occupied block (frozen canonical + localized active)
    if frozen > 0:
        C_lmo_full = np.hstack((orbocc_frozen, C_lmo))
    else:
        C_lmo_full = C_lmo

    log.info('LMO localization (%s): %d active occupied orbitals', method, nocc_active)
    return C_lmo, C_lmo_full


def make_paos(mf_or_mc, C_lmo, T_CutDO=0.02, s1e=None, with_df=None,
              T_CutMKN=0.0, doi_method='pao'):
    """Construct Projected Atomic Orbitals (PAOs) and per-LMO domains.

    PAOs are constructed by projecting the occupied MO space out of the AO
    basis, then normalized. This follows Eq. (1-5) of Riplinger & Neese 2013
    and Algorithm 1 of Jiang JCP 2024.

    Three domain assignment methods are available:

    doi_method='pao' (default, Jiang 2024):
        Per-PAO differential overlap integral DOI_{i,mu} = (i mu|i mu)^{1/2}.
        Computed via DF: DOI = ||B^Q_{iu}||_2. This is a Coulomb-weighted
        integral and tends to give larger domains than grid-based DOI.

    doi_method='grid' (Psi4-compatible):
        Grid-based differential overlap integral:
        DOI_{i,u} = sqrt( integral w(r) |phi_i(r)|^2 |phi_u(r)|^2 dr )
        Computed on a DFT-style numerical grid. This matches Psi4's DOI
        and gives physically compact domains. Atom completion is applied.

    doi_method='mulliken' (ORCA-compatible):
        Atom-based Mulliken population. Atom A is included in domain(i)
        if sum_{mu in A} (C_lmo * S)_{mu,i}^2 > T_CutMKN.
        All PAOs on included atoms enter the domain. This matches ORCA's
        domain construction with TCutMKN.

    Args:
        mf_or_mc: RHF or CASSCF object (for mol and mo_coeff).
        C_lmo (np.ndarray): Shape (nao, nocc_active). LMO coefficients.
        T_CutDO (float): Per-PAO DOI threshold (used when doi_method='pao').
        s1e (np.ndarray, optional): AO overlap matrix. Computed if not provided.
        with_df: DF object for integral-based DOI (used when doi_method='pao').
        T_CutMKN (float): Atom Mulliken population threshold (doi_method='mulliken').
            ORCA TightPNO default: 1e-3.
        doi_method (str): 'pao' for per-PAO DOI, 'mulliken' for atom-based Mulliken.

    Returns:
        C_pao (np.ndarray): Shape (nao, nao). PAO coefficients (all PAOs).
        pao_domains (list of np.ndarray): pao_domains[i] is an integer array
            of the AO indices belonging to the domain of LMO i.
        S_pao (np.ndarray): Shape (nao, nao). PAO overlap matrix.
        F_pao (np.ndarray): Shape (nao, nao). Fock matrix in PAO basis.
    """
    if hasattr(mf_or_mc, 'nelecas'):
        mol = mf_or_mc.mol
        mf = mf_or_mc._scf
    else:
        mol = mf_or_mc.mol
        mf = mf_or_mc

    log = logger.new_logger(mol)

    if s1e is None:
        s1e = mf.get_ovlp()   # (nao, nao)

    nao = s1e.shape[0]
    nocc_lmo = C_lmo.shape[1]

    # Full occupied coefficient matrix (for PAO projection)
    # We need all occupied orbitals, not just those being localized.
    # Use mf.mo_coeff occ block for the projection kernel.
    nocc_full = np.count_nonzero(mf.mo_occ > 1e-10)
    C_occ = mf.mo_coeff[:, :nocc_full]   # (nao, nocc_full)

    # PAO projection: C_pao = I - C_occ @ C_occ^T @ S
    # Each column of C_pao (shape nao x nao) is a PAO built from the
    # corresponding AO after removing the occupied space.
    #
    # C_pao[:,μ] = e_μ - sum_i |i><i|S|e_μ> = δ[:,μ] - C_occ @ (C_occ^T @ S[:,μ])
    # In matrix form: C_pao = I - C_occ @ C_occ^T @ S
    proj = np.dot(C_occ, np.dot(C_occ.T, s1e))   # (nao, nao)
    C_pao = np.eye(nao) - proj                     # (nao, nao); col = PAO

    # Normalize each PAO column so <μ̃|S|μ̃> = 1
    S_pao = reduce(np.dot, (C_pao.T, s1e, C_pao))   # (nao, nao) PAO overlap
    norms = np.sqrt(np.diag(S_pao))
    # Columns with norm close to zero correspond to occupied-space directions;
    # keep them but mark with zero norm (they drop out of pair calculations).
    norms_safe = np.where(norms > 1e-10, norms, 1.0)
    C_pao = C_pao / norms_safe[None, :]

    # Recompute S_pao and F_pao after normalization
    S_pao = reduce(np.dot, (C_pao.T, s1e, C_pao))
    fock_ao = mf.get_fock()
    F_pao = reduce(np.dot, (C_pao.T, fock_ao, C_pao))

    # LMO domain assignment: select PAOs for each LMO's local virtual space.

    pao_domains = []
    ao_labels = mol.ao_labels(fmt=False)
    atom_ids = np.array([lbl[0] for lbl in ao_labels])

    if doi_method == 'mulliken' and T_CutMKN > 0:
        # ORCA-compatible atom-based Mulliken domain assignment.
        # Atom A is in domain(i) if Mulliken_pop(A, LMO_i) > T_CutMKN.
        # All PAOs centered on included atoms are added to the domain.
        natom = mol.natm
        # Mulliken population matrix: P_{mu,i} = C_lmo[mu,i] * (S @ C_lmo)[mu,i]
        SC = np.dot(s1e, C_lmo)  # (nao, nocc)
        for i in range(nocc_lmo):
            pop_ao = C_lmo[:, i] * SC[:, i]   # (nao,) Mulliken AO populations
            pop_atom = np.zeros(natom)
            for a in range(natom):
                pop_atom[a] = np.sum(pop_ao[atom_ids == a])
            atoms_in = np.where(np.abs(pop_atom) > T_CutMKN)[0]
            if len(atoms_in) == 0:
                atoms_in = np.array([np.argmax(np.abs(pop_atom))])
            domain_i = np.where(np.isin(atom_ids, atoms_in))[0]
            pao_domains.append(domain_i)
        log.info('Domain method: Mulliken atom (T_CutMKN=%.1e), '
                 'avg atoms/LMO=%.1f, avg PAOs/LMO=%.1f',
                 T_CutMKN,
                 np.mean([len(np.unique(atom_ids[d])) for d in pao_domains]),
                 np.mean([len(d) for d in pao_domains]))

    elif T_CutDO > 0 and doi_method == 'grid':
        # Grid-based DOI matching Psi4 (dlpnobase.cc compute_overlap_ints):
        #   DOI_{i,u} = sqrt( integral w(r) |phi_i(r)|^2 |phi_u(r)|^2 dr )
        # Uses a DFT-style numerical grid. Atom completion is applied after
        # thresholding (if any PAO on an atom passes, all PAOs on that atom
        # are included).
        from pyscf.dft import gen_grid, numint
        grids = gen_grid.Grids(mol)
        grids.level = 1  # moderate grid, similar to Psi4 defaults
        grids.build()

        ao_labels = mol.ao_labels(fmt=False)
        atom_ids_local = np.array([lbl[0] for lbl in ao_labels])

        ni = numint.NumInt()
        doi_iu = np.zeros((nocc_lmo, nao))

        # Evaluate AO values on grid in blocks
        for ao_val, mask, weight, coords in ni.block_loop(mol, grids, nao):
            # ao_val: (npts, nao) AO values at grid points
            # weight: (npts,) quadrature weights
            npts = ao_val.shape[0]

            # LMO values at grid points: phi_i(r) = sum_mu C_lmo[mu,i] * chi_mu(r)
            lmo_vals = ao_val @ C_lmo  # (npts, nocc)
            # PAO values at grid points: phi_u(r) = sum_mu C_pao[mu,u] * chi_mu(r)
            pao_vals = ao_val @ C_pao  # (npts, nao)

            # Square and weight: w(r) * |phi_i(r)|^2 and |phi_u(r)|^2
            lmo_sq_w = lmo_vals ** 2 * weight[:, None]  # (npts, nocc)
            pao_sq = pao_vals ** 2  # (npts, nao)

            # Accumulate DOI^2 = sum_r w(r) |phi_i(r)|^2 |phi_u(r)|^2
            doi_iu += lmo_sq_w.T @ pao_sq  # (nocc, nao)

        doi_iu = np.sqrt(doi_iu)

        for i in range(nocc_lmo):
            doi = doi_iu[i]
            pao_inds = np.where(doi > T_CutDO)[0]
            if len(pao_inds) == 0:
                pao_inds = np.array([np.argmax(doi)])
            # Atom completion: if any PAO on atom passes, include all PAOs on that atom
            atoms_in = np.unique(atom_ids_local[pao_inds])
            domain_i = np.where(np.isin(atom_ids_local, atoms_in))[0]
            pao_domains.append(domain_i)
        log.info('Domain method: grid DOI (T_CutDO=%.1e), '
                 'avg atoms/LMO=%.1f, avg PAOs/LMO=%.1f',
                 T_CutDO,
                 np.mean([len(np.unique(atom_ids_local[d])) for d in pao_domains]),
                 np.mean([len(d) for d in pao_domains]))

    elif with_df is not None and T_CutDO > 0 and doi_method == 'pao':
        # Per-PAO DOI via DF (Jiang et al. 2024, Eq 58):
        #   DOI_{i,μ̃} = (iμ̃|iμ̃)^{1/2} = ||B^Q_{iμ̃}||₂
        # With atom completion: if ANY PAO on atom A passes the threshold,
        # ALL PAOs on atom A are included (matches Jiang/Psi4 contract_lists).
        from pyscf.cc.dlpno_tccsd.pno import _build_ovL
        ovL_lmo_pao = _build_ovL(with_df, C_lmo, C_pao)  # (nocc, nao, naux)

        ao_labels_df = mol.ao_labels(fmt=False)
        atom_ids_df = np.array([lbl[0] for lbl in ao_labels_df])
        for i in range(nocc_lmo):
            doi = np.sqrt(np.sum(ovL_lmo_pao[i] ** 2, axis=1))  # (nao,)
            pao_inds = np.where(doi > T_CutDO)[0]
            if len(pao_inds) == 0:
                pao_inds = np.array([np.argmax(doi)])
            # Atom completion: if any PAO on atom passes, include all on that atom
            atoms_in = np.unique(atom_ids_df[pao_inds])
            domain_i = np.where(np.isin(atom_ids_df, atoms_in))[0]
            pao_domains.append(domain_i)
        del ovL_lmo_pao

    else:
        # Fallback: Löwdin population-based (for when DF is not available)
        ao_labels = mol.ao_labels(fmt=False)
        atom_ids = np.array([lbl[0] for lbl in ao_labels])
        natom = mol.natm
        from scipy.linalg import sqrtm
        S_half = np.real(sqrtm(s1e))

        for i in range(nocc_lmo):
            lmo_i = C_lmo[:, i]
            pop_ao = (S_half @ lmo_i) ** 2
            total_pop = np.sum(pop_ao)
            if total_pop < 1e-15 or T_CutDO <= 0:
                domain_i = np.arange(nao)
            else:
                pop_atom = np.zeros(natom)
                for a in range(natom):
                    pop_atom[a] = np.sum(pop_ao[atom_ids == a])
                frac_atom = pop_atom / total_pop
                atoms_in_domain = list(np.where(frac_atom > T_CutDO)[0])
                completeness = sum(frac_atom[a] for a in atoms_in_domain)
                completeness_target = 1.0 - 2.0 * T_CutDO
                if completeness < completeness_target:
                    remaining = [a for a in np.argsort(frac_atom)[::-1]
                                 if a not in set(atoms_in_domain)]
                    for a in remaining:
                        atoms_in_domain.append(a)
                        completeness += frac_atom[a]
                        if completeness >= completeness_target:
                            break
                domain_i = np.where(np.isin(atom_ids, atoms_in_domain))[0]
            if len(domain_i) == 0:
                domain_i = np.where(pop_ao > 1e-12)[0]
            pao_domains.append(domain_i)

    log.info('PAO construction: nao=%d  avg domain size=%.1f',
             nao, np.mean([len(d) for d in pao_domains]))

    return C_pao, pao_domains, S_pao, F_pao


def split_localize_orbitals(mf, ncas, nelec_cas, method='pipek-mezey',
                            frozen=0, occ_natural=None):
    """Split-localize MOs according to Lang et al. JCTC 2020.

    Separately localizes each orbital subspace so that no mixing occurs
    between subspaces. For a closed-shell RHF reference, the four subspaces
    are:
      1. Doubly occupied inactive (core): MOs 0:ncore  (frozen:ncore if frozen>0)
      2. Doubly occupied active:           MOs ncore:nocc_full
      3. Singly occupied active:           (empty for RHF; relevant for CASSCF)
      4. Active virtual:                   MOs nocc_full:nocc_full+nvir_cas

    External virtual MOs (nocc_full+nvir_cas onwards) are left in the
    canonical basis (PAOs will be built from them for DLPNO).

    This split is required so that:
      - The DMRG-CI active space uses localized active orbitals
      - The DLPNO inactive LMOs are localized to their own subspace
      - CAS amplitudes extracted from DMRG are in the same local basis
        as the DLPNO pair correlation treatment

    Args:
        mf: Converged RHF object.
        ncas (int): Number of active (CAS) orbitals.
        nelec_cas (tuple): (n_alpha, n_beta) active electrons.
        method (str): 'pipek-mezey' (default) or 'boys'.
        frozen (int): Number of frozen core MOs (excluded from localization).
        occ_natural (np.ndarray, optional): Natural orbital occupations for the
            active space (from CASSCF 1-RDM). If provided, uses them to split
            doubly/singly occupied active MOs. If None, all active occupied
            are treated as doubly occupied (RHF assumption).

    Returns:
        C_mo_loc (np.ndarray): (nao, nmo) Full MO matrix with each subspace
            localized independently.
        C_lmo (np.ndarray): (nao, nocc_full - frozen) All occupied LMOs
            (inactive active occ), for use as DLPNO pair LMOs.
    """
    mol = mf.mol
    log = logger.new_logger(mol)
    nao = mf.mo_coeff.shape[0]
    nocc_full = np.count_nonzero(mf.mo_occ > 1e-10)
    nocc_cas_a = nelec_cas[0]
    ncore = nocc_full - nocc_cas_a
    nvir_cas = ncas - nocc_cas_a
    nmo = mf.mo_coeff.shape[1]

    C_mo_loc = mf.mo_coeff.copy()

    def _localize(C_block, label):
        """Localize a block of orbitals in-place; return localized block."""
        n = C_block.shape[1]
        if n <= 1:
            return C_block   # nothing to localize
        if method.lower() in ('pipek-mezey', 'pm'):
            mlo = lo.PipekMezey(mol, C_block)
            mlo.verbose = 0
            C_loc = mlo.kernel()
            for _ in range(3):
                C1 = mlo.stability_jacobi()[1]
                if C1 is C_loc:
                    break
                mlo = lo.PipekMezey(mol, C1)
                mlo.verbose = 0
                mlo.init_guess = None
                C_loc = mlo.kernel()
        elif method.lower() == 'boys':
            mlo = lo.Boys(mol, C_block)
            mlo.verbose = 0
            C_loc = mlo.kernel()
        else:
            C_loc = C_block
        log.info('  %s: %d orbitals localized', label, n)
        return C_loc

    # 1. Doubly occupied inactive (core): frozen:ncore
    if ncore > frozen:
        C_mo_loc[:, frozen:ncore] = _localize(
            mf.mo_coeff[:, frozen:ncore], 'inactive occupied')

    # 2. Doubly occupied active occupied: ncore:nocc_full
    if nocc_cas_a > 0:
        if occ_natural is not None:
            # Split doubly/singly occupied from natural orbital occupations
            # occ_natural[p] = natural occupation of active MO p (0..nocc_cas_a-1)
            doubly = occ_natural > 1.0    # occupation > 1 → doubly occupied
            singly = (occ_natural > 0.1) & (occ_natural <= 1.0)
            C_act_occ = mf.mo_coeff[:, ncore:nocc_full]
            if np.sum(doubly) > 0:
                C_mo_loc[:, ncore:ncore + np.sum(doubly)] = _localize(
                    C_act_occ[:, doubly], 'doubly occupied active')
            if np.sum(singly) > 0:
                start = ncore + np.sum(doubly)
                C_mo_loc[:, start:start + np.sum(singly)] = _localize(
                    C_act_occ[:, singly], 'singly occupied active')
        else:
            # RHF: all active occupied are doubly occupied
            C_mo_loc[:, ncore:nocc_full] = _localize(
                mf.mo_coeff[:, ncore:nocc_full], 'doubly occupied active')

    # 3. Active virtual: nocc_full:nocc_full+nvir_cas
    if nvir_cas > 0:
        C_mo_loc[:, nocc_full:nocc_full + nvir_cas] = _localize(
            mf.mo_coeff[:, nocc_full:nocc_full + nvir_cas], 'active virtual')

    # External virtual (nocc_full+nvir_cas:) left canonical

    # C_lmo = all occupied LMOs from frozen to nocc_full (for DLPNO pairs)
    C_lmo = C_mo_loc[:, frozen:nocc_full]

    log.info('Split localization complete: inactive=%d, active_occ=%d, '
             'active_vir=%d', max(0, ncore - frozen), nocc_cas_a, nvir_cas)

    return C_mo_loc, C_lmo


def build_localized_mo_coeff(mf, ncas, nelec_cas, C_lmo, method='pipek-mezey',
                             frozen=0):
    """Build full MO coefficient matrix with localized active space for CASCI.

    Takes the RHF canonical MO matrix and replaces the active space columns
    (ncore:ncore+ncas) with localized versions, so DMRG-CI runs in a local
    basis consistent with the DLPNO treatment.

    Active occupied (ncore:nocc) are taken from C_lmo (already localized by
    make_lmos). Active virtual (nocc:nocc+nvir_cas) are localized here with
    the same method.

    Args:
        mf: Converged RHF object.
        ncas (int): Number of active orbitals.
        nelec_cas (tuple): (n_alpha, n_beta) active electrons.
        C_lmo (np.ndarray): (nao, nocc_cas) Localized active occupied MOs from
            make_lmos(mf, frozen=ncore).
        method (str): Localization method for active virtual MOs.
        frozen (int): Number of frozen core orbitals (= ncore).

    Returns:
        C_full (np.ndarray): (nao, nmo) Full MO matrix with localized active
            space orbitals at columns ncore:ncore+ncas.
    """
    mol = mf.mol
    nocc_full = np.count_nonzero(mf.mo_occ > 1e-10)
    nocc_cas_a = nelec_cas[0]   # alpha electrons in CAS = active occupied
    ncore = nocc_full - nocc_cas_a
    nvir_cas = ncas - nocc_cas_a

    C_full = mf.mo_coeff.copy()

    # Replace active occupied columns with C_lmo
    # C_lmo has shape (nao, nocc_cas_a) from make_lmos(mf, frozen=ncore)
    assert C_lmo.shape[1] == nocc_cas_a, (
        f'C_lmo.shape[1]={C_lmo.shape[1]} != nocc_cas_a={nocc_cas_a}. '
        f'Call make_lmos(mf, frozen={ncore}) to get the active occupied LMOs.')
    C_full[:, ncore:nocc_full] = C_lmo

    # Localize active virtual MOs
    if nvir_cas > 1:
        C_vir_cas = mf.mo_coeff[:, nocc_full:nocc_full + nvir_cas]
        if method.lower() in ('pipek-mezey', 'pm'):
            mlo = lo.PipekMezey(mol, C_vir_cas)
            mlo.verbose = 0
            C_vir_cas_loc = mlo.kernel()
        elif method.lower() == 'boys':
            mlo = lo.Boys(mol, C_vir_cas)
            mlo.verbose = 0
            C_vir_cas_loc = mlo.kernel()
        else:
            C_vir_cas_loc = C_vir_cas   # keep canonical
        C_full[:, nocc_full:nocc_full + nvir_cas] = C_vir_cas_loc

    return C_full


def make_iaos(mf, C_lmo=None):
    """Construct Intrinsic Atomic Orbitals as alternative fragment basis.

    IAOs are constructed using the Knizia (JCTC 2013) method as implemented
    in pyscf.lo.iao. They provide a cleaner atom-centred fragment definition
    than PM orbitals for transition-metal systems.

    Args:
        mf: Converged RHF object.
        C_lmo (np.ndarray, optional): If given, only the occupied space spanned
            by C_lmo is used; otherwise mf.mo_coeff occ block is used.

    Returns:
        C_iao (np.ndarray): Shape (nao, niao). IAO coefficients.
        frag_iao_list (list): Default fragment assignment: each atom's IAOs
            form one fragment.
    """
    from pyscf.lo import iao as iao_module

    mol = mf.mol
    if C_lmo is None:
        nocc = np.count_nonzero(mf.mo_occ > 1e-10)
        mo_occ = mf.mo_coeff[:, :nocc]
    else:
        mo_occ = C_lmo

    C_iao = iao_module.iao(mol, mo_occ)

    # Default fragment list: one fragment per atom
    iao_ao_list = iao_module.reference_mol(mol).ao_labels(fmt=None)
    natom = mol.natom()
    atom_to_iao = [[] for _ in range(natom)]
    for idx, (iatom, *_) in enumerate(iao_ao_list):
        atom_to_iao[iatom].append(idx)

    frag_iao_list = [lst for lst in atom_to_iao if lst]
    return C_iao, frag_iao_list


def pao_domain_union(pao_domains_i, pao_domains_j):
    """Return sorted union of two PAO domain index arrays."""
    return np.union1d(pao_domains_i, pao_domains_j)


def _pivoted_cholesky(S, tol=1e-8):
    """Pivoted Cholesky decomposition to select important basis functions.

    Iteratively selects pivot indices (basis functions with largest residual
    diagonal) until the residual falls below tol. Matches Psi4's
    Matrix::pivoted_cholesky() used in PartialCholesky orthogonalization.

    Args:
        S: (n, n) symmetric positive semi-definite overlap matrix.
        tol: threshold for residual diagonal.

    Returns:
        pivots: sorted list of selected pivot indices.
    """
    n = S.shape[0]
    d = np.diag(S).copy()
    pivots = []
    L = np.zeros((n, n))

    for k in range(n):
        idx = np.argmax(d)
        if d[idx] < tol:
            break
        pivots.append(idx)
        L[idx, k] = np.sqrt(d[idx])
        for i in range(n):
            if i == idx:
                continue
            L[i, k] = (S[i, idx] - np.dot(L[i, :k], L[idx, :k])) / L[idx, k]
        d -= L[:, k] ** 2
        d = np.maximum(d, 0.0)

    return sorted(pivots)


def orthogonalize_pao_domain(C_pao, S_pao, domain_idx, S_cut=1e-6,
                             method='cholesky'):
    """Orthogonalize PAOs within a domain.

    Three methods available:
    - 'canonical': standard eigenvalue-based orthogonalization (PySCF default)
    - 'cholesky': pivoted Cholesky-based selection followed by canonical orth
    - 'psi4': Exact Psi4 PartialCholesky algorithm. Matches
      `BasisSetOrthogonalization::compute_partial_cholesky_orthog` in
      `libmints/orthog.cc`. Steps:
        1. Normalize the overlap matrix (S → S / sqrt(diag) outer-product)
        2. Sort columns by INCREASING off-diagonal sum (low-overlap first)
        3. LAPACK pivoted Cholesky (DPSTRF) with cutoff `S_cut`
        4. Canonical orthogonalization on the reduced sub-matrix with
           lindep_tol = 0 (Psi4 line 234: lindep_tolerance is 0.0)
        5. Pad back to original ordering
      This gives the EXACT same canonical PAO basis dimension as Psi4.

    Args:
        C_pao: (nao, npao) PAO coefficients.
        S_pao: (npao, npao) PAO overlap matrix.
        domain_idx: integer array of PAO indices in domain.
        S_cut: threshold for removing linear dependencies.
        method: 'canonical', 'cholesky', or 'psi4'.

    Returns:
        C_orth: (nao, n_orth) orthogonal PAOs in AO basis.
        X_orth: (n_domain, n_orth) domain PAO → orth transformation.
    """
    S_dom = S_pao[np.ix_(domain_idx, domain_idx)]
    n_dom = len(domain_idx)

    if method == 'psi4':
        # === Psi4 PartialCholesky exactly ===
        from scipy.linalg.lapack import dpstrf
        # 1. Normalize: S_norm[i,j] = S[i,j] / sqrt(S[i,i]*S[j,j])
        diag = np.diag(S_dom).copy()
        diag = np.where(diag > 0, diag, 1.0)
        norm = 1.0 / np.sqrt(diag)
        S_norm = S_dom * norm[:, None] * norm[None, :]
        # 2. Sort columns by increasing off-diagonal sum
        od = np.sum(np.abs(S_norm), axis=1) - np.abs(np.diag(S_norm))
        order = np.argsort(od, kind='stable')
        S_reord = S_norm[np.ix_(order, order)]
        # 3. LAPACK pivoted Cholesky (DPSTRF) with tol = S_cut
        c, piv, rank_c, info = dpstrf(S_reord.copy(), tol=S_cut, lower=1)
        if rank_c == 0:
            return np.zeros((C_pao.shape[0], 0)), np.zeros((n_dom, 0))
        # piv is 1-indexed Fortran; convert to 0-indexed
        cholesky_pivots_in_reord = (piv[:rank_c] - 1).tolist()
        # Translate from reordered indices back to original domain indices
        pivots = sorted(order[i] for i in cholesky_pivots_in_reord)
        # 4. Build sub-overlap matrix on selected pivots
        S_sub = S_norm[np.ix_(pivots, pivots)]
        # Canonical orthogonalization with lindep_tol = 0 (keep all > 0)
        eigvals, eigvecs = np.linalg.eigh(S_sub)
        keep = eigvals > 0.0
        if not np.any(keep):
            return np.zeros((C_pao.shape[0], 0)), np.zeros((n_dom, 0))
        X_sub = eigvecs[:, keep] / np.sqrt(eigvals[keep])
        # 5. Pad back to full domain (in normalized basis)
        X_orth = np.zeros((n_dom, X_sub.shape[1]))
        for m in range(X_sub.shape[1]):
            for k_idx, p in enumerate(pivots):
                X_orth[p, m] = X_sub[k_idx, m]
        # Unroll normalization: scale rows by 1/sqrt(diag) (Psi4 line 87-88)
        X_orth = X_orth * norm[:, None]

    elif method == 'cholesky':
        # Step 1: Pivoted Cholesky to select important PAOs
        pivots = _pivoted_cholesky(S_dom, tol=S_cut)
        n_chol = len(pivots)
        if n_chol == 0:
            return np.zeros((C_pao.shape[0], 0)), np.zeros((n_dom, 0))

        # Step 2: Canonical orthogonalization of Cholesky subset
        S_sub = S_dom[np.ix_(pivots, pivots)]
        eigvals, eigvecs = np.linalg.eigh(S_sub)
        keep = eigvals > S_cut
        if not np.any(keep):
            return np.zeros((C_pao.shape[0], 0)), np.zeros((n_dom, 0))
        X_sub = eigvecs[:, keep] / np.sqrt(eigvals[keep])

        # Step 3: Pad back to full domain
        X_orth = np.zeros((n_dom, X_sub.shape[1]))
        for m in range(X_sub.shape[1]):
            for k_idx, p in enumerate(pivots):
                X_orth[p, m] = X_sub[k_idx, m]

    else:
        # Standard canonical orthogonalization
        eigvals, eigvecs = np.linalg.eigh(S_dom)
        keep = eigvals > S_cut
        if not np.any(keep):
            return np.zeros((C_pao.shape[0], 0)), np.zeros((n_dom, 0))
        X_orth = eigvecs[:, keep] / np.sqrt(eigvals[keep])

    C_dom = C_pao[:, domain_idx]
    C_orth = np.dot(C_dom, X_orth)

    return C_orth, X_orth
