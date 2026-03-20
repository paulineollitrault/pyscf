"""
DMRG interface and cluster amplitude extraction for DLPNO-TCCSD(T).

Provides two modes:
  - run_dmrg_casci: DMRG-CI in a fixed (pre-localized) MO basis, NO orbital
    optimization. This is the correct mode for DLPNO-TCCSD(T): orbitals are
    localized first, then DMRG solves the CI problem in that fixed basis so
    the extracted amplitudes are expressed in the same localized basis as the
    DLPNO correlation treatment.
  - run_dmrg_casscf: DMRG-CASSCF with orbital optimization. Useful when
    starting from a poor initial guess (e.g. for large TM complexes), but
    the resulting amplitudes are in CASSCF-optimized (non-local) MOs and
    cannot be directly used in the DLPNO step.

Amplitude extraction follows the cluster decomposition of the RDMs:
    t1_cas[i,a] = gamma1[i,a+nocc_cas] / (eps_i - eps_a)
    t2_cas[i,j,a,b] from 2-RDM after subtracting t1*t1 disconnected part

References:
    Lee & Head-Gordon, J. Chem. Theory Comput. 2019, 15, 4S94 (Lee's repo)
    Lang et al., JCTC 2020, 16, 3028 (DLPNO-TCCSD)
"""

import numpy as np
from functools import reduce
from pyscf import lib
from pyscf.lib import logger


def run_dmrg_casci(mol, mf, ncas, nelec_cas, mo_coeff_loc,
                   maxM=1000, scratch='./dmrg_scratch', verbose=None):
    """Run DMRG-CI in a fixed localized MO basis (no orbital optimization).

    Uses CASCI (not CASSCF) so MO coefficients are frozen at mo_coeff_loc.
    This is the correct mode for DLPNO-TCCSD(T): orbitals must be localized
    BEFORE DMRG so the extracted CAS amplitudes live in the same local basis
    as the DLPNO correlation treatment.

    Args:
        mol: PySCF Mole object.
        mf: Converged RHF object.
        ncas (int): Number of active (CAS) orbitals.
        nelec_cas (tuple): (n_alpha, n_beta) electrons in CAS.
        mo_coeff_loc (np.ndarray): (nao, nmo) Full MO coefficient matrix with
            the active space columns (ncore:ncore+ncas) already localized.
        maxM (int): Maximum DMRG bond dimension.
        scratch (str): DMRG scratch directory.
        verbose (int, optional): Verbosity level.

    Returns:
        mc: Converged CASCI object with mc.mo_coeff (= mo_coeff_loc), mc.ci,
            mc.e_tot set. mc.fcisolver is a block2 DMRGci object.
    """
    log = logger.new_logger(mf, verbose)

    try:
        from pyscf import mcscf
        from pyscf.dmrgscf import DMRGCI
    except ImportError:
        raise ImportError(
            'block2 / pyscf-dmrgscf not found. Install with:\n'
            '  pip install block2\n'
            '  pip install pyscf-dmrgscf'
        )

    import os
    scratch = os.path.abspath(scratch)
    os.makedirs(scratch, exist_ok=True)

    mc = mcscf.CASCI(mf, ncas, nelec_cas)
    mc.fcisolver = DMRGCI(mol, maxM=maxM, tol=1e-10)
    mc.fcisolver.runtimeDir = scratch
    mc.fcisolver.scratchDirectory = scratch
    mc.fcisolver.threads = lib.num_threads()
    mc.fcisolver.memory = int(mf.max_memory * 0.8 / 1000)

    mc.fcisolver.scheduleSweeps = [0, 4, 8, 12, 16, 20]
    mc.fcisolver.scheduleMaxMs = [
        min(maxM // 4, 100),
        min(maxM // 2, 250),
        min(3 * maxM // 4, 500),
        maxM, maxM, maxM,
    ]
    mc.fcisolver.scheduleNoises = [1e-4, 1e-4, 1e-5, 1e-5, 0.0, 0.0]
    mc.fcisolver.scheduleTols = [1e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10]
    mc.fcisolver.twodot_to_onedot = 18

    mc.verbose = verbose if verbose is not None else mol.verbose

    log.info('Running DMRG-CI (no orbital opt): ncas=%d nelec=%s maxM=%d',
             ncas, nelec_cas, maxM)
    mc.kernel(mo_coeff_loc)   # pass fixed MO coefficients

    log.note('DMRG-CI E_tot = %.15g', mc.e_tot)
    return mc


def run_dmrg_casscf(mol, mf, ncas, nelec_cas, maxM=1000,
                    basis=None, scratch='./dmrg_scratch', verbose=None):
    """Run DMRG-CASSCF and return converged mc object.

    Uses block2 SU2 spin-adapted DMRG via the PySCF DMRGSCF interface.

    Args:
        mol: PySCF Mole object.
        mf: Converged RHF object (must use mc.mo_coeff as MO basis downstream).
        ncas (int): Number of CAS orbitals.
        nelec_cas (tuple): (n_alpha, n_beta) electrons in CAS.
        maxM (int): Maximum bond dimension for DMRG sweep schedule.
        basis (str, optional): Unused here (basis already in mol). Kept for API
            consistency with driver.py signature.
        scratch (str): DMRG scratch directory.
        verbose (int, optional): Verbosity level.

    Returns:
        mc: Converged CASSCF object with mc.mo_coeff, mc.ci, mc.e_tot set.
            mc.fcisolver is a block2 DMRGci object exposing make_rdm12.
    """
    log = logger.new_logger(mf, verbose)

    try:
        from pyscf import mcscf
        from pyscf.dmrgscf import DMRGCI, dmrgci
    except ImportError:
        raise ImportError(
            'block2 / pyscf-dmrgscf not found. Install with:\n'
            '  pip install block2\n'
            '  pip install pyscf-dmrgscf'
        )

    import os
    scratch = os.path.abspath(scratch)
    os.makedirs(scratch, exist_ok=True)

    # Set up DMRG solver with SU2 symmetry
    mc = mcscf.CASSCF(mf, ncas, nelec_cas)
    mc.fcisolver = DMRGCI(mol, maxM=maxM, tol=1e-10)
    mc.fcisolver.runtimeDir = scratch
    mc.fcisolver.scratchDirectory = scratch
    mc.fcisolver.threads = lib.num_threads()
    mc.fcisolver.memory = int(mf.max_memory * 0.8 / 1000)  # in GB

    # Sweep schedule: warmup → noise sweeps → final clean sweeps
    # Follows Olivares-Amaya et al. JCP 2015 recommended schedule
    mc.fcisolver.scheduleSweeps = [0, 4, 8, 12, 16, 20]
    mc.fcisolver.scheduleMaxMs = [
        min(maxM // 4, 100),
        min(maxM // 2, 250),
        min(3 * maxM // 4, 500),
        maxM, maxM, maxM,
    ]
    mc.fcisolver.scheduleNoises = [1e-4, 1e-4, 1e-5, 1e-5, 0.0, 0.0]
    mc.fcisolver.scheduleTols = [1e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-10]
    mc.fcisolver.twodot_to_onedot = 18

    mc.max_cycle_macro = 50
    mc.conv_tol = 1e-9
    mc.verbose = verbose if verbose is not None else mol.verbose

    log.info('Running DMRG-CASSCF: ncas=%d nelec=%s maxM=%d', ncas, nelec_cas, maxM)
    mc.kernel()

    if not mc.converged:
        log.warn('DMRG-CASSCF did not converge. Proceeding with current wavefunction.')

    log.note('DMRG-CASSCF E_tot = %.15g', mc.e_tot)
    return mc


def extract_amplitudes_from_mps(driver, ket, ncas, nelec_cas, nocc_cas, nvir_cas, log):
    """Extract exact CC amplitudes from a pyblock2 MPS via CI projection.

    Uses get_csf_coefficients (SZ mode) to read determinant coefficients
    directly from the MPS. Converts to CC amplitudes following Lang et al.
    eq (3)-(4):
        T_CAS^(1) = C^(1) / C0
        T_CAS^(2) = C^(2) / C0   (αβ component)

    Args:
        driver: pyblock2 DMRGDriver (SZ symmetry) with the converged MPS.
        ket: The converged MPS from DMRG-CI.
        ncas: Total number of CAS orbitals.
        nelec_cas: (nalpha, nbeta) active electrons.
        nocc_cas: Number of CAS occupied orbitals (= nalpha).
        nvir_cas: Number of CAS virtual orbitals (= ncas - nocc_cas).
        log: PySCF logger.

    Returns:
        t1_cas (np.ndarray): Shape (nocc_cas, nvir_cas).
        t2_cas (np.ndarray): Shape (nocc_cas, nocc_cas, nvir_cas, nvir_cas).
    """
    nalpha, nbeta = nelec_cas

    # Build the HF reference determinant in block2's SZ convention:
    # 0=empty, 1=alpha, 2=beta, 3=doubly occupied
    ref_det = np.zeros(ncas, dtype=np.uint8)
    ref_det[:nbeta] = 3
    ref_det[nbeta:nalpha] = 1

    # Build list of determinants to query
    dets_list = [ref_det.copy()]
    det_labels = [('ref', None)]

    # Singles: α excitation i→a
    for i in range(nocc_cas):
        for a in range(nvir_cas):
            a_orb = nocc_cas + a
            det = ref_det.copy()
            if det[i] == 3:
                det[i] = 2
            elif det[i] == 1:
                det[i] = 0
            else:
                continue
            if det[a_orb] == 0:
                det[a_orb] = 1
            elif det[a_orb] == 2:
                det[a_orb] = 3
            else:
                continue
            dets_list.append(det)
            det_labels.append(('s_alpha', (i, a)))

    # Singles: β excitation i→a
    for i in range(nocc_cas):
        for a in range(nvir_cas):
            a_orb = nocc_cas + a
            det = ref_det.copy()
            if det[i] == 3:
                det[i] = 1
            elif det[i] == 2:
                det[i] = 0
            else:
                continue
            if det[a_orb] == 0:
                det[a_orb] = 2
            elif det[a_orb] == 1:
                det[a_orb] = 3
            else:
                continue
            dets_list.append(det)
            det_labels.append(('s_beta', (i, a)))

    # Doubles: αβ excitation i(α)→a(α), j(β)→b(β)
    for i in range(nocc_cas):
        for a in range(nvir_cas):
            a_orb = nocc_cas + a
            for j in range(nocc_cas):
                for b in range(nvir_cas):
                    b_orb = nocc_cas + b
                    det = ref_det.copy()
                    # Remove alpha from i
                    if det[i] == 3:
                        det[i] = 2
                    elif det[i] == 1:
                        det[i] = 0
                    else:
                        continue
                    # Remove beta from j
                    if det[j] == 3:
                        det[j] = 1
                    elif det[j] == 2:
                        det[j] = 0
                    else:
                        continue
                    # Add alpha to a
                    if det[a_orb] == 0:
                        det[a_orb] = 1
                    elif det[a_orb] == 2:
                        det[a_orb] = 3
                    else:
                        continue
                    # Add beta to b
                    if det[b_orb] == 0:
                        det[b_orb] = 2
                    elif det[b_orb] == 1:
                        det[b_orb] = 3
                    else:
                        continue
                    dets_list.append(det)
                    det_labels.append(('d_ab', (i, j, a, b)))

    dets_array = np.array(dets_list, dtype=np.uint8)
    n_singles = sum(1 for l, _ in det_labels if l.startswith('s_'))
    n_doubles = sum(1 for l, _ in det_labels if l == 'd_ab')
    log.info('Extracting %d CI coefficients from MPS '
             '(1 ref + %d singles + %d doubles)',
             len(dets_list), n_singles, n_doubles)

    # Query all determinant coefficients at once
    _, dvals = driver.get_csf_coefficients(
        ket, cutoff=0.0, given_dets=dets_array, iprint=0)

    # Extract C0
    C0 = float(dvals[0])
    log.info('C0 (HF coefficient from MPS) = %.8f  (|C0|^2 = %.6f)', C0, C0**2)
    if abs(C0) < 1e-10:
        raise ValueError(
            f'Reference CI coefficient C0 ≈ 0 (got {C0:.3e}). '
            'The HF determinant has negligible weight in the DMRG wavefunction.')

    # ------------------------------------------------------------------
    # Phase correction: block2 uses site-ordered determinants (α,β
    # interleaved per site) while PySCF FCI uses spin-ordered (all α
    # first, then all β).  The reordering phase (-1)^{ΔN_swap} accounts
    # for this difference, where N_swap counts (β at k, α at k') pairs
    # with k < k' in the site-ordered product.
    # ------------------------------------------------------------------
    def _n_swap(det):
        """Count β-before-α reordering swaps for site-ordered → spin-ordered."""
        n = 0
        alpha_after = 0
        for k in range(len(det) - 1, -1, -1):
            if det[k] in (2, 3):  # β present
                n += alpha_after
            if det[k] in (1, 3):  # α present
                alpha_after += 1
        return n

    n_swap_ref = _n_swap(ref_det)

    # Build t1: average of α and β single excitation coefficients,
    # with the CC excitation phase and the block2→PySCF reordering phase.
    #   CC phase for single i→a: (-1)^i * (-1)^(nocc_cas-1)
    t1_cas = np.zeros((nocc_cas, nvir_cas))
    for idx, (label, data) in enumerate(det_labels):
        if label == 's_alpha':
            i, a = data
            dn = _n_swap(dets_list[idx]) - n_swap_ref
            ph_cc = (-1) ** (i + nocc_cas - 1)
            t1_cas[i, a] += ph_cc * (-1) ** dn * dvals[idx] / C0
        elif label == 's_beta':
            i, a = data
            dn = _n_swap(dets_list[idx]) - n_swap_ref
            ph_cc = (-1) ** (i + nocc_cas - 1)
            t1_cas[i, a] += ph_cc * (-1) ** dn * dvals[idx] / C0
    t1_cas *= 0.5  # average α and β

    # Build t2: αβ double excitation coefficients / C0.
    # Total phase = (-1)^(i+j) [CC excitation] × (-1)^{ΔN_swap} [reordering].

    t2_cas = np.zeros((nocc_cas, nocc_cas, nvir_cas, nvir_cas))
    for idx, (label, data) in enumerate(det_labels):
        if label == 'd_ab':
            i, j, a, b = data
            dn = _n_swap(dets_list[idx]) - n_swap_ref
            phase = (-1) ** (i + j + dn)
            t2_cas[i, j, a, b] = phase * dvals[idx] / C0

    log.note('CAS amplitudes extracted from MPS via exact CI projection '
             '(Lang et al. eq 3-4).')
    return t1_cas, t2_cas


def _ci_to_t2(ci, nocc_cas, ncas, nelec_cas):
    """Extract t2 amplitudes from a FCI CI vector by direct projection.

    Implements the TCCSD amplitude extraction formula (Kinoshita et al.
    J. Chem. Phys. 123, 074106, 2005):

        t2[i,j,a,b] = <Φ_{i→a(α), j→b(β)} | Ψ_CI / C0>

    where C0 = <Φ_0|Ψ_CI> is the reference CI coefficient.  This is the
    αβ spin component, which maps directly to PySCF's spatial t2 convention.

    The symmetry t2[i,j,a,b] = t2[j,i,b,a] is satisfied automatically for
    a singlet CI vector (α↔β symmetry of mc.ci).

    Args:
        ci: 2-D numpy CI vector, shape (nalpha_dets, nbeta_dets).
        nocc_cas: Number of occupied CAS orbitals.
        ncas: Total number of CAS orbitals.
        nelec_cas: (nalpha, nbeta) active electrons.

    Returns:
        t2: shape (nocc_cas, nocc_cas, nvir_cas, nvir_cas).
        C0: reference CI coefficient (scalar).
    """
    from pyscf.fci import cistring

    nvir_cas = ncas - nocc_cas
    nalpha, nbeta = nelec_cas

    alpha_strs = cistring.make_strings(range(ncas), nalpha)
    beta_strs  = cistring.make_strings(range(ncas), nbeta)

    # Reference: lowest nocc_cas orbitals occupied  (binary 00...011...1)
    ref_str = (1 << nocc_cas) - 1

    ref_alpha_idx = int(np.where(alpha_strs == ref_str)[0][0])
    ref_beta_idx  = int(np.where(beta_strs  == ref_str)[0][0])
    C0 = float(ci[ref_alpha_idx, ref_beta_idx])

    if abs(C0) < 1e-12:
        raise ValueError(
            f'Reference CI coefficient C0 ≈ 0 (got {C0:.3e}). '
            'The dominant determinant is not the HF reference; '
            'check that active-space occupied MOs have the lowest indices.')

    # String → array-index lookup
    alpha_str_to_idx = {int(s): idx for idx, s in enumerate(alpha_strs)}
    beta_str_to_idx  = {int(s): idx for idx, s in enumerate(beta_strs)}

    # Fermionic phase: (-1)^(# occupied orbitals with index < p in string s)
    def _phase(s, p):
        return (-1) ** bin(s & ((1 << p) - 1)).count('1')

    t2 = np.zeros((nocc_cas, nocc_cas, nvir_cas, nvir_cas))

    for i in range(nocc_cas):
        s_after_ann_i = ref_str & ~(1 << i)
        ph_ann_i = _phase(ref_str, i)

        for a in range(nvir_cas):
            a_orb = nocc_cas + a
            exc_alpha = s_after_ann_i | (1 << a_orb)
            alpha_idx = alpha_str_to_idx.get(exc_alpha, -1)
            if alpha_idx < 0:
                continue
            ph_alpha = ph_ann_i * _phase(s_after_ann_i, a_orb)

            for j in range(nocc_cas):
                s_after_ann_j = ref_str & ~(1 << j)
                ph_ann_j = _phase(ref_str, j)

                for b in range(nvir_cas):
                    b_orb = nocc_cas + b
                    exc_beta = s_after_ann_j | (1 << b_orb)
                    beta_idx = beta_str_to_idx.get(exc_beta, -1)
                    if beta_idx < 0:
                        continue
                    ph_beta = ph_ann_j * _phase(s_after_ann_j, b_orb)

                    t2[i, j, a, b] = (ph_alpha * ph_beta
                                      * ci[alpha_idx, beta_idx] / C0)
    return t2, C0


def _ci_to_t1(ci, nocc_cas, ncas, nelec_cas, C0):
    """Extract t1 amplitudes from CI vector by projection onto singly-excited dets.

    t1[i,a] = (1/sqrt(2)) * sum_spin <Φ_{i→a(σ)} | Ψ_CI / C0>

    For CASSCF with HF reference and Brillouin condition satisfied, t1 ≈ 0.
    This function is provided for completeness.
    """
    from pyscf.fci import cistring

    nvir_cas = ncas - nocc_cas
    nalpha, nbeta = nelec_cas

    alpha_strs = cistring.make_strings(range(ncas), nalpha)
    beta_strs  = cistring.make_strings(range(ncas), nbeta)

    ref_str = (1 << nocc_cas) - 1
    ref_alpha_idx = int(np.where(alpha_strs == ref_str)[0][0])
    ref_beta_idx  = int(np.where(beta_strs  == ref_str)[0][0])

    alpha_str_to_idx = {int(s): idx for idx, s in enumerate(alpha_strs)}
    beta_str_to_idx  = {int(s): idx for idx, s in enumerate(beta_strs)}

    def _phase(s, p):
        return (-1) ** bin(s & ((1 << p) - 1)).count('1')

    t1 = np.zeros((nocc_cas, nvir_cas))

    for i in range(nocc_cas):
        s_after_ann = ref_str & ~(1 << i)
        ph_ann = _phase(ref_str, i)

        for a in range(nvir_cas):
            a_orb = nocc_cas + a
            exc_str = s_after_ann | (1 << a_orb)
            ph_cre = _phase(s_after_ann, a_orb)
            phase = ph_ann * ph_cre

            # α excitation: ci[exc_alpha, ref_beta]
            alpha_idx = alpha_str_to_idx.get(exc_str, -1)
            if alpha_idx >= 0:
                t1[i, a] += phase * ci[alpha_idx, ref_beta_idx] / C0

            # β excitation: ci[ref_alpha, exc_beta]
            beta_idx = beta_str_to_idx.get(exc_str, -1)
            if beta_idx >= 0:
                t1[i, a] += phase * ci[ref_alpha_idx, beta_idx] / C0

        t1[i, :] *= 0.5  # average of α and β contributions

    return t1


def get_cas_amplitudes(mc, verbose=None):
    """Extract cluster amplitudes t1_cas and t2_cas from a CASSCF/DMRG wavefunction.

    Uses the TCCSD projection formula (Kinoshita et al. JCP 123, 074106, 2005):

        t2[i,j,a,b] = <Φ_{i→a(α), j→b(β)} | Ψ_CI / C0>

    where C0 = <Φ_0|Ψ_CI> is the leading CI coefficient (HF reference).
    For a standard FCI solver mc.ci is a 2-D numpy array and the formula is
    evaluated exactly.  For DMRG (mc.ci not a 2-D array), a fallback based on
    the α-β spin-resolved 2-RDM is used:

        t2[i,j,a,b] ≈ dm2_αβ[a+nocc, i, b+nocc, j] / C0²   (leading order)

    Index convention (all indices in the FULL MO list, 0-based):
        ncore    = mc.ncore
        nocc_cas = mc.nelecas[0]
        occ_cas_idx[p] = ncore + p
        vir_cas_idx[a] = ncore + nocc_cas + a

    Args:
        mc: Converged CASSCF/CASCI/DMRGSCF object.
        verbose: Verbosity level.

    Returns:
        t1_cas (np.ndarray): Shape (nocc_cas, nvir_cas).
        t2_cas (np.ndarray): Shape (nocc_cas, nocc_cas, nvir_cas, nvir_cas).
        occ_cas_idx (np.ndarray): CAS occupied orbital indices in full MO list.
        vir_cas_idx (np.ndarray): CAS virtual orbital indices in full MO list.
    """
    log = logger.new_logger(mc, verbose)

    ncore = mc.ncore
    ncas = mc.ncas
    nelec_cas = mc.nelecas  # (nalpha, nbeta)
    nocc_cas = nelec_cas[0]
    nvir_cas = ncas - nocc_cas

    occ_cas_idx = np.arange(ncore, ncore + nocc_cas)
    vir_cas_idx = np.arange(ncore + nocc_cas, ncore + ncas)

    log.info('CAS amplitude extraction: ncore=%d nocc_cas=%d nvir_cas=%d',
             ncore, nocc_cas, nvir_cas)

    if nvir_cas == 0:
        log.warn('CAS has no virtual orbitals. Returning zero amplitudes.')
        t1_cas = np.zeros((nocc_cas, 0))
        t2_cas = np.zeros((nocc_cas, nocc_cas, 0, 0))
        return t1_cas, t2_cas, occ_cas_idx, vir_cas_idx

    # ------------------------------------------------------------------
    # Primary path: direct CI projection (exact for standard FCI solver)
    # Try to get a 2-D CI array, reshaping from 1-D if needed.
    # ------------------------------------------------------------------
    ci = mc.ci
    ci_2d = None
    if isinstance(ci, np.ndarray):
        from pyscf.fci import cistring as _cistring
        ndet_a = _cistring.num_strings(ncas, nelec_cas[0])
        ndet_b = _cistring.num_strings(ncas, nelec_cas[1])
        if ci.ndim == 2 and ci.shape == (ndet_a, ndet_b):
            ci_2d = ci
        elif ci.ndim == 1 and ci.size == ndet_a * ndet_b:
            ci_2d = ci.reshape(ndet_a, ndet_b)
            log.debug('Reshaped 1-D CI vector (%d,) to 2-D (%d,%d)',
                      ci.size, ndet_a, ndet_b)

    if ci_2d is not None:
        log.debug('Using exact CI-vector projection for t2 extraction.')
        t2_cas, C0 = _ci_to_t2(ci_2d, nocc_cas, ncas, nelec_cas)
        t1_cas = _ci_to_t1(ci_2d, nocc_cas, ncas, nelec_cas, C0)
        log.debug('C0 = %.8f  (|C0|^2 = %.6f)', C0, C0**2)

    # ------------------------------------------------------------------
    # DMRG fallback: spin-free 2-RDM (approximate, kept for compatibility).
    # For production use, prefer extract_amplitudes_from_mps() with pyblock2.
    # ------------------------------------------------------------------
    else:
        log.warn('CI vector not accessible as 2-D array. Using approximate '
                 'spin-free 2-RDM formula for t2. For exact results, use '
                 'extract_amplitudes_from_mps() with pyblock2.')
        dm1, dm2 = mc.fcisolver.make_rdm12(mc.ci, ncas, nelec_cas)
        n_occ_diag = np.diag(dm1)[:nocc_cas] / 2.0
        n_vir_diag = np.diag(dm1)[nocc_cas:] / 2.0
        C0_sq = float(np.clip(np.prod(n_occ_diag) * np.prod(1.0 - n_vir_diag),
                              1e-6, 1.0))
        C0 = np.sqrt(C0_sq)
        io = slice(0, nocc_cas)
        iv = slice(nocc_cas, ncas)
        dm2_aibj = dm2[iv, io, iv, io]
        dm2_biaj = dm2_aibj.transpose(2, 1, 0, 3)
        t2_cas = np.einsum('aibj->ijab',
                           2.0 * dm2_aibj + dm2_biaj) / (6.0 * C0_sq)
        t1_cas = dm1[:nocc_cas, nocc_cas:] / (2.0 * C0)

    log.info('CAS amplitudes: |t1|=%.4g  |t2|=%.4g',
             np.linalg.norm(t1_cas), np.linalg.norm(t2_cas))
    log.debug('occ_cas_idx = %s', occ_cas_idx)
    log.debug('vir_cas_idx = %s', vir_cas_idx)

    return t1_cas, t2_cas, occ_cas_idx, vir_cas_idx


def cas_idx_to_full(mc):
    """Return index arrays mapping CAS sub-indices to full MO indices.

    Convenience function used by lccsd.py and lccsd_t.py.

    Returns:
        ncore, nocc_cas, nvir_cas, occ_cas_idx, vir_cas_idx
    """
    ncore = mc.ncore
    ncas = mc.ncas
    nocc_cas = mc.nelecas[0]
    nvir_cas = ncas - nocc_cas
    occ_cas_idx = np.arange(ncore, ncore + nocc_cas)
    vir_cas_idx = np.arange(ncore + nocc_cas, ncore + ncas)
    return ncore, nocc_cas, nvir_cas, occ_cas_idx, vir_cas_idx
