"""
Fragment CCSD kernel for DLPNO-TCCSD(T), with CAS amplitude injection.

This module implements the pair-space CCSD solve in the PNO basis, which is
the core computational step of DLPNO-TCCSD(T). It combines:

  1. Fragment CCSD infrastructure ported from Ye & Berkelbach lnocc/lnoccsd.py
     - fake_mf construction from local orbital subspace
     - DF integral transformation in PNO basis
     - Fragment energy projection via LO population weights

  2. CAS amplitude injection from Lee & Head-Gordon tccsd.py (ecCC-TCC)
     - For CAS pairs: overwrite t2new[i,j,a,b] with DMRG amplitudes
     - Implemented as a monkey-patched update_amps callback

Design: For each strong pair (i,j) not a CAS pair, we run a restricted CCSD
in the combined PNO space of that pair. For CAS pairs, we freeze the amplitudes
to the DMRG values by injecting t2_cas into every CCSD iteration.

The fragment energy is computed using the population-based projector:
    e_frag = sum_{ij, ab} t2[i,j,a,b] * K[i,j,a,b] * P_lo[i,frag]

where P_lo[i,frag] is the squared LMO projection onto the fragment.

References:
    Ye & Berkelbach, JCTC 2024, 20, 8948  [LNO-CCSD implementation]
    Lee & Head-Gordon, JCTC 2019, 15, 4594  [TCCSD t2 freeze]
    Lang et al., JCTC 2020, 16, 3028  [DLPNO-TCCSD]
"""

import numpy as np
from functools import reduce
from pyscf import lib, scf, ao2mo
from pyscf.ao2mo import _ao2mo
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc.ccsd import _ChemistsERIs


# ---------------------------------------------------------------------------
# Helpers for building a local CCSD object in the PNO basis
# ---------------------------------------------------------------------------

class _FakeMF:
    """Minimal mock mean-field object for PySCF CCSD.

    Stores precomputed integrals and orbital data for a pair's PNO subspace.
    Avoids the overhead of reinitializing a full PySCF RHF object for every
    fragment (following Ye's approach in lnoccsd.py get_e_hf / MODIFIED_CCSD).

    Important: get_fock() MUST return the full AO Fock (nao × nao) so that
    PySCF's CCSD can compute eris.fock = mo.T @ F_ao @ mo correctly.
    """
    def __init__(self, mol, mo_coeff_pair, mo_energy_pair, mo_occ_pair,
                 e_tot, fock_ao, _eri, verbose=0):
        self.mol = mol
        self.mo_coeff = mo_coeff_pair    # (nao, nmo_pair)
        self.mo_energy = mo_energy_pair  # (nmo_pair,)
        self.mo_occ = mo_occ_pair        # (nmo_pair,)  0 or 2
        self.e_tot = e_tot               # scalar (HF energy, from parent mf)
        self._fock_ao = fock_ao          # (nao, nao) AO Fock from parent RHF
        self._eri = _eri                 # 4c ERI in pair MO basis (compact)
        self.verbose = verbose
        self.stdout = None
        self.max_memory = 4000
        self.with_df = None
        # Attributes required by PySCF StreamObject / CCSD infrastructure
        self.chkfile = None
        self.converged = True
        self._keys = set()

    def get_ovlp(self):
        nao = self.mo_coeff.shape[0]
        return np.eye(nao)

    def get_hcore(self):
        # Approximate: h_core ≈ fock (good enough for CCSD which only uses fock)
        return self._fock_ao

    def get_fock(self, *args, **kwargs):
        # Return the full AO Fock so CCSD can compute eris.fock = mo.T @ F @ mo
        return self._fock_ao

    def make_rdm1(self, mo_coeff=None, mo_occ=None):
        if mo_coeff is None: mo_coeff = self.mo_coeff
        if mo_occ is None: mo_occ = self.mo_occ
        return np.dot(mo_coeff * mo_occ, mo_coeff.T)

    def istype(self, cls_or_name):
        """Check if this object is of a given type (needed by DFCCSD)."""
        if isinstance(cls_or_name, str):
            return cls_or_name in ('RHF', 'SCF')
        return isinstance(self, cls_or_name)

    def get_veff(self, mol=None, dm=None, *args, **kwargs):
        nao = self.mo_coeff.shape[0]
        return np.zeros((nao, nao))

    def energy_tot(self, dm=None, *args, **kwargs):
        return self.e_tot


def _make_pno_ccsd_eris(nocc_pair, nvir_pair, fock_pair, K_pno, T2_pno,
                         mo_energy_pair):
    """Build a minimal _ChemistsERIs-like object in the PNO basis.

    For a pair (i,j) with PNO virtual space, we have:
      - 2 occupied orbitals {i,j}  (in the local occupied subspace)
      - n_pno virtual orbitals     (PNOs for this pair)

    We need the CCSD ERIs: ovov, oovv, ovvo, ovoo, ovvv, oooo.
    These are constructed from the exchange integrals K_pno (the only
    two-electron integrals needed at MP2 level) and augmented with zeros
    for the remaining required blocks (consistent with the PNO basis where
    off-diagonal pair couplings enter through the LCCSD sigma-vector).

    For the full DLPNO-CCSD one would also need off-pair coupling integrals
    (n_ij residual couplings). Here we implement the simpler "pair-specific"
    approach where each pair is solved independently. Pair couplings are
    accounted for in the next iteration via sigma-vector terms.

    This is sufficient for the CAS amplitude injection purpose and closely
    follows what Ye's code does for the fragment impurity solver.

    Args:
        nocc_pair (int): Number of occupied orbitals in this pair (2 for ij).
        nvir_pair (int): Number of PNOs.
        fock_pair (np.ndarray): (nocc_pair+nvir_pair, nocc_pair+nvir_pair).
        K_pno (np.ndarray): (nvir_pair, nvir_pair) exchange integrals (ia|jb).
        T2_pno (np.ndarray): (nvir_pair, nvir_pair) MP2 doubles.
        mo_energy_pair (np.ndarray): (nocc_pair+nvir_pair,) diagonal Fock.

    Returns:
        eris: Object with attributes oooo, ovoo, oovv, ovov, ovvo, ovvv, vvvv,
              fock, mo_energy, nocc.
    """
    nocc = nocc_pair
    nvir = nvir_pair

    class _ERIs:
        pass

    eris = _ERIs()
    eris.nocc = nocc
    eris.fock = fock_pair
    eris.mo_energy = mo_energy_pair

    # For a pair (i,j), the exchange integrals are:
    # K[i,a,j,b] = (ia|jb)  ← these are the PNO pair integrals
    # We store them in chemist's ovov notation: eris.ovov[i,a,j,b] = (ia|jb)
    # For a 2-occupied system:
    #   ovov[nocc=2, nvir, nocc=2, nvir] where ovov[0,a,1,b] = K_ij[a,b]

    eris.oooo = np.zeros((nocc, nocc, nocc, nocc))
    eris.ovoo = np.zeros((nocc, nvir, nocc, nocc))
    eris.oovv = np.zeros((nocc, nocc, nvir, nvir))
    eris.ovov = np.zeros((nocc, nvir, nocc, nvir))
    eris.ovvo = np.zeros((nocc, nvir, nvir, nocc))
    eris.ovvv = np.zeros((nocc, nvir, nvir, nvir))
    eris.vvvv = np.zeros((nvir, nvir, nvir, nvir))

    # Fill (ia|jb) exchange: pair (0,a,1,b) = K[a,b], (1,a,0,b) = K[b,a]
    if nocc == 2 and nvir > 0:
        eris.ovov[0, :, 1, :] = K_pno
        eris.ovov[1, :, 0, :] = K_pno.T
        # ovvo[i,a,b,j] = (ia|jb) but transposed in last two → ovvo[i,a,b,j]=(ia|jb)
        # Standard relation: eris.ovvo[i,a,b,j] = eris.ovov[i,a,j,b]
        eris.ovvo[0, :, :, 1] = K_pno
        eris.ovvo[1, :, :, 0] = K_pno.T
        # oovv[i,j,a,b] = (ij|ab): for pair (0,1): (01|ab) = K_pno[a,b]
        # Note: (01|ab) ≠ (0a|1b) in general. We approximate by zero for
        # the off-diagonal oovv since we only have exchange integrals.
        # This is the standard "pair approximation" in DLPNO.

    return eris


# ---------------------------------------------------------------------------
# CAS amplitude injection (Tailored CC modification to update_amps)
# ---------------------------------------------------------------------------

def _make_cas_freeze_masks(nocc_pair, nvir_pair, i_cas_local, j_cas_local,
                            a_cas_local, b_cas_local):
    """Build boolean freeze masks for t1 and t2 CAS blocks.

    Args:
        nocc_pair (int): Number of occupied orbitals in pair subspace.
        nvir_pair (int): Number of virtual orbitals (PNOs) in pair subspace.
        i_cas_local, j_cas_local: Local occupied indices that are CAS.
        a_cas_local, b_cas_local: Local virtual indices that map to CAS vir.

    Returns:
        t1f (np.ndarray): (nocc_pair, nvir_pair) bool mask: True = freeze at CAS.
        t2f (np.ndarray): (nocc_pair, nocc_pair, nvir_pair, nvir_pair) bool mask.
    """
    t1f = np.zeros((nocc_pair, nvir_pair), dtype=bool)
    t2f = np.zeros((nocc_pair, nocc_pair, nvir_pair, nvir_pair), dtype=bool)

    # Mark CAS virtual block of singles
    np.ix_(i_cas_local, a_cas_local)  # for type-check only
    t1f[np.ix_(i_cas_local, a_cas_local)] = True

    # Mark full CAS (occ,occ,vir,vir) block of doubles
    t2f[np.ix_(i_cas_local, j_cas_local,
               a_cas_local, b_cas_local)] = True

    return t1f, t2f


# ---------------------------------------------------------------------------
# Fragment energy: population-projected CCSD energy
# ---------------------------------------------------------------------------

def _fragment_energy(oovv, t2, uocc_loc):
    """Compute fragment CCSD correlation energy via LMO projector.

    Follows get_fragment_energy in Ye's lnoccsd.py.
    The projector M = uocc_loc @ uocc_loc^T gives the weight of each pair (i,j)
    in the current fragment.

    Args:
        oovv (np.ndarray): (nocc, nocc, nvir, nvir) — K[i,j,a,b] = (ij|ab).
        t2 (np.ndarray): (nocc, nocc, nvir, nvir) — T2 + t1*t1 (tau).
        uocc_loc (np.ndarray): (nocc, n_lo) LO projection onto fragment.

    Returns:
        e_frag (float): Fragment correlation energy.
    """
    M = np.dot(uocc_loc, uocc_loc.T)  # (nocc, nocc)
    # e = 2*sum_ijab t2[i,j,a,b] * oovv[i,j,a,b] * M[i,j]
    #   -   sum_ijab t2[i,j,b,a] * oovv[i,j,a,b] * M[i,j]
    ed = 2.0 * np.einsum('ijab,ijab,ij->', t2, oovv, M)
    ex = -np.einsum('ijba,ijab,ij->', t2, oovv, M)
    return (ed + ex).real


# ---------------------------------------------------------------------------
# Inter-pair coupling helpers for the outer macro-iteration
# ---------------------------------------------------------------------------

def _project_t2_full(t2_kl, C_pno_kl, C_pno_ij, s1e):
    """Project T2[k,l,a,b] from PNO_{kl} to PNO_{ij} (both virtual indices).

    PNO coefficients satisfy C.T @ S @ C = I, so the projection is:
        T2_proj[a',b'] = S_{ij,kl}[a',a] * T2[a,b] * S_{kl,ij}[b,b']
    where S_{ij,kl} = C_pno_ij.T @ s1e @ C_pno_kl.
    """
    S = C_pno_ij.T @ (s1e @ C_pno_kl)  # (n_pno_ij, n_pno_kl)
    return S @ t2_kl @ S.T              # (n_pno_ij, n_pno_ij)


def _project_t2_half(t2_ik, C_pno_ik, C_pno_ij, s1e):
    """Project only the first virtual index of T2[i,k] from PNO_{ik} to PNO_{ij}.

    Returns shape (n_pno_ij, n_pno_ik): first index in PNO_ij, second in PNO_ik.
    """
    S = C_pno_ij.T @ (s1e @ C_pno_ik)  # (n_pno_ij, n_pno_ik)
    return S @ t2_ik                    # (n_pno_ij, n_pno_ik)


def _ovl_pno(lmo_idx, C_pno, C_pao, ovL_lmo_pao):
    """(lmo|pno,L) 3-index DF tensor in the PNO basis.

    ovL_lmo_pao[lmo, a_pao, L] stores (lmo,PAO|L) from pno._build_ovL.
    To get (lmo,PNO|L):
        ovL_ao[ν,L] = C_pao[ν,a] * ovL[lmo,a,L]   (PAO→AO back-transform)
        ovL_pno[b,L] = C_pno[ν,b] * ovL_ao[ν,L]
    """
    ovL_ao = C_pao @ ovL_lmo_pao[lmo_idx]  # (nao, naux)
    return C_pno.T @ ovL_ao                # (n_pno, naux)


def _compute_ladder(t2_ij, C_pno_ij, with_df):
    """Intra-pair ladder diagram: 0.5 * Σ_{cd} tau_{ij}[c,d] * (ac|bd).

    Uses DF to avoid building the full (npno^4) vvvv tensor:
        ladder[a,b] = 0.5 * Σ_L (Σ_c B[a,c,L]*t2[c,d]) * B[b,d,L]

    Returns (n_pno, n_pno) contribution to the T2 residual numerator.
    """
    n_pno = C_pno_ij.shape[1]
    if n_pno == 0:
        return np.zeros((0, 0))

    naux = with_df.get_naoaux()
    mo = np.asfortranarray(C_pno_ij)
    ijslice = (0, n_pno, 0, n_pno)

    # Contract on-the-fly: avoid storing full vvL
    ladder = np.zeros((n_pno, n_pno))
    buf = None
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]
        buf = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
        # B_L[a,c] for each L in this batch: shape (nL, n_pno, n_pno)
        B_L = buf.reshape(nL, n_pno, n_pno)
        # X_L[a,d] = Σ_c B_L[a,c] * t2[c,d] for each L
        X_L = np.einsum('Lac,cd->Lad', B_L, t2_ij)
        # ladder[a,b] += Σ_L X_L[a,d] * B_L[b,d]
        ladder += np.einsum('Lad,Lbd->ab', X_L, B_L)
        Lpq = None

    # Full factor (not halved): the caller builds the P(ij)-symmetrized
    # residual directly, and the ladder is already P(ij)-symmetric.
    return ladder


def _compute_foo_dressed_local(t2_pno_all, pno_spaces, nocc_lmo, cc_ints,
                               _pool=None):
    """Per-pair local-aux T2-dressed Fock occ-occ from cc_ints['Qma'].

    Equivalent to _compute_foo_dressed but skips ovL_pno_cache / with_df
    entirely. Uses cc_ints[key_mq]['Qma'] (n_local, nocc, n_pno) which
    holds fitted (Q | m a) for all m∈nocc and a∈PNO_mq with Q in the
    pair's local aux. The contraction is identical in structure to the
    legacy version, just summed over local-aux instead of full naux.
    """
    def _per_pair(key_mq):
        t2_mq_raw = t2_pno_all.get(key_mq)
        if t2_mq_raw is None or t2_mq_raw.shape[0] == 0:
            return key_mq, None
        ci = cc_ints.get(key_mq)
        if ci is None:
            return key_mq, None
        m, q = key_mq
        Qma = ci['Qma']
        ovL_m = Qma[:, m, :].T
        theta_mq = 2.0 * t2_mq_raw.T - t2_mq_raw
        X_mq = theta_mq @ ovL_m
        contrib_q = np.einsum('Lpa,aL->p', Qma, X_mq, optimize=True)
        contrib_m = None
        if m != q:
            theta_qm = 2.0 * t2_mq_raw - t2_mq_raw.T
            ovL_q = Qma[:, q, :].T
            X_qm = theta_qm @ ovL_q
            contrib_m = np.einsum('Lpa,aL->p', Qma, X_qm, optimize=True)
        return key_mq, (contrib_q, contrib_m)

    foo = np.zeros((nocc_lmo, nocc_lmo))
    pair_list = list(t2_pno_all.keys())
    if _pool is not None:
        results = list(_pool.map(_per_pair, pair_list))
    else:
        results = [_per_pair(k) for k in pair_list]
    for key_mq, payload in results:
        if payload is None:
            continue
        m, q = key_mq
        contrib_q, contrib_m = payload
        foo[:, q] += contrib_q
        if contrib_m is not None:
            foo[:, m] += contrib_m
    return foo


def _compute_foo_dressed(t2_pno_all, pno_spaces, nocc_lmo, with_df, C_lmo, s1e,
                         ovL_pno_cache=None):
    """Compute T2-dressed occupied Fock intermediate (global, once per iteration).

    PySCF: foo[i,j] += Σ_{a,k,b} voov[a,i,k,b] * theta[k,j,a,b]
    where theta[k,j,a,b] = 2*tau[j,k,a,b] - tau[k,j,a,b]  (tau=t2 with t1=0)

    In DLPNO: for each stored pair (m,n), theta involves t2_mn.
    foo[p,q] = Σ_{m ∈ partners(q)} Σ_{a,b∈PNO_{mq}} (ap|mb) * theta[m,q,a,b]

    Returns:
        foo_dressed (np.ndarray): (nocc, nocc) T2-dependent Fock correction.
    """
    if with_df is None:
        return np.zeros((nocc_lmo, nocc_lmo))

    foo = np.zeros((nocc_lmo, nocc_lmo))

    for key_mq, t2_mq_raw in t2_pno_all.items():
        if t2_mq_raw is None or t2_mq_raw.shape[0] == 0:
            continue
        m, q = key_mq  # m <= q
        C_pno_mq = pno_spaces[key_mq]['C_pno']
        n_mq = C_pno_mq.shape[1]
        if n_mq == 0:
            continue

        # theta[m,q,a,b] = 2*t2[q,m,a,b] - t2[m,q,a,b] = 2*t2_mq.T - t2_mq
        theta_mq = 2.0 * t2_mq_raw.T - t2_mq_raw

        # foo[p, q] += Σ_{a,b} (ap|mb) * theta_mq[a,b]  for all p
        # voov[a,p,m,b] = (a_{mq} p | m b_{mq})
        if ovL_pno_cache is not None:
            # Build voov_all_p from cached 3-index tensors:
            # voov[a,p,m,b] = ovL[mq,p][a,L] * ovL[mq,m][b,L]
            ovL_m = ovL_pno_cache.get((key_mq, m))
            if ovL_m is not None:
                # intermediate[b] = Σ_a theta[a,b] * ovL_m_a → contract later
                # foo[p,q] += Σ_{a,b} ovL_p[a,L] * ovL_m[b,L] * theta[a,b]
                #           = Σ_L (Σ_a ovL_p[a,L]*theta_col_L[a]) where
                #             theta_col_L needs ovL_m... Better:
                # foo[p,q] = Σ_L (ovL_p @ theta @ ovL_m.T)... no, trace.
                # voov[a,p,m,b] * theta[a,b] = ovL_p[a,L]*ovL_m[b,L]*theta[a,b]
                # Sum over a,b: = Σ_L (Σ_a ovL_p[a,L]*x_L) where x_L = Σ_b theta.T[b,a]*...
                # Simplify: result[p] = tr(ovL_p.T @ theta @ ovL_m) for each p
                # = Σ_L (ovL_p.T @ theta @ ovL_m)_LL... no.
                # Actually: Σ_{a,b} ovL_p[a,L] * theta[a,b] * ovL_m[b,L']  δ_{LL'}
                # = Σ_L (Σ_a ovL_p[a,L] * (Σ_b theta[a,b] * ovL_m[b,L]))
                # Let X[a,L] = Σ_b theta[a,b] * ovL_m[b,L] = theta @ ovL_m
                # Then result[p] = Σ_{a,L} ovL_p[a,L] * X[a,L] = tr(ovL_p.T @ X)
                #                = Σ_L (ovL_p.T @ X)_{L,L} ... no, just element-wise.
                # result[p] = Σ_{a,L} ovL_p[a,L] * X[a,L]
                # For all p simultaneously: build ovL_all_p of shape (n_mq, nocc, naux)?
                # That's what the original did. Instead, use the per-p cache:
                X_mq = theta_mq @ ovL_m  # (n_mq, naux)
                for p in range(nocc_lmo):
                    ovL_p = ovL_pno_cache.get((key_mq, p))
                    if ovL_p is not None:
                        foo[p, q] += np.einsum('aL,aL->', ovL_p, X_mq)
            else:
                # Fallback
                voov_all_p = with_df.ao2mo(
                    [C_pno_mq, C_lmo, C_lmo[:, [m]], C_pno_mq],
                    compact=False).reshape(n_mq, nocc_lmo, n_mq)
                foo[:, q] += np.einsum('apb,ab->p', voov_all_p, theta_mq)
        else:
            voov_all_p = with_df.ao2mo(
                [C_pno_mq, C_lmo, C_lmo[:, [m]], C_pno_mq],
                compact=False).reshape(n_mq, nocc_lmo, n_mq)
            foo[:, q] += np.einsum('apb,ab->p', voov_all_p, theta_mq)

        if m != q:
            # theta[q,m,a,b] = 2*t2[m,q,a,b] - t2[q,m,a,b] = 2*t2_mq - t2_mq.T
            theta_qm = 2.0 * t2_mq_raw - t2_mq_raw.T

            if ovL_pno_cache is not None:
                ovL_q = ovL_pno_cache.get((key_mq, q))
                if ovL_q is not None:
                    X_qm = theta_qm @ ovL_q  # (n_mq, naux)
                    for p in range(nocc_lmo):
                        ovL_p = ovL_pno_cache.get((key_mq, p))
                        if ovL_p is not None:
                            foo[p, m] += np.einsum('aL,aL->', ovL_p, X_qm)
                else:
                    voov_all_p2 = with_df.ao2mo(
                        [C_pno_mq, C_lmo, C_lmo[:, [q]], C_pno_mq],
                        compact=False).reshape(n_mq, nocc_lmo, n_mq)
                    foo[:, m] += np.einsum('apb,ab->p', voov_all_p2, theta_qm)
            else:
                voov_all_p2 = with_df.ao2mo(
                    [C_pno_mq, C_lmo, C_lmo[:, [q]], C_pno_mq],
                    compact=False).reshape(n_mq, nocc_lmo, n_mq)
                foo[:, m] += np.einsum('apb,ab->p', voov_all_p2, theta_qm)

    return foo


def _compute_foo_t1_dressed(t1_pno, pno_spaces, nocc_lmo, with_df, C_lmo,
                             ovL_pno_cache=None):
    """Compute T1-dressed occupied Fock correction (Eq. 98 of Jiang et al.).

    F̄_ij^{t1} = Σ_{k,c} [2(ij|kc) - (ic|kj)] t̃_k^c

    Using DF with 3-index tensors:
        ooL[i,j,Q] = (i_lmo j_lmo | Q)
        ovL_kk[c,Q] = (k_lmo c_kk | Q)

    Coulomb:  2 Σ_Q ooL[i,j,Q] * (ovL_kk.T @ t1_k)[Q]
    Exchange: -Σ_Q ovL_kk_i[c,Q] * t1_k[c] * ooL[k,j,Q]
              where ovL_kk_i = ovL_pno_cache[(kk, i)]

    Returns:
        foo_t1 (np.ndarray): (nocc, nocc) T1-dressed Fock correction.
    """
    foo_t1 = np.zeros((nocc_lmo, nocc_lmo))
    if with_df is None or ovL_pno_cache is None:
        return foo_t1

    # Build ooL[i,j,Q] = (i_lmo j_lmo | Q)  — reuse _build_ovL
    from pyscf.cc.dlpno_tccsd.pno import _build_ovL
    ooL = _build_ovL(with_df, C_lmo, C_lmo)  # (nocc, nocc, naux)

    for k in range(nocc_lmo):
        key_kk = (k, k)
        t1_k = t1_pno.get(k)
        if t1_k is None or t1_k.size == 0 or np.max(np.abs(t1_k)) < 1e-15:
            continue

        # ovL_kk_k[c, Q] = (k c_kk | Q)
        ovL_kk_k = ovL_pno_cache.get((key_kk, k))
        if ovL_kk_k is None:
            continue

        # z_coul[Q] = Σ_c ovL_kk_k[c,Q] * t1_k[c]
        z_coul = ovL_kk_k.T @ t1_k  # (naux,)

        # Coulomb: foo_t1[i,j] += 2 * Σ_Q ooL[i,j,Q] * z_coul[Q]
        foo_t1 += 2.0 * np.einsum('ijQ,Q->ij', ooL, z_coul)

        # Exchange: foo_t1[i,j] -= Σ_Q z_exch_i[Q] * ooL[k,j,Q]
        # z_exch_i[Q] = Σ_c ovL_kk_i[c,Q] * t1_k[c]
        # where ovL_kk_i = ovL_pno_cache[(kk, i)] = (i c_kk | Q)
        for i_idx in range(nocc_lmo):
            ovL_kk_i = ovL_pno_cache.get((key_kk, i_idx))
            if ovL_kk_i is None:
                continue
            z_exch = ovL_kk_i.T @ t1_k  # (naux,)
            foo_t1[i_idx, :] -= np.einsum('Q,jQ->j', z_exch, ooL[k])

    return foo_t1


def _project_t1_to_pair(t1_pno, i, key_kl, S_pno_cache, pno_spaces):
    """Project t1_pno[i] from diagonal PNO basis (i,i) to PNO basis of pair kl.

    Implements Eq. 70 of Jiang et al. JCP 2024:
        t̃_i^{a_kl} = S_{a_kl, a_ii}^{PNO} · t_i^{a_ii}

    Args:
        t1_pno (dict): i → np.ndarray(n_pno_ii,) T1 in diagonal PNO basis.
        i (int): Occupied LMO index.
        key_kl (tuple): (k, l) pair key with k <= l.
        S_pno_cache (dict): (key1, key2) → PNO overlap matrix.
        pno_spaces (dict): PNO space data.

    Returns:
        np.ndarray: (n_pno_kl,) T1 of LMO i projected into PNO_{kl} basis.
    """
    key_ii = (i, i)
    t1_i = t1_pno.get(i)
    if t1_i is None or t1_i.size == 0:
        return np.zeros(pno_spaces[key_kl]['C_pno'].shape[1])
    if key_ii == key_kl:
        return t1_i.copy()
    S = S_pno_cache.get((key_kl, key_ii))
    if S is not None:
        return S @ t1_i
    # Fallback: should not happen if cache is built correctly
    return np.zeros(pno_spaces[key_kl]['C_pno'].shape[1])


def _compute_t1_residual_psi4(t1_pno, t2_pno_all, pno_spaces,
                               fov_pno, F_lmo, eps_lmo,
                               nocc, S_pno_cache, cc_ints,
                               ovL_pno_cache=None, pair_lmo_idx=None,
                               t1_cache=None):
    """T1 residual EXACTLY matching Psi4's structure (DePrince Eqs 19-22).

    R[i, a_ii] = Fai[i,a_ii] + A[i,a] + C[i,a] - B[i,a] - A2[i,a]

    Where:
      A[i, a_ii] = Σ_k S(ki,ii)^T @ K_tilde_chem[ki] @ Tt[ki]      (Eq 20)
      C[i, a_ii] = Σ_k S(ik,ii)^T @ Tt[ik] @ Fkc[ki]^T              (Eq 22)
      B[i, a_ii] = Σ_{k,l} S(kl,ii)^T @ (Tt[kl] @ K_kilc^T)[:, i_kl]  (Eq 21)
      A2[i, a_ii] = Σ_{k,l} T_n[ii][l, :] * (K[kl] · U_ki)        (Eq 21)

    Notation:
      - K_tilde_chem[ki][a, c*N+d] = (k a | c d) — LMO is k, all PNOs in PNO_ki
      - K_bar[kl][m, c] = (mk|lc) — m occupied (in domain), c PNO_kl
      - K_kilc[kl] = K_bar[kl] + T_n[kl] @ K_iajb[kl] — dressed
      - Tt[ij] = 2*T[ij] - T[ij].T (antisym in virtual)
      - T_n[ij][k, :] = t1_k projected to PNO_ij basis (over k in domain)

    Args:
        t1_pno: dict i -> (n_pno_ii,) T1 amplitudes
        t2_pno_all: dict canonical_key -> (n_pno, n_pno) T2 amplitudes
        pno_spaces: dict with PNO data
        fov_pno: dict i -> (n_pno_ii,) bare Fock ov
        F_lmo: (nocc, nocc) Fock in LMO basis
        eps_lmo: (nocc,) LMO orbital energies
        nocc: number of active occupied
        S_pno_cache: dict (key1, key2) -> overlap matrix
        cc_ints: dict from compute_cc_integrals (must be present)

    Returns:
        r1_pno: dict i -> (n_pno_ii,) T1 residual
    """
    if cc_ints is None:
        raise ValueError("cc_ints required for Psi4-style T1 residual")

    # Phase 1: use pre-built t1 cache (or build one locally for back-compat).
    if t1_cache is None:
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, build_t1_cache,
        )
        _pi = PairIndex(
            pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
        t1_cache = build_t1_cache(
            t1_pno, _pi, S_pno_cache, pno_spaces)

    # Pre-compute Tt for all canonical pairs (in their canonical PNO basis)
    Tt_canon = {}
    for key, t2 in t2_pno_all.items():
        if t2 is not None and t2.shape[0] > 0:
            Tt_canon[key] = 2.0 * t2 - t2.T

    # T_n[(key, k)] = t1_k projected to pair key's PNO basis.
    # Phase 1: fill directly from the cached (nocc, n_pno) matrix.
    T_n = {}
    for key in t2_pno_all:
        if pno_spaces[key]['C_pno'].shape[1] == 0:
            continue
        cached_matrix = t1_cache[key]
        for k in range(nocc):
            T_n[(key, k)] = cached_matrix[k]

    # Initialize R1 from Fai_[i].  At iter 0 with T1=0, Fai_bar = 0 and
    # all dressing terms vanish, so R1 starts from just A + B.
    # The bare fov enters the R1 through the Fai_bar T1 dressing at
    # later iterations.  However, we still need fov_pno for the energy
    # formula (Eq 102: E += fov . t1).  For the R1 itself, we initialize
    # to zero and let the dressing terms build the Fai contribution.
    r1_pno = {}
    for i in range(nocc):
        key_ii = (i, i)
        if key_ii not in pno_spaces:
            r1_pno[i] = np.zeros(0)
            continue
        n_ii = pno_spaces[key_ii]['C_pno'].shape[1]
        if n_ii == 0:
            r1_pno[i] = np.zeros(0)
            continue
        r1_pno[i] = np.zeros(n_ii)

    _dbg_r1 = getattr(_compute_t1_residual_psi4, '_debug_r1_dump', False)
    _r1_iter = getattr(_compute_t1_residual_psi4, '_iter', 0)

    # ===========================================================
    # T1-dressing terms in Psi4's Fai_[i] (ccsd.cc lines 1631-1712)
    # ===========================================================
    # Psi4's full Fai_[i] for the diagonal pair (ii) is built in two
    # stages.  We replicate the *bare* (non T2-dressed) parts here using
    # the cc_ints DF intermediates.  We use the SET form
    #     t1_new = R / (eps - e_pno)
    # which absorbs the *diagonal* F_vv and F_oo[i,i] contributions into
    # the denominator, so the terms below contribute only the
    # off-diagonal pieces.
    #
    # Stage 1 — Fai_bar T1 dressing (ccsd.cc lines 1631-1654):
    #   gamma_q   = sum_{m,c} Qma[q,m,c] * T_n_ii[m,c]
    #   lambda_q  = Qab[q] @ T_n_ii.T            (n_pno, nocc)
    #   Fai_bar[a] += 2 * sum_q gamma_q * Qia[q,a]                (J)
    #   Fai_bar[a] -= sum_{q,k} lambda_q[a,k] * Qik[q,k]          (K)
    #
    # Stage 2 — F_vv * t1 / F_oo * t1 (ccsd.cc lines 1700-1712):
    #   In the bare semicanonical PNO basis Fab_bar[ii] = diag(e_pno),
    #   so the off-diagonal F_vv contribution vanishes; the diagonal
    #   piece is in the denominator.  For F_oo, we add the bare
    #   off-diagonal contribution explicitly (k != i):
    #     Fai[a] -= sum_{k!=i} F_lmo[k,i] * (t1[k] projected to PNO_ii)
    # --- Pre-compute Fij_bar (T1-dressed occupied Fock, Psi4 lines 1596-1597) ---
    # Fij_bar[k,i] = F_lmo[k,i]
    #   + sum over ordered pairs (k,i):
    #     2 * T_n_ki . K_bar_chem_ki - T_n_ik . K_bar_ik
    # For all ordered pairs (i,j) in Psi4:
    #   Fij_bar(i,j) += 2*T_n[ij] . K_bar_chem[ij] - T_n[ij] . K_bar[ji]
    Fij_bar = F_lmo.copy()
    for key_ij in t2_pno_all:
        ci_ij = cc_ints.get(key_ij)
        if ci_ij is None:
            continue
        i0, j0 = key_ij  # canonical (i0 <= j0)
        n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_ij == 0:
            continue
        # Build T_n_ij[m, c] for canonical pair
        T_n_ij_mat = np.zeros((nocc, n_ij))
        for m in range(nocc):
            tn_m = T_n.get((key_ij, m))
            if tn_m is not None:
                T_n_ij_mat[m] = tn_m
        # Ordered pair (i0, j0): K_bar_chem = ci_ij['K_bar_chem'], K_bar_ji = ci_ij['K_bar_ji']
        # Fij_bar(i0, j0) += 2 * T_n . K_bar_chem - T_n . K_bar[ji]
        Fij_bar[i0, j0] += (2.0 * np.sum(T_n_ij_mat * ci_ij['K_bar_chem'])
                            - np.sum(T_n_ij_mat * ci_ij['K_bar_ji']))
        if i0 != j0:
            # Ordered pair (j0, i0): K_bar_chem_ji, K_bar_ij
            # Psi4 builds K_bar_chem for ordered pair (j,i) from the ji index
            # K_bar_chem[ji] = q_pair_ji . Qma_ji — but our cc_ints stores for canonical (i0,j0)
            # For ordered (j0,i0): K_bar_chem uses j0 as "i-side", i0 as "j-side"
            # Equivalent: swap i_Qa↔j_Qa roles; q_pair unchanged
            # K_bar_chem_ji[m,a] = sum_Q q_pair[Q] * Qma[Q,m,a] (same as canonical)
            # K_bar[ij] for ordered (j0,i0) = K_bar_ij from our storage
            Fij_bar[j0, i0] += (2.0 * np.sum(T_n_ij_mat * ci_ij['K_bar_chem'])
                                - np.sum(T_n_ij_mat * ci_ij['K_bar_ij']))

    for i in range(nocc):
        key_ii = (i, i)
        if key_ii not in pno_spaces:
            continue
        n_ii = pno_spaces[key_ii]['C_pno'].shape[1]
        if n_ii == 0:
            continue
        e_pno_ii = pno_spaces[key_ii]['e_pno']
        t1_i = t1_pno.get(i)
        has_t1_i = t1_i is not None and t1_i.size > 0 and np.max(np.abs(t1_i)) > 1e-15

        ci_ii = cc_ints.get(key_ii)
        if ci_ii is None:
            continue

        Qma_ii = ci_ii['Qma']      # (n_local, nocc, n_ii)
        Qab_ii = ci_ii['Qab']      # (n_local, n_ii, n_ii)
        Qia_ii = ci_ii['i_Qa']     # (n_local, n_ii)
        Qik_ii = ci_ii['i_Qk']     # (n_local, nocc)

        # T_n_ii[m, c] = t1[m] projected to PNO_ii
        T_n_ii = np.zeros((nocc, n_ii))
        any_t1 = False
        for m in range(nocc):
            tn_m = T_n.get((key_ii, m))
            if tn_m is not None and tn_m.size == n_ii:
                T_n_ii[m] = tn_m
                if np.max(np.abs(tn_m)) > 1e-15:
                    any_t1 = True

        # ---- Stage 1: Fai_bar T1 dressing (ccsd.cc lines 1631-1654) ----
        if any_t1:
            # gamma[q] = Σ_{m,c} Qma[q,m,c] * T_n[m,c]
            gamma = Qma_ii.reshape(Qma_ii.shape[0], -1) @ T_n_ii.ravel()
            # J: r1 += 2 * Qia.T @ gamma
            r1_pno[i] += 2.0 * (Qia_ii.T @ gamma)
            # K: r1 -= Σ_q Σ_k Qab[q,a,k] * (Σ_k' Qik[q,k'] * T_n[k',c]) ... wait
            # Actually: y[q,c] = Σ_k Qik[q,k] * T_n[k,c]
            y = Qik_ii @ T_n_ii                # (n_local, n_ii)
            # r1[a] -= Σ_{q,c} Qab[q,a,c] * y[q,c] = Σ_q (Qab[q] @ y[q])[a]
            # Flatten: (Qab * y[:,None,:]).sum((0,2)) — efficient via matmul
            # = Σ_q,c Qab[q, :, c] * y[q, c] = Σ_q (Qab[q] @ y[q])[a]
            r1_pno[i] -= np.einsum('qac,qc->a', Qab_ii, y, optimize=True)

        # ---- Stage 2 & 3: Fab_bar @ t1 and -T_n.T @ Fia_bar @ t1 ----
        if has_t1_i and any_t1:
            # W[q, a, k] = Σ_b Qab[q,a,b] * T_n[k,b] = (Qab @ T_n.T)[q,a,k]
            W_qak = Qab_ii @ T_n_ii.T          # (n_local, n_ii, nocc)
            # Fab_bar = diag(e_pno) + 2*Σ_q gamma[q]*Qab[q] - Σ_{q,k} W[q,a,k]*Qma[q,k,c]
            Fab_bar_ii = np.diag(e_pno_ii)
            Fab_bar_ii += 2.0 * np.tensordot(gamma, Qab_ii, axes=(0, 0))
            Fab_bar_ii -= np.tensordot(W_qak, Qma_ii, axes=((0, 2), (0, 1)))
            r1_pno[i] += Fab_bar_ii @ t1_i

            # Fia_bar[m, a] = 2*Σ_q gamma[q]*Qma[q,m,a] - Σ_{q,k} Z[q,m,k]*Qma[q,k,b]
            # Z[q, m, k] = Σ_a Qma[q,m,a] * T_n[k,a] = Qma @ T_n.T
            Z_qmk = Qma_ii @ T_n_ii.T          # (n_local, nocc, nocc)
            Fia_bar_ii = 2.0 * np.tensordot(gamma, Qma_ii, axes=(0, 0))
            Fia_bar_ii -= np.tensordot(Z_qmk, Qma_ii, axes=((0, 1), (0, 1)))
            r1_pno[i] -= T_n_ii.T @ (Fia_bar_ii @ t1_i)
        elif has_t1_i:
            # No T1 dressing → bare Fab_bar = diag(e_pno)
            r1_pno[i] += e_pno_ii * t1_i

        # ---- Stage 4: -sum_k T_n[ii][k,a] * Fij_bar[k,i] ----
        # (ccsd.cc line 1709)
        # INCREMENT form: include ALL k (including k=i).  T_n_ii was already
        # built above (nocc, n_ii) with row k = t1_k in PNO_ii basis;
        # rows outside the cache are zero and contribute nothing, so this
        # is one matvec instead of nocc _project_t1_to_pair calls.
        r1_pno[i] -= Fij_bar[:, i] @ T_n_ii

    # Helper: get integral (k a_ki | c d) from cc_ints
    def _get_ki_data(k, i_lmo):
        key_ki = (min(k, i_lmo), max(k, i_lmo))
        if key_ki not in pno_spaces:
            return None
        ci = cc_ints.get(key_ki)
        if ci is None:
            return None
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        if n_ki == 0:
            return None
        k_Qa = ci['i_Qa'] if key_ki[0] == k else ci['j_Qa']
        return {'key': key_ki, 'n_ki': n_ki, 'k_Qa': k_Qa,
                'Qab': ci['Qab'], 'i_Qk': ci.get('i_Qk'), 'j_Qk': ci.get('j_Qk')}

    # ===========================================================
    # A and C terms: loop over ordered pairs (i, k) for each i
    # ===========================================================
    # Pre-compute LT1[(i, m)] = (key_im, L_im @ t1_m_in_im) once per (i, m).
    # Inside the k-loop below the original code recomputed this for each
    # of the nocc values of k; since LT1 depends only on (i, m) we can
    # hoist it. The k-dependent part (S_ki_im projection) stays inside.
    _LT1_cache = {}
    for m in range(nocc):
        t1_m = t1_pno.get(m)
        if t1_m is None or t1_m.size == 0:
            continue
        if np.max(np.abs(t1_m)) < 1e-15:
            continue
        for i_out in range(nocc):
            key_im = (min(i_out, m), max(i_out, m))
            ci_im = cc_ints.get(key_im)
            if ci_im is None:
                continue
            if pno_spaces[key_im]['C_pno'].shape[1] == 0:
                continue
            K_im = ci_im['K_iajb']
            L_im = 2.0 * K_im - K_im.T
            t1_m_in_im = t1_cache[key_im][m]
            _LT1_cache[(i_out, m)] = (key_im, L_im @ t1_m_in_im)

    for i in range(nocc):
        key_ii = (i, i)
        if key_ii not in pno_spaces or pno_spaces[key_ii]['C_pno'].shape[1] == 0:
            continue
        n_ii = pno_spaces[key_ii]['C_pno'].shape[1]

        for k in range(nocc):
            ki_data = _get_ki_data(k, i)
            if ki_data is None:
                continue
            key_ki = ki_data['key']
            n_ki = ki_data['n_ki']
            k_Qa = ki_data['k_Qa']
            Qab_ki = ki_data['Qab']

            # Get T2 for ordered pair (k, i): Tt_ki[a, c]
            if key_ki not in t2_pno_all:
                continue  # weak pair — no T2 amplitudes
            t2_canon = t2_pno_all[key_ki]
            if k <= i:
                # canonical key = (k, i), storage in (k,i) order
                t2_ki = t2_canon  # T_iajb[(k,i)][a, c] = t2[k,i]^{a,c}
            else:
                # canonical key = (i, k), storage in (i,k) order — transpose
                t2_ki = t2_canon.T
            Tt_ki = 2.0 * t2_ki - t2_ki.T  # antisymmetric in (a, c)

            # ----- A term -----
            # temp_A[d] = Σ_{a,c} (k a | c d) * Tt_ki[a, c]
            # = Σ_{Q, a, c} k_Qa[Q,a] * Qab[Q,c,d] * Tt_ki[a,c]
            # Z[Q, c] = Σ_a k_Qa[Q,a] * Tt_ki[a,c]
            Z = np.einsum('Qa,ac->Qc', k_Qa, Tt_ki)
            temp_A = np.einsum('Qc,Qcd->d', Z, Qab_ki)
            # Project to PNO_ii via S(ki, ii)^T
            if key_ki == key_ii:
                A_contrib = temp_A
            else:
                S_ii_ki = S_pno_cache.get((key_ii, key_ki))
                if S_ii_ki is not None:
                    A_contrib = S_ii_ki @ temp_A
                else:
                    A_contrib = None
            if A_contrib is not None:
                r1_pno[i] += A_contrib

            # ----- C term -----
            # C_i^a = Σ_k S(ik,ii)^T @ Tt[ik] @ Fkc[ki]^T
            # Fkc[ki] = fov[k] projected to PNO_ki (with T1 dressing,
            # but at iter 0 t1=0 so Fkc = bare fov_k in PNO_ki)
            # Get T2 for ordered pair (i, k): Tt_ik = T2[i,k] - T2[i,k].T (antisym in c,d)
            if i <= k:
                t2_ik = t2_canon  # canonical = (i,k) = (min,max)
            else:
                t2_ik = t2_canon.T
            Tt_ik = 2.0 * t2_ik - t2_ik.T

            # Fkc[ki] = fov_k in PNO_ki basis
            # Use T_n[(key_ki, k)] which is t1_k projected — but for Fkc we need
            # fov_k projected, not t1_k. Project fov_k to PNO_ki basis.
            fov_k_in_ki = _project_t1_to_pair(
                fov_pno, k, key_ki, S_pno_cache, pno_spaces)

            # T1 dressing of Fkc (Psi4 ccsd.cc lines 1685-1692):
            # Fkc[ij] += sum_m S(ij,im) @ L[im] @ S(im,mm) @ T1[m]
            # where L[im] = 2*K[im] - K[im].T, and the sum is over m in
            # the pair (i,k)'s LMO domain (for us: all m).
            # Here ij=ki (ordered pair for C-term), so:
            # Fkc_dress_ki = sum_m S(ki,im) @ L[im] @ S(im,mm) @ T1[m]
            # This dressing is evaluated in PNO_ki basis.
            fkc_dress = np.zeros_like(fov_k_in_ki)
            for m in range(nocc):
                entry = _LT1_cache.get((i, m))
                if entry is None:
                    continue
                key_im, LT1 = entry
                # S(ki, im) @ LT1 → project from PNO_im to PNO_ki
                if key_ki == key_im:
                    fkc_dress += LT1
                else:
                    S_ki_im = S_pno_cache.get((key_ki, key_im))
                    if S_ki_im is not None:
                        fkc_dress += S_ki_im @ LT1
            # Psi4's Fkc_ is PURELY T1-dressed (no bare fov).
            # The bare fov contribution enters through Fai_bar, not Fkc.
            fov_k_in_ki = fkc_dress

            # contrib = Tt_ik @ fov_k_in_ki  (= Tt_ik @ Fkc)
            # Note: PNO_ik == PNO_ki (same canonical pair)
            contrib_C_local = Tt_ik @ fov_k_in_ki
            if key_ki == key_ii:
                C_contrib = contrib_C_local
            else:
                S_ii_ik = S_pno_cache.get((key_ii, key_ki))
                if S_ii_ik is not None:
                    C_contrib = S_ii_ik @ contrib_C_local
                else:
                    C_contrib = None
            if C_contrib is not None:
                r1_pno[i] += C_contrib

    # ===========================================================
    # B and A2 terms: loop over ordered pairs (k, l)
    # ===========================================================
    # For each ordered pair (k, l), compute B_ia and A2 contributions
    # for each orbital i in pair (k,l)'s LMO domain.
    # For us: nocc is small, treat all i as in every pair domain.
    for key_kl in t2_pno_all:
        if pno_spaces[key_kl]['C_pno'].shape[1] == 0:
            continue
        # Process both ordered (k, l) and (l, k) when k != l
        k0, l0 = key_kl  # canonical
        orderings = [(k0, l0)] if k0 == l0 else [(k0, l0), (l0, k0)]

        for (k, l) in orderings:
            n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
            ci_kl = cc_ints.get(key_kl)
            if ci_kl is None:
                continue

            # T2 in ordering (k, l)
            t2_canon_kl = t2_pno_all[key_kl]
            if k <= l:
                t2_kl = t2_canon_kl
            else:
                t2_kl = t2_canon_kl.T
            Tt_kl = 2.0 * t2_kl - t2_kl.T

            # K_iajb[kl] in PNO_kl: bare exchange (kc|ld)
            K_iajb_kl = ci_kl['K_iajb']

            # K_bar[kl][m, c] = (mk|lc) — Psi4 stores as K_bar_ij[m, c] = (mi|jc)
            # For our pair with k as i-side and l as j-side:
            # If key_kl = (k, l) (k <= l), use K_bar_ij (q_io.T @ q_jv) where i_lmo=k, j_lmo=l
            # If key_kl = (l, k) (k > l), use K_bar_ji (q_jo.T @ q_iv) where j_lmo=k, i_lmo=l
            if key_kl[0] == k:
                # k is at key[0]=i_lmo, l is at key[1]=j_lmo
                K_bar_kl = ci_kl['K_bar_ij']  # (nocc, n_kl) = (m k | l c)
            else:
                # k is at key[1]=j_lmo, l is at key[0]=i_lmo
                K_bar_kl = ci_kl['K_bar_ji']  # (nocc, n_kl) = (m k | l c)

            # K_kilc = K_bar[kl] + T_n[kl] @ K_iajb[kl]
            # T_n[kl][m, :] = t1_m projected to PNO_kl
            T_n_kl = np.zeros((nocc, n_kl))
            for m in range(nocc):
                T_n_kl[m] = T_n.get((key_kl, m), np.zeros(n_kl))
            K_kilc = K_bar_kl + T_n_kl @ K_iajb_kl  # (nocc, n_kl)

            # B_ia[a_kl, m] = Σ_c Tt_kl[a, c] * K_kilc[m, c] = Tt_kl @ K_kilc.T
            B_ia = Tt_kl @ K_kilc.T  # (n_kl, nocc)

            # Only orbitals i in pair (k,l)'s LMO domain contribute (Psi4 line 2082-2083)
            if pair_lmo_idx is not None and key_kl in pair_lmo_idx:
                _i_list = pair_lmo_idx[key_kl]
            else:
                _i_list = range(nocc)
            for i in _i_list:
                key_ii = (i, i)
                if key_ii not in pno_spaces:
                    continue
                if pno_spaces[key_ii]['C_pno'].shape[1] == 0:
                    continue
                # B contribution
                if key_kl == key_ii:
                    B_contrib = B_ia[:, i]
                else:
                    S_ii_kl = S_pno_cache.get((key_ii, key_kl))
                    if S_ii_kl is not None:
                        B_contrib = S_ii_kl @ B_ia[:, i]
                    else:
                        B_contrib = None
                if B_contrib is not None:
                    r1_pno[i] -= B_contrib

                # A2 contribution (Psi4 ccsd.cc lines 2146-2152)
                # R1[i,a] -= T_n[ii][l, a] * dot(K_iajb[kl], U_ki)
                # where U_ki = S(kl, ki) @ Tt[ki] @ S(ki, kl)
                key_ki = (min(k, i), max(k, i))
                if key_ki in t2_pno_all and key_ki in pno_spaces:
                    n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
                    if n_ki > 0:
                        t2_ki_canon = t2_pno_all[key_ki]
                        t2_ki = t2_ki_canon if k <= i else t2_ki_canon.T
                        Tt_ki = 2.0 * t2_ki - t2_ki.T
                        # S(kl, ki) @ Tt[ki] @ S(ki, kl)
                        S_kl_ki = S_pno_cache.get((key_kl, key_ki))
                        S_ki_kl = S_pno_cache.get((key_ki, key_kl))
                        if key_kl == key_ki:
                            U_ki = Tt_ki
                        elif S_kl_ki is not None and S_ki_kl is not None:
                            U_ki = S_kl_ki @ Tt_ki @ S_ki_kl
                        else:
                            U_ki = None
                        if U_ki is not None:
                            scalar = np.sum(K_iajb_kl * U_ki)
                            # T_n[ii][l, :] in PNO_ii basis
                            T_n_l_ii = T_n.get((key_ii, l), np.zeros(
                                pno_spaces[key_ii]['C_pno'].shape[1]))
                            if T_n_l_ii.size > 0:
                                r1_pno[i] -= scalar * T_n_l_ii

    if _dbg_r1:
        for i in range(nocc):
            if r1_pno[i].size > 0:
                rms = float(np.sqrt(np.mean(r1_pno[i]*r1_pno[i])))
                mx = float(np.max(np.abs(r1_pno[i])))
                sm = float(np.sum(r1_pno[i]))
                print(f'  R1_OURS iter {_r1_iter} lmo({i}): rms={rms:.12e} '
                      f'max={mx:.12e} sum={sm:.12e} np={r1_pno[i].size}',
                      flush=True)

    return r1_pno


def _run_dlpno_lccsd(mf, C_lmo, pno_spaces, strong_pairs,
                     fock_ao, eps_lmo, s1e, conv_tol, max_cycle,
                     t2_cas=None, occ_cas_idx=None, vir_cas_idx=None,
                     mo_coeff_cas=None, diis_space=5,
                     damping=0.5, diis_start_cycle=0,
                     C_pao=None, use_t1_transform=True,
                     ncores=1, negligible_pairs=None, _pool=None):
    """DLPNO-CCSD with pair-local residual and per-pair PNO virtual spaces.

    Each pair (i,j) updates ONLY its own T2_ij in its own PNO basis.
    The T2 residual numerator is computed term-by-term:
      1. K_ij (exchange integral driving term)
      2. Inter-pair coupling (B/Woooo + G/Fock + ring) via _compute_pair_coupling
      3. Intra-pair ladder (vvvv contraction via DF)
      4. Self-pair Woooo and Fock contributions

    Then: t2_new_ij = N_ij / D_ij  where D_ij = eps_i + eps_j - e_a - e_b.

    This avoids the redundancy problems of projecting all pairs' T2 into a
    single PNO space and calling PySCF's update_amps.  Each pair's T2 lives
    exclusively in its own PNO basis — no cross-pair T2 projection is needed
    except in the coupling terms (which use PNO overlap matrices).

    For TCCSD: after each Jacobi step, the CAS block of each CAS pair's T2
    is restored to the DMRG value (Lang et al. eq. 14–15).  The CAS virtuals
    occupy the first nvir_cas_local columns of C_pno_ij (identity block in
    the extended PNO construction, eq. 10 of Lang et al.).

    Args:
        Same as _run_global_lccsd.

    Returns:
        e_total (float): Total DLPNO-CCSD correlation energy.
        t2_pno_all (dict): (i,j) → T2 in PNO[ij] basis (for (T) triples).
    """
    mol = mf.mol
    with_df = getattr(mf, 'with_df', None)
    nocc = C_lmo.shape[1]

    # Jiang uses bare integrals with explicit T1 dressing
    use_t1_transform = False

    # ------------------------------------------------------------------
    # Pre-compute quantities needed by the pair-local residual
    # ------------------------------------------------------------------
    F_lmo = reduce(np.dot, (C_lmo.T, fock_ao, C_lmo))

    # J_oo[i,j,k,l] = (ij|kl) in chemist notation — for Woooo coupling
    # Deferred: built from ooL_3idx after ovL cache is ready (avoids
    # redundant ao2mo call when DF is available).
    J_oo = None

    # PAO coefficients and (LMO,PAO|L) 3-index tensor for ring coupling.
    # C_pao must be the SAME PAOs used by pno.make_pnos, passed from the driver.
    from pyscf.cc.dlpno_tccsd.pno import _build_ovL
    if C_pao is None:
        # Fallback: build PAOs from scratch (may not match pno.py's PAOs)
        nao = C_lmo.shape[0]
        S_ao = s1e
        P_occ = C_lmo @ np.linalg.solve(C_lmo.T @ S_ao @ C_lmo, C_lmo.T @ S_ao)
        C_pao_raw = np.eye(nao) - P_occ
        S_pao = C_pao_raw.T @ S_ao @ C_pao_raw
        eigvals, eigvecs = np.linalg.eigh(S_pao)
        keep = eigvals > 1e-8
        C_pao = C_pao_raw @ eigvecs[:, keep] / np.sqrt(eigvals[keep])

    ovL_lmo_pao = None
    if with_df is not None:
        ovL_lmo_pao = _build_ovL(with_df, C_lmo, C_pao)

    # ------------------------------------------------------------------
    # T1 amplitudes: local in diagonal PNO basis (Jiang et al. JCP 2024).
    # t1_pno[i] = np.ndarray(n_pno_ii,) in PNO basis of pair (i,i).
    # fov_pno[i] = F_{a_ii, i} in the same basis (bare Fock ov block).
    # ------------------------------------------------------------------
    # Verify diagonal pairs exist for all occupied orbitals
    for i in range(nocc):
        if (i, i) not in pno_spaces:
            raise RuntimeError(
                f'Diagonal pair ({i},{i}) missing from pno_spaces; '
                'required for local T1 amplitudes')

    t1_pno = {}
    fov_pno = {}   # Fock ov block in diagonal PNO basis (reused in residual)
    for i in range(nocc):
        key_ii = (i, i)
        data_ii = pno_spaces[key_ii]
        C_pno_ii = data_ii['C_pno']
        n_pno_ii = C_pno_ii.shape[1]
        if n_pno_ii == 0:
            t1_pno[i] = np.zeros(0)
            fov_pno[i] = np.zeros(0)
            continue
        # F_{a_ii, i} = C_pno_ii^T @ fock_ao @ C_lmo[:,i]
        fov_i = C_pno_ii.T @ (fock_ao @ C_lmo[:, i])  # (n_pno_ii,)
        fov_pno[i] = fov_i
        # T1 initialisation: Psi4 starts T1 at zero (ccsd.cc line 1960).
        # The MP2 init t1 = -fov/D was causing iter 0 to have nonzero t1-
        # dependent terms, poisoning iter 1+ residuals vs Psi4.
        t1_pno[i] = np.zeros(data_ii['e_pno'].shape[0])

    # ------------------------------------------------------------------
    # CAS freeze bookkeeping (per-pair PNO convention)
    # In the extended PNO basis, CAS virtual MOs are the FIRST nvir_cas_local
    # columns of C_pno_ij (identity block, Lang et al. eq. 10).
    # ------------------------------------------------------------------
    cas_blocks = {}   # key → (slice, t2_cas_ij) for pairs with CAS virtuals
    if t2_cas is not None and occ_cas_idx is not None:
        for pair in strong_pairs:
            i, j = pair
            key = (min(i, j), max(i, j))
            if key not in pno_spaces:
                continue
            nvir_cas_local = pno_spaces[key].get('nvir_cas_local', 0)
            if nvir_cas_local == 0:
                continue
            i_cas = int(np.where(occ_cas_idx == i)[0][0])
            j_cas = int(np.where(occ_cas_idx == j)[0][0])
            t2c = t2_cas[i_cas, j_cas] if i <= j else t2_cas[j_cas, i_cas].T
            cas_blocks[key] = (slice(0, nvir_cas_local), t2c)

    # ------------------------------------------------------------------
    # Initialise T2 from MP2 (or stored T2_pno); inject CAS values
    # ------------------------------------------------------------------
    t2_pno_all = {}
    for pair in strong_pairs:
        i, j = pair
        key = (min(i, j), max(i, j))
        if key not in pno_spaces:
            continue
        data = pno_spaces[key]
        n_pno = data['C_pno'].shape[1]
        if n_pno == 0:
            t2_pno_all[key] = np.zeros((0, 0))
            continue
        T2_mp2 = data.get('T2_pno')
        K_pno  = data.get('K_pno')
        if T2_mp2 is not None and T2_mp2.shape == (n_pno, n_pno):
            t2_pno_all[key] = T2_mp2.copy()
        elif K_pno is not None:
            D = eps_lmo[i] + eps_lmo[j] - data['e_pno'][:, None] - data['e_pno'][None, :]
            t2_pno_all[key] = K_pno / np.where(np.abs(D) > 1e-12, D, 1e-12)
        else:
            # K_pno not computed (e.g. pure CAS pair): start from zero;
            # the CAS block will be overwritten by DMRG amplitudes below.
            t2_pno_all[key] = np.zeros((n_pno, n_pno))
        if key in cas_blocks:
            cas_sl, t2c = cas_blocks[key]
            t2_pno_all[key][cas_sl, cas_sl] = t2c

    # Add weak-pair MP2 T2 amplitudes so dressing builders (G_tilde, C_tilde,
    # D_tilde, B_tilde) see them, matching Psi4 which keeps T_iajb_ for ALL
    # pairs (strong + weak).  These weak amplitudes are NOT iterated, but
    # they DO contribute to the dressing of strong-pair residuals.
    keys_sorted = sorted(t2_pno_all.keys())  # strong-pair keys (unchanged)
    _strong_keys_set = set(keys_sorted)
    for key_w, data_w in pno_spaces.items():
        if key_w in _strong_keys_set:
            continue
        if data_w['C_pno'].shape[1] == 0:
            continue
        T2_w = data_w.get('T2_pno')
        if T2_w is not None:
            t2_pno_all[key_w] = T2_w.copy()

    n_active = len(keys_sorted)
    print(f'  DLPNO-CCSD: {n_active} pairs, CAS freeze = {len(cas_blocks)} pairs',
          flush=True)

    # ------------------------------------------------------------------
    # Pre-compute K_pno cache (exchange integrals for energy monitoring)
    # and initialise CAS pairs that lack K_pno.
    # ------------------------------------------------------------------
    K_pno_cache = {}
    for key in keys_sorted:
        i, j = key
        data = pno_spaces[key]
        C_pno_ij = data['C_pno']
        n_pno = C_pno_ij.shape[1]
        if n_pno == 0:
            continue

        K_pno = data.get('K_pno')
        if K_pno is not None:
            K_pno_cache[key] = K_pno
        elif with_df is not None:
            # Build exchange via DF
            K_pno_cache[key] = with_df.ao2mo(
                [C_lmo[:, [i]], C_pno_ij, C_lmo[:, [j]], C_pno_ij],
                compact=False).reshape(n_pno, n_pno)
        if pno_spaces[key].get('K_pno') is None and key in cas_blocks:
            if key in K_pno_cache:
                D_full = (eps_lmo[i] + eps_lmo[j]
                          - data['e_pno'][:, None] - data['e_pno'][None, :])
                D_full_safe = np.where(np.abs(D_full) > 1e-12, D_full, 1e-12)
                t2_pno_all[key] = K_pno_cache[key] / D_full_safe
                cas_sl, t2c = cas_blocks[key]
                t2_pno_all[key][cas_sl, cas_sl] = t2c

    # ------------------------------------------------------------------
    # Pre-compute ovL_pno cache: 3-index DF tensors in PNO basis.
    # With local DF fitting (Jiang Eq 63-64): each pair only uses aux
    # functions on atoms with significant Mulliken population (T_CutMKN).
    # ovL_pno_cache[(pair_key, lmo_idx)] = (n_pno, naux_local) tensor.
    # pair_aux_idx[pair_key] = integer array of global aux indices in domain.
    # ------------------------------------------------------------------
    T_CutMKN = 1e-3  # ORCA TightPNO default
    ovL_pno_cache = {}  # legacy/deprecated; kept as empty alias for residual
    pair_aux_idx = {}  # pair_key -> np.array of aux indices in local domain
    if with_df is not None:
        # --- Compute per-LMO local auxiliary domains (Eq 63-64) ---
        _ao_labels = mf.mol.ao_labels(fmt=False)
        _atom_ids = np.array([lbl[0] for lbl in _ao_labels])
        _natm = mf.mol.natm
        # Use RI auxiliary basis for local DF domain construction if available
        _ri_auxbasis = getattr(mf, '_ri_auxbasis', None)
        if _ri_auxbasis is not None:
            from pyscf import df
            _auxmol = df.addons.make_auxmol(mf.mol, _ri_auxbasis)
        else:
            _auxmol = with_df.auxmol
        _aux_atom_ids = np.array([lbl[0] for lbl in _auxmol.ao_labels(fmt=False)])
        _naux_full = with_df.get_naoaux()

        # Mulliken aux domain matching Psi4 dlpnobase.cc:667-704 EXACTLY:
        # P_i[u,v] = C[u,i] * S[u,v] * C[v,i]
        # mkn_pop[atom_u] += p_uv * p_uu / (p_uu + p_vv)
        # mkn_pop[atom_v] += p_uv * p_vv / (p_uu + p_vv)
        # Then: keep atom A if abs(mkn_pop[A]) > T_CUT_MKN  (absolute, NOT
        # normalized by total — this was a bug in our previous version that
        # used normalized q_iA = pop_atom / total).
        _lmo_aux_mask = []  # per-LMO boolean mask over aux functions
        _lmo_atom_set = []  # per-LMO set of atoms with Mulliken pop
        for i in range(nocc):
            c_i = C_lmo[:, i]
            P_i = s1e * c_i[:, None] * c_i[None, :]
            p_diag = np.diag(P_i)
            sum_diag = p_diag[:, None] + p_diag[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                w_u = np.where(sum_diag > 1e-15, p_diag[:, None] / sum_diag, 0.0)
                w_v = np.where(sum_diag > 1e-15, p_diag[None, :] / sum_diag, 0.0)
            contrib_u = P_i * w_u
            contrib_v = P_i * w_v
            mkn_pop = np.zeros(_natm)
            for a in range(_natm):
                mask_a = (_atom_ids == a)
                mkn_pop[a] = np.sum(contrib_u[mask_a, :]) + np.sum(contrib_v[:, mask_a])
            atoms_in = np.where(np.abs(mkn_pop) > T_CutMKN)[0]
            _lmo_aux_mask.append(np.isin(_aux_atom_ids, atoms_in))
            _lmo_atom_set.append(set(atoms_in.tolist()))

        # Pair aux domain = union of LMO i and LMO j aux domains
        # Include ALL pairs in pno_spaces (strong + weak + diagonal)
        _all_keys_init = list(keys_sorted)
        for key_w in pno_spaces:
            if key_w not in _all_keys_init and pno_spaces[key_w]['C_pno'].shape[1] > 0:
                _all_keys_init.append(key_w)

        # Pair LMO domain: m is in pair (i,j)'s domain iff pairs (i,m) AND
        # (j,m) are BOTH non-negligible (strong or weak).  This matches
        # Psi4's lmopair_to_lmos_ (dlpnobase.cc line 798-818) where
        # i_j_to_ij_[i][m] == -1 for discarded (negligible) pairs.
        # Using only the strong+weak set (t2_pno_all keys) — NOT the
        # full pno_spaces — is what makes this domain actually local:
        # for water chains, including negligible pairs leaves every pair
        # with domain = full nocc.
        pair_lmo_idx = {}
        _negligible_set = set(
            (min(p), max(p)) for p in (negligible_pairs or []))
        _non_negligible_set = {k for k in t2_pno_all.keys()
                                if k not in _negligible_set}
        for key in _all_keys_init:
            i, j = key
            domain_lmos = []
            for m in range(nocc):
                key_im = (min(i, m), max(i, m))
                key_jm = (min(j, m), max(j, m))
                if (key_im in _non_negligible_set
                        and key_jm in _non_negligible_set):
                    domain_lmos.append(m)
            pair_lmo_idx[key] = np.array(domain_lmos)
        _plens = np.array([len(v) for v in pair_lmo_idx.values()])
        print(f"  Pair LMO domains: mean={_plens.mean():.1f}  "
              f"max={_plens.max()} (of nocc={nocc})",
              flush=True)

        for key in _all_keys_init:
            i, j = key
            _pmask = _lmo_aux_mask[i] | _lmo_aux_mask[j]
            pair_aux_idx[key] = np.where(_pmask)[0]

        # ovL_pno_cache is no longer built up-front (Psi4-equivalent path).
        # It will be populated below from cc_ints (local-aux per pair) after
        # cc_ints is built. The full-naux _build_ovL_batched call (which used
        # to take ~30s for str 010 and 162 MB of memory) is removed.
        del _lmo_aux_mask, _ao_labels, _atom_ids, _aux_atom_ids

    # ------------------------------------------------------------------
    # Per-pair local DF: precompute ALL locally-fitted intermediates ONCE.
    # Matches Psi4 compute_cc_integrals(). Stored for the entire CCSD run.
    # ------------------------------------------------------------------
    from pyscf.cc.dlpno_tccsd.local_df import compute_cc_integrals_sparse
    import time as _time_cc
    _t_cc = _time_cc.perf_counter()
    # Use RI auxiliary basis for local DF if available, otherwise JK aux
    _ri_auxbasis = getattr(mf, '_ri_auxbasis', None)
    if _ri_auxbasis is not None:
        from pyscf import df
        _auxmol = df.addons.make_auxmol(mf.mol, _ri_auxbasis)
    else:
        _auxmol = with_df.auxmol
    _j2c = _auxmol.intor('int2c2e')
    # Include ALL pairs with PNOs (strong + weak), matching Psi4 which
    # builds cc_integrals for all unscreened pairs (0 screened for TightPNO).
    _all_keys_cc = list(keys_sorted)
    for key_w in pno_spaces:
        if key_w not in _all_keys_cc:
            if pno_spaces[key_w]['C_pno'].shape[1] > 0 and key_w in pair_aux_idx:
                _all_keys_cc.append(key_w)
    # Psi4-style sparse per-aux-Q build with X_pno + pair_paos. Defaults
    # to Psi4's TightPNO thresholds (1e-3); override via mf attributes.
    _t_mkn = getattr(mf, '_sparse_cc_T_CUT_MKN', 1e-3)
    _t_clmo = getattr(mf, '_sparse_cc_T_CUT_CLMO', 1e-3)
    _pao_domains = []
    for _i in range(nocc):
        _ki = (_i, _i)
        if _ki in pno_spaces and pno_spaces[_ki].get('pair_paos') is not None:
            _pao_domains.append(np.asarray(pno_spaces[_ki]['pair_paos']))
        else:
            _pao_domains.append(np.zeros(0, dtype=int))
    _cc_ints = compute_cc_integrals_sparse(
        mf.mol, _auxmol, C_lmo, C_pao, pno_spaces, pair_aux_idx,
        _j2c, _all_keys_cc, nocc, s1e=s1e,
        pao_domains=_pao_domains, strong_pair_keys=_all_keys_cc,
        T_CUT_MKN=_t_mkn, T_CUT_CLMO=_t_clmo,
        pair_lmo_idx=pair_lmo_idx,
        _pool=_pool)
    print(f'  Local DF integrals: {len(_cc_ints)} pairs, '
          f'{_time_cc.perf_counter() - _t_cc:.1f}s', flush=True)
    # Rebuild K_pno_cache from locally-fitted K_iajb
    for key in list(K_pno_cache.keys()):
        ci = _cc_ints.get(key)
        if ci is not None:
            K_pno_cache[key] = ci['K_iajb']


    # Pre-compute PNO overlap matrices S_pno_cache[(key_ij, key_kl)].
    #
    # Restrict upfront build to O(N^2) entries where key_kl has BOTH
    # LMOs in pair (i,j)'s local domain (pair_lmo_idx[ij]), matching
    # Psi4's S_PNO which is indexed by lmopair_to_lmos_[ij] pairs.
    # Any residual / compute_C_tilde / build_D_tilde site that asks for
    # an unrestricted (key_a, key_b) overlap will go through the
    # ``compute_S_pno`` helper with IDENTICAL numerics — no drift.
    # This drops setup from O(N^4) to O(N^2) memory + build time.
    from pyscf.cc.dlpno_tccsd.local_df import compute_S_pno as _compute_S_pno
    _all_pair_keys = [k for k in pno_spaces
                      if pno_spaces[k]['C_pno'].shape[1] > 0]
    _all_pair_set = set(_all_pair_keys)
    S_pao_full = C_pao.T @ s1e @ C_pao
    S_pno_cache = {}
    for key_ij in _all_pair_keys:
        if pair_lmo_idx is not None and key_ij in pair_lmo_idx:
            _dom = [int(x) for x in pair_lmo_idx[key_ij]]
            _partner_keys = [
                (k, l) for k in _dom for l in _dom
                if k <= l and (k, l) in _all_pair_set
            ]
        else:
            _partner_keys = _all_pair_keys
        for key_kl in _partner_keys:
            S_pno_cache[(key_ij, key_kl)] = _compute_S_pno(
                key_ij, key_kl, pno_spaces, S_pao_full, s1e)

    # ooL_3idx (full naux occ-occ), K_coul_cache (full naux exchange) and
    # J_oo (nocc^4) are NOT built. cc_ints['i_Qk', 'j_Qk', 'J_ij_kj',
    # 'K_ij_kj'] supply the equivalent local-aux per-pair quantities.
    ooL_3idx = None
    K_coul_cache = {}
    ovL_pno_bare = ovL_pno_cache  # alias; kept as empty {} for legacy refs
    ooL_bare = None

    # ------------------------------------------------------------------
    # Adiabatic turn-on of CAS amplitudes + Jacobi-DIIS loop
    # ------------------------------------------------------------------
    # The CAS T2 can be very large (|t2| ~ 1 for strongly correlated
    # systems).  Injecting it at full strength from cycle 0 creates a
    # perturbation that the Jacobi iteration cannot absorb, leading to
    # divergence.  Instead, we gradually scale the CAS block from the
    # MP2/CCSD values to the DMRG values over n_bootstrap steps.
    # Each step reconverges the external T2 before increasing the scale.
    n_bootstrap = 5 if cas_blocks else 1
    bootstrap_tol = max(conv_tol * 100, 1e-3)  # loose tol for intermediate steps

    for boot_step in range(n_bootstrap):
        scale = (boot_step + 1) / n_bootstrap if cas_blocks else 1.0

        # Update CAS blocks with scaled DMRG amplitudes
        # t2_cas_scaled = t2_mp2 + scale * (t2_dmrg - t2_mp2)
        for key in list(cas_blocks.keys()):
            cb = cas_blocks[key]
            cas_sl = cb[0]
            t2c_dmrg = cb[1]
            if key not in t2_pno_all:
                continue
            if boot_step == 0 and len(cb) == 2:
                # Save MP2 CAS block as interpolation reference
                t2c_mp2 = t2_pno_all[key][cas_sl, cas_sl].copy()
                cas_blocks[key] = (cas_sl, t2c_dmrg, t2c_mp2)
            if len(cas_blocks[key]) == 3:
                t2c_mp2 = cas_blocks[key][2]
                t2_pno_all[key][cas_sl, cas_sl] = (
                    t2c_mp2 + scale * (t2c_dmrg - t2c_mp2))
            else:
                t2_pno_all[key][cas_sl, cas_sl] = scale * t2c_dmrg

        if n_bootstrap > 1:
            print(f'  Bootstrap {boot_step+1}/{n_bootstrap}: '
                  f'CAS scale = {scale:.2f}', flush=True)

        mydiis = lib.diis.DIIS()
        mydiis.space = diis_space
        this_tol = conv_tol if boot_step == n_bootstrap - 1 else bootstrap_tol
        this_max = max(max_cycle, 100) if boot_step == n_bootstrap - 1 else max(30, max_cycle // 2)

        e_prev = 0.0
        import time as _time

        # Phase 0 of DLPNO-restructure: build the canonical pair index.
        # Read-only for now; later phases will migrate pair-keyed dicts
        # (t2_pno_all, S_pno_cache, cc_ints, ovL_pno_cache, ...) onto
        # TensorStore instances backed by this index.  See pair_index.py.
        #
        # NB: keys_sorted only captures strong pairs (frozen before the
        # weak-pair extension of t2_pno_all above L1172). Downstream
        # CCSD functions iterate t2_pno_all directly, so the PairIndex
        # must cover the union of strong + weak pairs — hence we use
        # list(t2_pno_all.keys()) here.
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, assert_consistent_with_dicts, build_t1_cache,
        )
        _pair_index = PairIndex(
            list(t2_pno_all.keys()),
            pno_spaces, pair_lmo_idx, nocc)
        assert_consistent_with_dicts(
            _pair_index, pno_spaces, pair_lmo_idx)
        if boot_step == 0:
            print(f'  [pair_index] {_pair_index!r}', flush=True)

        for cycle in range(this_max):
            # Phase 1: pre-project t1 into every pair's PNO basis once
            # per cycle, replacing ~1.5 M lazy _project_t1_to_pair calls.
            # ``_t1_cache[key]`` is a (nocc, n_pno[key]) matrix;
            # ``_t1_cache[key][k]`` is the t1_k projection into pair key.
            _t1_cache = build_t1_cache(
                t1_pno, _pair_index, S_pno_cache, pno_spaces)
            _t_cycle_start = _time.perf_counter()
            t2_new = {}
            t1_pno_old = {i: t1_pno[i].copy() for i in range(nocc)}
            # Tell compute_residual_v2 the current iteration so PTERM dumps
            # can be filtered/labeled per iter.
            from pyscf.cc.dlpno_tccsd.residual import compute_residual_v2 as _crv2
            _crv2._iter = cycle
            from pyscf.cc.dlpno_tccsd.residual import compute_all_df_terms_local as _cadf
            _cadf._iter = cycle
            _crv2._debug_pterm_all = (cycle <= 2) and getattr(_run_dlpno_lccsd, '_debug_pterm_iters', False)
            # Pass strong-pair set to T1 residual so it can optionally skip
            _compute_t1_residual_psi4._iter = cycle

            # ---- T1-transformed MOs or bare integrals ----
            if use_t1_transform:
                # C̃_lmo = C_lmo + ψ automatically T1-dresses ALL integrals
                C_lmo_t1 = C_lmo.copy()
                for ii in range(nocc):
                    key_ii = (ii, ii)
                    if key_ii in pno_spaces and t1_pno[ii].size > 0:
                        C_pno_ii = pno_spaces[key_ii]['C_pno']
                        C_lmo_t1[:, ii] += C_pno_ii @ t1_pno[ii]

                # Per-iter rebuild of full-naux ovL/ooL/J_oo/K_coul_cache
                # is removed. cc_ints is built once up-front; t1_ints /
                # t1_fock dress the local-aux per-pair intermediates each
                # iteration. K_coul_cache fallback at _update_pair is now
                # served by cc_ints['J_ij_kj'] for cross-pair Coulomb.
                _t_ovl = _t_ovl_done = _t_kcoul = _t_kcoul_done = \
                    _time.perf_counter()
                _t_foo = _time.perf_counter()
                foo_total = _compute_foo_dressed_local(
                    t2_pno_all, pno_spaces, nocc, _cc_ints, _pool=_pool)
                foo_bare = foo_total
                _t_foo_done = _time.perf_counter()
            else:
                _t_ovl = _t_kcoul = _t_foo = _time.perf_counter()
                foo_total = _compute_foo_dressed_local(
                    t2_pno_all, pno_spaces, nocc, _cc_ints, _pool=_pool)
                foo_bare = foo_total
                _t_ovl_done = _t_kcoul_done = _t_foo_done = _time.perf_counter()

            # ---- T1-dressed intermediates (precomputed once per iteration) ----
            _jiang_cache = None
            from pyscf.cc.dlpno_tccsd.residual import (
                compute_C_tilde, build_D_tilde, build_G_tilde,
                build_mixed_domain_integrals,
            )

            _t_jiang = _time.perf_counter()

            # T1-dressed ooL/ovL (Eqs 91-92)
            _tj0 = _time.perf_counter()
            _all_keys_j = list(keys_sorted)
            for ii in range(nocc):
                kii = (ii, ii)
                if kii not in _all_keys_j and kii in pno_spaces:
                    _all_keys_j.append(kii)
            # Local DF: skip global dressed ovL/ooL (per-pair t1_ints provides them)
            _jiang_ooL_d = ooL_3idx  # bare ooL — per-pair B_tilde uses local
            _jiang_ovL_d = None  # not needed (K_dressed_override per pair)
            _tj_ovl = _time.perf_counter() - _tj0

            # Unified per-pair local-aux pass: fvv_t1, C/D Term 2, ladder.
            # Reads from cc_ints['Qab', 'Qma'] (already fitted, local-aux
            # per pair) — Psi4-equivalent. No global aux iteration.
            from pyscf.cc.dlpno_tccsd.residual import compute_all_df_terms_local
            _tj0 = _time.perf_counter()
            _fvv_t1_pre, _c_t2_pre, _d_t2_pre, _jiang_ladder_all = \
                compute_all_df_terms_local(
                    t1_pno, fov_pno, t2_pno_all, pno_spaces, nocc,
                    _cc_ints, S_pno_cache, keys_sorted, _pool=_pool,
                    pair_lmo_idx=pair_lmo_idx, t1_cache=_t1_cache)
            _tj_df = _time.perf_counter() - _tj0

            # compute_C_tilde, build_D_tilde, t1_fock each dispatch their
            # own ~400 tasks via _pool.map. Running them in 3 driver
            # threads concurrently contends for the shared 32-worker pool
            # (task-queue lock, cache lines on shared caches). Running
            # them sequentially — each grabbing the full pool for its
            # own pass — avoids contention while keeping per-kernel
            # parallelism intact.
            from pyscf.cc.dlpno_tccsd.local_df import t1_fock
            compute_C_tilde._dump_iter = cycle
            _tj0 = _time.perf_counter()

            _tct_old_0 = _time.perf_counter()
            _jiang_C = compute_C_tilde(
                t1_pno, t2_pno_all, pno_spaces, nocc,
                ovL_pno_cache, ooL_3idx, S_pno_cache, with_df,
                _term2_precomputed=_c_t2_pre,
                cc_ints=_cc_ints,
                pair_lmo_idx=pair_lmo_idx, _pool=_pool,
                S_pao_full=S_pao_full, s1e=s1e,
                t1_cache=_t1_cache)
            _tct_old = _time.perf_counter() - _tct_old_0

            # Prototype: batched compute_C_tilde (Terms 3+4 vectorized).
            # Runs side-by-side for validation + timing; no pool, 32-thread BLAS.
            from pyscf.cc.dlpno_tccsd.residual import compute_C_tilde_batched
            compute_C_tilde_batched._dump_timing = (cycle == 5)
            _tct_new_0 = _time.perf_counter()
            _jiang_C_new = compute_C_tilde_batched(
                t1_pno, t2_pno_all, pno_spaces, nocc,
                ovL_pno_cache, ooL_3idx, S_pno_cache, with_df,
                _term2_precomputed=_c_t2_pre,
                cc_ints=_cc_ints,
                pair_lmo_idx=pair_lmo_idx, _pool=_pool,
                S_pao_full=S_pao_full, s1e=s1e,
                blas_threads=32)
            _tct_new = _time.perf_counter() - _tct_new_0
            _max_diff = 0.0
            _missing = 0
            _extra = 0
            for _k in _jiang_C:
                if _k in _jiang_C_new:
                    _d = float(np.max(np.abs(_jiang_C[_k] - _jiang_C_new[_k])))
                    if _d > _max_diff:
                        _max_diff = _d
                else:
                    _missing += 1
            for _k in _jiang_C_new:
                if _k not in _jiang_C:
                    _extra += 1
            print(f'  [c_tilde_proto] iter={cycle} '
                  f'old={_tct_old*1e3:.0f}ms new={_tct_new*1e3:.0f}ms '
                  f'max_diff={_max_diff:.2e} '
                  f'missing={_missing} extra={_extra}', flush=True)
            _jiang_D = build_D_tilde(
                t1_pno, t2_pno_all, pno_spaces, nocc,
                ovL_pno_cache, ooL_3idx, S_pno_cache, with_df,
                _term2_precomputed=_d_t2_pre,
                cc_ints=_cc_ints,
                pair_lmo_idx=pair_lmo_idx, _pool=_pool,
                S_pao_full=S_pao_full, s1e=s1e,
                t1_cache=_t1_cache)
            _local_Fkj, _local_df_Fab, _local_foo_t1 = t1_fock(
                _cc_ints, None, t1_pno, fov_pno, pno_spaces,
                S_pno_cache, F_lmo, eps_lmo, foo_total,
                _all_keys_j, nocc, _pool=_pool,
                pair_lmo_idx=pair_lmo_idx, t1_cache=_t1_cache)
            _tj_C = _time.perf_counter() - _tj0
            _tj_D = 0.0
            _tj_FG = 0.0
            _tj_Fab = 0.0

            # Mixed-domain integrals for C/D bold terms (negligible; serial is fine)
            _tj0 = _time.perf_counter()
            _jiang_K_mixed = build_mixed_domain_integrals(
                t2_pno_all, pno_spaces, nocc,
                ovL_pno_cache, ooL_3idx, S_pno_cache,
                cc_ints=_cc_ints)
            _tj_Km = _time.perf_counter() - _tj0

            _jiang_J_oo_d = None

            _local_df_G = build_G_tilde(
                t2_pno_all, t1_pno, pno_spaces, nocc,
                ovL_pno_cache, ooL_3idx, S_pno_cache,
                _local_Fkj, _local_foo_t1,
                cc_ints=_cc_ints,
                S_pao_full=S_pao_full, s1e=s1e)

            _t_jiang_done = _time.perf_counter()
            if getattr(_run_dlpno_lccsd, '_detail_jiang', False):
                print(f'    [jiang] ovL_d={_tj_ovl:.3f} df={_tj_df:.3f} '
                      f'C={_tj_C:.3f} D={_tj_D:.3f} FG={_tj_FG:.3f} '
                      f'Km={_tj_Km:.3f} Fab={_tj_Fab:.3f} '
                      f'total={_t_jiang_done-_t_jiang:.3f}s', flush=True)

            _jiang_cache = {
                'C_tilde': _jiang_C, 'D_tilde': _jiang_D,
                'G_tilde': None,
                'ovL_dressed': None,
                'ooL_dressed': None,
                'J_oo_d': None,
                'K_mixed': _jiang_K_mixed,
                'Fab_all': None,
                'ladder_all': _jiang_ladder_all,
            }

            _t_pairs = _time.perf_counter()

            # --- Batched B + E contributions across all strong pairs ---
            # Replaces the per-pair `for key_kl in t2_pno_all` loops that
            # dominate CCSD CPU time (~77% per profile).  Each strong pair
            # has its own B_tilde (from compute_B_tilde); precompute them
            # as a dict then call compute_B_E_batched once.
            from pyscf.cc.dlpno_tccsd.local_df import compute_B_tilde as _cB_fn
            from pyscf.cc.dlpno_tccsd.residual import compute_B_E_batched

            def _bt_one(_key):
                return _key, _cB_fn(_cc_ints, None, t2_pno_all, t1_pno,
                                    pno_spaces, S_pno_cache, _key, nocc,
                                    pair_lmo_idx=pair_lmo_idx,
                                    t1_cache=_t1_cache)
            _B_tilde_per_ij = {}
            if _pool is not None:
                for _k, _bt in _pool.map(_bt_one, keys_sorted):
                    _B_tilde_per_ij[_k] = _bt
            else:
                for _k in keys_sorted:
                    _, _bt = _bt_one(_k)
                    _B_tilde_per_ij[_k] = _bt

            # Batched B+E across all strong pairs — one pool dispatch.
            # The function iterates keys internally via _pool.map.
            _B_dict, _E_dict = compute_B_E_batched(
                keys_sorted, t2_pno_all, pno_spaces, S_pno_cache,
                _cc_ints, _B_tilde_per_ij, pair_lmo_idx, nocc,
                _pool=_pool, S_pao_full=S_pao_full, s1e=s1e)
            _BE_all = {'B': _B_dict, 'E': _E_dict}

            # Accumulator for per-pair timing (thread-safe via list append)
            _pair_timings = {'fab': [], 'resid': [], 'btilde': []}

            def _update_pair(key):
                """Compute T2 update for a single pair. Thread-safe (read-only
                access to shared caches, writes only to returned arrays)."""
                i, j = key
                data = pno_spaces[key]
                n_pno = data['C_pno'].shape[1]
                if n_pno == 0:
                    return key, np.zeros((0, 0))

                from pyscf.cc.dlpno_tccsd.residual import (
                    compute_residual_v2)
                jc = _jiang_cache

                # Per-pair intermediates: use local DF if available
                _tb0 = _time.perf_counter()
                _K_dressed_local = None
                _ladder_local = None
                if key in _cc_ints and _cc_ints[key] is not None:
                    from pyscf.cc.dlpno_tccsd.local_df import (
                        t1_ints, compute_ladder)
                    # T1-dressed K̃ (local DF)
                    _dressed = t1_ints(
                        _cc_ints, t1_pno, pno_spaces, S_pno_cache,
                        [key], nocc, pair_lmo_idx=pair_lmo_idx,
                        t1_cache=_t1_cache)
                    if key in _dressed:
                        _K_dressed_local = (_dressed[key]['i_Qa_t1'].T
                                            @ _dressed[key]['j_Qa_t1'])
                    # B_tilde reused from precomputed dict
                    _B_tilde_local = _B_tilde_per_ij.get(key)
                    _ladder_local = compute_ladder(
                        _cc_ints, t2_pno_all, t1_pno, pno_spaces,
                        S_pno_cache, key, nocc,
                        pair_lmo_idx=pair_lmo_idx,
                        t1_cache=_t1_cache)
                    # Fab from local DF
                    if _local_df_Fab is not None and key in _local_df_Fab:
                        Fab_ij = _local_df_Fab[key]
                    else:
                        Fab_ij = jc['Fab_all'][key]
                    # J_ij_kj from local DF
                    ci = _cc_ints[key]
                    J_pair = ci.get('J_ij_kj', {})
                    for k in range(nocc):
                        if (key, k) not in J_pair:
                            kj = (min(k, j), max(k, j))
                            kc = (key, kj, i, k)
                            if K_coul_cache and kc in K_coul_cache:
                                J_pair[(key, k)] = K_coul_cache[kc]
                    _K_mixed_local = ci.get('K_ij_kj', {})
                else:
                    _B_tilde_local = None
                if _B_tilde_local is None:
                    Fab_ij = jc['Fab_all'][key]
                    t1_i = _t1_cache[key][i]
                    t1_j = _t1_cache[key][j]
                    tau = t2_pno_all[key] + np.outer(t1_i, t1_j)
                    B_tilde_oo = jc['J_oo_d'][:, i, :, j].copy()
                    _naux_ij = ovL_pno_cache.get((key, 0))
                    if _naux_ij is not None:
                        _na = _naux_ij.shape[1]
                        ovL_stack = np.zeros((nocc, n_pno, _na))
                        for k in range(nocc):
                            _ok = ovL_pno_cache.get((key, k))
                            if _ok is not None:
                                ovL_stack[k] = _ok
                        P_stack = np.einsum('ba,kaQ->kbQ', tau.T, ovL_stack)
                        P_flat = P_stack.reshape(nocc, -1)
                        ovL_flat = ovL_stack.reshape(nocc, -1)
                        B_tilde_oo += P_flat @ ovL_flat.T
                    J_pair = {}
                    _K_mixed_local = None
                    for k in range(nocc):
                        kj = (min(k, j), max(k, j))
                        kc = (key, kj, i, k)
                        if K_coul_cache and kc in K_coul_cache:
                            J_pair[(key, k)] = K_coul_cache[kc]
                _pair_timings['btilde'].append(
                    _time.perf_counter() - _tb0)

                _tr0 = _time.perf_counter()
                R_ij = compute_residual_v2(
                    i, j, t2_pno_all, pno_spaces, nocc,
                    F_lmo, s1e, with_df, eps_lmo,
                    ovL_bare=ovL_pno_cache,
                    ooL_bare=ooL_3idx,
                    S_pno_cache=S_pno_cache,
                    K_coul_cache=K_coul_cache,
                    K_pno_bare=K_pno_cache,
                    ovL_dressed=jc['ovL_dressed'],
                    ooL_dressed=jc['ooL_dressed'],
                    B_tilde=(_B_tilde_local if _B_tilde_local is not None
                             else B_tilde_oo),
                    Fab={key: Fab_ij},
                    G_tilde=(_local_df_G if _local_df_G is not None
                             else jc['G_tilde']),
                    C_tilde_cache=jc['C_tilde'],
                    D_tilde_cache=jc['D_tilde'],
                    J_ij_kj=J_pair,
                    K_ij_kj=(_K_mixed_local if _K_mixed_local
                             else jc['K_mixed']),
                    t1_pno=t1_pno,
                    ladder_precomputed=({key: _ladder_local}
                        if _ladder_local is not None
                        else jc.get('ladder_all')),
                    K_dressed_override=_K_dressed_local,
                    cc_ints=_cc_ints,
                    pair_domain=pair_lmo_idx.get(key) if pair_lmo_idx else None,
                    B_term_override=(_BE_all['B'].get(key)
                                      if _BE_all is not None else None),
                    E_contrib_override=(_BE_all['E'].get(key)
                                         if _BE_all is not None else None),
                    S_pao_full=S_pao_full,
                    t1_cache=_t1_cache)
                _pair_timings['resid'].append(
                    _time.perf_counter() - _tr0)
                e_pno = data['e_pno']
                # Psi4 increment form (ccsd.cc line 2472):
                # T_new = T_old - R / (e_a + e_b - F_ii - F_jj)
                D_psi4 = (e_pno[:, None] + e_pno[None, :]
                          - eps_lmo[i] - eps_lmo[j])
                D_psi4 = np.where(np.abs(D_psi4) > 1e-12, D_psi4, 1e-12)
                T2_old = t2_pno_all[key]
                T2_ij_new = T2_old - R_ij / D_psi4
                if key in cas_blocks:
                    cb = cas_blocks[key]
                    cas_sl = cb[0]
                    if len(cb) == 3:
                        _, t2c_dmrg_ref, t2c_mp2 = cb
                        T2_ij_new[cas_sl, cas_sl] = (
                            t2c_mp2 + scale * (t2c_dmrg_ref - t2c_mp2))
                    else:
                        T2_ij_new[cas_sl, cas_sl] = cb[1]
                return key, T2_ij_new, R_ij

            r2_all = {}
            # Enable per-term profiling in compute_residual_v2 for this cycle
            from pyscf.cc.dlpno_tccsd.residual import compute_residual_v2 as _crv2
            _crv2._term_times = {}
            if _pool is not None:
                for key, T2_ij_new, R_ij in _pool.map(_update_pair, keys_sorted):
                    t2_new[key] = T2_ij_new
                    r2_all[key] = R_ij
            else:
                for key in keys_sorted:
                    _, T2_ij_new, R_ij = _update_pair(key)
                    t2_new[key] = T2_ij_new
                    r2_all[key] = R_ij
            _tt = _crv2._term_times
            _crv2._term_times = None
            if _tt:
                _n_calls = _tt.pop('_n', 1)
                _tot = sum(_tt.values())
                _per = ' '.join(f'{k}={v:.2f}' for k, v in sorted(_tt.items()))
                print(f'  [residual] n_calls={_n_calls} sum_CPU={_tot:.2f}s  {_per}', flush=True)

            _t_pairs_done = _time.perf_counter()
            if getattr(_run_dlpno_lccsd, '_detail_pairs', False):
                _sb = sum(_pair_timings['btilde'])
                _sr = sum(_pair_timings['resid'])
                print(f'    [pairs] btilde={_sb:.3f}s resid={_sr:.3f}s '
                      f'({len(_pair_timings["btilde"])} pairs)', flush=True)
            _pair_timings['fab'].clear()
            _pair_timings['btilde'].clear()
            _pair_timings['resid'].clear()
            # ---- Local T1 update ----
            if use_t1_transform and not getattr(
                    _run_dlpno_lccsd, '_dress_t1', False):
                # Use BARE integrals for T1 (standard split-basis approach).
                _t1_ovL = ovL_pno_bare
                _t1_foo = foo_bare
                _t1_ooL = ooL_bare
                _t1_fov = fov_pno
                _t1_C = C_lmo
            else:
                # Use the same integrals as T2 (dressed or bare).
                _t1_ovL = ovL_pno_cache
                _t1_foo = foo_total
                _t1_ooL = ooL_3idx
                _t1_C = C_lmo_t1 if use_t1_transform else C_lmo
                # Recompute fov with dressed MOs if using T1 transform
                if use_t1_transform and hasattr(
                        _run_dlpno_lccsd, '_dress_t1') and \
                        _run_dlpno_lccsd._dress_t1:
                    _t1_fov = {}
                    for ii in range(nocc):
                        key_ii = (ii, ii)
                        if key_ii not in pno_spaces:
                            _t1_fov[ii] = np.zeros(0)
                            continue
                        C_pno_ii = pno_spaces[key_ii]['C_pno']
                        if C_pno_ii.shape[1] == 0:
                            _t1_fov[ii] = np.zeros(0)
                            continue
                        _t1_fov[ii] = C_pno_ii.T @ (fock_ao @ _t1_C[:, ii])
                else:
                    _t1_fov = fov_pno
            _t1_start = _time.perf_counter()
            r1_pno = _compute_t1_residual_psi4(
                t1_pno, t2_pno_all, pno_spaces, _t1_fov, F_lmo, eps_lmo, nocc,
                S_pno_cache, _cc_ints, ovL_pno_cache=_t1_ovL,
                pair_lmo_idx=pair_lmo_idx, t1_cache=_t1_cache)
            t1_pno_new = {}
            for ii in range(nocc):
                key_ii = (ii, ii)
                if key_ii not in pno_spaces or r1_pno[ii].size == 0:
                    t1_pno_new[ii] = np.zeros(0) if ii not in t1_pno else t1_pno[ii].copy()
                    continue
                e_pno_ii = pno_spaces[key_ii]['e_pno']
                # Psi4 INCREMENT form (ccsd.cc line 2456):
                #   T_ia[a] -= R_ia[a] / (e_pno[a] - F_lmo[i,i])
                D_i = e_pno_ii - eps_lmo[ii]
                D_i = np.where(np.abs(D_i) > 1e-12, D_i, 1e-12)
                t1_pno_new[ii] = t1_pno[ii] - r1_pno[ii] / D_i
            t1_pno = t1_pno_new
            _t1_end = _time.perf_counter()

            # T1 diagnostics
            if cycle < 3 or cycle % 5 == 0:
                t1_norm = np.sqrt(sum(np.dot(t1_pno[ii], t1_pno[ii])
                                      for ii in range(nocc) if t1_pno[ii].size > 0))
                print(f'    [T1] |t1| = {t1_norm:.6f}', flush=True)

            # ---- DIIS on external amplitudes only (Psi4-style R-based) ----
            def _mask_cas(t2_dict):
                pieces = []
                for k in keys_sorted:
                    t2k = t2_dict[k].copy()
                    if k in cas_blocks:
                        cas_sl = cas_blocks[k][0]
                        t2k[cas_sl, cas_sl] = 0.0
                    pieces.append(t2k.ravel())
                return np.concatenate(pieces)

            t2_vec_new = _mask_cas(t2_new)
            t2_vec_old = _mask_cas(t2_pno_all)

            # Combine local T1 and T2 into a single amplitude vector for DIIS
            t1_vec_new = np.concatenate([t1_pno[ii].ravel() for ii in range(nocc)])
            t1_vec_old = np.concatenate([t1_pno_old[ii].ravel() for ii in range(nocc)])
            amp_new = np.concatenate([t1_vec_new, t2_vec_new])
            amp_old = np.concatenate([t1_vec_old, t2_vec_old])
            err_amp = amp_new - amp_old
            dT = np.max(np.abs(err_amp)) if err_amp.size > 0 else 0.0

            # Psi4-style DIIS: error vector is the raw residual R = (R1, R2),
            # not ΔT = -R/D.  The ΔT form blows up on pathological near-zero
            # orbital-energy denominators (ghost-CP basis with diffuse
            # contamination, e.g. S22-16 ethene-ethyne a_ghost).  R-based
            # DIIS mirrors Psi4 DLPNO-CCSD (ccsd.cc:2479-2499) which stores
            # (T1, T2) as the DIIS vector and (R1, R2) as the error vector
            # with DIIS_MAX_VECS=5 (Psi4 CC default).
            r1_vec = np.concatenate([r1_pno[ii].ravel() for ii in range(nocc)])
            r2_vec = _mask_cas(r2_all)
            err_vec = np.concatenate([r1_vec, r2_vec])
            if cycle >= diis_start_cycle and err_vec.size > 0:
                amp_new = mydiis.update(amp_new, err_vec)

            # ---- Unpack T1 and T2; restore CAS blocks ----
            offset = 0
            for ii in range(nocc):
                sz = t1_pno[ii].size
                t1_pno[ii] = amp_new[offset:offset + sz]
                offset += sz

            for k in keys_sorted:
                sz = t2_new[k].size
                t2_pno_all[k] = amp_new[offset:offset + sz].reshape(t2_new[k].shape)
                offset += sz

            for key, cb in cas_blocks.items():
                if key not in t2_pno_all:
                    continue
                cas_sl = cb[0]
                if len(cb) == 3:
                    _, t2c_dmrg_ref, t2c_mp2 = cb
                    t2_pno_all[key][cas_sl, cas_sl] = (
                        t2c_mp2 + scale * (t2c_dmrg_ref - t2c_mp2))
                else:
                    t2_pno_all[key][cas_sl, cas_sl] = cb[1]

            # Compute energy for monitoring (Eq. 102 of Jiang et al.)
            # E = Σ_i F_ia t1_ia + Σ_ij K_ij^{ab} (2τ-τ^T)
            # Use BARE integrals (canonical energy formula).
            e_cyc = sum(np.dot(fov_pno[ii], t1_pno[ii])
                        for ii in range(nocc) if t1_pno[ii].size > 0)
            for pair in strong_pairs:
                pi, pj = pair
                pk = (min(pi, pj), max(pi, pj))
                if pk not in t2_pno_all:
                    continue
                T2_p = t2_pno_all[pk]
                K_p = K_pno_cache.get(pk)
                if K_p is None:
                    K_p = pno_spaces.get(pk, {}).get('K_pno')
                if T2_p.shape[0] == 0 or K_p is None:
                    continue
                # tau = t2 + t1_i ⊗ t1_j in PNO basis (Phase 1: cached).
                t1i_pno = _t1_cache[pk][pk[0]]
                t1j_pno = _t1_cache[pk][pk[1]]
                tau_p = T2_p + np.outer(t1i_pno, t1j_pno)
                Tt_p = 2.0 * tau_p - tau_p.T
                e_p = np.einsum('ab,ab->', K_p, Tt_p)
                e_cyc += e_p if pi == pj else 2.0 * e_p
            dE = abs(e_cyc - e_prev) if cycle > 0 else float('inf')
            e_prev = e_cyc
            _t_cycle_end = _time.perf_counter()
            _dt_ovl = _t_ovl_done - _t_ovl
            _dt_kcoul = _t_kcoul_done - _t_kcoul
            _dt_foo = _t_foo_done - _t_foo
            _dt_pairs = _t_pairs_done - _t_pairs
            _dt_jiang = _t_jiang_done - _t_jiang
            _dt_total = _t_cycle_end - _t_cycle_start
            _jiang_str = f' jiang={_dt_jiang:.1f}'
            print(f'  Cycle {cycle + 1:3d}: dT = {dT:.3e}  E_corr = {e_cyc:.10f}'
                  f'  dE = {dE:.2e}  [{_dt_total:.1f}s: ovL={_dt_ovl:.1f} '
                  f'Kcoul={_dt_kcoul:.1f} foo={_dt_foo:.1f}'
                  f'{_jiang_str} pairs={_dt_pairs:.1f}]', flush=True)
            if dT < this_tol:
                print(f'  DLPNO-CCSD converged in {cycle + 1} cycles (amplitude).',
                      flush=True)
                break
            # Secondary energy-based convergence.  Local-orbital CCSD often
            # has dT oscillating at ~1e-3 while the energy is already
            # converged to μEh — the redundant PNO overlap across pairs
            # prevents dT from reaching conv_tol even when E is stable.
            # Fall back to |dE| < conv_tol (same order as amplitude tol).
            e_conv_tol = this_tol
            if cycle > 5 and dE < e_conv_tol:
                print(f'  DLPNO-CCSD converged in {cycle + 1} cycles (energy, '
                      f'dE={dE:.2e}).',
                      flush=True)
                break
        else:
            if boot_step == n_bootstrap - 1:
                print('  WARNING: DLPNO-CCSD did not converge.', flush=True)

    # ------------------------------------------------------------------
    # Total correlation energy (Eq. 102 of Jiang et al.)
    # E = Σ_i F_ia t1_ia + Σ_ij K_ij^{ab} (2τ-τ^T)
    # Use BARE integrals (canonical energy formula).
    # ------------------------------------------------------------------
    e_total = sum(np.dot(fov_pno[ii], t1_pno[ii])
                  for ii in range(nocc) if t1_pno[ii].size > 0)
    for pair in strong_pairs:
        i, j = pair
        key = (min(i, j), max(i, j))
        if key not in t2_pno_all or key not in pno_spaces:
            continue
        T2 = t2_pno_all[key]
        K = K_pno_cache.get(key)
        if K is None:
            K = pno_spaces[key].get('K_pno')
        if T2.shape[0] == 0 or K is None:
            continue
        t1i_pno = _project_t1_to_pair(
            t1_pno, key[0], key, S_pno_cache, pno_spaces)
        t1j_pno = _project_t1_to_pair(
            t1_pno, key[1], key, S_pno_cache, pno_spaces)
        tau = T2 + np.outer(t1i_pno, t1j_pno)
        Tt = 2.0 * tau - tau.T
        e_ij = np.einsum('ab,ab->', K, Tt)
        e_total += e_ij if i == j else 2.0 * e_ij

    return e_total, t2_pno_all, t1_pno


# ---------------------------------------------------------------------------
# Scalable coupled LCCSD (old Psi4-style: superseded by _run_dlpno_lccsd)
# ---------------------------------------------------------------------------

def run_lccsd(mf, C_lmo, pno_spaces, strong_pairs, cas_pairs,
              t1_cas, t2_cas, occ_cas_idx, vir_cas_idx,
              mo_coeff_cas, s1e=None,
              conv_tol=1e-7, max_cycle=50, ncores=1,
              C_pao=None, max_outer_cycle=30, outer_conv_tol=1e-8,
              verbose=None, negligible_pairs=None, _pool=None):
    """Run pair-local CCSD over all strong pairs, injecting CAS amplitudes.

    For each strong pair (i,j):
      - If (i,j) is a CAS pair: use DMRG amplitudes directly (t2_ij from t2_cas),
        do not run CCSD iterations for this pair.
      - Otherwise: run CCSD in the pair's PNO space with the standard
        update_amps, using mf as integral source.

    The total LCCSD correlation energy is the sum of fragment energies from
    all strong pairs (with double-counting removed by the 1/2 weight for i<j).

    Args:
        mf: RHF object with with_df set. Source of all integrals.
        C_lmo (np.ndarray): (nao, nocc_lmo). LMO coefficients.
        pno_spaces (dict): Output of pno.make_pnos.
        strong_pairs (list): List of (i,j) strong pairs.
        cas_pairs (set): Set of (i,j) CAS pairs (subset of strong_pairs).
        t1_cas (np.ndarray): (nocc_cas, nvir_cas) CAS singles amplitudes.
        t2_cas (np.ndarray): (nocc_cas, nocc_cas, nvir_cas, nvir_cas) CAS doubles.
        occ_cas_idx (np.ndarray): CAS occupied indices in full MO list.
        vir_cas_idx (np.ndarray): CAS virtual indices in full MO list.
        mo_coeff_cas (np.ndarray): mc.mo_coeff — full CASSCF-optimized MOs.
        s1e (np.ndarray, optional): AO overlap matrix.
        conv_tol (float): CCSD convergence threshold.
        max_cycle (int): Maximum CCSD iterations.
        verbose: Verbosity level.

    Returns:
        e_tccsd (float): DLPNO-TCCSD correlation energy (strong pairs only).
        t2_pno_all (dict): (i,j) → converged T2 in PNO basis (for (T) triples).
        t1_pno (dict): i → (n_pno_ii,) local T1 amplitudes in diagonal PNO
            basis following Jiang et al. JCP 2024.
    """
    log = logger.new_logger(mf, verbose)
    mol = mf.mol

    if s1e is None:
        s1e = mf.get_ovlp()

    nocc_lmo = C_lmo.shape[1]
    occ_cas_set = set(occ_cas_idx.tolist())

    # AO Fock: used for LMO orbital energies AND passed to fake_mf for pair CCSD
    fock_ao = mf.get_fock()
    F_lmo = reduce(np.dot, (C_lmo.T, fock_ao, C_lmo))
    eps_lmo = F_lmo.diagonal().real

    n_total = len(strong_pairs)
    print(f'  LCCSD: solving {n_total} strong pairs (ncores={ncores})...', flush=True)

    e_tccsd, t2_pno_all, t1_pno = _run_dlpno_lccsd(
        mf, C_lmo, pno_spaces, strong_pairs,
        fock_ao, eps_lmo, s1e, conv_tol, max_cycle,
        t2_cas=t2_cas, occ_cas_idx=occ_cas_idx,
        vir_cas_idx=vir_cas_idx, mo_coeff_cas=mo_coeff_cas,
        C_pao=C_pao,
        ncores=ncores, negligible_pairs=negligible_pairs,
        _pool=_pool)
    print(f'  E_TCCSD = {e_tccsd:.15g}', flush=True)

    return e_tccsd, t2_pno_all, t1_pno


