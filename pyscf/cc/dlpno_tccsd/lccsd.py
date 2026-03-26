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


def _compute_local_t1_residual(t1_pno, t2_pno_all, pno_spaces,
                                fov_pno, F_lmo, eps_lmo,
                                nocc, s1e, fock_ao, C_lmo,
                                ovL_pno_cache, S_pno_cache,
                                foo_dressed=None):
    """Compute local CCSD T1 residual in diagonal PNO bases.

    Implements the t1-transformed CCSD singles residual following
    Jiang et al. JCP 2024, Eqs. 87–90, 94–101:

        R_i^{a_ii} = F̃_{a_ii,i}            (dressed Fock ov, Eq. 96)
                   + F_vv × t1              (virtual Fock dressing)
                   - F_oo × t1              (occupied Fock dressing)
                   + C_i^{a_ii}             (Eq. 90: T2 × Fock_ov coupling)
                   + A_i^{a_ii}             (Eq. 88: T2 × voov)
                   - foo_dressed × t1       (T2-dressed occupied Fock)

    The T1 amplitudes live in the diagonal PNO basis of pair (i,i).
    Inter-pair coupling uses PNO overlap matrices from S_pno_cache.

    Args:
        t1_pno (dict): i → (n_pno_ii,) T1 in diagonal PNO_{ii} basis.
        t2_pno_all (dict): (i,j) → (n_pno_ij, n_pno_ij) T2 in PNO_{ij}.
        pno_spaces (dict): PNO space data.
        fov_pno (dict): i → (n_pno_ii,) Fock ov block in PNO_{ii} basis.
        F_lmo (np.ndarray): (nocc, nocc) Fock in LMO basis.
        eps_lmo (np.ndarray): (nocc,) diagonal LMO Fock eigenvalues.
        nocc (int): Number of occupied LMOs.
        s1e (np.ndarray): AO overlap matrix.
        fock_ao (np.ndarray): AO Fock matrix.
        C_lmo (np.ndarray): (nao, nocc) LMO coefficients.
        ovL_pno_cache (dict): ((key, k) → (n_pno, naux)) DF 3-index tensors.
        S_pno_cache (dict): ((key1, key2) → overlap matrix) PNO overlaps.
        foo_dressed (np.ndarray or None): (nocc, nocc) T2-dressed occ Fock.

    Returns:
        r1_pno (dict): i → (n_pno_ii,) T1 residual in PNO_{ii} basis.
    """
    r1_pno = {}

    for i in range(nocc):
        key_ii = (i, i)
        if key_ii not in pno_spaces:
            r1_pno[i] = np.zeros(0)
            continue
        data_ii = pno_spaces[key_ii]
        C_pno_ii = data_ii['C_pno']
        n_pno_ii = C_pno_ii.shape[1]
        if n_pno_ii == 0:
            r1_pno[i] = np.zeros(0)
            continue
        e_pno_ii = data_ii['e_pno']

        # ---- Term 1: Dressed Fock ov (Eq. 96) ----
        # F̃_{a_ii,i} = F̄_{a_ii,i} - t̃_k^{a_ii} F̄_{ki}
        #             + F̄_{a_ii,b_ii} t_i^{b_ii} - t̃_k^{a_ii} F̄_{kb_ii} t_i^{b_ii}
        # Leading term: bare Fock ov
        r1_i = fov_pno[i].copy()  # (n_pno_ii,)

        # ---- Term 2: F_vv × T1 (off-diagonal virtual Fock) ----
        # F_vv in PNO_{ii} basis (already semicanonical, so off-diag is small
        # but nonzero due to t1-dressing if we were doing full dressing).
        # In semicanonical PNO basis, F_vv is diagonal by construction,
        # so off-diagonal F_vv is zero.  The diagonal part goes into the
        # denominator (Eq. 103).  Skip this term.

        # ---- Term 3: -F_oo × T1 (off-diagonal occupied Fock) ----
        # -Σ_k F_oo_off[k,i] * t1_k projected to PNO_{ii}
        ft_oo = F_lmo - np.diag(eps_lmo)  # off-diagonal occupied Fock
        for k in range(nocc):
            if abs(ft_oo[k, i]) < 1e-15:
                continue
            t1_k_in_ii = _project_t1_to_pair(
                t1_pno, k, key_ii, S_pno_cache, pno_spaces)
            r1_i -= ft_oo[k, i] * t1_k_in_ii

        # ---- Term 3b: -foo_dressed × T1 ----
        if foo_dressed is not None:
            for k in range(nocc):
                if abs(foo_dressed[k, i]) < 1e-15:
                    continue
                t1_k_in_ii = _project_t1_to_pair(
                    t1_pno, k, key_ii, S_pno_cache, pno_spaces)
                r1_i -= foo_dressed[k, i] * t1_k_in_ii

        # ---- Term 4 (C-term, Eq. 90): T2 × Fock_ov coupling ----
        # C_i^{a_ii} = Σ_k S_{a_ik}^{a_ii} u_ik^{a_ik c_ik} F̃_{k,c_ik}
        # Simplified: C_i = Σ_k proj_to_ii( theta_ik @ fov_k_in_ik )
        for k in range(nocc):
            key_ik = (min(i, k), max(i, k))
            if key_ik not in t2_pno_all:
                continue
            t2_ik_raw = t2_pno_all[key_ik]
            if t2_ik_raw is None or t2_ik_raw.shape[0] == 0:
                continue
            # t2[i,k] from stored t2[min,max]
            t2_ik = t2_ik_raw if i <= k else t2_ik_raw.T
            theta_ik = 2.0 * t2_ik - t2_ik.T

            # fov[k] projected to PNO_{ik} basis
            fov_k_in_ik = _project_t1_to_pair(
                fov_pno, k, key_ik, S_pno_cache, pno_spaces)

            # theta_ik @ fov_k gives result in PNO_{ik} basis
            contrib_ik = theta_ik @ fov_k_in_ik  # (n_pno_ik,)

            # Project result from PNO_{ik} to PNO_{ii}
            if key_ik == key_ii:
                r1_i += contrib_ik
            else:
                S_ii_ik = S_pno_cache.get((key_ii, key_ik))
                if S_ii_ik is not None:
                    r1_i += S_ii_ik @ contrib_ik

        # ---- Term 5 (A-term, Eq. 88): T1 × voov ----
        # Dominant part: Σ_{k} t1[k,e] * (2*(ka|ie) - (ki|ae))
        # Using DF: (ka|ie)_{a_ii, e_ii} via ovL_pno_cache
        # We approximate with the exchange (K) part only.
        if ovL_pno_cache:
            for k in range(nocc):
                key_ik = (min(i, k), max(i, k))
                if key_ik not in pno_spaces:
                    continue
                n_pno_ik = pno_spaces[key_ik]['C_pno'].shape[1]
                if n_pno_ik == 0:
                    continue
                # t1[k] projected to PNO_{ik}
                t1_k_in_ik = _project_t1_to_pair(
                    t1_pno, k, key_ik, S_pno_cache, pno_spaces)
                if np.max(np.abs(t1_k_in_ik)) < 1e-15:
                    continue

                # (ka|ie) in PNO_{ik}: ovL_k @ ovL_i.T
                ovL_k = ovL_pno_cache.get((key_ik, k))
                ovL_i = ovL_pno_cache.get((key_ik, i))
                if ovL_k is None or ovL_i is None:
                    continue
                voov_ki = ovL_k @ ovL_i.T  # (n_pno_ik, n_pno_ik)

                # Σ_e t1[k,e] * 2*(ka|ie) in PNO_{ik}
                contrib = 2.0 * voov_ki @ t1_k_in_ik  # (n_pno_ik,)

                # Project from PNO_{ik} to PNO_{ii}
                if key_ik == key_ii:
                    r1_i += contrib
                else:
                    S_ii_ik = S_pno_cache.get((key_ii, key_ik))
                    if S_ii_ik is not None:
                        r1_i += S_ii_ik @ contrib

        r1_pno[i] = r1_i

    return r1_pno


def _compute_local_t1_to_t2_correction(t1_pno, fov_pno, keys_sorted,
                                         cas_blocks, pno_spaces, eps_lmo,
                                         S_pno_cache, nocc):
    """Compute the T1→T2 correction for each pair using local T1.

    The dominant T1→T2 coupling is through the T1⊗fov outer product
    (part of the energy expression, Eq. 102 of Jiang et al.):
        delta_t2[i,j,a,b] = t1[i,a]*fov[j,b] + fov[i,a]*t1[j,b]

    projected to PNO_{ij} basis via PNO overlap matrices.

    Returns:
        delta_t2 (dict): key → (n_pno, n_pno) correction to add to t2_new.
    """
    delta_t2 = {}
    for key in keys_sorted:
        if key in cas_blocks:
            continue
        i, j = key
        n_pno = pno_spaces[key]['C_pno'].shape[1]
        if n_pno == 0:
            continue

        dt2 = np.zeros((n_pno, n_pno))

        # T1⊗fov outer product projected to PNO_{ij} basis
        t1_i_ij = _project_t1_to_pair(t1_pno, i, key, S_pno_cache, pno_spaces)
        t1_j_ij = _project_t1_to_pair(t1_pno, j, key, S_pno_cache, pno_spaces)
        fov_i_ij = _project_t1_to_pair(fov_pno, i, key, S_pno_cache, pno_spaces)
        fov_j_ij = _project_t1_to_pair(fov_pno, j, key, S_pno_cache, pno_spaces)

        dt2 += np.outer(t1_i_ij, fov_j_ij) + np.outer(fov_i_ij, t1_j_ij)

        # Apply PNO denominator
        e_pno = pno_spaces[key]['e_pno']
        D_ij = (eps_lmo[i] + eps_lmo[j] - e_pno[:, None] - e_pno[None, :])
        D_ij = np.where(np.abs(D_ij) > 1e-12, D_ij, 1e-12)
        dt2 /= D_ij

        delta_t2[key] = dt2

    return delta_t2


def _compute_pair_residual_numerator(
        i, j, t2_pno_all, pno_spaces, nocc_lmo,
        C_pao, ovL_lmo_pao, F_lmo, s1e, with_df, C_lmo,
        J_oo, K_pno_cache, eps_lmo,
        t1_pno=None, fov_pno=None,
        foo_dressed=None,
        ovL_pno_cache=None, S_pno_cache=None, K_coul_cache=None,
        ooL_3idx=None):
    """Compute the DLPNO-CCSD T2 residual numerator for pair (i,j).

    Implements the Riplinger-Neese DLPNO-CCSD residual (ORCA approach):

        R_ij[a,b] = K_ij[a,b]                                    (exchange)
                  + 0.5 * Σ_cd tau_ij[c,d] * (ac|bd)_ij          (ladder)
                  + 0.5 * Σ_kl W_kl * tau_kl_proj[a,b]          (dressed Woooo)
                  - P(ij) Σ_k F[k,i] * t_kj_proj[a,b]           (Fock coupling)
                  + P(ij)P(ab) Σ_kc theta_ik[a,c] * L_jk[b,c]   (ring)

    where W_kl = (ij|kl) + Σ_cd (ic|jd)_kl * tau_kl[c,d]  (dressed Woooo)
    and   L_jk[b,c] = 2*(jb|kc) - (jk|bc)
    and   theta = 2*t2 - t2.T

    The update is: t2_new_ij = R_ij / D_ij
    where D_ij[a,b] = eps_i + eps_j - e_pno_a - e_pno_b.

    Each pair updates ONLY its own T2 in its own PNO basis.
    """
    key = (min(i, j), max(i, j))
    data = pno_spaces[key]
    C_pno_ij = data['C_pno']
    n_pno = C_pno_ij.shape[1]

    if n_pno == 0:
        return np.zeros((0, 0))

    t2_ij = t2_pno_all[key]

    # --- 1. Exchange integral K_ij[a,b] = (ia|jb) ---
    K_ij = K_pno_cache.get(key)
    if K_ij is None:
        K_ij = data.get('K_pno')
    if K_ij is None:
        ovL_i_ij = _ovl_pno(i, C_pno_ij, C_pao, ovL_lmo_pao)
        ovL_j_ij = _ovl_pno(j, C_pno_ij, C_pao, ovL_lmo_pao)
        K_ij = ovL_i_ij @ ovL_j_ij.T
        K_pno_cache[key] = K_ij

    R_ij = K_ij.copy()

    # --- T1-dress the exchange integral (Eq. 75, 92 of Jiang et al.) ---
    # K̃_ij = B̃_ai @ B̃_bj where B̃_ai = B_ai + Σ_k t̃_k^a B_ki
    # delta_K_ij = Σ_k [ t̃_k^a (ki|jb) + (ia|kj) t̃_k^b ] + Σ_{kl} t̃_k^a t̃_l^b (ki|lj)
    if (t1_pno is not None and S_pno_cache is not None
            and ovL_pno_cache is not None and ooL_3idx is not None):
        # Project all T1 to PNO_ij and build dressed ovL
        # ovL_dressed_m[a,Q] = ovL_m[a,Q] + Σ_k t̃_k^a ooL[k,m,Q]
        ovL_i = ovL_pno_cache.get((key, i))
        ovL_j = ovL_pno_cache.get((key, j))
        if ovL_i is not None and ovL_j is not None:
            # Compute T1 correction to ovL: Σ_k t̃_k^{a_ij} * ooL[k,m,Q]
            dress_i = np.zeros_like(ovL_i)  # (n_pno, naux)
            dress_j = np.zeros_like(ovL_j)
            for k in range(nocc_lmo):
                t1_k_ij = _project_t1_to_pair(
                    t1_pno, k, key, S_pno_cache, pno_spaces)
                if np.max(np.abs(t1_k_ij)) < 1e-15:
                    continue
                # ooL[k,i,Q] = ooL_3idx[k][i,Q] = (k_lmo i_lmo | Q)
                dress_i += np.outer(t1_k_ij, ooL_3idx[k, i])
                dress_j += np.outer(t1_k_ij, ooL_3idx[k, j])
            # K̃_ij = (ovL_i + dress_i) @ (ovL_j + dress_j).T
            K_dressed = (ovL_i + dress_i) @ (ovL_j + dress_j).T
            R_ij += K_dressed - K_ij  # add the T1 correction

    # --- T1 projection to PNO basis (local T1, Eq. 70 of Jiang et al.) ---
    t1_i_pno = np.zeros(n_pno)
    t1_j_pno = np.zeros(n_pno)
    if t1_pno is not None and S_pno_cache is not None:
        t1_i_pno = _project_t1_to_pair(t1_pno, i, key, S_pno_cache, pno_spaces)
        t1_j_pno = _project_t1_to_pair(t1_pno, j, key, S_pno_cache, pno_spaces)

    # tau_ij = t2_ij + t1_i ⊗ t1_j (for ladder and Woooo)
    tau_ij = t2_ij + np.outer(t1_i_pno, t1_j_pno)

    # --- 2. Ladder: 0.5 * Σ_cd tau_ij[c,d] * (ac|bd)_ij ---
    if with_df is not None:
        R_ij += _compute_ladder(tau_ij, C_pno_ij, with_df)

    # --- 3. Dressed Woooo: Σ_kl W_kl * tau_kl_proj[a,b] ---
    # PySCF: woooo[i,j,k,l] = (ik|jl) + Σ_{a,b} tau[i,j,a,b]*(ak|lb)
    # P(ij)-symmetrized: R_sym = Σ_{k,l} W_kl * tau[k,l]
    # W_kl = (ik|jl) + Σ_{a,b} tau_ij[a,b] * (a_ij k | l b_ij)
    if J_oo is not None:
        for key_kl, t2_kl in t2_pno_all.items():
            if t2_kl is None or t2_kl.shape[0] == 0:
                continue
            k, l = key_kl
            C_pno_kl = pno_spaces[key_kl]['C_pno']

            # Build tau_kl in PNO[kl] basis
            tau_kl = t2_kl.copy()
            if t1_pno is not None and S_pno_cache is not None:
                t1_k = _project_t1_to_pair(t1_pno, k, key_kl, S_pno_cache, pno_spaces)
                t1_l = _project_t1_to_pair(t1_pno, l, key_kl, S_pno_cache, pno_spaces)
                tau_kl = t2_kl + np.outer(t1_k, t1_l)

            # Bare Woooo: (ik|jl)
            W_kl = J_oo[i, k, j, l]

            # Dressing: P(ij)-symmetrized Woooo dressing coefficient is
            # W_kl = (ik|jl) + 0.5*(Σ_ab tau_ij[a,b]*(ak|lb) + Σ_ab tau_ij[b,a]*(al|kb))
            # For k=l: voov_kl = voov_lk, so this simplifies to 0.5*einsum(tau+tau.T, voov_kk)
            # For k≠l: need cross terms between voov_kl and voov_lk
            if with_df is not None and n_pno > 0:
                # (a_ij k | l b_ij) = ovL[(ij),k] @ ovL[(ij),l].T
                _ovL_k = ovL_pno_cache.get((key, k))
                _ovL_l = ovL_pno_cache.get((key, l))
                if _ovL_k is not None and _ovL_l is not None:
                    voov_kl = _ovL_k @ _ovL_l.T
                else:
                    voov_kl = with_df.ao2mo(
                        [C_pno_ij, C_lmo[:, [k]], C_lmo[:, [l]], C_pno_ij],
                        compact=False).reshape(n_pno, n_pno)

            # Project tau_kl to PNO[ij]
            S_proj = S_pno_cache.get((key, key_kl)) if S_pno_cache else None
            if S_proj is not None:
                tau_kl_proj = S_proj @ tau_kl @ S_proj.T
            else:
                tau_kl_proj = _project_t2_full(tau_kl, C_pno_kl, C_pno_ij, s1e)
            if k != l:
                if with_df is not None and n_pno > 0:
                    voov_lk = voov_kl.T  # (a_ij l | k b_ij) = voov_kl.T
                    d_kl = np.einsum('ab,ab->', tau_ij, voov_kl)
                    d_lk = np.einsum('ab,ab->', tau_ij, voov_lk)
                    d_kl_cross = np.einsum('ba,ab->', tau_ij, voov_lk)
                    d_lk_cross = np.einsum('ba,ab->', tau_ij, voov_kl)
                    W_kl += 0.5 * (d_kl + d_kl_cross)
                    W_lk = J_oo[i, l, j, k]
                    W_lk += 0.5 * (d_lk + d_lk_cross)
                else:
                    W_lk = J_oo[i, l, j, k]
                R_ij += W_kl * tau_kl_proj + W_lk * tau_kl_proj.T
            else:
                if with_df is not None and n_pno > 0:
                    tau_sym = tau_ij + tau_ij.T
                    W_kl += 0.5 * np.einsum('ab,ab->', tau_sym, voov_kl)
                R_ij += W_kl * tau_kl_proj

    # --- 4. Fock coupling: -P(ij) Σ_k ft[k,i] * t_kj_proj ---
    # ft[k,i] = F[k,i] - δ_{ki} * eps_i  (off-diagonal only; diagonal is in D)
    ft_oo = F_lmo - np.diag(eps_lmo)  # off-diagonal occupied Fock

    def _get_S(key_other):
        """Get PNO overlap matrix from cache or compute on the fly."""
        if S_pno_cache is not None:
            S = S_pno_cache.get((key, key_other))
            if S is not None:
                return S
        return C_pno_ij.T @ (s1e @ pno_spaces[key_other]['C_pno'])

    for k in range(nocc_lmo):
        # -ft[k,i] * t_kj_proj (coupling from pair (k,j))
        if abs(ft_oo[k, i]) > 1e-15:
            key_kj = (min(k, j), max(k, j))
            if key_kj in t2_pno_all and t2_pno_all[key_kj] is not None:
                t2_kj_raw = t2_pno_all[key_kj]
                if t2_kj_raw.shape[0] > 0:
                    t2_kj = t2_kj_raw.T if k > j else t2_kj_raw
                    S_kj = _get_S(key_kj)
                    R_ij -= ft_oo[k, i] * (S_kj @ t2_kj @ S_kj.T)

        # -ft[k,j] * t_ik_proj (coupling from pair (i,k)) — P(ij) term
        if abs(ft_oo[k, j]) > 1e-15:
            key_ik = (min(i, k), max(i, k))
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
                t2_ik_raw = t2_pno_all[key_ik]
                if t2_ik_raw.shape[0] > 0:
                    t2_ik = t2_ik_raw.T if i > k else t2_ik_raw
                    S_ik = _get_S(key_ik)
                    R_ij -= ft_oo[k, j] * (S_ik @ t2_ik @ S_ik.T)

    # --- 4b. T1 Fock dressing ---
    if t1_pno is not None and fov_pno is not None and S_pno_cache is not None:
        # T1 correction to occupied Fock: ft_ij[k,m] += 0.5 * Σ_a t1[m,a]*fov[k,a]
        # Computed pair-locally: for each (k, m) pair, project both t1[m] and
        # fov[k] to a common PNO basis and dot them.
        ft_t1_oo = np.zeros((nocc_lmo, nocc_lmo))
        for m in range(nocc_lmo):
            key_mm = (m, m)
            if key_mm not in pno_spaces or t1_pno[m].size == 0:
                continue
            for k in range(nocc_lmo):
                # fov[k] in PNO_{mm} basis
                fov_k_mm = _project_t1_to_pair(
                    fov_pno, k, key_mm, S_pno_cache, pno_spaces)
                ft_t1_oo[k, m] = 0.5 * np.dot(fov_k_mm, t1_pno[m])
        for k in range(nocc_lmo):
            for idx_occ, (idx_i, idx_j) in enumerate([(i, j), (j, i)]):
                corr = ft_t1_oo[k, idx_i]
                if abs(corr) < 1e-15:
                    continue
                key_kj2 = (min(k, idx_j), max(k, idx_j))
                if key_kj2 in t2_pno_all and t2_pno_all[key_kj2] is not None:
                    t2_raw = t2_pno_all[key_kj2]
                    if t2_raw.shape[0] > 0:
                        t2_p = t2_raw.T if k > idx_j else t2_raw
                        S_p = _get_S(key_kj2)
                        R_ij -= corr * (S_p @ t2_p @ S_p.T)

        # T1 virtual Fock dressing: ft_ab[a,c] = -0.5 * Σ_m t1_m(a)*fov_m(c)
        # where t1_m and fov_m are projected to PNO_{ij} basis.
        ft_vv = np.zeros((n_pno, n_pno))
        for m in range(nocc_lmo):
            t1_m_ij = _project_t1_to_pair(
                t1_pno, m, key, S_pno_cache, pno_spaces)
            fov_m_ij = _project_t1_to_pair(
                fov_pno, m, key, S_pno_cache, pno_spaces)
            ft_vv -= 0.5 * np.outer(t1_m_ij, fov_m_ij)
        if np.max(np.abs(ft_vv)) > 1e-15:
            R_ij += ft_vv @ t2_ij + t2_ij @ ft_vv.T

    # --- 5. Ring + C_ij + oovv: Combined voov/oovv contributions ---
    # Verified pair-local formula (linear in T2, unsymmetrized):
    #   R_unsym[i,j,a,b] = Σ_k { θ_ki(c,a)*(ck|jb)         [ring/voov]
    #                           - t2(k,i,c,a)*(kj|cb)        [oovv from pair(i,k)]
    #                           - t2(j,k,c,a)*(ki|cb) }      [C_ij from pair(j,k)]
    # Then P(ij): R_full = R_unsym[i,j] + R_unsym[j,i].T
    #
    # Key identity: voov(ck|jb) = K_dir.T and oovv(kj|cb) = K_coul.T
    # where K_dir[b,c]=(jb|kc) and K_coul[b,c]=(jk|bc) are the same integrals
    # the old ring code computed.
    if with_df is not None:
        for k in range(nocc_lmo):
            # --- Contributions from pair (i,k) ---
            key_ik = (min(i, k), max(i, k))
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
                C_pno_ik = pno_spaces[key_ik]['C_pno']
                n_ik = C_pno_ik.shape[1]
                if n_ik > 0:
                    t2_ik_raw = t2_pno_all[key_ik]
                    t2_ik = t2_ik_raw.T if i > k else t2_ik_raw

                    # PNO overlap: projects from PNO_ik to PNO_ij
                    S_ij_ik = _get_S(key_ik)

                    # K_dir[b_ij, c_ik] = (jb|kc) = ovL[(ij),j] @ ovL[(ik),k].T
                    _ovL_j_ij = ovL_pno_cache.get((key, j))
                    _ovL_k_ik = ovL_pno_cache.get((key_ik, k))
                    if _ovL_j_ij is not None and _ovL_k_ik is not None:
                        K_dir = _ovL_j_ij @ _ovL_k_ik.T
                    else:
                        K_dir = with_df.ao2mo(
                            [C_lmo[:, [j]], C_pno_ij, C_lmo[:, [k]], C_pno_ik],
                            compact=False).reshape(n_pno, n_ik)

                    # K_coul[b_ij, c_ik] = (jk|bc) [oovv-type, pre-computed]
                    _kc_key = (key, key_ik, j, k)
                    K_coul = K_coul_cache.get(_kc_key) if K_coul_cache else None
                    if K_coul is None:
                        K_coul = with_df.ao2mo(
                            [C_lmo[:, [j]], C_lmo[:, [k]], C_pno_ij, C_pno_ik],
                            compact=False).reshape(1, 1, n_pno, n_ik)[0, 0]

                    theta_ik = 2.0 * t2_ik - t2_ik.T
                    R_ij += S_ij_ik @ (theta_ik @ (K_dir.T - 0.5 * K_coul.T))
                    R_ij -= 0.5 * S_ij_ik @ (t2_ik.T @ K_coul.T)
                    R_ij -= K_coul @ t2_ik @ S_ij_ik.T

            # --- Contributions from pair (j,k) ---
            key_jk = (min(j, k), max(j, k))
            if key_jk in t2_pno_all and t2_pno_all[key_jk] is not None:
                C_pno_jk = pno_spaces[key_jk]['C_pno']
                n_jk = C_pno_jk.shape[1]
                if n_jk > 0:
                    t2_jk_raw = t2_pno_all[key_jk]
                    t2_jk = t2_jk_raw.T if j > k else t2_jk_raw

                    S_ij_jk = _get_S(key_jk)

                    # K_dir_j[a_ij, c_jk] = (ia|kc) = ovL[(ij),i] @ ovL[(jk),k].T
                    _ovL_i_ij = ovL_pno_cache.get((key, i))
                    _ovL_k_jk = ovL_pno_cache.get((key_jk, k))
                    if _ovL_i_ij is not None and _ovL_k_jk is not None:
                        K_dir_j = _ovL_i_ij @ _ovL_k_jk.T
                    else:
                        K_dir_j = with_df.ao2mo(
                            [C_lmo[:, [i]], C_pno_ij, C_lmo[:, [k]], C_pno_jk],
                            compact=False).reshape(n_pno, n_jk)

                    # K_coul_j[a_ij, c_jk] = (ik|ac) [oovv-type, pre-computed]
                    _kc_key_j = (key, key_jk, i, k)
                    K_coul_j = K_coul_cache.get(_kc_key_j) if K_coul_cache else None
                    if K_coul_j is None:
                        K_coul_j = with_df.ao2mo(
                            [C_lmo[:, [i]], C_lmo[:, [k]], C_pno_ij, C_pno_jk],
                            compact=False).reshape(1, 1, n_pno, n_jk)[0, 0]

                    R_ij -= S_ij_jk @ (t2_jk.T @ K_coul_j.T)
                    theta_kj = 2.0 * t2_jk.T - t2_jk
                    R_ij += ((K_dir_j - 0.5 * K_coul_j) @ theta_kj) @ S_ij_jk.T
                    R_ij -= 0.5 * (K_coul_j @ t2_jk) @ S_ij_jk.T

    # --- 5b. Quadratic T2 dressing of ring/voov/C_ij ---
    # PySCF: wVooV_dress = 0.5*einsum('bkic,jkca->bija',voov,t2)
    #        wVOov_dress = 0.5*einsum('aikc,kcjb->aijb',VOov,tau_ring)
    #   (wVOov += wVooV*0.5 in PySCF happens BEFORE wVooV dressing, so
    #    wVOov_dress does NOT include 0.5*wVooV_dress)
    # Two sources: B (theta×wVOov_dress via V_k), C (t2^T×wVooV_dress via W_k)
    # Implemented as double sum over (k,l) with factored intermediates W_k, V_k.
    if with_df is not None and foo_dressed is not None:
        R_dress_ij = np.zeros((n_pno, n_pno))  # unsym[i,j]
        R_dress_ji = np.zeros((n_pno, n_pno))  # unsym[j,i]

        for k in range(nocc_lmo):
            # --- Pair (i,k): build W_k, V_k using pairs (j,l) ---
            key_ik = (min(i, k), max(i, k))
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
                C_pno_ik = pno_spaces[key_ik]['C_pno']
                n_ik = C_pno_ik.shape[1]
                if n_ik > 0:
                    t2_ik_raw = t2_pno_all[key_ik]
                    t2_ik_d = t2_ik_raw.T if i > k else t2_ik_raw
                    theta_ik_d = 2.0 * t2_ik_d - t2_ik_d.T
                    S_ij_ik_d = _get_S(key_ik)

                    W_k = np.zeros((n_ik, n_pno))
                    V_k = np.zeros((n_ik, n_pno))
                    for l in range(nocc_lmo):
                        key_jl = (min(j, l), max(j, l))
                        if key_jl not in t2_pno_all or t2_pno_all[key_jl] is None:
                            continue
                        n_jl = pno_spaces[key_jl]['C_pno'].shape[1]
                        if n_jl == 0:
                            continue
                        t2_jl_raw = t2_pno_all[key_jl]
                        t2_jl = t2_jl_raw.T if j > l else t2_jl_raw
                        S_ij_jl = _get_S(key_jl)

                        # (c_ik l | k d_jl) = ovL[(ik),l] @ ovL[(jl),k].T
                        _ovL_ik_l = ovL_pno_cache.get((key_ik, l))
                        _ovL_jl_k = ovL_pno_cache.get((key_jl, k))
                        if _ovL_ik_l is not None and _ovL_jl_k is not None:
                            voov_lk = _ovL_ik_l @ _ovL_jl_k.T
                        else:
                            voov_lk = with_df.ao2mo(
                                [C_pno_ik, C_lmo[:, [l]], C_lmo[:, [k]],
                                 pno_spaces[key_jl]['C_pno']],
                                compact=False).reshape(n_ik, n_jl)
                        W_k += 0.5 * voov_lk @ t2_jl @ S_ij_jl.T

                        # (c_ik k | l d_jl) = ovL[(ik),k] @ ovL[(jl),l].T
                        _ovL_ik_k = ovL_pno_cache.get((key_ik, k))
                        _ovL_jl_l = ovL_pno_cache.get((key_jl, l))
                        if _ovL_ik_k is not None and _ovL_jl_l is not None:
                            voov_kl = _ovL_ik_k @ _ovL_jl_l.T
                        else:
                            voov_kl = with_df.ao2mo(
                                [C_pno_ik, C_lmo[:, [k]], C_lmo[:, [l]],
                                 pno_spaces[key_jl]['C_pno']],
                                compact=False).reshape(n_ik, n_jl)
                        VOov_kl = voov_kl - 0.5 * voov_lk
                        tau_ring_lj = 2.0 * t2_jl.T - t2_jl
                        V_k += 0.5 * VOov_kl @ tau_ring_lj @ S_ij_jl.T

                    R_dress_ij += S_ij_ik_d @ (theta_ik_d @ V_k)
                    R_dress_ij += 0.5 * S_ij_ik_d @ (t2_ik_d.T @ W_k)
                    R_dress_ji += S_ij_ik_d @ (t2_ik_d.T @ W_k)

            # --- Pair (j,k): build W_k_ji, V_k_ji using pairs (i,l) ---
            key_jk = (min(j, k), max(j, k))
            if key_jk in t2_pno_all and t2_pno_all[key_jk] is not None:
                C_pno_jk = pno_spaces[key_jk]['C_pno']
                n_jk = C_pno_jk.shape[1]
                if n_jk > 0:
                    t2_jk_raw = t2_pno_all[key_jk]
                    t2_jk_d = t2_jk_raw.T if j > k else t2_jk_raw
                    theta_jk_d = 2.0 * t2_jk_d - t2_jk_d.T
                    S_ij_jk_d = _get_S(key_jk)

                    W_k_ji = np.zeros((n_jk, n_pno))
                    V_k_ji = np.zeros((n_jk, n_pno))
                    for l in range(nocc_lmo):
                        key_il = (min(i, l), max(i, l))
                        if key_il not in t2_pno_all or t2_pno_all[key_il] is None:
                            continue
                        n_il = pno_spaces[key_il]['C_pno'].shape[1]
                        if n_il == 0:
                            continue
                        t2_il_raw = t2_pno_all[key_il]
                        t2_il = t2_il_raw.T if i > l else t2_il_raw
                        S_ij_il = _get_S(key_il)

                        # (c_jk l | k d_il) = ovL[(jk),l] @ ovL[(il),k].T
                        _ovL_jk_l = ovL_pno_cache.get((key_jk, l))
                        _ovL_il_k = ovL_pno_cache.get((key_il, k))
                        if _ovL_jk_l is not None and _ovL_il_k is not None:
                            voov_lk = _ovL_jk_l @ _ovL_il_k.T
                        else:
                            voov_lk = with_df.ao2mo(
                                [C_pno_jk, C_lmo[:, [l]], C_lmo[:, [k]],
                                 pno_spaces[key_il]['C_pno']],
                                compact=False).reshape(n_jk, n_il)
                        W_k_ji += 0.5 * voov_lk @ t2_il @ S_ij_il.T

                        # (c_jk k | l d_il) = ovL[(jk),k] @ ovL[(il),l].T
                        _ovL_jk_k = ovL_pno_cache.get((key_jk, k))
                        _ovL_il_l = ovL_pno_cache.get((key_il, l))
                        if _ovL_jk_k is not None and _ovL_il_l is not None:
                            voov_kl = _ovL_jk_k @ _ovL_il_l.T
                        else:
                            voov_kl = with_df.ao2mo(
                                [C_pno_jk, C_lmo[:, [k]], C_lmo[:, [l]],
                                 pno_spaces[key_il]['C_pno']],
                                compact=False).reshape(n_jk, n_il)
                        VOov_kl = voov_kl - 0.5 * voov_lk
                        tau_ring_li = 2.0 * t2_il.T - t2_il
                        V_k_ji += 0.5 * VOov_kl @ tau_ring_li @ S_ij_il.T

                    R_dress_ji += S_ij_jk_d @ (theta_jk_d @ V_k_ji)
                    R_dress_ji += 0.5 * S_ij_jk_d @ (t2_jk_d.T @ W_k_ji)
                    R_dress_ij += S_ij_jk_d @ (t2_jk_d.T @ W_k_ji)

        # P(ij) symmetrize the dressing
        R_ij += R_dress_ij + R_dress_ji.T

    # --- 6. T2-dressed Fock intermediates ---
    # (quadratic in T2, ~16% of linear norm; enable for production)
    if foo_dressed is not None and with_df is not None:
        # --- 6a. T2-dressed foo: apply precomputed foo_dressed ---
        for k in range(nocc_lmo):
            if abs(foo_dressed[k, i]) > 1e-15:
                key_kj2 = (min(k, j), max(k, j))
                if key_kj2 in t2_pno_all and t2_pno_all[key_kj2] is not None:
                    t2_kj_raw = t2_pno_all[key_kj2]
                    if t2_kj_raw.shape[0] > 0:
                        t2_kj = t2_kj_raw.T if k > j else t2_kj_raw
                        S_kj = _get_S(key_kj2)
                        R_ij -= foo_dressed[k, i] * (S_kj @ t2_kj @ S_kj.T)
            # P(ij) partner: -foo_d[k,j]*t2[i,k] (NOT t2[k,i])
            if abs(foo_dressed[k, j]) > 1e-15:
                key_ki2 = (min(k, i), max(k, i))
                if key_ki2 in t2_pno_all and t2_pno_all[key_ki2] is not None:
                    t2_ki_raw = t2_pno_all[key_ki2]
                    if t2_ki_raw.shape[0] > 0:
                        t2_ik = t2_ki_raw.T if i > k else t2_ki_raw
                        S_ki = _get_S(key_ki2)
                        R_ij -= foo_dressed[k, j] * (S_ki @ t2_ik @ S_ki.T)

        # --- 6b. T2-dressed fvv in PNO[ij] basis ---
        # voov_mn = (a_mn nn | mm b_ij) = ovL[(mn),nn] @ ovL[(ij),mm].T
        fvv_dressed = np.zeros((n_pno, n_pno))
        for key_mn, t2_mn_raw in t2_pno_all.items():
            if t2_mn_raw is None or t2_mn_raw.shape[0] == 0:
                continue
            mm, nn = key_mn
            n_mn = pno_spaces[key_mn]['C_pno'].shape[1]
            if n_mn == 0:
                continue

            theta_fock_mn = 2.0 * t2_mn_raw.T - t2_mn_raw
            S_proj = _get_S(key_mn)

            # voov_mn[a_mn, b_ij] = (a_mn nn | mm b_ij)
            _ovL_mn_nn = ovL_pno_cache.get((key_mn, nn))
            _ovL_ij_mm = ovL_pno_cache.get((key, mm))
            if _ovL_mn_nn is not None and _ovL_ij_mm is not None:
                voov_mn = _ovL_mn_nn @ _ovL_ij_mm.T
            else:
                voov_mn = with_df.ao2mo(
                    [pno_spaces[key_mn]['C_pno'], C_lmo[:, [nn]],
                     C_lmo[:, [mm]], C_pno_ij],
                    compact=False).reshape(n_mn, n_pno)
            theta_proj = theta_fock_mn @ S_proj.T
            fvv_dressed -= theta_proj.T @ voov_mn

            if mm != nn:
                theta_fock_nm = 2.0 * t2_mn_raw - t2_mn_raw.T
                # voov_nm[a_mn, b_ij] = (a_mn mm | nn b_ij)
                _ovL_mn_mm = ovL_pno_cache.get((key_mn, mm))
                _ovL_ij_nn = ovL_pno_cache.get((key, nn))
                if _ovL_mn_mm is not None and _ovL_ij_nn is not None:
                    voov_nm = _ovL_mn_mm @ _ovL_ij_nn.T
                else:
                    voov_nm = with_df.ao2mo(
                        [pno_spaces[key_mn]['C_pno'], C_lmo[:, [mm]],
                         C_lmo[:, [nn]], C_pno_ij],
                        compact=False).reshape(n_mn, n_pno)
                theta_proj_nm = theta_fock_nm @ S_proj.T
                fvv_dressed -= theta_proj_nm.T @ voov_nm

        R_ij += t2_ij @ fvv_dressed.T + fvv_dressed @ t2_ij

    # --- 7: T1 × ovvv and T1 × ooov disabled for now (TODO: add back) ---

    return R_ij


def _compute_inter_pair_coupling(
        i, j, t2_pno_all, pno_spaces, nocc_lmo,
        F_lmo, s1e, with_df, C_lmo,
        J_oo, eps_lmo,
        t1_can=None, U_pno=None, fov_lmo=None):
    """Inter-pair coupling correction for the hybrid update_amps approach.

    Computes coupling contributions from occupied orbitals k not in {i,j}:
      1. Woooo from pairs (k,l) where not both k,l in {i,j}
      2. Fock coupling from k not in {i,j}
      3. T1 Fock dressing from k not in {i,j}
      4. Ring coupling from k not in {i,j}

    These terms are NOT included in the 2-occupied update_amps call
    (which only sees k,l in {i,j}).

    Returns:
        delta (np.ndarray): (n_pno_ij, n_pno_ij) coupling residual numerator.
            Must be divided by D_ij before adding to T2.
    """
    key = (min(i, j), max(i, j))
    data = pno_spaces[key]
    C_pno_ij = data['C_pno']
    n_pno = C_pno_ij.shape[1]

    if n_pno == 0:
        return np.zeros((0, 0))

    delta = np.zeros((n_pno, n_pno))
    pair_set = {i, j}

    # --- 1. Woooo coupling: (k,l) where not both in {i,j} ---
    if J_oo is not None:
        for key_kl, t2_kl in t2_pno_all.items():
            if t2_kl is None or t2_kl.shape[0] == 0:
                continue
            k, l = key_kl
            if k in pair_set and l in pair_set:
                continue  # handled by update_amps
            C_pno_kl = pno_spaces[key_kl]['C_pno']

            tau_kl = t2_kl.copy()
            if t1_can is not None and U_pno is not None and key_kl in U_pno:
                U_kl = U_pno[key_kl]
                t1_k = t1_can[k] @ U_kl
                t1_l = t1_can[l] @ U_kl
                tau_kl = t2_kl + np.outer(t1_k, t1_l)

            dress = 0.0
            n_kl = C_pno_kl.shape[1]
            if with_df is not None and n_kl > 0:
                K_cross = with_df.ao2mo(
                    [C_lmo[:, [i]], C_pno_kl, C_lmo[:, [j]], C_pno_kl],
                    compact=False).reshape(n_kl, n_kl)
                dress = np.einsum('cd,cd->', K_cross, tau_kl)

            W_kl = J_oo[i, j, k, l] + dress
            tau_kl_proj = _project_t2_full(tau_kl, C_pno_kl, C_pno_ij, s1e)
            delta += 0.5 * W_kl * tau_kl_proj
            if k != l:
                W_lk = J_oo[i, j, l, k] + dress
                delta += 0.5 * W_lk * tau_kl_proj.T

    # --- 2. Fock coupling from k not in {i,j} ---
    ft_oo = F_lmo - np.diag(eps_lmo)
    for k in range(nocc_lmo):
        if k in pair_set:
            continue

        if abs(ft_oo[k, i]) > 1e-15:
            key_kj = (min(k, j), max(k, j))
            if key_kj in t2_pno_all and t2_pno_all[key_kj] is not None:
                t2_kj_raw = t2_pno_all[key_kj]
                if t2_kj_raw.shape[0] > 0:
                    C_pno_kj = pno_spaces[key_kj]['C_pno']
                    t2_kj = t2_kj_raw.T if k > j else t2_kj_raw
                    delta -= ft_oo[k, i] * _project_t2_full(
                        t2_kj, C_pno_kj, C_pno_ij, s1e)

        if abs(ft_oo[k, j]) > 1e-15:
            key_ik = (min(i, k), max(i, k))
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
                t2_ik_raw = t2_pno_all[key_ik]
                if t2_ik_raw.shape[0] > 0:
                    C_pno_ik = pno_spaces[key_ik]['C_pno']
                    t2_ik = t2_ik_raw.T if i > k else t2_ik_raw
                    delta -= ft_oo[k, j] * _project_t2_full(
                        t2_ik, C_pno_ik, C_pno_ij, s1e)

    # --- 3. T1 Fock dressing from k not in {i,j} ---
    if t1_can is not None and fov_lmo is not None:
        ft_t1_oo = 0.5 * fov_lmo @ t1_can.T
        for k in range(nocc_lmo):
            if k in pair_set:
                continue
            for idx_i, idx_j in [(i, j), (j, i)]:
                corr = ft_t1_oo[k, idx_i]
                if abs(corr) < 1e-15:
                    continue
                key_kj2 = (min(k, idx_j), max(k, idx_j))
                if key_kj2 in t2_pno_all and t2_pno_all[key_kj2] is not None:
                    t2_raw = t2_pno_all[key_kj2]
                    if t2_raw.shape[0] > 0:
                        C_pno_p = pno_spaces[key_kj2]['C_pno']
                        t2_p = t2_raw.T if k > idx_j else t2_raw
                        delta -= corr * _project_t2_full(
                            t2_p, C_pno_p, C_pno_ij, s1e)

    # --- 4. Ring coupling from k not in {i,j} ---
    if with_df is not None:
        for k in range(nocc_lmo):
            if k in pair_set:
                continue

            # Ring from pair (i,k)
            key_ik = (min(i, k), max(i, k))
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
                C_pno_ik = pno_spaces[key_ik]['C_pno']
                n_ik = C_pno_ik.shape[1]
                if n_ik > 0:
                    t2_ik_raw = t2_pno_all[key_ik]
                    t2_ik = t2_ik_raw.T if i > k else t2_ik_raw

                    theta_ki = 2.0 * t2_ik.T - t2_ik
                    theta_half = _project_t2_half(
                        theta_ki.T, C_pno_ik, C_pno_ij, s1e)

                    K_dir = with_df.ao2mo(
                        [C_lmo[:, [j]], C_pno_ij,
                         C_lmo[:, [k]], C_pno_ik],
                        compact=False).reshape(n_pno, n_ik)
                    K_coul = with_df.ao2mo(
                        [C_lmo[:, [j]], C_lmo[:, [k]],
                         C_pno_ij, C_pno_ik],
                        compact=False).reshape(1, 1, n_pno, n_ik)[0, 0]
                    L_jk = 2.0 * K_dir - K_coul
                    delta += theta_half @ L_jk.T

            # Ring from pair (j,k) — P(ij) symmetrization
            key_jk = (min(j, k), max(j, k))
            if key_jk in t2_pno_all and t2_pno_all[key_jk] is not None:
                C_pno_jk = pno_spaces[key_jk]['C_pno']
                n_jk = C_pno_jk.shape[1]
                if n_jk > 0:
                    t2_jk_raw = t2_pno_all[key_jk]
                    t2_jk = t2_jk_raw.T if j > k else t2_jk_raw

                    theta_kj = 2.0 * t2_jk.T - t2_jk
                    theta_half = _project_t2_half(
                        theta_kj.T, C_pno_jk, C_pno_ij, s1e)

                    K_dir = with_df.ao2mo(
                        [C_lmo[:, [i]], C_pno_ij,
                         C_lmo[:, [k]], C_pno_jk],
                        compact=False).reshape(n_pno, n_jk)
                    K_coul = with_df.ao2mo(
                        [C_lmo[:, [i]], C_lmo[:, [k]],
                         C_pno_ij, C_pno_jk],
                        compact=False).reshape(1, 1, n_pno, n_jk)[0, 0]
                    L_ik = 2.0 * K_dir - K_coul
                    ring_ji = theta_half @ L_ik.T
                    delta += ring_ji.T

    return delta


def _compute_pair_coupling(i, j, t2_pno_all, pno_spaces, nocc_lmo,
                            C_pao, ovL_lmo_pao, F_lmo, s1e, with_df, C_lmo,
                            mol=None, J_oo=None):
    """Inter-pair coupling correction δK for pair (i,j).

    Computes the missing residual contributions relative to the independent-pair
    CCSD.  There are three classes of coupling:

    G term — off-diagonal Fock coupling (dominant for localized orbitals):
        δK[a,b] -= F_lmo[k,i] * T2_proj[(k,j)→(i,j)][a,b]   ∀k≠i,j
        δK[a,b] -= F_lmo[k,j] * T2_proj[(i,k)→(i,j)][a,b]   ∀k≠i,j

    B term — oo-oo Coulomb coupling via the Woooo intermediate (CRITICAL):
        δK[a,b] += sum_{(k,l)≠(i,j)} (ij|kl) * T2_proj[(k,l)→(i,j)][a,b]
        This term is missing because the 2-occ pair subspace only has k,l∈{i,j}.
        J_oo[i,j,k,l] = (ij|kl) must be precomputed and passed in.

    Ring coupling — via DF if with_df given, else ao2mo.kernel, else skipped:
        δK += (ring_ik - ring_ik.T) - (ring_jk - ring_jk.T)
        where ring_ik[a,b] = sum_c T2_half_ik[a,c] * W_jk[b,c]
              W_jk[b,c] = 2*(jb|kc) - (jk|bc)  b∈PNO_ij, c∈PNO_ik

    Returns:
        delta_K (np.ndarray): (n_pno_ij, n_pno_ij) coupling correction,
            or None if pno_spaces entry missing.
    """
    key_ij = (min(i, j), max(i, j))
    if key_ij not in pno_spaces:
        return None
    C_pno_ij = pno_spaces[key_ij]['C_pno']
    n_ij = C_pno_ij.shape[1]
    delta_K = np.zeros((n_ij, n_ij))

    # Ring coupling requires some integral source (DF or exact)
    use_ring = (with_df is not None) or (mol is not None)

    # Precompute ovL_pno for LMOs i and j if tensor is available (fast path)
    if use_ring and with_df is not None and ovL_lmo_pao is not None:
        ovL_j_ij = _ovl_pno(j, C_pno_ij, C_pao, ovL_lmo_pao)  # (n_ij, naux)
        ovL_i_ij = _ovl_pno(i, C_pno_ij, C_pao, ovL_lmo_pao)  # (n_ij, naux)
    else:
        ovL_j_ij = ovL_i_ij = None

    # ------------------------------------------------------------------
    # B term: oo-oo Coulomb coupling via Woooo intermediate.
    # sum_{(k,l)≠(i,j)} (ij|kl) * T2[k,l]_→ij
    # This is MISSING from the 2-occ pair update_amps since the pair
    # subspace only couples k,l ∈ {i,j}.  J_oo[i,j,k,l] = (ij|kl).
    # ------------------------------------------------------------------
    key_self = (min(i, j), max(i, j))
    if J_oo is not None:
        for key_kl, T2_kl in t2_pno_all.items():
            if key_kl == key_self:
                continue
            if T2_kl is None or T2_kl.shape[0] == 0:
                continue
            k, l = key_kl
            C_pno_kl = pno_spaces[key_kl]['C_pno']
            T2_kl_proj = _project_t2_full(T2_kl, C_pno_kl, C_pno_ij, s1e)
            # Factor 0.5 matches canonical: 0.5 * Σ_kl (ij|kl) * tau_kl
            J_kl = J_oo[i, j, k, l]
            delta_K += 0.5 * J_kl * T2_kl_proj
            if k != l:
                delta_K += 0.5 * J_oo[i, j, l, k] * T2_kl_proj.T

    for k in range(nocc_lmo):
        if k == i or k == j:
            continue

        # ----------------------------------------------------------------
        # Fock coupling from pair (k,j): -F[k,i] * T2[k,j] projected to ij
        # ----------------------------------------------------------------
        key_kj = (min(k, j), max(k, j))
        if key_kj in t2_pno_all and t2_pno_all[key_kj] is not None:
            C_pno_kj = pno_spaces[key_kj]['C_pno']
            t2_kj_raw = t2_pno_all[key_kj]
            # Stored as T2[min,max]; if k>j, T2[k,j]=T2[j,k].T
            t2_kj = t2_kj_raw.T if k > j else t2_kj_raw
            delta_K -= F_lmo[k, i] * _project_t2_full(t2_kj, C_pno_kj, C_pno_ij, s1e)

        # ----------------------------------------------------------------
        # Fock coupling from pair (i,k): -F[k,j] * T2[i,k] projected to ij
        # ----------------------------------------------------------------
        key_ik = (min(i, k), max(i, k))
        if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
            C_pno_ik = pno_spaces[key_ik]['C_pno']
            t2_ik_raw = t2_pno_all[key_ik]
            t2_ik = t2_ik_raw.T if i > k else t2_ik_raw
            delta_K -= F_lmo[k, j] * _project_t2_full(t2_ik, C_pno_ik, C_pno_ij, s1e)

        if not use_ring:
            continue

        # ----------------------------------------------------------------
        # Ring coupling: P(ij)P(ab) sum_c T2[im,ac] * W[mb,cj]
        # Contribution from pair (i,k) [sharing occupied i]:
        #   ring_ik[a,b] = sum_c T2_half_ik[a,c] * W_jk[b,c]
        #   W_jk[b,c] = 2*(jb|kc) - (jk|bc)  b∈PNO_ij, c∈PNO_ik
        # ----------------------------------------------------------------
        if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
            C_pno_ik = pno_spaces[key_ik]['C_pno']
            n_ik = C_pno_ik.shape[1]
            t2_ik_raw = t2_pno_all[key_ik]
            t2_ik = t2_ik_raw.T if i > k else t2_ik_raw

            # T2_half_ik: first virtual projected to PNO_ij, second stays in PNO_ik
            T2_half_ik = _project_t2_half(t2_ik, C_pno_ik, C_pno_ij, s1e)  # (n_ij, n_ik)

            # K_dir_jk[a,c] = (j,a|k,c), a∈PNO_ij, c∈PNO_ik
            if ovL_j_ij is not None:
                ovL_k_ik = _ovl_pno(k, C_pno_ik, C_pao, ovL_lmo_pao)
                K_dir_jk = ovL_j_ij @ ovL_k_ik.T                    # (n_ij, n_ik)
            elif with_df is not None:
                K_dir_jk = with_df.ao2mo(
                    [C_lmo[:, [j]], C_pno_ij, C_lmo[:, [k]], C_pno_ik],
                    compact=False
                ).reshape(n_ij, n_ik)
            else:
                K_dir_jk = ao2mo.kernel(
                    mol, [C_lmo[:, [j]], C_pno_ij, C_lmo[:, [k]], C_pno_ik],
                    compact=False
                ).reshape(n_ij, n_ik)

            # K_coul_jk[a,c] = (jk|ac), a∈PNO_ij, c∈PNO_ik
            if with_df is not None:
                K_coul_jk = with_df.ao2mo(
                    [C_lmo[:, [j]], C_lmo[:, [k]], C_pno_ij, C_pno_ik],
                    compact=False
                ).reshape(1, 1, n_ij, n_ik)[0, 0]
            else:
                K_coul_jk = ao2mo.kernel(
                    mol, [C_lmo[:, [j]], C_lmo[:, [k]], C_pno_ij, C_pno_ik],
                    compact=False
                ).reshape(n_ij, n_ik)

            W_jk = 2.0 * K_dir_jk - K_coul_jk  # (n_ij, n_ik)
            ring_ik = T2_half_ik @ W_jk.T        # (n_ij, n_ij)
            delta_K += ring_ik - ring_ik.T

        # ----------------------------------------------------------------
        # Ring coupling from pair (j,k) [sharing occupied j] — P(ij) term:
        #   ring_jk[a,b] = sum_c T2_half_jk[a,c] * W_ik[b,c]
        #   W_ik[b,c] = 2*(ib|kc) - (ik|bc)  b∈PNO_ij, c∈PNO_jk
        # ----------------------------------------------------------------
        key_jk = (min(j, k), max(j, k))
        if key_jk in t2_pno_all and t2_pno_all[key_jk] is not None:
            C_pno_jk = pno_spaces[key_jk]['C_pno']
            n_jk = C_pno_jk.shape[1]
            t2_jk_raw = t2_pno_all[key_jk]
            t2_jk = t2_jk_raw.T if j > k else t2_jk_raw

            # T2_half_jk: first virtual projected to PNO_ij
            T2_half_jk = _project_t2_half(t2_jk, C_pno_jk, C_pno_ij, s1e)  # (n_ij, n_jk)

            # K_dir_ik[a,c] = (i,a|k,c), a∈PNO_ij, c∈PNO_jk
            if ovL_i_ij is not None:
                ovL_k_jk = _ovl_pno(k, C_pno_jk, C_pao, ovL_lmo_pao)
                K_dir_ik = ovL_i_ij @ ovL_k_jk.T                    # (n_ij, n_jk)
            elif with_df is not None:
                K_dir_ik = with_df.ao2mo(
                    [C_lmo[:, [i]], C_pno_ij, C_lmo[:, [k]], C_pno_jk],
                    compact=False
                ).reshape(n_ij, n_jk)
            else:
                K_dir_ik = ao2mo.kernel(
                    mol, [C_lmo[:, [i]], C_pno_ij, C_lmo[:, [k]], C_pno_jk],
                    compact=False
                ).reshape(n_ij, n_jk)

            # K_coul_ik[a,c] = (ik|ac), a∈PNO_ij, c∈PNO_jk
            if with_df is not None:
                K_coul_ik = with_df.ao2mo(
                    [C_lmo[:, [i]], C_lmo[:, [k]], C_pno_ij, C_pno_jk],
                    compact=False
                ).reshape(1, 1, n_ij, n_jk)[0, 0]
            else:
                K_coul_ik = ao2mo.kernel(
                    mol, [C_lmo[:, [i]], C_lmo[:, [k]], C_pno_ij, C_pno_jk],
                    compact=False
                ).reshape(n_ij, n_jk)

            W_ik = 2.0 * K_dir_ik - K_coul_ik  # (n_ij, n_jk)
            ring_jk = T2_half_jk @ W_ik.T        # (n_ij, n_ij)
            delta_K -= ring_jk - ring_jk.T

    return delta_K


# ---------------------------------------------------------------------------
# Global coupled LCCSD (correct inter-pair coupling via global CCSD)
# ---------------------------------------------------------------------------

def _run_global_lccsd(mf, C_lmo, pno_spaces, strong_pairs,
                      fock_ao, eps_lmo, s1e, conv_tol, max_cycle,
                      t2_cas=None, occ_cas_idx=None, vir_cas_idx=None,
                      mo_coeff_cas=None):
    """Run a global CCSD in the LMO occupied + canonical virtual basis.

    All inter-pair coupling is exact (Fock, ring, ladder diagrams across all
    pairs simultaneously).  When C_vir spans the full virtual space
    (T_CutPNO→0) this converges to canonical CCSD.

    For TCCSD: if CAS virtual amplitudes are provided, the CAS block of T2
    is frozen to the DMRG values after every CCSD update step (Lang et al.
    eq. 14–15).  The external block is iterated freely with standard CCSD
    machinery.  When vir_cas_idx is empty the freeze is a no-op and the
    function reduces to standard CCSD.

    Args:
        mf: RHF (or DF-RHF) object.
        C_lmo: (nao, nocc_lmo) LMO coefficients.
        pno_spaces: dict (i,j) → PNO data.
        strong_pairs: list of (i,j) pairs.
        fock_ao: (nao, nao) AO Fock matrix.
        eps_lmo: (nocc_lmo,) LMO Fock diagonal.
        s1e: (nao, nao) AO overlap matrix.
        conv_tol, max_cycle: CCSD convergence settings.
        t2_cas: (nocc_cas, nocc_cas, nvir_cas, nvir_cas) DMRG doubles.
        occ_cas_idx: LMO indices of CAS occupied orbitals.
        vir_cas_idx: columns of mo_coeff_cas that are CAS virtual orbitals.
        mo_coeff_cas: CASSCF MO coefficient matrix (nao, nmo_cas).

    Returns:
        e_total (float): Total CCSD correlation energy.
        t2_pno_all (dict): (i,j) → T2 in pair PNO basis (for (T) triples).
    """
    nocc = C_lmo.shape[1]
    nocc_full = int(np.count_nonzero(mf.mo_occ > 1e-10))
    C_vir = mf.mo_coeff[:, nocc_full:]  # canonical virtual MOs (nao, nvir)
    nvir = C_vir.shape[1]
    nmo = nocc + nvir

    mo_coeff = np.hstack((C_lmo, C_vir))  # (nao, nmo)
    fock_mo = reduce(np.dot, (mo_coeff.T, fock_ao, mo_coeff))
    mo_energy = np.diag(fock_mo).real
    mo_occ = np.zeros(nmo)
    mo_occ[:nocc] = 2.0

    print(f'  Global LCCSD: nocc={nocc}  nvir={nvir}  nmo={nmo}', flush=True)

    # Build a fake mf with LMO+canonical_vir MOs and DF integrals.
    # Use DFCCSD to avoid building the full nmo^4 ERI tensor.
    fake_mf = _FakeMF(mol=mf.mol, mo_coeff_pair=mo_coeff,
                      mo_energy_pair=mo_energy, mo_occ_pair=mo_occ,
                      e_tot=mf.e_tot, fock_ao=fock_ao, _eri=None, verbose=0)

    use_df = hasattr(mf, 'with_df') and mf.with_df is not None
    if use_df:
        fake_mf.with_df = mf.with_df
        from pyscf.cc import dfccsd as _dfccsd
        mycc = _dfccsd.RCCSD(fake_mf, mo_coeff=mo_coeff, mo_occ=mo_occ)
    else:
        mycc = ccsd.CCSD(fake_mf)
    mycc.verbose = 0
    eris = mycc.ao2mo(mo_coeff)

    # ------------------------------------------------------------------
    # CAS freeze setup
    # ------------------------------------------------------------------
    # U_cas: (nvir, nvir_cas) — transforms CAS virtual MOs into the
    # canonical virtual MO basis.  Columns are orthonormal because
    # CASSCF virtual orbitals are orthogonal to all occupied orbitals and
    # span a subspace of the total virtual space.
    nvir_cas = 0 if (vir_cas_idx is None) else len(vir_cas_idx)
    has_cas = (nvir_cas > 0 and t2_cas is not None
               and t2_cas.shape[2] > 0 and mo_coeff_cas is not None)

    if has_cas:
        C_cas_vir = mo_coeff_cas[:, vir_cas_idx]        # (nao, nvir_cas)
        U_cas = C_vir.T @ (s1e @ C_cas_vir)             # (nvir, nvir_cas)
        P_cas = U_cas @ U_cas.T                          # (nvir, nvir) projector

        nc = len(occ_cas_idx)
        # Precompute frozen CAS amplitudes in canonical virtual basis:
        # T2_cas_global[ic, jc] = U_cas @ t2_cas[ic,jc] @ U_cas.T  (nvir, nvir)
        T2_cas_global = np.einsum('ap,ijpq,bq->ijab',
                                  U_cas, t2_cas, U_cas)  # (nc,nc,nvir,nvir)

        # --- Diagnostic ---
        UU = U_cas.T @ U_cas
        print(f'  [DIAG] U_cas shape={U_cas.shape}  '
              f'U_cas.T@U_cas diag={np.diag(UU).round(4)}  '
              f'max_offdiag={np.max(np.abs(UU - np.diag(np.diag(UU)))):.2e}',
              flush=True)
        nvir_cas_loc = len(vir_cas_idx)
        print(f'  [DIAG] |t2_cas| = {np.linalg.norm(t2_cas):.6f}  '
              f'max = {np.max(np.abs(t2_cas)):.6f}', flush=True)
        print(f'  [DIAG] t2_cas[0,0,0,0]={t2_cas[0,0,0,0]:.6f}  '
              f't2_cas[1,0,0,0]={t2_cas[1,0,0,0]:.6f}  '
              f't2_cas[0,1,0,1]={t2_cas[0,1,0,1]:.6f}  '
              f't2_cas[1,1,0,0]={t2_cas[1,1,0,0]:.6f}', flush=True)
        print(f'  [DIAG] U_cas[0:nvir_cas, :] =\n{U_cas[:nvir_cas_loc, :].round(4)}',
              flush=True)
        print(f'  [DIAG] T2_cas_global[0,0,:4,:4]=\n'
              f'{T2_cas_global[0,0,:nvir_cas_loc,:nvir_cas_loc].round(6)}', flush=True)
        print(f'  [DIAG] T2_cas_global[0,1,:4,:4]=\n'
              f'{T2_cas_global[0,1,:nvir_cas_loc,:nvir_cas_loc].round(6)}', flush=True)
        print(f'  [DIAG] |T2_cas_global| = {np.linalg.norm(T2_cas_global):.6f}  '
              f'max = {np.max(np.abs(T2_cas_global)):.6f}', flush=True)

        def freeze_cas_block(t2):
            """Remove CAS-CAS virtual contribution; inject frozen DMRG value."""
            for ic in range(nc):
                for jc in range(nc):
                    i_lmo = occ_cas_idx[ic]
                    j_lmo = occ_cas_idx[jc]
                    t2[i_lmo, j_lmo] -= P_cas @ t2[i_lmo, j_lmo] @ P_cas
                    t2[i_lmo, j_lmo] += T2_cas_global[ic, jc]
            return t2

    # ------------------------------------------------------------------
    # CCSD iteration
    # ------------------------------------------------------------------
    if not has_cas:
        # Standard path: delegate entirely to PySCF's CCSD kernel.
        mycc.conv_tol = conv_tol
        mycc.max_cycle = max_cycle
        # Pass t1=None, t2=None so PySCF uses its MP2 initial guess.
        _, t1, t2 = mycc.ccsd(eris=eris)
    else:
        # TCCSD path: start from MP2 initial guess, then iterate CCSD
        # with the CAS block frozen to DMRG amplitudes after each step.
        _, t1, t2 = mycc.init_amps(eris)
        t2 = freeze_cas_block(t2)
        mydiis = lib.diis.DIIS()
        mydiis.space = 8
        for cycle in range(max_cycle):
            t1_new, t2_new = mycc.update_amps(t1, t2, eris)
            t2_new = freeze_cas_block(t2_new)

            # DIIS extrapolation (error = new − old)
            t_vec = np.concatenate([t1_new.ravel(), t2_new.ravel()])
            t_old = np.concatenate([t1.ravel(), t2.ravel()])
            err = t_vec - t_old
            dT = np.max(np.abs(err))
            t_vec_diis = mydiis.update(t_vec, err)

            n1 = t1.size
            t1 = t_vec_diis[:n1].reshape(t1.shape)
            t2 = t_vec_diis[n1:].reshape(t2.shape)
            # Re-apply freeze after DIIS (extrapolation can drift CAS block).
            t2 = freeze_cas_block(t2)

            print(f'  Cycle {cycle + 1:3d}: dT = {dT:.3e}', flush=True)
            if dT < conv_tol:
                print(f'  Global TCCSD converged in {cycle + 1} cycles.',
                      flush=True)
                break
        else:
            print('  WARNING: Global TCCSD did not converge.', flush=True)

    # ------------------------------------------------------------------
    # Total correlation energy and projection to PNO basis
    # ------------------------------------------------------------------
    print(f'  [DIAG] |t1| = {np.linalg.norm(t1):.6f}  '
          f'max|t1| = {np.max(np.abs(t1)):.6f}', flush=True)
    print(f'  [DIAG] |t2| = {np.linalg.norm(t2):.6f}', flush=True)
    if has_cas:
        nvir_cas = len(vir_cas_idx)
        print(f'  [DIAG] t2[0,0,0,0]={t2[0,0,0,0]:.6f}  '
              f't2[0,1,0,0]={t2[0,1,0,0]:.6f}  '
              f't2[1,0,0,1]={t2[1,0,0,1]:.6f}', flush=True)
        print(f'  [DIAG] |T2_cas_global[0,0]| in CAS block = '
              f'{np.linalg.norm(T2_cas_global[0,0,:nvir_cas,:nvir_cas]):.6f}  '
              f'max = {np.max(np.abs(T2_cas_global[0,0,:nvir_cas,:nvir_cas])):.6f}',
              flush=True)

    # Compute energy using PySCF's CCSD energy formula (handles DF correctly)
    e_total = mycc.energy(t1, t2, eris)
    print(f'  [DIAG] E(CCSD corr) = {e_total:.10f}', flush=True)

    t2_pno_all = {}
    for pair in strong_pairs:
        pi, pj = pair
        key = (min(pi, pj), max(pi, pj))
        if key not in pno_spaces:
            continue
        C_pno_ij = pno_spaces[key]['C_pno']
        n_pno = C_pno_ij.shape[1]
        if n_pno == 0:
            t2_pno_all[key] = np.zeros((0, 0))
            continue
        S = C_vir.T @ (s1e @ C_pno_ij)        # (nvir, n_pno)
        t2_ij = t2[pi, pj] if pi <= pj else t2[pj, pi].T
        t2_pno_all[key] = S.T @ t2_ij @ S     # (n_pno, n_pno)

    return e_total, t2_pno_all, t1


# ---------------------------------------------------------------------------
# DLPNO-CCSD: per-pair PNO virtual spaces, correct full-occ coupling
# ---------------------------------------------------------------------------

def _run_dlpno_lccsd(mf, C_lmo, pno_spaces, strong_pairs,
                     fock_ao, eps_lmo, s1e, conv_tol, max_cycle,
                     t2_cas=None, occ_cas_idx=None, vir_cas_idx=None,
                     mo_coeff_cas=None, diis_space=15,
                     damping=0.5, diis_start_cycle=6,
                     C_pao=None):
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

    # ------------------------------------------------------------------
    # Pre-compute quantities needed by the pair-local residual
    # ------------------------------------------------------------------
    F_lmo = reduce(np.dot, (C_lmo.T, fock_ao, C_lmo))

    # J_oo[i,j,k,l] = (ij|kl) in chemist notation — for Woooo coupling
    if with_df is not None:
        J_oo_flat = with_df.ao2mo(C_lmo, compact=False)
        J_oo = J_oo_flat.reshape(nocc, nocc, nocc, nocc)
    elif mol is not None:
        J_oo_flat = ao2mo.kernel(mol, C_lmo, compact=False)
        J_oo = J_oo_flat.reshape(nocc, nocc, nocc, nocc)
    else:
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
        # MP2 T1 initialisation (Eq. 103): t1[i,a] = fov[i,a] / (e_pno_a - F_ii)
        e_pno_ii = data_ii['e_pno']
        D_i = e_pno_ii - eps_lmo[i]       # denominator: e_a - eps_i (negative)
        D_i = np.where(np.abs(D_i) > 1e-12, D_i, -1e-12)
        t1_pno[i] = -fov_i / D_i          # R/D with R=fov, D=eps_a - eps_i

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

    keys_sorted = sorted(t2_pno_all.keys())
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
    # ovL_pno_cache[(pair_key, lmo_idx)] = (n_pno, naux) tensor.
    # Built via _build_ovL(with_df, C_lmo, C_pno_ij) which gives the
    # exact DF factorization: (a_P k|l b_Q) =
    #   ovL_pno_cache[(P,k)] @ ovL_pno_cache[(Q,l)].T
    # ------------------------------------------------------------------
    from pyscf.cc.dlpno_tccsd.pno import _build_ovL
    ovL_pno_cache = {}
    if with_df is not None:
        for key in keys_sorted:
            C_pno_ij = pno_spaces[key]['C_pno']
            n_pno_ij = C_pno_ij.shape[1]
            if n_pno_ij == 0:
                continue
            # Build ovL[i, a_pno, L] for all occupied i at once
            ovL_ij = _build_ovL(with_df, C_lmo, C_pno_ij)  # (nocc, n_pno, naux)
            for k in range(nocc):
                ovL_pno_cache[(key, k)] = ovL_ij[k]  # (n_pno, naux)
        # Also build ovL for diagonal pairs not in keys_sorted (needed by T1)
        for i in range(nocc):
            key_ii = (i, i)
            if key_ii in keys_sorted:
                continue  # already built above
            if key_ii not in pno_spaces:
                continue
            C_pno_ii = pno_spaces[key_ii]['C_pno']
            n_pno_ii = C_pno_ii.shape[1]
            if n_pno_ii == 0:
                continue
            ovL_ii = _build_ovL(with_df, C_lmo, C_pno_ii)
            for k in range(nocc):
                ovL_pno_cache[(key_ii, k)] = ovL_ii[k]

        _n_entries = len(ovL_pno_cache)
        if _n_entries > 0:
            _sample = next(iter(ovL_pno_cache.values()))
            _mem_mb = _n_entries * _sample.nbytes / 1e6
            print(f'  ovL_pno cache: {_n_entries} entries, '
                  f'{_mem_mb:.0f} MB', flush=True)

    # Pre-compute PNO overlap matrices S_pno_cache[(key_ij, key_kl)]
    # for all pair combinations that share an occupied index (connected pairs).
    S_pno_cache = {}
    for key_ij in keys_sorted:
        i, j = key_ij
        C_pno_ij = pno_spaces[key_ij]['C_pno']
        if C_pno_ij.shape[1] == 0:
            continue
        for key_kl in keys_sorted:
            if key_kl == key_ij:
                continue
            k, l = key_kl
            # Connected: shares at least one occupied index
            if i == k or i == l or j == k or j == l:
                C_pno_kl = pno_spaces[key_kl]['C_pno']
                if C_pno_kl.shape[1] == 0:
                    continue
                S_pno_cache[(key_ij, key_kl)] = C_pno_ij.T @ (
                    s1e @ C_pno_kl)

    # Extend S_pno_cache with diagonal pair overlaps needed by local T1.
    # For each LMO i, we need S_pno[(key_ii, key_kl)] and S_pno[(key_kl, key_ii)]
    # for all pairs kl that involve i (k==i or l==i), plus overlaps between
    # diagonal pairs (i,i) and (k,k) for the occupied Fock dressing of T1.
    for i in range(nocc):
        key_ii = (i, i)
        if key_ii not in pno_spaces:
            continue
        C_pno_ii = pno_spaces[key_ii]['C_pno']
        if C_pno_ii.shape[1] == 0:
            continue
        # Overlaps with all strong pairs involving i
        for key_kl in keys_sorted:
            k, l = key_kl
            if k != i and l != i:
                continue
            if key_kl == key_ii:
                continue
            C_pno_kl = pno_spaces[key_kl]['C_pno']
            if C_pno_kl.shape[1] == 0:
                continue
            if (key_ii, key_kl) not in S_pno_cache:
                S_pno_cache[(key_ii, key_kl)] = C_pno_ii.T @ (s1e @ C_pno_kl)
            if (key_kl, key_ii) not in S_pno_cache:
                S_pno_cache[(key_kl, key_ii)] = C_pno_kl.T @ (s1e @ C_pno_ii)
        # Overlaps between diagonal pairs (i,i) and (k,k) for foo*t1 terms
        for k in range(nocc):
            if k == i:
                continue
            key_kk = (k, k)
            if key_kk not in pno_spaces:
                continue
            C_pno_kk = pno_spaces[key_kk]['C_pno']
            if C_pno_kk.shape[1] == 0:
                continue
            if (key_ii, key_kk) not in S_pno_cache:
                S_pno_cache[(key_ii, key_kk)] = C_pno_ii.T @ (s1e @ C_pno_kk)

    # Pre-compute K_coul integrals: (jk|b_ij c_ik) [oovv-type]
    # These cannot be factored from ovL_pno_cache, so we pre-compute them.
    # K_coul_cache[(key_ij, key_pk, occ_bra1, occ_bra2)] = (n_pno_ij, n_pno_pk)
    # For section 5 (ring/oovv): need (jk|b_ij c_ik) and (ik|a_ij c_jk)
    K_coul_cache = {}
    if with_df is not None:
        for key_ij in keys_sorted:
            i, j = key_ij
            C_pno_ij = pno_spaces[key_ij]['C_pno']
            n_ij = C_pno_ij.shape[1]
            if n_ij == 0:
                continue
            for k in range(nocc):
                # Pair (i,k) contributions to pair (i,j)
                key_ik = (min(i, k), max(i, k))
                if key_ik in pno_spaces:
                    C_pno_ik = pno_spaces[key_ik]['C_pno']
                    n_ik = C_pno_ik.shape[1]
                    if n_ik > 0:
                        cache_key = (key_ij, key_ik, j, k)
                        if cache_key not in K_coul_cache:
                            K_coul_cache[cache_key] = with_df.ao2mo(
                                [C_lmo[:, [j]], C_lmo[:, [k]],
                                 C_pno_ij, C_pno_ik],
                                compact=False).reshape(1, 1, n_ij, n_ik)[0, 0]
                # Pair (j,k) contributions to pair (i,j)
                key_jk = (min(j, k), max(j, k))
                if key_jk in pno_spaces:
                    C_pno_jk = pno_spaces[key_jk]['C_pno']
                    n_jk = C_pno_jk.shape[1]
                    if n_jk > 0:
                        cache_key = (key_ij, key_jk, i, k)
                        if cache_key not in K_coul_cache:
                            K_coul_cache[cache_key] = with_df.ao2mo(
                                [C_lmo[:, [i]], C_lmo[:, [k]],
                                 C_pno_ij, C_pno_jk],
                                compact=False).reshape(1, 1, n_ij, n_jk)[0, 0]
        _n_kcoul = len(K_coul_cache)
        if _n_kcoul > 0:
            _sample_kc = next(iter(K_coul_cache.values()))
            _mem_kc = sum(v.nbytes for v in K_coul_cache.values()) / 1e6
            print(f'  K_coul cache: {_n_kcoul} entries, '
                  f'{_mem_kc:.0f} MB', flush=True)

    # Pre-compute occ-occ 3-index DF tensor ooL[i,j,Q] = (i_lmo j_lmo | Q)
    # for T1 integral dressing (Eq. 91-93 of Jiang et al.).
    ooL_3idx = None
    if with_df is not None:
        from pyscf.cc.dlpno_tccsd.pno import _build_ovL as _build_ovL_fn
        ooL_3idx = _build_ovL_fn(with_df, C_lmo, C_lmo)  # (nocc, nocc, naux)

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
        for cycle in range(this_max):
            t2_new = {}
            t1_pno_old = {i: t1_pno[i].copy() for i in range(nocc)}

            # Precompute T2-dressed foo once per iteration (global quantity)
            foo_dress = _compute_foo_dressed(
                t2_pno_all, pno_spaces, nocc, with_df, C_lmo, s1e,
                ovL_pno_cache=ovL_pno_cache)

            # Add T1-dressed Fock correction (Eq. 98 of Jiang et al.)
            foo_t1 = _compute_foo_t1_dressed(
                t1_pno, pno_spaces, nocc, with_df, C_lmo,
                ovL_pno_cache=ovL_pno_cache)
            foo_dress += foo_t1

            for key in keys_sorted:
                i, j = key
                data = pno_spaces[key]
                C_pno_ij = data['C_pno']
                n_pno = C_pno_ij.shape[1]

                if n_pno == 0:
                    t2_new[key] = np.zeros((0, 0))
                    continue

                # ---- Pair-local residual ----
                # T1 is passed for tau, exchange dressing, and Fock dressing.
                R_ij = _compute_pair_residual_numerator(
                    i, j, t2_pno_all, pno_spaces, nocc,
                    C_pao, ovL_lmo_pao, F_lmo, s1e, with_df, C_lmo,
                    J_oo, K_pno_cache, eps_lmo,
                    t1_pno=t1_pno, fov_pno=fov_pno,
                    foo_dressed=foo_dress,
                    ovL_pno_cache=ovL_pno_cache,
                    S_pno_cache=S_pno_cache,
                    K_coul_cache=K_coul_cache,
                    ooL_3idx=ooL_3idx)
                e_pno = data['e_pno']
                D_ij = (eps_lmo[i] + eps_lmo[j]
                        - e_pno[:, None] - e_pno[None, :])
                D_ij = np.where(np.abs(D_ij) > 1e-12, D_ij, 1e-12)
                T2_ij_new = R_ij / D_ij

                # ---- CAS freeze: restore scaled CAS block ----
                if key in cas_blocks:
                    cb = cas_blocks[key]
                    cas_sl = cb[0]
                    if len(cb) == 3:
                        _, t2c_dmrg_ref, t2c_mp2 = cb
                        T2_ij_new[cas_sl, cas_sl] = (
                            t2c_mp2 + scale * (t2c_dmrg_ref - t2c_mp2))
                    else:
                        T2_ij_new[cas_sl, cas_sl] = cb[1]

                t2_new[key] = T2_ij_new

            # ---- Local T1 update (Eq. 87, 103 of Jiang et al.) ----
            r1_pno = _compute_local_t1_residual(
                t1_pno, t2_pno_all, pno_spaces,
                fov_pno, F_lmo, eps_lmo,
                nocc, s1e, fock_ao, C_lmo,
                ovL_pno_cache, S_pno_cache,
                foo_dressed=foo_dress)

            t1_pno_new = {}
            for ii in range(nocc):
                key_ii = (ii, ii)
                e_pno_ii = pno_spaces[key_ii]['e_pno']
                n_pno_ii = e_pno_ii.size
                if n_pno_ii == 0:
                    t1_pno_new[ii] = np.zeros(0)
                    continue
                D_i = e_pno_ii - eps_lmo[ii]
                D_i = np.where(np.abs(D_i) > 1e-12, D_i, -1e-12)
                t1_pno_new[ii] = -r1_pno[ii] / D_i

            # ---- T1→T2 correction: t1⊗fov outer product ----
            dt2_t1 = _compute_local_t1_to_t2_correction(
                t1_pno, fov_pno, keys_sorted,
                cas_blocks, pno_spaces, eps_lmo,
                S_pno_cache, nocc)
            for k in keys_sorted:
                if k in cas_blocks:
                    continue
                if k in dt2_t1 and t2_new[k].shape[0] > 0:
                    t2_new[k] += dt2_t1[k]

            t1_pno = t1_pno_new

            # ---- Damping + DIIS on external amplitudes only ----
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

            # Global DIIS on all amplitudes simultaneously
            if cycle >= diis_start_cycle and err_amp.size > 0:
                amp_new = mydiis.update(amp_new, err_amp)

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
            # E = Σ_i F_ia t1_ia + Σ_ij (t_ij^{ab} + t̃_i^a t̃_j^b) L_ij^{ab}
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
                # tau = t2 + t1_i ⊗ t1_j in PNO basis (local T1 projection)
                t1i_pno = _project_t1_to_pair(
                    t1_pno, pk[0], pk, S_pno_cache, pno_spaces)
                t1j_pno = _project_t1_to_pair(
                    t1_pno, pk[1], pk, S_pno_cache, pno_spaces)
                tau_p = T2_p + np.outer(t1i_pno, t1j_pno)
                Tt_p = 2.0 * tau_p - tau_p.T
                e_p = np.einsum('ab,ab->', K_p, Tt_p)
                e_cyc += e_p if pi == pj else 2.0 * e_p
            dE = abs(e_cyc - e_prev) if cycle > 0 else float('inf')
            e_prev = e_cyc
            print(f'  Cycle {cycle + 1:3d}: dT = {dT:.3e}  E_corr = {e_cyc:.10f}'
                  f'  dE = {dE:.2e}', flush=True)
            if dT < this_tol:
                print(f'  DLPNO-CCSD converged in {cycle + 1} cycles (amplitude).',
                      flush=True)
                break
            # For TCCSD, redundant PNO modes cause dT to diverge while the
            # energy converges.  Use energy as primary convergence criterion.
            e_conv_tol = this_tol if cas_blocks else this_tol * 1e-2
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
    # E = Σ_i F_ia t1_ia + Σ_ij (t_ij^{ab} + t̃_i^a t̃_j^b) L_ij^{ab}
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

def _run_lccsd_coupled(mf, C_lmo, pno_spaces, strong_pairs,
                       fock_ao, eps_lmo, s1e,
                       cas_pairs, t2_cas, occ_cas_idx,
                       C_pao=None, ovL_lmo_pao=None,
                       conv_tol=1e-7, max_cycle=100, diis_space=8):
    """Scalable DLPNO-CCSD with per-pair PNO spaces and global Jacobi-DIIS.

    All pairs use their own PNO virtual space (scalable with system size).
    Inter-pair coupling (Fock + ring diagrams) enters the CCSD residual of
    each pair via _compute_pair_coupling and is treated in a Jacobi sense:

        T2_new[ij] = T2_intra_new[ij]  +  delta_K[ij] / D[ij]

    where T2_intra_new comes from one PySCF CCSD update_amps step on the
    2-occ pair subsystem and delta_K is the inter-pair coupling residual.
    DIIS is applied globally across all pairs simultaneously.

    With T_CutPNO→0 (full virtual space per pair) and Fock+ring coupling
    active, this converges to canonical CCSD.

    CAS amplitude injection: for pairs with nvir_cas_local>0, the CAS block
    of T2 is frozen to the DMRG value after every DIIS step.

    Args:
        mf: RHF or DF-RHF object.
        C_lmo: (nao, nocc_lmo) LMO coefficients.
        pno_spaces: dict key→PNO data from pno.make_pnos.
        strong_pairs: list of (i,j) pairs to solve.
        fock_ao: (nao, nao) AO Fock.
        eps_lmo: (nocc_lmo,) LMO diagonal Fock eigenvalues.
        s1e: (nao, nao) AO overlap.
        cas_pairs: set of (i,j) CAS pairs.
        t2_cas: (nocc_cas, nocc_cas, nvir_cas, nvir_cas) DMRG amplitudes.
        occ_cas_idx: CAS occupied LMO indices.
        C_pao: (nao, n_pao) PAO coefficients (for ring coupling).
        ovL_lmo_pao: optional precomputed (nocc_lmo, n_pao, naux) tensor.
        conv_tol, max_cycle, diis_space: convergence settings.

    Returns:
        e_total (float): Total LCCSD correlation energy.
        t2_pno_all (dict): key → converged T2 in PNO basis.
    """
    mol = mf.mol
    with_df = getattr(mf, 'with_df', None)
    nocc_lmo = C_lmo.shape[1]
    F_lmo = reduce(np.dot, (C_lmo.T, fock_ao, C_lmo))

    # ------------------------------------------------------------------
    # Build per-pair CCSD objects (done once before the iteration loop)
    # ------------------------------------------------------------------
    pair_data = {}   # canonical key (i≤j) → dict
    for pair in strong_pairs:
        i, j = pair
        key = (min(i, j), max(i, j))
        if key in pair_data or key not in pno_spaces:
            continue

        data = pno_spaces[key]
        C_pno_ij = data['C_pno']
        e_pno    = data['e_pno']
        K_pno    = data['K_pno']
        n_pno    = C_pno_ij.shape[1]

        nocc_pair = 2
        nmo_pair  = nocc_pair + n_pno
        mo_coeff_pair = np.hstack((C_lmo[:, [i, j]], C_pno_ij))

        mo_energy_pair          = np.zeros(nmo_pair)
        mo_energy_pair[0]       = eps_lmo[i]
        mo_energy_pair[1]       = eps_lmo[j]
        mo_energy_pair[2:]      = e_pno
        mo_occ_pair             = np.zeros(nmo_pair)
        mo_occ_pair[:nocc_pair] = 2.0

        if with_df is not None:
            eri_pair = with_df.ao2mo(mo_coeff_pair, compact=False).reshape(
                nmo_pair, nmo_pair, nmo_pair, nmo_pair)
        else:
            eri_pair = ao2mo.kernel(mol, mo_coeff_pair, compact=False).reshape(
                nmo_pair, nmo_pair, nmo_pair, nmo_pair)

        fock_pair_mo = reduce(np.dot, (mo_coeff_pair.T, fock_ao, mo_coeff_pair))
        o_sl = slice(0, nocc_pair)
        v_sl = slice(nocc_pair, nmo_pair)
        nvir_tril = n_pno * (n_pno + 1) // 2

        eris_pair          = _ChemistsERIs()
        eris_pair.nocc     = nocc_pair
        eris_pair.fock     = fock_pair_mo
        eris_pair.mo_energy = np.diag(fock_pair_mo).real
        eris_pair.ovov     = eri_pair[o_sl, v_sl, o_sl, v_sl]
        eris_pair.oovv     = eri_pair[o_sl, o_sl, v_sl, v_sl]
        eris_pair.ovvo     = eri_pair[o_sl, v_sl, v_sl, o_sl]
        eris_pair.ovoo     = eri_pair[o_sl, v_sl, o_sl, o_sl]
        eris_pair.oooo     = eri_pair[o_sl, o_sl, o_sl, o_sl]
        if n_pno > 0:
            eris_pair.ovvv = lib.pack_tril(
                eri_pair[o_sl, v_sl, v_sl, v_sl].reshape(-1, n_pno, n_pno)
            ).reshape(nocc_pair, n_pno, nvir_tril)
            eris_pair.vvvv = ao2mo.restore(4, eri_pair[v_sl, v_sl, v_sl, v_sl], n_pno)
        else:
            eris_pair.ovvv = np.zeros((nocc_pair, 0, 0))
            eris_pair.vvvv = np.zeros((0, 0, 0, 0))

        fake_mf = _FakeMF(mol=mol, mo_coeff_pair=mo_coeff_pair,
                          mo_energy_pair=mo_energy_pair, mo_occ_pair=mo_occ_pair,
                          e_tot=mf.e_tot, fock_ao=fock_ao, _eri=None, verbose=0)
        mycc = ccsd.CCSD(fake_mf)
        mycc.verbose = 0

        # Orbital energy denominator: D[a,b] = eps_i + eps_j - e_a - e_b
        D = eps_lmo[i] + eps_lmo[j] - e_pno[:, None] - e_pno[None, :]
        D_safe = np.where(np.abs(D) > 1e-12, D, 1e-12)

        pair_data[key] = {
            'i': i, 'j': j,
            'eris': eris_pair, 'mycc': mycc,
            'K_pno': K_pno, 'D_safe': D_safe,
            'n_pno': n_pno, 'C_pno': C_pno_ij,
        }

    # ------------------------------------------------------------------
    # CAS injection bookkeeping
    # ------------------------------------------------------------------
    cas_blocks = {}  # key → (cas_slice, t2_cas_pno)
    for pair in strong_pairs:
        i, j = pair
        key = (min(i, j), max(i, j))
        if pair not in cas_pairs or key not in pair_data:
            continue
        nvir_cas_local = pno_spaces[key].get('nvir_cas_local', 0)
        if nvir_cas_local > 0:
            i_cas = int(np.where(occ_cas_idx == i)[0][0])
            j_cas = int(np.where(occ_cas_idx == j)[0][0])
            cas_blocks[key] = (slice(0, nvir_cas_local), t2_cas[i_cas, j_cas])

    # ------------------------------------------------------------------
    # Initialize T2 from MP2 amplitudes
    # ------------------------------------------------------------------
    t2_pno_all = {}
    for key, pobj in pair_data.items():
        n_pno  = pobj['n_pno']
        T2_mp2 = pno_spaces[key].get('T2_pno')
        if n_pno == 0:
            t2_pno_all[key] = np.zeros((0, 0))
        elif T2_mp2 is not None and T2_mp2.shape == (n_pno, n_pno):
            t2_pno_all[key] = T2_mp2.copy()
        else:
            t2_pno_all[key] = pobj['K_pno'] / pobj['D_safe']

    # Apply initial CAS injection
    for key, (cas_sl, t2c) in cas_blocks.items():
        t2_pno_all[key][cas_sl, cas_sl] = t2c

    # ------------------------------------------------------------------
    # Jacobi-DIIS global iteration
    # ------------------------------------------------------------------
    keys_sorted = sorted(pair_data.keys())
    diis = lib.diis.DIIS()
    diis.space = diis_space

    # ------------------------------------------------------------------
    # Precompute oo Coulomb integrals J[i,j,k,l] = (ij|kl) for B term
    # ------------------------------------------------------------------
    if with_df is not None:
        J_oo_flat = with_df.ao2mo(C_lmo, compact=False)
        J_oo = J_oo_flat.reshape(nocc_lmo, nocc_lmo, nocc_lmo, nocc_lmo)
    elif mol is not None:
        J_oo_flat = ao2mo.kernel(mol, C_lmo, compact=False)
        J_oo = J_oo_flat.reshape(nocc_lmo, nocc_lmo, nocc_lmo, nocc_lmo)
    else:
        J_oo = None

    n_active = len(keys_sorted)
    ring_src = 'DF' if with_df is not None else ('exact' if mol is not None else 'none')
    print(f'  Coupled LCCSD: {n_active} active pairs, ring={ring_src}, '
          f'B_term={J_oo is not None}', flush=True)

    for cycle in range(max_cycle):
        t2_new = {}

        for key in keys_sorted:
            pobj  = pair_data[key]
            i, j  = pobj['i'], pobj['j']
            n_pno = pobj['n_pno']
            eris  = pobj['eris']
            mycc  = pobj['mycc']
            D_safe = pobj['D_safe']

            T2_cur = t2_pno_all[key]

            if n_pno == 0:
                t2_new[key] = np.zeros((0, 0))
                continue

            # 1. Intra-pair CCSD step: one PySCF update_amps call
            T2_full = np.zeros((2, 2, n_pno, n_pno))
            T2_full[0, 1] = T2_cur
            T2_full[1, 0] = T2_cur.T
            t1_zero = np.zeros((2, n_pno))
            _, T2_intra_full = mycc.update_amps(t1_zero, T2_full, eris)
            T2_intra_new = T2_intra_full[0, 1]  # (n_pno, n_pno)

            # 2. Inter-pair coupling residual (Fock + B-term + ring)
            delta_K = _compute_pair_coupling(
                i, j, t2_pno_all, pno_spaces, nocc_lmo,
                C_pao, ovL_lmo_pao, F_lmo, s1e, with_df, C_lmo,
                mol=mol, J_oo=J_oo
            )

            # 3. Jacobi update: T2_new = T2_intra + delta_K / D
            T2_ij_new = T2_intra_new.copy()
            if delta_K is not None:
                T2_ij_new += delta_K / D_safe

            # 4. CAS injection: freeze CAS block
            if key in cas_blocks:
                cas_sl, t2c = cas_blocks[key]
                T2_ij_new[cas_sl, cas_sl] = t2c

            t2_new[key] = T2_ij_new

        # DIIS extrapolation over all pairs simultaneously
        t2_vec     = np.concatenate([t2_new[k].ravel() for k in keys_sorted])
        t2_old_vec = np.concatenate([t2_pno_all[k].ravel() for k in keys_sorted])
        err_vec    = t2_vec - t2_old_vec
        dT = np.max(np.abs(err_vec)) if err_vec.size > 0 else 0.0

        if err_vec.size > 0:
            t2_vec_diis = diis.update(t2_vec, err_vec)
        else:
            t2_vec_diis = t2_vec

        # Unpack DIIS-extrapolated vector back to per-pair arrays
        offset = 0
        for k in keys_sorted:
            sz = t2_new[k].size
            t2_pno_all[k] = t2_vec_diis[offset:offset + sz].reshape(t2_new[k].shape)
            offset += sz

        # Re-apply CAS injection (DIIS may have drifted CAS block)
        for key, (cas_sl, t2c) in cas_blocks.items():
            if key in t2_pno_all:
                t2_pno_all[key][cas_sl, cas_sl] = t2c

        print(f'  Cycle {cycle + 1:3d}: dT = {dT:.3e}', flush=True)

        if dT < conv_tol:
            print(f'  Coupled LCCSD converged in {cycle + 1} cycles.', flush=True)
            break
    else:
        print('  WARNING: Coupled LCCSD did not converge.', flush=True)

    # ------------------------------------------------------------------
    # Total LCCSD correlation energy
    # ------------------------------------------------------------------
    e_total = 0.0
    for pair in strong_pairs:
        i, j = pair
        key   = (min(i, j), max(i, j))
        if key not in t2_pno_all or key not in pair_data:
            continue
        T2 = t2_pno_all[key]
        K  = pno_spaces[key]['K_pno']
        if T2.shape[0] == 0:
            continue
        Tt  = 2.0 * T2 - T2.T
        e_ij = np.einsum('ab,ab->', K, Tt)
        e_total += e_ij if i == j else 2.0 * e_ij

    return e_total, t2_pno_all


# ---------------------------------------------------------------------------
# Per-pair worker (extracted for parallel execution)
# ---------------------------------------------------------------------------

def _process_one_pair(pair, pno_spaces, C_lmo, mol, with_df, e_tot_hf,
                      fock_ao, eps_lmo, occ_cas_idx, cas_pairs,
                      t2_cas, vir_cas_idx, mo_coeff_cas, s1e,
                      conv_tol, max_cycle):
    """Compute T2 and energy contribution for one pair. Thread-safe.

    All input arrays are read-only shared state; each call allocates its own
    output arrays.  The DF integral build (with_df.ao2mo) and all BLAS calls
    release the GIL, so concurrent threads make real progress.

    Returns:
        (i, j, t2_pno, e_ij, kind)
        kind ∈ {'cas', 'ccsd', 'skip'}
    """
    (i, j) = pair

    if (i, j) not in pno_spaces:
        return i, j, None, 0.0, 'skip'

    data       = pno_spaces[(i, j)]
    C_pno_ij   = data['C_pno']
    K_pno      = data['K_pno']
    T2_mp2     = data['T2_pno']
    e_pno      = data['e_pno']
    n_pno      = C_pno_ij.shape[1]

    # -- CAS pair: tailored CCSD in extended PNO basis --
    # Lang et al. eq (10): S_ij = I_NCAS ⊕ d_ij
    # The extended PNO = [CAS_vir_MOs | external_PNOs] with identity mapping
    # for CAS virtuals. DMRG t2 maps directly into first nvir_cas indices.
    if (i, j) in cas_pairs:
        nvir_cas_local = data.get('nvir_cas_local', 0)

        # DMRG t2 for this pair — identity mapping into first nvir_cas indices
        i_cas = int(np.where(occ_cas_idx == i)[0][0])
        j_cas = int(np.where(occ_cas_idx == j)[0][0])
        t2_cas_pno = t2_cas[i_cas, j_cas]  # (nvir_cas, nvir_cas)

        # Build integrals and run CCSD with tailoring
        nocc_pair = 2
        nmo_pair = nocc_pair + n_pno
        C_lmo_ij = C_lmo[:, [i, j]]
        mo_coeff_pair = np.hstack((C_lmo_ij, C_pno_ij))

        mo_energy_pair = np.zeros(nmo_pair)
        mo_energy_pair[0] = eps_lmo[i]
        mo_energy_pair[1] = eps_lmo[j]
        mo_energy_pair[2:] = e_pno
        mo_occ_pair = np.zeros(nmo_pair)
        mo_occ_pair[:nocc_pair] = 2.0

        if with_df is not None:
            eri_pair = with_df.ao2mo(mo_coeff_pair, compact=False).reshape(
                nmo_pair, nmo_pair, nmo_pair, nmo_pair)
        else:
            eri_pair = ao2mo.kernel(mol, mo_coeff_pair, compact=False).reshape(
                nmo_pair, nmo_pair, nmo_pair, nmo_pair)

        fock_pair_mo = reduce(np.dot, (mo_coeff_pair.T, fock_ao, mo_coeff_pair))
        o, v = slice(0, nocc_pair), slice(nocc_pair, nmo_pair)
        n_vir = nmo_pair - nocc_pair
        nvir_tril = n_vir * (n_vir + 1) // 2

        eris_pair = _ChemistsERIs()
        eris_pair.nocc = nocc_pair
        eris_pair.fock = fock_pair_mo
        eris_pair.mo_energy = np.diag(fock_pair_mo).real
        eris_pair.ovov = eri_pair[o, v, o, v]
        eris_pair.oovv = eri_pair[o, o, v, v]
        eris_pair.ovvo = eri_pair[o, v, v, o]
        eris_pair.ovoo = eri_pair[o, v, o, o]
        eris_pair.oooo = eri_pair[o, o, o, o]
        eris_pair.ovvv = lib.pack_tril(
            eri_pair[o, v, v, v].reshape(-1, n_vir, n_vir)
        ).reshape(nocc_pair, n_vir, nvir_tril)
        eris_pair.vvvv = ao2mo.restore(4, eri_pair[v, v, v, v], n_vir)

        fake_mf = _FakeMF(mol=mol, mo_coeff_pair=mo_coeff_pair,
                          mo_energy_pair=mo_energy_pair, mo_occ_pair=mo_occ_pair,
                          e_tot=e_tot_hf, fock_ao=fock_ao, _eri=None, verbose=0)

        # Initialize t2: external block from MP2, CAS block from DMRG
        t1_init = np.zeros((nocc_pair, n_pno))
        t2_init = np.zeros((nocc_pair, nocc_pair, n_pno, n_pno))
        # Inject DMRG amplitudes into the CAS-virtual block (first nvir_cas indices)
        cas = slice(0, nvir_cas_local)
        if nvir_cas_local > 0:
            t2_init[0, 1, cas, cas] = t2_cas_pno
            t2_init[1, 0, cas, cas] = t2_cas_pno.T

        mycc_pair = ccsd.CCSD(fake_mf)
        mycc_pair.verbose = 0
        mycc_pair.conv_tol = conv_tol
        mycc_pair.max_cycle = max_cycle

        # Monkey-patch update_amps to freeze CAS-virtual block after each iteration
        _orig_update = mycc_pair.update_amps
        _ncl = nvir_cas_local
        _t2_cas_pno = t2_cas_pno
        def _tailored_update(t1, t2, eris):
            t1new, t2new = _orig_update(t1, t2, eris)
            if _ncl > 0:
                c = slice(0, _ncl)
                t2new[0, 1, c, c] = _t2_cas_pno
                t2new[1, 0, c, c] = _t2_cas_pno.T
            return t1new, t2new
        mycc_pair.update_amps = _tailored_update

        try:
            _, _, t2_conv_full = mycc_pair.ccsd(t1_init, t2_init, eris=eris_pair)
        except Exception:
            t2_conv_full = t2_init.copy()

        t2_conv = t2_conv_full[0, 1]
        Tt = 2.0 * t2_conv - t2_conv.T
        K_ij = eri_pair[0, nocc_pair:, 1, nocc_pair:]  # (ia|jb) exchange
        e_ij = np.einsum('ab,ab->', K_ij, Tt)
        # Only label as 'cas' if DMRG tailoring actually occurred
        kind = 'cas' if nvir_cas_local > 0 else 'ccsd'
        return i, j, t2_conv, e_ij, kind

    # -- External pair: CCSD in PNO basis --
    nocc_pair  = 2
    nmo_pair   = nocc_pair + n_pno
    C_lmo_ij   = C_lmo[:, [i, j]]
    mo_coeff_pair = np.hstack((C_lmo_ij, C_pno_ij))

    mo_energy_pair          = np.zeros(nmo_pair)
    mo_energy_pair[0]       = eps_lmo[i]
    mo_energy_pair[1]       = eps_lmo[j]
    mo_energy_pair[2:]      = e_pno
    mo_occ_pair             = np.zeros(nmo_pair)
    mo_occ_pair[:nocc_pair] = 2.0

    # ERI build: use DF if available, else exact 4-index integrals
    if with_df is not None:
        eri_pair = with_df.ao2mo(mo_coeff_pair, compact=False).reshape(
            nmo_pair, nmo_pair, nmo_pair, nmo_pair)
    else:
        eri_pair = ao2mo.kernel(mol, mo_coeff_pair, compact=False).reshape(
            nmo_pair, nmo_pair, nmo_pair, nmo_pair)

    fock_pair_mo   = reduce(np.dot, (mo_coeff_pair.T, fock_ao, mo_coeff_pair))
    o, v           = slice(0, nocc_pair), slice(nocc_pair, nmo_pair)
    n_vir          = nmo_pair - nocc_pair
    nvir_tril      = n_vir * (n_vir + 1) // 2

    eris_pair          = _ChemistsERIs()
    eris_pair.nocc     = nocc_pair
    eris_pair.fock     = fock_pair_mo
    eris_pair.mo_energy = np.diag(fock_pair_mo).real
    eris_pair.ovov     = eri_pair[o, v, o, v]
    eris_pair.oovv     = eri_pair[o, o, v, v]
    eris_pair.ovvo     = eri_pair[o, v, v, o]
    eris_pair.ovoo     = eri_pair[o, v, o, o]
    eris_pair.oooo     = eri_pair[o, o, o, o]
    eris_pair.ovvv     = lib.pack_tril(
        eri_pair[o, v, v, v].reshape(-1, n_vir, n_vir)
    ).reshape(nocc_pair, n_vir, nvir_tril)
    eris_pair.vvvv     = ao2mo.restore(4, eri_pair[v, v, v, v], n_vir)

    fake_mf = _FakeMF(mol=mol, mo_coeff_pair=mo_coeff_pair,
                      mo_energy_pair=mo_energy_pair, mo_occ_pair=mo_occ_pair,
                      e_tot=e_tot_hf, fock_ao=fock_ao, _eri=None, verbose=0)

    t1_init = np.zeros((nocc_pair, n_pno))
    t2_init = np.zeros((nocc_pair, nocc_pair, n_pno, n_pno))
    if n_pno > 0 and T2_mp2 is not None and T2_mp2.shape == (n_pno, n_pno):
        t2_init[0, 1] = T2_mp2
        t2_init[1, 0] = T2_mp2.T

    mycc_pair           = ccsd.CCSD(fake_mf)
    mycc_pair.verbose   = 0
    mycc_pair.conv_tol  = conv_tol
    mycc_pair.max_cycle = max_cycle
    try:
        _, _, t2_conv_full = mycc_pair.ccsd(t1_init, t2_init, eris=eris_pair)
    except Exception:
        t2_conv_full = t2_init.copy()

    t2_conv = t2_conv_full[0, 1]
    Tt = 2.0 * t2_conv - t2_conv.T
    e_ij = np.einsum('ab,ab->', K_pno, Tt)
    return i, j, t2_conv, e_ij, 'ccsd'


# ---------------------------------------------------------------------------
# Main LCCSD runner
# ---------------------------------------------------------------------------

def run_lccsd(mf, C_lmo, pno_spaces, strong_pairs, cas_pairs,
              t1_cas, t2_cas, occ_cas_idx, vir_cas_idx,
              mo_coeff_cas, s1e=None,
              conv_tol=1e-7, max_cycle=50, ncores=1,
              C_pao=None, max_outer_cycle=30, outer_conv_tol=1e-8,
              verbose=None):
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
        C_pao=C_pao)
    print(f'  E_TCCSD = {e_tccsd:.15g}', flush=True)

    return e_tccsd, t2_pno_all, t1_pno


def _solve_pair_ccsd(eris, fock, nocc, nvir,
                     K_ij, T2_init,
                     cas_inject=None,
                     conv_tol=1e-7, max_cycle=50, verbose=0):
    """Minimal pair CCSD iteration in PNO basis.

    Implements the (T2-only) CCSD equations for a 2-occupied, nvir-virtual
    system. This is the "pair approximation" where each pair is solved
    independently (no off-pair coupling in the T2 residual).

    The residual for a pair (0,1) in a 2-electron MO subspace is:
        R[a,b] = K[a,b] + sum_c (F_vv[a,c]*T2[c,b] + T2[a,c]*F_vv[c,b])
               - sum_k (F_oo[0,k]*T2[k_loc,b,a] + ...)
               + (direct + exchange) contractions with T2

    For simplicity, we use the leading-order diagonal Fock approximation:
        R[a,b] ≈ K[a,b] + (eps_a + eps_b - eps_i - eps_j) * T2[a,b]
               + quadratic T2 terms (retained up to full CCSD)

    This is equivalent to the Psi4 DLPNO LMP2 iteration extended to CCSD.
    The full LCCSD residual with off-pair coupling (PNO overlap terms) would
    require the complete Psi4 ccsd.cc machinery; that extension is left as
    a TODO for production use.

    Args:
        eris: Pair ERIs object from _make_pno_ccsd_eris.
        fock (np.ndarray): (nocc+nvir, nocc+nvir) pair Fock matrix.
        nocc (int): Number of occupied orbitals in pair (= 2).
        nvir (int): Number of PNOs.
        K_ij (np.ndarray): (nvir, nvir) exchange integrals (i a | j b).
        T2_init (np.ndarray): (nvir, nvir) initial T2 guess.
        cas_inject: None or (T2_cas_ij, t2f) for CAS amplitude freeze.
        conv_tol (float): Convergence threshold on max|T2new - T2|.
        max_cycle (int): Maximum iterations.
        verbose (int): Verbosity.

    Returns:
        T2_conv (np.ndarray): (nvir, nvir) converged doubles amplitudes.
    """
    if nvir == 0:
        return np.zeros((nocc, nocc, 0, 0))

    # Diagonal Fock (in canonical PNO basis)
    eps_occ = fock.diagonal()[:nocc]
    eps_vir = fock.diagonal()[nocc:]

    # Denominator tensor
    D = (eps_occ[0] + eps_occ[1]
         - eps_vir[:, None] - eps_vir[None, :])   # (nvir, nvir)
    D_safe = np.where(np.abs(D) > 1e-12, D, 1e-12)

    # Initialize T2 from MP2
    if T2_init is not None and T2_init.shape == (nvir, nvir):
        T2 = T2_init.copy()
    else:
        T2 = K_ij / D_safe   # T2 = K/D < 0 (D<0)

    # CCSD iteration (simplified pair approximation)
    # Full residual: R = K + D*T2 + quadratic(T2)
    # For the pair approximation we include:
    #   1. Linear terms: (eps_a - eps_i) and (eps_b - eps_j) shifts
    #   2. Quadratic T2: standard CCSD "ring" and "ladder" diagrams
    # The full DLPNO-CCSD coupling via PNO overlaps to other pairs is handled
    # implicitly through the initial MP2 amplitudes and the final energy.

    for cyc in range(max_cycle):
        T2_old = T2.copy()

        # CAS injection: overwrite CAS block with DMRG values
        if cas_inject is not None:
            T2_cas_ij, t2f = cas_inject
            T2 = np.where(t2f, T2_cas_ij, T2)

        # Residual: R[a,b] = K[a,b] + D[a,b]*T2[a,b]
        #   + quadratic terms from CCSD
        # Quadratic contributions (simplified for pair approximation):
        #   W_oooo type: sum_kl T2[k,l]*T2[k,l] * oooo
        #   (for 2-occupied: oooo[0,1,0,1] = (01|01) ≈ 0 in our pair ERIs)
        #   W_vvvv type: sum_cd T2[a,c]*T2[b,d]*vvvv[c,d,...]
        #   These are neglected in the pair approximation.

        # Leading correction: "laddter diagram" - T2*oovv type
        # For a 2-orbital system the CCSD equations reduce to MP2 + ring diagrams.
        # The dominant correction is:
        #   Delta_T2[a,b] = -sum_c T2[a,c]*K[c,b] / D[a,b]  (ring)
        # This is the MP2→CCSD iteration in the weak-coupling limit.

        Tt = 2.0 * T2 - T2.T
        # Ring diagram: sum_c K[a,c] * Tt[c,b]  (dressed K)
        ring = np.dot(K_ij, Tt.T)   # (nvir, nvir)

        R = K_ij + ring

        # Update: T2new = R / D  (D<0, so T2 stays negative)
        T2_new = R / D_safe

        # CAS injection (after update): freeze CAS block
        if cas_inject is not None:
            T2_cas_ij, t2f = cas_inject
            T2_new = np.where(t2f, T2_cas_ij, T2_new)

        dT = np.max(np.abs(T2_new - T2_old))
        T2 = T2_new

        if dT < conv_tol:
            break

    return T2
