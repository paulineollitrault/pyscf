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

import os
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
    # Phase II port: pyscf/lib/cc/dlpno_foo_dressed.c (PySCF native style).
    # The C kernel takes Qma directly on the pair-domain axis (no scatter-back)
    # and writes contributions only to nlmo_pair positions; caller scatters
    # into the global foo via ci['p_lmos'].
    import ctypes
    from pyscf import lib as _pyscflib
    _libcc = _pyscflib.load_library('libcc')
    _libcc.DLPNOfoo_dressed_pair.restype = None
    _libcc.DLPNOfoo_dressed_pair.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,            # out_q, out_m
        ctypes.c_void_p, ctypes.c_void_p,            # Qma, t2
        ctypes.c_int, ctypes.c_int, ctypes.c_int,    # m_pos, q_pos, need_m
        ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,  # n_local, nlmo_pair, n_pno
    ]

    def _per_pair(key_mq):
        t2_mq_raw = t2_pno_all.get(key_mq)
        if t2_mq_raw is None or t2_mq_raw.shape[0] == 0:
            return key_mq, None
        ci = cc_ints.get(key_mq)
        if ci is None:
            return key_mq, None
        m, q = key_mq
        Qma = np.ascontiguousarray(ci['Qma'])  # (n_local, nlmo_p, npno)
        t2_c = np.ascontiguousarray(t2_mq_raw)  # (n_pno, n_pno)
        n_local, nlmo_p, n_pno = Qma.shape
        m_pos = int(ci['p_lmos_dense'][m])
        q_pos = int(ci['p_lmos_dense'][q])
        if m_pos < 0 or q_pos < 0:
            return key_mq, None  # pair endpoints always in p_lmos by construction
        contrib_q = np.zeros(nlmo_p)
        if m != q:
            contrib_m = np.zeros(nlmo_p)
            _libcc.DLPNOfoo_dressed_pair(
                contrib_q.ctypes.data_as(ctypes.c_void_p),
                contrib_m.ctypes.data_as(ctypes.c_void_p),
                Qma.ctypes.data_as(ctypes.c_void_p),
                t2_c.ctypes.data_as(ctypes.c_void_p),
                m_pos, q_pos, 1,
                n_local, nlmo_p, n_pno,
            )
        else:
            contrib_m = None
            _libcc.DLPNOfoo_dressed_pair(
                contrib_q.ctypes.data_as(ctypes.c_void_p),
                contrib_q.ctypes.data_as(ctypes.c_void_p),  # alias when m == q
                Qma.ctypes.data_as(ctypes.c_void_p),
                t2_c.ctypes.data_as(ctypes.c_void_p),
                m_pos, q_pos, 0,
                n_local, nlmo_p, n_pno,
            )
        return key_mq, (contrib_q, contrib_m, ci['p_lmos'])

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
        contrib_q, contrib_m, p_lmos = payload
        foo[p_lmos, q] += contrib_q
        if contrib_m is not None:
            foo[p_lmos, m] += contrib_m
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
                               t1_cache=None, _pool=None,
                               Fij_bar_precomputed=None,
                               Fij_bar_precomputed_keys=None,
                               cc_ints_flat=None, pair_index=None):
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

    # T_n[(key, k)] semantics — t1_k projected into pair key's PNO basis —
    # is exactly ``t1_cache[key][k]`` (FlatTensorStore zero-fills missing
    # rows with the right shape). Consumers below read t1_cache directly,
    # so the per-(key, k) dict is no longer materialized.

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
    #
    # Optional fast path: t1_fock already accumulated this for the keys in
    # `Fij_bar_precomputed_keys` (strong + diagonals), so we reuse that
    # snapshot and only loop over the remaining (weak) pairs here.
    if Fij_bar_precomputed is not None:
        Fij_bar = Fij_bar_precomputed.copy()
        _skip_keys = (Fij_bar_precomputed_keys
                      if Fij_bar_precomputed_keys is not None else set())
    else:
        Fij_bar = F_lmo.copy()
        _skip_keys = set()
    for key_ij in t2_pno_all:
        if key_ij in _skip_keys:
            continue
        ci_ij = cc_ints.get(key_ij)
        if ci_ij is None:
            continue
        i0, j0 = key_ij  # canonical (i0 <= j0)
        n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_ij == 0:
            continue
        # T_n_ij[m, c] = t1_m projected to canonical pair's PNO basis —
        # already in t1_cache as a (nocc, n_ij) view.
        T_n_ij_mat = t1_cache[key_ij]
        # K_bar_chem/ij/ji are now reduced to (nlmo_p, npno); gather T_n
        # on p_lmos once for all three elementwise products.
        T_n_red = T_n_ij_mat[ci_ij['p_lmos']]
        # Fij_bar(i0, j0) += 2 * T_n . K_bar_chem - T_n . K_bar[ji]
        Fij_bar[i0, j0] += (2.0 * np.sum(T_n_red * ci_ij['K_bar_chem'])
                            - np.sum(T_n_red * ci_ij['K_bar_ji']))
        if i0 != j0:
            Fij_bar[j0, i0] += (2.0 * np.sum(T_n_red * ci_ij['K_bar_chem'])
                                - np.sum(T_n_red * ci_ij['K_bar_ij']))

    # Helper: get integral (k a_ki | c d) from cc_ints.
    # Hoisted out of per-i loop so the merged _per_i closure can use it.
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

    # Pre-compute LT1[(i, m)] = (key_im, L_im @ t1_m_in_im) once per (i, m).
    # Hoisted above the per-i loop so _per_i sees a fully-built cache.
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

    # =========================================================================
    # Per-i residual: Stages 1-4 (Fai_bar / Fab_bar / Fia_bar / Fij_bar @ T_n)
    # merged with A+C contributions. r1_pno[i] writes are disjoint per i, so
    # this is dispatched across `_pool.map` when available.
    # =========================================================================
    import time as _time_perI
    _per_i_dump = getattr(_compute_t1_residual_psi4, '_per_i_dump_timing', False)
    # stages_cy: Cython kernel call (Stages 1-3 fused); kloop: A+C k-loop
    _per_i_t = {'stages_cy': 0.0, 's4': 0.0, 'kloop': 0.0, 'kloop_n': 0}

    def _per_i(i):
        key_ii = (i, i)
        if key_ii not in pno_spaces:
            return i, None
        n_ii = pno_spaces[key_ii]['C_pno'].shape[1]
        if n_ii == 0:
            return i, None

        r1_i = np.zeros(n_ii)
        e_pno_ii = pno_spaces[key_ii]['e_pno']
        t1_i = t1_pno.get(i)
        has_t1_i = t1_i is not None and t1_i.size > 0 and np.max(np.abs(t1_i)) > 1e-15

        ci_ii = cc_ints.get(key_ii)
        if ci_ii is not None:
            # Phase II: kernel takes pair-domain Qma directly. M = nlmo_pair.
            Qma_ii = ci_ii['Qma']      # (n_local, nlmo_p, n_ii)
            Qab_ii = ci_ii['Qab']      # (n_local, n_ii, n_ii)
            Qia_ii = ci_ii['i_Qa']     # (n_local, n_ii)
            Qik_ii = ci_ii['i_Qk']     # (n_local, nlmo_p) — no scatter-back

            # T_n_ii_full[m, c] = t1[m] projected to PNO_ii (nocc, n_ii) —
            # used for Stage 4 (Fij_bar @ T_n on full-nocc axis).
            T_n_ii_full = t1_cache[key_ii]
            # Pair-domain T_n for the kernel (nlmo_p, n_ii).
            p_lmos_ii = ci_ii['p_lmos']
            T_n_ii_pair = np.ascontiguousarray(T_n_ii_full[p_lmos_ii])
            any_t1 = T_n_ii_full.size > 0 and bool(
                np.max(np.abs(T_n_ii_full)) > 1e-15)

            _ts1 = _time_perI.perf_counter() if _per_i_dump else 0.0
            # ---- Stages 1-3 in one C kernel: pyscf/lib/cc/dlpno_per_i.c ----
            if any_t1 or has_t1_i:
                import ctypes
                from pyscf import lib as _pyscflib
                _libcc_pi = _pyscflib.load_library('libcc')
                _libcc_pi.DLPNOper_i_stages123.restype = None
                _libcc_pi.DLPNOper_i_stages123.argtypes = [
                    ctypes.c_void_p,                          # r1_inout
                    ctypes.c_void_p, ctypes.c_void_p,         # Qma, Qab
                    ctypes.c_void_p, ctypes.c_void_p,         # Qia, Qik
                    ctypes.c_void_p, ctypes.c_void_p,         # T_n, t1_i
                    ctypes.c_void_p,                          # e_pno
                    ctypes.c_int,                             # do_stage_23
                    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,  # L, M, A
                ]
                _Qma_c = np.ascontiguousarray(Qma_ii)
                _Qab_c = np.ascontiguousarray(Qab_ii)
                _Qia_c = np.ascontiguousarray(Qia_ii)
                _Qik_c = np.ascontiguousarray(Qik_ii)
                _epn_c = np.ascontiguousarray(e_pno_ii)
                _t1_c = (np.ascontiguousarray(t1_i) if has_t1_i
                         else np.zeros(n_ii))
                if any_t1:
                    _L_dim, _M_dim, _A_dim = _Qma_c.shape
                    _libcc_pi.DLPNOper_i_stages123(
                        r1_i.ctypes.data_as(ctypes.c_void_p),
                        _Qma_c.ctypes.data_as(ctypes.c_void_p),
                        _Qab_c.ctypes.data_as(ctypes.c_void_p),
                        _Qia_c.ctypes.data_as(ctypes.c_void_p),
                        _Qik_c.ctypes.data_as(ctypes.c_void_p),
                        T_n_ii_pair.ctypes.data_as(ctypes.c_void_p),
                        _t1_c.ctypes.data_as(ctypes.c_void_p),
                        _epn_c.ctypes.data_as(ctypes.c_void_p),
                        1 if (has_t1_i and any_t1) else 0,
                        _L_dim, _M_dim, _A_dim,
                    )
                else:
                    # any_t1 == 0 implies Stages 1/2/3 all skipped except
                    # the has_t1_i shortcut: r1 += e_pno * t1
                    r1_i += e_pno_ii * t1_i
            _ts2 = _time_perI.perf_counter() if _per_i_dump else 0.0

            # ---- Stage 4: -sum_k T_n[ii][k,a] * Fij_bar[k,i] ----
            # Uses the full-nocc T_n because Fij_bar is a global (nocc, nocc).
            r1_i -= Fij_bar[:, i] @ T_n_ii_full
            _tsk = _time_perI.perf_counter() if _per_i_dump else 0.0
            if _per_i_dump:
                _per_i_t['stages_cy'] += _ts2 - _ts1
                _per_i_t['s4']        += _tsk - _ts2

        _tk0 = _time_perI.perf_counter() if _per_i_dump else 0.0
        # ---- A + C terms: inner k-loop over ordered pairs (i, k) / (k, i) ----
        for k in range(nocc):
            ki_data = _get_ki_data(k, i)
            if ki_data is None:
                continue
            key_ki = ki_data['key']
            k_Qa = ki_data['k_Qa']
            Qab_ki = ki_data['Qab']

            if key_ki not in t2_pno_all:
                continue
            t2_canon = t2_pno_all[key_ki]
            if k <= i:
                t2_ki = t2_canon
            else:
                t2_ki = t2_canon.T
            Tt_ki = 2.0 * t2_ki - t2_ki.T

            # ----- A term using precomputed K_tilde_chem (Psi4 parity) -----
            # Old form (2 per-iter L-axis einsums):
            #   Z = einsum('Qa,ac->Qc', k_Qa, Tt_ki)
            #   temp_A = einsum('Qc,Qcd->d', Z, Qab_ki)
            # New form (single matmul on cc_ints['K_tilde_chem_*']):
            #   K[a, c, d] = sum_Q k_Qa[Q, a] * Qab[Q, c, d]    (precomputed)
            #   temp_A[d] = sum_{a, c} Tt_ki[a, c] * K[a, c, d]
            #             = Tt_ki.ravel() @ K.reshape(n_ki², n_ki)
            ci_ki_full = cc_ints[key_ki]
            K_tc = (ci_ki_full['K_tilde_chem_i'] if key_ki[0] == k
                    else ci_ki_full['K_tilde_chem_j'])
            n_ki = ki_data['n_ki']
            temp_A = Tt_ki.ravel() @ K_tc.reshape(n_ki * n_ki, n_ki)
            if key_ki == key_ii:
                A_contrib = temp_A
            else:
                S_ii_ki = S_pno_cache.get((key_ii, key_ki))
                A_contrib = S_ii_ki @ temp_A if S_ii_ki is not None else None
            if A_contrib is not None:
                r1_i += A_contrib

            # ----- C term -----
            if i <= k:
                t2_ik = t2_canon
            else:
                t2_ik = t2_canon.T
            Tt_ik = 2.0 * t2_ik - t2_ik.T

            fov_k_in_ki = _project_t1_to_pair(
                fov_pno, k, key_ki, S_pno_cache, pno_spaces)

            # T1 dressing of Fkc (Psi4 ccsd.cc lines 1685-1692)
            fkc_dress = np.zeros_like(fov_k_in_ki)
            for m in range(nocc):
                entry = _LT1_cache.get((i, m))
                if entry is None:
                    continue
                key_im, LT1 = entry
                if key_ki == key_im:
                    fkc_dress += LT1
                else:
                    S_ki_im = S_pno_cache.get((key_ki, key_im))
                    if S_ki_im is not None:
                        fkc_dress += S_ki_im @ LT1
            fov_k_in_ki = fkc_dress

            contrib_C_local = Tt_ik @ fov_k_in_ki
            if key_ki == key_ii:
                C_contrib = contrib_C_local
            else:
                S_ii_ik = S_pno_cache.get((key_ii, key_ki))
                C_contrib = S_ii_ik @ contrib_C_local if S_ii_ik is not None else None
            if C_contrib is not None:
                r1_i += C_contrib

        if _per_i_dump:
            _per_i_t['kloop'] += _time_perI.perf_counter() - _tk0
            _per_i_t['kloop_n'] += 1
        return i, r1_i

    _per_i_wall_start = _time_perI.perf_counter() if _per_i_dump else 0.0
    if _pool is not None:
        _per_i_results = list(_pool.map(_per_i, range(nocc)))
    else:
        _per_i_results = [_per_i(i) for i in range(nocc)]
    for _i, _r1_i in _per_i_results:
        if _r1_i is not None:
            r1_pno[_i] = _r1_i
    _per_i_wall_end = _time_perI.perf_counter() if _per_i_dump else 0.0
    if _per_i_dump:
        print(f'  PER_I_DBG: wall={_per_i_wall_end - _per_i_wall_start:.3f}s '
              f'stages_cy={_per_i_t["stages_cy"]*1e3:.1f}ms '
              f's4={_per_i_t["s4"]*1e3:.1f}ms '
              f'kloop={_per_i_t["kloop"]*1e3:.1f}ms '
              f'kloop_n={_per_i_t["kloop_n"]}',
              flush=True)

    # ===========================================================
    # B and A2 terms: loop over ordered pairs (k, l)
    # ===========================================================
    # For each ordered pair (k, l), compute B_ia and A2 contributions
    # for each orbital i in pair (k,l)'s LMO domain. r1_pno[i] receives
    # contributions from many (key_kl, ordering) tuples → parallelize
    # with a per-worker reduction: each task returns a dict of per-i
    # contributions, accumulated serially in the main thread.
    # ---- Static per-task plan: hoists every dict lookup out of the
    # inner loop. Built once per CCSD run (keyed by t2_pno_all keys +
    # pno_spaces id), then read directly each cycle.
    # When cc_ints_flat and pair_index are passed (driver path), an
    # additional FLAT plan is built that lets a single batched Cython
    # kernel replace the per-task pool.map dispatch entirely.
    _pkl_cache_attr = getattr(_compute_t1_residual_psi4, '_per_kl_plan_cache', None)
    if _pkl_cache_attr is None:
        _pkl_cache_attr = {}
        _compute_t1_residual_psi4._per_kl_plan_cache = _pkl_cache_attr
    _pkl_plan_key = (id(cc_ints), id(pno_spaces), id(S_pno_cache),
                     id(pair_lmo_idx), id(t2_pno_all),
                     id(cc_ints_flat), id(pair_index))
    _pkl_plan = _pkl_cache_attr.get(_pkl_plan_key)
    if _pkl_plan is None:
        _ba_work = []
        for key_kl in t2_pno_all:
            if pno_spaces[key_kl]['C_pno'].shape[1] == 0:
                continue
            k0, l0 = key_kl
            if k0 == l0:
                _ba_work.append((key_kl, k0, l0))
            else:
                _ba_work.append((key_kl, k0, l0))
                _ba_work.append((key_kl, l0, k0))

        # Per-task: prebuild K_iajb_kl, K_bar_kl refs, k_first flag,
        # transpose flag, and the i-list with all per-i lookups resolved.
        _per_task_plan = []
        for arg in _ba_work:
            key_kl, k, l = arg
            ci_kl = cc_ints.get(key_kl)
            if ci_kl is None:
                _per_task_plan.append(None)
                continue
            n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
            K_iajb_kl = ci_kl['K_iajb']
            # K_bar_ij/ji reduced to (nlmo_p, n_kl) in cc_ints; the existing
            # per-task plan + Cython kernel iterate over global LMOs, so
            # scatter back to (nocc, n_kl) at this callsite. Kernels stay
            # live during the correctness port; will collapse this when we
            # sweep .pyx -> pyscf/lib/cc/dlpno_*.c.
            _K_bar_red = (ci_kl['K_bar_ij'] if key_kl[0] == k
                          else ci_kl['K_bar_ji'])
            K_bar_kl = np.zeros((nocc, n_kl))
            K_bar_kl[ci_kl['p_lmos']] = _K_bar_red
            t2_swap_kl = (k > l)

            i_list = (pair_lmo_idx[key_kl]
                      if pair_lmo_idx is not None and key_kl in pair_lmo_idx
                      else list(range(nocc)))
            inners = []
            for i in i_list:
                key_ii = (i, i)
                if key_ii not in pno_spaces:
                    continue
                n_pno_ii = pno_spaces[key_ii]['C_pno'].shape[1]
                if n_pno_ii == 0:
                    continue
                # B-side static
                S_ii_kl = (None if key_kl == key_ii
                           else S_pno_cache.get((key_ii, key_kl)))
                # A2-side static
                key_ki = (min(k, i), max(k, i))
                a2_n_ki = 0
                S_kl_ki = None
                S_ki_kl = None
                t2_swap_ki = False
                a2_diag = False
                if (key_ki in t2_pno_all and key_ki in pno_spaces
                        and pno_spaces[key_ki]['C_pno'].shape[1] > 0):
                    a2_n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
                    t2_swap_ki = (k > i)
                    if key_kl == key_ki:
                        a2_diag = True
                    else:
                        S_kl_ki = S_pno_cache.get((key_kl, key_ki))
                        S_ki_kl = S_pno_cache.get((key_ki, key_kl))
                inners.append((i, key_ii, n_pno_ii, S_ii_kl,
                               key_ki, a2_n_ki, t2_swap_ki, a2_diag,
                               S_kl_ki, S_ki_kl))
            _per_task_plan.append({
                'key_kl': key_kl, 'k': k, 'l': l, 'n_kl': n_kl,
                'K_iajb_kl': K_iajb_kl, 'K_bar_kl': K_bar_kl,
                't2_swap_kl': t2_swap_kl, 'inners': inners,
            })
        # ---- Batched plan: flat per-task + per-(t,i) metadata that the
        # single-call Cython kernel reads directly. Only built when the
        # driver passes cc_ints_flat + pair_index (we need the underlying
        # FlatTensorStore _buffers + _offsets).
        _batched_plan = None
        if (cc_ints_flat is not None and pair_index is not None
                and hasattr(t2_pno_all, '_buffer')):
            _b_n_kl = []
            _b_t2_swap_kl = []
            _b_K_iajb_off = []
            _b_K_bar_off = []
            _b_t2_kl_canon_off = []
            _b_T_n_kl_off = []
            _b_inner_off = [0]
            _b_i_arr = []
            _b_n_pno_ii = []
            _b_is_diag_kl_ii = []
            _b_has_S_ii_kl = []
            _b_S_ii_kl_off = []
            _b_has_A2 = []
            _b_is_diag_kl_ki = []
            _b_n_ki = []
            _b_t2_swap_ki = []
            _b_t2_ki_canon_off = []
            _b_S_kl_ki_off = []
            _b_S_ki_kl_off = []
            _b_T_n_l_ii_off = []
            _b_contrib_off = [0]
            _b_K_iajb_pieces = []
            _b_K_bar_pieces = []
            _b_K_iajb_off_running = 0
            _b_K_bar_off_running = 0

            _t2_offsets = np.asarray(t2_pno_all._offsets)
            # t1_cache offsets are stable across CCSD cycles: pno_spaces
            # shapes don't change so build_t1_cache always allocates the
            # same layout. Snapshot here.
            _t1c_offsets = np.asarray(t1_cache._offsets)
            _S_idx_matrix = S_pno_cache._idx_matrix
            _S_offsets = np.asarray(S_pno_cache._offsets)
            _canon_to_idx = pair_index.canonical_to_idx
            _abort_batched = False
            # Build our own consolidated S buffer that includes overflow
            # entries (which other phases — compute_C_tilde, build_D_tilde
            # — populate lazily before T1 residual runs). Plan-time offsets
            # point into _S_local rather than S_pno_cache._buffer, so the
            # kernel sees a single contiguous buffer that's complete.
            _S_local_pieces = []
            _S_local_off_running = 0
            _S_local_overflow_off = {}   # (ia, ib) -> offset into _S_local

            def _resolve_S(ia, ib):
                """Return offset into _S_local for S[(ia,ib)] (None if missing)."""
                idx = int(_S_idx_matrix[ia, ib])
                if idx >= 0:
                    return int(_S_offsets[idx])
                if (ia, ib) in S_pno_cache._overflow:
                    if (ia, ib) in _S_local_overflow_off:
                        return _S_local_overflow_off[(ia, ib)]
                    arr = S_pno_cache._overflow[(ia, ib)]
                    arr = np.ascontiguousarray(arr)
                    nonlocal _S_local_off_running
                    off = _S_buffer_size + _S_local_off_running
                    _S_local_overflow_off[(ia, ib)] = off
                    _S_local_pieces.append(arr.ravel())
                    _S_local_off_running += arr.size
                    return off
                return None

            _S_buffer_size = int(S_pno_cache._buffer.shape[0])

            for arg in _ba_work:
                key_kl, k, l = arg
                ci_kl = cc_ints.get(key_kl)
                if ci_kl is None:
                    # No-op task; preserve task slot with 0 inners
                    _b_n_kl.append(0)
                    _b_t2_swap_kl.append(0)
                    _b_K_iajb_off.append(0)
                    _b_K_bar_off.append(0)
                    _b_t2_kl_canon_off.append(0)
                    _b_T_n_kl_off.append(0)
                    _b_inner_off.append(_b_inner_off[-1])
                    continue
                n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
                canon_kl_idx = _canon_to_idx[key_kl]
                # K_iajb_kl + K_bar_kl: copy into our own static flat
                # buffers (own them, simpler than passing two cc_ints
                # buffers + a k_first selector).
                K_iajb_kl = np.ascontiguousarray(ci_kl['K_iajb'])
                _b_K_iajb_pieces.append(K_iajb_kl.ravel())
                _b_K_iajb_off.append(_b_K_iajb_off_running)
                _b_K_iajb_off_running += K_iajb_kl.size
                # K_bar reduced (nlmo_p, n_kl) -> scatter to (nocc, n_kl)
                # so the Cython kernel keeps its global-LMO iteration.
                _K_bar_red = (ci_kl['K_bar_ij'] if key_kl[0] == k
                              else ci_kl['K_bar_ji'])
                K_bar_kl = np.zeros((nocc, n_kl))
                K_bar_kl[ci_kl['p_lmos']] = _K_bar_red
                K_bar_kl = np.ascontiguousarray(K_bar_kl)
                _b_K_bar_pieces.append(K_bar_kl.ravel())
                _b_K_bar_off.append(_b_K_bar_off_running)
                _b_K_bar_off_running += K_bar_kl.size

                _b_n_kl.append(n_kl)
                _b_t2_swap_kl.append(1 if k > l else 0)
                _b_t2_kl_canon_off.append(int(_t2_offsets[canon_kl_idx]))
                _b_T_n_kl_off.append(int(_t1c_offsets[canon_kl_idx]))

                i_list = (pair_lmo_idx[key_kl]
                          if pair_lmo_idx is not None and key_kl in pair_lmo_idx
                          else list(range(nocc)))
                n_inner_for_task = 0
                for i in i_list:
                    key_ii = (i, i)
                    if key_ii not in pno_spaces:
                        continue
                    n_pno_ii = pno_spaces[key_ii]['C_pno'].shape[1]
                    if n_pno_ii == 0:
                        continue
                    canon_ii_idx = _canon_to_idx[key_ii]
                    _b_i_arr.append(i)
                    _b_n_pno_ii.append(n_pno_ii)
                    _b_contrib_off.append(_b_contrib_off[-1] + n_pno_ii)
                    n_inner_for_task += 1

                    # B-side
                    if key_kl == key_ii:
                        _b_is_diag_kl_ii.append(1)
                        _b_has_S_ii_kl.append(0)
                        _b_S_ii_kl_off.append(0)
                    else:
                        s_off = _resolve_S(canon_ii_idx, canon_kl_idx)
                        if s_off is not None:
                            _b_is_diag_kl_ii.append(0)
                            _b_has_S_ii_kl.append(1)
                            _b_S_ii_kl_off.append(s_off)
                        else:
                            _b_is_diag_kl_ii.append(0)
                            _b_has_S_ii_kl.append(0)
                            _b_S_ii_kl_off.append(0)

                    # A2-side
                    key_ki = (min(k, i), max(k, i))
                    a2_added = False
                    if (key_ki in t2_pno_all and key_ki in pno_spaces
                            and pno_spaces[key_ki]['C_pno'].shape[1] > 0):
                        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
                        canon_ki_idx = _canon_to_idx[key_ki]
                        if key_kl == key_ki:
                            _b_has_A2.append(1)
                            _b_is_diag_kl_ki.append(1)
                            _b_n_ki.append(n_ki)
                            _b_t2_swap_ki.append(1 if k > i else 0)
                            _b_t2_ki_canon_off.append(int(_t2_offsets[canon_ki_idx]))
                            _b_S_kl_ki_off.append(0)
                            _b_S_ki_kl_off.append(0)
                            _b_T_n_l_ii_off.append(
                                int(_t1c_offsets[canon_ii_idx]) + l * n_pno_ii)
                            a2_added = True
                        else:
                            s_kl_ki_off = _resolve_S(canon_kl_idx, canon_ki_idx)
                            s_ki_kl_off = _resolve_S(canon_ki_idx, canon_kl_idx)
                            if s_kl_ki_off is not None and s_ki_kl_off is not None:
                                _b_has_A2.append(1)
                                _b_is_diag_kl_ki.append(0)
                                _b_n_ki.append(n_ki)
                                _b_t2_swap_ki.append(1 if k > i else 0)
                                _b_t2_ki_canon_off.append(int(_t2_offsets[canon_ki_idx]))
                                _b_S_kl_ki_off.append(s_kl_ki_off)
                                _b_S_ki_kl_off.append(s_ki_kl_off)
                                _b_T_n_l_ii_off.append(
                                    int(_t1c_offsets[canon_ii_idx]) + l * n_pno_ii)
                                a2_added = True
                    if not a2_added:
                        _b_has_A2.append(0)
                        _b_is_diag_kl_ki.append(0)
                        _b_n_ki.append(0)
                        _b_t2_swap_ki.append(0)
                        _b_t2_ki_canon_off.append(0)
                        _b_S_kl_ki_off.append(0)
                        _b_S_ki_kl_off.append(0)
                        _b_T_n_l_ii_off.append(0)
                _b_inner_off.append(_b_inner_off[-1] + n_inner_for_task)

            if _b_K_iajb_pieces:
                _K_iajb_static = np.concatenate(_b_K_iajb_pieces)
                _K_bar_static = np.concatenate(_b_K_bar_pieces)
                # Consolidated S buffer: original flat tier + lazy overflow.
                if _S_local_pieces:
                    _S_consolidated = np.concatenate(
                        [S_pno_cache._buffer] + _S_local_pieces)
                else:
                    _S_consolidated = S_pno_cache._buffer
                _max_n_kl = max(_b_n_kl) if _b_n_kl else 1
                _max_n_ki = max(_b_n_ki) if _b_n_ki else 1
                _batched_plan = {
                    'M': nocc,
                    'n_tasks': len(_b_n_kl),
                    'n_total_inner': len(_b_i_arr),
                    'n_kl_arr': np.asarray(_b_n_kl, dtype=np.int32),
                    't2_swap_kl': np.asarray(_b_t2_swap_kl, dtype=np.int32),
                    'K_iajb_off': np.asarray(_b_K_iajb_off, dtype=np.int64),
                    'K_bar_off': np.asarray(_b_K_bar_off, dtype=np.int64),
                    't2_kl_canon_off': np.asarray(_b_t2_kl_canon_off, dtype=np.int64),
                    'T_n_kl_off': np.asarray(_b_T_n_kl_off, dtype=np.int64),
                    'inner_off': np.asarray(_b_inner_off, dtype=np.int64),
                    'i_arr': np.asarray(_b_i_arr, dtype=np.int32),
                    'n_pno_ii_arr': np.asarray(_b_n_pno_ii, dtype=np.int32),
                    'is_diag_kl_ii': np.asarray(_b_is_diag_kl_ii, dtype=np.int32),
                    'has_S_ii_kl': np.asarray(_b_has_S_ii_kl, dtype=np.int32),
                    'S_ii_kl_off': np.asarray(_b_S_ii_kl_off, dtype=np.int64),
                    'has_A2': np.asarray(_b_has_A2, dtype=np.int32),
                    'is_diag_kl_ki': np.asarray(_b_is_diag_kl_ki, dtype=np.int32),
                    'n_ki_arr': np.asarray(_b_n_ki, dtype=np.int32),
                    't2_swap_ki': np.asarray(_b_t2_swap_ki, dtype=np.int32),
                    't2_ki_canon_off': np.asarray(_b_t2_ki_canon_off, dtype=np.int64),
                    'S_kl_ki_off': np.asarray(_b_S_kl_ki_off, dtype=np.int64),
                    'S_ki_kl_off': np.asarray(_b_S_ki_kl_off, dtype=np.int64),
                    'T_n_l_ii_off': np.asarray(_b_T_n_l_ii_off, dtype=np.int64),
                    'contrib_off': np.asarray(_b_contrib_off, dtype=np.int64),
                    'K_iajb_static': _K_iajb_static,
                    'K_bar_static': _K_bar_static,
                    'S_consolidated': _S_consolidated,
                    'max_n_kl': _max_n_kl,
                    'max_n_ki': _max_n_ki,
                }

        _pkl_plan = {'_ba_work': _ba_work,
                     '_per_task_plan': _per_task_plan,
                     '_batched_plan': _batched_plan}
        _pkl_cache_attr[_pkl_plan_key] = _pkl_plan
    _ba_work = _pkl_plan['_ba_work']
    _per_task_plan = _pkl_plan['_per_task_plan']
    _batched_plan = _pkl_plan['_batched_plan']

    def _per_kl(arg_with_plan):
        plan = arg_with_plan
        if plan is None:
            return {}
        local = {}
        key_kl = plan['key_kl']
        k = plan['k']; l = plan['l']; n_kl = plan['n_kl']
        K_iajb_kl = plan['K_iajb_kl']
        K_bar_kl = plan['K_bar_kl']

        t2_canon_kl = t2_pno_all[key_kl]
        t2_kl = t2_canon_kl.T if plan['t2_swap_kl'] else t2_canon_kl
        Tt_kl = 2.0 * t2_kl - t2_kl.T

        T_n_kl = t1_cache[key_kl]
        K_kilc = K_bar_kl + T_n_kl @ K_iajb_kl
        B_ia = Tt_kl @ K_kilc.T

        for (i, key_ii, n_pno_ii, S_ii_kl,
             key_ki, a2_n_ki, t2_swap_ki, a2_diag,
             S_kl_ki, S_ki_kl) in plan['inners']:
            # B contribution
            if S_ii_kl is None and key_kl == key_ii:
                B_contrib = -B_ia[:, i]
                local[i] = B_contrib if i not in local else local[i] + B_contrib
            elif S_ii_kl is not None:
                B_contrib = -(S_ii_kl @ B_ia[:, i])
                local[i] = B_contrib if i not in local else local[i] + B_contrib

            # A2 contribution
            if a2_n_ki > 0:
                t2_ki_canon = t2_pno_all[key_ki]
                t2_ki = t2_ki_canon.T if t2_swap_ki else t2_ki_canon
                Tt_ki = 2.0 * t2_ki - t2_ki.T
                if a2_diag:
                    U_ki = Tt_ki
                elif S_kl_ki is not None and S_ki_kl is not None:
                    U_ki = S_kl_ki @ Tt_ki @ S_ki_kl
                else:
                    U_ki = None
                if U_ki is not None:
                    scalar = np.sum(K_iajb_kl * U_ki)
                    A2 = -(scalar * t1_cache[key_ii][l])
                    local[i] = A2 if i not in local else local[i] + A2
        return local

    _per_kl_wall_start = _time_perI.perf_counter() if _per_i_dump else 0.0
    if _batched_plan is not None and _batched_plan['n_tasks'] > 0:
        # Single batched Cython call replaces the entire pool.map dispatch.
        from pyscf.cc.dlpno_tccsd._per_kl_batched_cy import per_kl_batched
        from threadpoolctl import threadpool_limits
        bp = _batched_plan
        # Per-thread scratch — sized for max shapes across all tasks.
        _bp_scratch = getattr(_compute_t1_residual_psi4,
                              '_per_kl_batched_scratch', None)
        _NTH = min(64, bp['n_tasks'])
        _max_n_kl = bp['max_n_kl']
        _max_n_ki = bp['max_n_ki']
        _M = bp['M']
        if (_bp_scratch is None
                or _bp_scratch['num_threads'] != _NTH
                or _bp_scratch['max_n_kl'] != _max_n_kl
                or _bp_scratch['max_n_ki'] != _max_n_ki
                or _bp_scratch['M'] != _M):
            _bp_scratch = {
                'num_threads': _NTH,
                'max_n_kl': _max_n_kl, 'max_n_ki': _max_n_ki, 'M': _M,
                'Tt_kl': np.empty((_NTH, _max_n_kl * _max_n_kl)),
                'K_kilc': np.empty((_NTH, _M * _max_n_kl)),
                'B_ia': np.empty((_NTH, _max_n_kl * _M)),
                'Tt_ki': np.empty((_NTH, max(_max_n_ki, 1) ** 2)),
                'X': np.empty((_NTH, max(_max_n_ki, 1) * _max_n_kl)),
                'Z': np.empty((_NTH, max(_max_n_ki, 1) ** 2)),
            }
            _compute_t1_residual_psi4._per_kl_batched_scratch = _bp_scratch

        contrib_flat = np.zeros(int(bp['contrib_off'][-1]))
        with threadpool_limits(limits=1, user_api='blas'):
            if int(os.environ.get('DLPNO_C_CYCLE', '0')):
                # Native-C path: see pyscf/lib/cc/dlpno_t1_residual.c::
                # DLPNOper_kl_batched. Same flat-buffer plan + per-thread
                # scratch as the Cython kernel.
                import ctypes as _ct
                from pyscf import lib as _pyscflib
                _libcc = getattr(_compute_t1_residual_psi4, '_libcc', None)
                if _libcc is None:
                    _libcc = _pyscflib.load_library('libcc')
                    _libcc.DLPNOper_kl_batched.restype = None
                    _libcc.DLPNOper_kl_batched.argtypes = (
                        [_ct.c_int, _ct.c_int]                      # n_tasks, M
                        + [_ct.c_void_p] * 7                        # per-task arrays
                        + [_ct.c_void_p] * 14                       # per-(t,i) arrays
                        + [_ct.c_void_p] * 5                        # static + dynamic flat buffers
                        + [_ct.c_void_p, _ct.c_size_t] * 6          # 6 scratch (ptr, stride)
                        + [_ct.c_void_p, _ct.c_int])                # contrib_flat, num_threads
                    _compute_t1_residual_psi4._libcc = _libcc
                _libcc.DLPNOper_kl_batched(
                    int(bp['n_tasks']), int(_M),
                    bp['n_kl_arr'].ctypes.data_as(_ct.c_void_p),
                    bp['t2_swap_kl'].ctypes.data_as(_ct.c_void_p),
                    bp['K_iajb_off'].ctypes.data_as(_ct.c_void_p),
                    bp['K_bar_off'].ctypes.data_as(_ct.c_void_p),
                    bp['t2_kl_canon_off'].ctypes.data_as(_ct.c_void_p),
                    bp['T_n_kl_off'].ctypes.data_as(_ct.c_void_p),
                    bp['inner_off'].ctypes.data_as(_ct.c_void_p),
                    bp['i_arr'].ctypes.data_as(_ct.c_void_p),
                    bp['n_pno_ii_arr'].ctypes.data_as(_ct.c_void_p),
                    bp['is_diag_kl_ii'].ctypes.data_as(_ct.c_void_p),
                    bp['has_S_ii_kl'].ctypes.data_as(_ct.c_void_p),
                    bp['S_ii_kl_off'].ctypes.data_as(_ct.c_void_p),
                    bp['has_A2'].ctypes.data_as(_ct.c_void_p),
                    bp['is_diag_kl_ki'].ctypes.data_as(_ct.c_void_p),
                    bp['n_ki_arr'].ctypes.data_as(_ct.c_void_p),
                    bp['t2_swap_ki'].ctypes.data_as(_ct.c_void_p),
                    bp['t2_ki_canon_off'].ctypes.data_as(_ct.c_void_p),
                    bp['S_kl_ki_off'].ctypes.data_as(_ct.c_void_p),
                    bp['S_ki_kl_off'].ctypes.data_as(_ct.c_void_p),
                    bp['T_n_l_ii_off'].ctypes.data_as(_ct.c_void_p),
                    bp['contrib_off'].ctypes.data_as(_ct.c_void_p),
                    bp['K_iajb_static'].ctypes.data_as(_ct.c_void_p),
                    bp['K_bar_static'].ctypes.data_as(_ct.c_void_p),
                    bp['S_consolidated'].ctypes.data_as(_ct.c_void_p),
                    t2_pno_all._buffer.ctypes.data_as(_ct.c_void_p),
                    t1_cache._buffer.ctypes.data_as(_ct.c_void_p),
                    _bp_scratch['Tt_kl'].ctypes.data_as(_ct.c_void_p), _bp_scratch['Tt_kl'].shape[1],
                    _bp_scratch['K_kilc'].ctypes.data_as(_ct.c_void_p), _bp_scratch['K_kilc'].shape[1],
                    _bp_scratch['B_ia'].ctypes.data_as(_ct.c_void_p), _bp_scratch['B_ia'].shape[1],
                    _bp_scratch['Tt_ki'].ctypes.data_as(_ct.c_void_p), _bp_scratch['Tt_ki'].shape[1],
                    _bp_scratch['X'].ctypes.data_as(_ct.c_void_p), _bp_scratch['X'].shape[1],
                    _bp_scratch['Z'].ctypes.data_as(_ct.c_void_p), _bp_scratch['Z'].shape[1],
                    contrib_flat.ctypes.data_as(_ct.c_void_p),
                    int(_NTH),
                )
            else:
                per_kl_batched(
                    bp['n_tasks'], _M,
                    bp['n_kl_arr'], bp['t2_swap_kl'],
                    bp['K_iajb_off'], bp['K_bar_off'],
                    bp['t2_kl_canon_off'], bp['T_n_kl_off'],
                    bp['inner_off'],
                    bp['i_arr'], bp['n_pno_ii_arr'],
                    bp['is_diag_kl_ii'], bp['has_S_ii_kl'], bp['S_ii_kl_off'],
                    bp['has_A2'], bp['is_diag_kl_ki'], bp['n_ki_arr'],
                    bp['t2_swap_ki'], bp['t2_ki_canon_off'],
                    bp['S_kl_ki_off'], bp['S_ki_kl_off'],
                    bp['T_n_l_ii_off'], bp['contrib_off'],
                    bp['K_iajb_static'], bp['K_bar_static'],
                    bp['S_consolidated'],
                    t2_pno_all._buffer, t1_cache._buffer,
                    _bp_scratch['Tt_kl'], _bp_scratch['K_kilc'],
                    _bp_scratch['B_ia'], _bp_scratch['Tt_ki'],
                    _bp_scratch['X'], _bp_scratch['Z'],
                    contrib_flat,
                    _NTH,
                )
        # Aggregate per-(t,i) contributions into r1_pno[i]
        i_arr_np = bp['i_arr']
        contrib_off_np = bp['contrib_off']
        for ti in range(int(bp['n_total_inner'])):
            i = int(i_arr_np[ti])
            r1_pno[i] += contrib_flat[
                contrib_off_np[ti]:contrib_off_np[ti + 1]]
        if _per_i_dump:
            print(f'  PER_KL_DBG[batched]: wall={_time_perI.perf_counter() - _per_kl_wall_start:.3f}s '
                  f'n_tasks={bp["n_tasks"]} n_inner={bp["n_total_inner"]}',
                  flush=True)
    else:
        # Fall back to the pool.map + chunking path.
        _CHUNK = 32

        def _per_kl_chunk(chunk):
            return [_per_kl(p) for p in chunk]

        _chunks = [_per_task_plan[i:i + _CHUNK]
                   for i in range(0, len(_per_task_plan), _CHUNK)]
        if _pool is not None:
            _chunked_results = list(_pool.map(_per_kl_chunk, _chunks))
        else:
            _chunked_results = [[_per_kl(p) for p in c] for c in _chunks]
        for cr in _chunked_results:
            for local in cr:
                for i, contrib in local.items():
                    r1_pno[i] += contrib
        if _per_i_dump:
            print(f'  PER_KL_DBG[python]: wall={_time_perI.perf_counter() - _per_kl_wall_start:.3f}s '
                  f'n_tasks={len(_per_task_plan)}', flush=True)

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

    # Fine-grained pool: ~1000 small numpy tasks per pool.map (e.g.
    # compute_C_tilde_batched Phase 1) scale poorly on a 64-worker
    # pool due to GIL contention between BLAS calls. Benchmark on
    # water10-scale work shows the sweet spot is 4-8 workers; past
    # 16 workers parallelism degrades. Route the fine-grained per-
    # pair builders through this smaller pool; keep `_pool` (full
    # 64-worker pool) for coarse tasks like _update_pair and the
    # T1 residual per-LMO map.
    from concurrent.futures import ThreadPoolExecutor
    import os as _os
    _fine_n = int(_os.environ.get('DLPNO_FINE_POOL_SIZE', '8'))
    _fine_pool = (ThreadPoolExecutor(max_workers=_fine_n)
                  if _pool is not None else None)

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
    _t_post_ccints = _time_cc.perf_counter()

    # ------------------------------------------------------------------
    # Phase 2c + 2e of the restructure — now happens in one block right
    # after cc_ints is available.  The canonical PairIndex is built once
    # per CCSD run and the per-pair amplitude / integral dicts are
    # flattened onto contiguous FlatTensorStore buffers.  Every
    # downstream site still uses dict-style access, but the backing
    # memory layout now matches what Phase 4's Cython kernels expect.
    # ------------------------------------------------------------------
    from pyscf.cc.dlpno_tccsd.pair_index import (
        PairIndex, FlatTensorStore, assert_consistent_with_dicts,
        flatten_cc_ints_fields,
    )
    _pair_index = PairIndex(
        list(t2_pno_all.keys()), pno_spaces, pair_lmo_idx, nocc)
    assert_consistent_with_dicts(
        _pair_index, pno_spaces, pair_lmo_idx)
    print(f'  [pair_index] {_pair_index!r}', flush=True)

    # Phase 2e: flatten the 12 tensor fields of cc_ints onto shared
    # per-field FlatTensorStore buffers.  The dict keeps its structure;
    # each field's per-pair ndarray becomes a view into a contiguous
    # buffer.  The `_cc_ints_flat` handle is kept around for Phase 4's
    # Cython kernels (``.buffer`` / ``.offsets`` / ``.shapes``).
    _t_flat = _time_cc.perf_counter()
    _cc_ints_flat = flatten_cc_ints_fields(_cc_ints, _pair_index)
    print(f'  [cc_ints_flat] {len(_cc_ints_flat)} fields flattened, '
          f'{_time_cc.perf_counter() - _t_flat:.2f}s '
          f'total_buffer={sum(s.buffer.size for s in _cc_ints_flat.values())*8/2**20:.1f} MB',
          flush=True)

    # Rebuild K_pno_cache from locally-fitted K_iajb (now a view
    # into the flattened Qab/K_iajb field store).
    for key in list(K_pno_cache.keys()):
        ci = _cc_ints.get(key)
        if ci is not None:
            K_pno_cache[key] = ci['K_iajb']

    # Phase 2c: flatten the (n_pno, n_pno) amplitude / K_pno dicts.
    _pair_t2_shape = lambda p: (int(_pair_index.n_pno[p]),
                                int(_pair_index.n_pno[p]))
    t2_pno_all = FlatTensorStore.from_dict(
        _pair_index, t2_pno_all, shape_fn=_pair_t2_shape)
    K_pno_cache = FlatTensorStore.from_dict(
        _pair_index, K_pno_cache, shape_fn=_pair_t2_shape)

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
    # Build into a plain dict first — the upfront set is domain-
    # restricted and we don't yet know every (pair_a, pair_b) the
    # iteration will ask for.  Convert to FlatPairPairStore below;
    # later lazy additions go into the overflow dict.
    #
    # Parallel across `key_ij`: each task computes all its partner
    # overlaps independently (no cross-task writes).  Releases the GIL
    # via numpy @ / np.ix_, so ThreadPool scales near-linearly with
    # `_pool` size when OMP_NUM_THREADS=1.  Without the pool, falls
    # back to the serial path.
    # Native-C S_pno builder: per-pair-A all-partners-B in one C call.
    # Eliminates ~25k Python compute_S_pno dispatches on water-10 (~5s wall).
    _use_spno_c = bool(int(os.environ.get('DLPNO_C_CYCLE', '0')))
    if _use_spno_c:
        import ctypes as _ct
        from pyscf import lib as _pyscflib
        _libcc_spno = _pyscflib.load_library('libcc')
        _libcc_spno.DLPNObuild_S_pno_for_pair.restype = None
        _libcc_spno.DLPNObuild_S_pno_for_pair.argtypes = (
            [_ct.c_void_p, _ct.c_void_p,
             _ct.c_int, _ct.c_int, _ct.c_int]
            + [_ct.c_void_p] * 8
            + [_ct.c_void_p, _ct.c_size_t])
        _S_pao_full_c = np.ascontiguousarray(S_pao_full)
        _n_pao_total = _S_pao_full_c.shape[0]

    def _build_one_key(key_ij):
        if pair_lmo_idx is not None and key_ij in pair_lmo_idx:
            _dom = [int(x) for x in pair_lmo_idx[key_ij]]
            _partner_keys = [
                (k, l) for k in _dom for l in _dom
                if k <= l and (k, l) in _all_pair_set
            ]
        else:
            _partner_keys = _all_pair_keys

        if not _use_spno_c:
            out = {}
            for key_kl in _partner_keys:
                out[(key_ij, key_kl)] = _compute_S_pno(
                    key_ij, key_kl, pno_spaces, S_pao_full, s1e)
            return out

        # C path: gather partner X / pair_paos into flat buffers and dispatch
        # one DLPNObuild_S_pno_for_pair call per pair_a.
        pd_a = pno_spaces[key_ij]
        X_a_full = pd_a.get('X_pno')
        pp_a = pd_a.get('pair_paos')
        if X_a_full is None or pp_a is None:
            # Some partner key_kl might also lack X_pno; fall back.
            out = {}
            for key_kl in _partner_keys:
                out[(key_ij, key_kl)] = _compute_S_pno(
                    key_ij, key_kl, pno_spaces, S_pao_full, s1e)
            return out

        # Filter partners to those with X_pno set (Psi4 path).
        good = []
        fallback = []
        for key_kl in _partner_keys:
            pd_b = pno_spaces[key_kl]
            if (pd_b.get('X_pno') is not None
                    and pd_b.get('pair_paos') is not None):
                good.append(key_kl)
            else:
                fallback.append(key_kl)

        out = {}
        for key_kl in fallback:
            out[(key_ij, key_kl)] = _compute_S_pno(
                key_ij, key_kl, pno_spaces, S_pao_full, s1e)

        if not good:
            return out

        n_pao_a = int(np.asarray(pp_a).size)
        n_pno_a = int(X_a_full.shape[1])
        n_partners = len(good)
        partner_n_pao = np.empty(n_partners, dtype=np.int32)
        partner_n_pno = np.empty(n_partners, dtype=np.int32)
        for p, key_kl in enumerate(good):
            pd_b = pno_spaces[key_kl]
            partner_n_pao[p] = int(np.asarray(pd_b['pair_paos']).size)
            partner_n_pno[p] = int(pd_b['X_pno'].shape[1])
        pp_off = np.empty(n_partners + 1, dtype=np.int64)
        pp_off[0] = 0
        pp_off[1:] = np.cumsum(partner_n_pao.astype(np.int64))
        X_sizes = (partner_n_pao.astype(np.int64)
                   * partner_n_pno.astype(np.int64))
        X_off = np.empty(n_partners + 1, dtype=np.int64)
        X_off[0] = 0
        X_off[1:] = np.cumsum(X_sizes)
        S_sizes = (partner_n_pno.astype(np.int64) * n_pno_a)
        S_off = np.empty(n_partners + 1, dtype=np.int64)
        S_off[0] = 0
        S_off[1:] = np.cumsum(S_sizes)

        pp_flat = np.empty(int(pp_off[-1]), dtype=np.int64)
        X_flat = np.empty(int(X_off[-1]))
        for p, key_kl in enumerate(good):
            pd_b = pno_spaces[key_kl]
            pp_flat[pp_off[p]:pp_off[p + 1]] = np.asarray(
                pd_b['pair_paos'], dtype=np.int64)
            X_flat[X_off[p]:X_off[p + 1]] = np.ascontiguousarray(
                pd_b['X_pno']).ravel()
        S_flat = np.empty(int(S_off[-1]))

        pp_a_arr = np.ascontiguousarray(pp_a, dtype=np.int64)
        X_a_arr  = np.ascontiguousarray(X_a_full)
        _libcc_spno.DLPNObuild_S_pno_for_pair(
            pp_a_arr.ctypes.data_as(_ct.c_void_p),
            X_a_arr.ctypes.data_as(_ct.c_void_p),
            int(n_pao_a), int(n_pno_a), int(n_partners),
            partner_n_pao.ctypes.data_as(_ct.c_void_p),
            partner_n_pno.ctypes.data_as(_ct.c_void_p),
            pp_off.ctypes.data_as(_ct.c_void_p),
            pp_flat.ctypes.data_as(_ct.c_void_p),
            X_off.ctypes.data_as(_ct.c_void_p),
            X_flat.ctypes.data_as(_ct.c_void_p),
            S_off.ctypes.data_as(_ct.c_void_p),
            S_flat.ctypes.data_as(_ct.c_void_p),
            _S_pao_full_c.ctypes.data_as(_ct.c_void_p),
            _n_pao_total,
        )
        for p, key_kl in enumerate(good):
            n_pno_b = int(partner_n_pno[p])
            out[(key_ij, key_kl)] = (
                S_flat[S_off[p]:S_off[p + 1]]
                .reshape(n_pno_a, n_pno_b).copy())
        return out

    _t_spno = _time_cc.perf_counter()
    S_pno_cache = {}
    if _pool is not None:
        for sub in _pool.map(_build_one_key, _all_pair_keys):
            S_pno_cache.update(sub)
    else:
        for key_ij in _all_pair_keys:
            S_pno_cache.update(_build_one_key(key_ij))
    print(f'  [STAGE5_DBG] S_pno_cache upfront build: '
          f'{_time_cc.perf_counter() - _t_spno:.2f}s '
          f'(pairs={len(_all_pair_keys)})', flush=True)

    # Phase 2d: snapshot the upfront S_pno_cache into a flat
    # pair-of-pair buffer.  Any subsequent miss inside `_s_pno_getter`
    # lands in the overflow dict — the primary tier remains contiguous,
    # which is what Phase 4's Cython kernels consume via
    # ``.buffer`` / ``.offsets`` / ``.index_matrix()``.
    from pyscf.cc.dlpno_tccsd.pair_index import FlatPairPairStore
    S_pno_cache = FlatPairPairStore(_pair_index, initial=S_pno_cache)
    print(f'  [S_pno_cache] {S_pno_cache!r}', flush=True)

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

        # Phase 2c: _pair_index and flat t2_pno_all / K_pno_cache are
        # already built above (outside the bootstrap loop) — just use
        # them here.  build_t1_cache is pulled in per-cycle.
        from pyscf.cc.dlpno_tccsd.pair_index import build_t1_cache

        print(f'  [STAGE5_DBG] post-cc_ints setup before cycle loop: '
              f'{_time_cc.perf_counter() - _t_post_ccints:.2f}s', flush=True)
        _t_loop_start = _time_cc.perf_counter()
        for cycle in range(this_max):
            # Phase 1: pre-project t1 into every pair's PNO basis once
            # per cycle, replacing ~1.5 M lazy _project_t1_to_pair calls.
            # ``_t1_cache[key]`` is a (nocc, n_pno[key]) matrix;
            # ``_t1_cache[key][k]`` is the t1_k projection into pair key.
            _t1_cache = build_t1_cache(
                t1_pno, _pair_index, S_pno_cache, pno_spaces)

            # CCSD MONO real-data validation hook (env-gated; exits after).
            if cycle == 0 and int(os.environ.get(
                    'DLPNO_CCSD_MONO_TEST', '0')):
                # Inject non-zero t1 so validators exercise t1-dependent
                # terms (at cycle 0 t1=0 trivializes most dressing math).
                if int(os.environ.get('DLPNO_CCSD_MONO_INJECT_T1', '1')):
                    import numpy as _np_inj
                    _rng_inj = _np_inj.random.default_rng(99999)
                    for _ii in range(nocc):
                        if _ii in t1_pno and t1_pno[_ii].size > 0:
                            t1_pno[_ii] = _rng_inj.standard_normal(
                                t1_pno[_ii].size) * 0.01
                    # Rebuild t1_cache from the new t1_pno.
                    _t1_cache = build_t1_cache(
                        t1_pno, _pair_index, S_pno_cache, pno_spaces)
                from pyscf.cc.dlpno_tccsd._ccsd_solver import (
                    validate_t1_ints_real, validate_b_tilde_real,
                    validate_update_amps_and_energy_real,
                    validate_t1_fock_fia_bar_real, validate_t1_fock_real,
                    validate_t1_fock_finalize_real, validate_d_tilde_ph1_real,
                    validate_t1_residual_AC_init_real)
                validate_t1_ints_real(
                    _cc_ints, t1_pno, _t1_cache, pno_spaces, pair_lmo_idx,
                    F_lmo, eps_lmo, fov_pno, nocc, keys_sorted, S_pno_cache)
                validate_b_tilde_real(
                    _cc_ints, t1_pno, _t1_cache, pno_spaces, pair_lmo_idx,
                    F_lmo, eps_lmo, fov_pno, nocc, keys_sorted, S_pno_cache,
                    t2_pno_all)
                validate_update_amps_and_energy_real(
                    _cc_ints, t1_pno, _t1_cache, pno_spaces, pair_lmo_idx,
                    F_lmo, eps_lmo, fov_pno, nocc, keys_sorted, S_pno_cache,
                    t2_pno_all, K_pno_cache=K_pno_cache)
                validate_t1_fock_fia_bar_real(
                    _cc_ints, t1_pno, _t1_cache, pno_spaces, pair_lmo_idx,
                    F_lmo, eps_lmo, fov_pno, nocc, keys_sorted, S_pno_cache,
                    t2_pno_all)
                import numpy as _np_mono
                _foo_zero = _np_mono.zeros((nocc, nocc), dtype=_np_mono.float64)
                validate_t1_fock_real(
                    _cc_ints, t1_pno, _t1_cache, pno_spaces, pair_lmo_idx,
                    F_lmo, eps_lmo, fov_pno, nocc, keys_sorted, S_pno_cache,
                    t2_pno_all, _foo_zero)
                validate_t1_fock_finalize_real(
                    _cc_ints, t1_pno, _t1_cache, pno_spaces, pair_lmo_idx,
                    F_lmo, eps_lmo, fov_pno, nocc, keys_sorted, S_pno_cache,
                    t2_pno_all, _foo_zero)
                # phase_d_tilde_ph1 / phase_c_tilde_ph1 real-data validation
                # deferred: build_D_tilde_batched / compute_C_tilde_batched
                # return PH1 + PH2 combined; need separate PH1-only reference
                # path (call _d_tilde_ph1_batched_cy / _c_tilde_ph1_batched_cy
                # directly with the wrapper's internal plan).  Synthetic tests
                # already cover PH1 math; real-data confirms only the packer
                # for K_tilde_chem variants (covered by dump checksums above).
                validate_t1_residual_AC_init_real(
                    _cc_ints, t1_pno, _t1_cache, pno_spaces, pair_lmo_idx,
                    F_lmo, eps_lmo, fov_pno, nocc, keys_sorted, S_pno_cache,
                    t2_pno_all)
                # c-collapse-1/2 orchestration parity test.
                from pyscf.cc.dlpno_tccsd._ccsd_solver import (
                    validate_run_one_cycle_real)
                validate_run_one_cycle_real(
                    _cc_ints, t1_pno, _t1_cache, pno_spaces, pair_lmo_idx,
                    F_lmo, eps_lmo, fov_pno, nocc, keys_sorted, S_pno_cache,
                    t2_pno_all)
                # c-collapse-3 (per_kl wiring) — full R1 vs Python.
                from pyscf.cc.dlpno_tccsd._ccsd_solver import (
                    validate_run_one_cycle_with_per_kl)
                validate_run_one_cycle_with_per_kl(
                    _cc_ints, t1_pno, _t1_cache, pno_spaces, pair_lmo_idx,
                    F_lmo, eps_lmo, fov_pno, nocc, keys_sorted, S_pno_cache,
                    t2_pno_all, _cc_ints_flat, _pair_index, ovL_pno_cache)
                import sys as _sys
                _sys.exit(0)

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
            _compute_t1_residual_psi4._per_i_dump_timing = (cycle == 5)

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
                    t2_pno_all, pno_spaces, nocc, _cc_ints,
                    _pool=(_fine_pool or _pool))
                foo_bare = foo_total
                _t_foo_done = _time.perf_counter()
            else:
                _t_ovl = _t_kcoul = _t_foo = _time.perf_counter()
                foo_total = _compute_foo_dressed_local(
                    t2_pno_all, pno_spaces, nocc, _cc_ints,
                    _pool=(_fine_pool or _pool))
                foo_bare = foo_total
                _t_ovl_done = _t_kcoul_done = _t_foo_done = _time.perf_counter()

            # ---- T1-dressed intermediates (precomputed once per iteration) ----
            _jiang_cache = None
            from pyscf.cc.dlpno_tccsd.residual import (
                build_G_tilde, build_mixed_domain_integrals,
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

            # compute_all_df_terms_local is skipped: for every pair covered
            # by cc_ints (all pairs for non-CAS geometries like water10),
            # downstream C_tilde / D_tilde take their "cc_ints branch" and
            # ignore _term2_precomputed. fvv_t1_all and ladder_all were
            # already unused. Pass empty dicts as the fallback is only
            # hit for pairs missing from cc_ints — zero on this path.
            _c_t2_pre = {}
            _d_t2_pre = {}
            _tj_df = 0.0

            # compute_C_tilde, build_D_tilde, t1_fock each dispatch their
            # own ~400 tasks via _pool.map. Running them in 3 driver
            # threads concurrently contends for the shared 32-worker pool
            # (task-queue lock, cache lines on shared caches). Running
            # them sequentially — each grabbing the full pool for its
            # own pass — avoids contention while keeping per-kernel
            # parallelism intact.
            from pyscf.cc.dlpno_tccsd.local_df import t1_fock
            # compute_C_tilde_psi4 is a line-by-line Psi4 reference port
            # (slow Python, used for cross-validation against the upcoming
            # C port). compute_C_tilde_batched is the production fast path.
            # Default: batched. Opt into Psi4-Python via DLPNO_C_TILDE_PSI4=1.
            _use_psi4_C = (os.environ.get('DLPNO_C_TILDE_PSI4', '0') == '1')
            if _use_psi4_C:
                from pyscf.cc.dlpno_tccsd.residual import compute_C_tilde_psi4 as _cC_fn
            else:
                from pyscf.cc.dlpno_tccsd.residual import compute_C_tilde_batched as _cC_fn
                _cC_fn._dump_timing = (cycle == 5)

            _tj_c0 = _time.perf_counter()
            _jiang_C = _cC_fn(
                t1_pno, t2_pno_all, pno_spaces, nocc,
                ovL_pno_cache, ooL_3idx, S_pno_cache, with_df,
                _term2_precomputed=_c_t2_pre,
                cc_ints=_cc_ints,
                pair_lmo_idx=pair_lmo_idx, _pool=(_fine_pool or _pool),
                S_pao_full=S_pao_full, s1e=s1e,
                blas_threads=32, omp_threads=ncores,
                t1_cache=_t1_cache)
            _tj_C = _time.perf_counter() - _tj_c0

            _tj_d0 = _time.perf_counter()
            _use_psi4_D = (os.environ.get('DLPNO_D_TILDE_PSI4', '0') == '1')
            if _use_psi4_D:
                from pyscf.cc.dlpno_tccsd.residual import build_D_tilde_psi4 as _bD_fn
            else:
                from pyscf.cc.dlpno_tccsd.residual import build_D_tilde_batched as _bD_fn
            _jiang_D = _bD_fn(
                t1_pno, t2_pno_all, pno_spaces, nocc,
                ovL_pno_cache, ooL_3idx, S_pno_cache, with_df,
                _term2_precomputed=_d_t2_pre,
                cc_ints=_cc_ints,
                pair_lmo_idx=pair_lmo_idx, _pool=(_fine_pool or _pool),
                S_pao_full=S_pao_full, s1e=s1e,
                t1_cache=_t1_cache, omp_threads=ncores)
            _tj_D = _time.perf_counter() - _tj_d0

            _tj_f0 = _time.perf_counter()
            _local_Fkj, _local_df_Fab, _local_foo_t1, _local_Fij_bar = t1_fock(
                _cc_ints, None, t1_pno, fov_pno, pno_spaces,
                S_pno_cache, F_lmo, eps_lmo, foo_total,
                _all_keys_j, nocc, _pool=(_fine_pool or _pool),
                pair_lmo_idx=pair_lmo_idx, t1_cache=_t1_cache)
            # Keys whose d_ij/d_ji additions are already baked into
            # _local_Fij_bar — T1 residual augments only the remaining
            # (weak) pairs to avoid double-counting.
            _local_Fij_bar_keys = set(k for k in _all_keys_j
                                       if _cc_ints.get(k) is not None)
            _tj_Fab = _time.perf_counter() - _tj_f0

            _tj_km0 = _time.perf_counter()
            _jiang_K_mixed = build_mixed_domain_integrals(
                t2_pno_all, pno_spaces, nocc,
                ovL_pno_cache, ooL_3idx, S_pno_cache,
                cc_ints=_cc_ints)
            _tj_Km = _time.perf_counter() - _tj_km0

            _jiang_J_oo_d = None

            _tj_g0 = _time.perf_counter()
            _use_psi4_G = (os.environ.get('DLPNO_G_TILDE_PSI4', '0') == '1')
            if _use_psi4_G:
                from pyscf.cc.dlpno_tccsd.residual import build_G_tilde_psi4 as _bG_fn
            else:
                _bG_fn = build_G_tilde
            _local_df_G = _bG_fn(
                t2_pno_all, t1_pno, pno_spaces, nocc,
                ovL_pno_cache, ooL_3idx, S_pno_cache,
                _local_Fkj, _local_foo_t1,
                cc_ints=_cc_ints,
                S_pao_full=S_pao_full, s1e=s1e,
                _pool=(_fine_pool or _pool))
            _g_future = None
            _tj_FG = _time.perf_counter() - _tj_g0

            _t_jiang_done = _time.perf_counter()

            _jiang_cache = {
                'C_tilde': _jiang_C, 'D_tilde': _jiang_D,
                'G_tilde': None,
                'ovL_dressed': None,
                'ooL_dressed': None,
                'J_oo_d': None,
                'K_mixed': _jiang_K_mixed,
                'Fab_all': None,
                'ladder_all': None,
            }

            _t_pairs = _time.perf_counter()

            # --- Batched B + E contributions across all strong pairs ---
            # Hoist T1-dressing of DF integrals OUT of _update_pair: call
            # t1_ints once for ALL strong pairs at iteration start, then
            # compute_B_tilde and _update_pair both read from the shared
            # `_t1_dressed_all` dict (matches Psi4 ccsd.cc:2066-2067 where
            # t1_ints() runs once and populates i_Qk_t1_/i_Qa_t1_).
            from pyscf.cc.dlpno_tccsd.local_df import (
                t1_ints as _t1_ints_all,
                compute_B_tilde as _cB_fn,
                compute_ladder as _cL_fn,
            )
            _bt_pool = _fine_pool or _pool
            if int(os.environ.get('DLPNO_CCSD_MONO_DROPIN_T1_INTS', '0')):
                from pyscf.cc.dlpno_tccsd._ccsd_solver import (
                    t1_ints_via_class)
                _t1_dressed_all = t1_ints_via_class(
                    _cc_ints, t1_pno, pno_spaces, S_pno_cache,
                    keys_sorted, nocc,
                    pair_lmo_idx=pair_lmo_idx, t1_cache=_t1_cache,
                    fov_pno=fov_pno, F_lmo=F_lmo, eps_lmo=eps_lmo,
                    t2_pno_all=t2_pno_all)
            else:
                _t1_dressed_all = _t1_ints_all(
                    _cc_ints, t1_pno, pno_spaces, S_pno_cache,
                    keys_sorted, nocc,
                    pair_lmo_idx=pair_lmo_idx, t1_cache=_t1_cache,
                    _pool=_bt_pool)

            # B_tilde uses the shared dressed dict (no recomputation).
            def _bt_one(_key):
                return _key, _cB_fn(_cc_ints, _t1_dressed_all,
                                    t2_pno_all, t1_pno,
                                    pno_spaces, S_pno_cache, _key, nocc,
                                    pair_lmo_idx=pair_lmo_idx,
                                    t1_cache=_t1_cache)
            _t_bt0 = _time.perf_counter()
            if int(os.environ.get('DLPNO_CCSD_MONO_DROPIN_B_TILDE', '0')):
                from pyscf.cc.dlpno_tccsd._ccsd_solver import (
                    b_tilde_via_class)
                _B_tilde_per_ij = b_tilde_via_class(
                    _cc_ints, _t1_dressed_all, t2_pno_all, t1_pno,
                    pno_spaces, S_pno_cache, keys_sorted, nocc,
                    pair_lmo_idx, _t1_cache,
                    F_lmo=F_lmo, eps_lmo=eps_lmo, fov_pno=fov_pno)
            else:
                _B_tilde_per_ij = {}
                if _bt_pool is not None:
                    for _k, _bt in _bt_pool.map(_bt_one, keys_sorted):
                        _B_tilde_per_ij[_k] = _bt
                else:
                    for _k in keys_sorted:
                        _, _bt = _bt_one(_k)
                        _B_tilde_per_ij[_k] = _bt
            _t_bt = _time.perf_counter() - _t_bt0

            # Hoist compute_ladder similarly: per-pair calls run via the
            # same pool here, so _update_pair just looks up.
            def _ladder_one(_key):
                return _key, _cL_fn(_cc_ints, t2_pno_all, t1_pno,
                                    pno_spaces, S_pno_cache, _key, nocc,
                                    pair_lmo_idx=pair_lmo_idx,
                                    t1_cache=_t1_cache)
            _ladder_all = {}
            if _bt_pool is not None:
                for _k, _l in _bt_pool.map(_ladder_one, keys_sorted):
                    _ladder_all[_k] = _l
            else:
                for _k in keys_sorted:
                    _, _l = _ladder_one(_k)
                    _ladder_all[_k] = _l

            # Batched B+E across all strong pairs via plan-cached Cython
            # kernel (be_kernel).  Output matches the reference to FP
            # reordering noise (~3e-17); validated side-by-side over
            # many iterations prior to cutover.
            from pyscf.cc.dlpno_tccsd.residual import compute_B_E_batched_v2
            _t_be0 = _time.perf_counter()
            _B_dict, _E_dict = compute_B_E_batched_v2(
                keys_sorted, t2_pno_all, pno_spaces, S_pno_cache,
                _cc_ints, _B_tilde_per_ij, pair_lmo_idx, nocc,
                _pool=(_fine_pool or _pool),
                S_pao_full=S_pao_full, s1e=s1e,
                omp_threads=ncores)
            _BE_all = {'B': _B_dict, 'E': _E_dict}
            _t_be = _time.perf_counter() - _t_be0

            # Batched C and D contractions (Phase 5e).  Precomputes the
            # per-pair C_term and D_term tiles once, across all (ij, k)
            # items at once, bypassing the per-pair k-loop inside
            # compute_residual_v2. Lever-C pilot of inlining these (2026-04-24)
            # regressed water10 CCSD 128.5→139.5s (+8.6%): the Cython
            # c_kernel/d_kernel beats the Python inline k-loop despite the
            # extra gather/scatter.  Keep the batched path.
            from pyscf.cc.dlpno_tccsd.residual import compute_CD_terms_batched
            _t_cd0 = _time.perf_counter()
            _C_term_all, _D_term_all = compute_CD_terms_batched(
                keys_sorted, t2_pno_all, pno_spaces, S_pno_cache,
                _cc_ints, _jiang_C, _jiang_D,
                _jiang_K_mixed, K_coul_cache,
                pair_lmo_idx, nocc,
                S_pao_full=S_pao_full, s1e=s1e, omp_threads=ncores)
            _t_cd = _time.perf_counter() - _t_cd0

            # Batched G_term: moved out of per-pair residual (profile
            # showed the per-k inner loop in compute_residual_v2 was
            # 9s CPU/iter at water10, ~82% of residual CPU).  Building
            # a single plan bucketed by (n_ij, n_ik) across all strong
            # pairs turns ~30 small matmuls per pair per iter into a
            # handful of large batched matmuls.  Plan is cached across
            # iterations (static structure).
            # Join the backgrounded build_G_tilde before gterm needs it.
            # With the overlap, G's 0.62s/iter serial work runs concurrently
            # with bt + be + cd above, so its wall cost is absorbed.
            if _g_future is not None:
                _g_wait0 = _time.perf_counter()
                _local_df_G = _g_future.result()
                _tj_FG_wait = _time.perf_counter() - _g_wait0
                _tj_FG += _tj_FG_wait

            from pyscf.cc.dlpno_tccsd.residual import compute_G_term_batched
            _t_g0 = _time.perf_counter()
            _G_term_all = compute_G_term_batched(
                keys_sorted, t2_pno_all, pno_spaces, S_pno_cache,
                (_local_df_G if _local_df_G is not None
                 else jc['G_tilde']),
                pair_lmo_idx, nocc,
                S_pao_full=S_pao_full, s1e=s1e,
                _pool=_pool)
            _t_g = _time.perf_counter() - _t_g0

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
                    # Read from hoisted shared dicts (no per-pair t1_ints /
                    # compute_ladder calls — done once at iteration start).
                    _dressed_entry = _t1_dressed_all.get(key)
                    if _dressed_entry is not None:
                        _K_dressed_local = (_dressed_entry['i_Qa_t1'].T
                                            @ _dressed_entry['j_Qa_t1'])
                    _B_tilde_local = _B_tilde_per_ij.get(key)
                    _ladder_local = _ladder_all.get(key)
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
                    C_term_override=_C_term_all.get(key),
                    D_term_override=_D_term_all.get(key),
                    G_term_override=_G_term_all.get(key),
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
            _t1_resid_start = _time.perf_counter()
            r1_pno = _compute_t1_residual_psi4(
                t1_pno, t2_pno_all, pno_spaces, _t1_fov, F_lmo, eps_lmo, nocc,
                S_pno_cache, _cc_ints, ovL_pno_cache=_t1_ovL,
                pair_lmo_idx=pair_lmo_idx, t1_cache=_t1_cache, _pool=_pool,
                Fij_bar_precomputed=_local_Fij_bar,
                Fij_bar_precomputed_keys=_local_Fij_bar_keys,
                cc_ints_flat=_cc_ints_flat, pair_index=_pair_index)
            _t1_resid_dt = _time.perf_counter() - _t1_resid_start
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

            # CCSD MONO full-cycle validation hook (env-gated; exits after).
            # Hook runs AFTER cycle 0's T1+T2 update but BEFORE DIIS.  At
            # this point: r1_pno, r2_all, t1_pno (=t1_pno_new), t2_new,
            # t1_pno_old all in scope.  t2_pno_all is still OLD (DIIS at
            # line ~2750 hasn't run).
            if cycle == 0 and int(os.environ.get(
                    'DLPNO_CCSD_MONO_TEST_FULL', '0')):
                from pyscf.cc.dlpno_tccsd._ccsd_solver import (
                    validate_run_one_cycle_full_with_external_R2)
                validate_run_one_cycle_full_with_external_R2(
                    _cc_ints, t1_pno_old, t1_pno, t2_pno_all, t2_new,
                    r1_pno, r2_all,
                    pno_spaces, pair_lmo_idx, F_lmo, eps_lmo, fov_pno,
                    nocc, keys_sorted, S_pno_cache, _cc_ints_flat,
                    _pair_index, ovL_pno_cache, K_pno_cache,
                    g_tilde_pyscf=_local_df_G)
                import sys as _sys
                _sys.exit(0)

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
            _diis_start = _time.perf_counter()
            if cycle >= diis_start_cycle and err_vec.size > 0:
                amp_new = mydiis.update(amp_new, err_vec)
            _diis_dt = _time.perf_counter() - _diis_start

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
            _energy_start = _time.perf_counter()
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
            _energy_dt = _time.perf_counter() - _energy_start
            _t_cycle_end = _time.perf_counter()
            _dt_foo = _t_foo_done - _t_foo
            _dt_pairs = _t_pairs_done - _t_pairs
            _dt_jiang = _t_jiang_done - _t_jiang
            _dt_total = _t_cycle_end - _t_cycle_start
            _upd_rest = _dt_pairs - _t_bt - _t_be - _t_cd - _t_g
            _other_dt = (_dt_total - _dt_foo - _dt_jiang - _dt_pairs
                         - _t1_resid_dt - _diis_dt - _energy_dt)
            print(f'  Cycle {cycle + 1:3d}: dT = {dT:.3e}  '
                  f'E_corr = {e_cyc:.10f}  dE = {dE:.2e}  '
                  f'[{_dt_total:.1f}s: foo={_dt_foo:.2f} '
                  f'jiang={_dt_jiang:.2f}'
                  f'(C={_tj_C:.2f} D={_tj_D:.2f} Fock={_tj_Fab:.2f} '
                  f'Km={_tj_Km:.2f} G={_tj_FG:.2f}) '
                  f'pairs={_dt_pairs:.2f}(bt={_t_bt:.2f} be={_t_be:.2f} '
                  f'cd={_t_cd:.2f} gterm={_t_g:.2f} upd={_upd_rest:.2f}) '
                  f't1r={_t1_resid_dt:.2f} diis={_diis_dt:.2f} '
                  f'eng={_energy_dt:.2f} oth={_other_dt:.2f}]',
                  flush=True)
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

    if _fine_pool is not None:
        _fine_pool.shutdown(wait=True)

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


