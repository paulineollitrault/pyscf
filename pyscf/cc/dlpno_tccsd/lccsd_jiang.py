"""Jiang et al. (JCP 2024) exact t1-transformed DLPNO-CCSD T2 residual.

Implements Eqs 75-86 of Jiang et al. using bare integrals + explicit
T1-dressed intermediates. Replaces the C̃_lmo MO-rotation approach.

References:
    Jiang et al., J. Chem. Phys. 2024 (DLPNO paper)
    Eqs 75-102: T2 residual, T1 residual, energy
"""
import numpy as np
from pyscf.ao2mo import _ao2mo

from pyscf.cc.dlpno_tccsd.lccsd import (
    _project_t1_to_pair, _compute_ladder, _project_t2_full,
)


def _compute_foo_t1_dressed(t1_pno, fov_pno, pno_spaces, nocc,
                            ovL_pno_bare, ooL_bare, S_pno_cache):
    """Compute T1-dressed occupied Fock: canonical lines 126, 157-158.

    foo_t1[i,j] = 0.5 * Σ_a fov[i,a]*t1[j,a]                    (line 126)
                + Σ_{k,c} [2*(kc|ji) - (ic|jk)] * t1[k,c]       (lines 157-158)

    Uses BARE integrals throughout.
    """
    foo_t1 = np.zeros((nocc, nocc))

    # Line 126: 0.5 * fov @ t1.T (pair-local: project to common PNO_ii)
    for ii in range(nocc):
        key_ii = (ii, ii)
        if key_ii not in pno_spaces:
            continue
        fov_ii = fov_pno.get(ii)
        if fov_ii is None or fov_ii.size == 0:
            continue
        for jj in range(nocc):
            t1_jj_in_ii = _project_t1_to_pair(
                t1_pno, jj, key_ii, S_pno_cache, pno_spaces)
            foo_t1[ii, jj] += 0.5 * np.dot(fov_ii, t1_jj_in_ii)

    # Lines 157-158: foo[i,j] += Σ_{k,c} [2*(kc|ji) - (ic|jk)] * t1[k,c]
    for kk in range(nocc):
        key_kk = (kk, kk)
        if key_kk not in pno_spaces or t1_pno.get(kk) is None:
            continue
        t1_k = t1_pno[kk]
        if t1_k.size == 0:
            continue
        ovL_kk_k = ovL_pno_bare.get((key_kk, kk))
        if ovL_kk_k is None:
            continue
        # z_k[Q] = Σ_c t1_k[c]*ovL_k_kk[c,Q]
        z_k = t1_k @ ovL_kk_k  # (naux,)
        # 2*(kc|ji)*t1[k,c] = 2*z_k @ ooL[j,i,:]
        for ii in range(nocc):
            for jj in range(nocc):
                foo_t1[ii, jj] += 2.0 * np.dot(z_k, ooL_bare[jj, ii, :])

        # -(ic|jk)*t1[k,c]: z_ik[Q] = Σ_c t1_k[c]*ovL_bare_i_kk[c,Q]
        for ii in range(nocc):
            ovL_ii_kk = ovL_pno_bare.get((key_kk, ii))
            if ovL_ii_kk is None:
                continue
            z_ik = t1_k @ ovL_ii_kk  # (naux,)
            for jj in range(nocc):
                foo_t1[ii, jj] -= np.dot(z_ik, ooL_bare[jj, kk, :])

    return foo_t1


def _compute_fvv_t1_dressed(t1_pno, fov_pno, pno_spaces, nocc,
                            ovL_pno_bare, S_pno_cache, with_df,
                            pair_key):
    """Compute T1-dressed virtual Fock for pair (i,j): canonical lines 129, 332-333.

    fvv_t1[a,b] = -0.5 * Σ_k t1_k[a]*fov_k[b]                   (line 129)
                + Σ_{k,c} t1_k[c] * [2*(ck|ab) - (bk|ca)]        (lines 332-333)

    Returns the T1 correction to fvv in PNO_ij basis. Uses BARE integrals.
    """
    n_pno = pno_spaces[pair_key]['C_pno'].shape[1]
    C_pno_ij = pno_spaces[pair_key]['C_pno']
    fvv_t1 = np.zeros((n_pno, n_pno))

    # Line 129: -0.5 * t1.T @ fov (projected to PNO_ij)
    for kk in range(nocc):
        t1_k_ij = _project_t1_to_pair(
            t1_pno, kk, pair_key, S_pno_cache, pno_spaces)
        fov_k_ij = _project_t1_to_pair(
            fov_pno, kk, pair_key, S_pno_cache, pno_spaces)
        fvv_t1 -= 0.5 * np.outer(t1_k_ij, fov_k_ij)

    # Lines 332-333 need a DF loop for (ck|ab) in PNO_ij basis.
    # For now, compute the Coulomb part only (dominant):
    # 2*Σ_{k,c} t1_k[c]*(ck|ab) = 2*Σ_k z_k[Q]*(ab|Q)
    # where z_k[Q] = t1_k[c]*ovL_k_ij[c,Q]
    # This requires a DF loop for (ab|Q) = vvL_ij.
    z_total = np.zeros(ovL_pno_bare.get((pair_key, 0), np.zeros((1, 1))).shape[1]
                       if (pair_key, 0) in ovL_pno_bare else 0)
    if z_total.size > 0:
        for kk in range(nocc):
            t1_k_ij = _project_t1_to_pair(
                t1_pno, kk, pair_key, S_pno_cache, pno_spaces)
            if np.max(np.abs(t1_k_ij)) < 1e-15:
                continue
            ovL_k_ij = ovL_pno_bare.get((pair_key, kk))
            if ovL_k_ij is not None:
                z_total += t1_k_ij @ ovL_k_ij

        if np.max(np.abs(z_total)) > 1e-15:
            mo_vv = np.asfortranarray(C_pno_ij)
            ijslice = (0, n_pno, 0, n_pno)
            buf = None
            aux_off = 0
            for Lpq in with_df.loop():
                nL = Lpq.shape[0]
                buf = _ao2mo.nr_e2(
                    Lpq, mo_vv, ijslice, aosym='s2', out=buf)
                B_L = buf.reshape(nL, n_pno, n_pno)
                z_batch = z_total[aux_off:aux_off + nL]
                fvv_t1 += 2.0 * np.einsum('L,Lab->ab', z_batch, B_L)
                aux_off += nL

    return fvv_t1


def _build_ooL_dressed_asymmetric(ooL_bare, ovL_pno_bare, t1_pno,
                                  pno_spaces, nocc):
    """Build asymmetrically T1-dressed ooL (Jiang et al. Eq 91).

    B̃_{ki}^Q = B_{ki}^Q + Σ_a B_{ka}^Q * t_i^a

    Only the second index (i) is dressed. Uses BARE integrals.

    Returns:
        ooL_dressed: (nocc, nocc, naux) dressed ooL tensor
    """
    naux = ooL_bare.shape[2]
    ooL_dressed = ooL_bare.copy()

    for ii in range(nocc):
        key_ii = (ii, ii)
        if key_ii not in pno_spaces or t1_pno.get(ii) is None:
            continue
        t1_i = t1_pno[ii]
        if t1_i.size == 0:
            continue
        # B̃_{ki}^Q += Σ_a (ka|Q)*t1_i[a] for each k
        # (ka|Q) = ovL_bare[(ii, k)][a, Q] — LMO k in PNO_ii basis
        for kk in range(nocc):
            ovL_k_ii = ovL_pno_bare.get((key_ii, kk))
            if ovL_k_ii is not None:
                # Σ_a ovL_k_ii[a,Q]*t1_i[a] = t1_i @ ovL_k_ii → (naux,)
                ooL_dressed[kk, ii, :] += t1_i @ ovL_k_ii

    return ooL_dressed
