"""Jiang et al. (JCP 2024) exact t1-transformed DLPNO-CCSD T2 residual.

Implements Eqs 75-86 of Jiang et al. using bare integrals + explicit
T1-dressed intermediates. Replaces the C̃_lmo MO-rotation approach.

All integrals (ovL_pno_cache, ooL_3idx, K_coul_cache, J_oo) must be
BARE (computed from undressed C_lmo). T1 effects enter exclusively
through the dressed intermediates built here.

References:
    Jiang et al., J. Chem. Phys. 2024 (DLPNO paper)
    Eqs 75-102: T2 residual, T1 residual, energy
"""
import numpy as np
from pyscf.ao2mo import _ao2mo

from pyscf.cc.dlpno_tccsd.lccsd import (
    _project_t1_to_pair, _compute_ladder, _project_t2_full,
)


# ---------------------------------------------------------------------------
# Dressed DF integral builders (Eqs 91-93)
# ---------------------------------------------------------------------------

def _build_ooL_dressed(ooL_bare, ovL_pno_bare, t1_pno, pno_spaces, nocc):
    """Asymmetric T1-dressed ooL (Eq 91): B̃_{ki} = B_{ki} + B_{ka}·t_i^a.

    Only the second index (i) is dressed.
    """
    ooL_dressed = ooL_bare.copy()
    for ii in range(nocc):
        key_ii = (ii, ii)
        if key_ii not in pno_spaces:
            continue
        t1_i = t1_pno.get(ii)
        if t1_i is None or t1_i.size == 0:
            continue
        for kk in range(nocc):
            ovL_k_ii = ovL_pno_bare.get((key_ii, kk))
            if ovL_k_ii is not None:
                ooL_dressed[kk, ii, :] += t1_i @ ovL_k_ii
    return ooL_dressed


def build_dressed_ovL_cache(ovL_pno_bare, ooL_bare, t1_pno, pno_spaces,
                            S_pno_cache, nocc, with_df, keys):
    """Build dressed ovL cache for ALL (pair, LMO) combinations.

    B̃_{mi}^Q = B_{mi}^Q + Σ_b B_{mb_ij}^Q · t̃_i^{b_ij} - Σ_k t̃_k^{a_ij} · B_{ki}^Q

    Applies Eq 92 Terms 2-3 (linear in T1) to each ovL entry.
    Matches Psi4's i_Qa_t1_ construction (lines 1494-1495).

    Returns dict same format as ovL_pno_bare but with dressed values.
    """
    ovL_dressed = {}

    for key in keys:
        n_pno = pno_spaces[key]['C_pno'].shape[1]
        if n_pno == 0:
            continue
        C_pno_ij = pno_spaces[key]['C_pno']

        # Build T1_all projected to this pair's PNO basis
        T1_all = np.zeros((nocc, n_pno))
        for kk in range(nocc):
            T1_all[kk] = _project_t1_to_pair(
                t1_pno, kk, key, S_pno_cache, pno_spaces)

        # Term 2: -T1_all.T @ ooL_bare[:,m,:] for each m
        # Precompute for all m at once
        # delta_oo[m][a,Q] = -Σ_k T1_all[k,a]*ooL_bare[k,m,Q]
        # = -(T1_all.T @ ooL_bare[:,m,:]) for each m

        # Term 3: +vvL_ij × T1_m needs DF loop
        # Compute Σ_b (ab|Q)*T1_m^b for all m and all Q at once
        naux = ooL_bare.shape[2]

        # For each m, T1_m projected to PNO_ij:
        t1_proj = {}
        for m in range(nocc):
            t1_proj[m] = _project_t1_to_pair(
                t1_pno, m, key, S_pno_cache, pno_spaces)

        # Compute Term 3 (vvL × T1) via DF loop
        delta_vv = {m: np.zeros((n_pno, naux)) for m in range(nocc)}
        any_nonzero = any(np.max(np.abs(t1_proj[m])) > 1e-15 for m in range(nocc))
        if any_nonzero:
            mo_vv = np.asfortranarray(C_pno_ij)
            ijslice = (0, n_pno, 0, n_pno)
            buf = None
            aux_off = 0
            for Lpq in with_df.loop():
                nL = Lpq.shape[0]
                buf = _ao2mo.nr_e2(Lpq, mo_vv, ijslice, aosym='s2', out=buf)
                B_L = buf.reshape(nL, n_pno, n_pno)
                for m in range(nocc):
                    if np.max(np.abs(t1_proj[m])) > 1e-15:
                        delta_vv[m][:, aux_off:aux_off+nL] = np.einsum(
                            'Lab,b->aL', B_L, t1_proj[m])
                aux_off += nL

        # Build dressed ovL for each m
        for m in range(nocc):
            ovL_bare_entry = ovL_pno_bare.get((key, m))
            if ovL_bare_entry is None:
                continue
            # Term 2: -T1_all.T @ ooL_bare[:,m,:]
            delta_oo = -T1_all.T @ ooL_bare[:, m, :]  # (n_pno, naux)

            # Full dressed = bare + delta_vv(Term 3) + delta_oo(Term 2)
            ovL_dressed[(key, m)] = ovL_bare_entry + delta_vv[m] + delta_oo

    return ovL_dressed


def compute_C_tilde(t1_pno, t2_pno_all, pno_spaces, nocc,
                    ovL_pno_bare, ooL_bare, S_pno_cache, with_df,
                    _term2_precomputed=None, cc_ints=None):
    """Compute C_tilde (gamma intermediate, Eq 83) for all pairs.

    C_tilde[ki][a_ki, c_ki] = gamma_{ki}^{ac} =
        + Σ_b t̃_i^b · (kb|ac)              [Term 2: vovv × T1_i]
        - Σ_l t̃_l^a · (ki|lc)              [Term 1: ooov × T1_all]
        - Σ_l t̃_l^a · (K_kl × T1_i)        [Term 3: T1² × K]
        - (1/2) Σ_l S @ t2_li @ S @ K_kl @ S [Term 4: T2 × K]

    Following Psi4 ccsd.cc compute_C_tilde() lines 1694-1751.

    Returns dict: pair_key → (n_pno_ki, n_pno_ki) matrix.
    """
    C_tilde_all = {}

    # Iterate over ALL ordered (k,i) pairs, not just min/max.
    # gamma_{ki} depends on which LMO is "i" (the T1 source).
    all_pairs = set()
    for key in t2_pno_all:
        k, i = key  # stored as (min, max)
        all_pairs.add((k, i))
        all_pairs.add((i, k))

    # Term 2 can be precomputed via unified DF pass
    if _term2_precomputed is None:
        # Batched Term 2 via single DF pass (fallback)
        term2_data = {}
        for k, i in all_pairs:
            key_ki = (min(k, i), max(k, i))
            n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
            if n_ki == 0:
                continue
            t1_i_ki = _project_t1_to_pair(t1_pno, i, key_ki, S_pno_cache, pno_spaces)
            if np.max(np.abs(t1_i_ki)) < 1e-15:
                continue
            ovL_i_ki = ovL_pno_bare.get((key_ki, i))
            if ovL_i_ki is None:
                continue
            term2_data[(k, i)] = {
                'key_ki': key_ki, 'n_ki': n_ki,
                'C_pno': np.asfortranarray(pno_spaces[key_ki]['C_pno']),
                't1_i': t1_i_ki, 'ovL_i': ovL_i_ki,
                'result': np.zeros((n_ki, n_ki)),
            }
        if term2_data:
            aux_off = 0
            for Lpq in with_df.loop():
                nL = Lpq.shape[0]
                for (k, i), td in term2_data.items():
                    n_ki = td['n_ki']
                    buf = _ao2mo.nr_e2(Lpq, td['C_pno'],
                                       (0, n_ki, 0, n_ki), aosym='s2')
                    B_L = buf.reshape(nL, n_ki, n_ki)
                    z_i = td['t1_i'] @ td['ovL_i'][:, aux_off:aux_off+nL]
                    td['result'] += np.einsum('L,Lac->ac', z_i, B_L)
                aux_off += nL
        _term2_precomputed = {ki: td['result'] for ki, td in term2_data.items()}

    for k, i in all_pairs:
        key_ki = (min(k, i), max(k, i))  # storage key
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        if n_ki == 0:
            continue

        C_tilde_ki = np.zeros((n_ki, n_ki))

        # Term 2: Σ_b t1_i^b · (kb|ac) = Σ_{Q,b} t1_i[b]*k_Qa[Q,b]*Qab[Q,a,c]
        # Use local DF when cc_ints available, otherwise precomputed (global DF)
        if cc_ints is not None and key_ki in cc_ints and cc_ints[key_ki] is not None:
            ci_ki = cc_ints[key_ki]
            if key_ki[0] == k:
                k_Qa = ci_ki['i_Qa']
            else:
                k_Qa = ci_ki['j_Qa']
            Qab_ki = ci_ki['Qab']
            t1_i_ki = _project_t1_to_pair(t1_pno, i, key_ki, S_pno_cache, pno_spaces)
            z = k_Qa @ t1_i_ki
            C_tilde_ki += np.einsum('Q,Qac->ac', z, Qab_ki)
        elif (k, i) in _term2_precomputed:
            C_tilde_ki += _term2_precomputed[(k, i)]

        # --- Term 1: -Σ_l T1_all[l,a] · (ki|lc) ---
        # K_bar_chem[ki][l,c] = Σ_Q ooL[k,i,Q]*ovL_l_ki[c,Q] = (ki|lc)
        T1_all_ki = np.zeros((nocc, n_ki))
        for ll in range(nocc):
            T1_all_ki[ll] = _project_t1_to_pair(
                t1_pno, ll, key_ki, S_pno_cache, pno_spaces)
        K_bar_chem = np.zeros((nocc, n_ki))
        if cc_ints is not None and key_ki in cc_ints and cc_ints[key_ki] is not None:
            from pyscf.cc.dlpno_tccsd.local_df import get_local_ovL, get_local_ooL_vec
            _ooL_ki = get_local_ooL_vec(cc_ints, k, i, key_ki)
            if _ooL_ki is not None:
                for ll in range(nocc):
                    _ovL_l = get_local_ovL(cc_ints, key_ki, ll)
                    if _ovL_l is not None:
                        K_bar_chem[ll] = _ovL_l @ _ooL_ki
            else:
                ooL_ki = ooL_bare[k, i, :]
                for ll in range(nocc):
                    ovL_l_ki = ovL_pno_bare.get((key_ki, ll))
                    if ovL_l_ki is not None:
                        K_bar_chem[ll] = ovL_l_ki @ ooL_ki
        else:
            ooL_ki = ooL_bare[k, i, :]
            for ll in range(nocc):
                ovL_l_ki = ovL_pno_bare.get((key_ki, ll))
                if ovL_l_ki is not None:
                    K_bar_chem[ll] = ovL_l_ki @ ooL_ki
        C_tilde_ki -= T1_all_ki.T @ K_bar_chem  # (n_ki, n_ki)

        # --- Term 3: -Σ_l T1_l^a · (S @ K_kl @ T1_i_kl) ---
        for ll in range(nocc):
            key_kl = (min(k, ll), max(k, ll))
            if key_kl not in pno_spaces:
                continue
            n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
            if n_kl == 0:
                continue
            # K_kl[a_kl, b_kl] = (ka|lb) = exchange in PNO_kl
            if cc_ints is not None:
                from pyscf.cc.dlpno_tccsd.local_df import get_local_K
                _K = get_local_K(cc_ints, key_kl, k, ll)
                if _K is not None:
                    K_kl = _K
                else:
                    ovL_k_kl = ovL_pno_bare.get((key_kl, k))
                    ovL_l_kl = ovL_pno_bare.get((key_kl, ll))
                    if ovL_k_kl is None or ovL_l_kl is None:
                        continue
                    K_kl = ovL_k_kl @ ovL_l_kl.T
            else:
                ovL_k_kl = ovL_pno_bare.get((key_kl, k))
                ovL_l_kl = ovL_pno_bare.get((key_kl, ll))
                if ovL_k_kl is None or ovL_l_kl is None:
                    continue
                K_kl = ovL_k_kl @ ovL_l_kl.T

            # T1_i projected to PNO_kl
            t1_i_kl = _project_t1_to_pair(t1_pno, i, key_kl, S_pno_cache, pno_spaces)
            if np.max(np.abs(t1_i_kl)) < 1e-15:
                continue

            # (K_kl @ T1_i_kl)[a_kl] = Σ_b K_kl[a,b]*T1_i[b]
            Kt1 = K_kl @ t1_i_kl  # (n_kl,)

            # Project Kt1 from PNO_kl to PNO_ki
            S_ki_kl = S_pno_cache.get((key_ki, key_kl))
            if S_ki_kl is None:
                continue
            Kt1_ki = S_ki_kl @ Kt1  # (n_ki,)

            # T1_l projected to PNO_ki
            t1_l_ki = T1_all_ki[ll]

            # C_tilde -= outer(T1_l, Kt1_ki) = T1_l^a * Kt1[c]
            C_tilde_ki -= np.outer(t1_l_ki, Kt1_ki)

        # --- Term 4: -(1/2) Σ_l S @ t2_li @ S @ K_kl @ S ---
        for ll in range(nocc):
            key_li = (min(ll, i), max(ll, i))
            key_kl = (min(k, ll), max(k, ll))
            if key_li not in t2_pno_all or key_kl not in pno_spaces:
                continue
            t2_li_raw = t2_pno_all.get(key_li)
            if t2_li_raw is None or t2_li_raw.shape[0] == 0:
                continue
            n_li = pno_spaces[key_li]['C_pno'].shape[1]
            n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
            if n_li == 0 or n_kl == 0:
                continue

            t2_li = t2_li_raw.T if ll > i else t2_li_raw

            if cc_ints is not None:
                from pyscf.cc.dlpno_tccsd.local_df import get_local_K
                _K = get_local_K(cc_ints, key_kl, k, ll)
                if _K is not None:
                    K_kl = _K
                else:
                    ovL_k_kl = ovL_pno_bare.get((key_kl, k))
                    ovL_l_kl = ovL_pno_bare.get((key_kl, ll))
                    if ovL_k_kl is None or ovL_l_kl is None:
                        continue
                    K_kl = ovL_k_kl @ ovL_l_kl.T
            else:
                ovL_k_kl = ovL_pno_bare.get((key_kl, k))
                ovL_l_kl = ovL_pno_bare.get((key_kl, ll))
                if ovL_k_kl is None or ovL_l_kl is None:
                    continue
                K_kl = ovL_k_kl @ ovL_l_kl.T

            def _get_S_or_I(ka, kb, n_a):
                if ka == kb:
                    return np.eye(n_a)
                S = S_pno_cache.get((ka, kb))
                return S

            n_ki2 = pno_spaces[key_ki]['C_pno'].shape[1]
            S_ki_li = _get_S_or_I(key_ki, key_li, n_ki2)
            S_li_kl = _get_S_or_I(key_li, key_kl, n_li)
            S_kl_ki = _get_S_or_I(key_kl, key_ki, n_kl)
            if S_ki_li is None or S_li_kl is None or S_kl_ki is None:
                continue

            # Psi4: S(ki,li) @ T[li] @ S(li,kl) @ K[kl] @ S(kl,ki)
            C_temp = S_ki_li @ t2_li @ S_li_kl @ K_kl @ S_kl_ki
            C_tilde_ki -= 0.5 * C_temp  # Term 4: Jiang Eq 83

        C_tilde_all[(k, i)] = C_tilde_ki  # store with ORDERED (k,i) key

    return C_tilde_all


def _build_psi_ao(t1_pno, pno_spaces, C_lmo, nocc):
    """Build AO-basis T1 vectors: ψ_i(μ) = Σ_a C_pno_ii(μ,a)*t1_i(a).

    These are used to construct dressed integrals via DF half-transforms.
    """
    nao = C_lmo.shape[0]
    psi = np.zeros((nocc, nao))
    for ii in range(nocc):
        key_ii = (ii, ii)
        if key_ii in pno_spaces:
            t1_i = t1_pno.get(ii)
            if t1_i is not None and t1_i.size > 0:
                psi[ii] = pno_spaces[key_ii]['C_pno'] @ t1_i
    return psi


# ---------------------------------------------------------------------------
# T1-dressed Fock intermediates (Eqs 94-101, 85-86)
# ---------------------------------------------------------------------------

def _compute_foo_t1(t1_pno, fov_pno, pno_spaces, nocc,
                    ovL_pno_bare, ooL_bare, S_pno_cache):
    """T1-dressed occupied Fock correction (Eq 98).

    F̄_{ij} = F_{ij} + [2J-K]_{kc}^{ij} · t̃_k^c  [Eq 98, PySCF lines 157-158]

    The Eq 94 part (F̄_{kc}·t_j^c) is handled in build_Fkj via Fia_bar.
    Returns (nocc, nocc) T1 correction to add to bare foo.
    """
    foo_t1 = np.zeros((nocc, nocc))

    # NOTE: The 0.5*fov·t1 term (PySCF line 126) is NOT used here.
    # In Jiang/Psi4, this contribution enters through Eq 94 (build_Fkj)
    # via Fia_bar @ T1, which is handled separately.

    # Eq 98 / lines 157-158: F̄_{ij} += Σ_{k,c} [2(kc|ji) - (ic|jk)]·t1_k^c
    for kk in range(nocc):
        key_kk = (kk, kk)
        if key_kk not in pno_spaces:
            continue
        t1_k = t1_pno.get(kk)
        if t1_k is None or t1_k.size == 0:
            continue
        ovL_kk_k = ovL_pno_bare.get((key_kk, kk))
        if ovL_kk_k is None:
            continue
        z_k = t1_k @ ovL_kk_k  # (naux,): Σ_c t1_k^c·(kc|Q)
        # Coulomb: 2·Σ_Q z_k[Q]·ooL[j,i,Q]
        for ii in range(nocc):
            ooL_ji = ooL_bare[:, ii, :]  # (nocc, naux)
            foo_t1[ii, :] += 2.0 * (ooL_ji @ z_k)
        # Exchange: -Σ_Q z_ik[Q]·ooL[j,k,Q]
        for ii in range(nocc):
            ovL_ii_kk = ovL_pno_bare.get((key_kk, ii))
            if ovL_ii_kk is None:
                continue
            z_ik = t1_k @ ovL_ii_kk  # (naux,)
            foo_t1[ii, :] -= ooL_bare[:, kk, :] @ z_ik

    return foo_t1


def _compute_fvv_t1_pair(t1_pno, fov_pno, pno_spaces, nocc,
                         ovL_pno_bare, S_pno_cache, with_df, pair_key):
    """T1-dressed virtual Fock F̄_{ab} for pair ij (Eq 101, PySCF lines 332-333).

    Computes [2(ab|kc)-(ac|kb)]·t̃_k^c part of the dressed Fock.
    The Eq 97 part (-Σ_k t̃_k^a·F̄_{kb}) is handled in build_Fab via Fia_bar.
    Returns (n_pno, n_pno) T1 correction to fvv in PNO_ij basis.
    """
    n_pno = pno_spaces[pair_key]['C_pno'].shape[1]
    if n_pno == 0:
        return np.zeros((0, 0))
    C_pno_ij = pno_spaces[pair_key]['C_pno']
    fvv_t1 = np.zeros((n_pno, n_pno))

    # NOTE: The -0.5*t1@fov term (PySCF line 129) is NOT used here.
    # In Jiang/Psi4, this contribution enters through Eq 97 (build_Fab)
    # via -T_n.T @ Fia_bar, which is handled separately.

    # Eq 98 / lines 332-333: Σ_{k,c} t1_k^c·[2(ck|ab) - (bk|ca)]
    # Coulomb: 2·Σ_k (Σ_c t1_k^c·ovL_k[c,Q])·vvL[a,b,Q]
    # z_total[Q] = Σ_k Σ_c t1_k_ij^c · ovL_k_ij[c,Q]
    naux = 0
    for m in range(nocc):
        entry = ovL_pno_bare.get((pair_key, m))
        if entry is not None:
            naux = entry.shape[1]
            break
    if naux == 0:
        return fvv_t1

    z_total = np.zeros(naux)
    for kk in range(nocc):
        t1_k_ij = _project_t1_to_pair(
            t1_pno, kk, pair_key, S_pno_cache, pno_spaces)
        if np.max(np.abs(t1_k_ij)) < 1e-15:
            continue
        ovL_k_ij = ovL_pno_bare.get((pair_key, kk))
        if ovL_k_ij is not None:
            z_total += t1_k_ij @ ovL_k_ij

    # Precompute T1 projections and ovL for exchange part
    t1_proj_all = {}
    ovL_k_all = {}
    for kk in range(nocc):
        t1_k_ij = _project_t1_to_pair(
            t1_pno, kk, pair_key, S_pno_cache, pno_spaces)
        if np.max(np.abs(t1_k_ij)) > 1e-15:
            t1_proj_all[kk] = t1_k_ij
            ovL_k = ovL_pno_bare.get((pair_key, kk))
            if ovL_k is not None:
                ovL_k_all[kk] = ovL_k

    any_nonzero = len(t1_proj_all) > 0
    if any_nonzero:
        mo_vv = np.asfortranarray(C_pno_ij)
        ijslice = (0, n_pno, 0, n_pno)
        buf = None
        aux_off = 0
        for Lpq in with_df.loop():
            nL = Lpq.shape[0]
            buf = _ao2mo.nr_e2(Lpq, mo_vv, ijslice, aosym='s2', out=buf)
            B_L = buf.reshape(nL, n_pno, n_pno)  # (L, c, a) = (ca|L)
            z_batch = z_total[aux_off:aux_off + nL]

            # Coulomb: 2·Σ_k z_k[L]·(ab|L) = 2·z_total[L]·B_L[L,a,b]
            fvv_t1 += 2.0 * np.einsum('L,Lab->ab', z_batch, B_L)

            # Exchange: -Σ_{k,c} T1_k[c]·(bk|ca)
            # (bk|ca) = Σ_L ovL_k[b,L]*B_L[L,c,a]
            # Y_k[a,L] = Σ_c T1_k[c]*B_L[L,c,a]
            # fvv_exch[a,b] -= Σ_k Y_k_batch @ ovL_k_batch.T
            for kk in t1_proj_all:
                if kk not in ovL_k_all:
                    continue
                Y_k = np.einsum('c,Lca->aL', t1_proj_all[kk], B_L)
                ovL_k_batch = ovL_k_all[kk][:, aux_off:aux_off + nL]
                fvv_t1 -= Y_k @ ovL_k_batch.T

            aux_off += nL

    return fvv_t1


def compute_fvv_t1_all_pairs(t1_pno, fov_pno, pno_spaces, nocc,
                              ovL_pno_bare, S_pno_cache, with_df,
                              pair_keys):
    """Batched fvv_t1 for ALL pairs in a single DF pass.

    Same result as calling _compute_fvv_t1_pair per pair, but reads the
    DF integrals only once instead of N_pairs times.

    Returns dict: pair_key -> (n_pno, n_pno) fvv_t1 matrix.
    """
    # Precompute per-pair data: z_total, t1_projections, ovL, C_pno
    pair_data = {}
    for pk in pair_keys:
        n_pno = pno_spaces[pk]['C_pno'].shape[1]
        if n_pno == 0:
            continue
        C_pno_ij = pno_spaces[pk]['C_pno']

        # Get naux
        naux = 0
        for m in range(nocc):
            entry = ovL_pno_bare.get((pk, m))
            if entry is not None:
                naux = entry.shape[1]
                break
        if naux == 0:
            continue

        # z_total[Q] = Σ_k Σ_c t1_k^c · ovL_k[c,Q]
        z_total = np.zeros(naux)
        t1_proj_all = {}
        ovL_k_all = {}
        for kk in range(nocc):
            t1_k_ij = _project_t1_to_pair(
                t1_pno, kk, pk, S_pno_cache, pno_spaces)
            if np.max(np.abs(t1_k_ij)) < 1e-15:
                continue
            ovL_k_ij = ovL_pno_bare.get((pk, kk))
            if ovL_k_ij is not None:
                z_total += t1_k_ij @ ovL_k_ij
                t1_proj_all[kk] = t1_k_ij
                ovL_k_all[kk] = ovL_k_ij

        if len(t1_proj_all) == 0:
            continue

        pair_data[pk] = {
            'n_pno': n_pno,
            'C_pno': np.asfortranarray(C_pno_ij),
            'z_total': z_total,
            't1_proj': t1_proj_all,
            'ovL_k': ovL_k_all,
            'fvv_t1': np.zeros((n_pno, n_pno)),
        }

    if not pair_data:
        return {pk: np.zeros((pno_spaces[pk]['C_pno'].shape[1],) * 2)
                for pk in pair_keys}

    # Single DF pass: loop over batches, process all pairs per batch
    aux_off = 0
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]

        for pk, pd in pair_data.items():
            n_pno = pd['n_pno']
            ijslice = (0, n_pno, 0, n_pno)
            buf = _ao2mo.nr_e2(Lpq, pd['C_pno'], ijslice, aosym='s2')
            B_L = buf.reshape(nL, n_pno, n_pno)

            # Coulomb
            z_batch = pd['z_total'][aux_off:aux_off + nL]
            pd['fvv_t1'] += 2.0 * np.einsum('L,Lab->ab', z_batch, B_L)

            # Exchange
            for kk, t1_k in pd['t1_proj'].items():
                ovL_k = pd['ovL_k'].get(kk)
                if ovL_k is None:
                    continue
                Y_k = np.einsum('c,Lca->aL', t1_k, B_L)
                ovL_k_batch = ovL_k[:, aux_off:aux_off + nL]
                pd['fvv_t1'] -= Y_k @ ovL_k_batch.T

        aux_off += nL

    # Build result dict
    result = {}
    for pk in pair_keys:
        n_pno = pno_spaces[pk]['C_pno'].shape[1]
        if pk in pair_data:
            result[pk] = pair_data[pk]['fvv_t1']
        else:
            result[pk] = np.zeros((n_pno, n_pno))
    return result


# ---------------------------------------------------------------------------
# Complete Jiang et al. T2 residual (Eqs 75-81)
# ---------------------------------------------------------------------------

def compute_pair_residual_jiang(
        i, j, t2_pno_all, pno_spaces, nocc_lmo,
        F_lmo, s1e, with_df, C_lmo,
        J_oo, K_pno_cache, eps_lmo,
        t1_pno, fov_pno,
        foo_t2_dressed, foo_t1_dressed,
        fvv_t1_pair,
        ovL_pno_bare, S_pno_cache, K_coul_cache,
        ooL_bare, ooL_dressed,
        ovL_pno_dressed=None):
    """Compute the DLPNO-CCSD T2 residual using Jiang et al. Eqs 75-81.

    Uses BARE integrals for all 2e integrals, with T1 effects entering
    through explicit dressed intermediates.

    Args:
        foo_t2_dressed: (nocc, nocc) T2-dressed foo (theta × voov, bare integrals)
        foo_t1_dressed: (nocc, nocc) T1-dressed foo correction
        fvv_t1_pair: (n_pno, n_pno) T1-dressed fvv for this pair
        ovL_pno_bare: dict of BARE ovL tensors
        ooL_bare: (nocc, nocc, naux) BARE ooL
        ooL_dressed: (nocc, nocc, naux) asymmetrically dressed ooL (Eq 91)
    """
    key = (min(i, j), max(i, j))
    data = pno_spaces[key]
    C_pno_ij = data['C_pno']
    n_pno = C_pno_ij.shape[1]

    if n_pno == 0:
        return np.zeros((0, 0))

    def _get_S(key_other):
        """Get PNO overlap matrix from cache or compute."""
        if S_pno_cache is not None:
            S = S_pno_cache.get((key, key_other))
            if S is not None:
                return S
        return C_pno_ij.T @ (s1e @ pno_spaces[key_other]['C_pno'])

    t2_ij = t2_pno_all[key]

    # --- Project T1 to pair basis ---
    t1_i_pno = _project_t1_to_pair(t1_pno, i, key, S_pno_cache, pno_spaces)
    t1_j_pno = _project_t1_to_pair(t1_pno, j, key, S_pno_cache, pno_spaces)
    tau_ij = t2_ij + np.outer(t1_i_pno, t1_j_pno)

    # ===================================================================
    # K̃ (Eq 75): Dressed exchange = B̃_{ai}·B̃_{bj}
    # B̃_{ai} = B_{ai} - t̃_k^a·B_{ki} + B_{ab}·t_i^b - t̃_k^a·B_{kb}·t_i^b
    #
    # Our approach: start with BARE exchange K = ovL_i @ ovL_j.T
    # then add the three T1 correction terms.
    # ===================================================================
    ovL_i = ovL_pno_bare.get((key, i))
    ovL_j = ovL_pno_bare.get((key, j))

    # Control T1 dressing of exchange via function attribute
    _DRESS_EXCHANGE = getattr(compute_pair_residual_jiang, '_dress_exchange', False)

    if ovL_i is not None and ovL_j is not None:
        # Bare exchange
        K_bare = ovL_i @ ovL_j.T
        R_ij = K_bare.copy()

        # Term 2 of Eq 92: -Σ_k t̃_k^a·B_{ki} applied to both halves
        # ΔovL_i[a,Q] = -Σ_k T1_all[k,a]·ooL_bare[k,i,Q]
        _t1_all = np.zeros((nocc_lmo, n_pno))
        for k in range(nocc_lmo):
            _t1_all[k] = _project_t1_to_pair(
                t1_pno, k, key, S_pno_cache, pno_spaces)

        # Eq 92 Term 2: sign TBD (see debug results)
        _SIGN_TERM2 = getattr(compute_pair_residual_jiang, '_sign_term2', -1)
        delta_ovL_i = _SIGN_TERM2 * _t1_all.T @ ooL_bare[:, i, :]
        delta_ovL_j = _SIGN_TERM2 * _t1_all.T @ ooL_bare[:, j, :]

        # Term 3 of Eq 92: +B_{ab}·t_i^b → ovL already includes this
        # if computed from bare MOs. Actually, ovL_bare = (i a|Q) where
        # i is bare. The B_{ab}·t_i^b term needs vvL·t1, which is:
        # Σ_b (ab|Q)·t1_i^b — this requires a DF loop.
        # For the exchange, we can compute this efficiently:
        # ovL_dressed_i = ovL_bare_i + vvL·t1_i + delta_ovL_i
        # But vvL·t1_i requires a DF loop. Let's handle it:
        #
        # Actually, in the paper's Eq 92:
        # B̃_{ai} = B_{ai} + Σ_b B_{ab}·t_i^b - Σ_k t̃_k^a·B_{ki} - Σ_{k,b} t̃_k^a·B_{kb}·t_i^b
        #
        # We split into: ovL_dressed = ovL_bare + delta_vv + delta_oo + delta_cross
        # delta_vv = Σ_b vvL[a,b,Q]·t1_i^b (needs DF loop)
        # delta_oo = -T1_all.T @ ooL_bare[:,i,:] (computed above)
        # delta_cross = -T1_all.T @ (Σ_b ovL_bare_k[b,Q]·t1_i^b for each k)
        #
        # For delta_vv, we use a DF loop combined with the ladder:
        # Actually, let's just compute the full dressed ovL for i and j.

        # Compute vvL·t1 contribution via DF loop
        delta_vv_i = np.zeros((n_pno, ovL_i.shape[1]))
        delta_vv_j = np.zeros_like(delta_vv_i)
        if np.max(np.abs(t1_i_pno)) > 1e-15 or np.max(np.abs(t1_j_pno)) > 1e-15:
            mo_vv = np.asfortranarray(C_pno_ij)
            ijslice = (0, n_pno, 0, n_pno)
            buf = None
            aux_off = 0
            for Lpq in with_df.loop():
                nL = Lpq.shape[0]
                buf = _ao2mo.nr_e2(Lpq, mo_vv, ijslice, aosym='s2', out=buf)
                B_L = buf.reshape(nL, n_pno, n_pno)
                # delta_vv_i[a, aux_off:] = Σ_b B_L[L,a,b]·t1_i[b]
                delta_vv_i[:, aux_off:aux_off+nL] = np.einsum(
                    'Lab,b->aL', B_L, t1_i_pno)
                delta_vv_j[:, aux_off:aux_off+nL] = np.einsum(
                    'Lab,b->aL', B_L, t1_j_pno)
                aux_off += nL

        # Cross term: -Σ_k t̃_k^a · (Σ_b B_{kb}·t_i^b)
        # = -T1_all.T @ z_cross where z_cross[k,Q] = Σ_b ovL_bare_k[b,Q]·t1_i[b]
        z_cross_i = np.zeros((nocc_lmo, ovL_i.shape[1]))
        z_cross_j = np.zeros_like(z_cross_i)
        for k in range(nocc_lmo):
            ovL_k = ovL_pno_bare.get((key, k))
            if ovL_k is not None:
                z_cross_i[k] = t1_i_pno @ ovL_k
                z_cross_j[k] = t1_j_pno @ ovL_k
        delta_cross_i = -_t1_all.T @ z_cross_i  # (n_pno, naux)
        delta_cross_j = -_t1_all.T @ z_cross_j

        if _DRESS_EXCHANGE:
            # Which terms to include (for debugging)
            _TERMS = getattr(compute_pair_residual_jiang, '_exchange_terms', 'all')
            if _TERMS == 'oo_only':
                ovL_i_d = ovL_i + delta_ovL_i
                ovL_j_d = ovL_j + delta_ovL_j
            elif _TERMS == 'vv_only':
                ovL_i_d = ovL_i + delta_vv_i
                ovL_j_d = ovL_j + delta_vv_j
            elif _TERMS == 'oo_vv':
                ovL_i_d = ovL_i + delta_vv_i + delta_ovL_i
                ovL_j_d = ovL_j + delta_vv_j + delta_ovL_j
            else:  # all
                ovL_i_d = ovL_i + delta_vv_i + delta_ovL_i + delta_cross_i
                ovL_j_d = ovL_j + delta_vv_j + delta_ovL_j + delta_cross_j
            R_ij = ovL_i_d @ ovL_j_d.T
    else:
        K_bare = K_pno_cache.get(key, data.get('K_pno', np.zeros((n_pno, n_pno))))
        R_ij = K_bare.copy()

    # ===================================================================
    # A (Eq 76): Dressed ladder = tau × B̃_{ac} × B̃_{bd}
    # B̃_{ab} = B_{ab} - Σ_k t̃_k^a·B_{kb}  (Eq 93)
    # ===================================================================
    _DRESS_LADDER = getattr(compute_pair_residual_jiang, '_dress_ladder', False)
    if with_df is not None:
        if not _DRESS_LADDER or _t1_all is None:
            R_ij += _compute_ladder(tau_ij, C_pno_ij, with_df)
        else:
            # Dressed ladder: B̃_L[a,c] = B_L[a,c] - Σ_k T1_all[k,a]*ovL_k[c,L]
            if '_t1_all' not in dir() or _t1_all is None:
                _t1_all = np.zeros((nocc_lmo, n_pno))
                for k in range(nocc_lmo):
                    _t1_all[k] = _project_t1_to_pair(
                        t1_pno, k, key, S_pno_cache, pno_spaces)
            # Precompute correction tensor: corr[a,c,Q] = Σ_k T1[k,a]*ovL_k[c,Q]
            naux = ovL_pno_bare[(key, 0)].shape[1]
            ovL_stacked = np.stack(
                [ovL_pno_bare[(key, k)] for k in range(nocc_lmo)])  # (nocc, n_pno, naux)
            corr_full = np.einsum('ka,kcQ->acQ', _t1_all, ovL_stacked)  # (n_pno, n_pno, naux)

            mo = np.asfortranarray(C_pno_ij)
            ijslice = (0, n_pno, 0, n_pno)
            ladder = np.zeros((n_pno, n_pno))
            buf = None
            aux_off = 0
            for Lpq in with_df.loop():
                nL = Lpq.shape[0]
                buf = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
                B_L = buf.reshape(nL, n_pno, n_pno)  # bare (L, a, c)
                # Dress: B̃_L = B_L - corr_batch
                corr_batch = corr_full[:, :, aux_off:aux_off+nL].transpose(2, 0, 1)
                B_tilde = B_L - corr_batch  # (nL, n_pno, n_pno)
                X_L = np.einsum('Lac,cd->Lad', B_tilde, tau_ij)
                ladder += np.einsum('Lad,Lbd->ab', X_L, B_tilde)
                aux_off += nL
                Lpq = None
            R_ij += ladder

    # ===================================================================
    # B (Eq 77): Woooo with dressed β
    # β_{ij}^{kl} = B̃_{ki}·B̃_{lj} + tau_ij × B_{kc}·B_{ld}  (Eq 82)
    # Use ooL_dressed for the first part: β = ooL_dressed[k,i]·ooL_dressed[l,j]
    # ===================================================================
    _DEBUG_SKIP_WOOOO = getattr(compute_pair_residual_jiang, '_skip_woooo', False)
    if J_oo is not None and not _DEBUG_SKIP_WOOOO:
        # Build dressed J_oo from ooL_dressed
        ooL_d_flat = ooL_dressed.reshape(nocc_lmo * nocc_lmo, -1)
        J_oo_dressed = (ooL_d_flat @ ooL_d_flat.T).reshape(
            nocc_lmo, nocc_lmo, nocc_lmo, nocc_lmo)

        for key_kl, t2_kl in t2_pno_all.items():
            if t2_kl is None or t2_kl.shape[0] == 0:
                continue
            k, l = key_kl

            # tau_kl in PNO[kl]
            tau_kl = t2_kl.copy()
            t1_k = _project_t1_to_pair(t1_pno, k, key_kl, S_pno_cache, pno_spaces)
            t1_l = _project_t1_to_pair(t1_pno, l, key_kl, S_pno_cache, pno_spaces)
            tau_kl += np.outer(t1_k, t1_l)

            # β = dressed J_oo + tau×voov dressing
            W_kl = J_oo_dressed[i, k, j, l]

            # voov dressing: use dressed ovL if available
            _woo_src = ovL_pno_dressed if ovL_pno_dressed else ovL_pno_bare
            if with_df is not None and n_pno > 0:
                _ovL_k = _woo_src.get((key, k))
                _ovL_l = _woo_src.get((key, l))
                if _ovL_k is not None and _ovL_l is not None:
                    voov_kl = _ovL_k @ _ovL_l.T
                    tau_sym = tau_ij + tau_ij.T
                    if k != l:
                        d_kl = np.einsum('ab,ab->', tau_ij, voov_kl)
                        d_kl_cross = np.einsum('ba,ab->', tau_ij, voov_kl.T)
                        W_kl += 0.5 * (d_kl + d_kl_cross)
                    else:
                        W_kl += 0.5 * np.einsum('ab,ab->', tau_sym, voov_kl)

            S_proj = S_pno_cache.get((key, key_kl)) if S_pno_cache else None
            if S_proj is not None:
                tau_kl_proj = S_proj @ tau_kl @ S_proj.T
            else:
                tau_kl_proj = _project_t2_full(
                    tau_kl, pno_spaces[key_kl]['C_pno'], C_pno_ij, s1e)

            if k != l:
                d_lk = np.einsum('ab,ab->', tau_ij, voov_kl.T) if '_ovL_k' in dir() else 0
                d_lk_cross = np.einsum('ba,ab->', tau_ij, voov_kl) if '_ovL_k' in dir() else 0
                W_lk = J_oo_dressed[i, l, j, k]
                if with_df is not None and '_ovL_k' in dir():
                    W_lk += 0.5 * (d_lk + d_lk_cross)
                R_ij += W_kl * tau_kl_proj + W_lk * tau_kl_proj.T
            else:
                R_ij += W_kl * tau_kl_proj

    # ===================================================================
    # Section 4: Fock coupling — -P(ij) Σ_k F̃̃[k,i]·t_kj
    # F̃̃_{ij} = F_{ij} + foo_t2[i,j] + foo_t1[i,j]  (Eqs 85-86)
    # ===================================================================
    # Combined dressed Fock = bare + T2 dressing + T1 dressing
    ft_oo = F_lmo - np.diag(eps_lmo)
    if foo_t2_dressed is not None:
        ft_oo = ft_oo + foo_t2_dressed
    if foo_t1_dressed is not None:
        ft_oo = ft_oo + foo_t1_dressed
    # Additional: ft_ij += 0.5·t1·fov.T (canonical line 265)
    for kk in range(nocc_lmo):
        key_kk = (kk, kk)
        if key_kk not in pno_spaces:
            continue
        fov_kk = fov_pno.get(kk)
        if fov_kk is None or fov_kk.size == 0:
            continue
        for mm in range(nocc_lmo):
            t1_mm_in_kk = _project_t1_to_pair(
                t1_pno, mm, key_kk, S_pno_cache, pno_spaces)
            ft_oo[kk, mm] += 0.5 * np.dot(t1_mm_in_kk, fov_kk)

    for k in range(nocc_lmo):
        if abs(ft_oo[k, i]) > 1e-15:
            key_kj = (min(k, j), max(k, j))
            if key_kj in t2_pno_all and t2_pno_all[key_kj] is not None:
                t2_kj_raw = t2_pno_all[key_kj]
                if t2_kj_raw.shape[0] > 0:
                    t2_kj = t2_kj_raw.T if k > j else t2_kj_raw
                    S_kj = _get_S(key_kj)
                    R_ij -= ft_oo[k, i] * (S_kj @ t2_kj @ S_kj.T)
        if abs(ft_oo[k, j]) > 1e-15:
            key_ik = (min(i, k), max(i, k))
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
                t2_ik_raw = t2_pno_all[key_ik]
                if t2_ik_raw.shape[0] > 0:
                    t2_ik = t2_ik_raw.T if i > k else t2_ik_raw
                    S_ik = _get_S(key_ik)
                    R_ij -= ft_oo[k, j] * (S_ik @ t2_ik @ S_ik.T)

    # ===================================================================
    # E (Eq 80): t2 × F̃̃_{ab} (virtual Fock dressing)
    # F̃̃_{ab} = eps_a·δ_{ab} + fvv_t1 + fvv_t2
    # ===================================================================
    # T1 virtual Fock dressing
    if fvv_t1_pair is not None and np.max(np.abs(fvv_t1_pair)) > 1e-15:
        # Canonical: R[a,b] += Σ_c t2[a,c]*ft[b,c] + Σ_c t2.T[b,c]*ft[a,c]
        #          = t2 @ ft.T + ft @ t2
        R_ij += t2_ij @ fvv_t1_pair.T + fvv_t1_pair @ t2_ij

    # Line 266 analog: ft_ab also includes -0.5*t1.T@fov (separate from fvv_t1)
    if t1_pno is not None and fov_pno is not None:
        ft_ab_266 = np.zeros((n_pno, n_pno))
        for kk in range(nocc_lmo):
            t1_k_ij = _project_t1_to_pair(
                t1_pno, kk, key, S_pno_cache, pno_spaces)
            fov_k_ij = _project_t1_to_pair(
                fov_pno, kk, key, S_pno_cache, pno_spaces)
            ft_ab_266 -= 0.5 * np.outer(t1_k_ij, fov_k_ij)
        if np.max(np.abs(ft_ab_266)) > 1e-15:
            R_ij += t2_ij @ ft_ab_266.T + ft_ab_266 @ t2_ij

    # T2 virtual Fock dressing (section 6b of original code)
    if with_df is not None:
        fvv_t2_dressed = np.zeros((n_pno, n_pno))
        for key_mn, t2_mn_raw in t2_pno_all.items():
            if t2_mn_raw is None or t2_mn_raw.shape[0] == 0:
                continue
            mm, nn = key_mn
            n_mn = pno_spaces[key_mn]['C_pno'].shape[1]
            if n_mn == 0:
                continue
            theta_fock_mn = 2.0 * t2_mn_raw.T - t2_mn_raw
            S_proj = _get_S(key_mn)
            _ovL_fvv_src = ovL_pno_dressed if ovL_pno_dressed else ovL_pno_bare
            _ovL_mn_nn = _ovL_fvv_src.get((key_mn, nn))
            _ovL_ij_mm = _ovL_fvv_src.get((key, mm))
            if _ovL_mn_nn is not None and _ovL_ij_mm is not None:
                voov_mn = _ovL_mn_nn @ _ovL_ij_mm.T
            else:
                continue
            theta_proj = theta_fock_mn @ S_proj.T
            fvv_t2_dressed -= theta_proj.T @ voov_mn
            if mm != nn:
                theta_fock_nm = 2.0 * t2_mn_raw - t2_mn_raw.T
                _ovL_mn_mm = _ovL_fvv_src.get((key_mn, mm))
                _ovL_ij_nn = _ovL_fvv_src.get((key, nn))
                if _ovL_mn_mm is not None and _ovL_ij_nn is not None:
                    voov_nm = _ovL_mn_mm @ _ovL_ij_nn.T
                    theta_proj_nm = theta_fock_nm @ S_proj.T
                    fvv_t2_dressed -= theta_proj_nm.T @ voov_nm
        R_ij += t2_ij @ fvv_t2_dressed.T + fvv_t2_dressed @ t2_ij

    # ===================================================================
    # Ring / voov / oovv (sections 5 of original code)
    # K_dir uses dressed ovL (captures voov T1 dressing from MO rotation).
    # K_coul uses oo-dressed cache + vv corrections (Eq 93-like dressing).
    # ===================================================================
    if with_df is not None:
        # Build T1_all projected to PNO_ij (reused for vv corrections)
        _t1_all_ij = None
        if (t1_pno is not None and S_pno_cache is not None
                and ooL_dressed is not None and n_pno > 0):
            _t1_all_ij = np.zeros((nocc_lmo, n_pno))
            for kk in range(nocc_lmo):
                _t1_all_ij[kk] = _project_t1_to_pair(
                    t1_pno, kk, key, S_pno_cache, pno_spaces)

        for k in range(nocc_lmo):
            # --- Pair (i,k) contributions ---
            key_ik = (min(i, k), max(i, k))
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
                C_pno_ik = pno_spaces[key_ik]['C_pno']
                n_ik = C_pno_ik.shape[1]
                if n_ik > 0:
                    t2_ik_raw = t2_pno_all[key_ik]
                    t2_ik = t2_ik_raw.T if i > k else t2_ik_raw
                    S_ij_ik = _get_S(key_ik)

                    # K_dir (voov): use dressed ovL — captures T1 via MO rotation
                    _ovL_src = ovL_pno_dressed if ovL_pno_dressed else ovL_pno_bare
                    _ovL_j_ij = _ovL_src.get((key, j))
                    _ovL_k_ik = _ovL_src.get((key_ik, k))
                    K_dir = (_ovL_j_ij @ _ovL_k_ik.T
                             if _ovL_j_ij is not None and _ovL_k_ik is not None
                             else np.zeros((n_pno, n_ik)))

                    # K_coul (oovv): from cache (oo-dressed if available)
                    _kc_key = (key, key_ik, j, k)
                    K_coul = (K_coul_cache.get(_kc_key)
                              if K_coul_cache else None)
                    if K_coul is None:
                        K_coul = np.zeros((n_pno, n_ik))

                    # vv correction to K_coul: Eq 93-like dressing of virtual indices
                    # ΔK_coul_vv = -Σ_m T1_all_ij[m,b]*ovL_m_ik@ooL_d[j,k,:]
                    #             -Σ_m T1_all_ik[m,c]*ovL_m_ij@ooL_d[j,k,:]
                    if _t1_all_ij is not None and ooL_dressed is not None:
                        ooL_jk_d = ooL_dressed[j, k, :]
                        _t1_all_ik = np.zeros((nocc_lmo, n_ik))
                        for mm in range(nocc_lmo):
                            _t1_all_ik[mm] = _project_t1_to_pair(
                                t1_pno, mm, key_ik, S_pno_cache, pno_spaces)
                        W_ik = np.zeros((nocc_lmo, n_ik))
                        W_ij = np.zeros((nocc_lmo, n_pno))
                        for mm in range(nocc_lmo):
                            ovL_m_ik = ovL_pno_bare.get((key_ik, mm))
                            ovL_m_ij = ovL_pno_bare.get((key, mm))
                            if ovL_m_ik is not None:
                                W_ik[mm] = ovL_m_ik @ ooL_jk_d
                            if ovL_m_ij is not None:
                                W_ij[mm] = ovL_m_ij @ ooL_jk_d
                        K_coul = K_coul - _t1_all_ij.T @ W_ik - W_ij.T @ _t1_all_ik

                    theta_ik = 2.0 * t2_ik - t2_ik.T
                    R_ij += S_ij_ik @ (theta_ik @ (K_dir.T - 0.5 * K_coul.T))
                    R_ij -= 0.5 * S_ij_ik @ (t2_ik.T @ K_coul.T)
                    R_ij -= K_coul @ t2_ik @ S_ij_ik.T

            # --- Pair (j,k) contributions ---
            key_jk = (min(j, k), max(j, k))
            if key_jk in t2_pno_all and t2_pno_all[key_jk] is not None:
                C_pno_jk = pno_spaces[key_jk]['C_pno']
                n_jk = C_pno_jk.shape[1]
                if n_jk > 0:
                    t2_jk_raw = t2_pno_all[key_jk]
                    t2_jk = t2_jk_raw.T if j > k else t2_jk_raw
                    S_ij_jk = _get_S(key_jk)

                    # K_dir_j (voov): dressed ovL
                    _ovL_src_j = ovL_pno_dressed if ovL_pno_dressed else ovL_pno_bare
                    _ovL_i_ij = _ovL_src_j.get((key, i))
                    _ovL_k_jk = _ovL_src_j.get((key_jk, k))
                    K_dir_j = (_ovL_i_ij @ _ovL_k_jk.T
                               if _ovL_i_ij is not None and _ovL_k_jk is not None
                               else np.zeros((n_pno, n_jk)))

                    # K_coul_j (oovv): from cache (oo-dressed if available)
                    _kc_key_j = (key, key_jk, i, k)
                    K_coul_j = (K_coul_cache.get(_kc_key_j)
                                if K_coul_cache else None)
                    if K_coul_j is None:
                        K_coul_j = np.zeros((n_pno, n_jk))

                    # vv correction to K_coul_j
                    if _t1_all_ij is not None and ooL_dressed is not None:
                        ooL_ik_d = ooL_dressed[i, k, :]
                        _t1_all_jk = np.zeros((nocc_lmo, n_jk))
                        for mm in range(nocc_lmo):
                            _t1_all_jk[mm] = _project_t1_to_pair(
                                t1_pno, mm, key_jk, S_pno_cache, pno_spaces)
                        W_jk = np.zeros((nocc_lmo, n_jk))
                        W_ij2 = np.zeros((nocc_lmo, n_pno))
                        for mm in range(nocc_lmo):
                            ovL_m_jk = ovL_pno_bare.get((key_jk, mm))
                            ovL_m_ij = ovL_pno_bare.get((key, mm))
                            if ovL_m_jk is not None:
                                W_jk[mm] = ovL_m_jk @ ooL_ik_d
                            if ovL_m_ij is not None:
                                W_ij2[mm] = ovL_m_ij @ ooL_ik_d
                        K_coul_j = K_coul_j - _t1_all_ij.T @ W_jk - W_ij2.T @ _t1_all_jk

                    R_ij -= S_ij_jk @ (t2_jk.T @ K_coul_j.T)
                    theta_kj = 2.0 * t2_jk.T - t2_jk
                    R_ij += ((K_dir_j - 0.5 * K_coul_j) @ theta_kj) @ S_ij_jk.T
                    R_ij -= 0.5 * (K_coul_j @ t2_jk) @ S_ij_jk.T

    # ===================================================================
    # Section 5b: Quadratic T2 dressing of ring
    # (same as original code, using bare integrals)
    # ===================================================================
    if with_df is not None and foo_t2_dressed is not None:
        R_dress_ij = np.zeros((n_pno, n_pno))
        R_dress_ji = np.zeros((n_pno, n_pno))
        for k in range(nocc_lmo):
            key_ik = (min(i, k), max(i, k))
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
                n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
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
                        _5b_src = ovL_pno_dressed if ovL_pno_dressed else ovL_pno_bare
                        _ovL_ik_l = _5b_src.get((key_ik, l))
                        _ovL_jl_k = _5b_src.get((key_jl, k))
                        if _ovL_ik_l is not None and _ovL_jl_k is not None:
                            voov_lk = _ovL_ik_l @ _ovL_jl_k.T
                        else:
                            continue
                        W_k += 0.5 * voov_lk @ t2_jl @ S_ij_jl.T
                        _ovL_ik_k = _5b_src.get((key_ik, k))
                        _ovL_jl_l = _5b_src.get((key_jl, l))
                        if _ovL_ik_k is not None and _ovL_jl_l is not None:
                            voov_kl = _ovL_ik_k @ _ovL_jl_l.T
                        else:
                            continue
                        VOov_kl = voov_kl - 0.5 * voov_lk
                        tau_ring_lj = 2.0 * t2_jl.T - t2_jl
                        V_k += 0.5 * VOov_kl @ tau_ring_lj @ S_ij_jl.T
                    R_dress_ij += S_ij_ik_d @ (theta_ik_d @ V_k)
                    R_dress_ij += 0.5 * S_ij_ik_d @ (t2_ik_d.T @ W_k)
                    R_dress_ji += S_ij_ik_d @ (t2_ik_d.T @ W_k)

            # P(ij) partner (pair j,k)
            key_jk = (min(j, k), max(j, k))
            if key_jk in t2_pno_all and t2_pno_all[key_jk] is not None:
                n_jk = pno_spaces[key_jk]['C_pno'].shape[1]
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
                        _ovL_jk_l = _5b_src.get((key_jk, l))
                        _ovL_il_k = _5b_src.get((key_il, k))
                        if _ovL_jk_l is not None and _ovL_il_k is not None:
                            voov_lk_ji = _ovL_jk_l @ _ovL_il_k.T
                        else:
                            continue
                        W_k_ji += 0.5 * voov_lk_ji @ t2_il @ S_ij_il.T
                        _ovL_jk_k = _5b_src.get((key_jk, k))
                        _ovL_il_l = _5b_src.get((key_il, l))
                        if _ovL_jk_k is not None and _ovL_il_l is not None:
                            voov_kl_ji = _ovL_jk_k @ _ovL_il_l.T
                        else:
                            continue
                        VOov_kl_ji = voov_kl_ji - 0.5 * voov_lk_ji
                        tau_ring_li = 2.0 * t2_il.T - t2_il
                        V_k_ji += 0.5 * VOov_kl_ji @ tau_ring_li @ S_ij_il.T
                    R_dress_ji += S_ij_jk_d @ (theta_jk_d @ V_k_ji)
                    R_dress_ji += 0.5 * S_ij_jk_d @ (t2_jk_d.T @ W_k_ji)
                    R_dress_ij += S_ij_jk_d @ (t2_jk_d.T @ W_k_ji)

        R_ij += R_dress_ij + R_dress_ji.T

    return R_ij
