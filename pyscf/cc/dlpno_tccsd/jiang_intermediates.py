"""Jiang et al. dressed intermediates for DLPNO-CCSD.

Translates Psi4 ccsd.cc functions:
- compute_B_tilde() → build_B_tilde()
- t1_fock() → build_Fab(), build_Fkj(), build_G_tilde()
- compute_C_tilde() → [in lccsd_jiang.py]
- compute_D_tilde() → build_D_tilde()

All use BARE integrals (ovL_bare, ooL_bare) with explicit T1 dressing.
"""
import numpy as np
from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair


def _build_T1_all(t1_pno, pair_key, nocc, S_pno_cache, pno_spaces):
    """Build T1_all[m, a] = T1 of LMO m projected to PNO of pair_key."""
    n = pno_spaces[pair_key]['C_pno'].shape[1]
    T = np.zeros((nocc, n))
    for m in range(nocc):
        T[m] = _project_t1_to_pair(t1_pno, m, pair_key, S_pno_cache, pno_spaces)
    return T


def build_B_tilde(t2_pno_all, t1_pno, pno_spaces, nocc,
                  ovL_bare, ooL_bare, ooL_dressed, S_pno_cache):
    """Build B_tilde matrix (Eq 82): beta for Woooo.

    B_tilde[k,l] = Σ_Q B̃_{ki}^Q · B̃_{lj}^Q + Σ_{c,d} tau_ij^{cd} · (kc|ld)

    The first part uses ooL_dressed (Eq 91). The second uses bare ovL.

    Returns (nocc, nocc) matrix for the current pair (i,j).
    This must be called per-pair, not globally.
    """
    # B_tilde is pair-dependent through tau_ij. For Woooo, it enters as:
    # W_kl = B_tilde_ij(k,l) → summed with tau_kl × S
    # In Psi4, B_tilde[ij](k,l) = ooL_dressed[k,i,Q]*ooL_dressed[l,j,Q]
    #                              + Σ_{c,d} tau_ij[c,d]*ovL_k[c,Q]*ovL_l[d,Q]
    # This is computed per (i,j) pair.
    # For simplicity, we precompute just the ooL part:
    # B_tilde_oo[k,l] = ooL_dressed[k,:,Q] @ ooL_dressed[l,:,Q].T → (nocc,nocc,nocc,nocc)
    # This is J_oo_dressed.
    # The tau part is added inline in the pair residual.
    # So B_tilde = J_oo_dressed + tau×voov dressing (pair-specific).
    pass  # Handled inline in the residual


def build_G_tilde(t2_pno_all, t1_pno, pno_spaces, nocc,
                  ovL_bare, ooL_bare, S_pno_cache,
                  Fkj, foo_t1, cc_ints=None):
    """Build G_tilde (Eq 86): double-dressed Fock oo.

    G_tilde[k,j] = F̃_{kj} + Σ_l u_lj × K_il (bare exchange)

    Psi4 ccsd.cc lines 1820-1844.

    Args:
        Fkj: (nocc, nocc) = F̃_{kj} = dressed Fock oo (from build_Fkj)
        foo_t1: (nocc, nocc) = T1 correction to Fock (from _compute_foo_t1)
    """
    G = Fkj.copy()

    for i_idx in range(nocc):
        for j_idx in range(nocc):
            for l_idx in range(nocc):
                key_il = (min(i_idx, l_idx), max(i_idx, l_idx))
                key_lj = (min(l_idx, j_idx), max(l_idx, j_idx))

                if key_il not in t2_pno_all or key_lj not in t2_pno_all:
                    continue
                t2_il = t2_pno_all.get(key_il)
                t2_lj = t2_pno_all.get(key_lj)
                if t2_il is None or t2_lj is None:
                    continue
                if t2_il.shape[0] == 0 or t2_lj.shape[0] == 0:
                    continue

                # u_lj = 2*t2_lj - t2_lj.T (antisymmetrized)
                t2_lj_d = t2_lj.T if l_idx > j_idx else t2_lj
                u_lj = 2.0 * t2_lj_d - t2_lj_d.T

                # K_il = (ia|lb) = ovL_i_il @ ovL_l_il.T (bare exchange)
                if cc_ints is not None:
                    from pyscf.cc.dlpno_tccsd.local_df import get_local_K
                    _K = get_local_K(cc_ints, key_il, i_idx, l_idx)
                    if _K is not None:
                        K_il = _K
                    else:
                        ovL_i_il = ovL_bare.get((key_il, i_idx))
                        ovL_l_il = ovL_bare.get((key_il, l_idx))
                        if ovL_i_il is None or ovL_l_il is None:
                            continue
                        K_il = ovL_i_il @ ovL_l_il.T
                else:
                    ovL_i_il = ovL_bare.get((key_il, i_idx))
                    ovL_l_il = ovL_bare.get((key_il, l_idx))
                    if ovL_i_il is None or ovL_l_il is None:
                        continue
                    K_il = ovL_i_il @ ovL_l_il.T

                # Project u_lj to PNO_il: S(il,lj) @ u_lj @ S(lj,il)
                # When key_il == key_lj (always when i==j and l shares the
                # diagonal), the projection is identity (same PNO basis).
                if key_il == key_lj:
                    U_lj_proj = u_lj
                else:
                    S_il_lj = S_pno_cache.get((key_il, key_lj))
                    if S_il_lj is None:
                        continue
                    U_lj_proj = S_il_lj @ u_lj @ S_il_lj.T  # (n_il, n_il)

                # G[i,j] += K_il · U_lj.T = trace(K @ U.T) = Σ_{a,b} K[a,b]*U[b,a]
                G[i_idx, j_idx] += np.sum(K_il * U_lj_proj.T)

    return G


def _build_Fia_bar(t1_pno, pno_spaces, nocc, ovL_bare, S_pno_cache, pair_key):
    """Build T1-dressed ov Fock Fia_bar for a specific pair.

    Fia_bar[k, a] = Σ_Q [2*gamma_Q * ovL_k[a,Q] - (ovL_k @ T_n.T @ ovL_k)[k,a]]
    where gamma_Q = Σ_{m,c} T1_all[m,c] * ovL_m[c,Q]

    Returns (nocc, n_pno) matrix. Psi4 lines 1544-1571.
    """
    n_pno = pno_spaces[pair_key]['C_pno'].shape[1]
    if n_pno == 0:
        return np.zeros((nocc, 0))

    # Build T1_all projected to this pair's PNO basis
    T1_all = np.zeros((nocc, n_pno))
    for m in range(nocc):
        T1_all[m] = _project_t1_to_pair(
            t1_pno, m, pair_key, S_pno_cache, pno_spaces)

    # z_total[Q] = Σ_{m,c} T1_all[m,c] * ovL_m[c,Q]
    naux = 0
    for m in range(nocc):
        entry = ovL_bare.get((pair_key, m))
        if entry is not None:
            naux = entry.shape[1]
            break
    if naux == 0:
        return np.zeros((nocc, n_pno))

    z_total = np.zeros(naux)
    for m in range(nocc):
        ovL_m = ovL_bare.get((pair_key, m))
        if ovL_m is not None and np.max(np.abs(T1_all[m])) > 1e-15:
            z_total += T1_all[m] @ ovL_m  # (naux,)

    Fia_bar = np.zeros((nocc, n_pno))
    for k in range(nocc):
        ovL_k = ovL_bare.get((pair_key, k))
        if ovL_k is None:
            continue
        # J: 2 * gamma * ovL_k
        Fia_bar[k] += 2.0 * ovL_k @ z_total
        # K: -Σ_n (ovL_n @ (T1_n @ ovL_k))[a]
        for n in range(nocc):
            if np.max(np.abs(T1_all[n])) < 1e-15:
                continue
            ovL_n = ovL_bare.get((pair_key, n))
            if ovL_n is None:
                continue
            z_nk = T1_all[n] @ ovL_k  # (naux,)
            Fia_bar[k] -= ovL_n @ z_nk  # (n_pno,)

    return Fia_bar


def build_Fkj(F_lmo, eps_lmo, t1_pno, fov_pno, pno_spaces, nocc,
              ovL_bare, ooL_bare, S_pno_cache, foo_t2):
    """Build F̃_{kj} = dressed Fock oo (Eqs 94, 98).

    F̃_{kj} = F̄_{kj} + F̄_{kc}·t_j^c (Eq 94)
    where F̄_{kj} = F_{kj} + [2J-K]·t̃ (Eq 98)

    Combined with the T2-dressed foo contribution.
    """
    from pyscf.cc.dlpno_tccsd.lccsd_jiang import _compute_foo_t1

    # F̄_{kj} = F_{kj} + T1 dressing only (Eq 98).
    # The T2 contribution to G_tilde is added inside build_G_tilde
    # (Σ_l Tt_lj × K_il), matching Psi4 exactly.
    foo_t1 = _compute_foo_t1(
        t1_pno, fov_pno, pno_spaces, nocc,
        ovL_bare, ooL_bare, S_pno_cache)

    Fkj = F_lmo + foo_t1

    # Eq 94: F̃_{kj} += Σ_a F̄_{ka}(jj) · t̃_j^a
    # F̄_{ka} = fov_bare + [2J-K]·T1 (Fia_bar)
    # Psi4 line 1618-1622: Fkj(i,j) += Fia_bar[jj](i,:) · T1_j
    for j_idx in range(nocc):
        key_jj = (j_idx, j_idx)
        if key_jj not in pno_spaces or t1_pno[j_idx].size == 0:
            continue
        t1_j = t1_pno[j_idx]

        # Build full Fia_bar for diagonal pair (j,j)
        Fia_bar_jj = _build_Fia_bar(
            t1_pno, pno_spaces, nocc, ovL_bare, S_pno_cache, key_jj)

        # Add bare fov contribution (not in Fia_bar which starts from zero)
        for i_idx in range(nocc):
            fov_i_jj = _project_t1_to_pair(
                fov_pno, i_idx, key_jj, S_pno_cache, pno_spaces)
            Fkj[i_idx, j_idx] += np.dot(fov_i_jj + Fia_bar_jj[i_idx], t1_j)

    return Fkj, foo_t1


def build_Fab(t1_pno, fov_pno, pno_spaces, nocc,
              ovL_bare, S_pno_cache, with_df, pair_key):
    """Build F̃̃_{ab} (Eqs 85, 97, 101) for a specific pair.

    F̃̃_{ab} = F̃_{ab} - Σ_kl S·u_kl·K_kl·S (Eq 85)
    F̃_{ab} = F̄_{ab} - Σ_k t̃_k^a·F̄_{kb} (Eq 97)
    F̄_{ab} = ε_a·δ_{ab} + [2(ab|kc)-(ac|kb)]·t̃_k^c (Eq 101)

    Returns (n_pno, n_pno) matrix. The T2 part (Eq 85 subtraction) is
    handled inline in the residual following Psi4.
    """
    from pyscf.cc.dlpno_tccsd.lccsd_jiang import _compute_fvv_t1_pair

    # F̄_{ab} = ε_a·δ_{ab} + T1 corrections (Eq 101)
    n_pno = pno_spaces[pair_key]['C_pno'].shape[1]
    e_pno = pno_spaces[pair_key]['e_pno']

    # Start with PNO orbital energies
    Fab = np.diag(e_pno)

    # Add T1 corrections: [2(ab|kc)-(ac|kb)]·t̃_k^c
    fvv_t1 = _compute_fvv_t1_pair(
        t1_pno, fov_pno, pno_spaces, nocc,
        ovL_bare, S_pno_cache, with_df, pair_key)
    Fab += fvv_t1

    # Eq 97: F̃_{ab} = F̄_{ab} - Σ_k t̃_k^a · F̄_{kb}
    # Psi4 line 1640: Fab -= T_n_ij.T @ Fia_bar
    # F̄_{kb} = fov_bare + [2J-K]·T1 (full Fia_bar)
    Fia_bar_ij = _build_Fia_bar(
        t1_pno, pno_spaces, nocc, ovL_bare, S_pno_cache, pair_key)
    T1_all = np.zeros((nocc, n_pno))
    for kk in range(nocc):
        T1_all[kk] = _project_t1_to_pair(
            t1_pno, kk, pair_key, S_pno_cache, pno_spaces)
    # fov_bare for all occ in this pair's PNO domain
    fov_all = np.zeros((nocc, n_pno))
    for kk in range(nocc):
        fov_all[kk] = _project_t1_to_pair(
            fov_pno, kk, pair_key, S_pno_cache, pno_spaces)
    # Fab -= T1_all.T @ (fov_all + Fia_bar)
    Fab -= T1_all.T @ (fov_all + Fia_bar_ij)

    return Fab


def build_Fab_all(t1_pno, fov_pno, pno_spaces, nocc,
                  ovL_bare, S_pno_cache, with_df, pair_keys,
                  _fvv_t1_precomputed=None):
    """Precompute Fab for ALL pairs in a single DF pass.

    Replaces per-pair build_Fab() calls, avoiding redundant DF reads and
    the thread-safety issue with with_df.loop().

    Returns dict: pair_key -> (n_pno, n_pno) Fab matrix.
    """
    if _fvv_t1_precomputed is not None:
        fvv_t1_all = _fvv_t1_precomputed
    else:
        from pyscf.cc.dlpno_tccsd.lccsd_jiang import compute_fvv_t1_all_pairs
        fvv_t1_all = compute_fvv_t1_all_pairs(
            t1_pno, fov_pno, pno_spaces, nocc,
            ovL_bare, S_pno_cache, with_df, pair_keys)

    # Complete Fab for each pair (Eq 97: Fab -= T1.T @ (fov + Fia_bar))
    Fab_all = {}
    for pk in pair_keys:
        n_pno = pno_spaces[pk]['C_pno'].shape[1]
        if n_pno == 0:
            Fab_all[pk] = np.zeros((0, 0))
            continue

        e_pno = pno_spaces[pk]['e_pno']
        Fab = np.diag(e_pno) + fvv_t1_all[pk]

        # Eq 97: F̃_{ab} = F̄_{ab} - Σ_k t̃_k^a · F̄_{kb}
        Fia_bar_ij = _build_Fia_bar(
            t1_pno, pno_spaces, nocc, ovL_bare, S_pno_cache, pk)
        T1_all = _build_T1_all(t1_pno, pk, nocc, S_pno_cache, pno_spaces)
        fov_all = np.zeros((nocc, n_pno))
        for kk in range(nocc):
            fov_all[kk] = _project_t1_to_pair(
                fov_pno, kk, pk, S_pno_cache, pno_spaces)
        Fab -= T1_all.T @ (fov_all + Fia_bar_ij)

        Fab_all[pk] = Fab

    return Fab_all


def build_D_tilde(t1_pno, t2_pno_all, pno_spaces, nocc,
                  ovL_bare, ooL_bare, S_pno_cache, with_df,
                  _term2_precomputed=None, cc_ints=None):
    """Build D_tilde (delta, Eq 84) for all ordered (i,k) pairs.

    delta_{ik}^{ac} = Terms 1-4 of Eq 84, using M/L integrals.
    Term 2 uses a batched single DF pass over all (i,k) pairs.

    Following Psi4 ccsd.cc compute_D_tilde() lines 1753-1818.
    """
    from pyscf.ao2mo import _ao2mo

    D_tilde_all = {}

    # Iterate over all ordered (i,k) pairs
    all_pairs = set()
    for key in t2_pno_all:
        a, b = key
        all_pairs.add((a, b))
        all_pairs.add((b, a))

    # Term 2 can be precomputed via unified DF pass
    if _term2_precomputed is None:
        # Batched Term 2 via single DF pass (fallback)
        term2_data = {}
        for i_idx, k_idx in all_pairs:
            key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
            n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
            if n_ik == 0:
                continue
            t1_i_ik = _project_t1_to_pair(
                t1_pno, i_idx, key_ik, S_pno_cache, pno_spaces)
            if np.max(np.abs(t1_i_ik)) < 1e-15:
                continue
            ovL_k_ik = ovL_bare.get((key_ik, k_idx))
            if ovL_k_ik is None:
                continue
            term2_data[(i_idx, k_idx)] = {
                'key_ik': key_ik, 'n_ik': n_ik,
                'C_pno': np.asfortranarray(pno_spaces[key_ik]['C_pno']),
                't1_i': t1_i_ik, 'ovL_k': ovL_k_ik,
                'result': np.zeros((n_ik, n_ik)),
            }
        if term2_data:
            aux_off = 0
            for Lpq in with_df.loop():
                nL = Lpq.shape[0]
                for (i_idx, k_idx), td in term2_data.items():
                    n_ik = td['n_ik']
                    buf = _ao2mo.nr_e2(Lpq, td['C_pno'],
                                       (0, n_ik, 0, n_ik), aosym='s2')
                    B_L = buf.reshape(nL, n_ik, n_ik)
                    ovL_k_batch = td['ovL_k'][:, aux_off:aux_off+nL]
                    z_c = np.einsum('b,Lbc->Lc', td['t1_i'], B_L)
                    y = td['t1_i'] @ ovL_k_batch
                    td['result'] += 2.0 * ovL_k_batch @ z_c
                    td['result'] -= np.einsum('L,Lac->ac', y, B_L)
                aux_off += nL
            _term2_precomputed = {ik: td['result'] for ik, td in term2_data.items()}

    # --- Build D_tilde for each (i,k) pair ---
    for i_idx, k_idx in all_pairs:
        key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
        n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
        if n_ik == 0:
            continue

        D_tilde_ik = np.zeros((n_ik, n_ik))

        # T1_i projected to PNO_ik
        t1_i_ik = _project_t1_to_pair(
            t1_pno, i_idx, key_ik, S_pno_cache, pno_spaces)

        # T1_all projected to PNO_ik
        T1_all_ik = np.zeros((nocc, n_ik))
        for mm in range(nocc):
            T1_all_ik[mm] = _project_t1_to_pair(
                t1_pno, mm, key_ik, S_pno_cache, pno_spaces)

        # Term 2: Psi4 lines 1791-1801 use K_tilde_chem[ki] × t1_i.
        # K_tc[b, a*n+c] = Σ_Q k_Qa[Q,b] * Qab[Q,a,c]
        # Part A: D[a,b] += 2 * Σ_{Q,c} k_Qa[Q,b] * Qab[Q,a,c] * t1_i[c]
        # Part B: D[r,s] -= Σ_{Q,b} t1_i[b] * k_Qa[Q,b] * Qab[Q,s,r]
        key_ki = (min(k_idx, i_idx), max(k_idx, i_idx))
        if cc_ints is not None and key_ki in cc_ints and cc_ints[key_ki] is not None:
            ci_ki = cc_ints[key_ki]
            if key_ki[0] == k_idx:
                k_Qa = ci_ki['i_Qa']
            else:
                k_Qa = ci_ki['j_Qa']
            Qab_ki = ci_ki['Qab']
            # Part A: D[a,b] += 2 * Σ_{Q,c} k_Qa[Q,b]*Qab[Q,a,c]*t1_i[c]
            # z[Q,a] = Σ_c Qab[Q,a,c]*t1_i[c]
            z_Qa = np.einsum('Qac,c->Qa', Qab_ki, t1_i_ik)
            # D[a,b] += 2 * Σ_Q z_Qa[Q,a] * k_Qa[Q,b] = 2 * z_Qa.T @ k_Qa
            D_tilde_ik += 2.0 * z_Qa.T @ k_Qa
            # Part B: D[r,s] -= Σ_{Q,b} t1_i[b]*k_Qa[Q,b]*Qab[Q,s,r]
            # w[Q] = Σ_b t1_i[b]*k_Qa[Q,b]
            w = k_Qa @ t1_i_ik  # (n_local,)
            # D[r,s] -= Σ_Q w[Q]*Qab[Q,s,r]
            D_tilde_ik -= np.einsum('Q,Qsr->rs', w, Qab_ki)
        elif _term2_precomputed and (i_idx, k_idx) in _term2_precomputed:
            D_tilde_ik += _term2_precomputed[(i_idx, k_idx)]

        # --- Term 1: -Σ_l T1_all[l,a] · M_{ik}^{lc} ---
        # M_{ik}^{lc} = 2*(il|kc) - (ik|lc)
        # (il|kc) = Σ_Q ooL[i,l,Q]*ovL_k_ik[c,Q] → K_bar-like
        # (ik|lc) = Σ_Q ooL[i,k,Q]*ovL_l_ik[c,Q] → K_bar_chem-like
        _use_local_d1 = (cc_ints is not None and key_ik in cc_ints
                         and cc_ints[key_ik] is not None)
        for ll in range(nocc):
            if _use_local_d1:
                from pyscf.cc.dlpno_tccsd.local_df import (
                    get_local_ovL, get_local_ooL_vec)
                _ovL_k = get_local_ovL(cc_ints, key_ik, k_idx)
                _ovL_l = get_local_ovL(cc_ints, key_ik, ll)
                _ooL_il = get_local_ooL_vec(cc_ints, i_idx, ll, key_ik)
                _ooL_ik = get_local_ooL_vec(cc_ints, i_idx, k_idx, key_ik)
                if _ovL_k is not None and _ovL_l is not None and _ooL_il is not None and _ooL_ik is not None:
                    ilkc = _ovL_k @ _ooL_il
                    iklc = _ovL_l @ _ooL_ik
                else:
                    continue
            else:
                ooL_il = ooL_bare[i_idx, ll, :]
                ooL_ik = ooL_bare[i_idx, k_idx, :]
                ovL_k_ik_entry = ovL_bare.get((key_ik, k_idx))
                ovL_l_ik_entry = ovL_bare.get((key_ik, ll))
                if ovL_k_ik_entry is None or ovL_l_ik_entry is None:
                    continue
                ilkc = ovL_k_ik_entry @ ooL_il
                iklc = ovL_l_ik_entry @ ooL_ik
            # M^{lc} = 2*(il|kc) - (ik|lc)
            M_lc = 2.0 * ilkc - iklc  # (n_ik,)
            # D[a,c] -= T1_all[l,a] * M_lc[c]
            D_tilde_ik -= np.outer(T1_all_ik[ll], M_lc)

        # --- Term 3: -Σ_l T1_l^a · (L_lk × T1_i) ---
        # L_lk[a,b] = 2*(la|kb) - (lb|ka) in PNO_lk
        for ll in range(nocc):
            key_lk = (min(ll, k_idx), max(ll, k_idx))
            if key_lk not in pno_spaces:
                continue
            n_lk = pno_spaces[key_lk]['C_pno'].shape[1]
            if n_lk == 0:
                continue
            if cc_ints is not None:
                from pyscf.cc.dlpno_tccsd.local_df import get_local_K
                _K = get_local_K(cc_ints, key_lk, ll, k_idx)
                if _K is not None:
                    K_lk = _K
                else:
                    ovL_l_lk = ovL_bare.get((key_lk, ll))
                    ovL_k_lk = ovL_bare.get((key_lk, k_idx))
                    if ovL_l_lk is None or ovL_k_lk is None:
                        continue
                    K_lk = ovL_l_lk @ ovL_k_lk.T
            else:
                ovL_l_lk = ovL_bare.get((key_lk, ll))
                ovL_k_lk = ovL_bare.get((key_lk, k_idx))
                if ovL_l_lk is None or ovL_k_lk is None:
                    continue
                K_lk = ovL_l_lk @ ovL_k_lk.T
            L_lk = 2.0 * K_lk - K_lk.T

            t1_i_lk = _project_t1_to_pair(
                t1_pno, i_idx, key_lk, S_pno_cache, pno_spaces)
            if np.max(np.abs(t1_i_lk)) < 1e-15:
                continue
            Lt1 = L_lk @ t1_i_lk  # (n_lk,)
            # Project to PNO_ik
            S_ik_lk = S_pno_cache.get((key_ik, key_lk))
            if S_ik_lk is None:
                continue
            Lt1_ik = S_ik_lk @ Lt1  # (n_ik,)
            t1_l_ik = T1_all_ik[ll]
            D_tilde_ik -= np.outer(t1_l_ik, Lt1_ik)

        # --- Term 4: +(1/2) Σ_l S@u_il@S @ L_kl @ S ---
        for ll in range(nocc):
            key_il = (min(i_idx, ll), max(i_idx, ll))
            key_lk = (min(ll, k_idx), max(ll, k_idx))
            if key_il not in t2_pno_all or key_lk not in pno_spaces:
                continue
            t2_il = t2_pno_all.get(key_il)
            if t2_il is None or t2_il.shape[0] == 0:
                continue
            n_il = t2_il.shape[0]
            n_lk = pno_spaces[key_lk]['C_pno'].shape[1]
            if n_lk == 0:
                continue

            t2_il_d = t2_il.T if i_idx > ll else t2_il
            u_il = 2.0 * t2_il_d - t2_il_d.T

            if cc_ints is not None:
                from pyscf.cc.dlpno_tccsd.local_df import get_local_K
                _K = get_local_K(cc_ints, key_lk, ll, k_idx)
                if _K is not None:
                    K_lk = _K
                else:
                    ovL_l_lk = ovL_bare.get((key_lk, ll))
                    ovL_k_lk = ovL_bare.get((key_lk, k_idx))
                    if ovL_l_lk is None or ovL_k_lk is None:
                        continue
                    K_lk = ovL_l_lk @ ovL_k_lk.T
            else:
                ovL_l_lk = ovL_bare.get((key_lk, ll))
                ovL_k_lk = ovL_bare.get((key_lk, k_idx))
                if ovL_l_lk is None or ovL_k_lk is None:
                    continue
                K_lk = ovL_l_lk @ ovL_k_lk.T
            L_lk = 2.0 * K_lk - K_lk.T

            def _get_S_or_I(ka, kb, n_a):
                if ka == kb:
                    return np.eye(n_a)
                return S_pno_cache.get((ka, kb))

            n_ik2 = pno_spaces[key_ik]['C_pno'].shape[1]
            S_ik_il = _get_S_or_I(key_ik, key_il, n_ik2)
            S_il_lk = _get_S_or_I(key_il, key_lk, n_il)
            S_lk_ik = _get_S_or_I(key_lk, key_ik, n_lk)
            if S_ik_il is None or S_il_lk is None or S_lk_ik is None:
                continue

            # Psi4: S(ik,il) @ Tt[il] @ S(il,lk) @ L[lk] @ S(lk,ik)
            D_temp = S_ik_il @ u_il @ S_il_lk @ L_lk @ S_lk_ik
            D_tilde_ik += 0.5 * D_temp  # Term 4: Jiang Eq 84

        D_tilde_all[(i_idx, k_idx)] = D_tilde_ik

    return D_tilde_all


def build_mixed_domain_integrals(t2_pno_all, pno_spaces, nocc,
                                 ovL_bare, ooL_bare, S_pno_cache,
                                 cc_ints=None):
    """Build J_ij_kj and K_ij_kj: mixed-domain bare integrals for C/D terms.

    J_ij_kj[(key_ij, k)] = (ik|a_ij c_kj) = Coulomb with mixed PNO domains
    K_ij_kj[(key_ij, k)] = (ia_ij|kc_kj) = exchange with mixed PNO domains

    These are the "bold" terms in Eqs 78-79.
    """
    J_cache = {}
    K_cache = {}

    keys_sorted = sorted(t2_pno_all.keys())
    for key_ij in keys_sorted:
        i, j = key_ij
        n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_ij == 0:
            continue

        for k in range(nocc):
            key_kj = (min(k, j), max(k, j))
            if key_kj not in pno_spaces:
                continue
            n_kj = pno_spaces[key_kj]['C_pno'].shape[1]
            if n_kj == 0:
                continue

            # J(ik|a_ij c_kj) = Σ_Q ooL[i,k,Q] * vvL_mixed[a_ij, c_kj, Q]
            # But vvL_mixed requires mixed PNO basis transform.
            # For the Coulomb integral: (ik|ac) = ooL[i,k,Q] * (ac|Q)
            # where a is in PNO_ij and c is in PNO_kj.
            # (ac|Q) = C_pno_ij[:,a].T @ Lpq @ C_pno_kj[:,c] → mixed DF
            # This is exactly what K_coul_cache stores!
            # K_coul_cache[(key_ij, key_kj, i, k)] = (ik|a_ij c_kj)
            # So J_ij_kj = K_coul_cache[(key_ij, key_kj, i, k)]
            # Already computed! Just reference it.
            # We pass K_coul_cache directly instead.

            # K(ia_ij|kc_kj) = ovL_i_ij[a,Q] * ovL_k_kj[c,Q]
            if cc_ints is not None and key_ij in cc_ints and cc_ints[key_ij] is not None:
                _K_local = cc_ints[key_ij].get('K_ij_kj', {}).get((key_ij, k))
                if _K_local is not None:
                    K_cache[(key_ij, k)] = _K_local
                else:
                    ovL_i_ij = ovL_bare.get((key_ij, i))
                    ovL_k_kj = ovL_bare.get((key_kj, k))
                    if ovL_i_ij is not None and ovL_k_kj is not None:
                        K_cache[(key_ij, k)] = ovL_i_ij @ ovL_k_kj.T
            else:
                ovL_i_ij = ovL_bare.get((key_ij, i))
                ovL_k_kj = ovL_bare.get((key_kj, k))
                if ovL_i_ij is not None and ovL_k_kj is not None:
                    K_cache[(key_ij, k)] = ovL_i_ij @ ovL_k_kj.T

    return K_cache
