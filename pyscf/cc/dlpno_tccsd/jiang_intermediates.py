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
                  Fkj, foo_t1):
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
                ovL_i_il = ovL_bare.get((key_il, i_idx))
                ovL_l_il = ovL_bare.get((key_il, l_idx))
                if ovL_i_il is None or ovL_l_il is None:
                    continue
                K_il = ovL_i_il @ ovL_l_il.T

                # Project u_lj to PNO_il: S(il,lj) @ u_lj @ S(lj,il)
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

    # F̃ = F_bare + foo_T2 + foo_T1 (Eqs 94, 98)
    foo_t1 = _compute_foo_t1(
        t1_pno, fov_pno, pno_spaces, nocc,
        ovL_bare, ooL_bare, S_pno_cache)

    Fkj = F_lmo - np.diag(eps_lmo) + foo_t2 + foo_t1

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


def build_D_tilde(t1_pno, t2_pno_all, pno_spaces, nocc,
                  ovL_bare, ooL_bare, S_pno_cache, with_df):
    """Build D_tilde (delta, Eq 84) for all ordered (i,k) pairs.

    delta_{ik}^{ac} = Terms 1-4 of Eq 84, using M/L integrals.

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

        # --- Term 2: +t̃_i^b · M_{kb}^{ac} ---
        # M = 2K - J. In PNO_ik: M_{kb}^{ac} = 2*(ka|bc) - (kb|ac)
        # This needs (ka|bc) = vovv integral → DF loop
        # Psi4 uses K_tilde_chem_[ki] = ovL_k @ vvL.T for the K part
        # and reshapes for contraction with T1_i.
        # For now: compute via DF loop
        if np.max(np.abs(t1_i_ik)) > 1e-15:
            C_pno_ik = pno_spaces[key_ik]['C_pno']
            ovL_k_ik = ovL_bare.get((key_ik, k_idx))
            if ovL_k_ik is not None:
                mo_vv = np.asfortranarray(C_pno_ik)
                ijslice = (0, n_ik, 0, n_ik)
                buf = None
                # Term 2: +Σ_b T1_i[b] * M_{kb}^{ac}
                # M_{kb}^{ac} = 2*(ka|bc) - (kb|ac)
                # (ka|bc) = ovL_k[a,L]*B_L[L,b,c]
                # (kb|ac) = ovL_k[b,L]*B_L[L,a,c]
                D_tilde_ik_term2 = np.zeros((n_ik, n_ik))
                aux_off = 0
                for Lpq in with_df.loop():
                    nL = Lpq.shape[0]
                    buf = _ao2mo.nr_e2(Lpq, mo_vv, ijslice, aosym='s2', out=buf)
                    B_L = buf.reshape(nL, n_ik, n_ik)
                    ovL_k_batch = ovL_k_ik[:, aux_off:aux_off+nL]
                    # (ka|bc): ovL_k[a,L]*B_L[L,b,c]
                    # Need: Σ_b T1_i[b] * M_{kb}^{ac}
                    # M_{kb}^{ac} = 2*(ka|bc) - (kb|ac)
                    # = 2*ovL_k[a,L]*B_L[L,b,c] - ovL_k[b,L]*B_L[L,a,c]
                    # Σ_b T1[b] * M[b,a,c]:
                    # = 2*Σ_b T1[b]*ovL_k[a,L]*B_L[L,b,c] - Σ_b T1[b]*ovL_k[b,L]*B_L[L,a,c]
                    # = 2*ovL_k[a,L]*Σ_b T1[b]*B_L[L,b,c] - (T1@ovL_k_batch)[L]*B_L[L,a,c]
                    z_c = np.einsum('b,Lbc->Lc', t1_i_ik, B_L)  # (nL, n_ik)
                    y = t1_i_ik @ ovL_k_batch  # (nL,)
                    # Term: 2*ovL_k[a,L]*z_c[L,c]
                    D_tilde_ik_term2 += 2.0 * ovL_k_batch @ z_c
                    # Term: -y[L]*B_L[L,a,c]
                    D_tilde_ik_term2 -= np.einsum('L,Lac->ac', y, B_L)
                    aux_off += nL
                D_tilde_ik += D_tilde_ik_term2  # Add Term 2 contribution

        # --- Term 1: -Σ_l T1_all[l,a] · M_{ik}^{lc} ---
        # M_{ik}^{lc} = 2*(il|kc) - (ik|lc)
        # (il|kc) = Σ_Q ooL[i,l,Q]*ovL_k_ik[c,Q] → K_bar-like
        # (ik|lc) = Σ_Q ooL[i,k,Q]*ovL_l_ik[c,Q] → K_bar_chem-like
        ooL_ik = ooL_bare[i_idx, k_idx, :]  # (naux,)
        for ll in range(nocc):
            ooL_il = ooL_bare[i_idx, ll, :]  # (naux,)
            ovL_k_ik_entry = ovL_bare.get((key_ik, k_idx))
            ovL_l_ik_entry = ovL_bare.get((key_ik, ll))
            if ovL_k_ik_entry is None or ovL_l_ik_entry is None:
                continue
            # (il|kc) = ooL[i,l,Q] * ovL_k[c,Q] = Σ_Q ooL_il[Q]*ovL_k[c,Q]
            ilkc = ovL_k_ik_entry @ ooL_il  # (n_ik,) for each c
            # (ik|lc) = ooL[i,k,Q] * ovL_l[c,Q]
            iklc = ovL_l_ik_entry @ ooL_ik  # (n_ik,) for each c
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
            ovL_l_lk = ovL_bare.get((key_lk, ll))
            ovL_k_lk = ovL_bare.get((key_lk, k_idx))
            if ovL_l_lk is None or ovL_k_lk is None:
                continue
            K_lk = ovL_l_lk @ ovL_k_lk.T  # (n_lk, n_lk)
            L_lk = 2.0 * K_lk - K_lk.T  # L = 2K - K^swap

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

            ovL_l_lk = ovL_bare.get((key_lk, ll))
            ovL_k_lk = ovL_bare.get((key_lk, k_idx))
            if ovL_l_lk is None or ovL_k_lk is None:
                continue
            # Psi4: L[lk] = 2*K[lk] - K[lk].T where K[lk] = (la|kb)
            K_lk = ovL_l_lk @ ovL_k_lk.T  # K[lk] = (la|kb)
            L_lk = 2.0 * K_lk - K_lk.T    # L[lk] = 2*(la|kb) - (kb|la)

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
                                 ovL_bare, ooL_bare, S_pno_cache):
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
            ovL_i_ij = ovL_bare.get((key_ij, i))
            ovL_k_kj = ovL_bare.get((key_kj, k))
            if ovL_i_ij is not None and ovL_k_kj is not None:
                K_cache[(key_ij, k)] = ovL_i_ij @ ovL_k_kj.T

    return K_cache
