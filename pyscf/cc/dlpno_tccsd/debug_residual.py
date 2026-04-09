"""Debug: compute T2 residual term by term for CCD (T1=0).

Simplified clean implementation to compare against compute_residual_v2.
Each term is computed independently and returned for comparison.
"""
import numpy as np
from pyscf.ao2mo import _ao2mo


def compute_ccd_residual_debug(i, j, t2_pno_all, pno_spaces, nocc,
                                ovL_cache, S_cache, K_cache, K_coul_cache,
                                ooL, F_lmo, with_df):
    """Compute CCD T2 residual numerator for pair (i,j).

    Returns dict of per-term contributions and the total R.
    All at T1=0 (no dressing).
    """
    key = (min(i,j), max(i,j))
    data = pno_spaces[key]
    C_pno = data['C_pno']
    n = C_pno.shape[1]
    if n == 0:
        return {}

    t2_ij = t2_pno_all[key]
    eps = data['e_pno']

    def S(ka, kb):
        if ka == kb: return np.eye(pno_spaces[ka]['C_pno'].shape[1])
        return S_cache.get((ka, kb), np.zeros((0,0)))

    def get_t2(p, q):
        pk = (min(p,q), max(p,q))
        if pk not in t2_pno_all: return None
        t = t2_pno_all[pk]
        return t.T if p > q else t

    terms = {}

    # === K: bare exchange ===
    K_ij = K_cache.get(key)
    if K_ij is None:
        ovL_i = ovL_cache.get((key, i))
        ovL_j = ovL_cache.get((key, j))
        K_ij = ovL_i @ ovL_j.T
    terms['K'] = K_ij.copy()

    # === A: ladder ===
    # A[a,b] = sum_{Q,c,d} B_ac^Q * tau_ij[c,d] * B_bd^Q
    # At T1=0: tau = T2
    A = np.zeros((n, n))
    mo = np.asfortranarray(C_pno)
    ijslice = (0, n, 0, n)
    buf = None; aux_off = 0
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]
        buf = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
        B_L = buf.reshape(nL, n, n)  # (L, a, b) = B_ab^Q
        X_L = np.einsum('Lac,cd->Lad', B_L, t2_ij)
        A += np.einsum('Lad,Lbd->ab', X_L, B_L)
        aux_off += nL
    terms['A'] = A

    # === B: Woooo ===
    # B[a,b] = sum_{k,l} beta[k,l] * S @ tau[kl] @ S^T
    # beta[k,l] = (ki|lj) + sum_{c,d} tau_ij[c,d] * (kc|ld)_ij
    B = np.zeros((n, n))
    for k in range(nocc):
        for l in range(nocc):
            kl = (min(k,l), max(k,l))
            t2_kl = get_t2(k, l)
            if t2_kl is None or t2_kl.shape[0] == 0:
                continue

            # beta = (ki|lj) + voov contribution
            beta = np.sum(ooL[k, i] * ooL[l, j])  # (ki|lj)

            ovL_k = ovL_cache.get((key, k))
            ovL_l = ovL_cache.get((key, l))
            if ovL_k is not None and ovL_l is not None:
                # sum_{c,d} T2_ij[c,d] * ovL_k[c,Q] * ovL_l[d,Q]
                beta += np.sum(t2_ij * (ovL_k @ ovL_l.T))

            # tau_kl projected: S @ t2[kl] @ S^T
            S_kl = S(key, kl)
            tau_proj = S_kl @ t2_kl @ S_kl.T  # at T1=0, tau=T2
            B += beta * tau_proj
    terms['B'] = B

    # === E: fvv dressing ===
    # E_tilde = diag(eps) - sum_{k,l} S @ (Tt[kl] @ K[kl]^T) @ S^T
    # E[a,b] = T2[a,c] * E_tilde[c,b] + E_tilde[a,c] * T2[c,b]
    # For our direct-substitution: E_tilde WITHOUT diag(eps)
    E_tilde = np.zeros((n, n))
    for k in range(nocc):
        for l in range(nocc):
            kl = (min(k,l), max(k,l))
            t2_kl_raw = t2_pno_all.get(kl)
            if t2_kl_raw is None or t2_kl_raw.shape[0] == 0:
                continue
            t2_kl = t2_kl_raw.T if k > l else t2_kl_raw
            Tt_kl = 2.0 * t2_kl - t2_kl.T

            ovL_k_kl = ovL_cache.get((kl, k))
            ovL_l_kl = ovL_cache.get((kl, l))
            if ovL_k_kl is None or ovL_l_kl is None:
                continue
            K_kl = ovL_k_kl @ ovL_l_kl.T

            E_temp = Tt_kl @ K_kl.T  # (n_kl, n_kl)
            S_kl_ij = S(key, kl)
            E_tilde -= S_kl_ij @ E_temp @ S_kl_ij.T

    E = t2_ij @ E_tilde.T + E_tilde @ t2_ij
    terms['E'] = E
    terms['E_tilde'] = E_tilde

    # === C: ring (gamma) ===
    # C[a,b] = -sum_k gamma(ki)[a,c] * T[kj]^T[c,b'] * S[b',b]
    # gamma(ki) = J_ij_kj + C_tilde_ki projected
    # At T1=0: C_tilde Term 4 only = -0.5 * sum_l S@T2[li]@S@K[kl]@S
    C = np.zeros((n, n))
    for k in range(nocc):
        key_kj = (min(k,j), max(k,j))
        t2_kj = get_t2(k, j)
        if t2_kj is None or t2_kj.shape[0] == 0:
            continue

        # gamma = J(ik|a_ij c_kj) via K_coul
        gamma = np.zeros((n, pno_spaces[key_kj]['C_pno'].shape[1]))

        # J_ij_kj: (ik|a_ij c_kj) = sum_Q ooL[i,k,Q] * ovL_a_ij[a,Q] ... no
        # Actually J_ij_kj = (ik|ac) where a in PNO_ij, c in PNO_kj
        # = sum_Q ooL[i,k,Q] * vvL_cross[a_ij, c_kj, Q]
        # We can build it from: J[a,c] = sum_Q ovL_i_ij[a,Q] * ovL_k_kj[c,Q]
        # Wait - that's (ia|kc) not (ik|ac). These differ:
        # (ia|kc) = ovL_i_ij @ ovL_k_kj.T  (exchange-type)
        # (ik|ac) = ooL[i,k] * vvL_ij_kj   (Coulomb-type, cross-domain)

        # In Psi4: J_ij_kj_ is precomputed as the Coulomb-type integral
        # In our code: J_ij_kj is from K_coul_cache
        J_kc = K_coul_cache.get((key, key_kj, i, k))
        if J_kc is not None:
            gamma += J_kc

        # C_tilde contribution (Term 4 only at T1=0)
        key_ik = (min(i,k), max(i,k))
        key_ki_ordered = (k, i)  # C_tilde uses ordered key
        # At T1=0, C_tilde = Term 4 = -0.5 * sum_l S@T2[li]@S@K[kl]@S
        # This is precomputed. For now, compute inline.
        n_ik = pno_spaces.get(key_ik, {}).get('C_pno', np.zeros((0,0))).shape[1] if key_ik in pno_spaces else 0
        if n_ik > 0:
            C_tilde_ki = np.zeros((n_ik, n_ik))
            for l in range(nocc):
                key_li = (min(l,i), max(l,i))
                key_kl = (min(k,l), max(k,l))
                if key_li not in t2_pno_all or key_kl not in pno_spaces:
                    continue
                t2_li_raw = t2_pno_all[key_li]
                if t2_li_raw is None or t2_li_raw.shape[0] == 0:
                    continue
                t2_li = t2_li_raw.T if l > i else t2_li_raw
                n_li = t2_li.shape[0]
                n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
                if n_kl == 0: continue

                ovL_k_kl = ovL_cache.get((key_kl, k))
                ovL_l_kl = ovL_cache.get((key_kl, l))
                if ovL_k_kl is None or ovL_l_kl is None: continue
                K_kl = ovL_k_kl @ ovL_l_kl.T

                S_ki_li = S(key_ik, key_li)
                S_li_kl = S(key_li, key_kl)
                S_kl_ki = S(key_kl, key_ik)
                if S_ki_li is None or S_li_kl is None or S_kl_ki is None:
                    continue
                C_tilde_ki -= 0.5 * S_ki_li @ t2_li @ S_li_kl @ K_kl @ S_kl_ki

            S_ij_ik = S(key, key_ik)
            S_ik_kj = S(key_ik, key_kj)
            gamma += S_ij_ik @ C_tilde_ki @ S_ik_kj

        S_kj_ij = S(key_kj, key)
        C -= gamma @ t2_kj.T @ S_kj_ij

    # P symmetrize C: Rn = 0.5*C + C^T (for i==j: 1.5*(C+C^T))
    if i == j:
        C_sym = 0.5 * C + C.T  # Rn_ij
        C_sym += (0.5 * C + C.T).T  # + Rn_ji.T = same for i==j
        # Actually for i==j: Rn = Rn_ij + Rn_ji.T where Rn_ji = Rn_ij
        # So total = 2 * (0.5*C + C.T)... hmm
        # Psi4: C_ij_total = 0.5*C + C^T; Rn_ij += C_ij_total
        # Then: R = R_sym + Rn_ij + Rn_ji.T
        # For i==j: Rn_ji = Rn_ij, so R = R_sym + Rn_ij + Rn_ij.T
        # = R_sym + (0.5C + C^T) + (0.5C + C^T)^T
        # = R_sym + 0.5C + C^T + 0.5C^T + C = R_sym + 1.5C + 1.5C^T
        C_total = 1.5 * (C + C.T)
    else:
        # Need C_ij and C_ji separately...
        # For simplicity, just return C_ij part
        C_total = 0.5 * C + C.T  # approximate for i!=j
    terms['C'] = C_total

    # === D: ring (delta) ===
    # Similar to C but with L = 2K-J and u = 2T-T^T
    D = np.zeros((n, n))
    for k in range(nocc):
        key_ik = (min(i,k), max(i,k))
        key_jk = (min(j,k), max(j,k))
        t2_jk = get_t2(j, k)
        if t2_jk is None or t2_jk.shape[0] == 0:
            continue
        u_jk = 2.0 * t2_jk - t2_jk.T

        n_ik = pno_spaces.get(key_ik, {}).get('C_pno', np.zeros((0,0))).shape[1] if key_ik in pno_spaces else 0
        if n_ik == 0: continue

        S_ij_jk = S(key, key_jk)
        S_jk_ik = S(key_jk, key_ik)
        U_jk_proj = S_ij_jk @ u_jk @ S_jk_ik  # (n_ij, n_ik)

        # D_tilde (Term 4 at T1=0)
        D_tilde_ik = np.zeros((n_ik, n_ik))
        for l in range(nocc):
            key_il = (min(i,l), max(i,l))
            key_lk = (min(l,k), max(l,k))
            if key_il not in t2_pno_all or key_lk not in pno_spaces:
                continue
            t2_il_raw = t2_pno_all[key_il]
            if t2_il_raw is None or t2_il_raw.shape[0] == 0:
                continue
            t2_il = t2_il_raw.T if i > l else t2_il_raw
            u_il = 2.0 * t2_il - t2_il.T
            n_il = t2_il.shape[0]
            n_lk = pno_spaces[key_lk]['C_pno'].shape[1]
            if n_lk == 0: continue

            ovL_l_lk = ovL_cache.get((key_lk, l))
            ovL_k_lk = ovL_cache.get((key_lk, k))
            if ovL_l_lk is None or ovL_k_lk is None: continue
            K_lk = ovL_l_lk @ ovL_k_lk.T
            L_lk = 2.0 * K_lk - K_lk.T

            S_ik_il = S(key_ik, key_il)
            S_il_lk = S(key_il, key_lk)
            S_lk_ik = S(key_lk, key_ik)
            D_tilde_ik += 0.5 * S_ik_il @ u_il @ S_il_lk @ L_lk @ S_lk_ik

        S_ij_ik = S(key, key_ik)
        D_temp = S_ij_ik @ D_tilde_ik @ U_jk_proj.T

        # Bold L term
        # K_aikc = (ia|kc) exchange in mixed domain
        ovL_i_ij = ovL_cache.get((key, i))
        ovL_k_jk = ovL_cache.get((key_jk, k))
        if ovL_i_ij is not None and ovL_k_jk is not None:
            K_aikc = ovL_i_ij @ ovL_k_jk.T  # (n_ij, n_jk) = (ia|kc)
        else:
            K_aikc = np.zeros((n, pno_spaces[key_jk]['C_pno'].shape[1]))

        J_aikc = K_coul_cache.get((key, key_jk, i, k))
        if J_aikc is None:
            J_aikc = np.zeros_like(K_aikc)

        L_aikc = 2.0 * K_aikc - J_aikc
        S_jk_ij = S(key_jk, key)
        D_temp += L_aikc @ u_jk.T @ S_jk_ij
        D_temp *= 0.5
        D += D_temp

    # For i==j: D_ij + D_ji.T = D + D.T (since D_ji = D with i↔j)
    if i == j:
        D_total = D + D.T
    else:
        D_total = D  # approximate for i!=j
    terms['D'] = D_total

    # === G: Fock oo coupling ===
    # G[a,b] = -sum_k T2[ik](a,b) * F_kj (off-diagonal only)
    G = np.zeros((n, n))
    for k in range(nocc):
        key_ik = (min(i,k), max(i,k))
        t2_ik = get_t2(i, k)
        if t2_ik is None or t2_ik.shape[0] == 0:
            continue

        # F_kj without diagonal (our convention)
        f_kj = F_lmo[k, j] - (F_lmo[j, j] if k == j else 0)

        S_ij_ik = S(key, key_ik)
        T_proj = S_ij_ik @ t2_ik @ S_ij_ik.T
        G -= T_proj * f_kj

    # For i==j: G_ij + G_ji.T
    if i == j:
        G_total = G + G.T
    else:
        G_total = G
    terms['G'] = G_total

    # Total R = K + A + B + E + C + D + G
    R = terms['K'] + terms['A'] + terms['B'] + terms['E']
    R += terms['C'] + terms['D'] + terms['G']
    terms['R'] = R

    return terms
