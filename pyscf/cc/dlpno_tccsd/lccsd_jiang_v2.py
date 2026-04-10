"""Jiang et al. T2 residual — exact implementation following Psi4 ccsd.cc.

This is a clean reimplementation that follows the Psi4 code line-by-line,
using two buffers (R for symmetric terms, Rn for non-symmetric C/D/G terms)
and final P̂ symmetrization: R_final = R + Rn[ij] + Rn[ji].T

All integrals are BARE. T1 enters only through:
- i_Qa_t1 (dressed ovL for exchange, Eq 92)
- B_tilde (dressed ooL for Woooo, Eq 82)
- Fab_ (dressed Fock, Eqs 85/97/101)
- G_tilde (dressed Fock oo, Eq 86)
- C_tilde / D_tilde (gamma/delta intermediates, Eqs 83-84)
"""
import numpy as np
from pyscf.ao2mo import _ao2mo
from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair


def compute_residual_v2(
        i, j, t2_pno_all, pno_spaces, nocc,
        F_lmo, s1e, with_df, eps_lmo,
        # Pre-built intermediates (built once per iteration):
        ovL_bare,       # dict: (pair_key, m) -> (n_pno, naux)
        ooL_bare,       # (nocc, nocc, naux)
        S_pno_cache,    # dict: (key1, key2) -> overlap matrix
        K_coul_cache,   # dict: (key_ij, key_pk, occ1, occ2) -> (n_ij, n_pk)
        K_pno_bare,     # dict: pair_key -> (n_pno, n_pno) bare exchange
        # T1-dressed intermediates:
        ovL_dressed,    # dict: (pair_key, m) -> dressed ovL (Eq 92)
        ooL_dressed,    # (nocc, nocc, naux) asymmetric (Eq 91)
        B_tilde,        # (nocc, nocc) dressed beta for Woooo
        Fab,            # dict: pair_key -> (n_pno, n_pno) double-dressed Fock vv
        G_tilde,        # (nocc, nocc) double-dressed Fock oo
        C_tilde_cache,  # dict: (k,i) -> (n_ki, n_ki) gamma intermediate
        D_tilde_cache,  # dict: (i,k) -> (n_ik, n_ik) delta intermediate
        # Mixed-domain integrals:
        J_ij_kj,        # dict: (ij, k) -> (n_ij, n_kj) bare Coulomb (ik|a_ij c_kj)
        K_ij_kj,        # dict: (ij, k) -> (n_ij, n_kj) bare exchange (ia_ij|kc_kj)
        t1_pno=None,
        ladder_precomputed=None,
        K_dressed_override=None,
        cc_ints=None,
):
    """Compute T2 residual following Psi4 ccsd.cc lines 2052-2221 exactly.

    Returns P̂-symmetrized R_ij = R_sym + Rn[ij] + Rn[ji].T
    """
    key = (min(i, j), max(i, j))
    data = pno_spaces[key]
    C_pno_ij = data['C_pno']
    n_pno = C_pno_ij.shape[1]

    if n_pno == 0:
        return np.zeros((0, 0))

    def _get_S(key_other):
        if S_pno_cache is not None:
            S = S_pno_cache.get((key, key_other))
            if S is not None:
                return S
        return C_pno_ij.T @ (s1e @ pno_spaces[key_other]['C_pno'])

    def _get_S2(key_a, key_b):
        """Get S overlap between any two pair keys, with identity fallback."""
        if key_a == key_b:
            return np.eye(pno_spaces[key_a]['C_pno'].shape[1])
        if S_pno_cache is not None:
            S = S_pno_cache.get((key_a, key_b))
            if S is not None:
                return S
        return pno_spaces[key_a]['C_pno'].T @ (s1e @ pno_spaces[key_b]['C_pno'])

    t2_ij = t2_pno_all[key]

    # === Symmetric buffer (K̃, A, B, E) ===
    R_sym = np.zeros((n_pno, n_pno))
    _DEBUG_PTERM = getattr(compute_residual_v2, '_debug_pterm', False) or \
                   getattr(compute_residual_v2, '_debug_pterm_all', False)

    def _rms(M):
        return float(np.sqrt(np.mean(M*M)))

    # --- K̃ (Eq 75): dressed exchange ---
    K_term = np.zeros((n_pno, n_pno))
    if K_dressed_override is not None:
        K_term += K_dressed_override
    else:
        ovL_i_d = ovL_dressed.get((key, i))
        ovL_j_d = ovL_dressed.get((key, j))
        if ovL_i_d is not None and ovL_j_d is not None:
            K_term += ovL_i_d @ ovL_j_d.T
    R_sym += K_term
    if _DEBUG_PTERM:
        print(f"  PTERM pair({i},{j}): K_rms={_rms(K_term):.12f} R_K={_rms(R_sym):.12f}")

    # --- A (Eq 76): dressed ladder ---
    # B̃_{ab} = B_{ab} - Σ_k T1_all[k,a]*ovL_k[b,Q] (Eq 93)
    T1_all_ij = np.zeros((nocc, n_pno))
    if t1_pno is not None:
        for kk in range(nocc):
            T1_all_ij[kk] = _project_t1_to_pair(
                t1_pno, kk, key, S_pno_cache, pno_spaces)

    t1_i_pno = T1_all_ij[i] if t1_pno else np.zeros(n_pno)
    t1_j_pno = T1_all_ij[j] if t1_pno else np.zeros(n_pno)
    tau_ij = t2_ij + np.outer(t1_i_pno, t1_j_pno)

    A_term = np.zeros((n_pno, n_pno))
    if ladder_precomputed is not None and key in ladder_precomputed:
        A_term = ladder_precomputed[key]
        R_sym += A_term
    elif with_df is not None:
        # Fallback: compute ladder inline (per-pair DF loop)
        naux = ooL_bare.shape[2]
        ovL_stacked = np.stack(
            [ovL_bare[(key, k)] for k in range(nocc)])  # (nocc, n_pno, naux)
        corr_full = np.einsum('ka,kcQ->acQ', T1_all_ij, ovL_stacked)

        mo = np.asfortranarray(C_pno_ij)
        ijslice = (0, n_pno, 0, n_pno)
        ladder = np.zeros((n_pno, n_pno))
        buf = None
        aux_off = 0
        for Lpq in with_df.loop():
            nL = Lpq.shape[0]
            buf = _ao2mo.nr_e2(Lpq, mo, ijslice, aosym='s2', out=buf)
            B_L = buf.reshape(nL, n_pno, n_pno)
            corr_batch = corr_full[:, :, aux_off:aux_off+nL].transpose(2, 0, 1)
            B_tilde_L = B_L - corr_batch  # dressed B̃_{ab}
            X_L = np.einsum('Lac,cd->Lad', B_tilde_L, tau_ij)
            ladder += np.einsum('Lad,Lbd->ab', X_L, B_tilde_L)
            aux_off += nL
        R_sym += ladder
        A_term = ladder
    if _DEBUG_PTERM:
        print(f"  PTERM pair({i},{j}): A_rms={_rms(A_term):.12f} R_KA={_rms(R_sym):.12f}")

    # --- B (Eq 77/82): Woooo with dressed β ---
    # Psi4 uses BARE T2 (T_iajb_), NOT tau. See ccsd.cc line 2137.
    B_term = np.zeros((n_pno, n_pno))
    for key_kl, t2_kl in t2_pno_all.items():
        if t2_kl is None or t2_kl.shape[0] == 0:
            continue
        k, l = key_kl

        S_proj = _get_S(key_kl)
        t2_kl_proj = S_proj @ t2_kl @ S_proj.T

        beta_kl = B_tilde[k, l]
        if k != l:
            beta_lk = B_tilde[l, k]
            B_term += beta_kl * t2_kl_proj + beta_lk * t2_kl_proj.T
        else:
            B_term += beta_kl * t2_kl_proj
    R_sym += B_term
    if _DEBUG_PTERM:
        print(f"  PTERM pair({i},{j}): B_rms={_rms(B_term):.12f} R_KAB={_rms(R_sym):.12f}")

    # --- E (Eq 80): t2 × F̃̃_{ab} ---
    # E_tilde = Fab_[ij] - Σ_kl S @ u_kl × K_kl @ S
    # Psi4: starts with Fab_[ij], subtracts u×K (bare K_iajb)
    # Fab includes diag(e_pno) per Eq 101. But our code puts e_pno in the
    # denominator D, not the numerator. Subtract the diagonal to avoid
    # double-counting. (Psi4 uses T -= R/D iterative update which handles
    # this differently.)
    Fab_ij = Fab.get(key, np.zeros((n_pno, n_pno))).copy()
    e_pno = data['e_pno']
    # Match Psi4 exactly: keep full Fab including diag(e_pno).
    # The (e_a+e_b)*T term enters R; balanced by -(F_ii+F_jj)*T from G.
    # Update is T -= R/D_psi4, so R = 0 at convergence (full Psi4 residual).
    E_tilde = Fab_ij.copy()
    # Psi4 E term: subtract u×K from E_tilde (Eq 85: F̃̃ = F̃ - u×K)
    # This handles the fvv T2 dressing. C_tilde/D_tilde Term 4 handles
    # a DIFFERENT T2 contribution (through the C/D ring terms).
    for key_kl, t2_kl in t2_pno_all.items():
        if t2_kl is None or t2_kl.shape[0] == 0:
            continue
        k, l = key_kl
        u_kl = 2.0 * t2_kl - t2_kl.T
        if cc_ints is not None:
            from pyscf.cc.dlpno_tccsd.local_df import get_local_K
            _K = get_local_K(cc_ints, key_kl, k, l)
            if _K is not None:
                K_kl = _K
            else:
                ovL_k_kl = ovL_bare.get((key_kl, k))
                ovL_l_kl = ovL_bare.get((key_kl, l))
                if ovL_k_kl is None or ovL_l_kl is None:
                    continue
                K_kl = ovL_k_kl @ ovL_l_kl.T
        else:
            ovL_k_kl = ovL_bare.get((key_kl, k))
            ovL_l_kl = ovL_bare.get((key_kl, l))
            if ovL_k_kl is None or ovL_l_kl is None:
                continue
            K_kl = ovL_k_kl @ ovL_l_kl.T
        S_kl_ij = _get_S(key_kl)
        # (k,l) contribution
        E_tilde -= S_kl_ij @ (u_kl @ K_kl.T) @ S_kl_ij.T
        # (l,k) contribution for off-diagonal
        if k != l:
            E_tilde -= S_kl_ij @ ((2.0*t2_kl.T - t2_kl) @ K_kl) @ S_kl_ij.T

    if _DEBUG_PTERM:
        print(f"  PTERM pair({i},{j}): Etilde_rms={_rms(E_tilde):.12f}")
        # Decompose: Fab and the subtraction
        E_tilde_diag = np.diag(E_tilde)
        Fab_diag = np.diag(Fab_ij)
        sub = Fab_ij - E_tilde
        print(f"  PTERM_DECOMP pair({i},{j}): Fab_rms={_rms(Fab_ij):.12f} "
              f"Fab_diag_first={Fab_diag[0]:.10f} sub_rms={_rms(sub):.12f}")
    # Apply: R += t2 @ E_tilde.T + E_tilde @ t2 (Psi4 lines 2146-2147)
    E_term = t2_ij @ E_tilde.T + E_tilde @ t2_ij
    R_sym += E_term
    if _DEBUG_PTERM:
        print(f"  PTERM pair({i},{j}): E_rms={_rms(E_term):.12f} R_KABE={_rms(R_sym):.12f}")

    # === Non-symmetric terms (C, D, G) ===
    # Compute C_ij and C_ji in one pass, then form the full P̂ result.
    # P̂(0.5*C + C_ji) = 0.5*(C_ij + C_ij.T) + (C_ji + C_ji.T) for i≠j
    #                  = 1.5*(C + C.T) for i=j (since C_ji = C_ij)
    Rn_ij = np.zeros((n_pno, n_pno))

    # --- C (Eq 78) ---
    _SKIP_C = getattr(compute_residual_v2, '_skip_C', False)
    if C_tilde_cache is not None and not _SKIP_C:
        C_ij = np.zeros((n_pno, n_pno))
        C_ji = np.zeros((n_pno, n_pno))
        for k in range(nocc):
            # --- C_ij: gamma(ki) × t2(kj) ---
            key_ik = (min(i, k), max(i, k))
            key_kj = (min(k, j), max(k, j))
            if key_kj in t2_pno_all and t2_pno_all[key_kj] is not None:
                t2_kj_raw = t2_pno_all[key_kj]
                if t2_kj_raw.shape[0] > 0:
                    t2_kj = t2_kj_raw.T if k > j else t2_kj_raw
                    n_kj = t2_kj.shape[0]
                    gamma_ij = np.zeros((n_pno, n_kj))
                    # Bold: J(ik|a_ij c_kj)
                    if J_ij_kj:
                        J_b = J_ij_kj.get((key, k))
                        if J_b is not None:
                            gamma_ij += J_b
                    # C_tilde(ki)
                    ct = C_tilde_cache.get((k, i))
                    if ct is not None:
                        S_ij_ik = _get_S(key_ik)
                        S_ik_kj = _get_S2(key_ik, key_kj)
                        if S_ij_ik is not None and S_ik_kj is not None:
                            gamma_ij += S_ij_ik @ ct @ S_ik_kj
                    S_kj_ij = _get_S(key_kj)
                    C_ij -= gamma_ij @ t2_kj.T @ S_kj_ij.T

            # --- C_ji: gamma(kj) × t2(ki) ---
            key_jk = (min(j, k), max(j, k))
            key_ki = (min(k, i), max(k, i))
            if key_ki in t2_pno_all and t2_pno_all[key_ki] is not None:
                t2_ki_raw = t2_pno_all[key_ki]
                if t2_ki_raw.shape[0] > 0:
                    t2_ki = t2_ki_raw.T if k > i else t2_ki_raw
                    n_ki = t2_ki.shape[0]
                    gamma_ji = np.zeros((n_pno, n_ki))
                    # Bold: J(jk|a_ij c_ki) = KC[(key_ij, key_ki, j, k)]
                    if K_coul_cache:
                        J_b_ji = K_coul_cache.get((key, key_ki, j, k))
                        if J_b_ji is not None:
                            gamma_ji += J_b_ji
                    # C_tilde(kj)
                    ct_j = C_tilde_cache.get((k, j))
                    if ct_j is not None:
                        S_ij_jk = _get_S(key_jk)
                        S_jk_ki = _get_S2(key_jk, key_ki)
                        if S_ij_jk is not None and S_jk_ki is not None:
                            gamma_ji += S_ij_jk @ ct_j @ S_jk_ki
                    S_ki_ij = _get_S(key_ki)
                    C_ji -= gamma_ji @ t2_ki.T @ S_ki_ij.T

        # P̂: Rn[ij] + Rn[ji].T where Rn[ij] = 0.5*C_ij + C_ij.T
        Rn_C_ij = 0.5 * C_ij + C_ij.T
        Rn_C_ji = 0.5 * C_ji + C_ji.T
        C_term = Rn_C_ij + Rn_C_ji.T
        Rn_ij += C_term
        if _DEBUG_PTERM:
            # Print C_ij and C_ji separately to compare with Psi4 (which prints
            # the unsymmetrized C_ij per ordered pair).
            print(f"  PTERM pair({i},{j}): C_ij_unsym_rms={_rms(C_ij):.12f} C_ji_unsym_rms={_rms(C_ji):.12f}")
    else:
        C_term = np.zeros((n_pno, n_pno))

    # --- D (Eq 79): antisymmetric ring with delta ---
    _SKIP_D = getattr(compute_residual_v2, '_skip_D', False)
    if D_tilde_cache is not None and not _SKIP_D:
        D_ij = np.zeros((n_pno, n_pno))
        D_ji = np.zeros((n_pno, n_pno))
        for k in range(nocc):
            key_ik = (min(i, k), max(i, k))
            key_jk = (min(j, k), max(j, k))

            # D_ij: delta(ik) × u(jk)
            if key_jk in t2_pno_all and t2_pno_all[key_jk] is not None:
                t2_jk_raw = t2_pno_all[key_jk]
                if t2_jk_raw.shape[0] > 0:
                    t2_jk = t2_jk_raw.T if j > k else t2_jk_raw
                    u_jk = 2.0 * t2_jk - t2_jk.T
                    S_ij_jk = _get_S(key_jk)
                    S_jk_ik = _get_S2(key_jk, key_ik)
                    if S_ij_jk is not None and S_jk_ik is not None:
                        U_jk_proj = S_ij_jk @ u_jk @ S_jk_ik
                        D_temp = np.zeros((n_pno, n_pno))
                        dt = D_tilde_cache.get((i, k))
                        if dt is not None:
                            S_ij_ik = _get_S(key_ik)
                            D_temp += S_ij_ik @ dt @ U_jk_proj.T
                        # Bold: L = 2K-J with mixed domains
                        K_b = K_ij_kj.get((key, k)) if K_ij_kj else None
                        J_b = J_ij_kj.get((key, k)) if J_ij_kj else None
                        if K_b is not None and J_b is not None:
                            D_temp += (2.0*K_b - J_b) @ u_jk.T @ S_ij_jk.T
                        D_ij += 0.5 * D_temp

            # D_ji: delta(jk) × u(ik)
            if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
                t2_ik_raw = t2_pno_all[key_ik]
                if t2_ik_raw.shape[0] > 0:
                    t2_ik = t2_ik_raw.T if i > k else t2_ik_raw
                    u_ik = 2.0 * t2_ik - t2_ik.T
                    S_ij_ik2 = _get_S(key_ik)
                    S_ik_jk = _get_S2(key_ik, key_jk)
                    if S_ij_ik2 is not None and S_ik_jk is not None:
                        U_ik_proj = S_ij_ik2 @ u_ik @ S_ik_jk
                        D_temp_j = np.zeros((n_pno, n_pno))
                        dt_j = D_tilde_cache.get((j, k))
                        if dt_j is not None:
                            S_ij_jk2 = _get_S(key_jk)
                            D_temp_j += S_ij_jk2 @ dt_j @ U_ik_proj.T
                        # Bold for D_ji: L(jk|a_ij c_ik) = 2K-J cross-domain
                        # Use local DF cross-integrals (J_ji_ki, K_ji_ki) when available
                        _ci_ij = cc_ints.get(key) if cc_ints is not None else None
                        K_ji_local = _ci_ij.get('K_ji_ki', {}).get((key, k)) if _ci_ij else None
                        J_ji_local = _ci_ij.get('J_ji_ki', {}).get((key, k)) if _ci_ij else None
                        if K_ji_local is not None and J_ji_local is not None:
                            D_temp_j += (2.0*K_ji_local - J_ji_local) @ u_ik.T @ S_ij_ik2.T
                        else:
                            # Fallback to global DF
                            ovL_j_d = ovL_bare.get((key, j))
                            ovL_k_ik_d = ovL_bare.get((key_ik, k))
                            J_ji = K_coul_cache.get((key, key_ik, j, k)) if K_coul_cache else None
                            if ovL_j_d is not None and ovL_k_ik_d is not None and J_ji is not None:
                                K_ji = ovL_j_d @ ovL_k_ik_d.T
                                D_temp_j += (2.0*K_ji - J_ji) @ u_ik.T @ S_ij_ik2.T
                        D_ji += 0.5 * D_temp_j

        D_term = D_ij + D_ji.T
        Rn_ij += D_term
        if _DEBUG_PTERM:
            print(f"  PTERM pair({i},{j}): D_ij_unsym_rms={_rms(D_ij):.12f} D_ji_unsym_rms={_rms(D_ji):.12f}")
    else:
        D_term = np.zeros((n_pno, n_pno))

    # --- G (Eq 81): Fock oo coupling ---
    _SKIP_G = getattr(compute_residual_v2, '_skip_G', False)
    G_ij = np.zeros((n_pno, n_pno))
    G_ji = np.zeros((n_pno, n_pno))
    for k in (range(nocc) if not _SKIP_G else []):
        key_ik = (min(i, k), max(i, k))
        key_jk = (min(j, k), max(j, k))
        if key_ik in t2_pno_all and t2_pno_all[key_ik] is not None:
            t2_ik_raw = t2_pno_all[key_ik]
            if t2_ik_raw.shape[0] > 0:
                t2_ik = t2_ik_raw.T if i > k else t2_ik_raw
                S_ij_ik = _get_S(key_ik)
                G_ij -= (S_ij_ik @ t2_ik @ S_ij_ik.T) * G_tilde[k, j]
        if key_jk in t2_pno_all and t2_pno_all[key_jk] is not None:
            t2_jk_raw = t2_pno_all[key_jk]
            if t2_jk_raw.shape[0] > 0:
                t2_jk = t2_jk_raw.T if j > k else t2_jk_raw
                S_ij_jk = _get_S(key_jk)
                G_ji -= (S_ij_jk @ t2_jk @ S_ij_jk.T) * G_tilde[k, i]
    G_term = G_ij + G_ji.T
    Rn_ij += G_term

    if _DEBUG_PTERM:
        print(f"  PTERM pair({i},{j}): G_ij_unsym_rms={_rms(G_ij):.12f} G_ji_unsym_rms={_rms(G_ji):.12f}")
        R_total = R_sym + Rn_ij
        print(f"  PTERM pair({i},{j}): Rn={_rms(Rn_ij):.12f} R_total={_rms(R_total):.12f}")
    if _DEBUG_PTERM and not getattr(compute_residual_v2, '_dump_done', False) and \
       not getattr(compute_residual_v2, '_debug_pterm_all', False):
        if i == 0 and j == 0:
            print(f"  DUMP_R2 pair(0,0) npno={n_pno}")
            for a in range(min(n_pno, 5)):
                for b in range(min(n_pno, 5)):
                    print(f"  DUMP_R2[{a},{b}]={R_total[a,b]:.15e}")
            print(f"  DUMP_T2 pair(0,0)")
            for a in range(min(n_pno, 5)):
                for b in range(min(n_pno, 5)):
                    print(f"  DUMP_T2[{a},{b}]={t2_ij[a,b]:.15e}")
            print(f"  DUMP_K pair(0,0)")
            K_dump = K_dressed_override if K_dressed_override is not None else K_term
            for a in range(min(n_pno, 5)):
                for b in range(min(n_pno, 5)):
                    print(f"  DUMP_K[{a},{b}]={K_dump[a,b]:.15e}")
            print(f"  DUMP_EPNO pair(0,0)")
            for a in range(min(n_pno, 10)):
                print(f"  DUMP_EPNO[{a}]={e_pno[a]:.15e}")
            compute_residual_v2._dump_done = True

    # === Section 5b: Quadratic T2 dressing of ring ===
    # NOT used when C_tilde/D_tilde include Term 4 (Jiang's formulation).
    # Kept for reference / ORCA-compatible mode.
    _use_5b = getattr(compute_residual_v2, '_use_5b', False)
    if with_df is not None and _use_5b:
        R_dress_ij = np.zeros((n_pno, n_pno))
        R_dress_ji = np.zeros((n_pno, n_pno))
        for k in range(nocc):
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
                    for l in range(nocc):
                        key_jl = (min(j, l), max(j, l))
                        if key_jl not in t2_pno_all or t2_pno_all[key_jl] is None:
                            continue
                        n_jl = pno_spaces[key_jl]['C_pno'].shape[1]
                        if n_jl == 0:
                            continue
                        t2_jl_raw = t2_pno_all[key_jl]
                        t2_jl = t2_jl_raw.T if j > l else t2_jl_raw
                        S_ij_jl = _get_S(key_jl)
                        _ovL_ik_l = ovL_bare.get((key_ik, l))
                        _ovL_jl_k = ovL_bare.get((key_jl, k))
                        if _ovL_ik_l is None or _ovL_jl_k is None:
                            continue
                        voov_lk = _ovL_ik_l @ _ovL_jl_k.T
                        W_k += 0.5 * voov_lk @ t2_jl @ S_ij_jl.T
                        _ovL_ik_k = ovL_bare.get((key_ik, k))
                        _ovL_jl_l = ovL_bare.get((key_jl, l))
                        if _ovL_ik_k is None or _ovL_jl_l is None:
                            continue
                        voov_kl = _ovL_ik_k @ _ovL_jl_l.T
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
                    for l in range(nocc):
                        key_il = (min(i, l), max(i, l))
                        if key_il not in t2_pno_all or t2_pno_all[key_il] is None:
                            continue
                        n_il = pno_spaces[key_il]['C_pno'].shape[1]
                        if n_il == 0:
                            continue
                        t2_il_raw = t2_pno_all[key_il]
                        t2_il = t2_il_raw.T if i > l else t2_il_raw
                        S_ij_il = _get_S(key_il)
                        _ovL_jk_l = ovL_bare.get((key_jk, l))
                        _ovL_il_k = ovL_bare.get((key_il, k))
                        if _ovL_jk_l is None or _ovL_il_k is None:
                            continue
                        voov_lk_ji = _ovL_jk_l @ _ovL_il_k.T
                        W_k_ji += 0.5 * voov_lk_ji @ t2_il @ S_ij_il.T
                        _ovL_jk_k = ovL_bare.get((key_jk, k))
                        _ovL_il_l = ovL_bare.get((key_il, l))
                        if _ovL_jk_k is None or _ovL_il_l is None:
                            continue
                        voov_kl_ji = _ovL_jk_k @ _ovL_il_l.T
                        VOov_kl_ji = voov_kl_ji - 0.5 * voov_lk_ji
                        tau_ring_li = 2.0 * t2_il.T - t2_il
                        V_k_ji += 0.5 * VOov_kl_ji @ tau_ring_li @ S_ij_il.T
                    R_dress_ji += S_ij_jk_d @ (theta_jk_d @ V_k_ji)
                    R_dress_ji += 0.5 * S_ij_jk_d @ (t2_jk_d.T @ W_k_ji)
                    R_dress_ij += S_ij_jk_d @ (t2_jk_d.T @ W_k_ji)
        R_sym += R_dress_ij + R_dress_ji.T

    # === R_final = R_sym + Rn (already fully P̂-symmetrized) ===
    return R_sym + Rn_ij
