"""Per-pair local DF integral computation for DLPNO-CCSD.

Matches Psi4's architecture exactly:
  1. compute_cc_integrals(): precompute per-pair fitted 2-index and 3-index intermediates
  2. t1_ints(): dress stored intermediates with T1 each iteration
  3. t1_fock(): build dressed Fock matrices from stored intermediates
  4. Residual: uses only precomputed intermediates

All integrals for pair (ij) use pair (ij)'s local aux domain with J_local^{-1/2}.
"""

import numpy as np


def get_local_ovL(cc_ints, pair_key, lmo_idx):
    """Get locally-fitted ovL for LMO lmo_idx in pair pair_key's PNO basis.

    Returns (npno, n_local) or None if not available.
    """
    ci = cc_ints.get(pair_key)
    if ci is None:
        return None
    return ci['Qma'][:, lmo_idx, :].T  # (npno, n_local)


def get_local_ooL_vec(cc_ints, k, l, pair_key):
    """Get locally-fitted ooL[k,l] in pair pair_key's local fitting.

    Returns (n_local,) or None if not available.
    """
    ci = cc_ints.get(pair_key)
    if ci is None:
        return None
    i_lmo, j_lmo = pair_key  # i_lmo = min, j_lmo = max
    # We need (Q|kl) = raw_oo[k, l, aux_idx] @ jhi
    # From stored: i_Qk[:, m] = fitted (Q | m * i_lmo), j_Qk[:, m] = fitted (Q | m * j_lmo)
    if l == i_lmo:
        return ci['i_Qk'][:, k]  # fitted (Q | k * i_lmo) = (Q | k * l)
    elif l == j_lmo:
        return ci['j_Qk'][:, k]  # fitted (Q | k * j_lmo) = (Q | k * l)
    else:
        # l is neither i nor j of the pair — can't get from stored intermediates
        # This shouldn't happen for C_tilde/D_tilde which access ooL[k,i] or ooL[i,k]
        # where i is one of the pair's LMOs
        return None


def get_local_K(cc_ints, pair_key, lmo1, lmo2):
    """Get locally-fitted K = ovL_lmo1 @ ovL_lmo2.T for pair pair_key.

    Returns (npno, npno) or None.
    """
    ci = cc_ints.get(pair_key)
    if ci is None:
        return None
    ovL1 = ci['Qma'][:, lmo1, :].T  # (npno, n_local)
    ovL2 = ci['Qma'][:, lmo2, :].T  # (npno, n_local)
    return ovL1 @ ovL2.T  # (npno, npno)


def compute_cc_integrals(mol, auxmol, C_lmo, pno_spaces, pair_aux_idx,
                         j2c, keys, nocc):
    """Precompute ALL per-pair locally-fitted intermediates.

    Matches Psi4 ccsd.cc compute_cc_integrals() lines 1095-1470.
    Builds raw AO 3-center integrals ONCE, then transforms and fits per pair.

    Args:
        mol, auxmol: PySCF Mole objects.
        C_lmo: (nao, nocc) LMO coefficients (bare or T1-dressed).
        pno_spaces: dict pair_key -> {'C_pno': (nao, npno), 'e_pno': ...}
        pair_aux_idx: dict pair_key -> np.array of local aux indices.
        j2c: (naux, naux) full Coulomb metric.
        keys: list of pair keys (strong pairs).
        nocc: number of active occupied orbitals.

    Returns:
        cc_ints: dict pair_key -> dict of intermediates:
            'K_iajb': (npno, npno) exchange
            'K_mnij': (nocc, nocc) oo exchange (full occ, not local domain)
            'K_bar_ij': (nocc, npno) = (mi|ja) K_bar for LMO i side
            'K_bar_ji': (nocc, npno) = (mj|ia) K_bar for LMO j side
            'K_bar_chem': (nocc, npno) = Σ_Q q_pair[Q]*Qma[Q,m,a]
            'J_ijab': (npno, npno) Coulomb vv
            'i_Qa': (naux_local, npno) bare (Q|ia) fitted
            'j_Qa': (naux_local, npno) bare (Q|ja) fitted
            'i_Qk': (naux_local, nocc) bare (Q|ik) fitted
            'j_Qk': (naux_local, nocc) bare (Q|jk) fitted
            'Qma': (naux_local, nocc, npno) ovL per Q fitted
            'Qab': (naux_local, npno, npno) vvL per Q fitted
            'n_local': int
            'aux_idx': np.array
    """
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()

    # Compute raw AO 3-center integrals ONCE: (nao, nao, naux)
    pmol = mol + auxmol
    shls_slice = (0, mol.nbas, 0, mol.nbas, mol.nbas, mol.nbas + auxmol.nbas)
    raw_3c = pmol.intor('int3c2e', shls_slice=shls_slice)

    # Precompute raw oo integrals: raw_oo[k,l,Q] = C_lmo^T @ raw_3c @ C_lmo
    # First half: tmp_oo[mu, Q, k] = raw_3c @ C_lmo
    tmp_oo = np.tensordot(raw_3c, C_lmo, axes=([1], [0]))  # (nao, naux, nocc)
    raw_oo = np.tensordot(C_lmo, tmp_oo, axes=([0], [0]))   # (nocc, naux, nocc)
    raw_oo = raw_oo.transpose(0, 2, 1)  # (nocc, nocc, naux) = raw(k, l, Q)
    del tmp_oo

    cc_ints = {}
    for key in keys:
        i, j = key
        C_pno = pno_spaces[key]['C_pno']
        npno = C_pno.shape[1]
        if npno == 0:
            cc_ints[key] = None
            continue
        if key not in pair_aux_idx:
            cc_ints[key] = None
            continue

        aux_idx = pair_aux_idx[key]
        n_local = len(aux_idx)

        # Local J^{-1/2}
        j2c_local = j2c[np.ix_(aux_idx, aux_idx)]
        eigvals, eigvecs = np.linalg.eigh(j2c_local)
        keep = eigvals > 1e-14  # Match Psi4's eigenvalue truncation
        jhi = (eigvecs[:, keep] * (1.0 / np.sqrt(eigvals[keep]))) @ eigvecs[:, keep].T

        # Half-transform raw_3c with C_pno: tmp[mu, Q, a] = raw_3c @ C_pno
        tmp_pno = np.tensordot(raw_3c, C_pno, axes=([1], [0]))  # (nao, naux, npno)

        # --- Build raw integrals and apply local J^{-1/2} ---

        # q_iv[Q, a] = C_lmo[:,i]^T @ tmp_pno → fitted with J^{-1/2}
        raw_iv = np.tensordot(C_lmo[:, i], tmp_pno, axes=([0], [0]))  # (naux, npno)
        raw_jv = np.tensordot(C_lmo[:, j], tmp_pno, axes=([0], [0]))
        q_iv = jhi @ raw_iv[aux_idx]  # (n_local, npno)
        q_jv = jhi @ raw_jv[aux_idx]

        # q_io[Q, k] = raw_oo[k, i, Q] fitted
        q_io = jhi @ raw_oo[:, i, :][:, aux_idx].T  # (n_local, nocc)
        q_jo = jhi @ raw_oo[:, j, :][:, aux_idx].T

        # q_pair[Q] = raw_oo[i, j, Q] fitted
        q_pair = jhi @ raw_oo[i, j, aux_idx]  # (n_local,)

        # Qma[Q, m, a] = C_lmo^T @ tmp_pno, fitted
        raw_ma = np.tensordot(C_lmo, tmp_pno, axes=([0], [0]))  # (nocc, naux, npno)
        Qma = np.zeros((n_local, nocc, npno))
        for m in range(nocc):
            Qma[:, m, :] = jhi @ raw_ma[m, aux_idx, :]  # (n_local, npno)

        # Qab[Q, a, b] = C_pno^T @ tmp_pno, fitted
        raw_ab = np.tensordot(C_pno, tmp_pno, axes=([0], [0]))  # (npno, naux, npno)
        Qab = np.zeros((n_local, npno, npno))
        for a in range(npno):
            Qab[:, a, :] = jhi @ raw_ab[a, aux_idx, :]

        del tmp_pno, raw_iv, raw_jv, raw_ma, raw_ab

        # --- Build 2-index intermediates ---
        K_iajb = q_iv.T @ q_jv           # (npno, npno)
        K_mnij = q_io.T @ q_jo           # (nocc, nocc)
        K_bar_ij = q_io.T @ q_jv         # (nocc, npno) = (mi|ja)
        K_bar_ji = q_jo.T @ q_iv         # (nocc, npno) = (mj|ia)
        K_bar_chem = np.einsum('Q,Qma->ma', q_pair, Qma)  # (nocc, npno)
        J_ijab = np.einsum('Q,Qab->ab', q_pair, Qab)      # (npno, npno)

        # J_ij_kj and K_ij_kj (cross-pair integrals)
        # J_ij_kj[(key, k)][a_ij, c_kj] = Σ_Q q_pair[Q] * cross_vvL[Q, a_ij, c_kj]
        # K_ij_kj[(key, k)][a_ij, c_kj] = Σ_Q q_iv[Q, a_ij] * q_kv_kj[Q, c_kj]
        J_ij_kj = {}
        K_ij_kj_dict = {}
        key_set = set(keys)
        for k in range(nocc):
            key_kj = (min(k, j), max(k, j))
            if key_kj not in key_set or key_kj not in pno_spaces:
                continue
            C_pno_kj = pno_spaces[key_kj]['C_pno']
            n_kj = C_pno_kj.shape[1]
            if n_kj == 0:
                continue

            # Cross-pair vvL: C_pno_kj^T @ tmp_pno_for_ij → but tmp_pno is deleted
            # Recompute cross vvL from raw_3c
            tmp_kj = np.tensordot(raw_3c, C_pno_kj, axes=([1], [0]))  # (nao, naux, n_kj)
            raw_cross = np.tensordot(C_pno, tmp_kj, axes=([0], [0]))   # (npno, naux, n_kj)
            # J: Σ_Q ooL_fitted[i,k,Q] * cross_fitted[Q, a, c]
            cross_local = jhi @ raw_cross[:, aux_idx, :].transpose(1, 0, 2).reshape(
                len(aux_idx), npno * n_kj)
            cross_fitted = cross_local.reshape(n_local, npno, n_kj)
            # ooL[i,k] fitted with pair (ij)'s J^{-1/2}
            q_ik = jhi @ raw_oo[i, k, aux_idx]  # (n_local,)
            J_ij_kj[(key, k)] = np.einsum('Q,Qac->ac', q_ik, cross_fitted)

            # K: Σ_Q q_iv[Q, a] * q_kv_kj[Q, c]
            # q_kv_kj[Q, c] = C_lmo[:,k]^T @ tmp_kj, fitted with pair ij's J
            raw_kv_kj = np.tensordot(C_lmo[:, k], tmp_kj, axes=([0], [0]))  # (naux, n_kj)
            q_kv_kj = jhi @ raw_kv_kj[aux_idx]  # (n_local, n_kj)
            K_ij_kj_dict[(key, k)] = q_iv.T @ q_kv_kj  # (npno, n_kj)
            del tmp_kj, raw_cross

        # Also build cross-pair integrals from j-side: (j*a_ij | k*c_ki)
        # Needed for D_ji bold term, matching Psi4's K_ij_kj_[ji][k].
        # These use pair (i,j)'s local fitting (same J^{-1/2}).
        J_ji_ki = {}
        K_ji_ki_dict = {}
        for k in range(nocc):
            key_ki = (min(k, i), max(k, i))
            if key_ki not in key_set or key_ki not in pno_spaces:
                continue
            C_pno_ki = pno_spaces[key_ki]['C_pno']
            n_ki = C_pno_ki.shape[1]
            if n_ki == 0:
                continue
            tmp_ki = np.tensordot(raw_3c, C_pno_ki, axes=([1], [0]))
            raw_cross_ji = np.tensordot(C_pno, tmp_ki, axes=([0], [0]))
            cross_local_ji = jhi @ raw_cross_ji[:, aux_idx, :].transpose(1, 0, 2).reshape(
                len(aux_idx), npno * n_ki)
            cross_fitted_ji = cross_local_ji.reshape(n_local, npno, n_ki)
            q_jk = jhi @ raw_oo[j, k, aux_idx]
            J_ji_ki[(key, k)] = np.einsum('Q,Qac->ac', q_jk, cross_fitted_ji)
            raw_kv_ki = np.tensordot(C_lmo[:, k], tmp_ki, axes=([0], [0]))
            q_kv_ki = jhi @ raw_kv_ki[aux_idx]
            K_ji_ki_dict[(key, k)] = q_jv.T @ q_kv_ki  # (npno, n_ki)
            del tmp_ki, raw_cross_ji

        cc_ints[key] = {
            'K_iajb': K_iajb,
            'K_mnij': K_mnij,
            'K_bar_ij': K_bar_ij,
            'K_bar_ji': K_bar_ji,
            'K_bar_chem': K_bar_chem,
            'J_ijab': J_ijab,
            'J_ij_kj': J_ij_kj,
            'K_ij_kj': K_ij_kj_dict,
            'J_ji_ki': J_ji_ki,       # j-side cross-pair: (j*a_ij | k*c_ki)
            'K_ji_ki': K_ji_ki_dict,
            # 3-index (stored for T1 dressing and Term A)
            'i_Qa': q_iv.copy(),     # (n_local, npno) — Psi4 convention
            'j_Qa': q_jv.copy(),     # (n_local, npno)
            'i_Qk': q_io.copy(),     # (n_local, nocc)
            'j_Qk': q_jo.copy(),
            'Qma': Qma,              # (n_local, nocc, npno)
            'Qab': Qab,              # (n_local, npno, npno)
            'n_local': n_local,
            'aux_idx': aux_idx,
        }

    del raw_3c, raw_oo
    return cc_ints


def t1_ints(cc_ints, t1_pno, pno_spaces, S_pno_cache, keys, nocc):
    """Build T1-dressed DF intermediates, matching Psi4 t1_ints().

    For each pair (ij), builds:
        i_Qa_t1[Q, a] = i_Qa[Q,a] - i_Qk[Q,:] @ T1_all[:,a]
                       + Σ_b (Qab[Q,a,b] - T1_all^T @ Qma[Q,:,b]) * t1_i[b]

    Returns:
        dressed: dict pair_key -> {'i_Qa_t1': (n_local, npno), 'j_Qa_t1': ...}
    """
    from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair

    dressed = {}
    for key in keys:
        ci = cc_ints.get(key)
        if ci is None:
            continue
        i, j = key
        npno = pno_spaces[key]['C_pno'].shape[1]
        n_local = ci['n_local']

        # Project T1 to pair's PNO basis
        T1_all = np.zeros((nocc, npno))
        for k in range(nocc):
            T1_all[k] = _project_t1_to_pair(
                t1_pno, k, key, S_pno_cache, pno_spaces)

        Qma = ci['Qma']  # (n_local, nocc, npno)
        Qab = ci['Qab']  # (n_local, npno, npno)

        def _dress_one(lmo_idx, Qa_key):
            """Build i_Qa_t1 for one LMO matching Psi4 t1_ints exactly. Vectorized."""
            i_Qa = ci[Qa_key]  # (n_local, npno)
            i_Qk = ci[Qa_key.replace('Qa', 'Qk')]  # (n_local, nocc)

            t1_lmo = T1_all[lmo_idx]
            # Term 1: bare
            # Term 2: -i_Qk @ T1_all
            # Terms 3+4 vectorized:
            #   Σ_b Qab[Q,a,b] * t1_lmo[b] = Qab @ t1_lmo  → (n_local, npno)
            #   Σ_b (T1_all^T @ Qma[Q,:,b]) * t1_lmo[b]
            #     = Σ_{b,m} T1_all[m,a] * Qma[Q,m,b] * t1_lmo[b]
            #     = Σ_m T1_all[m,a] * (Qma @ t1_lmo)[Q,m]
            #     = (Qma @ t1_lmo) @ T1_all = (n_local, npno)
            result = i_Qa - i_Qk @ T1_all
            # Term 3: Σ_b Qab[Q,a,b] * t1_lmo[b]
            result += np.einsum('Qab,b->Qa', Qab, t1_lmo)
            # Term 4: -Σ_{b,m} T1_all[m,a] * Qma[Q,m,b] * t1_lmo[b]
            #       = -(Qma @ t1_lmo) @ T1_all
            qma_t1 = np.einsum('Qmb,b->Qm', Qma, t1_lmo)  # (n_local, nocc)
            result -= qma_t1 @ T1_all  # (n_local, npno)
            return result

        dressed[key] = {
            'i_Qa_t1': _dress_one(i, 'i_Qa'),
            'j_Qa_t1': _dress_one(j, 'j_Qa'),
        }

    return dressed


def t1_fock(cc_ints, dressed_ints, t1_pno, fov_pno, pno_spaces,
            S_pno_cache, F_lmo, eps_lmo, foo_t2, keys, nocc):
    """Build dressed Fock matrices matching Psi4 t1_fock().

    Returns:
        Fkj: (nocc, nocc) dressed occupied Fock
        Fab_all: dict pair_key -> (npno, npno) dressed virtual Fock
        foo_t1: (nocc, nocc) T1 part of foo
    """
    from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair

    # Step 1: Fkj dressing from per-pair K_bar_chem and K_bar
    Fkj = F_lmo.copy()
    for key in keys:
        ci = cc_ints.get(key)
        if ci is None:
            continue
        i, j = key
        npno = pno_spaces[key]['C_pno'].shape[1]

        T1_all = np.zeros((nocc, npno))
        for k in range(nocc):
            T1_all[k] = _project_t1_to_pair(
                t1_pno, k, key, S_pno_cache, pno_spaces)

        # Psi4 line 1546-1547:
        # Fkj(i,j) += 2 * T_n · K_bar_chem - T_n · K_bar_ji
        Fkj[i, j] += 2.0 * np.sum(T1_all * ci['K_bar_chem']) \
                    - np.sum(T1_all * ci['K_bar_ji'])
        if i != j:
            Fkj[j, i] += 2.0 * np.sum(T1_all * ci['K_bar_chem']) \
                        - np.sum(T1_all * ci['K_bar_ij'])

    # Step 2: Fab per pair from Qma, Qab
    Fab_all = {}
    for key in keys:
        ci = cc_ints.get(key)
        if ci is None:
            continue
        i, j = key
        npno = pno_spaces[key]['C_pno'].shape[1]
        n_local = ci['n_local']
        Qma = ci['Qma']
        Qab = ci['Qab']

        T1_all = np.zeros((nocc, npno))
        for k in range(nocc):
            T1_all[k] = _project_t1_to_pair(
                t1_pno, k, key, S_pno_cache, pno_spaces)

        # Fab = diag(e_pno) + J/K dressing from Qma/Qab
        e_pno = pno_spaces[key]['e_pno']
        Fab = np.diag(e_pno)

        # gamma[Q] = Σ_{m,a} T1_all[m,a] * Qma[Q,m,a]
        gamma = np.einsum('ma,Qma->Q', T1_all, Qma)  # (n_local,)

        # J contribution: 2 * Σ_Q gamma[Q] * Qab[Q,a,b]
        Fab += 2.0 * np.einsum('Q,Qab->ab', gamma, Qab)

        # K contribution: -Σ_Q (Qab[Q] @ T1_all^T) @ Qma[Q]
        # = -einsum('Qab,nb,Qnc->ac', Qab, T1_all, Qma)
        Y = np.einsum('Qab,nb->Qan', Qab, T1_all)  # (n_local, npno, nocc)
        Fab -= np.einsum('Qan,Qnc->ac', Y, Qma)

        # Psi4 ccsd.cc line 1639-1640:
        #   Fab_[ij] = Fab_bar[ij] - T_n.T @ Fia_bar[ij]
        # Fia_bar[k,a] = 2*gamma*Qma[k,a] - Qma@T_n.T@Qma  (lines 1602-1628)
        Fia_bar = 2.0 * np.einsum('Qka,Q->ka', Qma, gamma)
        Z = np.einsum('nb,Qkb->Qnk', T1_all, Qma)
        Fia_bar -= np.einsum('Qna,Qnk->ka', Qma, Z)

        Fab -= T1_all.T @ Fia_bar

        Fab_all[key] = Fab

    # Eq 94: Fkj += Σ_a Fia_bar_jj · t1_j
    for j_idx in range(nocc):
        key_jj = (j_idx, j_idx)
        ci = cc_ints.get(key_jj)
        if ci is None:
            continue
        t1_j = t1_pno.get(j_idx)
        if t1_j is None or t1_j.size == 0:
            continue
        npno = pno_spaces[key_jj]['C_pno'].shape[1]
        T1_all = np.zeros((nocc, npno))
        for k in range(nocc):
            T1_all[k] = _project_t1_to_pair(
                t1_pno, k, key_jj, S_pno_cache, pno_spaces)
        gamma = np.einsum('ma,Qma->Q', T1_all, ci['Qma'])
        Qma_jj = ci['Qma']
        Fia_bar_jj = 2.0 * np.einsum('Qka,Q->ka', Qma_jj, gamma)
        Z_jj = np.einsum('nb,Qkb->Qnk', T1_all, Qma_jj)
        Fia_bar_jj -= np.einsum('Qna,Qnk->ka', Qma_jj, Z_jj)
        # Psi4 ccsd.cc line 1621: Fkj_(i,j) += Fia_bar[jj](i,:) . T_ia[j]
        for i_idx in range(nocc):
            Fkj[i_idx, j_idx] += np.dot(Fia_bar_jj[i_idx], t1_j)

    if getattr(t1_fock, '_dump_fkj', False):
        Fkj_step1 = Fkj - F_lmo  # Step 1 only (before Step 2 was added above)
        # Wait - Step 2 already added. Need to separate.
        # Recompute Step 2 contribution
        Fkj_step2 = np.zeros_like(Fkj)
        for j_idx2 in range(nocc):
            key_jj2 = (j_idx2, j_idx2)
            ci2 = cc_ints.get(key_jj2)
            if ci2 is None: continue
            t1_j2 = t1_pno.get(j_idx2)
            if t1_j2 is None or t1_j2.size == 0: continue
            npno2 = pno_spaces[key_jj2]['C_pno'].shape[1]
            T1_all2 = np.zeros((nocc, npno2))
            for k2 in range(nocc):
                T1_all2[k2] = _project_t1_to_pair(t1_pno, k2, key_jj2, S_pno_cache, pno_spaces)
            gamma2 = np.einsum('ma,Qma->Q', T1_all2, ci2['Qma'])
            Fia_bar2 = 2.0 * np.einsum('Qka,Q->ka', ci2['Qma'], gamma2)
            Z2 = np.einsum('nb,Qkb->Qnk', T1_all2, ci2['Qma'])
            Fia_bar2 -= np.einsum('Qna,Qnk->ka', ci2['Qma'], Z2)
            for i_idx2 in range(nocc):
                Fkj_step2[i_idx2, j_idx2] = np.dot(Fia_bar2[i_idx2], t1_j2)
        step1_only = Fkj - F_lmo - Fkj_step2
        print(f"  FKJ_DBG: Step1 [0,1]={step1_only[0,1]:.12e} [1,0]={step1_only[1,0]:.12e}", flush=True)
        print(f"  FKJ_DBG: Step2 [0,1]={Fkj_step2[0,1]:.12e} [1,0]={Fkj_step2[1,0]:.12e}", flush=True)
        print(f"  FKJ_DBG: Total [0,1]={(Fkj-F_lmo)[0,1]:.12e} [1,0]={(Fkj-F_lmo)[1,0]:.12e}", flush=True)

    foo_t1 = Fkj - F_lmo
    # NOTE: Psi4's Fkj_ does NOT include foo_t2. The T2 contribution to G_tilde
    # is added inside compute_G_tilde. Previously we added foo_t2 here as a
    # workaround for a bug in build_G_tilde that skipped the i==j diagonal
    # contribution; that bug is now fixed.

    return Fkj, Fab_all, foo_t1


def compute_B_tilde(cc_ints, dressed_ints, t2_pno_all, t1_pno,
                    pno_spaces, S_pno_cache, key, nocc):
    """Build B_tilde for pair (ij) matching Psi4's precomputed B_tilde.

    B_tilde[k,l] = (ki|lj)_dressed + Σ_{a,b} tau[a,b] * (ka|lb)

    where (ki|lj)_dressed uses dressed ooL and
    (ka|lb) uses bare ovL from cc_ints.
    """
    from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair

    ci = cc_ints.get(key)
    if ci is None:
        return np.zeros((nocc, nocc))

    i, j = key
    npno = pno_spaces[key]['C_pno'].shape[1]
    n_local = ci['n_local']

    T1_all = np.zeros((nocc, npno))
    if t1_pno is not None:
        for k in range(nocc):
            T1_all[k] = _project_t1_to_pair(
                t1_pno, k, key, S_pno_cache, pno_spaces)
    t1_i = T1_all[i]
    t1_j = T1_all[j]

    # i_Qk_t1[Q, k] = i_Qk[Q,k] + Σ_a Qma[Q,k,a] * t1_i[a]
    i_Qk_t1 = ci['i_Qk'].copy()  # (n_local, nocc)
    i_Qk_t1 += np.einsum('Qka,a->Qk', ci['Qma'], t1_i)

    j_Qk_t1 = ci['j_Qk'].copy()
    j_Qk_t1 += np.einsum('Qka,a->Qk', ci['Qma'], t1_j)

    # J_oo_dressed[k,l] = i_Qk_t1[Q,k] * j_Qk_t1[Q,l]
    B_tilde = i_Qk_t1.T @ j_Qk_t1  # (nocc, nocc)

    # voov: B[k,l] += Σ_{Q} Qma[Q,k,:] @ T2 @ Qma[Q,l,:].T
    # Psi4 ccsd.cc line 1685: uses bare T_iajb, NOT tau
    T2_ij = t2_pno_all[key]
    Qma_arr = ci['Qma']  # (n_local, nocc, npno)
    P = np.einsum('ab,Qka->kbQ', T2_ij, Qma_arr)  # (nocc, npno, n_local)
    B_tilde += np.einsum('kbQ,Qlb->kl', P, Qma_arr)

    return B_tilde


def compute_ladder(cc_ints, t2_pno_all, t1_pno, pno_spaces,
                   S_pno_cache, key, nocc):
    """Compute ladder term A for pair (ij) matching Psi4 Term A.

    A[a,b] = Σ_Q Qab_t1[Q,a,c] * T2[c,d] * Qab_t1[Q,b,d]
    """
    from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair

    ci = cc_ints.get(key)
    if ci is None:
        return np.zeros((0, 0))

    i, j = key
    npno = pno_spaces[key]['C_pno'].shape[1]
    n_local = ci['n_local']
    Qab = ci['Qab']
    Qma = ci['Qma']

    T1_all = np.zeros((nocc, npno))
    if t1_pno is not None:
        for k in range(nocc):
            T1_all[k] = _project_t1_to_pair(
                t1_pno, k, key, S_pno_cache, pno_spaces)

    T2_ij = t2_pno_all[key]
    # Vectorized: build Qab_t1[Q,a,b] = Qab[Q,a,b] - (T1_all^T @ Qma)[Q,a,b]
    # = Qab - einsum('mb,Qma->Qab', T1, Qma)... wait that's wrong shape
    # Original: Qab_t1[a,b] = Qab[Q,a,b] - (T1_all.T @ Qma[Q])[a,b]
    # T1_all.T @ Qma[Q] = (npno, nocc) @ (nocc, npno) = (npno, npno)
    # So Qab_t1[Q] = Qab[Q] - T1_all.T @ Qma[Q]
    # Vectorized: Qab_t1 = Qab - einsum('na,Qnb->Qab', T1_all, Qma)
    Qab_t1 = Qab - np.einsum('na,Qnb->Qab', T1_all, Qma)
    if getattr(compute_ladder, '_dump_debug', False) and (i,j) in [(0,1),(2,3)]:
        t1_norm = np.linalg.norm(T1_all)
        t2_norm = np.linalg.norm(T2_ij)
        ladder_norm = np.linalg.norm(Qab_t1)
        print(f"  LADDER_DBG pair({i},{j}): |T1|={t1_norm:.10f} "
              f"|T2|={t2_norm:.10f} |Qab_t1|={ladder_norm:.10f} "
              f"T2[0,0]={T2_ij[0,0]:.12e} T2[0,1]={T2_ij[0,1]:.12e} T2[1,1]={T2_ij[1,1]:.12e}",
              flush=True)
    # ladder = Σ_Q Qab_t1[Q] @ T2 @ Qab_t1[Q].T
    X = np.einsum('Qac,cd->Qad', Qab_t1, T2_ij)
    ladder = np.einsum('Qad,Qbd->ab', X, Qab_t1)
    return ladder
