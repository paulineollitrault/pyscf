"""DLPNO-CCSD residual and T1-dressed intermediates.

Implements the DLPNO-CCSD T2 residual (Jiang et al. JCP 2024, Eqs 75-86)
as a Psi4-compatible two-buffer formulation (R + P̂(Rn)) with all T1
dressing via explicit intermediates (B_tilde, C_tilde, D_tilde, G_tilde,
Fab, Fkj). All integrals are BARE (computed from undressed C_lmo); T1
enters exclusively through the dressed intermediates built here.
"""
import numpy as np
from pyscf.ao2mo import _ao2mo
from pyscf.cc.dlpno_tccsd.lccsd import (
    _project_t1_to_pair, _compute_ladder, _project_t2_full,
)
from pyscf.cc.dlpno_tccsd.local_df import (
    get_local_K, get_local_ovL, get_local_ooL_vec,
)


def _chunked_map(pool, fn, items, chunks_per_worker=4):
    """Dispatch many small tasks to a ThreadPoolExecutor in chunks.

    ThreadPoolExecutor.map has ~5-50 us of dispatch overhead per task,
    which dominates when tasks are <1 ms. Chunking into N_workers*4 groups
    collapses overhead into ~N_workers dispatches while keeping load
    balancing healthy (4 chunks per worker).

    Yields results in submission order, matching pool.map semantics.
    """
    if pool is None:
        for it in items:
            yield fn(it)
        return
    items = list(items)
    n = len(items)
    if n == 0:
        return
    workers = getattr(pool, '_max_workers', 1)
    chunk_size = max(1, (n + workers * chunks_per_worker - 1)
                        // (workers * chunks_per_worker))

    def _process_chunk(chunk):
        return [fn(x) for x in chunk]

    chunks = [items[i:i + chunk_size] for i in range(0, n, chunk_size)]
    for chunk_result in pool.map(_process_chunk, chunks):
        yield from chunk_result


# =========================================================================
# T1-dressed intermediates: G_tilde, D_tilde, Fab, Fkj, Fia_bar (Eqs 82-86)
# =========================================================================



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
                K_il = get_local_K(cc_ints, key_il, i_idx, l_idx)
                if K_il is None:
                    continue

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






def build_D_tilde(t1_pno, t2_pno_all, pno_spaces, nocc,
                  ovL_bare, ooL_bare, S_pno_cache, with_df,
                  _term2_precomputed=None, cc_ints=None,
                  pair_lmo_idx=None, _pool=None):
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
                    td['result'] -= np.tensordot(y, B_L, axes=(0, 0))
                aux_off += nL
            _term2_precomputed = {ik: td['result'] for ik, td in term2_data.items()}

    # --- Build D_tilde for each (i,k) pair ---
    def _process_ik(ik_tuple):
        i_idx, k_idx = ik_tuple
        key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
        n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
        if n_ik == 0:
            return ik_tuple, None

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
        if key_ki in cc_ints and cc_ints[key_ki] is not None:
            ci_ki = cc_ints[key_ki]
            if key_ki[0] == k_idx:
                k_Qa = ci_ki['i_Qa']
            else:
                k_Qa = ci_ki['j_Qa']
            Qab_ki = ci_ki['Qab']
            # Part A: D[a,b] += 2 * Σ_{Q,c} k_Qa[Q,b]*Qab[Q,a,c]*t1_i[c]
            # z[Q,a] = Σ_c Qab[Q,a,c]*t1_i[c]
            z_Qa = np.tensordot(Qab_ki, t1_i_ik, axes=(2, 0))
            # D[a,b] += 2 * Σ_Q z_Qa[Q,a] * k_Qa[Q,b] = 2 * z_Qa.T @ k_Qa
            D_tilde_ik += 2.0 * z_Qa.T @ k_Qa
            # Part B: D[r,s] -= Σ_{Q,b} t1_i[b]*k_Qa[Q,b]*Qab[Q,s,r]
            # w[Q] = Σ_b t1_i[b]*k_Qa[Q,b]
            w = k_Qa @ t1_i_ik  # (n_local,)
            # D[r,s] -= Σ_Q w[Q]*Qab[Q,s,r]
            D_tilde_ik -= np.tensordot(w, Qab_ki, axes=(0, 0)).T
        elif _term2_precomputed and (i_idx, k_idx) in _term2_precomputed:
            D_tilde_ik += _term2_precomputed[(i_idx, k_idx)]

        # --- Term 1: -Σ_l T1_all[l,a] · M_{ik}^{lc} ---
        # M_{ik}^{lc} = 2*(il|kc) - (ik|lc)
        # (il|kc) = Σ_Q ooL[i,l,Q]*ovL_k_ik[c,Q] → K_bar-like
        # (ik|lc) = Σ_Q ooL[i,k,Q]*ovL_l_ik[c,Q] → K_bar_chem-like
        # Psi4 (ccsd.cc:1790) restricts l to lmopair_to_lmos_[ik]
        _domain_ik_t1 = (set(pair_lmo_idx[key_ik].tolist())
                         if pair_lmo_idx is not None and key_ik in pair_lmo_idx
                         else set(range(nocc)))
        _use_local_d1 = (key_ik in cc_ints and cc_ints[key_ik] is not None)
        for ll in range(nocc):
            if ll not in _domain_ik_t1:
                continue
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
        # Psi4 restricts l to lmopair_to_lmos_[ik] (ccsd.cc:1792)
        _domain_ik = (set(pair_lmo_idx[key_ik].tolist())
                      if pair_lmo_idx is not None and key_ik in pair_lmo_idx
                      else set(range(nocc)))
        for ll in range(nocc):
            if ll not in _domain_ik:
                continue
            key_lk = (min(ll, k_idx), max(ll, k_idx))
            if key_lk not in pno_spaces:
                continue
            n_lk = pno_spaces[key_lk]['C_pno'].shape[1]
            if n_lk == 0:
                continue
            K_lk = get_local_K(cc_ints, key_lk, ll, k_idx)
            if K_lk is None:
                continue
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
        # Psi4 restricts l to lmopair_to_lmos_[ik] (ccsd.cc:1803)
        for ll in range(nocc):
            if ll not in _domain_ik:
                continue
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

            K_lk = get_local_K(cc_ints, key_lk, ll, k_idx)
            if K_lk is None:
                continue
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

        return ik_tuple, D_tilde_ik

    if _pool is not None:
        for ik, val in _pool.map(_process_ik, list(all_pairs)):
            if val is not None:
                D_tilde_all[ik] = val
    else:
        for ik in all_pairs:
            _, val = _process_ik(ik)
            if val is not None:
                D_tilde_all[ik] = val
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
            if key_ij in cc_ints and cc_ints[key_ij] is not None:
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

# =========================================================================
# Dressed DF integrals (ooL/ovL) and C_tilde (Eqs 83, 91-93)
# =========================================================================
# ---------------------------------------------------------------------------
# Dressed DF integral builders (Eqs 91-93)
# ---------------------------------------------------------------------------



def compute_C_tilde(t1_pno, t2_pno_all, pno_spaces, nocc,
                    ovL_pno_bare, ooL_bare, S_pno_cache, with_df,
                    _term2_precomputed=None, cc_ints=None,
                    pair_lmo_idx=None, _pool=None):
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
                    td['result'] += np.tensordot(z_i, B_L, axes=(0, 0))
                aux_off += nL
        _term2_precomputed = {ki: td['result'] for ki, td in term2_data.items()}

    def _process_ki(ki_tuple):
        k, i = ki_tuple
        key_ki = (min(k, i), max(k, i))  # storage key
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        if n_ki == 0:
            return ki_tuple, None

        C_tilde_ki = np.zeros((n_ki, n_ki))
        _dump_terms = ((k, i) == (10, 10)
                       and getattr(compute_C_tilde, '_dump_iter', -1) == 1)
        _term_log = {} if _dump_terms else None

        # Term 2: Σ_b t1_i^b · (kb|ac) = Σ_{Q,b} t1_i[b]*k_Qa[Q,b]*Qab[Q,a,c]
        # Use local DF when cc_ints available, otherwise precomputed (global DF)
        if key_ki in cc_ints and cc_ints[key_ki] is not None:
            ci_ki = cc_ints[key_ki]
            if key_ki[0] == k:
                k_Qa = ci_ki['i_Qa']
            else:
                k_Qa = ci_ki['j_Qa']
            Qab_ki = ci_ki['Qab']
            t1_i_ki = _project_t1_to_pair(t1_pno, i, key_ki, S_pno_cache, pno_spaces)
            z = k_Qa @ t1_i_ki
            T2_contrib = np.tensordot(z, Qab_ki, axes=(0, 0))
            C_tilde_ki += T2_contrib
            if _dump_terms:
                ev = np.sort(np.linalg.eigvalsh(0.5*(T2_contrib+T2_contrib.T)))[::-1]
                _term_log['T2'] = (float(np.sqrt(np.mean(T2_contrib**2))), ev[0], ev[-1])
        elif (k, i) in _term2_precomputed:
            C_tilde_ki += _term2_precomputed[(k, i)]

        # --- Term 1: -Σ_l T1_all[l,a] · (ki|lc) ---
        # K_bar_chem[ki][l,c] = Σ_Q ooL[k,i,Q]*ovL_l_ki[c,Q] = (ki|lc)
        T1_all_ki = np.zeros((nocc, n_ki))
        for ll in range(nocc):
            T1_all_ki[ll] = _project_t1_to_pair(
                t1_pno, ll, key_ki, S_pno_cache, pno_spaces)
        K_bar_chem = np.zeros((nocc, n_ki))
        if key_ki in cc_ints and cc_ints[key_ki] is not None:
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
        # Psi4 line 1722 restricts the contraction over l to lmopair_to_lmos_[ki]
        # (T_n_ij_[ki] has shape (nlmo_ki, npno_ki)).
        _domain_ki = (set(pair_lmo_idx[key_ki].tolist())
                      if pair_lmo_idx is not None and key_ki in pair_lmo_idx
                      else set(range(nocc)))
        if _domain_ki != set(range(nocc)):
            _mask = np.zeros(nocc, dtype=bool)
            for _l in _domain_ki:
                _mask[_l] = True
            T1_all_ki[~_mask] = 0.0
            K_bar_chem[~_mask] = 0.0
        T1_contrib = -T1_all_ki.T @ K_bar_chem
        C_tilde_ki += T1_contrib  # (n_ki, n_ki)
        if _dump_terms:
            ev = np.sort(np.linalg.eigvalsh(0.5*(T1_contrib+T1_contrib.T)))[::-1]
            _term_log['T1'] = (float(np.sqrt(np.mean(T1_contrib**2))), ev[0], ev[-1])

        # --- Term 3: -Σ_l T1_l^a · (S @ K_kl @ T1_i_kl) ---
        # Psi4 restricts l to lmopair_to_lmos_[ki] (ccsd.cc:1724)
        T3_contrib = np.zeros_like(C_tilde_ki) if _dump_terms else None
        for ll in range(nocc):
            if ll not in _domain_ki:
                continue
            key_kl = (min(k, ll), max(k, ll))
            if key_kl not in pno_spaces:
                continue
            n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
            if n_kl == 0:
                continue
            # K_kl[a_kl, b_kl] = (ka|lb) = exchange in PNO_kl
            K_kl = get_local_K(cc_ints, key_kl, k, ll)
            if K_kl is None:
                continue

            # T1_i projected to PNO_kl
            t1_i_kl = _project_t1_to_pair(t1_pno, i, key_kl, S_pno_cache, pno_spaces)
            if np.max(np.abs(t1_i_kl)) < 1e-15:
                continue

            # Per Eq 83: t1_i contracts with b (first index of K_kl^{bc} where
            # K_iajb[kl][a,b]=(ka|lb) has "a" at k-partner position).
            Kt1 = K_kl.T @ t1_i_kl  # (n_kl,)
            S_ki_kl = S_pno_cache.get((key_ki, key_kl))
            if S_ki_kl is None:
                continue
            Kt1_ki = S_ki_kl @ Kt1  # (n_ki,)
            t1_l_ki = T1_all_ki[ll]
            contrib3 = -np.outer(t1_l_ki, Kt1_ki)
            C_tilde_ki += contrib3
            if _dump_terms:
                T3_contrib += contrib3
        if _dump_terms:
            ev = np.sort(np.linalg.eigvalsh(0.5*(T3_contrib+T3_contrib.T)))[::-1]
            _term_log['T3'] = (float(np.sqrt(np.mean(T3_contrib**2))), ev[0], ev[-1])

        # --- Term 4: -(1/2) Σ_l S @ t2_li @ S @ K_kl @ S ---
        # Psi4 restricts l to lmopair_to_lmos_[ki] (ccsd.cc:1736)
        T4_contrib = np.zeros_like(C_tilde_ki)
        for ll in range(nocc):
            if ll not in _domain_ki:
                continue
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

            K_kl = get_local_K(cc_ints, key_kl, k, ll)
            if K_kl is None:
                continue

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
            contrib4 = -0.5 * C_temp
            C_tilde_ki += contrib4  # Term 4: Jiang Eq 83
            T4_contrib += contrib4
        if _dump_terms:
            ev = np.sort(np.linalg.eigvalsh(0.5*(T4_contrib+T4_contrib.T)))[::-1]
            _term_log['T4'] = (float(np.sqrt(np.mean(T4_contrib**2))), ev[0], ev[-1])
            for _tname, (_trms, _t0, _tl) in _term_log.items():
                print(f"  CTILDE_TERMS_US iter 1 ki=(10,10) {_tname}: "
                      f"rms={_trms:.6e} ev_top={_t0:.6e} ev_bot={_tl:.6e}",
                      flush=True)

        return ki_tuple, C_tilde_ki  # store with ORDERED (k,i) key

    if _pool is not None:
        for ki, val in _pool.map(_process_ki, list(all_pairs)):
            if val is not None:
                C_tilde_all[ki] = val
    else:
        for ki in all_pairs:
            _, val = _process_ki(ki)
            if val is not None:
                C_tilde_all[ki] = val
    return C_tilde_all



# ---------------------------------------------------------------------------
# T1-dressed Fock intermediates (Eqs 94-101, 85-86)
# ---------------------------------------------------------------------------


def compute_all_df_terms_local(t1_pno, fov_pno, t2_pno_all, pno_spaces,
                                nocc, cc_ints, S_pno_cache, pair_keys,
                                _pool=None):
    """Per-pair, local-aux-per-pair version of compute_all_df_terms.

    Drop-in replacement that reads from ``cc_ints[pk]['Qab' / 'Qma']``
    (already fitted, local-aux per pair) instead of iterating
    ``with_df.loop()`` over global aux blocks. Each pair's contractions
    sum over the pair's local aux only — Psi4-equivalent and avoids the
    per-iteration full-naux ovL rebuild (12s of `_ao2mo.nr_e2` on str 010).

    Same return signature as ``compute_all_df_terms``:
        fvv_t1_all, c_term2_all, d_term2_all, ladder_all

    Args:
        t1_pno: dict i -> (n_pno_ii,)
        fov_pno: unused (kept for signature compat)
        t2_pno_all: dict pk -> (n_pno, n_pno)
        pno_spaces: dict from make_pnos
        nocc: int
        cc_ints: dict pk -> {'Qab': (n_local, n, n), 'Qma': (n_local, nocc, n), ...}
        S_pno_cache: dict (key1, key2) -> (n1, n2) overlap
        pair_keys: list of pair keys for which fvv_t1 and ladder are returned.
    """
    # All ordered (k, i) pairs that appear in t2 (both directions).
    all_ordered = set()
    for key in t2_pno_all:
        a, b = key
        all_ordered.add((a, b))
        all_ordered.add((b, a))

    fvv_t1_all = {}
    c_term2_all = {}
    d_term2_all = {}
    ladder_all = {}

    # ============================================================
    # Per-pair: fvv_t1 + ladder. Pure function — safe to parallelize.
    # ============================================================
    def _fvv_ladder(pk):
        n = pno_spaces[pk]['C_pno'].shape[1]
        if n == 0:
            return pk, np.zeros((n, n)), np.zeros((n, n))
        ci = cc_ints.get(pk)
        if ci is None:
            return pk, np.zeros((n, n)), np.zeros((n, n))

        Qab = ci['Qab']         # (n_local, n, n)
        Qma = ci['Qma']         # (n_local, nocc, n)
        i, j = pk
        T1_all = np.zeros((nocc, n))
        for kk in range(nocc):
            T1_all[kk] = _project_t1_to_pair(
                t1_pno, kk, pk, S_pno_cache, pno_spaces)

        # fvv_t1
        z = np.einsum('Lka,ka->L', Qma, T1_all, optimize=True)
        fvv = 2.0 * np.einsum('L,Lab->ab', z, Qab, optimize=True)
        tmp = np.einsum('Lab,kb->Lka', Qab, T1_all, optimize=True)
        fvv -= np.einsum('Lka,Lkb->ab', tmp, Qma, optimize=True)

        # ladder
        tau = t2_pno_all[pk] + np.outer(T1_all[i], T1_all[j])
        corr = np.einsum('ka,Lkb->Lab', T1_all, Qma, optimize=True)
        B_tilde = Qab - corr
        X_L = np.matmul(B_tilde, tau)
        ladder = (X_L.transpose(1, 0, 2).reshape(n, -1) @
                  B_tilde.transpose(1, 0, 2).reshape(n, -1).T)
        # Debug: print for diagonal (0,0), off-diag (0,5), diagonal (5,5)
        _it = getattr(compute_all_df_terms_local, '_iter', 0)
        if _it <= 2 and pk in [(0, 0), (0, 5), (5, 5)]:
            print(f'  LADDER_DBG iter {_it} '
                  f'pair{pk}: |T1_all|={np.linalg.norm(T1_all):.3e} '
                  f'|T2|={np.linalg.norm(t2_pno_all[pk]):.3e} '
                  f'|tau|={np.linalg.norm(tau):.3e} '
                  f'|corr|={np.linalg.norm(corr):.3e} '
                  f'|B_tilde|={np.linalg.norm(B_tilde):.3e} '
                  f'|Qab|={np.linalg.norm(Qab):.3e} '
                  f'|Qma|={np.linalg.norm(Qma):.3e} '
                  f'|ladder|={np.linalg.norm(ladder):.3e}',
                  flush=True)
        return pk, fvv, ladder

    # ============================================================
    # C_tilde Term 2: per (k, i) ordered pair
    # ============================================================
    def _c_term2(ki_tuple):
        k, i = ki_tuple
        key_ki = (min(k, i), max(k, i))
        n = pno_spaces[key_ki]['C_pno'].shape[1]
        if n == 0:
            return ki_tuple, None
        ci = cc_ints.get(key_ki)
        if ci is None:
            return ki_tuple, None
        t1_i = _project_t1_to_pair(t1_pno, i, key_ki, S_pno_cache, pno_spaces)
        if np.max(np.abs(t1_i)) < 1e-15:
            return ki_tuple, None
        Qma = ci['Qma']
        Qab = ci['Qab']
        z_i = Qma[:, i, :] @ t1_i
        return ki_tuple, np.einsum('L,Lab->ab', z_i, Qab, optimize=True)

    # ============================================================
    # D_tilde Term 2: per (i, k) ordered pair
    # ============================================================
    def _d_term2(ik_tuple):
        i_idx, k_idx = ik_tuple
        key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
        n = pno_spaces[key_ik]['C_pno'].shape[1]
        if n == 0:
            return ik_tuple, None
        ci = cc_ints.get(key_ik)
        if ci is None:
            return ik_tuple, None
        t1_i = _project_t1_to_pair(
            t1_pno, i_idx, key_ik, S_pno_cache, pno_spaces)
        if np.max(np.abs(t1_i)) < 1e-15:
            return ik_tuple, None
        Qma = ci['Qma']
        Qab = ci['Qab']
        ovL_k = Qma[:, k_idx, :].T
        z_c = Qab @ t1_i
        y = Qma[:, k_idx, :] @ t1_i
        result = 2.0 * (ovL_k @ z_c)
        result -= np.einsum('L,Lab->ab', y, Qab, optimize=True)
        return ik_tuple, result

    # Unified dispatch: one _pool.map (chunked) covering all three
    # task kinds. Removes 2 synchronization barriers per CCSD iteration
    # and amortizes pool dispatch overhead across chunks.
    ordered_list = list(all_ordered)
    work = ([('fvv_ladder', pk) for pk in pair_keys]
            + [('c_term2', ki) for ki in ordered_list]
            + [('d_term2', ik) for ik in ordered_list])

    def _dispatch(item):
        kind, key = item
        if kind == 'fvv_ladder':
            return kind, _fvv_ladder(key)
        elif kind == 'c_term2':
            return kind, _c_term2(key)
        else:  # d_term2
            return kind, _d_term2(key)

    for kind, payload in _chunked_map(_pool, _dispatch, work):
        if kind == 'fvv_ladder':
            pk, fvv, ladder = payload
            fvv_t1_all[pk] = fvv
            ladder_all[pk] = ladder
        elif kind == 'c_term2':
            ki, val = payload
            if val is not None:
                c_term2_all[ki] = val
        else:  # d_term2
            ik, val = payload
            if val is not None:
                d_term2_all[ik] = val

    # Pad missing pair_keys with zeros (matching old API)
    for pk in pair_keys:
        n = pno_spaces[pk]['C_pno'].shape[1]
        if pk not in fvv_t1_all:
            fvv_t1_all[pk] = np.zeros((n, n))
        if pk not in ladder_all:
            ladder_all[pk] = np.zeros((n, n))

    return fvv_t1_all, c_term2_all, d_term2_all, ladder_all


# =========================================================================
# T2 residual: Psi4-compatible two-buffer formulation
# =========================================================================


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
        pair_domain=None,
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

    # Pair LMO domain: Psi4 restricts k/l loops to lmopair_to_lmos_[ij]
    if pair_domain is not None:
        _domain_set = set(pair_domain)
    else:
        _domain_set = set(range(nocc))

    # === Symmetric buffer (K̃, A, B, E) ===
    R_sym = np.zeros((n_pno, n_pno))
    _DEBUG_PTERM = getattr(compute_residual_v2, '_debug_pterm', False) or \
                   getattr(compute_residual_v2, '_debug_pterm_all', False)
    _ITER = getattr(compute_residual_v2, '_iter', 0)

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
        print(f"  PTERM iter {_ITER} pair({i},{j}): K_rms={_rms(K_term):.12f} R_K={_rms(R_sym):.12f}")

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

    A_term = ladder_precomputed[key] if (
        ladder_precomputed is not None and key in ladder_precomputed
    ) else np.zeros((n_pno, n_pno))
    R_sym += A_term
    if _DEBUG_PTERM:
        print(f"  PTERM iter {_ITER} pair({i},{j}): A_rms={_rms(A_term):.12f} R_KA={_rms(R_sym):.12f}")

    # --- B (Eq 77/82): Woooo with dressed β ---
    # Psi4 uses BARE T2 (T_iajb_), NOT tau. See ccsd.cc line 2137.
    B_term = np.zeros((n_pno, n_pno))
    for key_kl, t2_kl in t2_pno_all.items():
        if t2_kl is None or t2_kl.shape[0] == 0:
            continue
        k, l = key_kl
        if k not in _domain_set or l not in _domain_set:
            continue

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
        print(f"  PTERM iter {_ITER} pair({i},{j}): B_rms={_rms(B_term):.12f} R_KAB={_rms(R_sym):.12f}")

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
        if k not in _domain_set or l not in _domain_set:
            continue
        u_kl = 2.0 * t2_kl - t2_kl.T
        # Use ONLY local DF — Psi4's K_iajb is always local DF
        _K = get_local_K(cc_ints, key_kl, k, l)
        if _K is None:
            print(f"WARN: No local DF K for pair {key_kl}, skipping E contribution")
            continue
        K_kl = _K
        S_kl_ij = _get_S(key_kl)
        # (k,l) contribution
        E_tilde -= S_kl_ij @ (u_kl @ K_kl.T) @ S_kl_ij.T
        # (l,k) contribution for off-diagonal
        if k != l:
            E_tilde -= S_kl_ij @ ((2.0*t2_kl.T - t2_kl) @ K_kl) @ S_kl_ij.T

    if _DEBUG_PTERM:
        print(f"  PTERM iter {_ITER} pair({i},{j}): Etilde_rms={_rms(E_tilde):.12f}")
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
        print(f"  PTERM iter {_ITER} pair({i},{j}): E_rms={_rms(E_term):.12f} R_KABE={_rms(R_sym):.12f}")

    # === Non-symmetric terms (C, D, G) ===
    # Compute C_ij and C_ji in one pass, then form the full P̂ result.
    # P̂(0.5*C + C_ji) = 0.5*(C_ij + C_ij.T) + (C_ji + C_ji.T) for i≠j
    #                  = 1.5*(C + C.T) for i=j (since C_ji = C_ij)
    Rn_ij = np.zeros((n_pno, n_pno))

    # --- C (Eq 78) ---
    if C_tilde_cache is not None:
        C_ij = np.zeros((n_pno, n_pno))
        C_ji = np.zeros((n_pno, n_pno))
        for k in sorted(_domain_set):
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
                    # Bold: J(jk|a_ij c_ki) — use LOCAL DF (J_ji_ki) when
                    # available so it matches Psi4's J_ij_kj_[(j,i)][k_ij]
                    # (fitted with pair ij's local aux metric).  Falling back
                    # to global DF (K_coul_cache) introduces small systematic
                    # differences vs the other bold terms which use local DF.
                    _ci_ij_c = cc_ints.get(key)
                    J_ji_local_c = (_ci_ij_c.get('J_ji_ki', {}).get((key, k))
                                    if _ci_ij_c else None)
                    if J_ji_local_c is not None:
                        gamma_ji += J_ji_local_c
                    elif K_coul_cache:
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
            print(f"  PTERM iter {_ITER} pair({i},{j}): C_ij_unsym_rms={_rms(C_ij):.12f} C_ji_unsym_rms={_rms(C_ji):.12f}")
    else:
        C_term = np.zeros((n_pno, n_pno))

    # --- D (Eq 79): antisymmetric ring with delta ---
    if D_tilde_cache is not None:
        D_ij = np.zeros((n_pno, n_pno))
        D_ji = np.zeros((n_pno, n_pno))
        for k in sorted(_domain_set):
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
                        # Use ONLY local DF cross-integrals (J_ji_ki, K_ji_ki)
                        _ci_ij = cc_ints.get(key)
                        K_ji_local = _ci_ij.get('K_ji_ki', {}).get((key, k)) if _ci_ij else None
                        J_ji_local = _ci_ij.get('J_ji_ki', {}).get((key, k)) if _ci_ij else None
                        if K_ji_local is not None and J_ji_local is not None:
                            D_temp_j += (2.0*K_ji_local - J_ji_local) @ u_ik.T @ S_ij_ik2.T
                        D_ji += 0.5 * D_temp_j

        D_term = D_ij + D_ji.T
        Rn_ij += D_term
        if _DEBUG_PTERM:
            print(f"  PTERM iter {_ITER} pair({i},{j}): D_ij_unsym_rms={_rms(D_ij):.12f} D_ji_unsym_rms={_rms(D_ji):.12f}")
    else:
        D_term = np.zeros((n_pno, n_pno))

    # --- G (Eq 81): Fock oo coupling ---
    G_ij = np.zeros((n_pno, n_pno))
    G_ji = np.zeros((n_pno, n_pno))
    for k in range(nocc):
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
        print(f"  PTERM iter {_ITER} pair({i},{j}): G_ij_unsym_rms={_rms(G_ij):.12f} G_ji_unsym_rms={_rms(G_ji):.12f}")
        R_total = R_sym + Rn_ij
        print(f"  PTERM iter {_ITER} pair({i},{j}): Rn={_rms(Rn_ij):.12f} R_total={_rms(R_total):.12f}")
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

    # === R_final = R_sym + Rn (already fully P̂-symmetrized) ===
    return R_sym + Rn_ij
