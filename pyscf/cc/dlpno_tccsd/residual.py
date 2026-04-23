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
                  Fkj, foo_t1, cc_ints=None,
                  S_pao_full=None, s1e=None):
    """Build G_tilde (Eq 86): double-dressed Fock oo.

    G_tilde[k,j] = F̃_{kj} + Σ_l u_lj × K_il (bare exchange)

    Psi4 ccsd.cc lines 1820-1844.

    Args:
        Fkj: (nocc, nocc) = F̃_{kj} = dressed Fock oo (from build_Fkj)
        foo_t1: (nocc, nocc) = T1 correction to Fock (from _compute_foo_t1)
    """
    G = Fkj.copy()
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    # Pre-compute u_lj = 2*t2_lj - t2_lj.T for every (l, j) with non-empty t2.
    # Only depends on (l, j); the original triple loop recomputed it across
    # the i axis (nocc× redundant).
    u_lj_cache = {}
    for l_idx in range(nocc):
        for j_idx in range(nocc):
            key_lj = (min(l_idx, j_idx), max(l_idx, j_idx))
            if key_lj not in t2_pno_all:
                continue
            t2_lj = t2_pno_all.get(key_lj)
            if t2_lj is None or t2_lj.shape[0] == 0:
                continue
            t2_lj_d = t2_lj.T if l_idx > j_idx else t2_lj
            u_lj_cache[(l_idx, j_idx)] = (key_lj, 2.0 * t2_lj_d - t2_lj_d.T)

    # Swap loop order to (i, l, j) so K_il / t2_il setup happens once per
    # (i, l) instead of once per (i, l, j).
    for i_idx in range(nocc):
        for l_idx in range(nocc):
            key_il = (min(i_idx, l_idx), max(i_idx, l_idx))
            if key_il not in t2_pno_all:
                continue
            t2_il = t2_pno_all.get(key_il)
            if t2_il is None or t2_il.shape[0] == 0:
                continue
            K_il = get_local_K(cc_ints, key_il, i_idx, l_idx)
            if K_il is None:
                continue

            for j_idx in range(nocc):
                u_entry = u_lj_cache.get((l_idx, j_idx))
                if u_entry is None:
                    continue
                key_lj, u_lj = u_entry

                if key_il == key_lj:
                    U_lj_proj = u_lj
                else:
                    S_il_lj = _s_pno_get(key_il, key_lj)
                    if S_il_lj is None:
                        continue
                    U_lj_proj = S_il_lj @ u_lj @ S_il_lj.T

                # G[i,j] += K_il · U_lj.T = trace(K @ U.T)
                G[i_idx, j_idx] += np.sum(K_il * U_lj_proj.T)

    return G






def _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e):
    """Return a function that retrieves S_pno[a,b] with uniform numerics.

    On cache hit: returns the cached matrix.
    On cache miss: computes via the same X_pno + S_pao_full path as the
    build (falls back to C_pno+s1e only when X_pno unavailable, e.g.
    CAS pairs) and POPULATES the cache so subsequent lookups hit.

    This makes the upfront O(N^2) restricted build work correctly: any
    key not pre-built is computed lazily once and cached.  After the
    first CCSD iteration the cache contains every actually-accessed
    entry; later iterations are pure hits.
    """
    from pyscf.cc.dlpno_tccsd.local_df import compute_S_pno

    def _get(ka, kb, default=None):
        if S_pno_cache is not None:
            S = S_pno_cache.get((ka, kb))
            if S is not None:
                return S
        if S_pao_full is None:
            return default
        S = compute_S_pno(ka, kb, pno_spaces, S_pao_full, s1e)
        if S_pno_cache is not None:
            # CPython dict insert is atomic under the GIL; concurrent
            # threads may compute the same entry but the last write wins
            # and all writes have identical contents — no data race.
            S_pno_cache[(ka, kb)] = S
        return S
    return _get


def build_D_tilde(t1_pno, t2_pno_all, pno_spaces, nocc,
                  ovL_bare, ooL_bare, S_pno_cache, with_df,
                  _term2_precomputed=None, cc_ints=None,
                  pair_lmo_idx=None, _pool=None,
                  S_pao_full=None, s1e=None,
                  t1_cache=None):
    """Build D_tilde (delta, Eq 84) for all ordered (i,k) pairs.

    delta_{ik}^{ac} = Terms 1-4 of Eq 84, using M/L integrals.
    Term 2 uses a batched single DF pass over all (i,k) pairs.

    Following Psi4 ccsd.cc compute_D_tilde() lines 1753-1818.
    """
    from pyscf.ao2mo import _ao2mo

    D_tilde_all = {}

    # Phase 1: cache t1 projections once (or reuse one passed from driver).
    if t1_cache is None:
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, build_t1_cache,
        )
        _pi = PairIndex(
            pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
        t1_cache = build_t1_cache(
            t1_pno, _pi, S_pno_cache, pno_spaces)

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
            t1_i_ik = t1_cache[key_ik][i_idx]
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

    # Unified S_pno accessor (cache hit → cached value; cache miss →
    # same X_pno + S_pao_full path as the upfront build, so restricting
    # the cache is safe to machine precision).
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    # --- Build D_tilde for each (i,k) pair ---
    def _process_ik(ik_tuple):
        i_idx, k_idx = ik_tuple
        key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
        n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
        if n_ik == 0:
            return ik_tuple, None

        D_tilde_ik = np.zeros((n_ik, n_ik))

        # T1_i projected to PNO_ik
        t1_i_ik = t1_cache[key_ik][i_idx]

        # Restrict LMO iteration to pair (i,k)'s local domain for Term 1.
        if pair_lmo_idx is not None and key_ik in pair_lmo_idx:
            _ll_idx_ik = np.asarray(pair_lmo_idx[key_ik])
        else:
            _ll_idx_ik = np.arange(nocc)

        # Phase 1: T1_all_ik = full (nocc, n_pno_ik) T1 projection table.
        # Cached copy already has zero rows for LMOs outside the domain, so
        # downstream Term 3 indexing T1_all_ik[ll] just works.
        T1_all_ik = t1_cache[key_ik]

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

            t1_i_lk = t1_cache[key_lk][i_idx]
            if np.max(np.abs(t1_i_lk)) < 1e-15:
                continue
            Lt1 = L_lk @ t1_i_lk  # (n_lk,)
            # Project to PNO_ik
            S_ik_lk = _s_pno_get(key_ik, key_lk)
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
                return _s_pno_get(ka, kb)

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
                    pair_lmo_idx=None, _pool=None,
                    S_pao_full=None, s1e=None,
                    t1_cache=None):
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
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    # Phase 1: t1 projections are cached once per CCSD cycle.  If caller
    # passed ``t1_cache``, use it; otherwise build a local one (back-compat
    # for callers that predate the cache-based migration).
    if t1_cache is None:
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, build_t1_cache,
        )
        _pi = PairIndex(
            pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
        t1_cache = build_t1_cache(
            t1_pno, _pi, S_pno_cache, pno_spaces)

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
            t1_i_ki = t1_cache[key_ki][i]
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
            t1_i_ki = t1_cache[key_ki][i]
            z = k_Qa @ t1_i_ki
            T2_contrib = np.tensordot(z, Qab_ki, axes=(0, 0))
            C_tilde_ki += T2_contrib
            if _dump_terms:
                ev = np.sort(np.linalg.eigvalsh(0.5*(T2_contrib+T2_contrib.T)))[::-1]
                _term_log['T2'] = (float(np.sqrt(np.mean(T2_contrib**2))), ev[0], ev[-1])
        elif (k, i) in _term2_precomputed:
            C_tilde_ki += _term2_precomputed[(k, i)]

        # --- Term 1: -Σ_l T1_all[l,a] · (ki|lc) ---
        # Restricted to l ∈ lmopair_to_lmos_[ki] (Psi4 line 1722).
        if pair_lmo_idx is not None and key_ki in pair_lmo_idx:
            _ll_idx = np.asarray(pair_lmo_idx[key_ki])
        else:
            _ll_idx = np.arange(nocc)
        _domain_ki = set(int(x) for x in _ll_idx)

        # Phase 1: fancy-index into cached (nocc, n_pno) matrix replaces
        # the per-l projection loop.
        T1_local_ki = np.ascontiguousarray(
            t1_cache[key_ki][np.asarray(_ll_idx, dtype=np.intp)])

        K_bar_chem_local = np.zeros((len(_ll_idx), n_ki))
        if key_ki in cc_ints and cc_ints[key_ki] is not None:
            from pyscf.cc.dlpno_tccsd.local_df import get_local_ovL, get_local_ooL_vec
            _ooL_ki = get_local_ooL_vec(cc_ints, k, i, key_ki)
            if _ooL_ki is not None:
                for _li, _ll in enumerate(_ll_idx):
                    _ovL_l = get_local_ovL(cc_ints, key_ki, int(_ll))
                    if _ovL_l is not None:
                        K_bar_chem_local[_li] = _ovL_l @ _ooL_ki
            else:
                ooL_ki = ooL_bare[k, i, :]
                for _li, _ll in enumerate(_ll_idx):
                    ovL_l_ki = ovL_pno_bare.get((key_ki, int(_ll)))
                    if ovL_l_ki is not None:
                        K_bar_chem_local[_li] = ovL_l_ki @ ooL_ki
        else:
            ooL_ki = ooL_bare[k, i, :]
            for _li, _ll in enumerate(_ll_idx):
                ovL_l_ki = ovL_pno_bare.get((key_ki, int(_ll)))
                if ovL_l_ki is not None:
                    K_bar_chem_local[_li] = ovL_l_ki @ ooL_ki

        T1_contrib = -T1_local_ki.T @ K_bar_chem_local
        C_tilde_ki += T1_contrib  # (n_ki, n_ki)

        # Scatter T1_local_ki into T1_all_ki for downstream Term 3/Term 4
        # code that still uses full-nocc indexing (T1_all_ki[ll]).
        T1_all_ki = np.zeros((nocc, n_ki))
        for _li, _ll in enumerate(_ll_idx):
            T1_all_ki[int(_ll)] = T1_local_ki[_li]
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
            t1_i_kl = t1_cache[key_kl][i]
            if np.max(np.abs(t1_i_kl)) < 1e-15:
                continue

            # Per Eq 83: t1_i contracts with b (first index of K_kl^{bc} where
            # K_iajb[kl][a,b]=(ka|lb) has "a" at k-partner position).
            Kt1 = K_kl.T @ t1_i_kl  # (n_kl,)
            S_ki_kl = _s_pno_get(key_ki, key_kl)
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
                return _s_pno_get(ka, kb)

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


# =========================================================================
# Prototype: compute_C_tilde_batched (plan-cached)
# =========================================================================
# Architecture test for "no per-pair Python loops; batched matmul with
# BLAS threading."  Terms 1 and 2 run per-pair (same code path as the
# reference).  Terms 3 and 4 — the per-l inner Python loops inside the
# reference — are replaced with a gather/bucket/batched-matmul/scatter
# pipeline across ALL (k, i, l) triples, with BLAS threading enabled for
# the matmul phase via threadpool_limits.
#
# The "plan" (bucket shape grouping, pre-stacked constant tensors K/S,
# scatter indices) is built once per run and cached on the function
# object.  Per-cycle work is only: gather T1/T2 values (which change),
# batched matmul, scatter.
# =========================================================================


def _build_c_tilde_t34_plan(
        all_pairs, pno_spaces, pair_lmo_idx, t2_pno_all,
        S_pno_cache, cc_ints, _s_pno_get, nocc):
    """Build the one-time plan for Terms 3 and 4.

    Returns dict with keys ``'t3'`` and ``'t4'``, each a list of per-shape
    bucket dicts carrying pre-stacked constant tensors (K, S) and the
    index arrays for T1/T2 gather + output scatter.  Per-cycle execution
    only needs to gather T1/T2 values and run batched matmuls; no Python
    loop over triples.
    """
    from pyscf.cc.dlpno_tccsd.local_df import get_local_K

    # --- Collect all valid (k, i, l) triples for Terms 3 and 4 ---
    t3_items = []  # (k, i, l, K_kl, S_ki_kl, key_kl, key_ki, n_ki, n_kl)
    t4_items = []  # (k, i, S_ki_li, S_li_kl, K_kl, S_kl_ki, key_li, transpose, n_ki, n_kl, n_li)

    for (k, i) in all_pairs:
        key_ki = (min(k, i), max(k, i))
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        if n_ki == 0:
            continue
        if pair_lmo_idx is not None and key_ki in pair_lmo_idx:
            _ll_idx = pair_lmo_idx[key_ki]
        else:
            _ll_idx = np.arange(nocc)

        for ll_raw in _ll_idx:
            ll = int(ll_raw)
            key_kl = (min(k, ll), max(k, ll))

            # Term 3 gather
            if key_kl in pno_spaces:
                n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
                if n_kl > 0:
                    K_kl = get_local_K(cc_ints, key_kl, k, ll)
                    if K_kl is not None:
                        S_ki_kl = _s_pno_get(key_ki, key_kl)
                        if S_ki_kl is not None:
                            t3_items.append((k, i, ll, K_kl, S_ki_kl,
                                             key_kl, key_ki, n_ki, n_kl))

            # Term 4 gather
            key_li = (min(ll, i), max(ll, i))
            if key_li not in t2_pno_all:
                continue
            if key_kl not in pno_spaces:
                continue
            n_li = pno_spaces[key_li]['C_pno'].shape[1]
            n_kl = pno_spaces[key_kl]['C_pno'].shape[1]
            if n_li == 0 or n_kl == 0:
                continue
            K_kl = get_local_K(cc_ints, key_kl, k, ll)
            if K_kl is None:
                continue
            transpose = (ll > i)
            S_ki_li = (np.eye(n_ki) if key_ki == key_li
                       else _s_pno_get(key_ki, key_li))
            S_li_kl = (np.eye(n_li) if key_li == key_kl
                       else _s_pno_get(key_li, key_kl))
            S_kl_ki = (np.eye(n_kl) if key_kl == key_ki
                       else _s_pno_get(key_kl, key_ki))
            if S_ki_li is None or S_li_kl is None or S_kl_ki is None:
                continue
            t4_items.append((k, i, S_ki_li, S_li_kl, K_kl, S_kl_ki,
                             key_li, transpose, n_ki, n_kl, n_li))

    # --- Global pair→(n_ki, flat_slot) map: covers Phase 1 + Phase 2 outputs ---
    # Every (k, i) with n_ki > 0 gets a slot in flat_out[n_ki].  Phase 1
    # results are copied into flat_out before Numba kernels accumulate
    # Terms 3 + 4 on top; at function exit flat_out is unpacked into
    # C_tilde_all dict.
    pairs_by_n_ki = {}
    pair_to_slot = {}
    for (k, i) in all_pairs:
        key_ki = (min(k, i), max(k, i))
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        if n_ki == 0:
            continue
        if n_ki not in pairs_by_n_ki:
            pairs_by_n_ki[n_ki] = []
        slot = len(pairs_by_n_ki[n_ki])
        pairs_by_n_ki[n_ki].append((k, i))
        pair_to_slot[(k, i)] = slot

    # --- Bucket Term 3 by (n_ki, n_kl); pre-stack K, S, gather index arrays ---
    t3_by_shape = {}
    for it in t3_items:
        t3_by_shape.setdefault((it[7], it[8]), []).append(it)

    t3_buckets = []
    for (n_ki, n_kl), items in t3_by_shape.items():
        N = len(items)
        K = np.empty((N, n_kl, n_kl))
        S = np.empty((N, n_ki, n_kl))
        t1i_keys = []      # list of (key_kl, i)  → T1_cache lookup for t1_i in PNO_kl
        T1l_keys = []      # list of (key_ki, l)  → T1_cache lookup for t1_l in PNO_ki
        # item_idx points into flat_out[n_ki] directly (GLOBAL slot),
        # so the Numba kernel can scatter without per-bucket unpacking.
        item_idx = np.empty(N, dtype=np.intp)
        for n, it in enumerate(items):
            k, i, ll = it[0], it[1], it[2]
            K[n] = it[3]
            S[n] = it[4]
            key_kl, key_ki = it[5], it[6]
            t1i_keys.append((key_kl, i))
            T1l_keys.append((key_ki, ll))
            item_idx[n] = pair_to_slot[(k, i)]
        t3_buckets.append({
            'K': K, 'S': S,
            't1i_keys': t1i_keys, 'T1l_keys': T1l_keys,
            'n_ki': n_ki, 'n_kl': n_kl,
            'item_idx': item_idx,
        })

    # --- Bucket Term 4 by (n_ki, n_kl, n_li); pre-stack 4 S matrices + K ---
    t4_by_shape = {}
    for it in t4_items:
        t4_by_shape.setdefault((it[8], it[9], it[10]), []).append(it)

    t4_buckets = []
    for (n_ki, n_kl, n_li), items in t4_by_shape.items():
        N = len(items)
        S_ki_li = np.empty((N, n_ki, n_li))
        S_li_kl = np.empty((N, n_li, n_kl))
        K = np.empty((N, n_kl, n_kl))
        S_kl_ki = np.empty((N, n_kl, n_ki))
        t2_sources = []    # list of (key_li, transpose)
        item_idx = np.empty(N, dtype=np.intp)
        for n, it in enumerate(items):
            k, i = it[0], it[1]
            S_ki_li[n] = it[2]
            S_li_kl[n] = it[3]
            K[n] = it[4]
            S_kl_ki[n] = it[5]
            t2_sources.append((it[6], bool(it[7])))
            item_idx[n] = pair_to_slot[(k, i)]
        t4_buckets.append({
            'S_ki_li': S_ki_li, 'S_li_kl': S_li_kl,
            'K': K, 'S_kl_ki': S_kl_ki,
            't2_sources': t2_sources,
            'n_ki': n_ki, 'n_kl': n_kl, 'n_li': n_li,
            'item_idx': item_idx,
        })

    return {
        't3': t3_buckets, 't4': t4_buckets,
        'pairs_by_n_ki': pairs_by_n_ki,
        'pair_to_slot': pair_to_slot,
    }


def compute_C_tilde_batched(
        t1_pno, t2_pno_all, pno_spaces, nocc,
        ovL_pno_bare, ooL_bare, S_pno_cache, with_df,
        _term2_precomputed=None, cc_ints=None,
        pair_lmo_idx=None, _pool=None,
        S_pao_full=None, s1e=None,
        blas_threads=32):
    """Drop-in replacement for compute_C_tilde with Terms 3+4 batched.

    Signature matches compute_C_tilde exactly plus one kwarg
    ``blas_threads`` controlling the threadpool_limits scope used during
    the batched phase (set to None to keep the caller's BLAS thread
    count).  Output dict format and values match the reference to
    machine precision.
    """
    import time as _time_dbg
    _dbg = getattr(compute_C_tilde_batched, '_dump_timing', False)
    _pt = {}
    _t0 = _time_dbg.perf_counter()

    from pyscf.cc.dlpno_tccsd.local_df import (
        get_local_K, get_local_ovL, get_local_ooL_vec,
    )

    C_tilde_all = {}
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    # All ordered (k, i) pairs gamma_{ki} is defined over.
    all_pairs = set()
    canonical_keys = set()
    for key in t2_pno_all:
        a, b = key
        all_pairs.add((a, b))
        all_pairs.add((b, a))
        canonical_keys.add(key)

    # ------------------------------------------------------------------
    # T1 projection cache: T1_cache[(canonical_pair, l)] = (n_pno,)
    # Built once up front; every phase reads from it.
    # ------------------------------------------------------------------
    T1_cache = {}
    for pk in canonical_keys:
        if pno_spaces[pk]['C_pno'].shape[1] == 0:
            continue
        for l in range(nocc):
            T1_cache[(pk, l)] = _project_t1_to_pair(
                t1_pno, l, pk, S_pno_cache, pno_spaces)
    _pt['T1_cache'] = _time_dbg.perf_counter() - _t0; _t0 = _time_dbg.perf_counter()

    # ------------------------------------------------------------------
    # Term 2 DF-pass fallback (only when cc_ints doesn't cover a pair).
    # This is the same code as the reference's fallback branch.
    # ------------------------------------------------------------------
    if _term2_precomputed is None:
        term2_data = {}
        for k, i in all_pairs:
            key_ki = (min(k, i), max(k, i))
            n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
            if n_ki == 0:
                continue
            t1_i_ki = T1_cache.get((key_ki, i))
            if t1_i_ki is None or np.max(np.abs(t1_i_ki)) < 1e-15:
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

    # ------------------------------------------------------------------
    # Phase 1: Terms 1 + 2 per-pair (same shape as reference; pool if given)
    # ------------------------------------------------------------------
    def _process_ki_terms12(ki_tuple):
        k, i = ki_tuple
        key_ki = (min(k, i), max(k, i))
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        if n_ki == 0:
            return ki_tuple, None

        C_tilde_ki = np.zeros((n_ki, n_ki))

        # --- Term 2 ---
        if key_ki in cc_ints and cc_ints[key_ki] is not None:
            ci_ki = cc_ints[key_ki]
            k_Qa = ci_ki['i_Qa'] if key_ki[0] == k else ci_ki['j_Qa']
            Qab_ki = ci_ki['Qab']
            t1_i_ki = T1_cache.get((key_ki, i), np.zeros(n_ki))
            z = k_Qa @ t1_i_ki
            C_tilde_ki += np.tensordot(z, Qab_ki, axes=(0, 0))
        elif (k, i) in _term2_precomputed:
            C_tilde_ki += _term2_precomputed[(k, i)]

        # --- Term 1: -Σ_l T1_all[l,a] · (ki|lc) ---
        if pair_lmo_idx is not None and key_ki in pair_lmo_idx:
            _ll_idx = np.asarray(pair_lmo_idx[key_ki])
        else:
            _ll_idx = np.arange(nocc)

        T1_local_ki = np.zeros((len(_ll_idx), n_ki))
        for _li, _ll in enumerate(_ll_idx):
            T1_local_ki[_li] = T1_cache.get(
                (key_ki, int(_ll)), np.zeros(n_ki))

        K_bar_chem_local = np.zeros((len(_ll_idx), n_ki))
        if key_ki in cc_ints and cc_ints[key_ki] is not None:
            _ooL_ki = get_local_ooL_vec(cc_ints, k, i, key_ki)
            if _ooL_ki is not None:
                for _li, _ll in enumerate(_ll_idx):
                    _ovL_l = get_local_ovL(cc_ints, key_ki, int(_ll))
                    if _ovL_l is not None:
                        K_bar_chem_local[_li] = _ovL_l @ _ooL_ki
            else:
                ooL_ki = ooL_bare[k, i, :]
                for _li, _ll in enumerate(_ll_idx):
                    ovL_l_ki = ovL_pno_bare.get((key_ki, int(_ll)))
                    if ovL_l_ki is not None:
                        K_bar_chem_local[_li] = ovL_l_ki @ ooL_ki
        else:
            ooL_ki = ooL_bare[k, i, :]
            for _li, _ll in enumerate(_ll_idx):
                ovL_l_ki = ovL_pno_bare.get((key_ki, int(_ll)))
                if ovL_l_ki is not None:
                    K_bar_chem_local[_li] = ovL_l_ki @ ooL_ki

        C_tilde_ki += -T1_local_ki.T @ K_bar_chem_local
        return ki_tuple, C_tilde_ki

    if _pool is not None:
        for ki, val in _pool.map(_process_ki_terms12, list(all_pairs)):
            if val is not None:
                C_tilde_all[ki] = val
    else:
        for ki in all_pairs:
            _, val = _process_ki_terms12(ki)
            if val is not None:
                C_tilde_all[ki] = val
    _pt['phase1_t12'] = _time_dbg.perf_counter() - _t0; _t0 = _time_dbg.perf_counter()

    # ------------------------------------------------------------------
    # Phase 2: Terms 3 + 4 via plan cache.
    #
    # Plan depends only on the pair structure (pno_spaces, pair_lmo_idx,
    # S_pno_cache contents, cc_ints integrals) — all fixed after Stage 3.
    # Built once per CCSD run and reused every cycle; per-cycle cost is
    # just T1/T2 gather + batched matmul + scatter.
    # ------------------------------------------------------------------
    plan_key = tuple(sorted(t2_pno_all.keys()))
    _cache_attr = getattr(compute_C_tilde_batched, '_plan_cache', None)
    if _cache_attr is None:
        _cache_attr = {}
        compute_C_tilde_batched._plan_cache = _cache_attr
    plan = _cache_attr.get(plan_key)
    if plan is None:
        plan = _build_c_tilde_t34_plan(
            all_pairs, pno_spaces, pair_lmo_idx, t2_pno_all,
            S_pno_cache, cc_ints, _s_pno_get, nocc)
        _cache_attr[plan_key] = plan

    # ------------------------------------------------------------------
    # Migrate Phase 1 results into flat per-n_ki buffers; Numba kernels
    # will accumulate Terms 3 + 4 on top.  All scatter-add happens inside
    # Numba (race-free via two-stage: parallel compute → serial add).
    # ------------------------------------------------------------------
    from pyscf.cc.dlpno_tccsd._c_tilde_numba import (
        t3_kernel as _t3_numba, t4_kernel as _t4_numba,
    )

    _t_alloc = _time_dbg.perf_counter()
    flat_out = {}
    for n_ki, pairs in plan['pairs_by_n_ki'].items():
        buf = np.zeros((len(pairs), n_ki, n_ki))
        for slot, pair in enumerate(pairs):
            v = C_tilde_all.get(pair)
            if v is not None:
                buf[slot] = v
        flat_out[n_ki] = buf
    _pt['flat_alloc'] = _time_dbg.perf_counter() - _t_alloc

    _pt['t3_gather'] = 0.0; _pt['t3_numba'] = 0.0
    _pt['t4_gather'] = 0.0; _pt['t4_numba'] = 0.0

    # ----- Term 3 via Numba kernel -----
    for bucket in plan['t3']:
        _tg = _time_dbg.perf_counter()
        t1i = np.ascontiguousarray(
            np.array([T1_cache[key] for key in bucket['t1i_keys']]))
        T1l = np.ascontiguousarray(
            np.array([T1_cache[key] for key in bucket['T1l_keys']]))
        _pt['t3_gather'] += _time_dbg.perf_counter() - _tg

        _tn = _time_dbg.perf_counter()
        _t3_numba(bucket['K'], bucket['S'], t1i, T1l,
                  bucket['item_idx'], flat_out[bucket['n_ki']])
        _pt['t3_numba'] += _time_dbg.perf_counter() - _tn

    # ----- Term 4 via Numba kernel -----
    for bucket in plan['t4']:
        _tg = _time_dbg.perf_counter()
        t2_arr = np.ascontiguousarray(np.array([
            (t2_pno_all[k_].T if tr else t2_pno_all[k_])
            for (k_, tr) in bucket['t2_sources']
        ]))
        _pt['t4_gather'] += _time_dbg.perf_counter() - _tg

        _tn = _time_dbg.perf_counter()
        _t4_numba(bucket['S_ki_li'], t2_arr, bucket['S_li_kl'],
                  bucket['K'], bucket['S_kl_ki'],
                  bucket['item_idx'], flat_out[bucket['n_ki']])
        _pt['t4_numba'] += _time_dbg.perf_counter() - _tn

    # Unpack flat_out back into C_tilde_all dict.
    _t_unpack = _time_dbg.perf_counter()
    for n_ki, pairs in plan['pairs_by_n_ki'].items():
        buf = flat_out[n_ki]
        for slot, pair in enumerate(pairs):
            C_tilde_all[pair] = buf[slot]
    _pt['unpack'] = _time_dbg.perf_counter() - _t_unpack

    if _dbg:
        _per = ' '.join(f'{k}={v*1e3:.0f}ms' for k, v in _pt.items())
        _tot = sum(_pt.values())
        print(f'  [c_tilde_batched_timing] tot={_tot*1e3:.0f}ms '
              f'{_per}', flush=True)

    return C_tilde_all


# ---------------------------------------------------------------------------
# T1-dressed Fock intermediates (Eqs 94-101, 85-86)
# ---------------------------------------------------------------------------


def compute_all_df_terms_local(t1_pno, fov_pno, t2_pno_all, pno_spaces,
                                nocc, cc_ints, S_pno_cache, pair_keys,
                                _pool=None, pair_lmo_idx=None,
                                t1_cache=None):
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
    # Phase 1: cache t1 projections once (or reuse one from driver).
    if t1_cache is None:
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, build_t1_cache,
        )
        _pi = PairIndex(
            pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
        t1_cache = build_t1_cache(
            t1_pno, _pi, S_pno_cache, pno_spaces)

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

        Qab = ci['Qab']                 # (n_local, n, n)
        Qma_full = ci['Qma']            # (n_local, nocc, n)
        i, j = pk

        # Restrict inner k-sum to pair's local LMO domain (Psi4
        # lmopair_to_lmos_[ij]). For each k in the full nocc, the
        # contribution is ~zero for k outside the domain; restricting
        # collapses O(nocc·n²) tensor contractions to O(nlmo·n²).
        if pair_lmo_idx is not None and pk in pair_lmo_idx:
            lmo_idx = np.asarray(pair_lmo_idx[pk])
        else:
            lmo_idx = np.arange(nocc)
        Qma = Qma_full[:, lmo_idx, :]   # (n_local, nlmo, n)

        # Phase 1: fancy-index into cached (nocc, n_pno) projection matrix.
        T1_local = np.ascontiguousarray(
            t1_cache[pk][np.asarray(lmo_idx, dtype=np.intp)])

        # fvv_t1
        z = np.einsum('Lka,ka->L', Qma, T1_local, optimize=True)
        fvv = 2.0 * np.einsum('L,Lab->ab', z, Qab, optimize=True)
        tmp = np.einsum('Lab,kb->Lka', Qab, T1_local, optimize=True)
        fvv -= np.einsum('Lka,Lkb->ab', tmp, Qma, optimize=True)

        # ladder — t1_i and t1_j: cache has zero rows for out-of-domain
        # LMOs, so indexing the cached matrix covers both paths in one line.
        t1_i = t1_cache[pk][i]
        t1_j = t1_cache[pk][j]
        tau = t2_pno_all[pk] + np.outer(t1_i, t1_j)
        corr = np.einsum('ka,Lkb->Lab', T1_local, Qma, optimize=True)
        B_tilde = Qab - corr
        X_L = np.matmul(B_tilde, tau)
        ladder = (X_L.transpose(1, 0, 2).reshape(n, -1) @
                  B_tilde.transpose(1, 0, 2).reshape(n, -1).T)
        # Debug: print for diagonal (0,0), off-diag (0,5), diagonal (5,5)
        _it = getattr(compute_all_df_terms_local, '_iter', 0)
        if _it <= 2 and pk in [(0, 0), (0, 5), (5, 5)]:
            print(f'  LADDER_DBG iter {_it} '
                  f'pair{pk}: |T1_all|={np.linalg.norm(T1_local):.3e} '
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
        t1_i = t1_cache[key_ki][i]
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
        t1_i = t1_cache[key_ik][i_idx]
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
# Batched B and E terms — moved out of per-pair compute_residual_v2 so
# that the many small S @ t2 @ S.T ops can be vectorised into batched BLAS.
# =========================================================================


def compute_B_E_batched(strong_keys, t2_pno_all, pno_spaces, S_pno_cache,
                        cc_ints, B_tilde_per_ij, pair_lmo_idx, nocc, _pool=None,
                        S_pao_full=None, s1e=None):
    """Compute B and E-contribution dicts for all strong pairs, batched.

    B_ij = Σ_{kl ∈ domain_ij²} β[k,l] · S_{kl→ij} @ t2_kl @ S_{kl→ij}.T
           + (l≠k:) β[l,k] · S_{kl→ij} @ t2_kl.T @ S_{kl→ij}.T

    E_contrib_ij = Σ_{kl ∈ domain_ij²} S_{kl→ij} @ (u_kl @ K_kl.T
                                       + (k≠l: (2t2_kl.T - t2_kl) @ K_kl))
                                       @ S_{kl→ij}.T

    where u_kl = 2 t2_kl - t2_kl.T.  Both terms share identical projection
    structure; batching them together halves the BLAS overhead relative
    to two separate loops.

    ``B_tilde_per_ij`` is a dict keyed by strong pair: each entry is the
    pair's (nocc, nocc) B_tilde matrix.

    Returns:
        B_all: dict key_ij → (n_ij, n_ij) B term
        E_contrib_all: dict key_ij → (n_ij, n_ij) E subtracted contribution
    """
    B_all = {}
    E_all = {}
    _s_pno_get = _s_pno_getter(S_pno_cache, pno_spaces, S_pao_full, s1e)

    def _per_ij(key_ij):
        i, j = key_ij
        n_ij = pno_spaces[key_ij]['C_pno'].shape[1]
        if n_ij == 0:
            return key_ij, np.zeros((n_ij, n_ij)), np.zeros((n_ij, n_ij))

        B_tilde = B_tilde_per_ij[key_ij]
        domain = (set(int(x) for x in pair_lmo_idx[key_ij])
                  if pair_lmo_idx is not None and key_ij in pair_lmo_idx
                  else set(range(nocc)))

        # --- Gather phase: collect pointers only (no per-kl math). ---
        # Bucket by n_kl so each bucket is a uniform-shape batched contraction.
        # Each entry carries (S, t2, K, beta_kl, beta_lk, same) where same = (k==l).
        buckets = {}  # n_kl -> list of (S, t2, K, beta_kl, beta_lk, same)
        for key_kl, t2_kl in t2_pno_all.items():
            if t2_kl is None or t2_kl.shape[0] == 0:
                continue
            k, l = key_kl
            if k not in domain or l not in domain:
                continue
            S = _s_pno_get(key_ij, key_kl)
            if S is None:
                continue
            K_kl = get_local_K(cc_ints, key_kl, k, l)
            if K_kl is None:
                continue
            same = (k == l)
            beta_kl = B_tilde[k, l]
            beta_lk = 0.0 if same else B_tilde[l, k]
            n_kl = t2_kl.shape[0]
            buckets.setdefault(n_kl, []).append(
                (S, t2_kl, K_kl, beta_kl, beta_lk, same))

        if not buckets:
            return key_ij, np.zeros((n_ij, n_ij)), np.zeros((n_ij, n_ij))

        # --- Batched phase: stack, then do per-bucket batched operations. ---
        # This replaces ~n_kept per-element numpy calls (which dominated at
        # small PNO size via BLAS-dispatch overhead) with O(1) big BLAS calls.
        B_sum = np.zeros((n_ij, n_ij))
        E_sum = np.zeros((n_ij, n_ij))
        for n_kl, items in buckets.items():
            N = len(items)
            S_arr = np.empty((N, n_ij, n_kl))
            T_arr = np.empty((N, n_kl, n_kl))
            K_arr = np.empty((N, n_kl, n_kl))
            b_kl_arr = np.empty(N)
            b_lk_arr = np.empty(N)
            same_arr = np.zeros(N, dtype=bool)
            for n, (S, t2, K, bkl, blk, same) in enumerate(items):
                S_arr[n] = S
                T_arr[n] = t2
                K_arr[n] = K
                b_kl_arr[n] = bkl
                b_lk_arr[n] = blk
                same_arr[n] = same

            T_T = T_arr.transpose(0, 2, 1)

            # TB[n] = β_kl·t2_n (for k==l)  OR  β_kl·t2_n + β_lk·t2_n.T (k!=l)
            TB = b_kl_arr[:, None, None] * T_arr
            if (~same_arr).any():
                TB = TB + b_lk_arr[:, None, None] * T_T

            # UK[n] = u·K.T (for k==l)  OR  u·K.T + (2 t2.T - t2)·K (k!=l)
            u = 2.0 * T_arr - T_T                        # (N, n_kl, n_kl)
            UK = np.matmul(u, K_arr.transpose(0, 2, 1))  # (N, n_kl, n_kl)
            if (~same_arr).any():
                v = 2.0 * T_T - T_arr
                vk = np.matmul(v, K_arr)
                vk[same_arr] = 0.0
                UK = UK + vk

            # B_sum += Σ_n S_n @ TB_n @ S_n.T  (batched)
            S_T = S_arr.transpose(0, 2, 1)
            B_sum += np.matmul(np.matmul(S_arr, TB), S_T).sum(axis=0)
            E_sum += np.matmul(np.matmul(S_arr, UK), S_T).sum(axis=0)

        return key_ij, B_sum, E_sum

    if _pool is not None:
        for key, B, E in _pool.map(_per_ij, list(strong_keys)):
            B_all[key] = B
            E_all[key] = E
    else:
        for key in strong_keys:
            key, B, E = _per_ij(key)
            B_all[key] = B
            E_all[key] = E

    return B_all, E_all


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
        B_term_override=None,
        E_contrib_override=None,
        S_pao_full=None,
        t1_cache=None,
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

    from pyscf.cc.dlpno_tccsd.local_df import compute_S_pno as _compute_S_pno
    _getS_misses = [0, 0]  # [get_S, get_S2]

    def _get_S(key_other):
        if S_pno_cache is not None:
            S = S_pno_cache.get((key, key_other))
            if S is not None:
                return S
        _getS_misses[0] += 1
        if S_pao_full is not None:
            return _compute_S_pno(key, key_other, pno_spaces, S_pao_full, s1e)
        return C_pno_ij.T @ (s1e @ pno_spaces[key_other]['C_pno'])

    def _get_S2(key_a, key_b):
        """Get S overlap between any two pair keys, with identity fallback."""
        if key_a == key_b:
            return np.eye(pno_spaces[key_a]['C_pno'].shape[1])
        if S_pno_cache is not None:
            S = S_pno_cache.get((key_a, key_b))
            if S is not None:
                return S
        _getS_misses[1] += 1
        if S_pao_full is not None:
            return _compute_S_pno(key_a, key_b, pno_spaces, S_pao_full, s1e)
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
    # Phase 1: use pre-built T1 cache; fall back to lazy build otherwise.
    if t1_cache is not None:
        T1_all_ij = t1_cache[key]
    elif t1_pno is not None:
        T1_all_ij = np.zeros((nocc, n_pno))
        for kk in range(nocc):
            T1_all_ij[kk] = _project_t1_to_pair(
                t1_pno, kk, key, S_pno_cache, pno_spaces)
    else:
        T1_all_ij = np.zeros((nocc, n_pno))

    t1_i_pno = T1_all_ij[i] if t1_pno else np.zeros(n_pno)
    t1_j_pno = T1_all_ij[j] if t1_pno else np.zeros(n_pno)
    tau_ij = t2_ij + np.outer(t1_i_pno, t1_j_pno)

    A_term = ladder_precomputed[key] if (
        ladder_precomputed is not None and key in ladder_precomputed
    ) else np.zeros((n_pno, n_pno))
    R_sym += A_term
    if _DEBUG_PTERM:
        print(f"  PTERM iter {_ITER} pair({i},{j}): A_rms={_rms(A_term):.12f} R_KA={_rms(R_sym):.12f}")

    import time as _time
    _pt = {}
    _t0 = _time.perf_counter()
    # --- B (Eq 77/82): Woooo with dressed β ---
    # Use batched override if provided (compute_B_E_batched); else in-place.
    if B_term_override is not None:
        B_term = B_term_override
    else:
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
    _pt['B'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
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
    if E_contrib_override is not None:
        E_tilde -= E_contrib_override
    else:
        for key_kl, t2_kl in t2_pno_all.items():
            if t2_kl is None or t2_kl.shape[0] == 0:
                continue
            k, l = key_kl
            if k not in _domain_set or l not in _domain_set:
                continue
            u_kl = 2.0 * t2_kl - t2_kl.T
            _K = get_local_K(cc_ints, key_kl, k, l)
            if _K is None:
                print(f"WARN: No local DF K for pair {key_kl}, skipping E contribution")
                continue
            K_kl = _K
            S_kl_ij = _get_S(key_kl)
            E_tilde -= S_kl_ij @ (u_kl @ K_kl.T) @ S_kl_ij.T
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
    _pt['E'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
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
        _pt['C'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()
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
    _pt['D'] = _time.perf_counter() - _t0; _t0 = _time.perf_counter()

    # --- G (Eq 81): Fock oo coupling ---
    # Psi4 restricts the k sum to lmopair_to_lmos_[ij]; doing so here
    # makes the S_pno_cache restriction self-consistent (no lookups for
    # k outside domain) and matches Psi4's formula.
    G_ij = np.zeros((n_pno, n_pno))
    G_ji = np.zeros((n_pno, n_pno))
    for k in sorted(_domain_set):
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
    _pt['G'] = _time.perf_counter() - _t0

    # Dump per-term timings into a module-level aggregator (optional)
    _accum = getattr(compute_residual_v2, '_term_times', None)
    if _accum is not None:
        for k_, v_ in _pt.items():
            _accum[k_] = _accum.get(k_, 0.0) + v_
        _accum['_n'] = _accum.get('_n', 0) + 1
        _accum['miss_S'] = _accum.get('miss_S', 0) + _getS_misses[0]
        _accum['miss_S2'] = _accum.get('miss_S2', 0) + _getS_misses[1]

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
