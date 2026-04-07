"""Unified single-DF-pass computation of fvv_t1, C_tilde Term 2, D_tilde Term 2.

Merges three separate DF loops into one pass over the DF integrals, reducing
I/O overhead and improving cache utilization.
"""
import numpy as np
from pyscf.ao2mo import _ao2mo
from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair


def compute_df_terms_batched(t1_pno, fov_pno, t2_pno_all, pno_spaces, nocc,
                              ovL_pno_bare, S_pno_cache, with_df,
                              pair_keys):
    """Single DF pass computing fvv_t1 (Fab), C_tilde Term 2, D_tilde Term 2.

    Returns:
        fvv_t1_all: dict pair_key -> (n_pno, n_pno) for Fab construction
        c_term2_all: dict (k, i) -> (n_ki, n_ki) C_tilde Term 2 contributions
        d_term2_all: dict (i_idx, k_idx) -> (n_ik, n_ik) D_tilde Term 2 contributions
    """
    # === Prepare data for all three computations ===

    # Fab/fvv_t1: one entry per pair_key
    fab_data = {}
    for pk in pair_keys:
        n_pno = pno_spaces[pk]['C_pno'].shape[1]
        if n_pno == 0:
            continue
        naux = 0
        for m in range(nocc):
            entry = ovL_pno_bare.get((pk, m))
            if entry is not None:
                naux = entry.shape[1]
                break
        if naux == 0:
            continue

        z_total = np.zeros(naux)
        t1_proj_all = {}
        ovL_k_all = {}
        for kk in range(nocc):
            t1_k = _project_t1_to_pair(t1_pno, kk, pk, S_pno_cache, pno_spaces)
            if np.max(np.abs(t1_k)) < 1e-15:
                continue
            ovL_k = ovL_pno_bare.get((pk, kk))
            if ovL_k is not None:
                z_total += t1_k @ ovL_k
                t1_proj_all[kk] = t1_k
                ovL_k_all[kk] = ovL_k
        if not t1_proj_all:
            continue
        fab_data[pk] = {
            'n': n_pno, 'C': np.asfortranarray(pno_spaces[pk]['C_pno']),
            'z': z_total, 't1': t1_proj_all, 'ovL': ovL_k_all,
            'result': np.zeros((n_pno, n_pno)),
        }

    # C_tilde Term 2: one entry per ordered (k, i)
    c_data = {}
    all_ordered = set()
    for key in t2_pno_all:
        a, b = key
        all_ordered.add((a, b))
        all_ordered.add((b, a))

    for k, i in all_ordered:
        key_ki = (min(k, i), max(k, i))
        n_ki = pno_spaces[key_ki]['C_pno'].shape[1]
        if n_ki == 0:
            continue
        t1_i = _project_t1_to_pair(t1_pno, i, key_ki, S_pno_cache, pno_spaces)
        if np.max(np.abs(t1_i)) < 1e-15:
            continue
        ovL_i = ovL_pno_bare.get((key_ki, i))
        if ovL_i is None:
            continue
        c_data[(k, i)] = {
            'key': key_ki, 'n': n_ki,
            'C': np.asfortranarray(pno_spaces[key_ki]['C_pno']),
            't1': t1_i, 'ovL': ovL_i,
            'result': np.zeros((n_ki, n_ki)),
        }

    # D_tilde Term 2: one entry per ordered (i_idx, k_idx)
    d_data = {}
    for i_idx, k_idx in all_ordered:
        key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
        n_ik = pno_spaces[key_ik]['C_pno'].shape[1]
        if n_ik == 0:
            continue
        t1_i = _project_t1_to_pair(t1_pno, i_idx, key_ik, S_pno_cache, pno_spaces)
        if np.max(np.abs(t1_i)) < 1e-15:
            continue
        ovL_k = ovL_pno_bare.get((key_ik, k_idx))
        if ovL_k is None:
            continue
        d_data[(i_idx, k_idx)] = {
            'key': key_ik, 'n': n_ik,
            'C': np.asfortranarray(pno_spaces[key_ik]['C_pno']),
            't1': t1_i, 'ovL': ovL_k,
            'result': np.zeros((n_ik, n_ik)),
        }

    # === Single DF pass ===
    # Collect all unique (C_pno, n_pno) that need AO→MO transforms.
    # Many entries share the same C_pno (same pair key), so we can avoid
    # redundant transforms by grouping.
    aux_off = 0
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]

        # --- Fab/fvv_t1 ---
        for pk, fd in fab_data.items():
            n = fd['n']
            buf = _ao2mo.nr_e2(Lpq, fd['C'], (0, n, 0, n), aosym='s2')
            B_L = buf.reshape(nL, n, n)
            # Coulomb: z[L] @ B[L,a,b] -> tensordot over L
            z_batch = fd['z'][aux_off:aux_off+nL]
            fd['result'] += 2.0 * np.tensordot(z_batch, B_L, axes=([0], [0]))
            # Exchange
            for kk, t1_k in fd['t1'].items():
                ovL_k = fd['ovL'].get(kk)
                if ovL_k is None:
                    continue
                # Y[a,L] = sum_c t1[c]*B[L,c,a] = (B reshaped) @ t1
                Y_k = np.tensordot(B_L, t1_k, axes=([1], [0])).transpose(1, 0)
                fd['result'] -= Y_k @ ovL_k[:, aux_off:aux_off+nL].T

        # --- C_tilde Term 2 + D_tilde Term 2 ---
        # Cache AO→MO transforms by canonical pair key: (k,i) and (i,k)
        # share the same key_ki and C_pno, so B_L is identical.
        _B_L_cache = {}
        for (k, i), cd in c_data.items():
            canon_key = cd['key']
            if canon_key not in _B_L_cache:
                n = cd['n']
                buf = _ao2mo.nr_e2(Lpq, cd['C'], (0, n, 0, n), aosym='s2')
                _B_L_cache[canon_key] = buf.reshape(nL, n, n)
            B_L = _B_L_cache[canon_key]
            z_i = cd['t1'] @ cd['ovL'][:, aux_off:aux_off+nL]
            cd['result'] += np.tensordot(z_i, B_L, axes=([0], [0]))

        for (i_idx, k_idx), dd in d_data.items():
            canon_key = dd['key']
            if canon_key not in _B_L_cache:
                n = dd['n']
                buf = _ao2mo.nr_e2(Lpq, dd['C'], (0, n, 0, n), aosym='s2')
                _B_L_cache[canon_key] = buf.reshape(nL, n, n)
            B_L = _B_L_cache[canon_key]
            z_c = np.tensordot(B_L, dd['t1'], axes=([1], [0]))
            y = dd['t1'] @ dd['ovL'][:, aux_off:aux_off+nL]
            dd['result'] += 2.0 * dd['ovL'][:, aux_off:aux_off+nL] @ z_c
            dd['result'] -= np.tensordot(y, B_L, axes=([0], [0]))

        aux_off += nL

    # === Package results ===
    fvv_t1_all = {}
    for pk in pair_keys:
        n = pno_spaces[pk]['C_pno'].shape[1]
        fvv_t1_all[pk] = fab_data[pk]['result'] if pk in fab_data else np.zeros((n, n))

    c_term2_all = {ki: cd['result'] for ki, cd in c_data.items()}
    d_term2_all = {ik: dd['result'] for ik, dd in d_data.items()}

    return fvv_t1_all, c_term2_all, d_term2_all


def compute_ladder_all_pairs(t1_pno, t2_pno_all, pno_spaces, nocc,
                              ovL_bare, S_pno_cache, with_df,
                              pair_keys):
    """Precompute the dressed ladder (Eq 76) for ALL pairs in a single DF pass.

    Replaces per-pair DF loops in compute_residual_v2. The ladder is the
    dominant cost per pair (~0.5s each), so batching N pairs into 1 DF pass
    eliminates N-1 redundant DF reads.

    Returns dict: pair_key -> (n_pno, n_pno) ladder matrix.
    """
    # Precompute per-pair data needed during the DF loop
    pair_data = {}
    for pk in pair_keys:
        i, j = pk
        n_pno = pno_spaces[pk]['C_pno'].shape[1]
        if n_pno == 0:
            continue

        C_pno_ij = pno_spaces[pk]['C_pno']

        # T1 projections for all occ
        T1_all = np.zeros((nocc, n_pno))
        if t1_pno is not None:
            for kk in range(nocc):
                T1_all[kk] = _project_t1_to_pair(
                    t1_pno, kk, pk, S_pno_cache, pno_spaces)

        t1_i = T1_all[i]
        t1_j = T1_all[j]
        tau = t2_pno_all[pk] + np.outer(t1_i, t1_j)

        # T1 correction tensor: corr[a,c,Q] = sum_k T1[k,a]*ovL[k,c,Q]
        # BLAS: T1.T @ ovL.reshape(nocc, n*Q) -> reshape to (n, n, Q)
        ovL_stacked = np.stack(
            [ovL_bare.get((pk, k), np.zeros((n_pno, 0)))
             for k in range(nocc)])  # (nocc, n_pno, naux)
        naux = ovL_stacked.shape[2]
        if naux == 0:
            continue
        corr_full = (T1_all.T @ ovL_stacked.reshape(nocc, -1)).reshape(
            n_pno, n_pno, naux)

        pair_data[pk] = {
            'n': n_pno,
            'C': np.asfortranarray(C_pno_ij),
            'tau': tau,
            'corr': corr_full,
            'ladder': np.zeros((n_pno, n_pno)),
        }

    if not pair_data:
        return {pk: np.zeros((pno_spaces[pk]['C_pno'].shape[1],) * 2)
                for pk in pair_keys}

    # Single DF pass: compute ladder for all pairs per batch
    aux_off = 0
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]

        for pk, pd in pair_data.items():
            n = pd['n']
            buf = _ao2mo.nr_e2(Lpq, pd['C'], (0, n, 0, n), aosym='s2')
            B_L = buf.reshape(nL, n, n)

            # Dress: B̃_L = B_L - corr[:,:,batch]
            corr_batch = pd['corr'][:, :, aux_off:aux_off+nL].transpose(2, 0, 1)
            B_tilde_L = B_L - corr_batch

            # Contract: ladder += sum_L B̃ × tau × B̃.T
            # Use BLAS instead of einsum: 4.4x faster
            n = pd['n']
            X_L = B_tilde_L @ pd['tau']  # batch matmul (nL,n,n)@(n,n)
            # einsum('Lad,Lbd->ab') via reshape + dgemm
            pd['ladder'] += (X_L.transpose(1, 0, 2).reshape(n, -1) @
                             B_tilde_L.transpose(1, 0, 2).reshape(n, -1).T)

        aux_off += nL

    # Package results
    result = {}
    for pk in pair_keys:
        n = pno_spaces[pk]['C_pno'].shape[1]
        result[pk] = pair_data[pk]['ladder'] if pk in pair_data else np.zeros((n, n))
    return result
