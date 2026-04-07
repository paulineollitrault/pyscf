"""Unified single-DF-pass for ALL DF-dependent Jiang intermediates.

Merges fvv_t1 (Fab), C_tilde Term 2, D_tilde Term 2, and the dressed
ladder (Eq 76) into ONE pass over the DF integrals. Caches AO→MO
transforms per unique pair key within each batch to avoid redundant work.
"""
import numpy as np
from pyscf.ao2mo import _ao2mo
from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair


def compute_all_df_terms(t1_pno, fov_pno, t2_pno_all, pno_spaces, nocc,
                          ovL_bare, S_pno_cache, with_df, pair_keys):
    """Single DF pass computing fvv_t1, C_tilde Term 2, D_tilde Term 2, ladder.

    Returns:
        fvv_t1_all: dict pair_key -> (n_pno, n_pno)
        c_term2_all: dict (k, i) -> (n_ki, n_ki)
        d_term2_all: dict (i, k) -> (n_ik, n_ik)
        ladder_all: dict pair_key -> (n_pno, n_pno)
    """
    # ====================================================================
    # Prepare per-computation data structures
    # ====================================================================

    # All ordered (k,i) pairs for C_tilde and D_tilde
    all_ordered = set()
    for key in t2_pno_all:
        a, b = key
        all_ordered.add((a, b))
        all_ordered.add((b, a))

    # --- Fab/fvv_t1 data ---
    fab_data = {}
    for pk in pair_keys:
        n = pno_spaces[pk]['C_pno'].shape[1]
        if n == 0:
            continue
        naux = 0
        for m in range(nocc):
            entry = ovL_bare.get((pk, m))
            if entry is not None:
                naux = entry.shape[1]
                break
        if naux == 0:
            continue
        z_total = np.zeros(naux)
        t1_proj = {}
        ovL_k = {}
        for kk in range(nocc):
            t1_k = _project_t1_to_pair(t1_pno, kk, pk, S_pno_cache, pno_spaces)
            if np.max(np.abs(t1_k)) < 1e-15:
                continue
            ovl = ovL_bare.get((pk, kk))
            if ovl is not None:
                z_total += t1_k @ ovl
                t1_proj[kk] = t1_k
                ovL_k[kk] = ovl
        if t1_proj:
            fab_data[pk] = {
                'n': n, 'C': np.asfortranarray(pno_spaces[pk]['C_pno']),
                'z': z_total, 't1': t1_proj, 'ovL': ovL_k,
                'result': np.zeros((n, n)),
            }

    # --- C_tilde Term 2 data ---
    c_data = {}
    for k, i in all_ordered:
        key_ki = (min(k, i), max(k, i))
        n = pno_spaces[key_ki]['C_pno'].shape[1]
        if n == 0:
            continue
        t1_i = _project_t1_to_pair(t1_pno, i, key_ki, S_pno_cache, pno_spaces)
        if np.max(np.abs(t1_i)) < 1e-15:
            continue
        ovL_i = ovL_bare.get((key_ki, i))
        if ovL_i is None:
            continue
        c_data[(k, i)] = {
            'key': key_ki, 'n': n,
            'C': np.asfortranarray(pno_spaces[key_ki]['C_pno']),
            't1': t1_i, 'ovL': ovL_i,
            'result': np.zeros((n, n)),
        }

    # --- D_tilde Term 2 data ---
    d_data = {}
    for i_idx, k_idx in all_ordered:
        key_ik = (min(i_idx, k_idx), max(i_idx, k_idx))
        n = pno_spaces[key_ik]['C_pno'].shape[1]
        if n == 0:
            continue
        t1_i = _project_t1_to_pair(t1_pno, i_idx, key_ik, S_pno_cache, pno_spaces)
        if np.max(np.abs(t1_i)) < 1e-15:
            continue
        ovL_k = ovL_bare.get((key_ik, k_idx))
        if ovL_k is None:
            continue
        d_data[(i_idx, k_idx)] = {
            'key': key_ik, 'n': n,
            'C': np.asfortranarray(pno_spaces[key_ik]['C_pno']),
            't1': t1_i, 'ovL': ovL_k,
            'result': np.zeros((n, n)),
        }

    # --- Ladder data ---
    ladder_data = {}
    for pk in pair_keys:
        i, j = pk
        n = pno_spaces[pk]['C_pno'].shape[1]
        if n == 0:
            continue
        T1_all = np.zeros((nocc, n))
        if t1_pno is not None:
            for kk in range(nocc):
                T1_all[kk] = _project_t1_to_pair(
                    t1_pno, kk, pk, S_pno_cache, pno_spaces)
        tau = t2_pno_all[pk] + np.outer(T1_all[i], T1_all[j])
        ovL_stacked = np.stack(
            [ovL_bare.get((pk, k), np.zeros((n, 0))) for k in range(nocc)])
        naux = ovL_stacked.shape[2]
        if naux == 0:
            continue
        # BLAS: corr = T1.T @ ovL.reshape -> reshape
        corr = (T1_all.T @ ovL_stacked.reshape(nocc, -1)).reshape(n, n, naux)
        ladder_data[pk] = {
            'n': n, 'C': np.asfortranarray(pno_spaces[pk]['C_pno']),
            'tau': tau, 'corr': corr,
            'ladder': np.zeros((n, n)),
        }

    # ====================================================================
    # Single DF pass: process all computations per batch
    # ====================================================================
    aux_off = 0
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]

        # Cache AO→MO transforms per unique pair key
        B_L_cache = {}

        def get_B_L(key, n, C):
            if key not in B_L_cache:
                buf = _ao2mo.nr_e2(Lpq, C, (0, n, 0, n), aosym='s2')
                B_L_cache[key] = buf.reshape(nL, n, n)
            return B_L_cache[key]

        # --- Fab/fvv_t1 ---
        for pk, fd in fab_data.items():
            n = fd['n']
            B_L = get_B_L(pk, n, fd['C'])
            z_batch = fd['z'][aux_off:aux_off+nL]
            fd['result'] += 2.0 * np.tensordot(z_batch, B_L, axes=([0], [0]))
            for kk, t1_k in fd['t1'].items():
                ovl = fd['ovL'].get(kk)
                if ovl is None:
                    continue
                Y_k = np.tensordot(B_L, t1_k, axes=([1], [0])).transpose(1, 0)
                fd['result'] -= Y_k @ ovl[:, aux_off:aux_off+nL].T

        # --- C_tilde Term 2 ---
        for (k, i), cd in c_data.items():
            B_L = get_B_L(cd['key'], cd['n'], cd['C'])
            z_i = cd['t1'] @ cd['ovL'][:, aux_off:aux_off+nL]
            cd['result'] += np.tensordot(z_i, B_L, axes=([0], [0]))

        # --- D_tilde Term 2 ---
        for (i_idx, k_idx), dd in d_data.items():
            B_L = get_B_L(dd['key'], dd['n'], dd['C'])
            z_c = np.tensordot(B_L, dd['t1'], axes=([1], [0]))
            y = dd['t1'] @ dd['ovL'][:, aux_off:aux_off+nL]
            dd['result'] += 2.0 * dd['ovL'][:, aux_off:aux_off+nL] @ z_c
            dd['result'] -= np.tensordot(y, B_L, axes=([0], [0]))

        # --- Ladder ---
        for pk, ld in ladder_data.items():
            n = ld['n']
            B_L = get_B_L(pk, n, ld['C'])
            corr_batch = ld['corr'][:, :, aux_off:aux_off+nL].transpose(2, 0, 1)
            B_tilde = B_L - corr_batch
            X_L = B_tilde @ ld['tau']
            ld['ladder'] += (X_L.transpose(1, 0, 2).reshape(n, -1) @
                             B_tilde.transpose(1, 0, 2).reshape(n, -1).T)

        aux_off += nL

    # ====================================================================
    # Package results
    # ====================================================================
    fvv_t1_all = {}
    for pk in pair_keys:
        n = pno_spaces[pk]['C_pno'].shape[1]
        fvv_t1_all[pk] = fab_data[pk]['result'] if pk in fab_data else np.zeros((n, n))

    c_term2_all = {ki: cd['result'] for ki, cd in c_data.items()}
    d_term2_all = {ik: dd['result'] for ik, dd in d_data.items()}

    ladder_all = {}
    for pk in pair_keys:
        n = pno_spaces[pk]['C_pno'].shape[1]
        ladder_all[pk] = ladder_data[pk]['ladder'] if pk in ladder_data else np.zeros((n, n))

    return fvv_t1_all, c_term2_all, d_term2_all, ladder_all
