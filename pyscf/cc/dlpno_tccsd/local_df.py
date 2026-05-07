"""Per-pair local DF integral computation for DLPNO-CCSD.

Matches Psi4's architecture exactly:
  1. build_screening_maps(): precompute atom-localized LMO/PAO/aux sparsity maps
  2. build_sparse_df_arrays(): compute qij_[Q], qia_[Q], qab_[Q] per aux function
     with Boughton-Pulay LMO refit, stored as sparse per-aux-atom neighborhoods.
     Memory: O(naux × atomic_neighborhood²), not O(naux × N_AO²).
  3. compute_cc_integrals(): per-pair extraction from sparse arrays.
  4. t1_ints() / t1_fock(): T1 dressing using stored per-pair intermediates.

All integrals for pair (ij) use pair (ij)'s local aux domain with J_local^{-1/2}.
"""

import os
import numpy as np


def build_screening_maps(mol, auxmol, C_lmo, pao_domains, s1e, strong_pair_keys,
                         T_CUT_MKN=1e-3, T_CUT_CLMO=1e-3, C_pao=None,
                         _pool=None):
    """Build atom-localized sparsity maps matching Psi4 dlpnobase.cc.

    Psi4 uses TWO separate atom-locality criteria for each LMO:
      * lmo_to_atoms[i] : atoms where max |C_lmo[μ, i]| > T_CUT_CLMO.
        Used to define bfs1 (which AOs contribute to (i μ | Q) integrals).
      * lmo_to_riatoms[i] : aux atoms where Mulliken pop > T_CUT_MKN.
        Used to define per-pair local aux domains.

    The LMO-reachability set `riatom_to_lmos_ext[A]` is the inverse of the
    pair-EXTENDED map `lmo_to_riatoms_ext` (= lmo_to_riatoms unioned with the
    riatoms of its strong-pair partners).

    Args:
        mol, auxmol: PySCF Mole objects.
        C_lmo: (nao, nocc) LMO coefficients.
        pao_domains: list[nocc] of np.arrays (AO indices per LMO's PAO domain).
        s1e: (nao, nao) AO overlap.
        strong_pair_keys: list of (i, j) strong pairs (ordered, i<=j).
        T_CUT_MKN, T_CUT_CLMO: screening thresholds (Psi4 defaults 1e-3).

    Returns:
        dict of maps (see module docstring).
    """
    nao = mol.nao_nr()
    natm = mol.natm
    nocc = C_lmo.shape[1]
    ao_labels = mol.ao_labels(fmt=False)
    atom_ids = np.array([lbl[0] for lbl in ao_labels])
    aux_labels = auxmol.ao_labels(fmt=False)
    aux_atom_ids = np.array([lbl[0] for lbl in aux_labels])
    atom_to_bf = [np.where(atom_ids == a)[0] for a in range(natm)]
    atom_to_aux = [np.where(aux_atom_ids == a)[0] for a in range(natm)]

    # --- lmo_to_atoms / lmo_to_paoatoms / lmo_to_riatoms ---
    # All three are per-LMO independent computations. We parallelize over
    # nocc via the shared thread pool when provided. Each worker releases
    # the GIL for its numpy operations, so this scales well.
    def _per_lmo(i):
        # (1) lmo_to_atoms: atoms where C_lmo has significant coefficient.
        has_coeff = np.zeros(natm, dtype=bool)
        for a in range(natm):
            m = atom_ids == a
            if np.any(np.abs(C_lmo[m, i]) > T_CUT_CLMO):
                has_coeff[a] = True
        atoms_i = np.where(has_coeff)[0]

        # (2) lmo_to_paoatoms: atoms hosting each LMO's PAO domain.
        # Include atoms centering the PAO indices AND atoms with
        # significant C_pao amplitude (project_s22_ladder_bug.md).
        if len(pao_domains[i]) == 0:
            paoatoms_i = np.zeros(0, dtype=int)
        else:
            indexed_atoms = set(atom_ids[pao_domains[i]].tolist())
            if C_pao is not None:
                C_slice = C_pao[:, pao_domains[i]]
                per_ao = np.max(np.abs(C_slice), axis=1)
                for A in range(natm):
                    if A in indexed_atoms:
                        continue
                    mask_A = (atom_ids == A)
                    if np.any(per_ao[mask_A] > T_CUT_CLMO):
                        indexed_atoms.add(A)
            paoatoms_i = np.array(sorted(indexed_atoms), dtype=int)

        # (3) lmo_to_riatoms: aux atoms via Mulliken population.
        c = C_lmo[:, i]
        P = s1e * c[:, None] * c[None, :]
        pd = np.diag(P)
        sd = pd[:, None] + pd[None, :]
        with np.errstate(divide='ignore', invalid='ignore'):
            w_u = np.where(sd > 1e-15, pd[:, None] / sd, 0.0)
            w_v = np.where(sd > 1e-15, pd[None, :] / sd, 0.0)
        pop = np.zeros(natm)
        for a in range(natm):
            m = atom_ids == a
            pop[a] = np.sum((P * w_u)[m, :]) + np.sum((P * w_v)[:, m])
        riatoms_i = np.where(np.abs(pop) > T_CUT_MKN)[0]

        return atoms_i, paoatoms_i, riatoms_i

    if _pool is not None and nocc > 1:
        _trips = list(_pool.map(_per_lmo, range(nocc)))
    else:
        _trips = [_per_lmo(i) for i in range(nocc)]
    lmo_to_atoms    = [t[0] for t in _trips]
    lmo_to_paoatoms = [t[1] for t in _trips]
    lmo_to_riatoms  = [t[2] for t in _trips]

    # --- Extend: union with riatoms of strong-pair partners ---
    pair_neighbors = [set() for _ in range(nocc)]
    for key in strong_pair_keys:
        i, j = key
        pair_neighbors[i].add(j)
        pair_neighbors[j].add(i)
    lmo_to_riatoms_ext = []
    for i in range(nocc):
        s = set(lmo_to_riatoms[i].tolist())
        for j in pair_neighbors[i]:
            s.update(lmo_to_riatoms[j].tolist())
        lmo_to_riatoms_ext.append(np.array(sorted(s), dtype=int))

    # --- riatom_to_lmos_ext[A] = inverse (LMOs where A ∈ lmo_to_riatoms_ext[i]) ---
    riatom_to_lmos_ext = [[] for _ in range(natm)]
    for i in range(nocc):
        for a in lmo_to_riatoms_ext[i]:
            riatom_to_lmos_ext[a].append(i)
    riatom_to_lmos_ext = [np.array(lst, dtype=int) for lst in riatom_to_lmos_ext]

    # --- riatom_to_atoms1[A]: AO atoms reachable via lmos_ext (LMO side, bfs1) ---
    riatom_to_atoms1 = []
    for a in range(natm):
        s = set()
        for i in riatom_to_lmos_ext[a]:
            s.update(lmo_to_atoms[i].tolist())
        riatom_to_atoms1.append(np.array(sorted(s), dtype=int))

    # --- riatom_to_atoms2[A]: AO atoms reachable via lmos_ext → pao domain (bfs2) ---
    riatom_to_atoms2 = []
    for a in range(natm):
        s = set()
        for i in riatom_to_lmos_ext[a]:
            s.update(lmo_to_paoatoms[i].tolist())
        riatom_to_atoms2.append(np.array(sorted(s), dtype=int))

    # --- riatom_to_paos_ext[A]: PAO indices reachable via lmos_ext ---
    riatom_to_paos_ext = []
    for a in range(natm):
        s = set()
        for i in riatom_to_lmos_ext[a]:
            s.update(pao_domains[i].tolist())
        riatom_to_paos_ext.append(np.array(sorted(s), dtype=int))

    # --- bfs1[A], bfs2[A]: AO indices on atoms1/atoms2 ---
    def _bfs(atoms_list):
        out = []
        for a_list in atoms_list:
            if len(a_list) == 0:
                out.append(np.zeros(0, dtype=int))
            else:
                out.append(np.sort(np.concatenate(
                    [atom_to_bf[aa] for aa in a_list])))
        return out
    riatom_to_bfs1 = _bfs(riatom_to_atoms1)
    riatom_to_bfs2 = _bfs(riatom_to_atoms2)

    # --- Dense inverse maps for O(1) lookup in hot loops ---
    riatom_to_lmos_ext_dense = np.full((natm, nocc), -1, dtype=np.int64)
    for a in range(natm):
        for idx, i in enumerate(riatom_to_lmos_ext[a]):
            riatom_to_lmos_ext_dense[a, i] = idx
    riatom_to_paos_ext_dense = np.full((natm, nao), -1, dtype=np.int64)
    for a in range(natm):
        for idx, u in enumerate(riatom_to_paos_ext[a]):
            riatom_to_paos_ext_dense[a, u] = idx
    riatom_to_bfs1_dense = np.full((natm, nao), -1, dtype=np.int64)
    for a in range(natm):
        for idx, mu in enumerate(riatom_to_bfs1[a]):
            riatom_to_bfs1_dense[a, mu] = idx
    riatom_to_bfs2_dense = np.full((natm, nao), -1, dtype=np.int64)
    for a in range(natm):
        for idx, mu in enumerate(riatom_to_bfs2[a]):
            riatom_to_bfs2_dense[a, mu] = idx

    # --- aux-atom → aux-shell lookup ---
    aux_shell_to_atom = np.array(
        [auxmol.bas_atom(sh) for sh in range(auxmol.nbas)], dtype=int)
    atom_to_aux_sh = [np.where(aux_shell_to_atom == a)[0] for a in range(natm)]
    aux_shell_loc = auxmol.ao_loc_nr()  # (nbas+1,) shell-to-aux-func offsets

    # --- AO shell boundaries (for shls_slice calls) ---
    ao_shell_to_atom = np.array(
        [mol.bas_atom(sh) for sh in range(mol.nbas)], dtype=int)
    atom_to_ao_sh = [np.where(ao_shell_to_atom == a)[0] for a in range(natm)]
    ao_shell_loc = mol.ao_loc_nr()

    # --- per-aux-atom Boughton-Pulay refit C_lmo ---
    SC_lmo = s1e @ C_lmo   # (nao, nocc)
    def _c_refit_one_atom(a):
        bfs1 = riatom_to_bfs1[a]
        lmos = riatom_to_lmos_ext[a]
        if len(bfs1) == 0 or len(lmos) == 0:
            return np.zeros((len(bfs1), len(lmos)))
        S_aa = s1e[np.ix_(bfs1, bfs1)]
        rhs = SC_lmo[bfs1][:, lmos]
        return np.linalg.solve(S_aa, rhs)
    if _pool is not None and natm > 1:
        c_refit = list(_pool.map(_c_refit_one_atom, range(natm)))
    else:
        c_refit = [_c_refit_one_atom(a) for a in range(natm)]

    return {
        'natm': natm, 'nao': nao, 'nocc': nocc,
        'atom_ids': atom_ids, 'aux_atom_ids': aux_atom_ids,
        'atom_to_bf': atom_to_bf, 'atom_to_aux': atom_to_aux,
        'atom_to_aux_sh': atom_to_aux_sh, 'atom_to_ao_sh': atom_to_ao_sh,
        'aux_shell_to_atom': aux_shell_to_atom,
        'aux_shell_loc': aux_shell_loc, 'ao_shell_loc': ao_shell_loc,
        'lmo_to_atoms': lmo_to_atoms, 'lmo_to_paoatoms': lmo_to_paoatoms,
        'lmo_to_riatoms': lmo_to_riatoms,
        'lmo_to_riatoms_ext': lmo_to_riatoms_ext,
        'riatom_to_lmos_ext': riatom_to_lmos_ext,
        'riatom_to_atoms1': riatom_to_atoms1,
        'riatom_to_atoms2': riatom_to_atoms2,
        'riatom_to_bfs1': riatom_to_bfs1,
        'riatom_to_bfs2': riatom_to_bfs2,
        'riatom_to_paos_ext': riatom_to_paos_ext,
        'riatom_to_lmos_ext_dense': riatom_to_lmos_ext_dense,
        'riatom_to_paos_ext_dense': riatom_to_paos_ext_dense,
        'riatom_to_bfs1_dense': riatom_to_bfs1_dense,
        'riatom_to_bfs2_dense': riatom_to_bfs2_dense,
        'c_refit': c_refit,
    }


def build_sparse_df_arrays(mol, auxmol, C_lmo, C_pao, maps, _pool=None):
    """Build sparse per-aux DF integrals qij_[Q], qia_[Q], qab_[Q].

    For each aux function Q on atom A_Q, stores:
        qij_[Q] of shape (|lmos_ext[A_Q]|, |lmos_ext[A_Q]|)  = (i j | Q)
        qia_[Q] of shape (|lmos_ext[A_Q]|, |paos_ext[A_Q]|)  = (i a | Q)
        qab_[Q] of shape (|paos_ext[A_Q]|, |paos_ext[A_Q]|)  = (a b | Q)

    Integrals are built shell-by-shell with atom-level screening:
        (μν | Q) computed only for μ ∈ bfs1[A_Q], ν ∈ bfs2[A_Q],
        then transformed via BP-refitted C_lmo (LMO side) and sliced C_pao (PAO side).

    Memory: Σ_Q |lmos_ext|² + |lmos_ext|×|paos_ext| + |paos_ext|². For large
    systems this is O(naux × const²) — linear in naux, not O(naux × nao²).

    Returns dict with 'qij', 'qia', 'qab': each a list of length naux.
    """
    naux = auxmol.nao_nr()
    pmol = mol + auxmol
    aux_shell_to_atom = maps['aux_shell_to_atom']
    aux_shell_loc = maps['aux_shell_loc']
    atom_to_ao_sh = maps['atom_to_ao_sh']
    ao_shell_loc = maps['ao_shell_loc']
    riatom_to_atoms1 = maps['riatom_to_atoms1']
    riatom_to_atoms2 = maps['riatom_to_atoms2']
    riatom_to_bfs1 = maps['riatom_to_bfs1']
    riatom_to_bfs2 = maps['riatom_to_bfs2']
    riatom_to_paos_ext = maps['riatom_to_paos_ext']
    c_refit = maps['c_refit']

    qij = [None] * naux
    qia = [None] * naux
    qab = [None] * naux

    def _process_shell(Q_sh):
        centerQ = aux_shell_to_atom[Q_sh]
        nq = aux_shell_loc[Q_sh + 1] - aux_shell_loc[Q_sh]
        q_start = aux_shell_loc[Q_sh]

        bfs1 = riatom_to_bfs1[centerQ]
        bfs2 = riatom_to_bfs2[centerQ]
        lmos_ext = maps['riatom_to_lmos_ext'][centerQ]
        paos_ext = riatom_to_paos_ext[centerQ]
        C_r = c_refit[centerQ]  # (|bfs1|, |lmos_ext|)

        # AO shell ranges for bfs1/bfs2: for AO-shell-level intor we need
        # to include shells whose AO indices touch bfs1/bfs2. Easier:
        # compute (μν | q) for μ ∈ union of bfs1's atoms, then slice.
        atoms1 = riatom_to_atoms1[centerQ]
        atoms2 = riatom_to_atoms2[centerQ]
        if len(bfs1) == 0 or len(bfs2) == 0 or len(lmos_ext) == 0:
            # Empty neighborhood — no contribution from this Q
            return [(q_start + q,
                     np.zeros((len(lmos_ext), len(lmos_ext))),
                     np.zeros((len(lmos_ext), len(paos_ext))),
                     np.zeros((len(paos_ext), len(paos_ext))))
                    for q in range(nq)]

        # Build shell slices for atoms1, atoms2
        sh1_list = np.concatenate([atom_to_ao_sh[a] for a in atoms1])
        sh2_list = np.concatenate([atom_to_ao_sh[a] for a in atoms2])
        sh1_min, sh1_max = sh1_list.min(), sh1_list.max() + 1
        sh2_min, sh2_max = sh2_list.min(), sh2_list.max() + 1

        buf = pmol.intor(
            'int3c2e',
            shls_slice=(sh1_min, sh1_max, sh2_min, sh2_max,
                        mol.nbas + Q_sh, mol.nbas + Q_sh + 1))

        bf1_start = ao_shell_loc[sh1_min]
        bf2_start = ao_shell_loc[sh2_min]
        rel_bfs1 = bfs1 - bf1_start
        rel_bfs2 = bfs2 - bf2_start
        mn_block = buf[rel_bfs1][:, rel_bfs2, :]

        if atoms1.tolist() == atoms2.tolist():
            mn1_block = mn_block
        else:
            buf1 = pmol.intor(
                'int3c2e',
                shls_slice=(sh1_min, sh1_max, sh1_min, sh1_max,
                            mol.nbas + Q_sh, mol.nbas + Q_sh + 1))
            mn1_block = buf1[rel_bfs1][:, rel_bfs1, :]

        if atoms1.tolist() == atoms2.tolist():
            mn2_block = mn_block
        else:
            buf2 = pmol.intor(
                'int3c2e',
                shls_slice=(sh2_min, sh2_max, sh2_min, sh2_max,
                            mol.nbas + Q_sh, mol.nbas + Q_sh + 1))
            mn2_block = buf2[rel_bfs2][:, rel_bfs2, :]

        C_pao_slice = C_pao[np.ix_(bfs2, paos_ext)]
        results = []
        for qi in range(nq):
            Q = q_start + qi
            results.append((
                Q,
                C_r.T @ mn1_block[:, :, qi] @ C_r,
                C_r.T @ mn_block[:, :, qi] @ C_pao_slice,
                C_pao_slice.T @ mn2_block[:, :, qi] @ C_pao_slice,
            ))
        return results

    Q_shells = list(range(auxmol.nbas))
    if _pool is not None:
        all_results = list(_pool.map(_process_shell, Q_shells))
    else:
        all_results = [_process_shell(Q_sh) for Q_sh in Q_shells]
    for shell_results in all_results:
        for Q, qij_Q, qia_Q, qab_Q in shell_results:
            qij[Q] = qij_Q
            qia[Q] = qia_Q
            qab[Q] = qab_Q

    return {'qij': qij, 'qia': qia, 'qab': qab}


_S_PNO_BATCHED_LIBCC = None


def compute_S_pno_batched(key_a, partner_keys, pno_spaces,
                          S_pao_full, s1e, _S_pao_full_c=None):
    """Batched S_pno: one C kernel call for all (key_a, partner_keys[p]).

    Returns a dict {partner_key: (n_pno_a, n_pno_b) ndarray}. Handles the
    fallback (any partner without X_pno) by falling through to the slow
    per-pair path for those entries only.
    """
    pd_a = pno_spaces[key_a]
    X_a_full = pd_a.get('X_pno')
    pp_a = pd_a.get('pair_paos')
    if X_a_full is None or pp_a is None:
        return {kb: compute_S_pno(key_a, kb, pno_spaces, S_pao_full, s1e)
                for kb in partner_keys}

    good = []
    fallback = []
    for kb in partner_keys:
        pd_b = pno_spaces[kb]
        if (pd_b.get('X_pno') is not None
                and pd_b.get('pair_paos') is not None):
            good.append(kb)
        else:
            fallback.append(kb)

    out = {kb: compute_S_pno(key_a, kb, pno_spaces, S_pao_full, s1e)
           for kb in fallback}
    if not good:
        return out

    global _S_PNO_BATCHED_LIBCC
    if _S_PNO_BATCHED_LIBCC is None:
        import ctypes as _ct
        from pyscf import lib as _pyscflib
        _lib = _pyscflib.load_library('libcc')
        _lib.DLPNObuild_S_pno_for_pair.restype = None
        _lib.DLPNObuild_S_pno_for_pair.argtypes = (
            [_ct.c_void_p, _ct.c_void_p,
             _ct.c_int, _ct.c_int, _ct.c_int]
            + [_ct.c_void_p] * 8
            + [_ct.c_void_p, _ct.c_size_t])
        _S_PNO_BATCHED_LIBCC = _lib
    import ctypes as _ct
    if _S_pao_full_c is None:
        _S_pao_full_c = np.ascontiguousarray(S_pao_full)
    _n_pao_total = _S_pao_full_c.shape[0]

    n_pao_a = int(np.asarray(pp_a).size)
    n_pno_a = int(X_a_full.shape[1])
    n_partners = len(good)
    partner_n_pao = np.empty(n_partners, dtype=np.int32)
    partner_n_pno = np.empty(n_partners, dtype=np.int32)
    for p, kb in enumerate(good):
        pd_b = pno_spaces[kb]
        partner_n_pao[p] = int(np.asarray(pd_b['pair_paos']).size)
        partner_n_pno[p] = int(pd_b['X_pno'].shape[1])
    pp_off = np.empty(n_partners + 1, dtype=np.int64)
    pp_off[0] = 0
    pp_off[1:] = np.cumsum(partner_n_pao.astype(np.int64))
    X_sizes = (partner_n_pao.astype(np.int64)
               * partner_n_pno.astype(np.int64))
    X_off = np.empty(n_partners + 1, dtype=np.int64)
    X_off[0] = 0
    X_off[1:] = np.cumsum(X_sizes)
    S_sizes = (partner_n_pno.astype(np.int64) * n_pno_a)
    S_off = np.empty(n_partners + 1, dtype=np.int64)
    S_off[0] = 0
    S_off[1:] = np.cumsum(S_sizes)

    pp_flat = np.empty(int(pp_off[-1]), dtype=np.int64)
    X_flat = np.empty(int(X_off[-1]))
    for p, kb in enumerate(good):
        pd_b = pno_spaces[kb]
        pp_flat[pp_off[p]:pp_off[p + 1]] = np.asarray(
            pd_b['pair_paos'], dtype=np.int64)
        X_flat[X_off[p]:X_off[p + 1]] = np.ascontiguousarray(
            pd_b['X_pno']).ravel()
    S_flat = np.empty(int(S_off[-1]))
    pp_a_arr = np.ascontiguousarray(pp_a, dtype=np.int64)
    X_a_arr = np.ascontiguousarray(X_a_full)
    _S_PNO_BATCHED_LIBCC.DLPNObuild_S_pno_for_pair(
        pp_a_arr.ctypes.data_as(_ct.c_void_p),
        X_a_arr.ctypes.data_as(_ct.c_void_p),
        int(n_pao_a), int(n_pno_a), int(n_partners),
        partner_n_pao.ctypes.data_as(_ct.c_void_p),
        partner_n_pno.ctypes.data_as(_ct.c_void_p),
        pp_off.ctypes.data_as(_ct.c_void_p),
        pp_flat.ctypes.data_as(_ct.c_void_p),
        X_off.ctypes.data_as(_ct.c_void_p),
        X_flat.ctypes.data_as(_ct.c_void_p),
        S_off.ctypes.data_as(_ct.c_void_p),
        S_flat.ctypes.data_as(_ct.c_void_p),
        _S_pao_full_c.ctypes.data_as(_ct.c_void_p),
        _n_pao_total,
    )
    for p, kb in enumerate(good):
        n_pno_b = int(partner_n_pno[p])
        out[kb] = (S_flat[S_off[p]:S_off[p + 1]]
                   .reshape(n_pno_a, n_pno_b).copy())
    return out


def compute_S_pno(key_a, key_b, pno_spaces, S_pao_full, s1e):
    """Compute the (n_pno_a, n_pno_b) PNO overlap matrix S_ab.

    Uses the Psi4-style PAO-domain path when both pairs have X_pno /
    pair_paos set:  S = X_a^T @ S_pao[pao_a, pao_b] @ X_b.
    Falls back to C_pno + s1e (AO basis) only when one side has no
    X_pno — needed for CAS pairs whose basis spills past the pair PAO
    domain via C_cas_vir.

    This is the *single authoritative* formula for all PNO overlaps so
    the upfront S_pno_cache build and any on-demand recomputation
    (cache miss) agree to machine precision.

    Args:
        key_a, key_b: pair keys (i, j) tuples.
        pno_spaces: dict from make_pnos; each entry carries
            'C_pno', optionally 'X_pno' and 'pair_paos'.
        S_pao_full: (npao, npao) PAO overlap C_pao^T @ s1e @ C_pao.
        s1e: (nao, nao) AO overlap.

    Returns:
        (n_pno_a, n_pno_b) overlap matrix.
    """
    pd_a = pno_spaces[key_a]
    pd_b = pno_spaces[key_b]
    X_a = pd_a.get('X_pno')
    X_b = pd_b.get('X_pno')
    pp_a = pd_a.get('pair_paos')
    pp_b = pd_b.get('pair_paos')
    if X_a is None or X_b is None or pp_a is None or pp_b is None:
        return pd_a['C_pno'].T @ (s1e @ pd_b['C_pno'])
    return X_a.T @ S_pao_full[np.ix_(pp_a, pp_b)] @ X_b


def get_local_ovL(cc_ints, pair_key, lmo_idx):
    """Get locally-fitted ovL for LMO lmo_idx in pair pair_key's PNO basis.

    Returns (npno, n_local) or None if not available.
    """
    ci = cc_ints.get(pair_key)
    if ci is None:
        return None
    # Qma reduced to (n_local, nlmo_p, npno); translate global lmo_idx.
    lmo_in_p = int(ci['p_lmos_dense'][lmo_idx])
    if lmo_in_p < 0:
        if int(os.environ.get('DLPNO_DEBUG_OOD', '0')):
            _c = getattr(get_local_ovL, '_ood', {})
            _c[(pair_key, lmo_idx)] = _c.get((pair_key, lmo_idx), 0) + 1
            get_local_ovL._ood = _c
        return None
    return ci['Qma'][:, lmo_in_p, :].T  # (npno, n_local)


def get_local_ooL_vec(cc_ints, k, l, pair_key):
    """Get locally-fitted ooL[k,l] in pair pair_key's local fitting.

    Returns (n_local,) or None if not available.
    """
    ci = cc_ints.get(pair_key)
    if ci is None:
        return None
    i_lmo, j_lmo = pair_key
    # i_Qk/j_Qk reduced to (n_local, nlmo_p); translate global k -> p_dense.
    k_red = int(ci['p_lmos_dense'][k])
    if k_red < 0:
        if int(os.environ.get('DLPNO_DEBUG_OOD', '0')):
            _c = getattr(get_local_ooL_vec, '_ood', {})
            _c[(pair_key, k, l)] = _c.get((pair_key, k, l), 0) + 1
            get_local_ooL_vec._ood = _c
        return None
    if l == i_lmo:
        return ci['i_Qk'][:, k_red]
    elif l == j_lmo:
        return ci['j_Qk'][:, k_red]
    else:
        return None


def get_local_K(cc_ints, pair_key, lmo1, lmo2):
    """Get locally-fitted K = ovL_lmo1 @ ovL_lmo2.T for pair pair_key.

    Returns (npno, npno) or None. Memoized per (pair_key, lmo1, lmo2);
    safe because cc_ints['Qma'] is built once and frozen during CCSD.
    """
    ci = cc_ints.get(pair_key)
    if ci is None:
        return None
    cache = ci.get('_K_cache')
    if cache is None:
        cache = {}
        ci['_K_cache'] = cache
    k = (lmo1, lmo2)
    K = cache.get(k)
    if K is None:
        Qma = ci['Qma']  # (n_local, nlmo_p, npno) reduced
        p_dense = ci['p_lmos_dense']
        l1, l2 = int(p_dense[lmo1]), int(p_dense[lmo2])
        if l1 < 0 or l2 < 0:
            if int(os.environ.get('DLPNO_DEBUG_OOD', '0')):
                _c = getattr(get_local_K, '_ood', {})
                _c[(pair_key, lmo1, lmo2)] = _c.get((pair_key, lmo1, lmo2), 0) + 1
                get_local_K._ood = _c
            return None
        # ovL_lmo[Q, a] = Qma[Q, lmo, a]; K[a,b] = Σ_Q ovL1[Q,a]*ovL2[Q,b]
        K = Qma[:, l1, :].T @ Qma[:, l2, :]
        cache[k] = K
    return K


def pno_in_AO(pno_spaces, key, C_pao):
    """Reconstruct C_pno in AO basis from Psi4-style PAO-domain storage.

    Equivalent to ``pno_spaces[key]['C_pno']`` but rebuilt on demand from
    ``X_pno`` (|domain_ij|, npno) and ``pair_paos`` (|domain_ij|,)::

        C_pno = C_pao[:, pair_paos] @ X_pno

    For CAS pairs (where the PNO basis extends beyond the pair PAO domain
    via ``C_cas_vir``), ``X_pno`` is ``None`` and we fall back to the
    cached AO-basis ``C_pno``.

    Args:
        pno_spaces: dict from make_pnos.
        key: pair key (i, j).
        C_pao: (nao, npao) PAO coefficients.

    Returns:
        np.ndarray (nao, npno) PNO coefficients in AO basis.
    """
    pd = pno_spaces[key]
    X_pno = pd.get('X_pno')
    if X_pno is None:
        return pd['C_pno']
    pair_paos = pd['pair_paos']
    return C_pao[:, pair_paos] @ X_pno



def compute_cc_integrals_sparse(mol, auxmol, C_lmo, C_pao, pno_spaces,
                                pair_aux_idx, j2c, keys, nocc, s1e=None,
                                pao_domains=None, strong_pair_keys=None,
                                T_CUT_MKN=1e-3, T_CUT_CLMO=1e-3,
                                screening_maps=None, sparse_arrays=None,
                                pair_lmo_idx=None,
                                _pool=None):
    """Per-pair fitted intermediates via sparse per-aux-Q sparse storage.

    Drop-in replacement for ``compute_cc_integrals`` that uses Psi4-style
    PAO-domain X_pno transforms. Same output schema. Shares the screening
    maps and sparse arrays from ``build_screening_maps`` /
    ``build_sparse_df_arrays`` (build them here if not passed in).

    With T_CUT_MKN, T_CUT_CLMO → 0, sparse → dense and the per-pair tensors
    must agree with ``compute_cc_integrals`` to BLAS noise (~1e-10).

    Args:
        mol, auxmol: PySCF Mole objects.
        C_lmo: (nao, nocc) LMO coefficients (full PySCF basis).
        C_pao: (nao, nao) PAO coefficients.
        pno_spaces: dict from ``make_pnos`` — must carry ``X_pno`` and
            ``pair_paos`` (Phase 1 storage).
        pair_aux_idx: dict pair_key -> np.array of local aux-function idx.
        j2c: (naux, naux) Coulomb metric.
        keys: list of pair keys.
        nocc: number of correlated occupieds.
        s1e: (nao, nao) AO overlap (computed from mol if None).
        pao_domains: list[nocc] of np.arrays of PAO AO indices. If None,
            reconstructed from ``pno_spaces[(i,i)]['domain_ij']``.
        strong_pair_keys: pair list driving lmo_to_riatoms_ext extension. If
            None, uses ``keys``.
        T_CUT_MKN, T_CUT_CLMO: sparsity thresholds. Set very small to recover
            dense behavior for validation.
        screening_maps, sparse_arrays: optional pre-built outputs to avoid
            recomputing across multiple pair sweeps.

    Returns:
        cc_ints (dict): same schema as ``compute_cc_integrals``.
    """
    if s1e is None:
        s1e = mol.intor('int1e_ovlp')
    if pao_domains is None:
        pao_domains = []
        for i in range(nocc):
            key_ii = (i, i)
            if key_ii in pno_spaces \
                    and pno_spaces[key_ii].get('pair_paos') is not None:
                pao_domains.append(np.asarray(pno_spaces[key_ii]['pair_paos']))
            else:
                pao_domains.append(np.zeros(0, dtype=int))
    if strong_pair_keys is None:
        strong_pair_keys = list(keys)

    import time as _ccints_setup_time
    _t_setup0 = _ccints_setup_time.perf_counter()
    if screening_maps is None:
        screening_maps = build_screening_maps(
            mol, auxmol, C_lmo, pao_domains, s1e, strong_pair_keys,
            T_CUT_MKN=T_CUT_MKN, T_CUT_CLMO=T_CUT_CLMO, C_pao=C_pao)
    _t_screening = _ccints_setup_time.perf_counter() - _t_setup0
    _t_setup0 = _ccints_setup_time.perf_counter()
    if sparse_arrays is None:
        sparse_arrays = build_sparse_df_arrays(
            mol, auxmol, C_lmo, C_pao, screening_maps, _pool=_pool)
    _t_sparse = _ccints_setup_time.perf_counter() - _t_setup0

    qij = sparse_arrays['qij']
    qia = sparse_arrays['qia']
    qab = sparse_arrays['qab']

    aux_atom_ids = screening_maps['aux_atom_ids']
    riatom_to_lmos_ext = screening_maps['riatom_to_lmos_ext']
    riatom_to_paos_ext = screening_maps['riatom_to_paos_ext']
    riatom_to_lmos_ext_dense = screening_maps['riatom_to_lmos_ext_dense']
    riatom_to_paos_ext_dense = screening_maps['riatom_to_paos_ext_dense']
    natm = screening_maps['natm']

    # Per-atom stacked sparse arrays — once globally so per-pair loops can
    # batch over all aux Q's centered on the same atom (BLAS-3 amortizes
    # across the n_aux_at_A Q's of each atom). All Q's of the same atom
    # share the same lmos_ext/paos_ext neighborhood, so stacking is well
    # defined.
    naux = auxmol.nao_nr()
    _t_setup0 = _ccints_setup_time.perf_counter()
    aux_at_atom = [np.where(aux_atom_ids == A)[0] for A in range(natm)]
    qij_atom = [None] * natm
    qia_atom = [None] * natm
    qab_atom = [None] * natm
    # Per-atom np.stack of qij/qia/qab columns is independent across atoms.
    # Match the lccsd_t.py path that pool-parallelises the same shape; on
    # water-22 the serial loop was ~0.5 s and scales O(N²).
    def _stack_atom(A):
        Qs = aux_at_atom[A]
        if len(Qs) == 0:
            return None, None, None
        nl = len(riatom_to_lmos_ext[A])
        np_ = len(riatom_to_paos_ext[A])
        if nl == 0 or np_ == 0:
            return None, None, None
        return (np.stack([qij[Q] for Q in Qs]),
                np.stack([qia[Q] for Q in Qs]),
                np.stack([qab[Q] for Q in Qs]))
    if _pool is not None and natm > 1:
        _stacks = list(_pool.map(_stack_atom, range(natm)))
    else:
        _stacks = [_stack_atom(A) for A in range(natm)]
    for A, (qj, qa, qb) in enumerate(_stacks):
        qij_atom[A] = qj
        qia_atom[A] = qa
        qab_atom[A] = qb
    # Map global Q → position within its atom's Q-stack
    aux_pos_in_atom = -np.ones(naux, dtype=np.int64)
    for A in range(natm):
        for pos, Q in enumerate(aux_at_atom[A]):
            aux_pos_in_atom[Q] = pos
    _t_atom_stacks = _ccints_setup_time.perf_counter() - _t_setup0

    cc_ints = {}
    key_set = set(keys)

    # Phase II: pyscf/lib/cc/dlpno_partner.c — replaces _cc_ints_partner_cy.pyx.
    import ctypes as _ctypes_pa
    from pyscf import lib as _pyscflib_pa
    _libcc_pa = _pyscflib_pa.load_library('libcc')
    _libcc_pa.DLPNOpartner_apply.restype = None
    _libcc_pa.DLPNOpartner_apply.argtypes = [
        _ctypes_pa.c_void_p, _ctypes_pa.c_void_p,           # proj_ij, qia_b
        _ctypes_pa.c_void_p, _ctypes_pa.c_void_p,           # local_Q, idx
        _ctypes_pa.c_void_p,                                # X
        _ctypes_pa.c_void_p, _ctypes_pa.c_void_p,           # raw_cross_out, raw_kv_out
        _ctypes_pa.c_int, _ctypes_pa.c_int,                 # k_s, do_proj
        _ctypes_pa.c_size_t, _ctypes_pa.c_size_t,           # nQp, npno
        _ctypes_pa.c_size_t, _ctypes_pa.c_size_t,           # np_full, nl
        _ctypes_pa.c_size_t, _ctypes_pa.c_size_t,           # n_kj, npp
        _ctypes_pa.c_size_t,                                # n_local_total
    ]

    def partner_apply(proj_ij, qia_b, k_s,
                      local_Q, idx, X,
                      raw_cross_out, raw_kv_out, do_proj):
        nQp = local_Q.shape[0]
        npno = proj_ij.shape[1] if proj_ij is not None else 0
        np_full = qia_b.shape[2]
        nl = qia_b.shape[1]
        n_kj = X.shape[1]
        npp = idx.shape[0]
        n_local_total = raw_cross_out.shape[0]
        _libcc_pa.DLPNOpartner_apply(
            proj_ij.ctypes.data_as(_ctypes_pa.c_void_p),
            qia_b.ctypes.data_as(_ctypes_pa.c_void_p),
            local_Q.ctypes.data_as(_ctypes_pa.c_void_p),
            idx.ctypes.data_as(_ctypes_pa.c_void_p),
            X.ctypes.data_as(_ctypes_pa.c_void_p),
            raw_cross_out.ctypes.data_as(_ctypes_pa.c_void_p),
            raw_kv_out.ctypes.data_as(_ctypes_pa.c_void_p),
            int(k_s), int(do_proj),
            nQp, npno, np_full, nl, n_kj, npp, n_local_total,
        )

    # === Native-C centerQ kernel (DLPNO_C_CYCLE=1): replaces the Python
    # body of the inner `for centerQ in unique_centers:` loop with a
    # single C call per centerQ. See pyscf/lib/cc/dlpno_pair_centerQ.c.
    _use_centerQ_c = bool(int(os.environ.get('DLPNO_C_CYCLE', '0')))
    _libcc_centerQ = None
    if _use_centerQ_c:
        import ctypes as _ctypes_cQ
        from pyscf import lib as _pyscflib_cQ
        _libcc_centerQ = _pyscflib_cQ.load_library('libcc')
        _libcc_centerQ.DLPNOpair_centerQ_step.restype = None
        _libcc_centerQ.DLPNOpair_centerQ_step.argtypes = (
            [_ctypes_cQ.c_void_p] * 3                     # qij/qia/qab atom-full
            + [_ctypes_cQ.c_void_p, _ctypes_cQ.c_void_p]   # local_Q, atom_pos
            + [_ctypes_cQ.c_int, _ctypes_cQ.c_int]         # i_s, j_s
            + [_ctypes_cQ.c_void_p] * 5                    # ij_u_in_Q, ext_kept_pos/lmos, X_ij, pair_used_in_Q
            + [_ctypes_cQ.c_size_t] * 9                    # shapes (incl n_red)
            + [_ctypes_cQ.c_void_p] * 8)                   # outputs
        _libcc_centerQ.DLPNOcross_partner_assemble.restype = None
        _libcc_centerQ.DLPNOcross_partner_assemble.argtypes = (
            [_ctypes_cQ.c_int]
            + [_ctypes_cQ.c_void_p] * 13
            + [_ctypes_cQ.c_size_t] * 3)
        _libcc_centerQ.DLPNOpartners_centerQ_step.restype = None
        _libcc_centerQ.DLPNOpartners_centerQ_step.argtypes = (
            [_ctypes_cQ.c_void_p] * 7            # proj, qia_atom, atom_pos, local_Q, paos_dense, lmos_dense, pair_used_inv
            + [_ctypes_cQ.c_int]                  # n_partners
            + [_ctypes_cQ.c_void_p] * 8           # 8 flat partner arrays
            + [_ctypes_cQ.c_size_t] * 8           # nQp..nocc, n_red
            + [_ctypes_cQ.c_void_p] * 2)          # raw_cross_flat, raw_kv_flat

    # === DBG_CCINTS section timers (read DLPNO_CCINTS_DBG=1) ===
    import time as _ccints_time
    import threading as _ccints_threading
    _dbg_ccints = bool(int(os.environ.get('DLPNO_CCINTS_DBG', '0')))
    _dbg_acc = {'setup': 0.0, 'centerQ_loop': 0.0, 'jhi_eigh': 0.0,
                'jhi_apply': 0.0, 'final_KJ': 0.0, 'cross_kj': 0.0,
                'partner_calls': 0.0, 'centerQ_inner': 0.0,
                'partner_enum': 0.0, 'pair_alloc': 0.0,
                'partner_flat': 0.0, 'flat_scatter': 0.0,
                'returnpack': 0.0, 'pair_total': 0.0,
                'cQ_pair_kernel': 0.0, 'cQ_partner_kernel': 0.0}
    _dbg_lock = _ccints_threading.Lock()
    def _dbg_add(k, v):
        if _dbg_ccints:
            with _dbg_lock:
                _dbg_acc[k] += v

    # === Per-pair scaling stats (read DLPNO_CCINTS_STATS=1) ===
    _stats_ccints = bool(int(os.environ.get('DLPNO_CCINTS_STATS', '0')))
    _stats_lock = _ccints_threading.Lock()
    _stats_pairs = []   # list of dicts: {npno, n_local, nlmo_p,
                        #                 ncenters, n_partners, work}

    def _process_pair(key):
        """Build cc_ints[key] entry. Pure function — safe for thread parallel."""
        _t_pair_start = _ccints_time.perf_counter() if _dbg_ccints else 0.0
        try:
            return _process_pair_inner(key)
        finally:
            if _dbg_ccints:
                _dbg_add('pair_total',
                         _ccints_time.perf_counter() - _t_pair_start)

    def _process_pair_inner(key):
        if key not in pair_aux_idx:
            return key, None
        i, j = key
        pd = pno_spaces[key]
        X_pno_ij = pd.get('X_pno')
        pair_paos_ij = pd.get('pair_paos')
        if X_pno_ij is None or pair_paos_ij is None:
            return key, None
        npno = X_pno_ij.shape[1]
        if npno == 0:
            return key, None

        aux_idx = np.asarray(pair_aux_idx[key])
        n_local = len(aux_idx)
        if n_local == 0:
            return key, None
        _t_pe_start = _ccints_time.perf_counter() if _dbg_ccints else 0.0
        if pair_lmo_idx is not None and key in pair_lmo_idx:
            _k_iter = pair_lmo_idx[key]
        else:
            _k_iter = range(nocc)
        kj_partners = []
        ki_partners = []
        for k in _k_iter:
            k = int(k)
            key_kj = (min(k, j), max(k, j))
            if key_kj in key_set and key_kj in pno_spaces:
                Xp = pno_spaces[key_kj].get('X_pno')
                if Xp is not None and Xp.shape[1] > 0:
                    kj_partners.append((k, key_kj, Xp.shape[1]))
            key_ki = (min(k, i), max(k, i))
            if key_ki in key_set and key_ki in pno_spaces:
                Xp = pno_spaces[key_ki].get('X_pno')
                if Xp is not None and Xp.shape[1] > 0:
                    ki_partners.append((k, key_ki, Xp.shape[1]))
        if _dbg_ccints:
            _dbg_add('partner_enum',
                     _ccints_time.perf_counter() - _t_pe_start)

        # Pre-fit accumulators.  raw_io/raw_jo/raw_ma's LMO axis is set
        # to ``nocc`` below once we know which LMOs actually get
        # populated from the centerQ stacks (the union of
        # riatom_to_lmos_ext over all of this pair's aux centers).
        _t_alloc_start = _ccints_time.perf_counter() if _dbg_ccints else 0.0
        raw_iv = np.zeros((n_local, npno))
        raw_jv = np.zeros((n_local, npno))
        raw_ab = np.zeros((n_local, npno, npno))
        raw_pair = np.zeros(n_local)
        # When _use_centerQ_c, the per-partner raw_cross / raw_kv arrays
        # live in flat buffers (built once per pair below) so the C
        # partners_centerQ kernel writes into them directly across centerQ
        # iterations and the cross_partner kernel reads them without a
        # dict→flat conversion. The Python fallback keeps dicts.
        if _use_centerQ_c:
            raw_cross_kj = raw_kv_kj = raw_cross_ji = raw_kv_ki = None
        else:
            raw_cross_kj = {k: np.zeros((n_local, npno, n_kj))
                            for k, _, n_kj in kj_partners}
            raw_kv_kj = {k: np.zeros((n_local, n_kj))
                         for k, _, n_kj in kj_partners}
            raw_cross_ji = {k: np.zeros((n_local, npno, n_ki))
                            for k, _, n_ki in ki_partners}
            raw_kv_ki = {k: np.zeros((n_local, n_ki))
                         for k, _, n_ki in ki_partners}
        if _dbg_ccints:
            _dbg_add('pair_alloc',
                     _ccints_time.perf_counter() - _t_alloc_start)

        # Session: per-pair flat partner buffers for the C centerQ
        # partners kernel.  Built once per pair (iteration-invariant since
        # X_pno_kj / pair_paos_kj are static); the per-(centerQ, side)
        # kernel call writes into raw_cross_kj_flat / raw_kv_kj_flat at
        # absolute offsets.  After the centerQ loop ends, the contents are
        # copied back into the per-k dicts (raw_cross_kj[k] etc.) for the
        # downstream cross_partner kernel to consume.
        if _use_centerQ_c:
            def _build_partner_flat(partner_data, n_partners):
                if n_partners == 0:
                    return None
                k_arr     = np.empty(n_partners, dtype=np.int64)
                n_kj_arr  = np.empty(n_partners, dtype=np.int64)
                pp_sizes  = np.empty(n_partners, dtype=np.int64)
                for p, (k, _X, pp, n_kj) in enumerate(partner_data):
                    k_arr[p]    = k
                    n_kj_arr[p] = n_kj
                    pp_sizes[p] = len(pp)
                pp_off = np.empty(n_partners + 1, dtype=np.int64)
                pp_off[0] = 0
                pp_off[1:] = np.cumsum(pp_sizes)
                X_sizes = pp_sizes * n_kj_arr
                X_off = np.empty(n_partners + 1, dtype=np.int64)
                X_off[0] = 0
                X_off[1:] = np.cumsum(X_sizes)
                cross_sizes = n_local * npno * n_kj_arr
                cross_off = np.empty(n_partners + 1, dtype=np.int64)
                cross_off[0] = 0
                cross_off[1:] = np.cumsum(cross_sizes)
                kv_sizes = n_local * n_kj_arr
                kv_off = np.empty(n_partners + 1, dtype=np.int64)
                kv_off[0] = 0
                kv_off[1:] = np.cumsum(kv_sizes)

                pp_flat = np.empty(int(pp_off[-1]), dtype=np.int64)
                X_flat  = np.empty(int(X_off[-1]))
                for p, (_k, X_p, pp_p, _n_kj) in enumerate(partner_data):
                    pp_flat[pp_off[p]:pp_off[p + 1]] = (
                        np.asarray(pp_p, dtype=np.int64))
                    X_flat[X_off[p]:X_off[p + 1]] = X_p.ravel()
                raw_cross_flat = np.zeros(int(cross_off[-1]))
                raw_kv_flat    = np.zeros(int(kv_off[-1]))
                return {
                    'k_arr': k_arr, 'n_kj_arr': n_kj_arr,
                    'pp_off': pp_off, 'pp_flat': pp_flat,
                    'X_off': X_off, 'X_flat': X_flat,
                    'cross_off': cross_off, 'kv_off': kv_off,
                    'raw_cross_flat': raw_cross_flat,
                    'raw_kv_flat': raw_kv_flat,
                }
            _t_pf_start = _ccints_time.perf_counter() if _dbg_ccints else 0.0
            _kj_pdat = [(k, pno_spaces[key]['X_pno'],
                         np.asarray(pno_spaces[key]['pair_paos']), n_kj)
                        for k, key, n_kj in kj_partners]
            _ki_pdat = [(k, pno_spaces[key]['X_pno'],
                         np.asarray(pno_spaces[key]['pair_paos']), n_ki)
                        for k, key, n_ki in ki_partners]
            _kj_flat = _build_partner_flat(_kj_pdat, len(kj_partners))
            _ki_flat = _build_partner_flat(_ki_pdat, len(ki_partners))
            if _dbg_ccints:
                _dbg_add('partner_flat',
                         _ccints_time.perf_counter() - _t_pf_start)

        pair_paos_ij = np.asarray(pair_paos_ij)

        # Group pair-aux Q's by centerQ → batched per-(pair, centerQ) ops.
        centers_of_aux = aux_atom_ids[aux_idx]
        unique_centers = np.unique(centers_of_aux)

        # Pair's "extended LMO" set — union of riatom_to_lmos_ext over
        # this pair's aux centers.  This is exactly the set of global
        # LMO indices that get populated in raw_io/raw_jo/raw_ma, so it
        # is the correct reduced axis: O(constant) for localized
        # systems instead of O(nocc).  No integral information is lost
        # because rows outside the union are zero in the original
        # formulation too.
        # Working LMO axis = pair_lmo_idx (F-coupling neighborhood) when
        # available, else fall back to the dense union over aux centers.
        # Using pair_lmo_idx from the start cuts per-pair work tensors
        # from (n_local, nocc, npno) to (n_local, nlmo_pair, npno) — the
        # dominant scaling factor in cc_ints. Rows for LMOs outside
        # pair_lmo_idx have zero contribution downstream anyway (they're
        # cropped at the storage step) so dropping them now is exact.
        if pair_lmo_idx is not None and key in pair_lmo_idx:
            _pl = np.asarray(pair_lmo_idx[key], dtype=np.int64)
            # Defensive: include endpoints i, j (Psi4-faithful invariant).
            if i not in _pl:
                _pl = np.append(_pl, i)
            if j != i and j not in _pl:
                _pl = np.append(_pl, j)
            p_lmos = np.unique(_pl).astype(np.int64)
        elif len(unique_centers) > 0:
            p_lmos = np.unique(np.concatenate(
                [riatom_to_lmos_ext[c] for c in unique_centers])
            ).astype(np.int64)
        else:
            p_lmos = np.zeros(0, dtype=np.int64)
        nlmo_p = len(p_lmos)
        p_lmos_dense = np.full(nocc, -1, dtype=np.int64)
        if nlmo_p > 0:
            p_lmos_dense[p_lmos] = np.arange(nlmo_p)

        # Now allocate the LMO-axis accumulators at the reduced size
        raw_io = np.zeros((n_local, nlmo_p))
        raw_jo = np.zeros((n_local, nlmo_p))
        raw_ma = np.zeros((n_local, nlmo_p, npno))

        # Per-partner (k, key) pre-cache of pair_paos_kj/ki & X_pno_kj/ki —
        # so we don't do dict lookups inside the centerQ loop.
        kj_data = []
        for k, key_kj, n_kj in kj_partners:
            X_kj = pno_spaces[key_kj]['X_pno']
            pp_kj = np.asarray(pno_spaces[key_kj]['pair_paos'])
            kj_data.append((k, X_kj, pp_kj, n_kj))
        ki_data = []
        for k, key_ki, n_ki in ki_partners:
            X_ki = pno_spaces[key_ki]['X_pno']
            pp_ki = np.asarray(pno_spaces[key_ki]['pair_paos'])
            ki_data.append((k, X_ki, pp_ki, n_ki))

        # Pair's "used PAO" set: union of pair_paos over self + all partners.
        # proj_ij_out only needs columns at PAOs in this set (the rest are
        # never read by the partner kernel). For water-22 this collapses
        # np_full ≈ 487 to n_red ≈ 150-250 — eliminating per-pair O(N) cost
        # in proj_ij build.
        _pair_used_pieces = [pair_paos_ij]
        for _k, _X, _pp, _n in kj_data:
            _pair_used_pieces.append(_pp)
        for _k, _X, _pp, _n in ki_data:
            _pair_used_pieces.append(_pp)
        pair_used_pao_global = np.unique(np.concatenate(
            _pair_used_pieces).astype(np.int64))

        if _stats_ccints:
            n_partners = len(kj_data) + len(ki_data)
            sum_n_kj = sum(d[3] for d in kj_data) + sum(d[3] for d in ki_data)
            np_full_avg = (sum(qab_atom[c].shape[1]
                               for c in unique_centers
                               if qab_atom[c] is not None)
                           / max(1, len(unique_centers)))
            with _stats_lock:
                _stats_pairs.append({
                    'i': int(i), 'j': int(j),
                    'npno': int(npno),
                    'n_local': int(n_local),
                    'nlmo_p': int(nlmo_p),
                    'ncenters': int(len(unique_centers)),
                    'n_partners': int(n_partners),
                    'sum_n_kj': int(sum_n_kj),
                    'np_avg': float(np_full_avg),
                    'n_pao_pair': int(len(pair_paos_ij)),
                })

        _t_centerQ_start = _ccints_time.perf_counter() if _dbg_ccints else 0.0
        for centerQ in unique_centers:
            ext_lmos = riatom_to_lmos_ext[centerQ]
            if len(ext_lmos) == 0 or qij_atom[centerQ] is None:
                continue

            # Q's belonging to this centerQ that are in pair_aux:
            mask_C = centers_of_aux == centerQ
            local_Q = np.where(mask_C)[0]                   # positions in n_local
            global_Q = aux_idx[local_Q]                     # global aux indices
            atom_pos = aux_pos_in_atom[global_Q]            # positions in atom's stack
            # Slice the per-atom stacks to just this pair's Q's. Skipped
            # on the C path (atom_pos indirection happens inside the
            # kernel) to avoid 3 fancy-index copies per centerQ.
            if not _use_centerQ_c:
                qij_b = qij_atom[centerQ][atom_pos]         # (nQp, nl, nl)
                qia_b = qia_atom[centerQ][atom_pos]         # (nQp, nl, np)
                qab_b = qab_atom[centerQ][atom_pos]         # (nQp, np, np)
            else:
                qij_b = qia_b = qab_b = None

            # Pair (ij)'s PAOs that fall inside centerQ's PAO neighborhood
            ij_pao_pos = riatom_to_paos_ext_dense[centerQ, pair_paos_ij]
            ij_mask = ij_pao_pos >= 0
            ij_u_in_pair = np.where(ij_mask)[0]
            ij_u_in_Q = ij_pao_pos[ij_mask]
            X_ij_slice = X_pno_ij[ij_u_in_pair]             # (|pair∩Q|, npno)

            i_s = riatom_to_lmos_ext_dense[centerQ, i]
            j_s = riatom_to_lmos_ext_dense[centerQ, j]

            # Intersect ext_lmos (LMOs with density on centerQ) with the
            # pair's interacting-LMO domain p_lmos. Only these rows need
            # to be populated — entries for LMOs outside p_lmos are
            # never read downstream.
            ext_local = p_lmos_dense[ext_lmos]
            keep_mask = ext_local >= 0
            if not np.any(keep_mask):
                # Nothing to contribute from this centerQ; still need to
                # handle the diagonal i_s/j_s and PAO pieces below.
                ext_kept_lmos = np.zeros(0, dtype=np.int64)
                ext_kept_pos = np.zeros(0, dtype=np.int64)
            else:
                ext_kept_pos = np.where(keep_mask)[0]       # rows of qij_b
                ext_kept_lmos = ext_local[keep_mask]        # cols of raw_io

            np_full = (qab_atom[centerQ].shape[1] if qab_b is None
                       else qab_b.shape[1])
            if _use_centerQ_c:
                # Native-C path: pass FULL per-atom stacks + atom_pos.
                # The kernel does the atom-page indirection internally;
                # Python no longer pre-slices `qij_atom[centerQ][atom_pos]`
                # (eliminates 3 fancy-index copies per centerQ).
                _qij_full = qij_atom[centerQ]
                _qia_full = qia_atom[centerQ]
                _qab_full = qab_atom[centerQ]
                _local_Q_long = np.ascontiguousarray(local_Q, dtype=np.int64)
                _atom_pos_long = np.ascontiguousarray(atom_pos, dtype=np.int64)
                _ij_u_in_Q_long = (
                    np.ascontiguousarray(ij_u_in_Q, dtype=np.int64)
                    if len(ij_u_in_Q) > 0
                    else np.zeros(0, dtype=np.int64))
                _ekp = np.ascontiguousarray(ext_kept_pos, dtype=np.int64)
                _ekl = np.ascontiguousarray(ext_kept_lmos, dtype=np.int64)
                _X_slice = np.ascontiguousarray(X_ij_slice)
                _nQp_c = int(_local_Q_long.size)
                _nl_c = int(_qij_full.shape[1])
                _np_full_c = int(_qab_full.shape[1])
                _npp_c = int(len(ij_u_in_Q))
                _n_kept_c = int(_ekp.size)
                # Per-centerQ pair_used_in_Q: positions in [0, np_full)
                # for the global PAOs in pair_used_pao_global. Sorted.
                # pair_used_inv: full→red position map (length np_full).
                _full_pos_for_used = riatom_to_paos_ext_dense[
                    centerQ, pair_used_pao_global]
                _used_mask = _full_pos_for_used >= 0
                _pair_used_in_Q = np.sort(
                    _full_pos_for_used[_used_mask]).astype(np.int64)
                _n_red_c = int(_pair_used_in_Q.size)
                _pair_used_inv = np.full(
                    _np_full_c, -1, dtype=np.int64)
                if _n_red_c > 0:
                    _pair_used_inv[_pair_used_in_Q] = np.arange(
                        _n_red_c, dtype=np.int64)
                if _npp_c > 0 and _n_red_c > 0:
                    proj_ij = np.empty((_nQp_c, npno, _n_red_c))
                else:
                    proj_ij = None
                _proj_ptr = (proj_ij if proj_ij is not None
                             else np.empty(0))
                _t_pk_start = (_ccints_time.perf_counter()
                               if _dbg_ccints else 0.0)
                _libcc_centerQ.DLPNOpair_centerQ_step(
                    _qij_full.ctypes.data_as(_ctypes_cQ.c_void_p),
                    _qia_full.ctypes.data_as(_ctypes_cQ.c_void_p),
                    _qab_full.ctypes.data_as(_ctypes_cQ.c_void_p),
                    _local_Q_long.ctypes.data_as(_ctypes_cQ.c_void_p),
                    _atom_pos_long.ctypes.data_as(_ctypes_cQ.c_void_p),
                    int(i_s), int(j_s),
                    _ij_u_in_Q_long.ctypes.data_as(_ctypes_cQ.c_void_p),
                    _ekp.ctypes.data_as(_ctypes_cQ.c_void_p),
                    _ekl.ctypes.data_as(_ctypes_cQ.c_void_p),
                    _X_slice.ctypes.data_as(_ctypes_cQ.c_void_p),
                    _pair_used_in_Q.ctypes.data_as(_ctypes_cQ.c_void_p),
                    _nQp_c, _nl_c, _np_full_c,
                    npno, _npp_c, _n_kept_c, n_local, nlmo_p, _n_red_c,
                    raw_io.ctypes.data_as(_ctypes_cQ.c_void_p),
                    raw_jo.ctypes.data_as(_ctypes_cQ.c_void_p),
                    raw_iv.ctypes.data_as(_ctypes_cQ.c_void_p),
                    raw_jv.ctypes.data_as(_ctypes_cQ.c_void_p),
                    raw_pair.ctypes.data_as(_ctypes_cQ.c_void_p),
                    raw_ma.ctypes.data_as(_ctypes_cQ.c_void_p),
                    raw_ab.ctypes.data_as(_ctypes_cQ.c_void_p),
                    _proj_ptr.ctypes.data_as(_ctypes_cQ.c_void_p),
                )
                if _dbg_ccints:
                    _dbg_add('cQ_pair_kernel',
                             _ccints_time.perf_counter() - _t_pk_start)
            else:
                # raw_io[local_Q, p_lmos_local] = qij_b[:, i_s, kept]
                if i_s >= 0 and ext_kept_lmos.size > 0:
                    raw_io[np.ix_(local_Q, ext_kept_lmos)] = \
                        qij_b[:, i_s, :][:, ext_kept_pos]
                if j_s >= 0 and ext_kept_lmos.size > 0:
                    raw_jo[np.ix_(local_Q, ext_kept_lmos)] = \
                        qij_b[:, j_s, :][:, ext_kept_pos]
                if i_s >= 0 and j_s >= 0:
                    raw_pair[local_Q] = qij_b[:, i_s, j_s]

                if len(ij_u_in_Q) > 0:
                    # qia restricted to pair's PAOs:
                    qia_b_pp = qia_b[:, :, ij_u_in_Q]            # (nQp, nl, npp)
                    if i_s >= 0:
                        raw_iv[local_Q] = qia_b_pp[:, i_s, :] @ X_ij_slice
                    if j_s >= 0:
                        raw_jv[local_Q] = qia_b_pp[:, j_s, :] @ X_ij_slice

                    # raw_ma: keep only ext_lmos that are in p_lmos
                    if ext_kept_lmos.size > 0:
                        nQp = len(local_Q)
                        qia_b_pp_kept = qia_b_pp[:, ext_kept_pos, :]
                        ma_b_kept = (
                            qia_b_pp_kept.reshape(
                                nQp * ext_kept_pos.size, -1)
                            @ X_ij_slice
                        ).reshape(nQp, ext_kept_pos.size, npno)
                        raw_ma[np.ix_(local_Q, ext_kept_lmos)] = ma_b_kept

                    # raw_ab: X^T qab_b_pp X
                    qab_b_pp = qab_b[
                        :, ij_u_in_Q[:, None], ij_u_in_Q[None, :]]
                    tmp = (qab_b_pp.reshape(nQp * len(ij_u_in_Q), -1)
                           @ X_ij_slice).reshape(
                               nQp, len(ij_u_in_Q), npno)
                    raw_ab[local_Q] = np.matmul(X_ij_slice.T, tmp)

                # proj_ij for cross-pair partners
                if len(ij_u_in_Q) > 0:
                    if len(ij_u_in_Q) == np_full and np.array_equal(
                            ij_u_in_Q, np.arange(np_full)):
                        qab_ij = qab_b
                    else:
                        qab_ij = qab_b[:, ij_u_in_Q, :]
                    proj_ij = np.matmul(X_ij_slice.T, qab_ij)
                else:
                    proj_ij = None

            # Per-partner work fused into one Cython kernel that reads
            # proj_ij / qia_b directly with index arrays — avoids the
            # forced copy on proj_ij[:, :, idx] (fancy on last axis) and
            # eliminates per-partner Python+numpy dispatch overhead.
            # Profile: this loop pair was 73.5% of cc_ints CPU.
            local_Q_long = np.ascontiguousarray(local_Q, dtype=np.int64)
            do_proj = 1 if proj_ij is not None else 0
            _proj_arg = (proj_ij if proj_ij is not None
                         else np.empty((1, 1, 1)))

            if _use_centerQ_c:
                # Native-C path: one call per side processes all partners
                # for this centerQ. Eliminates the per-partner Python prep
                # loop (~120k iterations across CCSD setup on water-10).
                _paos_dense_at = np.ascontiguousarray(
                    riatom_to_paos_ext_dense[centerQ].astype(np.int64))
                _lmos_dense_at = np.ascontiguousarray(
                    riatom_to_lmos_ext_dense[centerQ].astype(np.int64))
                _qia_full = qia_atom[centerQ]
                _atom_pos_long = np.ascontiguousarray(atom_pos, dtype=np.int64)
                _proj_c = np.ascontiguousarray(_proj_arg)
                _nQp = local_Q_long.shape[0]
                _nl_at = _qia_full.shape[1]
                _np_full = _qia_full.shape[2]
                _nao_pao_total = riatom_to_paos_ext_dense.shape[1]

                _t_pn_start = (_ccints_time.perf_counter()
                               if _dbg_ccints else 0.0)
                if _kj_flat is not None and do_proj:
                    _libcc_centerQ.DLPNOpartners_centerQ_step(
                        _proj_c.ctypes.data_as(_ctypes_cQ.c_void_p),
                        _qia_full.ctypes.data_as(_ctypes_cQ.c_void_p),
                        _atom_pos_long.ctypes.data_as(_ctypes_cQ.c_void_p),
                        local_Q_long.ctypes.data_as(_ctypes_cQ.c_void_p),
                        _paos_dense_at.ctypes.data_as(_ctypes_cQ.c_void_p),
                        _lmos_dense_at.ctypes.data_as(_ctypes_cQ.c_void_p),
                        _pair_used_inv.ctypes.data_as(_ctypes_cQ.c_void_p),
                        len(kj_partners),
                        _kj_flat['k_arr'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _kj_flat['n_kj_arr'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _kj_flat['pp_off'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _kj_flat['pp_flat'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _kj_flat['X_off'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _kj_flat['X_flat'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _kj_flat['cross_off'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _kj_flat['kv_off'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _nQp, npno, _np_full, _nl_at,
                        n_local, _nao_pao_total, nocc, _n_red_c,
                        _kj_flat['raw_cross_flat'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _kj_flat['raw_kv_flat'].ctypes.data_as(_ctypes_cQ.c_void_p),
                    )
                if _ki_flat is not None and do_proj:
                    _libcc_centerQ.DLPNOpartners_centerQ_step(
                        _proj_c.ctypes.data_as(_ctypes_cQ.c_void_p),
                        _qia_full.ctypes.data_as(_ctypes_cQ.c_void_p),
                        _atom_pos_long.ctypes.data_as(_ctypes_cQ.c_void_p),
                        local_Q_long.ctypes.data_as(_ctypes_cQ.c_void_p),
                        _paos_dense_at.ctypes.data_as(_ctypes_cQ.c_void_p),
                        _lmos_dense_at.ctypes.data_as(_ctypes_cQ.c_void_p),
                        _pair_used_inv.ctypes.data_as(_ctypes_cQ.c_void_p),
                        len(ki_partners),
                        _ki_flat['k_arr'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _ki_flat['n_kj_arr'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _ki_flat['pp_off'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _ki_flat['pp_flat'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _ki_flat['X_off'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _ki_flat['X_flat'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _ki_flat['cross_off'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _ki_flat['kv_off'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _nQp, npno, _np_full, _nl_at,
                        n_local, _nao_pao_total, nocc, _n_red_c,
                        _ki_flat['raw_cross_flat'].ctypes.data_as(_ctypes_cQ.c_void_p),
                        _ki_flat['raw_kv_flat'].ctypes.data_as(_ctypes_cQ.c_void_p),
                    )
                if _dbg_ccints:
                    _dbg_add('cQ_partner_kernel',
                             _ccints_time.perf_counter() - _t_pn_start)
            else:
                for k, X_kj, pp_kj, n_kj in kj_data:
                    k_s = int(riatom_to_lmos_ext_dense[centerQ, k])
                    kj_pao_pos = riatom_to_paos_ext_dense[centerQ, pp_kj]
                    kj_mask = kj_pao_pos >= 0
                    kj_u_in_pair = np.where(kj_mask)[0]
                    kj_u_in_Q = kj_pao_pos[kj_mask]
                    if len(kj_u_in_Q) == 0:
                        continue
                    X_kj_slice = np.ascontiguousarray(X_kj[kj_u_in_pair])
                    kj_u_in_Q_long = np.ascontiguousarray(kj_u_in_Q, dtype=np.int64)
                    partner_apply(
                        _proj_arg, qia_b, k_s,
                        local_Q_long, kj_u_in_Q_long, X_kj_slice,
                        raw_cross_kj[k], raw_kv_kj[k],
                        do_proj,
                    )

                for k, X_ki, pp_ki, n_ki in ki_data:
                    k_s = int(riatom_to_lmos_ext_dense[centerQ, k])
                    ki_pao_pos = riatom_to_paos_ext_dense[centerQ, pp_ki]
                    ki_mask = ki_pao_pos >= 0
                    ki_u_in_pair = np.where(ki_mask)[0]
                    ki_u_in_Q = ki_pao_pos[ki_mask]
                    if len(ki_u_in_Q) == 0:
                        continue
                    X_ki_slice = np.ascontiguousarray(X_ki[ki_u_in_pair])
                    ki_u_in_Q_long = np.ascontiguousarray(ki_u_in_Q, dtype=np.int64)
                    partner_apply(
                        _proj_arg, qia_b, k_s,
                        local_Q_long, ki_u_in_Q_long, X_ki_slice,
                        raw_cross_ji[k], raw_kv_ki[k],
                        do_proj,
                    )

        if _dbg_ccints:
            _dbg_add('centerQ_loop',
                     _ccints_time.perf_counter() - _t_centerQ_start)
        # C path: skip flat→dict scatter — _run_one_side reads the flat
        # buffers directly. Python fallback already wrote into the dicts.
        _t_jhi_start = _ccints_time.perf_counter() if _dbg_ccints else 0.0
        # Apply local J^{-1/2}
        j2c_local = j2c[np.ix_(aux_idx, aux_idx)]
        eigvals, eigvecs = np.linalg.eigh(j2c_local)
        keep = eigvals > 1e-14
        jhi = (eigvecs[:, keep] * (1.0 / np.sqrt(eigvals[keep]))) \
            @ eigvecs[:, keep].T

        q_iv = jhi @ raw_iv
        q_jv = jhi @ raw_jv
        q_io_pfit = jhi @ raw_io                # (n_local, nlmo_p) p_lmos-axis
        q_jo_pfit = jhi @ raw_jo
        q_pair = jhi @ raw_pair
        Qma_pfit = (jhi @ raw_ma.reshape(n_local, -1)
                    ).reshape(n_local, nlmo_p, npno)
        Qab = (jhi @ raw_ab.reshape(n_local, -1)).reshape(n_local, npno, npno)

        # Working axis = storage axis (pair_lmo_idx). No crop needed —
        # the centerQ loop already populated rows only for LMOs in p_lmos.
        q_io = q_io_pfit
        q_jo = q_jo_pfit
        Qma  = Qma_pfit

        if _dbg_ccints:
            _dbg_add('jhi_apply',
                     _ccints_time.perf_counter() - _t_jhi_start)
        _t_finalKJ_start = _ccints_time.perf_counter() if _dbg_ccints else 0.0
        K_iajb = q_iv.T @ q_jv
        # K_mnij removed: dead code (built but never read by any consumer).
        # K_bar_ij/ji/chem all reduced on the p_lmos axis: (nlmo_p, npno).
        K_bar_ij = q_io.T @ q_jv
        K_bar_ji = q_jo.T @ q_iv
        K_bar_chem = np.tensordot(q_pair, Qma, axes=(0, 0))
        J_ijab = np.tensordot(q_pair, Qab, axes=(0, 0))
        # Psi4 ccsd.cc:1402 K_tilde_chem (L pre-summed (q_iv|Qab) tensors).
        # Stored once here so compute_C_tilde, build_D_tilde, and the T1
        # residual all share without per-iter rebuilds.
        Qab_flat = Qab.reshape(n_local, npno * npno)
        K_tilde_chem_i = np.ascontiguousarray(q_iv.T @ Qab_flat)
        K_tilde_chem_j = np.ascontiguousarray(q_jv.T @ Qab_flat)

        if _dbg_ccints:
            _dbg_add('final_KJ',
                     _ccints_time.perf_counter() - _t_finalKJ_start)
        _t_cross_start = _ccints_time.perf_counter() if _dbg_ccints else 0.0

        # === Cross-partner final J/K assembly: native-C path under
        # DLPNO_C_CYCLE=1 (DLPNOcross_partner_assemble in
        # pyscf/lib/cc/dlpno_cross_partner.c). Same math as the per-
        # partner Python loop below; eliminates ~24K Python ctypes
        # dispatches per CCSD run on water-10. ===
        if _use_centerQ_c and (kj_partners or ki_partners):
            # Z_iv = q_iv^T @ jhi  (npno, n_local), Z_jv = q_jv^T @ jhi
            Z_iv = np.ascontiguousarray(q_iv.T @ jhi)
            Z_jv = np.ascontiguousarray(q_jv.T @ jhi)
            jhi_c = np.ascontiguousarray(jhi)
            q_io_c = np.ascontiguousarray(q_io)
            q_jo_c = np.ascontiguousarray(q_jo)

            def _run_one_side(partners, flat_buf,
                              q_io_or_jo_c, Z_iv_or_jv):
                """Process all partners on one side (kj or ki) in one C call.
                Reads raw_cross/raw_kv directly from the per-pair flat
                buffer populated by DLPNOpartners_centerQ_step — no
                dict→flat scatter required.

                Returns (J_dict, K_dict) keyed by k (global LMO index)."""
                if not partners or flat_buf is None:
                    return {}, {}
                n_partners = len(partners)
                k_loc_arr = np.empty(n_partners, dtype=np.int64)
                n_kj_arr  = flat_buf['n_kj_arr']
                JK_sizes  = np.empty(n_partners, dtype=np.int64)
                for p, (k, _key, n_kj) in enumerate(partners):
                    k_loc_arr[p] = int(p_lmos_dense[k])
                    JK_sizes[p]  = npno * n_kj
                cross_off = flat_buf['cross_off']
                kv_off    = flat_buf['kv_off']
                JK_off = np.empty(n_partners + 1, dtype=np.int64); JK_off[0] = 0
                JK_off[1:] = np.cumsum(JK_sizes)
                raw_cross_flat = flat_buf['raw_cross_flat']
                raw_kv_flat    = flat_buf['raw_kv_flat']
                J_out_flat = np.empty(int(JK_off[-1]))
                K_out_flat = np.empty(int(JK_off[-1]))
                _libcc_centerQ.DLPNOcross_partner_assemble(
                    int(n_partners),
                    k_loc_arr.ctypes.data_as(_ctypes_cQ.c_void_p),
                    n_kj_arr.ctypes.data_as(_ctypes_cQ.c_void_p),
                    cross_off.ctypes.data_as(_ctypes_cQ.c_void_p),
                    raw_cross_flat.ctypes.data_as(_ctypes_cQ.c_void_p),
                    kv_off.ctypes.data_as(_ctypes_cQ.c_void_p),
                    raw_kv_flat.ctypes.data_as(_ctypes_cQ.c_void_p),
                    jhi_c.ctypes.data_as(_ctypes_cQ.c_void_p),
                    q_io_or_jo_c.ctypes.data_as(_ctypes_cQ.c_void_p),
                    Z_iv_or_jv.ctypes.data_as(_ctypes_cQ.c_void_p),
                    JK_off.ctypes.data_as(_ctypes_cQ.c_void_p),
                    J_out_flat.ctypes.data_as(_ctypes_cQ.c_void_p),
                    JK_off.ctypes.data_as(_ctypes_cQ.c_void_p),
                    K_out_flat.ctypes.data_as(_ctypes_cQ.c_void_p),
                    n_local, nlmo_p, npno,
                )
                J_dict, K_dict = {}, {}
                for p, (k, _key, n_kj) in enumerate(partners):
                    J_dict[k] = J_out_flat[
                        JK_off[p]:JK_off[p + 1]].reshape(npno, n_kj).copy()
                    K_dict[k] = K_out_flat[
                        JK_off[p]:JK_off[p + 1]].reshape(npno, n_kj).copy()
                return J_dict, K_dict

            J_kj_byk, K_kj_byk = _run_one_side(
                kj_partners, _kj_flat, q_io_c, Z_iv)
            J_ki_byk, K_ki_byk = _run_one_side(
                ki_partners, _ki_flat, q_jo_c, Z_jv)
            J_ij_kj      = {(key, k): J_kj_byk[k] for k in J_kj_byk}
            K_ij_kj_dict = {(key, k): K_kj_byk[k] for k in K_kj_byk}
            J_ji_ki      = {(key, k): J_ki_byk[k] for k in J_ki_byk}
            K_ji_ki_dict = {(key, k): K_ki_byk[k] for k in K_ki_byk}
        else:
            J_ij_kj = {}
            K_ij_kj_dict = {}
            for k, key_kj, _ in kj_partners:
                cross_fitted = (jhi @ raw_cross_kj[k].reshape(n_local, -1)
                                ).reshape(n_local, npno, raw_cross_kj[k].shape[2])
                k_loc = int(p_lmos_dense[k])
                q_ik = q_io[:, k_loc]
                J_ij_kj[(key, k)] = np.tensordot(q_ik, cross_fitted, axes=(0, 0))
                q_kv_kj = jhi @ raw_kv_kj[k]
                K_ij_kj_dict[(key, k)] = q_iv.T @ q_kv_kj

            J_ji_ki = {}
            K_ji_ki_dict = {}
            for k, key_ki, _ in ki_partners:
                cross_fitted = (jhi @ raw_cross_ji[k].reshape(n_local, -1)
                                ).reshape(n_local, npno, raw_cross_ji[k].shape[2])
                k_loc = int(p_lmos_dense[k])
                q_jk = q_jo[:, k_loc]
                J_ji_ki[(key, k)] = np.tensordot(q_jk, cross_fitted, axes=(0, 0))
                q_kv_ki = jhi @ raw_kv_ki[k]
                K_ji_ki_dict[(key, k)] = q_jv.T @ q_kv_ki

        if _dbg_ccints:
            _dbg_add('cross_kj',
                     _ccints_time.perf_counter() - _t_cross_start)
        return key, {
            'K_iajb': K_iajb,
            'K_bar_ij': K_bar_ij,
            'K_bar_ji': K_bar_ji,
            'K_bar_chem': K_bar_chem,  # (nlmo_p, npno) reduced
            'K_tilde_chem_i': K_tilde_chem_i,  # (npno, npno²) Psi4 ccsd.cc:1402
            'K_tilde_chem_j': K_tilde_chem_j,  # (npno, npno²) for j-direction
            'J_ijab': J_ijab,
            'J_ij_kj': J_ij_kj,
            'K_ij_kj': K_ij_kj_dict,
            'J_ji_ki': J_ji_ki,
            'K_ji_ki': K_ji_ki_dict,
            'i_Qa': q_iv.copy(),
            'j_Qa': q_jv.copy(),
            'i_Qk': q_io.copy(),     # (n_local, nlmo_p) reduced
            'j_Qk': q_jo.copy(),     # (n_local, nlmo_p) reduced
            'Qma': Qma,              # (n_local, nlmo_p, npno) reduced
            'Qab': Qab,
            'n_local': n_local,
            'aux_idx': aux_idx,
            # LMO-axis metadata: ci['p_lmos'] is the sorted global LMO
            # indices along the reduced "nocc" axis of i_Qk / j_Qk / Qma
            # / K_mnij / K_bar_chem. Consumers that previously did
            # ci['Qma'][:, pair_lmo_idx[key], :] now use ci['Qma']
            # directly (pair_lmo_idx[key] == p_lmos by construction).
            'p_lmos': p_lmos,
            'p_lmos_dense': p_lmos_dense,
        }

    # Dispatch: parallel via _pool if provided, else serial. Each pair's
    # work is read-only on shared inputs (pno_spaces, pair_aux_idx, sparse
    # arrays) and writes only to its own local arrays before returning the
    # cc_ints entry — thread-safe.
    _t_pool_start = _ccints_setup_time.perf_counter()
    if _pool is not None:
        for k, entry in _pool.map(_process_pair, keys):
            cc_ints[k] = entry
    else:
        for key in keys:
            k, entry = _process_pair(key)
            cc_ints[k] = entry
    _t_pool_wall = _ccints_setup_time.perf_counter() - _t_pool_start

    if _dbg_ccints:
        _items = sorted(_dbg_acc.items(), key=lambda kv: -kv[1])
        _summary = ' '.join(f'{n}={v:.2f}s' for n, v in _items if v > 0.0)
        print(f"[CCINTS_DBG] (n_pairs={len(keys)}) {_summary}", flush=True)
        print(f"[CCINTS_DBG] OUTER walls: screening={_t_screening:.2f}s "
              f"sparse_arrays={_t_sparse:.2f}s atom_stacks={_t_atom_stacks:.2f}s "
              f"pool_dispatch={_t_pool_wall:.2f}s", flush=True)

    if _stats_ccints and _stats_pairs:
        _np = len(_stats_pairs)
        def _avg(field):
            return sum(p[field] for p in _stats_pairs) / _np
        def _max(field):
            return max(p[field] for p in _stats_pairs)
        def _sum(field):
            return sum(p[field] for p in _stats_pairs)
        # Per-pair work proxies (units of FLOPs ~ leading O()):
        # phase A (centerQ loop): ncenters * (n_partners*npno*np_avg + npno*nlmo_p*np_avg)
        # phase B (final KJ): n_local * npno^2 * nlmo_p
        # phase C (cross_kj): n_partners * n_local * npno * sum_n_kj
        wA = sum(p['ncenters'] * (p['n_partners'] * p['npno'] * p['np_avg']
                                  + p['npno'] * p['nlmo_p'] * p['np_avg'])
                 for p in _stats_pairs)
        wB = sum(p['n_local'] * p['npno']**2 * p['nlmo_p']
                 for p in _stats_pairs)
        wC = sum(p['n_partners'] * p['n_local'] * p['npno'] * p['sum_n_kj']
                 / max(1, p['n_partners'])
                 for p in _stats_pairs)
        print(f"[CCINTS_STATS] n_pairs={_np}", flush=True)
        print(f"  per-pair avg: npno={_avg('npno'):.1f} "
              f"n_local={_avg('n_local'):.1f} "
              f"nlmo_p={_avg('nlmo_p'):.1f} "
              f"ncenters={_avg('ncenters'):.1f} "
              f"n_partners={_avg('n_partners'):.1f} "
              f"sum_n_kj={_avg('sum_n_kj'):.1f}", flush=True)
        print(f"  per-pair max: npno={_max('npno')} "
              f"n_local={_max('n_local')} "
              f"nlmo_p={_max('nlmo_p')} "
              f"ncenters={_max('ncenters')} "
              f"n_partners={_max('n_partners')}", flush=True)
        print(f"  totals: n_local_sum={_sum('n_local')} "
              f"nlmo_p_sum={_sum('nlmo_p')} "
              f"partners_sum={_sum('n_partners')}", flush=True)
        print(f"  workA={wA:.2e} workB={wB:.2e} workC={wC:.2e}", flush=True)

    # Debug: zero p_lmos\pair_lmo_idx rows in selected cc_ints fields, to
    # localize which consumer(s) drift the energy when those rows go away.
    # Set env DLPNO_ZERO_EXTRA_LMOS to a comma-separated subset of:
    #   {Qma, i_Qk, j_Qk, K_bar_chem, K_bar_ij, K_bar_ji, all}
    _zero_fields_env = os.environ.get('DLPNO_ZERO_EXTRA_LMOS', '')
    if _zero_fields_env and pair_lmo_idx is not None:
        _zero_set = set(_zero_fields_env.split(','))
        if 'all' in _zero_set:
            _zero_set = {'Qma', 'i_Qk', 'j_Qk',
                         'K_bar_chem', 'K_bar_ij', 'K_bar_ji'}
        n_pairs_touched = 0
        n_extras_total = 0
        for key, ci in cc_ints.items():
            if ci is None or key not in pair_lmo_idx:
                continue
            pdom = set(int(x) for x in pair_lmo_idx[key])
            p_lmos_arr = ci['p_lmos']
            extras = np.array(
                [i for i, l in enumerate(p_lmos_arr)
                 if int(l) not in pdom], dtype=np.intp)
            if extras.size == 0:
                continue
            n_pairs_touched += 1
            n_extras_total += int(extras.size)
            for fld in _zero_set:
                if fld == 'Qma':
                    ci['Qma'][:, extras, :] = 0.0
                elif fld == 'i_Qk':
                    ci['i_Qk'][:, extras] = 0.0
                elif fld == 'j_Qk':
                    ci['j_Qk'][:, extras] = 0.0
                elif fld in ('K_bar_chem', 'K_bar_ij', 'K_bar_ji'):
                    ci[fld][extras] = 0.0
        print(f"[DLPNO_ZERO_EXTRA_LMOS={_zero_fields_env}] "
              f"zeroed {n_extras_total} p_lmos\\pair_lmo_idx rows across "
              f"{n_pairs_touched} pairs in fields {sorted(_zero_set)}",
              flush=True)

    return cc_ints


def t1_ints(cc_ints, t1_pno, pno_spaces, S_pno_cache, keys, nocc,
            pair_lmo_idx=None, t1_cache=None, _pool=None):
    """Build T1-dressed DF intermediates, matching Psi4 t1_ints().

    For each pair (ij), builds:
        i_Qa_t1[Q, a] = i_Qa[Q,a] - i_Qk[Q,:] @ T1_all[:,a]
                       + Σ_b (Qab[Q,a,b] - T1_all^T @ Qma[Q,:,b]) * t1_i[b]
        i_Qk_t1[Q, k] = i_Qk[Q,k] + Σ_a Qma[Q, k, a] * t1_i[a]   (Psi4 ccsd.cc:1510-1519)
        (and the j-counterparts)

    The implicit `k` sum inside this formula is restricted to
    pair_lmo_idx[key] (Psi4 lmopair_to_lmos_[ij]) when provided.

    Returns: dict per key with 4 entries: i_Qa_t1, j_Qa_t1, i_Qk_t1, j_Qk_t1.
    The Qk_t1 entries are the same quantities compute_B_tilde used to
    rebuild inline; downstream consumers should read from this dict.

    `_pool`: optional ThreadPoolExecutor for parallel per-key iteration.
    """
    # Phase 1: use pre-built t1 projection cache if provided; else build.
    if t1_cache is None:
        from pyscf.cc.dlpno_tccsd.pair_index import (
            PairIndex, build_t1_cache,
        )
        _pi = PairIndex(
            pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
        t1_cache = build_t1_cache(
            t1_pno, _pi, S_pno_cache, pno_spaces)

    _use_c_cycle = bool(int(os.environ.get('DLPNO_C_CYCLE', '0')))
    if _use_c_cycle:
        # Lazy-init libcc for the t1_ints C kernel. See
        # pyscf/lib/cc/dlpno_t1_ints.c::DLPNOt1_ints_pair_side.
        import ctypes
        from pyscf import lib as _pyscflib
        _libcc_t1 = getattr(t1_ints, '_libcc', None)
        if _libcc_t1 is None:
            _libcc_t1 = _pyscflib.load_library('libcc')
            _libcc_t1.DLPNOt1_ints_pair_side.restype = None
            _libcc_t1.DLPNOt1_ints_pair_side.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,                # Qa_t1_out, Qk_t1_out
                ctypes.c_void_p, ctypes.c_void_p,                # Qa_full, Qk_local
                ctypes.c_void_p, ctypes.c_void_p,                # Qma, Qab
                ctypes.c_void_p, ctypes.c_void_p,                # t1_lmo, T1_local
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,  # n_local, nlmo, npno
            ]
            t1_ints._libcc = _libcc_t1
    else:
        _libcc_t1 = None

    def _dress_one_pair(key):
        ci = cc_ints.get(key)
        if ci is None:
            return key, None
        i, j = key

        if pair_lmo_idx is not None and key in pair_lmo_idx:
            lmo_idx = np.asarray(pair_lmo_idx[key])
        else:
            lmo_idx = np.arange(nocc)

        T1_local = np.ascontiguousarray(
            t1_cache[key][np.asarray(lmo_idx, dtype=np.intp)])
        # Qma reduced to (n_local, nlmo_p, npno); translate global lmo_idx.
        _lmo_idx_in_p = np.asarray(
            ci['p_lmos_dense'])[lmo_idx].astype(np.intp)
        Qma = ci['Qma'][:, _lmo_idx_in_p, :]  # (n_local, nlmo, npno)
        Qab = ci['Qab']                       # (n_local, npno, npno)

        if _libcc_t1 is not None:
            import ctypes
            Qma_c = np.ascontiguousarray(Qma)
            Qab_c = np.ascontiguousarray(Qab)
            T1_local_c = np.ascontiguousarray(T1_local)
            n_local_c = Qma_c.shape[0]
            nlmo_c = Qma_c.shape[1]
            npno_c = Qma_c.shape[2]

            def _dress(lmo_global, Qa_key):
                Qa_full = np.ascontiguousarray(ci[Qa_key])
                Qk_local = np.ascontiguousarray(
                    ci[Qa_key.replace('Qa', 'Qk')][:, _lmo_idx_in_p])
                t1_lmo = np.ascontiguousarray(
                    t1_cache[key][int(lmo_global)])
                result_qa = np.empty((n_local_c, npno_c))
                result_qk = np.empty((n_local_c, nlmo_c))
                _libcc_t1.DLPNOt1_ints_pair_side(
                    result_qa.ctypes.data_as(ctypes.c_void_p),
                    result_qk.ctypes.data_as(ctypes.c_void_p),
                    Qa_full.ctypes.data_as(ctypes.c_void_p),
                    Qk_local.ctypes.data_as(ctypes.c_void_p),
                    Qma_c.ctypes.data_as(ctypes.c_void_p),
                    Qab_c.ctypes.data_as(ctypes.c_void_p),
                    t1_lmo.ctypes.data_as(ctypes.c_void_p),
                    T1_local_c.ctypes.data_as(ctypes.c_void_p),
                    n_local_c, nlmo_c, npno_c,
                )
                return result_qa, result_qk
        else:
            def _dress(lmo_global, Qa_key):
                Qa_full = ci[Qa_key]                                 # (n_local, npno)
                # i_Qk/j_Qk reduced; translate global lmo_idx -> p_lmos position.
                Qk_local = ci[Qa_key.replace('Qa', 'Qk')][:, _lmo_idx_in_p]  # (n_local, nlmo)

                t1_lmo = t1_cache[key][int(lmo_global)]
                qma_t1 = np.einsum('Qmb,b->Qm', Qma, t1_lmo)         # (n_local, nlmo)

                # i_Qa_t1: Psi4 ccsd.cc:1521-1534
                result_qa = Qa_full - Qk_local @ T1_local            # (n_local, npno)
                result_qa += np.einsum('Qab,b->Qa', Qab, t1_lmo)
                result_qa -= qma_t1 @ T1_local                       # (n_local, npno)

                # i_Qk_t1: Psi4 ccsd.cc:1510-1519. Same quantity compute_B_tilde
                # rebuilt inline; produce here so consumers can share.
                result_qk = Qk_local + qma_t1                        # (n_local, nlmo)

                return result_qa, result_qk

        i_Qa_t1, i_Qk_t1 = _dress(i, 'i_Qa')
        j_Qa_t1, j_Qk_t1 = _dress(j, 'j_Qa')

        return key, {
            'i_Qa_t1': i_Qa_t1,
            'j_Qa_t1': j_Qa_t1,
            'i_Qk_t1': i_Qk_t1,   # (n_local, nlmo) — pair-LMO-domain only
            'j_Qk_t1': j_Qk_t1,
        }

    dressed = {}
    if _pool is not None:
        for key, val in _pool.map(_dress_one_pair, list(keys)):
            if val is not None:
                dressed[key] = val
    else:
        for key in keys:
            _, val = _dress_one_pair(key)
            if val is not None:
                dressed[key] = val

    # T1INTS_DUMP: parity dump vs Psi4 ccsd.cc:1491 t1_ints output.
    # Psi4 stores ordered pairs (i,j) AND (j,i); each entry has i_Qk_t1
    # for its "i-side". To match Psi4's total, our canonical-pair store
    # contributes: diag: ‖i_Qk_t1‖² once; off-diag: ‖i_Qk_t1‖² + ‖j_Qk_t1‖².
    if int(os.environ.get('DLPNO_DUMP_T1INTS', '0')):
        _it = getattr(t1_ints, '_iter', 0)
        t1_ints._iter = _it + 1
        if _it <= 2:
            iqk_fro2 = 0.0
            iqa_fro2 = 0.0
            n_strong_ordered = 0
            for key, val in dressed.items():
                i, j = key
                iqk_fro2 += float(np.sum(val['i_Qk_t1'] ** 2))
                iqa_fro2 += float(np.sum(val['i_Qa_t1'] ** 2))
                n_strong_ordered += 1
                if i != j:
                    iqk_fro2 += float(np.sum(val['j_Qk_t1'] ** 2))
                    iqa_fro2 += float(np.sum(val['j_Qa_t1'] ** 2))
                    n_strong_ordered += 1
            print(f"T1INTS_DUMP iter={_it} n_strong={n_strong_ordered} "
                  f"iQk_fro2={iqk_fro2:.12e} "
                  f"iQa_fro2={iqa_fro2:.12e}",
                  flush=True)

    return dressed


def t1_fock(cc_ints, dressed_ints, t1_pno, fov_pno, pno_spaces,
            S_pno_cache, F_lmo, eps_lmo, foo_t2, keys, nocc, _pool=None,
            pair_lmo_idx=None, t1_cache=None):
    """Build dressed Fock matrices matching Psi4 t1_fock().

    Returns:
        Fkj: (nocc, nocc) dressed occupied Fock (full, includes Eq 94)
        Fab_all: dict pair_key -> (npno, npno) dressed virtual Fock
        foo_t1: (nocc, nocc) T1 part of foo
        Fij_bar_snapshot: (nocc, nocc) Fkj BEFORE Eq 94's Fia_bar_jj term —
            shared with T1 residual to avoid recomputing per-pair d_ij/d_ji.

    LMO sums inside each per-pair dressing are restricted to
    pair_lmo_idx[key] (Psi4 lmopair_to_lmos_[ij]) when provided.
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

    def _pair_domain(key):
        if pair_lmo_idx is not None and key in pair_lmo_idx:
            return np.asarray(pair_lmo_idx[key])
        return np.arange(nocc)

    # Batched prange kernel: replaces the per-pair pool.map with one
    # Cython call that runs prange over all pairs in an isolated OpenMP
    # team (not sharing threads with the Python pool). BLAS forced to
    # 1 thread during the call to avoid oversubscription.
    #
    # Static gather (Qab, Qma, K_bar_*, e_pno, all shape/offset arrays,
    # scratch buffers) is cached on the function across CCSD iterations —
    # cc_ints is built once per CCSD run, so only T1 and output buffers
    # are rebuilt each cycle.
    from pyscf.cc.dlpno_tccsd._t1_fock_batched_cy import t1_fock_batched
    from threadpoolctl import threadpool_limits

    Fkj = F_lmo.copy()
    Fab_all = {}

    valid_keys = [k for k in keys if cc_ints.get(k) is not None]
    N = len(valid_keys)

    if N > 0:
        # Plan key — stable across CCSD iterations within one run.
        plan_key = (id(cc_ints), tuple(valid_keys),
                    id(pair_lmo_idx) if pair_lmo_idx is not None else 0)
        plan = getattr(t1_fock, '_batched_plan', None)
        if plan is None or plan.get('key') != plan_key:
            nlmo_arr = np.zeros(N, dtype=np.int32)
            npno_arr = np.zeros(N, dtype=np.int32)
            n_local_arr = np.zeros(N, dtype=np.int32)
            need_dji_arr = np.zeros(N, dtype=np.int32)

            K_chem_list = [None] * N
            K_ji_list = [None] * N
            K_ij_list = [None] * N
            Qma_list = [None] * N
            Qab_list = [None] * N
            e_pno_list = [None] * N
            lmo_idx_list = [None] * N

            for p, key in enumerate(valid_keys):
                ci = cc_ints[key]
                i, j = key
                npno = pno_spaces[key]['C_pno'].shape[1]
                lmo_idx = np.asarray(_pair_domain(key), dtype=np.intp)
                nlmo = lmo_idx.size
                n_local = ci['Qma'].shape[0]

                lmo_idx_list[p] = lmo_idx
                # K_bar_chem/ij/ji are reduced to (nlmo_p, npno); translate
                # global lmo_idx -> position within p_lmos before fancy-index.
                _lmo_idx_in_p = np.asarray(
                    ci['p_lmos_dense'])[lmo_idx].astype(np.intp)
                K_chem_list[p] = np.ascontiguousarray(
                    ci['K_bar_chem'][_lmo_idx_in_p])
                K_ji_list[p] = np.ascontiguousarray(
                    ci['K_bar_ji'][_lmo_idx_in_p])
                need_dji = (i != j)
                K_ij_list[p] = (np.ascontiguousarray(
                                ci['K_bar_ij'][_lmo_idx_in_p])
                                if need_dji else K_ji_list[p])
                Qma_list[p] = np.ascontiguousarray(
                    ci['Qma'][:, _lmo_idx_in_p, :])
                Qab_list[p] = np.ascontiguousarray(ci['Qab'])
                e_pno_list[p] = np.ascontiguousarray(pno_spaces[key]['e_pno'])

                nlmo_arr[p] = nlmo
                npno_arr[p] = npno
                n_local_arr[p] = n_local
                need_dji_arr[p] = int(need_dji)

            def _flat(arrs):
                sizes = np.array([a.size for a in arrs], dtype=np.int64)
                offsets = np.empty(len(arrs) + 1, dtype=np.int64)
                offsets[0] = 0
                offsets[1:] = np.cumsum(sizes)
                buf = np.empty(int(offsets[-1]))
                for idx, a in enumerate(arrs):
                    buf[offsets[idx]:offsets[idx + 1]] = a.ravel()
                return buf, offsets

            K_chem_flat, K_chem_off = _flat(K_chem_list)
            K_ji_flat, K_ji_off = _flat(K_ji_list)
            K_ij_flat, K_ij_off = _flat(K_ij_list)
            Qma_flat, Qma_off = _flat(Qma_list)
            Qab_flat, Qab_off = _flat(Qab_list)
            e_pno_flat, e_pno_off = _flat(e_pno_list)

            # T1 offsets (reused each cycle, buffer rebuilt below)
            T1_sizes = (nlmo_arr.astype(np.int64) * npno_arr.astype(np.int64))
            T1_off = np.empty(N + 1, dtype=np.int64)
            T1_off[0] = 0
            T1_off[1:] = np.cumsum(T1_sizes)
            T1_total = int(T1_off[-1])

            Fab_sizes = (npno_arr.astype(np.int64) ** 2)
            Fab_off = np.empty(N + 1, dtype=np.int64)
            Fab_off[0] = 0
            Fab_off[1:] = np.cumsum(Fab_sizes)
            Fab_total = int(Fab_off[-1])

            max_n_local = int(n_local_arr.max())
            max_nlmo = int(nlmo_arr.max())
            max_npno = int(npno_arr.max())
            num_threads = min(32, N)  # 64 threads makes scratch 1.3GB; 32 halves it with same speedup

            scratch = {
                'gamma': np.empty((num_threads, max_n_local)),
                'Y_trans': np.empty(
                    (num_threads, max_n_local * max_npno * max_nlmo)),
                'Y_alt': np.empty(
                    (num_threads, max_n_local * max_nlmo * max_npno)),
                'Fia': np.empty((num_threads, max_nlmo * max_npno)),
                'Z_stacked': np.empty(
                    (num_threads, max_n_local * max_nlmo * max_nlmo)),
                'Z_xxx': np.empty(
                    (num_threads, max_n_local * max_nlmo * max_nlmo)),
            }
            # First-touch scratch to pay the page-fault cost once.
            for buf in scratch.values():
                buf.fill(0.0)

            plan = {
                'key': plan_key,
                'valid_keys': valid_keys,
                'lmo_idx_list': lmo_idx_list,
                'nlmo_arr': nlmo_arr, 'npno_arr': npno_arr,
                'n_local_arr': n_local_arr, 'need_dji_arr': need_dji_arr,
                'K_chem_flat': K_chem_flat, 'K_chem_off': K_chem_off,
                'K_ji_flat': K_ji_flat, 'K_ji_off': K_ji_off,
                'K_ij_flat': K_ij_flat, 'K_ij_off': K_ij_off,
                'Qma_flat': Qma_flat, 'Qma_off': Qma_off,
                'Qab_flat': Qab_flat, 'Qab_off': Qab_off,
                'e_pno_flat': e_pno_flat, 'e_pno_off': e_pno_off,
                'T1_off': T1_off, 'T1_total': T1_total,
                'Fab_off': Fab_off, 'Fab_total': Fab_total,
                'scratch': scratch, 'num_threads': num_threads,
            }
            t1_fock._batched_plan = plan

        # Per-cycle: build T1_flat and output buffers.
        T1_flat = np.empty(plan['T1_total'])
        T1_off = plan['T1_off']
        for p, key in enumerate(plan['valid_keys']):
            lmo_idx = plan['lmo_idx_list'][p]
            T1_flat[T1_off[p]:T1_off[p + 1]] = t1_cache[key][lmo_idx].ravel()

        d_flat = np.zeros(N * 2)
        Fab_flat = np.zeros(plan['Fab_total'])
        sc = plan['scratch']

        with threadpool_limits(limits=1, user_api='blas'):
            if int(os.environ.get('DLPNO_C_CYCLE', '0')):
                # Native-C path: matches Psi4 ccsd.cc:1571 t1_fock Step 1
                # (d_ij/d_ji) + Step 2 (Fia/Fab dressing). See
                # pyscf/lib/cc/dlpno_t1_fock.c::DLPNOt1_fock_batched.
                # Same flat-buffer plan + per-thread scratch as Cython.
                import ctypes
                from pyscf import lib as _pyscflib
                _libcc = getattr(t1_fock, '_libcc', None)
                if _libcc is None:
                    _libcc = _pyscflib.load_library('libcc')
                    _libcc.DLPNOt1_fock_batched.restype = None
                    _libcc.DLPNOt1_fock_batched.argtypes = (
                        [ctypes.c_void_p] * 18 +              # 7 ptr/off pairs + 4 shape arrays
                        [ctypes.c_void_p] +                    # is_strong_pair (nullable)
                        [ctypes.c_void_p, ctypes.c_size_t] * 6 +  # 6 scratch (ptr + stride)
                        [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # d, Fab, Fab_off
                         ctypes.c_size_t, ctypes.c_int])
                    t1_fock._libcc = _libcc
                _libcc.DLPNOt1_fock_batched(
                    T1_flat.ctypes.data_as(ctypes.c_void_p),
                    T1_off.ctypes.data_as(ctypes.c_void_p),
                    plan['K_chem_flat'].ctypes.data_as(ctypes.c_void_p),
                    plan['K_chem_off'].ctypes.data_as(ctypes.c_void_p),
                    plan['K_ji_flat'].ctypes.data_as(ctypes.c_void_p),
                    plan['K_ji_off'].ctypes.data_as(ctypes.c_void_p),
                    plan['K_ij_flat'].ctypes.data_as(ctypes.c_void_p),
                    plan['K_ij_off'].ctypes.data_as(ctypes.c_void_p),
                    plan['Qma_flat'].ctypes.data_as(ctypes.c_void_p),
                    plan['Qma_off'].ctypes.data_as(ctypes.c_void_p),
                    plan['Qab_flat'].ctypes.data_as(ctypes.c_void_p),
                    plan['Qab_off'].ctypes.data_as(ctypes.c_void_p),
                    plan['e_pno_flat'].ctypes.data_as(ctypes.c_void_p),
                    plan['e_pno_off'].ctypes.data_as(ctypes.c_void_p),
                    plan['nlmo_arr'].ctypes.data_as(ctypes.c_void_p),
                    plan['npno_arr'].ctypes.data_as(ctypes.c_void_p),
                    plan['n_local_arr'].ctypes.data_as(ctypes.c_void_p),
                    plan['need_dji_arr'].ctypes.data_as(ctypes.c_void_p),
                    None,  # is_strong_pair: baseline iterates valid_keys = strong+diag, no need to skip
                    sc['gamma'].ctypes.data_as(ctypes.c_void_p), sc['gamma'].shape[1],
                    sc['Y_trans'].ctypes.data_as(ctypes.c_void_p), sc['Y_trans'].shape[1],
                    sc['Y_alt'].ctypes.data_as(ctypes.c_void_p), sc['Y_alt'].shape[1],
                    sc['Fia'].ctypes.data_as(ctypes.c_void_p), sc['Fia'].shape[1],
                    sc['Z_stacked'].ctypes.data_as(ctypes.c_void_p), sc['Z_stacked'].shape[1],
                    sc['Z_xxx'].ctypes.data_as(ctypes.c_void_p), sc['Z_xxx'].shape[1],
                    d_flat.ctypes.data_as(ctypes.c_void_p),
                    Fab_flat.ctypes.data_as(ctypes.c_void_p),
                    plan['Fab_off'].ctypes.data_as(ctypes.c_void_p),
                    N, plan['num_threads'],
                )
            else:
                t1_fock_batched(
                    T1_flat, T1_off,
                    plan['K_chem_flat'], plan['K_chem_off'],
                    plan['K_ji_flat'], plan['K_ji_off'],
                    plan['K_ij_flat'], plan['K_ij_off'],
                    plan['Qma_flat'], plan['Qma_off'],
                    plan['Qab_flat'], plan['Qab_off'],
                    plan['e_pno_flat'], plan['e_pno_off'],
                    plan['nlmo_arr'], plan['npno_arr'],
                    plan['n_local_arr'], plan['need_dji_arr'],
                    sc['gamma'], sc['Y_trans'], sc['Y_alt'],
                    sc['Fia'], sc['Z_stacked'], sc['Z_xxx'],
                    d_flat, Fab_flat, plan['Fab_off'],
                    plan['num_threads'],
                )

        # Scatter outputs
        Fab_off_plan = plan['Fab_off']
        npno_arr_plan = plan['npno_arr']
        for p, key in enumerate(plan['valid_keys']):
            i, j = key
            npno = int(npno_arr_plan[p])
            Fkj[i, j] += d_flat[p * 2]
            if i != j:
                Fkj[j, i] += d_flat[p * 2 + 1]
            Fab_all[key] = (
                Fab_flat[Fab_off_plan[p]:Fab_off_plan[p + 1]]
                .reshape(npno, npno).copy())

    # Snapshot Fij_bar (T1-dressed Fock minus Eq 94's Fia_bar_jj term)
    # — shared with the T1 residual so it doesn't recompute the same per-pair
    # d_ij / d_ji additions. The T1 residual augments with weak-pair
    # contributions before use; Eq 94 stays exclusive to t1_fock's Fkj since
    # the T1 residual builds Fia_bar_ii separately in Stage 3.
    Fij_bar_snapshot = Fkj.copy()

    # Eq 94: Fkj += Σ_a Fia_bar_jj · t1_j — use local LMO domain of (jj, jj)
    for j_idx in range(nocc):
        key_jj = (j_idx, j_idx)
        ci = cc_ints.get(key_jj)
        if ci is None:
            continue
        t1_j = t1_pno.get(j_idx)
        if t1_j is None or t1_j.size == 0:
            continue
        npno = pno_spaces[key_jj]['C_pno'].shape[1]
        lmo_idx = _pair_domain(key_jj)
        # Phase 1: fancy-index the cached matrix.
        T1_local = np.ascontiguousarray(
            t1_cache[key_jj][np.asarray(lmo_idx, dtype=np.intp)])
        # Qma reduced; translate global lmo_idx -> p_lmos position.
        _lmo_idx_in_p = np.asarray(
            ci['p_lmos_dense'])[lmo_idx].astype(np.intp)
        Qma_jj = ci['Qma'][:, _lmo_idx_in_p, :]   # (n_local, nlmo, npno)
        gamma = Qma_jj.reshape(Qma_jj.shape[0], -1) @ T1_local.ravel()
        Fia_bar_jj = 2.0 * np.tensordot(gamma, Qma_jj, axes=(0, 0))
        Z_jj = T1_local @ Qma_jj.transpose(0, 2, 1)
        Fia_bar_jj -= np.tensordot(Z_jj, Qma_jj, axes=((0, 1), (0, 1)))
        Fkj[lmo_idx, j_idx] += Fia_bar_jj @ t1_j

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
            T1_all2 = t1_cache[key_jj2]
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

    # FKJ_DUMP: parity dump vs Psi4 ccsd.cc t1_fock Fkj_ output.
    if int(os.environ.get('DLPNO_DUMP_FKJ', '0')):
        _it = getattr(t1_fock, '_iter', 0)
        t1_fock._iter = _it + 1
        if _it <= 2:
            rms = float(np.sqrt((Fkj ** 2).mean()))
            sm = float(Fkj.sum())
            tr = float(np.trace(Fkj))
            fro = float(np.linalg.norm(Fkj, 'fro'))
            off = Fkj - np.diag(np.diag(Fkj))
            off_fro = float(np.linalg.norm(off, 'fro'))
            print(f"FKJ_DUMP iter={_it} nocc={Fkj.shape[0]} "
                  f"rms={rms:.12e} sum={sm:.12e} tr={tr:.12e} "
                  f"fro={fro:.12e} off_fro={off_fro:.12e}",
                  flush=True)

    return Fkj, Fab_all, foo_t1, Fij_bar_snapshot


def compute_B_tilde(cc_ints, dressed_ints, t2_pno_all, t1_pno,
                    pno_spaces, S_pno_cache, key, nocc,
                    pair_lmo_idx=None, t1_cache=None):
    """Build B_tilde for pair (ij) matching Psi4's precomputed B_tilde.

    B_tilde[k_ij, l_ij] = (ki|lj)_dressed + Σ_{a,b} tau[a,b] * (ka|lb)

    The k, l sums are restricted to pair_lmo_idx[key] (Psi4's
    lmopair_to_lmos_[ij]) when provided. Returns a tuple

        (B_local, p_lmos_dense)

    matching Psi4's per-pair (nlmo_ij, nlmo_ij) storage. ``B_local`` is the
    pair-domain matrix; ``p_lmos_dense`` is a (nocc,) int array mapping a
    global LMO index k → its pair-domain position k_ij (or -1 if k is not
    in the pair domain). Consumers do ``B_local[p_lmos_dense[k],
    p_lmos_dense[l]]``. Returns ``None`` if the pair has no integrals.
    """
    ci = cc_ints.get(key)
    if ci is None:
        return None

    i, j = key
    npno = pno_spaces[key]['C_pno'].shape[1]

    if pair_lmo_idx is not None and key in pair_lmo_idx:
        lmo_idx = np.asarray(pair_lmo_idx[key])
    else:
        lmo_idx = np.arange(nocc)
    nlmo = len(lmo_idx)

    # Qma reduced to (n_local, nlmo_p, npno); translate global lmo_idx
    # -> p_lmos position. Reused by fallback Qk slicing below.
    _lmo_idx_in_p = np.asarray(
        ci['p_lmos_dense'])[lmo_idx].astype(np.intp)
    Qma = ci['Qma'][:, _lmo_idx_in_p, :]     # (n_local, nlmo, npno)

    # Read T1-dressed Qk from precomputed dressed_ints if provided
    # (matches Psi4 ccsd.cc:1688 which reads from i_Qk_t1_/j_Qk_t1_
    # populated by t1_ints()). Falls back to inline if absent.
    if (dressed_ints is not None and key in dressed_ints
            and 'i_Qk_t1' in dressed_ints[key]):
        i_Qk_t1 = dressed_ints[key]['i_Qk_t1']     # (n_local, nlmo)
        j_Qk_t1 = dressed_ints[key]['j_Qk_t1']
    else:
        i_Qk = ci['i_Qk'][:, _lmo_idx_in_p]      # (n_local, nlmo)
        j_Qk = ci['j_Qk'][:, _lmo_idx_in_p]
        if t1_cache is not None:
            t1_i = t1_cache[key][i]
            t1_j = t1_cache[key][j]
        elif t1_pno is not None:
            from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair
            t1_i = _project_t1_to_pair(
                t1_pno, i, key, S_pno_cache, pno_spaces)
            t1_j = _project_t1_to_pair(
                t1_pno, j, key, S_pno_cache, pno_spaces)
        else:
            t1_i = np.zeros(npno)
            t1_j = np.zeros(npno)
        i_Qk_t1 = i_Qk.copy()
        i_Qk_t1 += np.einsum('Qka,a->Qk', Qma, t1_i)
        j_Qk_t1 = j_Qk.copy()
        j_Qk_t1 += np.einsum('Qka,a->Qk', Qma, t1_j)

    T2_ij = t2_pno_all[key]

    if int(os.environ.get('DLPNO_C_CYCLE', '0')) and nlmo > 0 and npno > 0:
        # Native-C path: matches Psi4 ccsd.cc:1742 compute_B_tilde.
        # See pyscf/lib/cc/dlpno_b_tilde.c. Validation gates:
        # water-4 E_TCCSD(T)=-304.98979787, water-10 E_TCCSD=-2.13088299002.
        import ctypes
        from pyscf import lib as _pyscflib
        _libcc = getattr(compute_B_tilde, '_libcc', None)
        if _libcc is None:
            _libcc = _pyscflib.load_library('libcc')
            _libcc.DLPNOcompute_B_tilde_pair.restype = None
            _libcc.DLPNOcompute_B_tilde_pair.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
            ]
            compute_B_tilde._libcc = _libcc
        i_Qk_c = np.ascontiguousarray(i_Qk_t1)
        j_Qk_c = np.ascontiguousarray(j_Qk_t1)
        Qma_c = np.ascontiguousarray(Qma)
        T2_c = np.ascontiguousarray(T2_ij)
        n_local_c = i_Qk_c.shape[0]
        B_local = np.empty((nlmo, nlmo))
        _libcc.DLPNOcompute_B_tilde_pair(
            B_local.ctypes.data_as(ctypes.c_void_p),
            i_Qk_c.ctypes.data_as(ctypes.c_void_p),
            j_Qk_c.ctypes.data_as(ctypes.c_void_p),
            Qma_c.ctypes.data_as(ctypes.c_void_p),
            T2_c.ctypes.data_as(ctypes.c_void_p),
            n_local_c, nlmo, npno,
        )
    else:
        B_local = i_Qk_t1.T @ j_Qk_t1            # (nlmo, nlmo)

        # voov dressing (bare T2 per Psi4)
        P = np.einsum('ab,Qka->kbQ', T2_ij, Qma)     # (nlmo, npno, n_local)
        B_local += np.einsum('kbQ,Qlb->kl', P, Qma)

    # BTILDE_DUMP: parity dump vs Psi4 ccsd.cc compute_B_tilde.
    # Track per-key call count: first time we see (cc_ints, key) is iter 0.
    if int(os.environ.get('DLPNO_DUMP_BTILDE', '0')):
        _counts = getattr(compute_B_tilde, '_counts', None)
        if _counts is None:
            _counts = {}
            compute_B_tilde._counts = _counts
        counts_key = (id(cc_ints), key)
        _it = _counts.get(counts_key, 0)
        _counts[counts_key] = _it + 1
        if _it <= 2:
            i, j = key
            rms = float(np.sqrt((B_local ** 2).mean())) if B_local.size else 0.0
            sm = float(B_local.sum())
            li = ','.join(str(int(x)) for x in lmo_idx)
            flat = ','.join(f'{v:.12e}' for v in B_local.ravel())
            print(f"BTILDE_DUMP iter={_it} pair=({i},{j}) nlmo={nlmo} "
                  f"rms={rms:.12e} sum={sm:.12e} lmo_idx=[{li}] B=[{flat}]",
                  flush=True)

    # Per-pair Psi4 layout: return reduced (nlmo, nlmo) plus a global→pair-domain
    # map so consumers do B_local[p_dense[k], p_dense[l]] without scattering.
    p_lmos_dense = np.full(nocc, -1, dtype=np.intp)
    p_lmos_dense[lmo_idx] = np.arange(nlmo, dtype=np.intp)
    return B_local, p_lmos_dense


def compute_ladder(cc_ints, t2_pno_all, t1_pno, pno_spaces,
                   S_pno_cache, key, nocc, pair_lmo_idx=None,
                   t1_cache=None):
    """Compute ladder term A for pair (ij) matching Psi4 Term A.

    A[a,b] = Σ_Q Qab_t1[Q,a,c] * T2[c,d] * Qab_t1[Q,b,d]

    The `k` (LMO) sum inside Qab_t1 is restricted to the pair's local LMO
    domain (lmopair_to_lmos_[ij] in Psi4) when pair_lmo_idx is provided.
    This turns a per-pair cost of O(n_local · nocc · npno²) into
    O(n_local · nlmo_ij · npno²), restoring DLPNO linear-scaling.
    """
    ci = cc_ints.get(key)
    if ci is None:
        return np.zeros((0, 0))

    i, j = key
    npno = pno_spaces[key]['C_pno'].shape[1]
    Qab = ci['Qab']
    Qma_full = ci['Qma']   # (n_local, nlmo_p, npno) reduced

    if pair_lmo_idx is not None and key in pair_lmo_idx:
        lmo_idx = np.asarray(pair_lmo_idx[key])
    else:
        lmo_idx = np.arange(nocc)
    # Translate global lmo_idx -> position within p_lmos.
    _lmo_idx_in_p = np.asarray(
        ci['p_lmos_dense'])[lmo_idx].astype(np.intp)
    Qma = Qma_full[:, _lmo_idx_in_p, :]      # (n_local, nlmo, npno)

    # Phase 1: use cache if given; otherwise build or fall back to zero.
    if t1_cache is not None:
        T1_local = np.ascontiguousarray(
            t1_cache[key][np.asarray(lmo_idx, dtype=np.intp)])
    elif t1_pno is not None:
        from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair
        T1_local = np.zeros((len(lmo_idx), npno))
        for ki, k in enumerate(lmo_idx):
            T1_local[ki] = _project_t1_to_pair(
                t1_pno, int(k), key, S_pno_cache, pno_spaces)
    else:
        T1_local = np.zeros((len(lmo_idx), npno))

    T2_ij = t2_pno_all[key]
    # Qab_t1[Q,a,b] = Qab[Q,a,b] - Σ_{k ∈ lmo_idx} T1_local[k,a] * Qma[Q,k,b]
    Qab_t1 = Qab - (T1_local.T @ Qma)
    X = Qab_t1 @ T2_ij                        # (Q, npno, npno)
    A_local = np.tensordot(X, Qab_t1, axes=((0, 2), (0, 2)))

    # LADDER_DUMP: parity dump vs Psi4 ccsd.cc:2317. Track per-key call count.
    if int(os.environ.get('DLPNO_DUMP_LADDER', '0')):
        _counts = getattr(compute_ladder, '_counts', None)
        if _counts is None:
            _counts = {}
            compute_ladder._counts = _counts
        counts_key = (id(cc_ints), key)
        _it = _counts.get(counts_key, 0)
        _counts[counts_key] = _it + 1
        if _it <= 2 and A_local.size:
            i, j = key
            rms = float(np.sqrt((A_local ** 2).mean()))
            sm = float(A_local.sum())
            fro = float(np.linalg.norm(A_local, 'fro'))
            tr = float(np.trace(A_local))
            print(f"LADDER_DUMP iter={_it} pair=({i},{j}) npno={npno} "
                  f"rms={rms:.12e} sum={sm:.12e} fro={fro:.12e} tr={tr:.12e}",
                  flush=True)

    return A_local
