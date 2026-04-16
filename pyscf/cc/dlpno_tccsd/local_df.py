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

import numpy as np


def build_screening_maps(mol, auxmol, C_lmo, pao_domains, s1e, strong_pair_keys,
                         T_CUT_MKN=1e-3, T_CUT_CLMO=1e-3):
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

    # --- lmo_to_atoms: atoms where C_lmo has significant coefficient ---
    lmo_to_atoms = []
    for i in range(nocc):
        has_coeff = np.zeros(natm, dtype=bool)
        for a in range(natm):
            m = atom_ids == a
            if np.any(np.abs(C_lmo[m, i]) > T_CUT_CLMO):
                has_coeff[a] = True
        lmo_to_atoms.append(np.where(has_coeff)[0])

    # --- lmo_to_paoatoms: atoms hosting each LMO's PAO domain ---
    lmo_to_paoatoms = []
    for i in range(nocc):
        if len(pao_domains[i]) == 0:
            lmo_to_paoatoms.append(np.zeros(0, dtype=int))
        else:
            lmo_to_paoatoms.append(np.unique(atom_ids[pao_domains[i]]))

    # --- lmo_to_riatoms: aux atoms via Mulliken population ---
    lmo_to_riatoms = []
    for i in range(nocc):
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
        lmo_to_riatoms.append(np.where(np.abs(pop) > T_CUT_MKN)[0])

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
    c_refit = []
    for a in range(natm):
        bfs1 = riatom_to_bfs1[a]
        lmos = riatom_to_lmos_ext[a]
        if len(bfs1) == 0 or len(lmos) == 0:
            c_refit.append(np.zeros((len(bfs1), len(lmos))))
            continue
        S_aa = s1e[np.ix_(bfs1, bfs1)]
        rhs = SC_lmo[bfs1][:, lmos]
        c_refit.append(np.linalg.solve(S_aa, rhs))

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


def build_sparse_df_arrays(mol, auxmol, C_lmo, C_pao, maps):
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

    # Iterate aux SHELLS (each shell's aux functions share centerQ and thus bfs1/bfs2)
    for Q_sh in range(auxmol.nbas):
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
            for q in range(nq):
                qij[q_start + q] = np.zeros((len(lmos_ext), len(lmos_ext)))
                qia[q_start + q] = np.zeros((len(lmos_ext), len(paos_ext)))
                qab[q_start + q] = np.zeros((len(paos_ext), len(paos_ext)))
            continue

        # Build shell slices for atoms1, atoms2
        sh1_list = np.concatenate([atom_to_ao_sh[a] for a in atoms1])
        sh2_list = np.concatenate([atom_to_ao_sh[a] for a in atoms2])
        # Shell ranges must be contiguous for intor slicing — atoms are sorted,
        # but shells of each atom ARE contiguous within the atom's shell range.
        # Use the min/max shell and extract the desired subset afterward.
        sh1_min, sh1_max = sh1_list.min(), sh1_list.max() + 1
        sh2_min, sh2_max = sh2_list.min(), sh2_list.max() + 1

        # Intor over this slice: (μν | Q_sh) for μ∈[sh1_min..sh1_max), ν∈[sh2_min..sh2_max)
        buf = pmol.intor(
            'int3c2e',
            shls_slice=(sh1_min, sh1_max, sh2_min, sh2_max,
                        mol.nbas + Q_sh, mol.nbas + Q_sh + 1))
        # buf shape (nao1_range, nao2_range, nq)

        # Need AO indices within sh1/sh2 ranges that land in bfs1/bfs2
        bf1_start, bf1_end = ao_shell_loc[sh1_min], ao_shell_loc[sh1_max]
        bf2_start, bf2_end = ao_shell_loc[sh2_min], ao_shell_loc[sh2_max]
        rel_bfs1 = bfs1 - bf1_start  # indices into buf
        rel_bfs2 = bfs2 - bf2_start
        # Sanity: all must be in [0, bf_end - bf_start)
        mn_block = buf[rel_bfs1][:, rel_bfs2, :]  # (|bfs1|, |bfs2|, nq)

        # Also compute (μν' | Q) with μ,ν' ∈ bfs1 (for qij). Reuse intor if
        # bfs1==bfs2 or sub-slice. For simplicity compute separately:
        sh1a_min, sh1a_max = sh1_min, sh1_max
        if atoms1.tolist() == atoms2.tolist():
            mn1_block = mn_block  # (|bfs1|, |bfs1|, nq) = same
        else:
            buf1 = pmol.intor(
                'int3c2e',
                shls_slice=(sh1_min, sh1_max, sh1_min, sh1_max,
                            mol.nbas + Q_sh, mol.nbas + Q_sh + 1))
            mn1_block = buf1[rel_bfs1][:, rel_bfs1, :]

        # qab needs (μν | Q) with μ, ν ∈ bfs2
        if atoms1.tolist() == atoms2.tolist():
            mn2_block = mn_block
        else:
            buf2 = pmol.intor(
                'int3c2e',
                shls_slice=(sh2_min, sh2_max, sh2_min, sh2_max,
                            mol.nbas + Q_sh, mol.nbas + Q_sh + 1))
            mn2_block = buf2[rel_bfs2][:, rel_bfs2, :]

        # Transform
        # C_pao_slice: (|bfs2|, |paos_ext|) — C_pao restricted to bfs2 rows, paos_ext cols
        C_pao_slice = C_pao[np.ix_(bfs2, paos_ext)]

        for qi in range(nq):
            Q = q_start + qi
            # qij[Q][i, j] = Σ_{μν} C_refit[μ, i] (μν|Q)_{bfs1 bfs1} C_refit[ν, j]
            qij[Q] = C_r.T @ mn1_block[:, :, qi] @ C_r
            # qia[Q][i, u] = Σ_{μν} C_refit[μ, i] (μν|Q)_{bfs1 bfs2} C_pao_slice[ν, u]
            qia[Q] = C_r.T @ mn_block[:, :, qi] @ C_pao_slice
            # qab[Q][u, v] = Σ_{μν} C_pao_slice[μ, u] (μν|Q)_{bfs2 bfs2} C_pao_slice[ν, v]
            qab[Q] = C_pao_slice.T @ mn2_block[:, :, qi] @ C_pao_slice

    return {'qij': qij, 'qia': qia, 'qab': qab}


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
        Qma = ci['Qma']  # (n_local, nocc, npno)
        # ovL_lmo[Q, a] = Qma[Q, lmo, a]; K[a,b] = Σ_Q ovL1[Q,a]*ovL2[Q,b]
        K = Qma[:, lmo1, :].T @ Qma[:, lmo2, :]
        cache[k] = K
    return K


def compute_cc_integrals(mol, auxmol, C_lmo, pno_spaces, pair_aux_idx,
                         j2c, keys, nocc, aux_block_size=None):
    """Precompute ALL per-pair locally-fitted intermediates (aux-blocked).

    Matches Psi4 ccsd.cc compute_cc_integrals() — processes auxiliary shells
    in blocks, avoiding O(nao² × naux) raw_3c and O(nao × naux × npno × N_pairs)
    tmp_pno_cache that would be prohibitive for large systems.

    Memory scaling:
      - Transient per-block: O(nao² × n_aux_block) + Σ_pairs O(nao × n_aux_block × npno)
      - Per-pair persistent: O(n_aux_pair × npno²) × N_pairs (output tensors only)

    Args:
        mol, auxmol: PySCF Mole objects.
        C_lmo: (nao, nocc) LMO coefficients.
        pno_spaces: dict pair_key -> {'C_pno': (nao, npno)}.
        pair_aux_idx: dict pair_key -> np.array of local aux indices.
        j2c: (naux, naux) full Coulomb metric.
        keys: list of pair keys.
        nocc: number of active occupied orbitals.
        aux_block_size: aux shells per block (auto-chosen if None).

    Returns:
        cc_ints: dict pair_key -> dict of fitted intermediates (same schema
        as Psi4-matched original implementation).
    """
    nao = mol.nao_nr()
    naux = auxmol.nao_nr()
    pmol = mol + auxmol
    aux_nbas = auxmol.nbas
    aux_ao_loc = auxmol.ao_loc_nr()

    # Auto-tune block size: aim for ~0.5 GB transient raw_block
    if aux_block_size is None:
        target_bytes = 512 * 1024 * 1024
        bytes_per_shell = nao * nao * 8 * 5  # p shell ~5 funcs upper bound
        aux_block_size = max(1, min(aux_nbas, target_bytes // max(bytes_per_shell, 1)))

    # Initialize per-pair accumulators indexed by pair's aux_idx (small).
    pair_data = {}
    key_set = set(keys)
    for key in keys:
        if key not in pair_aux_idx:
            continue
        i, j = key
        C_pno = pno_spaces[key]['C_pno']
        npno = C_pno.shape[1]
        if npno == 0:
            continue
        aux_idx = pair_aux_idx[key]
        n_local = len(aux_idx)
        aux_inv = -np.ones(naux, dtype=np.int64)
        aux_inv[aux_idx] = np.arange(n_local)  # global Q → local-pair Q position
        # Cross-pair sets: kj and ki partners that are also keys
        kj_partners = []
        ki_partners = []
        for k in range(nocc):
            key_kj = (min(k, j), max(k, j))
            if key_kj in key_set and key_kj in pno_spaces \
                    and pno_spaces[key_kj]['C_pno'].shape[1] > 0:
                kj_partners.append((k, key_kj, pno_spaces[key_kj]['C_pno'].shape[1]))
            key_ki = (min(k, i), max(k, i))
            if key_ki in key_set and key_ki in pno_spaces \
                    and pno_spaces[key_ki]['C_pno'].shape[1] > 0:
                ki_partners.append((k, key_ki, pno_spaces[key_ki]['C_pno'].shape[1]))
        pair_data[key] = {
            'C_pno': C_pno, 'npno': npno, 'aux_idx': aux_idx,
            'n_local': n_local, 'aux_inv': aux_inv,
            # Raw (pre-fit) accumulators. Layout chosen to match tensordot
            # output so block-updates use basic slicing (no strided copies).
            #   raw_iv, raw_jv: (n_local, npno)
            #   raw_ma:         (nocc, n_local, npno)
            #   raw_ab:         (npno, n_local, npno)
            'raw_iv': np.zeros((n_local, npno)),
            'raw_jv': np.zeros((n_local, npno)),
            'raw_ma': np.zeros((nocc, n_local, npno)),
            'raw_ab': np.zeros((npno, n_local, npno)),
            'raw_cross_kj': {k: np.zeros((npno, n_local, n_kj))
                             for k, _, n_kj in kj_partners},
            'raw_cross_ji': {k: np.zeros((npno, n_local, n_ki))
                             for k, _, n_ki in ki_partners},
            'raw_kv_kj': {k: np.zeros((n_local, n_kj))
                          for k, _, n_kj in kj_partners},
            'raw_kv_ki': {k: np.zeros((n_local, n_ki))
                          for k, _, n_ki in ki_partners},
            'kj_partners': kj_partners,
            'ki_partners': ki_partners,
        }

    # raw_oo full (nocc, nocc, naux) — needs all aux for per-pair slicing
    raw_oo = np.zeros((nocc, nocc, naux))

    # =============================================================
    # Aux-blocked main loop
    # =============================================================
    for sh_start in range(0, aux_nbas, aux_block_size):
        sh_end = min(sh_start + aux_block_size, aux_nbas)
        Q0 = aux_ao_loc[sh_start]
        Q1 = aux_ao_loc[sh_end]

        # (nao, nao, nQ) block
        raw_block = pmol.intor(
            'int3c2e',
            shls_slice=(0, mol.nbas, 0, mol.nbas,
                        mol.nbas + sh_start, mol.nbas + sh_end))

        # raw_oo[k, l, Q∈block]
        tmp_oo = np.tensordot(raw_block, C_lmo, axes=([1], [0]))  # (nao, nQ, nocc)
        raw_oo[:, :, Q0:Q1] = np.tensordot(
            C_lmo, tmp_oo, axes=([0], [0])).transpose(0, 2, 1)
        del tmp_oo

        # Cache half-transformed tmp_pno per pair within this block.
        # Populated lazily and shared between outer pair + cross-pair users.
        tmp_pno_blocks = {}
        Q_range = np.arange(Q0, Q1)

        def _tmp_pno(key_):
            if key_ not in tmp_pno_blocks:
                tmp_pno_blocks[key_] = np.tensordot(
                    raw_block, pno_spaces[key_]['C_pno'], axes=([1], [0]))
            return tmp_pno_blocks[key_]

        for key, pd in pair_data.items():
            aux_inv = pd['aux_inv']
            Q_to_local = aux_inv[Q_range]
            mask = Q_to_local >= 0
            if not mask.any():
                # Still may need tmp_pno_blocks[key] for OTHER pairs' cross use
                continue
            block_Q = np.where(mask)[0]
            local_Q = Q_to_local[mask]

            tmp = _tmp_pno(key)
            tmp_sub = tmp[:, block_Q, :]
            i, j = key

            pd['raw_iv'][local_Q] = np.tensordot(
                C_lmo[:, i], tmp_sub, axes=([0], [0]))
            pd['raw_jv'][local_Q] = np.tensordot(
                C_lmo[:, j], tmp_sub, axes=([0], [0]))
            # raw_ma[m, local_Q, a] = Σ_μ C_lmo[μ, m] * tmp_sub[μ, Q, a]
            pd['raw_ma'][:, local_Q, :] = np.tensordot(
                C_lmo, tmp_sub, axes=([0], [0]))
            # raw_ab[a, local_Q, b] = Σ_μ C_pno[μ, a] * tmp_sub[μ, Q, b]
            pd['raw_ab'][:, local_Q, :] = np.tensordot(
                pd['C_pno'], tmp_sub, axes=([0], [0]))

            # Cross-pair
            for k, key_kj, _ in pd['kj_partners']:
                tmp_kj_sub = _tmp_pno(key_kj)[:, block_Q, :]
                pd['raw_cross_kj'][k][:, local_Q, :] = np.tensordot(
                    pd['C_pno'], tmp_kj_sub, axes=([0], [0]))
                pd['raw_kv_kj'][k][local_Q] = np.tensordot(
                    C_lmo[:, k], tmp_kj_sub, axes=([0], [0]))
            for k, key_ki, _ in pd['ki_partners']:
                tmp_ki_sub = _tmp_pno(key_ki)[:, block_Q, :]
                pd['raw_cross_ji'][k][:, local_Q, :] = np.tensordot(
                    pd['C_pno'], tmp_ki_sub, axes=([0], [0]))
                pd['raw_kv_ki'][k][local_Q] = np.tensordot(
                    C_lmo[:, k], tmp_ki_sub, axes=([0], [0]))

        del tmp_pno_blocks, raw_block

    # =============================================================
    # Apply local J^{-1/2} per pair and build final outputs
    # =============================================================
    cc_ints = {}
    for key in keys:
        i, j = key
        if key not in pair_data:
            cc_ints[key] = None
            continue
        pd = pair_data[key]
        npno = pd['npno']
        aux_idx = pd['aux_idx']
        n_local = pd['n_local']
        C_pno = pd['C_pno']

        # Local J^{-1/2}
        j2c_local = j2c[np.ix_(aux_idx, aux_idx)]
        eigvals, eigvecs = np.linalg.eigh(j2c_local)
        keep = eigvals > 1e-14
        jhi = (eigvecs[:, keep] * (1.0 / np.sqrt(eigvals[keep]))) @ eigvecs[:, keep].T

        # Fitted 3-index quantities
        q_iv = jhi @ pd['raw_iv']                    # (n_local, npno)
        q_jv = jhi @ pd['raw_jv']
        q_io = jhi @ raw_oo[:, i, :][:, aux_idx].T   # (n_local, nocc)
        q_jo = jhi @ raw_oo[:, j, :][:, aux_idx].T
        q_pair = jhi @ raw_oo[i, j, aux_idx]         # (n_local,)

        Qma = np.zeros((n_local, nocc, npno))
        raw_ma = pd['raw_ma']          # (nocc, n_local, npno)
        for m in range(nocc):
            Qma[:, m, :] = jhi @ raw_ma[m]
        Qab = np.zeros((n_local, npno, npno))
        raw_ab = pd['raw_ab']          # (npno, n_local, npno)
        for a in range(npno):
            Qab[:, a, :] = jhi @ raw_ab[a]

        # 2-index intermediates
        K_iajb = q_iv.T @ q_jv
        K_mnij = q_io.T @ q_jo
        K_bar_ij = q_io.T @ q_jv
        K_bar_ji = q_jo.T @ q_iv
        K_bar_chem = np.tensordot(q_pair, Qma, axes=(0, 0))
        J_ijab = np.tensordot(q_pair, Qab, axes=(0, 0))

        # Cross-pair J/K integrals
        J_ij_kj = {}
        K_ij_kj_dict = {}
        for k, key_kj, n_kj in pd['kj_partners']:
            cross_fitted = np.zeros((n_local, npno, n_kj))
            rc = pd['raw_cross_kj'][k]  # (npno, n_local, n_kj)
            for a in range(npno):
                cross_fitted[:, a, :] = jhi @ rc[a]
            q_ik = jhi @ raw_oo[i, k, aux_idx]
            J_ij_kj[(key, k)] = np.tensordot(q_ik, cross_fitted, axes=(0, 0))
            q_kv_kj = jhi @ pd['raw_kv_kj'][k]
            K_ij_kj_dict[(key, k)] = q_iv.T @ q_kv_kj

        J_ji_ki = {}
        K_ji_ki_dict = {}
        for k, key_ki, n_ki in pd['ki_partners']:
            cross_fitted = np.zeros((n_local, npno, n_ki))
            rc = pd['raw_cross_ji'][k]  # (npno, n_local, n_ki)
            for a in range(npno):
                cross_fitted[:, a, :] = jhi @ rc[a]
            q_jk = jhi @ raw_oo[j, k, aux_idx]
            J_ji_ki[(key, k)] = np.tensordot(q_jk, cross_fitted, axes=(0, 0))
            q_kv_ki = jhi @ pd['raw_kv_ki'][k]
            K_ji_ki_dict[(key, k)] = q_jv.T @ q_kv_ki

        cc_ints[key] = {
            'K_iajb': K_iajb,
            'K_mnij': K_mnij,
            'K_bar_ij': K_bar_ij,
            'K_bar_ji': K_bar_ji,
            'K_bar_chem': K_bar_chem,
            'J_ijab': J_ijab,
            'J_ij_kj': J_ij_kj,
            'K_ij_kj': K_ij_kj_dict,
            'J_ji_ki': J_ji_ki,
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
        gamma = Qma.reshape(Qma.shape[0], -1) @ T1_all.ravel()  # (n_local,)

        # J contribution: 2 * Σ_Q gamma[Q] * Qab[Q,a,b]
        Fab += 2.0 * np.tensordot(gamma, Qab, axes=(0, 0))

        # K contribution: -Σ_Q (Qab[Q] @ T1_all^T) @ Qma[Q]
        # Y[Q,a,n] = Σ_b Qab[Q,a,b] * T1_all[n,b]
        Y = Qab @ T1_all.T  # (n_local, npno, nocc)
        # Fab[a,c] -= Σ_{Q,n} Y[Q,a,n] * Qma[Q,n,c]
        Fab -= np.tensordot(Y, Qma, axes=((0, 2), (0, 1)))

        # Psi4 ccsd.cc line 1639-1640:
        #   Fab_[ij] = Fab_bar[ij] - T_n.T @ Fia_bar[ij]
        # Fia_bar[k,a] = 2*gamma*Qma[k,a] - Qma@T_n.T@Qma  (lines 1602-1628)
        Fia_bar = 2.0 * np.tensordot(gamma, Qma, axes=(0, 0))
        # Z[Q,n,k] = Σ_b T1_all[n,b] * Qma[Q,k,b]
        Z = T1_all @ Qma.transpose(0, 2, 1)  # (n_local, nocc, nocc→k)
        # Fia_bar[k,a] -= Σ_{Q,n} Qma[Q,n,a] * Z[Q,n,k]
        Fia_bar -= np.tensordot(Z, Qma, axes=((0, 1), (0, 1)))

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
        Qma_jj = ci['Qma']
        gamma = Qma_jj.reshape(Qma_jj.shape[0], -1) @ T1_all.ravel()
        Fia_bar_jj = 2.0 * np.tensordot(gamma, Qma_jj, axes=(0, 0))
        Z_jj = T1_all @ Qma_jj.transpose(0, 2, 1)
        Fia_bar_jj -= np.tensordot(Z_jj, Qma_jj, axes=((0, 1), (0, 1)))
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
    # Qab_t1[Q,a,b] = Qab[Q,a,b] - Σ_n T1[n,a] * Qma[Q,n,b]
    # (T1_all.T @ Qma broadcasts over Q): (npno,nocc) @ (Q,nocc,npno) = (Q,npno,npno)
    Qab_t1 = Qab - (T1_all.T @ Qma)
    # ladder[a,b] = Σ_{Q,c,d} Qab_t1[Q,a,c] * T2[c,d] * Qab_t1[Q,b,d]
    X = Qab_t1 @ T2_ij                        # (Q, npno, npno)
    return np.tensordot(X, Qab_t1, axes=((0, 2), (0, 2)))
