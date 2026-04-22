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
                         T_CUT_MKN=1e-3, T_CUT_CLMO=1e-3, C_pao=None):
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
    # Include:
    #   (a) atoms that CENTER the PAO indices in pao_domains[i], AND
    #   (b) atoms on which those PAO vectors have non-negligible AMPLITUDE
    #       via the orthogonalization tail (C_pao[atom_AOs, pao_domains[i]]
    #       above T_CUT_CLMO).
    # Rationale: PAOs are (1 - P_occ) I orthogonalized in the pair domain;
    # for CP-ghost systems the orthogonalization pushes significant
    # amplitude onto ghost atoms.  If bfs2 excludes those atoms, the
    # sparse integral (Q|ab) misses cancelling contributions and the
    # T2 ladder term A blows up by ~20x.  Cf. project_s22_ladder_bug.md.
    lmo_to_paoatoms = []
    for i in range(nocc):
        if len(pao_domains[i]) == 0:
            lmo_to_paoatoms.append(np.zeros(0, dtype=int))
            continue
        # (a) Atoms centering the PAO indices
        indexed_atoms = set(atom_ids[pao_domains[i]].tolist())
        # (b) Atoms with significant C_pao amplitude on those PAO columns.
        # Skip (b) if C_pao isn't available (backward compat).
        if C_pao is not None:
            C_slice = C_pao[:, pao_domains[i]]                # (nao, |dom|)
            per_ao = np.max(np.abs(C_slice), axis=1)
            for A in range(natm):
                if A in indexed_atoms:
                    continue
                mask_A = (atom_ids == A)
                if np.any(per_ao[mask_A] > T_CUT_CLMO):
                    indexed_atoms.add(A)
        lmo_to_paoatoms.append(
            np.array(sorted(indexed_atoms), dtype=int))

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
    return ci['Qma'][:, lmo_idx, :].T  # (npno, n_local)


def get_local_ooL_vec(cc_ints, k, l, pair_key):
    """Get locally-fitted ooL[k,l] in pair pair_key's local fitting.

    Returns (n_local,) or None if not available.
    """
    ci = cc_ints.get(pair_key)
    if ci is None:
        return None
    i_lmo, j_lmo = pair_key
    if l == i_lmo:
        return ci['i_Qk'][:, k]
    elif l == j_lmo:
        return ci['j_Qk'][:, k]
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
        Qma = ci['Qma']  # (n_local, nocc, npno)
        # ovL_lmo[Q, a] = Qma[Q, lmo, a]; K[a,b] = Σ_Q ovL1[Q,a]*ovL2[Q,b]
        K = Qma[:, lmo1, :].T @ Qma[:, lmo2, :]
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

    if screening_maps is None:
        screening_maps = build_screening_maps(
            mol, auxmol, C_lmo, pao_domains, s1e, strong_pair_keys,
            T_CUT_MKN=T_CUT_MKN, T_CUT_CLMO=T_CUT_CLMO, C_pao=C_pao)
    if sparse_arrays is None:
        sparse_arrays = build_sparse_df_arrays(
            mol, auxmol, C_lmo, C_pao, screening_maps)

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
    aux_at_atom = [np.where(aux_atom_ids == A)[0] for A in range(natm)]
    qij_atom = [None] * natm
    qia_atom = [None] * natm
    qab_atom = [None] * natm
    for A in range(natm):
        Qs = aux_at_atom[A]
        if len(Qs) == 0:
            continue
        nl = len(riatom_to_lmos_ext[A])
        np_ = len(riatom_to_paos_ext[A])
        if nl == 0 or np_ == 0:
            continue
        qij_atom[A] = np.stack([qij[Q] for Q in Qs])    # (nQA, nl, nl)
        qia_atom[A] = np.stack([qia[Q] for Q in Qs])    # (nQA, nl, np)
        qab_atom[A] = np.stack([qab[Q] for Q in Qs])    # (nQA, np, np)
    # Map global Q → position within its atom's Q-stack
    aux_pos_in_atom = -np.ones(naux, dtype=np.int64)
    for A in range(natm):
        for pos, Q in enumerate(aux_at_atom[A]):
            aux_pos_in_atom[Q] = pos

    cc_ints = {}
    key_set = set(keys)

    def _process_pair(key):
        """Build cc_ints[key] entry. Pure function — safe for thread parallel."""
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

        # Cross-pair partner enumeration, restricted to pair (i,j)'s local
        # LMO domain when pair_lmo_idx is provided.  This drops per-pair
        # cost from O(nocc) to O(nlmo_ij), turning cc_ints build from
        # O(N^3) into O(N^2).
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

        # Pre-fit accumulators.  raw_io/raw_jo/raw_ma's LMO axis is set
        # to ``nocc`` below once we know which LMOs actually get
        # populated from the centerQ stacks (the union of
        # riatom_to_lmos_ext over all of this pair's aux centers).
        raw_iv = np.zeros((n_local, npno))
        raw_jv = np.zeros((n_local, npno))
        raw_ab = np.zeros((n_local, npno, npno))
        raw_pair = np.zeros(n_local)
        raw_cross_kj = {k: np.zeros((n_local, npno, n_kj))
                        for k, _, n_kj in kj_partners}
        raw_kv_kj = {k: np.zeros((n_local, n_kj))
                     for k, _, n_kj in kj_partners}
        raw_cross_ji = {k: np.zeros((n_local, npno, n_ki))
                        for k, _, n_ki in ki_partners}
        raw_kv_ki = {k: np.zeros((n_local, n_ki))
                     for k, _, n_ki in ki_partners}

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
        if len(unique_centers) > 0:
            _ext_union = np.unique(np.concatenate(
                [riatom_to_lmos_ext[c] for c in unique_centers]))
        else:
            _ext_union = np.zeros(0, dtype=np.int64)
        p_lmos = _ext_union.astype(np.int64)
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

        for centerQ in unique_centers:
            ext_lmos = riatom_to_lmos_ext[centerQ]
            if len(ext_lmos) == 0 or qij_atom[centerQ] is None:
                continue

            # Q's belonging to this centerQ that are in pair_aux:
            mask_C = centers_of_aux == centerQ
            local_Q = np.where(mask_C)[0]                   # positions in n_local
            global_Q = aux_idx[local_Q]                     # global aux indices
            atom_pos = aux_pos_in_atom[global_Q]            # positions in atom's stack
            # Slice the per-atom stacks to just this pair's Q's
            qij_b = qij_atom[centerQ][atom_pos]             # (nQp, nl, nl)
            qia_b = qia_atom[centerQ][atom_pos]             # (nQp, nl, np)
            qab_b = qab_atom[centerQ][atom_pos]             # (nQp, np, np)

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
                    qia_b_pp_kept = qia_b_pp[:, ext_kept_pos, :]  # (nQp, n_kept, npp)
                    ma_b_kept = (
                        qia_b_pp_kept.reshape(nQp * ext_kept_pos.size, -1)
                        @ X_ij_slice
                    ).reshape(nQp, ext_kept_pos.size, npno)
                    raw_ma[np.ix_(local_Q, ext_kept_lmos)] = ma_b_kept

                # raw_ab: X^T qab_b_pp X — one big reshape+gemm for tmp,
                # then batched gemm over Q for the second contraction.
                qab_b_pp = qab_b[:, ij_u_in_Q[:, None], ij_u_in_Q[None, :]]
                tmp = (qab_b_pp.reshape(nQp * len(ij_u_in_Q), -1)
                       @ X_ij_slice).reshape(nQp, len(ij_u_in_Q), npno)
                raw_ab[local_Q] = np.matmul(X_ij_slice.T, tmp)

            # Cross-pair partners.
            # Optimization: project the IJ side first ONCE per (pair, centerQ)
            # via X_ij.T @ qab_b. Then per-partner work reduces to one 1D
            # fancy slice + one batched matmul (nQp, npno, npp_kj) @ (npp_kj,
            # n_kj). Empirically ~10× faster than per-partner 2D fancy index
            # at sizes where pair_paos is large (medium systems).
            np_full = qab_b.shape[1]
            if len(ij_u_in_Q) > 0:
                # qab_ij[Q, u, v] = qab_b[Q, ij_u_in_Q[u], v]
                if len(ij_u_in_Q) == np_full and np.array_equal(
                        ij_u_in_Q, np.arange(np_full)):
                    qab_ij = qab_b
                else:
                    qab_ij = qab_b[:, ij_u_in_Q, :]      # (nQp, npp_ij, np_full)
                # proj_ij[Q, A_ij, v] = sum_u X_ij_slice[u, A_ij] * qab_ij[Q, u, v]
                proj_ij = np.matmul(X_ij_slice.T, qab_ij)  # (nQp, npno, np_full)
            else:
                proj_ij = None

            for k, X_kj, pp_kj, n_kj in kj_data:
                k_s = riatom_to_lmos_ext_dense[centerQ, k]
                kj_pao_pos = riatom_to_paos_ext_dense[centerQ, pp_kj]
                kj_mask = kj_pao_pos >= 0
                kj_u_in_pair = np.where(kj_mask)[0]
                kj_u_in_Q = kj_pao_pos[kj_mask]
                if len(kj_u_in_Q) == 0:
                    continue
                X_kj_slice = X_kj[kj_u_in_pair]              # (|kj∩Q|, n_kj)
                if k_s >= 0:
                    raw_kv_kj[k][local_Q] = (qia_b[:, k_s, kj_u_in_Q]
                                             @ X_kj_slice)
                if proj_ij is not None:
                    sub = proj_ij[:, :, kj_u_in_Q]           # (nQp, npno, npp_kj)
                    raw_cross_kj[k][local_Q] = sub @ X_kj_slice

            for k, X_ki, pp_ki, n_ki in ki_data:
                k_s = riatom_to_lmos_ext_dense[centerQ, k]
                ki_pao_pos = riatom_to_paos_ext_dense[centerQ, pp_ki]
                ki_mask = ki_pao_pos >= 0
                ki_u_in_pair = np.where(ki_mask)[0]
                ki_u_in_Q = ki_pao_pos[ki_mask]
                if len(ki_u_in_Q) == 0:
                    continue
                X_ki_slice = X_ki[ki_u_in_pair]
                if k_s >= 0:
                    raw_kv_ki[k][local_Q] = (qia_b[:, k_s, ki_u_in_Q]
                                             @ X_ki_slice)
                if proj_ij is not None:
                    sub = proj_ij[:, :, ki_u_in_Q]           # (nQp, npno, npp_ki)
                    raw_cross_ji[k][local_Q] = sub @ X_ki_slice

        # Apply local J^{-1/2}
        j2c_local = j2c[np.ix_(aux_idx, aux_idx)]
        eigvals, eigvecs = np.linalg.eigh(j2c_local)
        keep = eigvals > 1e-14
        jhi = (eigvecs[:, keep] * (1.0 / np.sqrt(eigvals[keep]))) \
            @ eigvecs[:, keep].T

        q_iv = jhi @ raw_iv
        q_jv = jhi @ raw_jv
        q_io_red = jhi @ raw_io                  # (n_local, nlmo_p) reduced
        q_jo_red = jhi @ raw_jo                  # (n_local, nlmo_p) reduced
        q_pair = jhi @ raw_pair
        # raw_ma is reduced (n_local, nlmo_p, npno) — small jhi matmul.
        Qma_red = (jhi @ raw_ma.reshape(n_local, -1)
                    ).reshape(n_local, nlmo_p, npno)
        Qab = (jhi @ raw_ab.reshape(n_local, -1)).reshape(n_local, npno, npno)

        # Scatter reduced fitted quantities into full-nocc shape so
        # downstream consumers (compute_B_tilde, t1_fock Fij_bar,
        # lccsd.py Fij_bar update, etc.) that index by global LMO
        # continue to work unchanged. The compute savings come from
        # the small jhi matmul above; memory footprint is the same as
        # before (full-nocc) but zeros in rows outside p_lmos.
        q_io = np.zeros((n_local, nocc))
        q_jo = np.zeros((n_local, nocc))
        Qma = np.zeros((n_local, nocc, npno))
        q_io[:, p_lmos] = q_io_red
        q_jo[:, p_lmos] = q_jo_red
        Qma[:, p_lmos, :] = Qma_red

        K_iajb = q_iv.T @ q_jv
        K_mnij = q_io.T @ q_jo                    # (nocc, nocc) full
        K_bar_ij = q_io.T @ q_jv
        K_bar_ji = q_jo.T @ q_iv
        K_bar_chem = np.tensordot(q_pair, Qma, axes=(0, 0))
        J_ijab = np.tensordot(q_pair, Qab, axes=(0, 0))

        J_ij_kj = {}
        K_ij_kj_dict = {}
        for k, key_kj, _ in kj_partners:
            cross_fitted = (jhi @ raw_cross_kj[k].reshape(n_local, -1)
                            ).reshape(n_local, npno, raw_cross_kj[k].shape[2])
            # q_io now uses the reduced (p_lmos) axis; k is a global LMO
            # index, kj_partners is built from pair_lmo_idx[key], so k is
            # guaranteed to be in p_lmos — map via p_lmos_dense.
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

        return key, {
            'K_iajb': K_iajb,
            'K_mnij': K_mnij,       # (nlmo_p, nlmo_p) reduced
            'K_bar_ij': K_bar_ij,
            'K_bar_ji': K_bar_ji,
            'K_bar_chem': K_bar_chem,  # (nlmo_p, npno) reduced
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
    if _pool is not None:
        for k, entry in _pool.map(_process_pair, keys):
            cc_ints[k] = entry
    else:
        for key in keys:
            k, entry = _process_pair(key)
            cc_ints[k] = entry

    return cc_ints


def t1_ints(cc_ints, t1_pno, pno_spaces, S_pno_cache, keys, nocc,
            pair_lmo_idx=None):
    """Build T1-dressed DF intermediates, matching Psi4 t1_ints().

    For each pair (ij), builds:
        i_Qa_t1[Q, a] = i_Qa[Q,a] - i_Qk[Q,:] @ T1_all[:,a]
                       + Σ_b (Qab[Q,a,b] - T1_all^T @ Qma[Q,:,b]) * t1_i[b]

    The implicit `k` sum inside this formula is restricted to
    pair_lmo_idx[key] (Psi4 lmopair_to_lmos_[ij]) when provided.
    """
    from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair

    dressed = {}
    for key in keys:
        ci = cc_ints.get(key)
        if ci is None:
            continue
        i, j = key
        npno = pno_spaces[key]['C_pno'].shape[1]

        if pair_lmo_idx is not None and key in pair_lmo_idx:
            lmo_idx = np.asarray(pair_lmo_idx[key])
        else:
            lmo_idx = np.arange(nocc)

        # Project T1 only for LMOs in pair's domain
        T1_local = np.zeros((len(lmo_idx), npno))
        for ki, k in enumerate(lmo_idx):
            T1_local[ki] = _project_t1_to_pair(
                t1_pno, int(k), key, S_pno_cache, pno_spaces)

        Qma = ci['Qma'][:, lmo_idx, :]       # (n_local, nlmo, npno)
        Qab = ci['Qab']                       # (n_local, npno, npno)

        def _dress_one(lmo_global, Qa_key):
            i_Qa = ci[Qa_key]                                 # (n_local, npno)
            i_Qk_local = ci[Qa_key.replace('Qa', 'Qk')][:, lmo_idx]  # (n_local, nlmo)

            t1_lmo = _project_t1_to_pair(
                t1_pno, int(lmo_global), key, S_pno_cache, pno_spaces)
            result = i_Qa - i_Qk_local @ T1_local              # (n_local, npno)
            result += np.einsum('Qab,b->Qa', Qab, t1_lmo)
            qma_t1 = np.einsum('Qmb,b->Qm', Qma, t1_lmo)       # (n_local, nlmo_p)
            result -= qma_t1 @ T1_local                         # (n_local, npno)
            return result

        dressed[key] = {
            'i_Qa_t1': _dress_one(i, 'i_Qa'),
            'j_Qa_t1': _dress_one(j, 'j_Qa'),
        }

    return dressed


def t1_fock(cc_ints, dressed_ints, t1_pno, fov_pno, pno_spaces,
            S_pno_cache, F_lmo, eps_lmo, foo_t2, keys, nocc, _pool=None,
            pair_lmo_idx=None):
    """Build dressed Fock matrices matching Psi4 t1_fock().

    Returns:
        Fkj: (nocc, nocc) dressed occupied Fock
        Fab_all: dict pair_key -> (npno, npno) dressed virtual Fock
        foo_t1: (nocc, nocc) T1 part of foo

    LMO sums inside each per-pair dressing are restricted to
    pair_lmo_idx[key] (Psi4 lmopair_to_lmos_[ij]) when provided.
    """
    from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair

    def _pair_domain(key):
        if pair_lmo_idx is not None and key in pair_lmo_idx:
            return np.asarray(pair_lmo_idx[key])
        return np.arange(nocc)

    # Combined per-pair work (Step 1 Fkj contribution + Step 2 Fab)
    def _per_pair(key):
        ci = cc_ints.get(key)
        if ci is None:
            return key, None
        i, j = key
        npno = pno_spaces[key]['C_pno'].shape[1]
        lmo_idx = _pair_domain(key)

        T1_local = np.zeros((len(lmo_idx), npno))
        for ki, k in enumerate(lmo_idx):
            T1_local[ki] = _project_t1_to_pair(
                t1_pno, int(k), key, S_pno_cache, pno_spaces)

        # Step 1 Fkj contributions — K_bar_* are (nocc, npno); restrict rows.
        K_bar_chem_l = ci['K_bar_chem'][lmo_idx]
        K_bar_ji_l = ci['K_bar_ji'][lmo_idx]
        d_ij = (2.0 * np.sum(T1_local * K_bar_chem_l)
                - np.sum(T1_local * K_bar_ji_l))
        d_ji = None
        if i != j:
            K_bar_ij_l = ci['K_bar_ij'][lmo_idx]
            d_ji = (2.0 * np.sum(T1_local * K_bar_chem_l)
                    - np.sum(T1_local * K_bar_ij_l))

        # Step 2 Fab
        Qma = ci['Qma'][:, lmo_idx, :]       # (n_local, nlmo, npno)
        Qab = ci['Qab']                       # (n_local, npno, npno)
        e_pno = pno_spaces[key]['e_pno']
        Fab = np.diag(e_pno)
        gamma = Qma.reshape(Qma.shape[0], -1) @ T1_local.ravel()  # (n_local,)
        Fab += 2.0 * np.tensordot(gamma, Qab, axes=(0, 0))
        Y = Qab @ T1_local.T                  # (n_local, npno, nlmo)
        Fab -= np.tensordot(Y, Qma, axes=((0, 2), (0, 1)))

        Fia_bar = 2.0 * np.tensordot(gamma, Qma, axes=(0, 0))     # (nlmo, npno)
        Z = T1_local @ Qma.transpose(0, 2, 1)                     # (n_local, nlmo, nlmo)
        Fia_bar -= np.tensordot(Z, Qma, axes=((0, 1), (0, 1)))    # (nlmo, npno)
        Fab -= T1_local.T @ Fia_bar
        return key, (d_ij, d_ji, Fab)

    Fkj = F_lmo.copy()
    Fab_all = {}
    if _pool is not None:
        results = list(_pool.map(_per_pair, keys))
    else:
        results = [_per_pair(k) for k in keys]
    for key, payload in results:
        if payload is None:
            continue
        d_ij, d_ji, Fab = payload
        i, j = key
        Fkj[i, j] += d_ij
        if d_ji is not None:
            Fkj[j, i] += d_ji
        Fab_all[key] = Fab

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
        T1_local = np.zeros((len(lmo_idx), npno))
        for ki, k in enumerate(lmo_idx):
            T1_local[ki] = _project_t1_to_pair(
                t1_pno, int(k), key_jj, S_pno_cache, pno_spaces)
        Qma_jj = ci['Qma'][:, lmo_idx, :]     # (n_local, nlmo, npno)
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
                    pno_spaces, S_pno_cache, key, nocc,
                    pair_lmo_idx=None):
    """Build B_tilde for pair (ij) matching Psi4's precomputed B_tilde.

    B_tilde[k,l] = (ki|lj)_dressed + Σ_{a,b} tau[a,b] * (ka|lb)

    The k, l sums are restricted to pair_lmo_idx[key] (Psi4's
    lmopair_to_lmos_[ij]) when provided.  Local entries are scattered into
    a (nocc, nocc) output so downstream residual.py indexing is unchanged.
    """
    from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair

    ci = cc_ints.get(key)
    if ci is None:
        return np.zeros((nocc, nocc))

    i, j = key
    npno = pno_spaces[key]['C_pno'].shape[1]

    if pair_lmo_idx is not None and key in pair_lmo_idx:
        lmo_idx = np.asarray(pair_lmo_idx[key])
    else:
        lmo_idx = np.arange(nocc)
    nlmo = len(lmo_idx)

    Qma = ci['Qma'][:, lmo_idx, :]           # (n_local, nlmo, npno)
    i_Qk = ci['i_Qk'][:, lmo_idx]            # (n_local, nlmo)
    j_Qk = ci['j_Qk'][:, lmo_idx]            # (n_local, nlmo)

    if t1_pno is not None:
        t1_i = _project_t1_to_pair(t1_pno, i, key, S_pno_cache, pno_spaces)
        t1_j = _project_t1_to_pair(t1_pno, j, key, S_pno_cache, pno_spaces)
    else:
        t1_i = np.zeros(npno)
        t1_j = np.zeros(npno)

    # i_Qk_t1[Q, kl] = i_Qk[Q, kl] + Σ_a Qma[Q, kl, a] * t1_i[a]
    i_Qk_t1 = i_Qk.copy()
    i_Qk_t1 += np.einsum('Qka,a->Qk', Qma, t1_i)
    j_Qk_t1 = j_Qk.copy()
    j_Qk_t1 += np.einsum('Qka,a->Qk', Qma, t1_j)

    B_local = i_Qk_t1.T @ j_Qk_t1            # (nlmo, nlmo)

    # voov dressing (bare T2 per Psi4)
    T2_ij = t2_pno_all[key]
    P = np.einsum('ab,Qka->kbQ', T2_ij, Qma)     # (nlmo, npno, n_local)
    B_local += np.einsum('kbQ,Qlb->kl', P, Qma)

    # Scatter local (nlmo × nlmo) into global (nocc × nocc) output
    B_tilde = np.zeros((nocc, nocc))
    B_tilde[np.ix_(lmo_idx, lmo_idx)] = B_local
    return B_tilde


def compute_ladder(cc_ints, t2_pno_all, t1_pno, pno_spaces,
                   S_pno_cache, key, nocc, pair_lmo_idx=None):
    """Compute ladder term A for pair (ij) matching Psi4 Term A.

    A[a,b] = Σ_Q Qab_t1[Q,a,c] * T2[c,d] * Qab_t1[Q,b,d]

    The `k` (LMO) sum inside Qab_t1 is restricted to the pair's local LMO
    domain (lmopair_to_lmos_[ij] in Psi4) when pair_lmo_idx is provided.
    This turns a per-pair cost of O(n_local · nocc · npno²) into
    O(n_local · nlmo_ij · npno²), restoring DLPNO linear-scaling.
    """
    from pyscf.cc.dlpno_tccsd.lccsd import _project_t1_to_pair

    ci = cc_ints.get(key)
    if ci is None:
        return np.zeros((0, 0))

    i, j = key
    npno = pno_spaces[key]['C_pno'].shape[1]
    Qab = ci['Qab']
    Qma_full = ci['Qma']   # (n_local, nocc, npno)

    if pair_lmo_idx is not None and key in pair_lmo_idx:
        lmo_idx = np.asarray(pair_lmo_idx[key])
    else:
        lmo_idx = np.arange(nocc)
    Qma = Qma_full[:, lmo_idx, :]            # (n_local, nlmo, npno)

    T1_local = np.zeros((len(lmo_idx), npno))
    if t1_pno is not None:
        for ki, k in enumerate(lmo_idx):
            T1_local[ki] = _project_t1_to_pair(
                t1_pno, int(k), key, S_pno_cache, pno_spaces)

    T2_ij = t2_pno_all[key]
    # Qab_t1[Q,a,b] = Qab[Q,a,b] - Σ_{k ∈ lmo_idx} T1_local[k,a] * Qma[Q,k,b]
    Qab_t1 = Qab - (T1_local.T @ Qma)
    X = Qab_t1 @ T2_ij                        # (Q, npno, npno)
    return np.tensordot(X, Qab_t1, axes=((0, 2), (0, 2)))
