"""
External-space perturbative triples (T) for DLPNO-TCCSD(T).

Computes the (T) energy correction restricted to triples that have at least
one index outside the CAS space. Pure-CAS triples are excluded because they
are already contained in the DMRG wavefunction.

The triple PNO space for each triplet (i,j,k) is the intersection of the
pair PNO spaces: PNO(ijk) = PNOs shared between pairs (ij), (ik), (jk).
This follows the TNO/triple-PNO approach of Jiang JCP 2024 / Psi4 lccsd_t.cc.

The energy formula is the standard CCSD(T) perturbative triples expression:
    e_T = sum_{i>j>k, abc} W[a,b,c] * r3(W)[a,b,c] / D[a,b,c]
where W is the W3 intermediate and r3 is the cyclic antisymmetrizer.

W[a,b,c] is built from the 6-term canonical formula using DF 3-index integrals:
    W[a,b,c] = sum_f (ia|bf)*t2_jk[c,f] + sum_f (jb|af)*t2_ik[c,f]
             + sum_f (kc|af)*t2_ij[b,f] + sum_f (ia|cf)*t2_kj[b,f]
             + sum_f (jb|cf)*t2_ki[a,f] + sum_f (kc|bf)*t2_ji[a,f]
where (xa|bf) = sum_L ovL_x[a,L] * vvL[b,f,L].

CAS double-counting prevention (Lang et al. 2020, Sec. II.C):
    All active T1 and T2 amplitudes are zeroed before computing (T).
    All triples (i,j,k) are included — no triples are skipped.

References:
    Lee & Head-Gordon, JCTC 2019, 15, 4594  (TCCSD(T) CAS exclusion)
    Jiang, JCP 2024  (DLPNO-(T) triple PNO intersection)
    Stanton, Chem. Phys. Lett. 1997, 281, 130  ((T) energy formula)
    Ye & Berkelbach, JCTC 2024  (LNO-(T) for PySCF infrastructure)
    ccsd_t_slow.py: canonical reference formula
"""

import os
import numpy as np
from functools import reduce
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo


def _tmem(label):
    """Print [TMEM/<label>] process RSS if DLPNO_MEM_PROBE=1 (triples phase)."""
    if os.environ.get('DLPNO_MEM_PROBE'):
        try:
            with open('/proc/self/statm') as _fh:
                _rss = int(_fh.read().split()[1]) * 4096 / (1024**3)
            print(f'  [TMEM/{label}] RSS={_rss:.2f} GiB', flush=True)
        except Exception:
            pass


def _mem_available_gb():
    """Best-effort kernel MemAvailable in GiB (None if /proc unreadable).

    MemAvailable already subtracts everything resident (base + sparse-DF are
    live when we read it before the energy pass) and adds back reclaimable
    page cache, so it is the true headroom the per-triple working sets can
    grow into. It does NOT count glibc-retained freed heap as available, so
    it errs on the safe (fewer-threads) side vs the real peak — good for OOM.
    """
    try:
        with open('/proc/meminfo') as _fh:
            for _ln in _fh:
                if _ln.startswith('MemAvailable:'):
                    return int(_ln.split()[1]) / (1024 ** 2)  # kB -> GiB
    except Exception:
        pass
    return None


def _t_worker_count(pool_workers, n_tno_max=None):
    """Number of parallel workers for the (T) energy pass, memory-aware.

    The (T) peak is working-set-bound: RSS ~= base + sparse_DF + W * nthreads,
    where W is the per-thread C scratch (dlpno_triples_orch.c), grow-and-keep
    to the largest triple: W ~= 8 * (3*n_tno^2*naux + ~12*n_tno^3) bytes.  So
    the memory-safe worker count is  N = MemAvailable / W  (base and sparse_DF
    are already resident when MemAvailable is read).

    Knobs (all optional; if none set -> use the whole pool, i.e. no change):
      DLPNO_T_MAX_WORKERS    explicit int hard cap (the robust throttle), or
                             'auto' to force the memory calc below.
      DLPNO_MEM_BUDGET_GB    headroom to fill; default = live MemAvailable.
                             Setting it (without DLPNO_T_MAX_WORKERS) also
                             enables auto sizing.
      DLPNO_T_GB_PER_WORKER  per-thread working set W in GiB (default 5.0;
                             basis/system dependent, ~4.5 measured on
                             rxn_12/def2-TZVP, n_tno_max~180).
      DLPNO_T_MEM_RESERVE_GB safety margin kept free (default 8.0).
    """
    _env = os.environ.get('DLPNO_T_MAX_WORKERS')
    _budget_env = os.environ.get('DLPNO_MEM_BUDGET_GB')
    if _env and _env.lower() != 'auto':
        try:
            return max(1, min(int(_env), pool_workers))   # explicit hard cap
        except ValueError:
            pass
    _auto = (_env is not None and _env.lower() == 'auto') \
        or _budget_env is not None
    if not _auto:
        return pool_workers               # no request -> unchanged behaviour
    # --- auto sizing ---
    budget = float(_budget_env) if _budget_env else _mem_available_gb()
    if not budget or budget <= 0:
        return pool_workers
    gbpw = float(os.environ.get('DLPNO_T_GB_PER_WORKER', '5.0'))
    reserve = float(os.environ.get('DLPNO_T_MEM_RESERVE_GB', '8.0'))
    n = int((budget - reserve) / max(gbpw, 1e-6))
    return max(1, min(n, pool_workers))


def _triple_pno_union(pno_spaces, i, j, k, s1e, t2_for_T=None,
                      T_CutTNO=1e-9, S_cut=1e-6):
    """Compute the TNO space from the averaged triplet density (Jiang 2024 eq.62).

    1. Pool PNO columns from pairs ij, ik, jk.
    2. Orthogonalize via canonical orthogonalization in S metric.
    3. Project pair densities into this common space, average → D_ijk.
    4. Diagonalize D_ijk, truncate at T_CutTNO → final TNO basis.

    Args:
        pno_spaces (dict): Output of pno.make_pnos.
        i, j, k (int): Triple LMO indices.
        s1e (np.ndarray): (nao, nao) AO overlap matrix S.
        t2_for_T (dict, optional): T2 amplitudes in PNO basis for each pair.
            If provided, pair densities are built from these (CCSD amplitudes).
            Otherwise, initial PNO T2 from pno_spaces is used.
        T_CutTNO (float): TNO occupation number truncation threshold.
        S_cut (float): Eigenvalue threshold for canonical orthogonalization.

    Returns:
        C_tno (np.ndarray): (nao, n_tno) S-orthonormal TNO coefficients.
        n_tno (int): Number of TNOs.
    """
    ij = (min(i,j), max(i,j))
    ik = (min(i,k), max(i,k))
    jk = (min(j,k), max(j,k))

    if ij not in pno_spaces or ik not in pno_spaces or jk not in pno_spaces:
        return None, 0

    C_ij = pno_spaces[ij]['C_pno']
    C_ik = pno_spaces[ik]['C_pno']
    C_jk = pno_spaces[jk]['C_pno']

    if C_ij.shape[1] == 0 and C_ik.shape[1] == 0 and C_jk.shape[1] == 0:
        return np.zeros((s1e.shape[0], 0)), 0

    # Pool all PNO columns from the three pairs
    cols = [c for c in (C_ij, C_ik, C_jk) if c.shape[1] > 0]
    C_pool = np.hstack(cols)   # (nao, n_ij + n_ik + n_jk)

    # Canonical orthogonalization in S metric: diagonalize S_pool, keep > S_cut
    S_pool = reduce(np.dot, (C_pool.T, s1e, C_pool))
    eigvals, eigvecs = np.linalg.eigh(S_pool)
    keep = eigvals > S_cut
    n_union = int(np.sum(keep))

    if n_union == 0:
        return np.zeros((s1e.shape[0], 0)), 0

    X = eigvecs[:, keep] / np.sqrt(eigvals[keep])
    C_union = np.dot(C_pool, X)   # (nao, n_union), S-orthonormal

    # If no T2 available or T_CutTNO <= 0, return the full union
    if T_CutTNO <= 0:
        return C_union, n_union

    # Build averaged triplet density D_ijk = (D_ij + D_ik + D_jk) / 3
    # Project each pair's T2 into the union basis, compute pair density
    D_avg = np.zeros((n_union, n_union))
    for pair_key, ii, jj in [(ij, i, j), (ik, i, k), (jk, j, k)]:
        C_p = pno_spaces[pair_key]['C_pno']
        if C_p.shape[1] == 0:
            continue
        # Projection matrix: PNO(pair) -> union TNO
        U = reduce(np.dot, (C_p.T, s1e, C_union))   # (n_pno, n_union)
        # Get T2 amplitudes
        if t2_for_T is not None and pair_key in t2_for_T:
            T2_p = t2_for_T[pair_key]
        elif pno_spaces[pair_key].get('T2_pno') is not None:
            T2_p = pno_spaces[pair_key]['T2_pno']
        else:
            continue
        # Project T2 to union basis
        T2_u = reduce(np.dot, (U.T, T2_p, U))   # (n_union, n_union)
        Tt_u = 2.0 * T2_u - T2_u.T
        # Pair density in union basis
        D_p = np.dot(Tt_u, T2_u.T) + np.dot(T2_u, Tt_u.T)
        if ii == jj:
            D_p *= 0.5
        D_avg += D_p

    D_avg /= 3.0

    # Diagonalize triplet density and truncate at T_CutTNO
    tno_occ, tno_vecs = np.linalg.eigh(D_avg)
    keep_tno = np.abs(tno_occ) > T_CutTNO
    n_tno = int(np.sum(keep_tno))

    if n_tno == 0:
        # Keep at least the largest
        n_tno = 1
        keep_tno[np.argmax(np.abs(tno_occ))] = True

    # Transform union → truncated TNO
    C_tno = np.dot(C_union, tno_vecs[:, keep_tno])   # (nao, n_tno)

    return C_tno, n_tno


def _triple_pno_union_psi4(pno_spaces, i, j, k, C_pao, S_pao_full, F_pao_full,
                            t2_for_T=None, T_CutTNO=1e-9, S_cut_domain=1e-8,
                            pao_domains_triple=None):
    """Psi4-style TNO construction in orthogonalized triple-PAO basis.

    Mirrors Psi4 DLPNOCCSD_T::tno_transform (triples.cc:431):
    (1) triple_paos = union of pair_paos for (ij), (jk), (ik) in PAO
        index space (no orthogonalization yet).
    (2) Canonical orthogonalization of S_pao[triple_paos, triple_paos]
        → X_pao_ijk (npao_ijk, npao_can_ijk).
    (3) Project each pair's density D_ij = Tt_ij @ T_ij.T + Tt_ij.T @ T_ij
        into the triple's orthogonal-PAO basis.
    (4) D_ijk = (D_ij + D_jk + D_ik) / 3; diagonalize → TNOs.
    (5) Truncate at T_CutTNO, canonicalize in F basis → X_tno_canonical.
    (6) X_tno_ijk = X_pao_ijk @ X_tno_canonical
        shape (npao_ijk, n_tno) — maps triple's RAW PAO domain to
        canonical TNO. This is what Psi4 stores as X_tno_[ijk] and what
        compute_lccsd_t0 uses to slice sparse qia[Q]/qij[Q]/qab[Q].

    Returns:
        C_tno_sc   : (nao, n_tno) TNO in AO basis (S-orthonormal)
        n_tno      : int
        X_tno_ijk  : (npao_ijk, n_tno) triple-PAO → canonical-TNO
        triple_paos: (npao_ijk,) int PAO indices (sorted)
        eps_tno_sc : (n_tno,) TNO orbital energies
    """
    ij = (min(i, j), max(i, j))
    ik = (min(i, k), max(i, k))
    jk = (min(j, k), max(j, k))
    nao = C_pao.shape[0]

    if not all(key in pno_spaces for key in (ij, ik, jk)):
        return None, 0, None, None, None

    # (1) triple_paos: union of per-LMO PAO domains [i, j, k]
    # Psi4 lmotriplet_to_paos_[ijk] = union of lmo_to_paos[i, j, k] rebuilt
    # at the triples stage with T_CUT_DO_TRIPLES (triples.cc:316-329).
    # Fallback: union of pair_paos for (ij, jk, ik) — the CCSD-stage domain.
    if pao_domains_triple is not None:
        pao_set = set()
        for lmo in (i, j, k):
            pao_set.update(int(x) for x in np.asarray(pao_domains_triple[lmo]).tolist())
    else:
        pao_set = set()
        for key in (ij, jk, ik):
            pp = pno_spaces[key].get('pair_paos')
            if pp is not None:
                pao_set.update(int(x) for x in np.asarray(pp).tolist())
    if not pao_set:
        return np.zeros((nao, 0)), 0, None, None, None
    triple_paos = np.array(sorted(pao_set), dtype=np.int64)

    # (2) Canonical orthogonalization of triple's PAO domain
    from pyscf.cc.dlpno_tccsd.local_orbs import orthogonalize_pao_domain
    C_orth_ijk, X_pao_ijk = orthogonalize_pao_domain(
        C_pao, S_pao_full, triple_paos,
        S_cut=S_cut_domain, method='psi4')
    npao_can_ijk = X_pao_ijk.shape[1]
    if npao_can_ijk == 0:
        return np.zeros((nao, 0)), 0, None, None, None

    # F_orth_ijk and D_ijk: single C kernel call folds
    #   F_orth = X.T @ F_pao[trip,trip] @ X
    #   D_ijk = (1/3) Σ_keys S.T @ D_pair @ S, with S, D_pair built per key.
    # Pack 3-key data into flat buffers for the C kernel.
    keys_list   = [ij, jk, ik]
    ij_lmos     = [(i, j), (j, k), (i, k)]
    n_keys      = 3
    pair_paos_n = np.zeros(n_keys, dtype=np.int32)
    n_pno_arr   = np.zeros(n_keys, dtype=np.int32)
    same_lmo    = np.zeros(n_keys, dtype=np.int32)
    pp_lists    = []
    Xpno_lists  = []
    T2_lists    = []
    for kk, (key, (ii_l, jj_l)) in enumerate(zip(keys_list, ij_lmos)):
        X_pno_pair = pno_spaces[key].get('X_pno')
        pp_pair = np.asarray(pno_spaces[key].get('pair_paos'))
        if t2_for_T is not None and key in t2_for_T:
            T2_p = t2_for_T[key]
        else:
            T2_p = pno_spaces[key].get('T2_pno')
        if (X_pno_pair is None or X_pno_pair.shape[1] == 0
                or T2_p is None or T2_p.size == 0
                or pp_pair.size == 0):
            pp_lists.append(np.empty(0, dtype=np.int64))
            Xpno_lists.append(np.empty(0, dtype=np.float64))
            T2_lists.append(np.empty(0, dtype=np.float64))
            continue
        pair_paos_n[kk] = pp_pair.size
        n_pno_arr[kk]   = X_pno_pair.shape[1]
        same_lmo[kk]    = 1 if ii_l == jj_l else 0
        pp_lists.append(np.ascontiguousarray(pp_pair, dtype=np.int64))
        Xpno_lists.append(
            np.ascontiguousarray(X_pno_pair, dtype=np.float64).ravel())
        T2_lists.append(
            np.ascontiguousarray(T2_p, dtype=np.float64).ravel())

    pair_paos_off = np.zeros(n_keys + 1, dtype=np.int64)
    Xpno_off      = np.zeros(n_keys + 1, dtype=np.int64)
    T2_off        = np.zeros(n_keys + 1, dtype=np.int64)
    for kk in range(n_keys):
        pair_paos_off[kk + 1] = pair_paos_off[kk] + pp_lists[kk].size
        Xpno_off[kk + 1]      = Xpno_off[kk]      + Xpno_lists[kk].size
        T2_off[kk + 1]        = T2_off[kk]        + T2_lists[kk].size
    pair_paos_flat = (np.concatenate(pp_lists)   if any(x.size for x in pp_lists)
                      else np.empty(0, dtype=np.int64))
    Xpno_flat      = (np.concatenate(Xpno_lists) if any(x.size for x in Xpno_lists)
                      else np.empty(0, dtype=np.float64))
    T2_flat        = (np.concatenate(T2_lists)   if any(x.size for x in T2_lists)
                      else np.empty(0, dtype=np.float64))

    F_orth_ijk = np.zeros((npao_can_ijk, npao_can_ijk), dtype=np.float64)
    D_ijk      = np.zeros((npao_can_ijk, npao_can_ijk), dtype=np.float64)

    triple_paos_c = np.ascontiguousarray(triple_paos, dtype=np.int64)
    X_pao_ijk_c   = np.ascontiguousarray(X_pao_ijk)
    F_pao_c       = np.ascontiguousarray(F_pao_full)
    S_pao_c       = np.ascontiguousarray(S_pao_full)

    import ctypes as _ct
    from pyscf import lib as _pl
    _libcc_t = getattr(_triple_pno_union_psi4, '_libcc', None)
    if _libcc_t is None:
        _libcc_t = _pl.load_library('libcc')
        _libcc_t.DLPNObuild_triple_tno_body.restype = None
        _libcc_t.DLPNObuild_triple_tno_body.argtypes = (
            [_ct.c_int] * 3                # n_pao_ijk, n_pao_can, n_pao_total
            + [_ct.c_void_p] * 4           # triple_paos, X_pao_ijk, F_pao, S_pao
            + [_ct.c_int]                  # n_keys
            + [_ct.c_void_p] * 9           # arrays + outputs
        )
        _triple_pno_union_psi4._libcc = _libcc_t

    _libcc_t.DLPNObuild_triple_tno_body(
        int(triple_paos.size), int(npao_can_ijk), int(F_pao_full.shape[0]),
        triple_paos_c.ctypes.data_as(_ct.c_void_p),
        X_pao_ijk_c.ctypes.data_as(_ct.c_void_p),
        F_pao_c.ctypes.data_as(_ct.c_void_p),
        S_pao_c.ctypes.data_as(_ct.c_void_p),
        int(n_keys),
        pair_paos_n.ctypes.data_as(_ct.c_void_p),
        pair_paos_off.ctypes.data_as(_ct.c_void_p),
        pair_paos_flat.ctypes.data_as(_ct.c_void_p),
        n_pno_arr.ctypes.data_as(_ct.c_void_p),
        Xpno_off.ctypes.data_as(_ct.c_void_p),
        Xpno_flat.ctypes.data_as(_ct.c_void_p),
        T2_off.ctypes.data_as(_ct.c_void_p),
        T2_flat.ctypes.data_as(_ct.c_void_p),
        same_lmo.ctypes.data_as(_ct.c_void_p),
        F_orth_ijk.ctypes.data_as(_ct.c_void_p),
        D_ijk.ctypes.data_as(_ct.c_void_p),
    )
    tno_occ, tno_vecs = np.linalg.eigh(D_ijk)
    order = np.argsort(tno_occ)[::-1]
    tno_occ = tno_occ[order]
    tno_vecs = tno_vecs[:, order]
    keep = np.abs(tno_occ) >= T_CutTNO
    n_tno = int(np.sum(keep))
    if n_tno == 0:
        n_tno = 1
        keep[0] = True
    X_tno_initial = tno_vecs[:, keep]

    # (6) Canonicalize in F basis
    F_in_tno = X_tno_initial.T @ F_orth_ijk @ X_tno_initial
    eps_tno_sc, tno_canon = np.linalg.eigh(F_in_tno)
    X_tno_canonical = X_tno_initial @ tno_canon

    # (7) X_tno_ijk maps triple's raw PAO domain → canonical TNO
    X_tno_ijk = X_pao_ijk @ X_tno_canonical              # (npao_ijk, n_tno)

    # TNO in AO basis (S-orthonormal by construction)
    C_tno_sc = C_pao[:, triple_paos] @ X_tno_ijk         # (nao, n_tno)

    return C_tno_sc, n_tno, X_tno_ijk, triple_paos, eps_tno_sc


def _preload_df_integrals(with_df):
    """Preload all DF 3-index integrals into memory.

    Returns Lpq_full as a contiguous (naux, nao_pair) array in packed
    triangular format, ready for _ao2mo.nr_e2.  Reading the HDF5 file
    once up-front makes all subsequent integral transforms thread-safe
    and eliminates redundant I/O.

    Pre-allocates the destination and copies chunks straight into it —
    the previous chunks-list + np.vstack roundtrip doubled the memory
    traffic (chunk → list → final array) and the vstack ran serial in
    Python, costing ~0.5 s on water-22 and scaling N^2 with system size.
    """
    naux = with_df.get_naoaux()
    Lpq_full = None
    p1 = 0
    for Lpq in with_df.loop():
        nL = Lpq.shape[0]
        if Lpq_full is None:
            # First chunk reveals nao_pair (size of axis 1).
            Lpq_full = np.empty((naux, Lpq.shape[1]), dtype=Lpq.dtype)
        Lpq_full[p1:p1 + nL] = Lpq
        p1 += nL
    if Lpq_full is None:
        return np.zeros((0, 0), dtype=np.float64)
    return Lpq_full


def _build_ovL_tno(Lpq_full, C_lmo, C_tno, lmo_indices):
    """Build (LMO_i, TNO_a | L) 3-index DF tensor for selected LMOs.

    Args:
        Lpq_full (np.ndarray): (naux, nao_pair) preloaded DF integrals.
        C_lmo (np.ndarray): (nao, nocc_lmo) LMO coefficients.
        C_tno (np.ndarray): (nao, n_tno) TNO coefficients in AO basis.
        lmo_indices (list): LMO orbital indices to compute (e.g. [i, j, k]).

    Returns:
        ovL (np.ndarray): (len(lmo_indices), n_tno, naux)
    """
    n_tno = C_tno.shape[1]
    naux = Lpq_full.shape[0]
    n_sel = len(lmo_indices)

    C_occ_sel = C_lmo[:, lmo_indices]   # (nao, n_sel)
    nmo_sel = n_sel + n_tno
    mo = np.asarray(np.hstack((C_occ_sel, C_tno)), order='F')
    ijslice = (0, n_sel, n_sel, nmo_sel)

    buf = _ao2mo.nr_e2(Lpq_full, mo, ijslice, aosym='s2')
    ovL = buf.reshape(naux, n_sel, n_tno).transpose(1, 2, 0).copy()
    return ovL


def _build_vvL_tno(Lpq_full, C_tno):
    """Build (TNO_a, TNO_b | L) 3-index DF tensor in TNO basis.

    Args:
        Lpq_full (np.ndarray): (naux, nao_pair) preloaded DF integrals.
        C_tno (np.ndarray): (nao, n_tno) TNO coefficients.

    Returns:
        vvL (np.ndarray): (n_tno, n_tno, naux)
    """
    n_tno = C_tno.shape[1]
    naux = Lpq_full.shape[0]

    mo = np.asfortranarray(C_tno)
    ijslice = (0, n_tno, 0, n_tno)

    buf = _ao2mo.nr_e2(Lpq_full, mo, ijslice, aosym='s2')
    vvL = buf.reshape(naux, n_tno, n_tno).transpose(1, 2, 0).copy()
    return vvL


def _build_ooL_triple(Lpq_full, C_lmo, lmo_indices):
    """Build (n_sel, n_sel, naux) occ-occ DF integrals for selected LMO indices.

    Args:
        Lpq_full (np.ndarray): (naux, nao_pair) preloaded DF integrals.
        C_lmo (np.ndarray): (nao, nocc_lmo) LMO coefficients.
        lmo_indices (list): LMO orbital indices [i, j, k] or [i, k].

    Returns:
        ooL (np.ndarray): (n_sel, n_sel, naux) occ-occ DF tensor in LMO basis.
    """
    n_sel = len(lmo_indices)
    naux = Lpq_full.shape[0]
    C_occ_sel = C_lmo[:, lmo_indices]   # (nao, n_sel)
    mo = np.asfortranarray(C_occ_sel)
    ijslice = (0, n_sel, 0, n_sel)

    buf = _ao2mo.nr_e2(Lpq_full, mo, ijslice, aosym='s2')
    ooL = buf.reshape(naux, n_sel, n_sel).transpose(1, 2, 0).copy()
    return ooL


def _w3_intermediate(t2_sc, ovL_sc, ooL_sc, vvL_sc, eps_occ, eps_vir,
                     t1_sc=None, fvo_sc=None, sum_all_occ=False,
                     occ_indices=None,
                     ooL_sc_full=None, t2_sc_full=None,
                     K_ooov=None):
    """Compute (T) energy for one triple i<=j<=k using Eq 53 (Jiang 2024).

    Uses the base W (single occupied assignment, no P_L permutation) with the
    Eq 53 closed-shell antisymmetrizer acting on virtual indices. The factor 6
    (= 2 spin × 3 closed-shell) converts from the spin-orbital formula.

    The vooo (A*t2) term sums over ALL occupied LMOs when ooL_sc_full and
    t2_sc_full are provided, matching Psi4's triplet-domain approach.

    The occupied degeneracy factor 1/(1+δ_ij+δ_jk+δ_ik+2δ_ijδ_jkδ_ik)
    is applied when occ_indices=(i,j,k) is given.

    Args:
        t2_sc: (nocc_t, nocc_t, n, n) T2 in SC occ × SC vir.
        ovL_sc: (nocc_t, n, naux) SC-occ × TNO DF integrals.
        ooL_sc: (nocc_t, nocc_t, naux) or larger for full-occ A*t2.
        vvL_sc: (n, n, naux) TNO × TNO DF integrals.
        eps_occ: SC occupied orbital energies.
        eps_vir: (n,) SC virtual orbital energies.
        occ_indices: (i, j, k) global occupied indices for degeneracy factor.
        ooL_sc_full: (nocc_t, nocc_all, naux) for full-occ vooo term.
        t2_sc_full: (nocc_all, nocc_t, n, n) for full-occ vooo term.

    Returns:
        et_ijk (float): (T) energy contribution including spin factor.
    """
    n = len(eps_vir)
    if n == 0:
        return 0.0

    nocc_triple = ovL_sc.shape[0]
    naux = ovL_sc.shape[2]

    # --- Denominator ---
    D_vir = eps_vir[:, None, None] + eps_vir[None, :, None] + eps_vir[None, None, :]
    D_occ = eps_occ[0] + eps_occ[1] + eps_occ[2]
    D = D_vir - D_occ  # eps_abc - eps_ijk (positive for occupied < virtual)

    # --- Occupied degeneracy factor ---
    if occ_indices is not None:
        i, j, k = occ_indices
        dij = int(i == j); djk = int(j == k); dik = int(i == k)
        occ_denom = 1 + dij + djk + dik + 2 * dij * djk * dik
    else:
        occ_denom = 1

    # --- Build P_L-assembled W (Jiang Eq 47+49) ---
    # P_L simultaneously permutes occupied (i,j,k) and virtual (a,b,c).
    # For each S_3 permutation σ: W[a,b,c] += base(σ(i),σ(j),σ(k))[σ(a),σ(b),σ(c)]
    # The transpose maps base[x,y,z] → W[a,b,c] by inverting the virtual permutation.
    trans = [lambda x: x,                     # (i,j,k) → identity
             lambda x: x.transpose(0,2,1),    # (i,k,j) → (a,c,b)
             lambda x: x.transpose(1,0,2),    # (j,i,k) → (b,a,c)
             lambda x: x.transpose(2,0,1),    # (j,k,i) → (b,c,a)
             lambda x: x.transpose(1,2,0),    # (k,i,j) → (c,a,b)
             lambda x: x.transpose(2,1,0)]    # (k,j,i) → (c,b,a)

    # Precompute K_ab[ip] = Σ_L ovL_sc[ip,a,L] * vvL_sc[b,f,L] for the 3
    # distinct occupied-index slots. The perm loop visits each ip twice,
    # so caching here avoids 3 redundant einsums per triple (×442 triples).
    # tensordot is ~3-5× faster than einsum for simple 3-axis contractions
    # since it goes straight to BLAS without the einsum path planner.
    K_ab_cache = [None, None, None]
    for ip in range(3):
        # shape (n, n, n): indexed [a, f, b]
        t = np.tensordot(ovL_sc[ip], vvL_sc, axes=([1], [2]))
        # reorder to [a, b, f]
        K_ab_cache[ip] = t.transpose(0, 2, 1)

    # Permutation index tables — shared by K_ovvv phase and K_ooov phase.
    p_table = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    _ip_arr = np.array([p[0] for p in p_table])   # (6,) ovL-slot of each perm
    _iq_arr = np.array([p[1] for p in p_table])   # (6,) oo-slot
    _ir_arr = np.array([p[2] for p in p_table])   # (6,) t2-slot

    # --- Phase 1: K_ovvv contribution (Psi4 triples.cc:827-830) ---
    # Per perm (A, B, C): base += K_Avvv[a, b, f] * t2_{CB}[c, f]
    W = np.zeros((n, n, n))
    for pidx in range(6):
        ip, iq, ir = p_table[pidx]
        K_ab = K_ab_cache[ip]
        t2_rq = t2_sc[ir, iq]
        base = (K_ab.reshape(n * n, n) @ t2_rq.T).reshape(n, n, n)
        W += trans[pidx](base)

    # --- Phase 2: K_ooov subtraction (Psi4 triples.cc:832-848) ---
    # Psi4 streams m one at a time: per perm (A, B, C), per m, build
    # T_Am[a,b] from pair (A, m) and subtract T_Am[a,b] × K_{Bo Cv}[m, c].
    # We vectorize the m-sum (and batch across the 6 perms) into one
    # matmul: sub[p, a, b, c] = Σ_m K_ooov[ip,iq,a,m] · t2[m, ir, b, c].
    if K_ooov is not None and t2_sc_full is not None:
        K_batch = K_ooov[_ip_arr, _iq_arr]                         # (6, n, m_dom)
        t2_batch = t2_sc_full[:, _ir_arr].transpose(1, 0, 2, 3)    # (6, m_dom, n, n)
        m_dom = K_batch.shape[2]
        sub = np.matmul(
            K_batch, t2_batch.reshape(6, m_dom, n * n)
        ).reshape(6, n, n, n)
        # Apply per-perm virtual-index transpose (inverts P_L) and subtract.
        W -= sub[0]
        W -= sub[1].transpose(0, 2, 1)
        W -= sub[2].transpose(1, 0, 2)
        W -= sub[3].transpose(2, 0, 1)
        W -= sub[4].transpose(1, 2, 0)
        W -= sub[5].transpose(2, 1, 0)
    else:
        # Fallback for legacy callers (no pre-built K_ooov, or in-block-only).
        for pidx in range(6):
            ip, iq, ir = p_table[pidx]
            if ooL_sc_full is not None and t2_sc_full is not None:
                A_al = ovL_sc[ip] @ ooL_sc_full[iq].T
                t2_mbc = t2_sc_full[:, ir]
            else:
                A_al = ovL_sc[ip] @ ooL_sc[iq].T
                t2_mbc = t2_sc[:, ir]
            m = t2_mbc.shape[0]
            W -= trans[pidx](
                (A_al @ t2_mbc.reshape(m, n * n)).reshape(n, n, n))

    # --- T = -W/D ---
    T = -W / D

    # --- V = W + T1 disconnected (Psi4 lines 930-962) ---
    V = W.copy()
    if t1_sc is not None and fvo_sc is not None:
        K_jk = ovL_sc[1] @ ovL_sc[2].T
        K_ik = ovL_sc[0] @ ovL_sc[2].T
        K_ij = ovL_sc[0] @ ovL_sc[1].T
        V += (t1_sc[0][:, None, None] * K_jk[None, :, :]
              + t1_sc[1][None, :, None] * K_ik[:, None, :]
              + t1_sc[2][None, None, :] * K_ij[:, :, None])

    # --- Energy: Eq 53 antisymmetrizer on virtual indices ---
    et = (8 * np.sum(V * T)
          - 4 * np.sum(V.transpose(2, 1, 0) * T)
          - 4 * np.sum(V.transpose(0, 2, 1) * T)
          - 4 * np.sum(V.transpose(1, 0, 2) * T)
          + 2 * np.sum(V.transpose(1, 2, 0) * T)
          + 2 * np.sum(V.transpose(2, 0, 1) * T))

    return float(et / occ_denom)


def _zero_cas_t2_amplitudes(t2_pno_all, pno_spaces, occ_cas_idx, C_cas_vir,
                            s1e, cas_proj_thresh=0.5):
    """Zero T2 amplitudes with CAS virtual character for TCC (T) calculation.

    Per Lang et al. 2020, Sec. II.C: to prevent double-counting of static
    correlation, all active single and double amplitudes are set to zero
    before computing the (T) correction.

    The PNO construction (pno.py) structures CAS-occupied pair PNOs as
    C_pno = [C_cas_vir | C_ext_pno]  (Lang et al. eq. 10: S^ij = I_NCAS ⊕ d^ij).
    The first n_cas_vir PNOs ARE the CAS virtuals; we zero exactly those
    rows/columns.  For non-CAS pairs, the PNOs are pure external and
    nothing is zeroed.

    Args:
        t2_pno_all (dict): {(i,j): t2_array} T2 amplitudes in PNO basis.
        pno_spaces (dict): PNO information from make_pnos.
        occ_cas_idx (array-like): Indices of CAS occupied LMOs.
        C_cas_vir (np.ndarray): (nao, n_cas_vir) CAS virtual MO coefficients.
        s1e (np.ndarray): (nao, nao) AO overlap matrix.
        cas_proj_thresh (float): Unused (kept for API compatibility).

    Returns:
        t2_zeroed (dict): Copy of t2_pno_all with CAS components zeroed.
    """
    occ_cas_set = set(int(x) for x in occ_cas_idx) if occ_cas_idx is not None else set()
    n_cas_vir = C_cas_vir.shape[1] if C_cas_vir is not None else 0
    t2_zeroed = {}

    for pair_key, t2 in t2_pno_all.items():
        i, j = pair_key

        # Only CAS-occupied pairs have the [CAS_vir | ext_PNO] structure.
        # Zero only the pure CAS-CAS block (both virtual indices in CAS).
        # Mixed CAS-ext amplitudes are part of T_ext and must be kept —
        # they have at least one index outside the CAS.
        if (n_cas_vir > 0
                and i in occ_cas_set and j in occ_cas_set
                and t2.shape[0] >= n_cas_vir):
            t2_new = t2.copy()
            t2_new[:n_cas_vir, :n_cas_vir] = 0.0
            t2_zeroed[pair_key] = t2_new
        else:
            t2_zeroed[pair_key] = t2.copy()

    return t2_zeroed


def _build_triple_local_DF(i, j, k, X_tno_ijk, triple_paos, triple_domain,
                            sparse_df, screening, j2c_full, lmo_aux_mask):
    """Psi4-style per-triple DF integrals on a local aux domain.

    Mirrors Psi4 DLPNOCCSD_T::compute_lccsd_t0 (triples.cc:667+). Aux Q's
    are grouped by atom center; each center's Q-stack is contracted in
    one batched matmul, amortizing BLAS-3 across the n_aux_at_A Q's of
    each atom (same pattern as compute_cc_integrals for CCSD pairs).

      q_iv[q, a_tno] = Σ_{u ∈ paos_ext[Q] ∩ triple_paos}
                          qia[Q][i_sparse, u_in_Q]
                          * X_tno_ijk[u_in_triple_paos, a_tno]

      q_vv[q, a, b]  = Σ_{u, v ∈ paos_ext[Q] ∩ triple_paos}
                          X_tno_ijk[u_tp, a]
                          * qab[Q][u_in_Q, v_in_Q]
                          * X_tno_ijk[v_tp, b]

      q_io[q, m]     = qij[Q][i_sparse, m_sparse]

    After the per-center loop, apply local J^{-1/2}.

    Args:
        X_tno_ijk: (npao_ijk, n_tno) — triple's raw-PAO → canonical-TNO
            transform (from _triple_pno_union_psi4).
        triple_paos: (npao_ijk,) global PAO indices in the triple.
        triple_domain: list of global LMO indices (sorted) for ooL rows.

    Returns:
        ovL_sc  (3, n_tno, naux_ijk), vvL_sc (n_tno, n_tno, naux_ijk),
        ooL_sc  (3, n_domain, naux_ijk), all fitted with local J^{-1/2}.
    """
    qij_atom = sparse_df.get('qij_atom')
    qia_atom = sparse_df.get('qia_atom')
    qab_atom = sparse_df.get('qab_atom')
    aux_pos_in_atom = sparse_df.get('aux_pos_in_atom')
    aux_atom_ids = sparse_df.get('aux_atom_ids')
    riatom_to_lmos_ext_dense = screening['riatom_to_lmos_ext_dense']
    riatom_to_paos_ext_dense = screening['riatom_to_paos_ext_dense']

    aux_mask = lmo_aux_mask[i] | lmo_aux_mask[j] | lmo_aux_mask[k]
    aux_idx = np.where(aux_mask)[0]
    naux_ijk = aux_idx.size
    n_tno = X_tno_ijk.shape[1]
    n_domain = len(triple_domain)

    if naux_ijk == 0 or n_tno == 0:
        return (np.zeros((3, n_tno, 0)),
                np.zeros((n_tno, n_tno, 0)),
                np.zeros((3, n_domain, 0)))

    # Local J^{-1/2}: stays in NumPy (LAPACK eigh is already optimal).
    j_loc = j2c_full[np.ix_(aux_idx, aux_idx)]
    evals, evecs = np.linalg.eigh(j_loc)
    keep_e = evals > 1e-14
    jhi = (evecs[:, keep_e] * (1.0 / np.sqrt(evals[keep_e]))
           ) @ evecs[:, keep_e].T
    jhi_c = np.ascontiguousarray(jhi)

    # Per-center metadata: group aux_idx by centerQ.
    centers_of_aux = aux_atom_ids[aux_idx]                  # (naux_ijk,)
    atom_pos_all = aux_pos_in_atom[aux_idx]                 # (naux_ijk,)
    order = np.argsort(centers_of_aux, kind='stable')
    centers_sorted = centers_of_aux[order]
    local_Q_sorted = order.astype(np.int64)                 # positions in naux_ijk
    atom_pos_sorted = atom_pos_all[order].astype(np.int64)
    unique_centers, first_idx, counts = np.unique(
        centers_sorted, return_index=True, return_counts=True)
    center_atoms_arr = unique_centers.astype(np.int64)
    center_off = np.empty(unique_centers.size + 1, dtype=np.int64)
    center_off[0] = 0
    center_off[1:] = np.cumsum(counts.astype(np.int64))

    # Outputs (allocate row-major).
    ovL_sc = np.zeros((3, n_tno, naux_ijk), dtype=np.float64)
    vvL_sc = np.zeros((n_tno, n_tno, naux_ijk), dtype=np.float64)
    ooL_sc = np.zeros((3, n_domain, naux_ijk), dtype=np.float64)

    triple_paos_arr = np.ascontiguousarray(triple_paos, dtype=np.int64)
    triple_domain_arr = np.ascontiguousarray(triple_domain, dtype=np.int64)
    X_tno_c = np.ascontiguousarray(X_tno_ijk)

    n_lmo_global = riatom_to_lmos_ext_dense.shape[1]
    n_pao_global = riatom_to_paos_ext_dense.shape[1]
    lmos_dense_c = np.ascontiguousarray(
        riatom_to_lmos_ext_dense, dtype=np.int64)
    paos_dense_c = np.ascontiguousarray(
        riatom_to_paos_ext_dense, dtype=np.int64)

    import ctypes as _ct
    from pyscf import lib as _pl
    _libcc_tdf = getattr(_build_triple_local_DF, '_libcc', None)
    if _libcc_tdf is None:
        _libcc_tdf = _pl.load_library('libcc')
        _libcc_tdf.DLPNObuild_triple_local_DF.restype = None
        _libcc_tdf.DLPNObuild_triple_local_DF.argtypes = (
            [_ct.c_int] * 10                  # i,j,k,n_tno,n_domain,n_pao_ijk,naux_ijk,n_centers,n_lmo_g,n_pao_g
            + [_ct.c_void_p] * 21
        )
        _build_triple_local_DF._libcc = _libcc_tdf

    _libcc_tdf.DLPNObuild_triple_local_DF(
        int(i), int(j), int(k),
        int(n_tno), int(n_domain), int(triple_paos_arr.size),
        int(naux_ijk), int(center_atoms_arr.size),
        int(n_lmo_global), int(n_pao_global),
        X_tno_c.ctypes.data_as(_ct.c_void_p),
        triple_paos_arr.ctypes.data_as(_ct.c_void_p),
        triple_domain_arr.ctypes.data_as(_ct.c_void_p),
        center_atoms_arr.ctypes.data_as(_ct.c_void_p),
        center_off.ctypes.data_as(_ct.c_void_p),
        local_Q_sorted.ctypes.data_as(_ct.c_void_p),
        atom_pos_sorted.ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_off'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qia_atom_off'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qab_atom_off'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_n_aux'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_n_lmo'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qab_atom_n_pao'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_flat'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qia_atom_flat'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qab_atom_flat'].ctypes.data_as(_ct.c_void_p),
        lmos_dense_c.ctypes.data_as(_ct.c_void_p),
        paos_dense_c.ctypes.data_as(_ct.c_void_p),
        jhi_c.ctypes.data_as(_ct.c_void_p),
        ovL_sc.ctypes.data_as(_ct.c_void_p),
        vvL_sc.ctypes.data_as(_ct.c_void_p),
        ooL_sc.ctypes.data_as(_ct.c_void_p),
    )
    return ovL_sc, vvL_sc, ooL_sc


# Per-LMO domain-partner sets cache. The (T) per-triple `triple_domain`
# scan in `_orch_phase1`, `_orch`, and `_process_one_triple` was
# O(nocc_lmo) per triple, which makes (T) wall scale ≈ O(N²) on top of
# the triple-count growth. Replace with `partners[i] & partners[j] &
# partners[k]` — a 3-way set intersection of bounded local neighbourhoods.
# Mirrors the fix applied to `_run_triples_omp` in commit a771bc241; this
# extends it to the default pool.map path.
_partners_cache = {}

def _build_partners(domain_set, nocc_lmo):
    """Return per-LMO partner sets: partners[i] = {m : (m, i) ∈ domain_set}.

    Cached by id(domain_set); the same set is reused across all triples
    in a (T) pass, so the O(|domain_set|) build runs once.
    """
    key = id(domain_set)
    p = _partners_cache.get(key)
    if p is not None and len(p) == nocc_lmo:
        return p
    p = [set() for _ in range(nocc_lmo)]
    for (a, b) in domain_set:
        p[a].add(b)
        p[b].add(a)
    _partners_cache[key] = p
    return p


# Lock for protecting the global pair arena build from cache-races.
# Without it, all 32 pool threads see `_pair_arena = None` on their first
# call and each redundantly rebuilds (32 × ~1.5s = ~48s of wasted work
# at water-22, dominating the (T) wall).
_ORCH_ARENA_LOCK = __import__('threading').Lock()


def _build_pair_arena(_gak, pno_spaces, t2_for_T):
    """Build the cycle-invariant flat-buffer arena over all pairs.
    Called once per (T) phase (under _ORCH_ARENA_LOCK to prevent races
    between pool threads). Returns the gcache tuple consumed by
    `_orch`.
    """
    all_pks = sorted(pno_spaces.keys())
    pair_to_idx = {pk: idx for idx, pk in enumerate(all_pks)}
    n_pairs = len(all_pks)
    g_pao_n = np.zeros(n_pairs, dtype=np.int32)
    g_pno_n = np.zeros(n_pairs, dtype=np.int32)
    for p, pk in enumerate(all_pks):
        pd = pno_spaces[pk]
        pp_arr = pd.get('pair_paos')
        xp_arr = pd.get('X_pno')
        if pp_arr is not None:
            g_pao_n[p] = int(np.asarray(pp_arr).size)
        if xp_arr is not None:
            g_pno_n[p] = int(xp_arr.shape[1])
    g_pp_off = np.zeros(n_pairs + 1, dtype=np.int64)
    g_pp_off[1:] = np.cumsum(g_pao_n.astype(np.int64))
    g_X_off = np.zeros(n_pairs + 1, dtype=np.int64)
    g_X_off[1:] = np.cumsum(
        g_pao_n.astype(np.int64) * g_pno_n.astype(np.int64))
    g_T2_off = np.zeros(n_pairs + 1, dtype=np.int64)
    g_T2_off[1:] = np.cumsum(
        g_pno_n.astype(np.int64) * g_pno_n.astype(np.int64))
    g_pp_flat = np.empty(int(g_pp_off[-1]), dtype=np.int64)
    g_X_flat  = np.empty(int(g_X_off[-1]), dtype=np.float64)
    g_T2_flat = np.zeros(int(g_T2_off[-1]), dtype=np.float64)
    for p, pk in enumerate(all_pks):
        pd = pno_spaces[pk]
        pp_arr = pd.get('pair_paos')
        xp_arr = pd.get('X_pno')
        if pp_arr is not None and g_pao_n[p] > 0:
            g_pp_flat[g_pp_off[p]:g_pp_off[p + 1]] = np.asarray(
                pp_arr, dtype=np.int64)
        if (xp_arr is not None and g_pao_n[p] > 0
                and g_pno_n[p] > 0):
            g_X_flat[g_X_off[p]:g_X_off[p + 1]] = np.ascontiguousarray(
                xp_arr).ravel()
        if pk in t2_for_T and g_pno_n[p] > 0:
            g_T2_flat[g_T2_off[p]:g_T2_off[p + 1]] = np.ascontiguousarray(
                t2_for_T[pk]).ravel()
    # NOTE: an attempt to de-duplicate the amplitude/PNO storage here (repoint
    # the per-pair pno_spaces/t2_for_T entries to zero-copy views into this flat
    # arena and free the originals) was numerically neutral but gave NO RSS
    # reduction: the freed per-pair arrays are small heap allocations (~tens of
    # KB) that glibc retains rather than returning to the OS, and the (T)-entry
    # base is dominated by glibc-retained-free CCSD pages (already reused by the
    # energy pass), not live amplitude data.  Removed — see STREAMING_DEV.md.
    return (_gak, pair_to_idx, g_pao_n, g_pno_n,
            g_pp_off, g_pp_flat, g_X_off, g_X_flat,
            g_T2_off, g_T2_flat)
# Active (pair -> id) map for the C-level (pair, RI-atom) H-cache.  Set by
# run_lccsd_t_ext around the TIGHT energy pass only (the prescreen pass uses
# different loose integral stacks and must never share cache entries), read
# concurrently by pool workers inside _orch.  None = cache disabled.
_QVV_PAIR_ID_MAP = None


def _orch(i, j, k, pno_spaces, t2_for_T,
                C_pao, S_pao_full, F_pao_full, F_lmo,
                sparse_df, screening, j2c_full, lmo_aux_mask,
                pao_domains_triple, nonneg_set, T_CutTNO,
                t1_pno):
    """Phase 3c-2/3 helper: full per-triple body in one C call.

    Calls DLPNOcompute_one_triple_E_T0 which does TNO + DF + U cache +
    t2_block + K_ab + K_ooov + K_*_for_V + W3 + energy in one shot.
    Returns et_ijk.
    """
    import time as _orch_time
    _t_start = 0.0
    ij = (min(i, j), max(i, j))
    ik = (min(i, k), max(i, k))
    jk = (min(j, k), max(j, k))
    if (ij not in pno_spaces or ik not in pno_spaces or jk not in pno_spaces):
        return 0.0
    if (ij not in t2_for_T or ik not in t2_for_T or jk not in t2_for_T):
        return 0.0

    # 1. triple_paos
    if pao_domains_triple is not None:
        pao_set = set()
        for lmo in (i, j, k):
            pao_set.update(int(x) for x in
                           np.asarray(pao_domains_triple[lmo]).tolist())
    else:
        pao_set = set()
        for key in (ij, jk, ik):
            pp = pno_spaces[key].get('pair_paos')
            if pp is not None:
                pao_set.update(int(x) for x in np.asarray(pp).tolist())
    if not pao_set:
        return 0.0
    triple_paos = np.array(sorted(pao_set), dtype=np.int64)
    n_pao_ijk = int(triple_paos.size)
    n_pao_total = int(F_pao_full.shape[0])
    nocc_lmo = int(F_lmo.shape[0])
    naux_total = int(j2c_full.shape[0])

    # 2. triple_domain — partner-set intersection (was O(nocc_lmo) scan)
    _domain_set = nonneg_set if nonneg_set is not None else set(t2_for_T.keys())
    _p = _build_partners(_domain_set, nocc_lmo)
    triple_domain = sorted(_p[i] & _p[j] & _p[k])
    triple_domain_arr = np.asarray(triple_domain, dtype=np.int64)
    n_dom = int(triple_domain_arr.size)
    _t_s1 = 0.0

    # 3. 3-pair arena
    keys_3 = [ij, jk, ik]
    _pm = _QVV_PAIR_ID_MAP
    pair_ids_3 = np.array([(_pm.get(key, -1) if _pm is not None else -1)
                           for key in keys_3], dtype=np.int64)
    ij_lmos = [(i, j), (j, k), (i, k)]
    pair_paos_n_3 = np.zeros(3, dtype=np.int32)
    n_pno_arr_3   = np.zeros(3, dtype=np.int32)
    same_lmo_3    = np.zeros(3, dtype=np.int32)
    pp_lists, X_lists, T2_lists = [], [], []
    for kk, (key, (ii_l, jj_l)) in enumerate(zip(keys_3, ij_lmos)):
        Xp = pno_spaces[key].get('X_pno')
        pp = np.asarray(pno_spaces[key].get('pair_paos'))
        T2_p = t2_for_T[key]
        if Xp is None or Xp.shape[1] == 0 or pp.size == 0 or T2_p.size == 0:
            pp_lists.append(np.empty(0, dtype=np.int64))
            X_lists.append(np.empty(0, dtype=np.float64))
            T2_lists.append(np.empty(0, dtype=np.float64))
            continue
        pair_paos_n_3[kk] = pp.size
        n_pno_arr_3[kk]   = Xp.shape[1]
        same_lmo_3[kk]    = 1 if ii_l == jj_l else 0
        pp_lists.append(np.ascontiguousarray(pp, dtype=np.int64))
        X_lists.append(np.ascontiguousarray(Xp, dtype=np.float64).ravel())
        T2_lists.append(np.ascontiguousarray(T2_p, dtype=np.float64).ravel())
    pair_paos_off_3 = np.zeros(4, dtype=np.int64)
    X_pno_off_3     = np.zeros(4, dtype=np.int64)
    T2_off_3        = np.zeros(4, dtype=np.int64)
    for kk in range(3):
        pair_paos_off_3[kk + 1] = pair_paos_off_3[kk] + pp_lists[kk].size
        X_pno_off_3[kk + 1]     = X_pno_off_3[kk]     + X_lists[kk].size
        T2_off_3[kk + 1]        = T2_off_3[kk]        + T2_lists[kk].size
    pair_paos_flat_3 = (np.concatenate(pp_lists) if any(x.size for x in pp_lists)
                        else np.empty(0, dtype=np.int64))
    X_pno_flat_3 = (np.concatenate(X_lists) if any(x.size for x in X_lists)
                    else np.empty(0, dtype=np.float64))
    T2_flat_3 = (np.concatenate(T2_lists) if any(x.size for x in T2_lists)
                 else np.empty(0, dtype=np.float64))
    _t_s2 = 0.0

    # 4. u_pks list — covers (r,m), (r,r), (r,r2) for r in [i,j,k], m in domain
    triple_lmo = [i, j, k]
    _u_pks_seen = set()
    _u_pks = []
    def _add(pk):
        if pk in _u_pks_seen: return
        if pk not in pno_spaces: return
        pd = pno_spaces[pk]
        if pd.get('X_pno') is None or pd['X_pno'].shape[1] == 0: return
        if pk not in t2_for_T: return
        _u_pks_seen.add(pk); _u_pks.append(pk)
    for r in triple_lmo:
        for m in triple_domain:
            _add((min(r, m), max(r, m)))
        _add((r, r))
    for r in triple_lmo:
        for r2 in triple_lmo:
            _add((min(r, r2), max(r, r2)))
    n_u_pks = len(_u_pks)
    u_pk_to_idx = {pk: idx for idx, pk in enumerate(_u_pks)}

    if n_u_pks == 0:
        return 0.0
    _t_s3 = 0.0

    # Global pair arena — cached across triples within a CCSD run.
    # Eliminates the per-triple ~2 MB X_pno/T2 marshalling that was the
    # bottleneck (139 ms/call → expect <30 ms/call).  Per-triple offsets
    # then point INTO the global flat arrays — no data copy.
    _gak = (id(pno_spaces), id(t2_for_T))
    gcache = getattr(_orch, '_pair_arena', None)
    if gcache is None or gcache[0] != _gak:
        # Double-checked locking: serialize the build so only one thread
        # does it; all others see the cached result.
        with _ORCH_ARENA_LOCK:
            gcache = getattr(_orch, '_pair_arena', None)
            if gcache is None or gcache[0] != _gak:
                gcache = _build_pair_arena(_gak, pno_spaces, t2_for_T)
                _orch._pair_arena = gcache
    (_, pair_to_idx,
     g_pao_n, g_pno_n,
     g_pp_off, g_pp_flat, g_X_off, g_X_flat,
     g_T2_off, g_T2_flat) = gcache
    _t_s4 = 0.0

    # Per-triple: just look up offsets into the global arena.  No data copy.
    u_pao_n = np.empty(n_u_pks, dtype=np.int32)
    u_pno_n = np.empty(n_u_pks, dtype=np.int32)
    u_pp_off = np.empty(n_u_pks + 1, dtype=np.int64)
    u_X_off = np.empty(n_u_pks + 1, dtype=np.int64)
    u_T2_off = np.empty(n_u_pks + 1, dtype=np.int64)
    for p, pk_ in enumerate(_u_pks):
        idx = pair_to_idx[pk_]
        u_pao_n[p] = g_pao_n[idx]
        u_pno_n[p] = g_pno_n[idx]
        u_pp_off[p] = g_pp_off[idx]
        u_X_off[p]  = g_X_off[idx]
        u_T2_off[p] = g_T2_off[idx]
    # Last entry (for cumulative-bound checks in C — kernels read up to
    # offsets[n_u_pks-1] + size, never offsets[n_u_pks]).  Keep matching
    # patterns just in case.
    u_pp_off[n_u_pks] = g_pp_off[-1]
    u_X_off[n_u_pks]  = g_X_off[-1]
    u_T2_off[n_u_pks] = g_T2_off[-1]
    # Aliases (no copy) to global flat arrays.
    u_pp_flat = g_pp_flat
    u_X_flat  = g_X_flat
    u_T2_flat = g_T2_flat

    # 5. Index tables for t2_block, w3, t1
    t2_block_idx = np.full(9, -1, dtype=np.int32)
    t2_block_tflag = np.zeros(9, dtype=np.int8)
    for p in range(3):
        for q in range(3):
            lp, lq = triple_lmo[p], triple_lmo[q]
            pk = (min(lp, lq), max(lp, lq))
            if pk in u_pk_to_idx:
                t2_block_idx[p*3 + q] = u_pk_to_idx[pk]
                t2_block_tflag[p*3 + q] = 1 if lp > lq else 0
    w3_idx = np.full(3 * n_dom, -1, dtype=np.int32)
    w3_tflag = np.zeros(3 * n_dom, dtype=np.int8)
    for r in range(3):
        lr = triple_lmo[r]
        for l_local in range(n_dom):
            ll = triple_domain[l_local]
            pk = (min(lr, ll), max(lr, ll))
            if pk in u_pk_to_idx:
                w3_idx[r * n_dom + l_local] = u_pk_to_idx[pk]
                w3_tflag[r * n_dom + l_local] = 1 if ll > lr else 0
    t1_diag_idx = np.full(3, -1, dtype=np.int32)
    for r in range(3):
        pk = (triple_lmo[r], triple_lmo[r])
        if pk in u_pk_to_idx:
            t1_diag_idx[r] = u_pk_to_idx[pk]
    t1_lmo_idx_arr = np.array(triple_lmo, dtype=np.int64)

    # 6. t1 flat (per-LMO) — cached on _orch across triples
    has_t1 = 1 if t1_pno is not None else 0
    if has_t1:
        cached_t1 = getattr(_orch, '_t1_cache', None)
        if cached_t1 is None or cached_t1[0] is not t1_pno:
            t1_sizes = np.zeros(nocc_lmo, dtype=np.int64)
            for lmo, vec in t1_pno.items():
                if vec is not None:
                    t1_sizes[lmo] = vec.size
            t1_off = np.empty(nocc_lmo + 1, dtype=np.int64); t1_off[0] = 0
            t1_off[1:] = np.cumsum(t1_sizes)
            t1_flat = np.zeros(int(t1_off[-1]), dtype=np.float64)
            for lmo, vec in t1_pno.items():
                if vec is not None and vec.size:
                    t1_flat[t1_off[lmo]:t1_off[lmo + 1]] = vec
            cached_t1 = (t1_pno, t1_off, t1_flat)
            _orch._t1_cache = cached_t1
        _, t1_off, t1_flat = cached_t1
    else:
        t1_off = np.zeros(nocc_lmo + 1, dtype=np.int64)
        t1_flat = np.zeros(1, dtype=np.float64)

    # 7. occ degeneracy + eps
    dij = int(i == j); djk = int(j == k); dik = int(i == k)
    occ_denom = 1 + dij + djk + dik + 2 * dij * djk * dik
    eps_i = float(F_lmo[i, i])
    eps_j = float(F_lmo[j, j])
    eps_k = float(F_lmo[k, k])

    # 8. Globals — cache across triples (heavy bool→int64 copies otherwise)
    _ck = (id(j2c_full), id(F_pao_full), id(S_pao_full),
           id(lmo_aux_mask), id(sparse_df), id(screening))
    cached = getattr(_orch, '_globals_cache', None)
    if cached is None or cached[0] != _ck:
        cached = (_ck,
                  np.ascontiguousarray(F_pao_full),
                  np.ascontiguousarray(S_pao_full),
                  np.ascontiguousarray(j2c_full),
                  np.ascontiguousarray(sparse_df['aux_atom_ids'], dtype=np.int64),
                  np.ascontiguousarray(sparse_df['aux_pos_in_atom'], dtype=np.int64),
                  np.ascontiguousarray(
                      screening['riatom_to_lmos_ext_dense'], dtype=np.int64),
                  np.ascontiguousarray(
                      screening['riatom_to_paos_ext_dense'], dtype=np.int64),
                  np.ascontiguousarray(lmo_aux_mask.astype(np.int64)))
        _orch._globals_cache = cached
    (_, F_pao_c, S_pao_c, j2c_c, aux_atom_ids, aux_pos_in_atom,
     lmo_dense_c, pao_dense_c, lmo_aux_mask_c) = cached
    triple_paos_c = np.ascontiguousarray(triple_paos, dtype=np.int64)
    _t_s10 = 0.0

    # 9. ctypes setup + call
    import ctypes as _ct
    from pyscf import lib as _pl
    _libcc = getattr(_orch, '_libcc', None)
    if _libcc is None:
        _libcc = _pl.load_library('libcc')
        _libcc.DLPNOcompute_one_triple_E_T0.restype = _ct.c_double
        _libcc.DLPNOcompute_one_triple_E_T0.argtypes = (
            [_ct.c_int] * 3                # i, j, k
            + [_ct.c_int]                  # n_pao_ijk
            + [_ct.c_void_p]               # triple_paos
            + [_ct.c_int]                  # n_dom
            + [_ct.c_void_p]               # triple_domain
            + [_ct.c_void_p] * 9           # 3-pair: 9 ptrs
            + [_ct.c_int]                  # n_u_pks
            + [_ct.c_void_p] * 8           # u_pao_n, u_pno_n, u_pp_off, u_pp_flat,
                                           # u_X_off, u_X_flat, u_T2_off, u_T2_flat
            + [_ct.c_void_p] * 2           # t2_block_idx, t2_block_tflag
            + [_ct.c_void_p] * 2           # w3_idx, w3_tflag
            + [_ct.c_int]                  # has_t1
            + [_ct.c_void_p] * 4           # t1_off, t1_flat, t1_diag_idx, t1_lmo_idx
            + [_ct.c_double, _ct.c_double, _ct.c_double]  # eps_i, eps_j, eps_k
            + [_ct.c_int]                  # occ_denom
            + [_ct.c_int] * 3              # nocc_lmo, n_pao_total, naux_total
            + [_ct.c_void_p] * 14          # F_pao, S_pao, j2c, aux_atom_ids, aux_pos,
                                           # qij_off, qia_off, qab_off,
                                           # qij_n_aux, qij_n_lmo, qab_n_pao,
                                           # qij_flat, qia_flat, qab_flat
            + [_ct.c_void_p] * 3           # lmo_dense, pao_dense, lmo_aux_mask
            + [_ct.c_double, _ct.c_double] # T_CutTNO, S_cut_domain
            + [_ct.c_int]                  # pre_n_tno (0 = compute internally)
            + [_ct.c_void_p] * 2           # pre_X_tno_ijk, pre_eps_tno (NULL)
            + [_ct.c_void_p]               # pair_ids_3 (H-cache; -1s disable)
        )
        _orch._libcc = _libcc

    et = _libcc.DLPNOcompute_one_triple_E_T0(
        int(i), int(j), int(k),
        int(n_pao_ijk), triple_paos_c.ctypes.data_as(_ct.c_void_p),
        int(n_dom), triple_domain_arr.ctypes.data_as(_ct.c_void_p),
        pair_paos_n_3.ctypes.data_as(_ct.c_void_p),
        pair_paos_off_3.ctypes.data_as(_ct.c_void_p),
        pair_paos_flat_3.ctypes.data_as(_ct.c_void_p),
        n_pno_arr_3.ctypes.data_as(_ct.c_void_p),
        X_pno_off_3.ctypes.data_as(_ct.c_void_p),
        X_pno_flat_3.ctypes.data_as(_ct.c_void_p),
        T2_off_3.ctypes.data_as(_ct.c_void_p),
        T2_flat_3.ctypes.data_as(_ct.c_void_p),
        same_lmo_3.ctypes.data_as(_ct.c_void_p),
        int(n_u_pks),
        u_pao_n.ctypes.data_as(_ct.c_void_p),
        u_pno_n.ctypes.data_as(_ct.c_void_p),
        u_pp_off.ctypes.data_as(_ct.c_void_p),
        u_pp_flat.ctypes.data_as(_ct.c_void_p),
        u_X_off.ctypes.data_as(_ct.c_void_p),
        u_X_flat.ctypes.data_as(_ct.c_void_p),
        u_T2_off.ctypes.data_as(_ct.c_void_p),
        u_T2_flat.ctypes.data_as(_ct.c_void_p),
        t2_block_idx.ctypes.data_as(_ct.c_void_p),
        t2_block_tflag.ctypes.data_as(_ct.c_void_p),
        w3_idx.ctypes.data_as(_ct.c_void_p),
        w3_tflag.ctypes.data_as(_ct.c_void_p),
        int(has_t1),
        t1_off.ctypes.data_as(_ct.c_void_p),
        t1_flat.ctypes.data_as(_ct.c_void_p),
        t1_diag_idx.ctypes.data_as(_ct.c_void_p),
        t1_lmo_idx_arr.ctypes.data_as(_ct.c_void_p),
        eps_i, eps_j, eps_k,
        int(occ_denom),
        int(nocc_lmo), int(n_pao_total), int(naux_total),
        F_pao_c.ctypes.data_as(_ct.c_void_p),
        S_pao_c.ctypes.data_as(_ct.c_void_p),
        j2c_c.ctypes.data_as(_ct.c_void_p),
        aux_atom_ids.ctypes.data_as(_ct.c_void_p),
        aux_pos_in_atom.ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_off'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qia_atom_off'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qab_atom_off'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_n_aux'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_n_lmo'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qab_atom_n_pao'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_flat'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qia_atom_flat'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qab_atom_flat'].ctypes.data_as(_ct.c_void_p),
        lmo_dense_c.ctypes.data_as(_ct.c_void_p),
        pao_dense_c.ctypes.data_as(_ct.c_void_p),
        lmo_aux_mask_c.ctypes.data_as(_ct.c_void_p),
        float(T_CutTNO), float(1e-8),
        0, None, None,  # pre_n_tno=0 (compute TNO internally)
        pair_ids_3.ctypes.data_as(_ct.c_void_p),
    )
    return float(et)


def _run_triples_omp(valid_triples,
                      pno_spaces, t2_for_T,
                      C_pao, S_pao_full, F_pao_full, F_lmo,
                      sparse_df, screening, j2c_full, lmo_aux_mask,
                      pao_domains_triple, nonneg_set, T_CutTNO,
                      t1_pno):
    """Phase 3c-4: full OMP-over-triples driver.

    Builds the per-CCSD pair arena once, flattens per-triple data into
    arrays, and calls DLPNOcompute_E_T0_omp ONCE to compute et_per_triple
    for the entire triples list.  Replaces pool.map dispatch.
    """
    import time as _t
    _prof = (False)
    _t_start = 0.0
    n_triples = len(valid_triples)
    if n_triples == 0:
        return np.zeros(0)
    nocc_lmo = int(F_lmo.shape[0])
    n_pao_total = int(F_pao_full.shape[0])
    naux_total = int(j2c_full.shape[0])
    _domain_set = nonneg_set if nonneg_set is not None else set(t2_for_T.keys())

    # Build (or reuse) global pair arena.
    _gak = (id(pno_spaces), id(t2_for_T))
    gcache = getattr(_run_triples_omp, '_pair_arena', None)
    if gcache is None or gcache[0] != _gak:
        all_pks = sorted(pno_spaces.keys())
        pair_to_idx = {pk: idx for idx, pk in enumerate(all_pks)}
        n_pairs = len(all_pks)
        g_pao_n = np.zeros(n_pairs, dtype=np.int32)
        g_pno_n = np.zeros(n_pairs, dtype=np.int32)
        for p, pk in enumerate(all_pks):
            pd = pno_spaces[pk]
            pp_arr = pd.get('pair_paos')
            xp_arr = pd.get('X_pno')
            if pp_arr is not None:
                g_pao_n[p] = int(np.asarray(pp_arr).size)
            if xp_arr is not None:
                g_pno_n[p] = int(xp_arr.shape[1])
        g_pp_off = np.zeros(n_pairs + 1, dtype=np.int64)
        g_pp_off[1:] = np.cumsum(g_pao_n.astype(np.int64))
        g_X_off = np.zeros(n_pairs + 1, dtype=np.int64)
        g_X_off[1:] = np.cumsum(
            g_pao_n.astype(np.int64) * g_pno_n.astype(np.int64))
        g_T2_off = np.zeros(n_pairs + 1, dtype=np.int64)
        g_T2_off[1:] = np.cumsum(
            g_pno_n.astype(np.int64) * g_pno_n.astype(np.int64))
        g_pp_flat = np.empty(int(g_pp_off[-1]), dtype=np.int64)
        g_X_flat  = np.empty(int(g_X_off[-1]), dtype=np.float64)
        g_T2_flat = np.zeros(int(g_T2_off[-1]), dtype=np.float64)
        for p, pk in enumerate(all_pks):
            pd = pno_spaces[pk]
            if g_pao_n[p] > 0:
                g_pp_flat[g_pp_off[p]:g_pp_off[p + 1]] = np.asarray(
                    pd['pair_paos'], dtype=np.int64)
            if g_pao_n[p] > 0 and g_pno_n[p] > 0:
                g_X_flat[g_X_off[p]:g_X_off[p + 1]] = np.ascontiguousarray(
                    pd['X_pno']).ravel()
            if pk in t2_for_T and g_pno_n[p] > 0:
                g_T2_flat[g_T2_off[p]:g_T2_off[p + 1]] = np.ascontiguousarray(
                    t2_for_T[pk]).ravel()
        gcache = (_gak, pair_to_idx, n_pairs,
                  g_pao_n, g_pno_n,
                  g_pp_off, g_pp_flat, g_X_off, g_X_flat,
                  g_T2_off, g_T2_flat)
        _run_triples_omp._pair_arena = gcache
    (_, pair_to_idx, n_pairs,
     g_pao_n, g_pno_n,
     g_pp_off, g_pp_flat, g_X_off, g_X_flat,
     g_T2_off, g_T2_flat) = gcache

    # Pre-compute per-LMO domain-partner set:
    #   partners[i] = {m : (min(m,i), max(m,i)) in _domain_set}
    # This turns the per-triple `for m in range(nocc_lmo)` scan into
    # a 3-way set intersection of bounded size (each partner set is the
    # F-coupling neighbourhood of one LMO).
    _partners = [set() for _ in range(nocc_lmo)]
    for (a, b) in _domain_set:
        _partners[a].add(b)
        _partners[b].add(a)

    # Build per-triple arena.
    ijk_list = np.asarray(valid_triples, dtype=np.int64).reshape(n_triples, 3)
    tp_lists = []      # triple_paos for each triple
    td_lists = []      # triple_domain
    pair_idx_3 = np.full((n_triples, 3), -1, dtype=np.int32)
    same_lmo_3 = np.zeros((n_triples, 3), dtype=np.int8)
    t2b_idx = np.full((n_triples, 9), -1, dtype=np.int32)
    t2b_tflag = np.zeros((n_triples, 9), dtype=np.int8)
    t1d_idx = np.full((n_triples, 3), -1, dtype=np.int32)
    eps_ijk = np.zeros((n_triples, 3), dtype=np.float64)
    occ_denom_arr = np.ones(n_triples, dtype=np.int32)
    upk_lists = []
    w3_idx_lists = []
    w3_tflag_lists = []
    m_dom_arr = np.zeros(n_triples, dtype=np.int32)
    valid_mask = np.ones(n_triples, dtype=bool)

    for t, (i, j, k) in enumerate(valid_triples):
        ij = (min(i, j), max(i, j))
        ik = (min(i, k), max(i, k))
        jk = (min(j, k), max(j, k))
        if (ij not in pair_to_idx or jk not in pair_to_idx or ik not in pair_to_idx):
            valid_mask[t] = False
            tp_lists.append(np.empty(0, dtype=np.int64))
            td_lists.append(np.empty(0, dtype=np.int64))
            upk_lists.append(np.empty(0, dtype=np.int32))
            w3_idx_lists.append(np.empty(0, dtype=np.int32))
            w3_tflag_lists.append(np.empty(0, dtype=np.int8))
            continue
        # triple_paos (sorted union)
        if pao_domains_triple is not None:
            pao_set = set()
            for lmo in (i, j, k):
                pao_set.update(int(x) for x in
                               np.asarray(pao_domains_triple[lmo]).tolist())
        else:
            pao_set = set()
            for key in (ij, jk, ik):
                pp = pno_spaces[key].get('pair_paos')
                if pp is not None:
                    pao_set.update(int(x) for x in np.asarray(pp).tolist())
        if not pao_set:
            valid_mask[t] = False
            tp_lists.append(np.empty(0, dtype=np.int64))
            td_lists.append(np.empty(0, dtype=np.int64))
            upk_lists.append(np.empty(0, dtype=np.int32))
            w3_idx_lists.append(np.empty(0, dtype=np.int32))
            w3_tflag_lists.append(np.empty(0, dtype=np.int8))
            continue
        triple_paos = np.array(sorted(pao_set), dtype=np.int64)
        tp_lists.append(triple_paos)

        # triple_domain = partners[i] ∩ partners[j] ∩ partners[k]
        triple_domain = sorted(
            _partners[i] & _partners[j] & _partners[k])
        td_lists.append(np.asarray(triple_domain, dtype=np.int64))
        n_dom = len(triple_domain)
        m_dom_arr[t] = n_dom

        # 3-pair indices
        pair_idx_3[t, 0] = pair_to_idx[ij]
        pair_idx_3[t, 1] = pair_to_idx[jk]
        pair_idx_3[t, 2] = pair_to_idx[ik]
        same_lmo_3[t, 0] = 1 if i == j else 0
        same_lmo_3[t, 1] = 1 if j == k else 0
        same_lmo_3[t, 2] = 1 if i == k else 0

        # u_pks list
        triple_lmo = (i, j, k)
        seen = set()
        upks = []
        for r in triple_lmo:
            for m in triple_domain:
                pk = (min(r, m), max(r, m))
                if pk not in seen and pk in pair_to_idx:
                    pd = pno_spaces[pk]
                    if (pd.get('X_pno') is not None
                            and pd['X_pno'].shape[1] > 0
                            and pk in t2_for_T):
                        seen.add(pk); upks.append(pk)
            pk_d = (r, r)
            if pk_d not in seen and pk_d in pair_to_idx:
                pd = pno_spaces[pk_d]
                if (pd.get('X_pno') is not None
                        and pd['X_pno'].shape[1] > 0
                        and pk_d in t2_for_T):
                    seen.add(pk_d); upks.append(pk_d)
        for r in triple_lmo:
            for r2 in triple_lmo:
                pk = (min(r, r2), max(r, r2))
                if pk not in seen and pk in pair_to_idx:
                    pd = pno_spaces[pk]
                    if (pd.get('X_pno') is not None
                            and pd['X_pno'].shape[1] > 0
                            and pk in t2_for_T):
                        seen.add(pk); upks.append(pk)
        upks_local = {pk: idx for idx, pk in enumerate(upks)}
        upk_idx_arr = np.array([pair_to_idx[pk] for pk in upks], dtype=np.int32)
        upk_lists.append(upk_idx_arr)

        # t2_block_idx (uses LOCAL u_pks indices, not global pair_idx)
        for p in range(3):
            for q in range(3):
                lp, lq = triple_lmo[p], triple_lmo[q]
                pk = (min(lp, lq), max(lp, lq))
                if pk in upks_local:
                    t2b_idx[t, p * 3 + q] = upks_local[pk]
                    t2b_tflag[t, p * 3 + q] = 1 if lp > lq else 0

        # w3 indices
        w3_idx_t = np.full(3 * n_dom, -1, dtype=np.int32)
        w3_tflag_t = np.zeros(3 * n_dom, dtype=np.int8)
        for r in range(3):
            lr = triple_lmo[r]
            for l_local in range(n_dom):
                ll = triple_domain[l_local]
                pk = (min(lr, ll), max(lr, ll))
                if pk in upks_local:
                    w3_idx_t[r * n_dom + l_local] = upks_local[pk]
                    w3_tflag_t[r * n_dom + l_local] = 1 if ll > lr else 0
        w3_idx_lists.append(w3_idx_t)
        w3_tflag_lists.append(w3_tflag_t)

        # t1 diag idx
        for r in range(3):
            pk = (triple_lmo[r], triple_lmo[r])
            if pk in upks_local:
                t1d_idx[t, r] = upks_local[pk]

        # eps_ijk + occ_denom
        eps_ijk[t, 0] = float(F_lmo[i, i])
        eps_ijk[t, 1] = float(F_lmo[j, j])
        eps_ijk[t, 2] = float(F_lmo[k, k])
        dij = int(i == j); djk = int(j == k); dik = int(i == k)
        occ_denom_arr[t] = 1 + dij + djk + dik + 2 * dij * djk * dik

    # Flatten variable-length per-triple data
    tp_off = np.zeros(n_triples + 1, dtype=np.int64)
    tp_off[1:] = np.cumsum([a.size for a in tp_lists])
    tp_flat = np.concatenate(tp_lists) if tp_lists else np.empty(0, dtype=np.int64)
    td_off = np.zeros(n_triples + 1, dtype=np.int64)
    td_off[1:] = np.cumsum([a.size for a in td_lists])
    td_flat = np.concatenate(td_lists) if td_lists else np.empty(0, dtype=np.int64)
    upk_off = np.zeros(n_triples + 1, dtype=np.int64)
    upk_off[1:] = np.cumsum([a.size for a in upk_lists])
    upk_idx_flat = np.concatenate(upk_lists) if upk_lists else np.empty(0, dtype=np.int32)
    w3_off = np.zeros(n_triples + 1, dtype=np.int64)
    w3_off[1:] = np.cumsum([a.size for a in w3_idx_lists])
    w3_idx_flat = np.concatenate(w3_idx_lists) if w3_idx_lists else np.empty(0, dtype=np.int32)
    w3_tflag_flat = np.concatenate(w3_tflag_lists) if w3_tflag_lists else np.empty(0, dtype=np.int8)

    # IMPORTANT: t2b_idx and w3_idx use LOCAL u_pks indices.  But the C
    # function expects them to index the per-triple u_pks list directly.
    # The C function looks up u_pao_n_t_i[u_idx], u_pp_off_t[u_idx], etc.
    # So local indices are correct (they're 0..n_u_pks_t-1).

    # T1 flat (per-LMO)
    has_t1 = 1 if t1_pno is not None else 0
    if has_t1:
        cached_t1 = getattr(_run_triples_omp, '_t1_cache', None)
        if cached_t1 is None or cached_t1[0] is not t1_pno:
            t1_sizes = np.zeros(nocc_lmo, dtype=np.int64)
            for lmo, vec in t1_pno.items():
                if vec is not None:
                    t1_sizes[lmo] = vec.size
            t1_off = np.empty(nocc_lmo + 1, dtype=np.int64); t1_off[0] = 0
            t1_off[1:] = np.cumsum(t1_sizes)
            t1_flat = np.zeros(int(t1_off[-1]), dtype=np.float64)
            for lmo, vec in t1_pno.items():
                if vec is not None and vec.size:
                    t1_flat[t1_off[lmo]:t1_off[lmo + 1]] = vec
            cached_t1 = (t1_pno, t1_off, t1_flat)
            _run_triples_omp._t1_cache = cached_t1
        _, t1_off, t1_flat = cached_t1
    else:
        t1_off = np.zeros(nocc_lmo + 1, dtype=np.int64)
        t1_flat = np.zeros(1, dtype=np.float64)

    # Cached globals
    _ck = (id(j2c_full), id(F_pao_full), id(S_pao_full),
           id(lmo_aux_mask), id(sparse_df), id(screening))
    cached = getattr(_run_triples_omp, '_globals_cache', None)
    if cached is None or cached[0] != _ck:
        cached = (_ck,
                  np.ascontiguousarray(F_pao_full),
                  np.ascontiguousarray(S_pao_full),
                  np.ascontiguousarray(j2c_full),
                  np.ascontiguousarray(sparse_df['aux_atom_ids'], dtype=np.int64),
                  np.ascontiguousarray(sparse_df['aux_pos_in_atom'], dtype=np.int64),
                  np.ascontiguousarray(
                      screening['riatom_to_lmos_ext_dense'], dtype=np.int64),
                  np.ascontiguousarray(
                      screening['riatom_to_paos_ext_dense'], dtype=np.int64),
                  np.ascontiguousarray(lmo_aux_mask.astype(np.int64)))
        _run_triples_omp._globals_cache = cached
    (_, F_pao_c, S_pao_c, j2c_c, aux_atom_ids, aux_pos_in_atom,
     lmo_dense_c, pao_dense_c, lmo_aux_mask_c) = cached

    # ctypes
    import ctypes as _ct
    from pyscf import lib as _pl
    _libcc = getattr(_run_triples_omp, '_libcc', None)
    if _libcc is None:
        _libcc = _pl.load_library('libcc')
        _libcc.DLPNOcompute_E_T0_omp.restype = _ct.c_double
        _libcc.DLPNOcompute_E_T0_omp.argtypes = (
            [_ct.c_int]                 # n_triples
            + [_ct.c_void_p] * 5        # ijk_list, tp_off, tp_flat, td_off, td_flat
            + [_ct.c_void_p] * 6        # pair_idx_3, same_lmo_3, t2b_idx, t2b_tflag, upk_off, upk_idx_flat
            + [_ct.c_void_p] * 3        # w3_off, w3_idx_flat, w3_tflag_flat
            + [_ct.c_void_p] * 4        # m_dom_arr, t1d_idx, eps_ijk, occ_denom_arr
            + [_ct.c_int]               # n_pairs_total
            + [_ct.c_void_p] * 8        # global pair arena: g_pao_n, g_pno_n, g_pp_off, g_pp_flat, g_X_off, g_X_flat, g_T2_off, g_T2_flat
            + [_ct.c_int]               # has_t1
            + [_ct.c_void_p] * 2        # t1_off, t1_flat
            + [_ct.c_int] * 3           # nocc_lmo, n_pao_total, naux_total
            + [_ct.c_void_p] * 14       # F_pao..qab_atom_flat
            + [_ct.c_void_p] * 3        # lmo_dense, pao_dense, lmo_aux_mask
            + [_ct.c_double, _ct.c_double]  # T_CutTNO, S_cut_domain
            + [_ct.c_void_p]            # et_per_triple
        )
        _run_triples_omp._libcc = _libcc

    et_per_triple = np.zeros(n_triples, dtype=np.float64)
    _libcc.DLPNOcompute_E_T0_omp(
        int(n_triples),
        ijk_list.ctypes.data_as(_ct.c_void_p),
        tp_off.ctypes.data_as(_ct.c_void_p),
        tp_flat.ctypes.data_as(_ct.c_void_p),
        td_off.ctypes.data_as(_ct.c_void_p),
        td_flat.ctypes.data_as(_ct.c_void_p),
        pair_idx_3.ctypes.data_as(_ct.c_void_p),
        same_lmo_3.ctypes.data_as(_ct.c_void_p),
        t2b_idx.ctypes.data_as(_ct.c_void_p),
        t2b_tflag.ctypes.data_as(_ct.c_void_p),
        upk_off.ctypes.data_as(_ct.c_void_p),
        upk_idx_flat.ctypes.data_as(_ct.c_void_p),
        w3_off.ctypes.data_as(_ct.c_void_p),
        w3_idx_flat.ctypes.data_as(_ct.c_void_p),
        w3_tflag_flat.ctypes.data_as(_ct.c_void_p),
        m_dom_arr.ctypes.data_as(_ct.c_void_p),
        t1d_idx.ctypes.data_as(_ct.c_void_p),
        eps_ijk.ctypes.data_as(_ct.c_void_p),
        occ_denom_arr.ctypes.data_as(_ct.c_void_p),
        int(n_pairs),
        g_pao_n.ctypes.data_as(_ct.c_void_p),
        g_pno_n.ctypes.data_as(_ct.c_void_p),
        g_pp_off.ctypes.data_as(_ct.c_void_p),
        g_pp_flat.ctypes.data_as(_ct.c_void_p),
        g_X_off.ctypes.data_as(_ct.c_void_p),
        g_X_flat.ctypes.data_as(_ct.c_void_p),
        g_T2_off.ctypes.data_as(_ct.c_void_p),
        g_T2_flat.ctypes.data_as(_ct.c_void_p),
        int(has_t1),
        t1_off.ctypes.data_as(_ct.c_void_p),
        t1_flat.ctypes.data_as(_ct.c_void_p),
        int(nocc_lmo), int(n_pao_total), int(naux_total),
        F_pao_c.ctypes.data_as(_ct.c_void_p),
        S_pao_c.ctypes.data_as(_ct.c_void_p),
        j2c_c.ctypes.data_as(_ct.c_void_p),
        aux_atom_ids.ctypes.data_as(_ct.c_void_p),
        aux_pos_in_atom.ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_off'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qia_atom_off'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qab_atom_off'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_n_aux'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_n_lmo'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qab_atom_n_pao'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qij_atom_flat'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qia_atom_flat'].ctypes.data_as(_ct.c_void_p),
        sparse_df['qab_atom_flat'].ctypes.data_as(_ct.c_void_p),
        lmo_dense_c.ctypes.data_as(_ct.c_void_p),
        pao_dense_c.ctypes.data_as(_ct.c_void_p),
        lmo_aux_mask_c.ctypes.data_as(_ct.c_void_p),
        float(T_CutTNO), float(1e-8),
        et_per_triple.ctypes.data_as(_ct.c_void_p),
    )
    return et_per_triple


def _process_one_triple(i, j, k,
                        pno_spaces, t2_for_T,
                        Lpq_full, C_lmo, fock_ao, F_lmo, s1e,
                        t1_pno=None, T_CutTNO=1e-9,
                        nonneg_set=None,
                        sparse_df=None, screening=None,
                        j2c_full=None, lmo_aux_mask=None, C_pao=None,
                        S_pao_full=None, F_pao_full=None,
                        pao_domains_triple=None):
    """Compute (T) energy contribution for one triple (i,j,k).

    Phase 3c-2/3: entire per-triple body in one C call (DLPNOcompute_one_triple_E_T0).
    Returns et_ijk (float), or 0.0 if the triple should be skipped.
    """
    ij = (min(i, j), max(i, j))
    ik = (min(i, k), max(i, k))
    jk = (min(j, k), max(j, k))

    if (ij not in pno_spaces or ik not in pno_spaces or jk not in pno_spaces):
        return 0.0
    if (ij not in t2_for_T or ik not in t2_for_T or jk not in t2_for_T):
        return 0.0

    return _orch(
        i, j, k, pno_spaces, t2_for_T,
        C_pao, S_pao_full, F_pao_full, F_lmo,
        sparse_df, screening, j2c_full, lmo_aux_mask,
        pao_domains_triple, nonneg_set, T_CutTNO,
        t1_pno)


def _process_degenerate_pair(i, k,
                             pno_spaces, t2_for_T,
                             Lpq_full, C_lmo, fock_ao, F_lmo, s1e,
                             t1_pno=None, T_CutTNO=1e-9,
                             nonneg_set=None,
                             sparse_df=None, screening=None,
                             j2c_full=None, lmo_aux_mask=None,
                             C_pao=None,
                             S_pao_full=None, F_pao_full=None,
                             pao_domains_triple=None):
    """Compute (T) energy from degenerate occupied triples {i,i,k} and {i,k,k}.

    With the Eq 53 formula, _process_one_triple handles degenerate triples
    correctly via the occ_indices degeneracy factor. We just call it for both
    (i,i,k) and (i,k,k).
    """
    kwargs = dict(pno_spaces=pno_spaces, t2_for_T=t2_for_T,
                  Lpq_full=Lpq_full, C_lmo=C_lmo, fock_ao=fock_ao,
                  F_lmo=F_lmo, s1e=s1e, t1_pno=t1_pno, T_CutTNO=T_CutTNO,
                  nonneg_set=nonneg_set,
                  sparse_df=sparse_df, screening=screening,
                  j2c_full=j2c_full, lmo_aux_mask=lmo_aux_mask,
                  C_pao=C_pao,
                  S_pao_full=S_pao_full, F_pao_full=F_pao_full,
                  pao_domains_triple=pao_domains_triple)
    et_iik = _process_one_triple(i, i, k, **kwargs)
    et_ikk = _process_one_triple(i, k, k, **kwargs)
    return et_iik + et_ikk


def run_lccsd_t_ext(mf, C_lmo, pno_spaces, strong_pairs,
                    t2_pno_all, occ_cas_idx,
                    t1_pno=None,
                    C_cas_vir=None,
                    vir_cas_idx=None,
                    cas_proj_thresh=0.5,
                    T_CutTNO=1e-9,
                    T_CutTriplesWeak=1e-7,
                    ncores=1,
                    negligible_pairs=None,
                    weak_pairs=None,
                    C_pao=None,
                    doi_iu=None,
                    verbose=None,
                    _pool=None):
    """Compute the (T) energy correction for DLPNO-TCCSD(T).

    Loops over all distinct triples (i<j<k).  Per Lang et al. 2020
    Sec. II.C, double-counting is prevented by zeroing all active T2
    amplitudes before computing (T), NOT by skipping CAS triples.

    The disconnected V intermediate (T1 × vvoo + T2 × Fock) is included
    following the canonical formula from ccsd_t_slow.py:
        z = r3(w + 0.5*v) / D;  E = sum w * z

    Args:
        mf: RHF object (for Fock matrix and with_df).
        C_lmo (np.ndarray): (nao, nocc_lmo) LMO coefficients.
        pno_spaces (dict): Output of pno.make_pnos.
        strong_pairs (list): Strong pairs from pair classification.
        t2_pno_all (dict): Converged T2 amplitudes per pair from run_lccsd.
        occ_cas_idx (np.ndarray): CAS occupied indices.
        t1_pno (dict): i → (n_pno_ii,) local T1 amplitudes in diagonal PNO
            basis (Jiang et al. JCP 2024).  If None, the V intermediate is omitted.
        C_cas_vir (np.ndarray): (nao, n_cas_vir) CAS virtual MO coefficients.
            Required for TCC; if None, T2 zeroing is skipped (warning issued).
        vir_cas_idx (array-like): CAS virtual indices (optional, for logging).
        cas_proj_thresh (float): Zero PNO if CAS-virtual overlap > threshold.
        T_CutTNO (float): TNO eigenvalue threshold (Lang et al. default 1e-9).
        verbose: Verbosity.

    Returns:
        e_t (float): (T) energy correction.
    """
    log = logger.new_logger(mf, verbose)
    import time as _time
    import os as _os_tp
    _tp_dbg = bool(int(_os_tp.environ.get('DLPNO_T_PROFILE', '0')))
    _tp_t = [_time.perf_counter()]
    def _tp(label):
        if _tp_dbg:
            now = _time.perf_counter()
            print(f'  [T-PROFILE] {label}: {now - _tp_t[0]:.2f}s', flush=True)
            _tp_t[0] = now

    if not hasattr(mf, 'with_df') or mf.with_df is None:
        import warnings
        warnings.warn(
            'mf.with_df is None — (T) correction requires density fitting. '
            'Returning e_t = 0.',
            UserWarning, stacklevel=2)
        return 0.0

    occ_cas_set = set(occ_cas_idx.tolist())
    nocc_lmo = C_lmo.shape[1]

    # AO overlap matrix (needed for S-metric PNO overlaps and CAS zeroing)
    s1e = mf.get_ovlp()
    _tp("setup (s1e)")

    # Zero CAS amplitudes to prevent double-counting static correlation
    # (Lang et al. Section II.A: set all active amplitudes to zero for (T))
    if C_cas_vir is not None and C_cas_vir.shape[1] > 0:
        log.info('TCC (T): zeroing T2 with CAS character (n_cas_vir=%d, thresh=%.2f)',
                 C_cas_vir.shape[1], cas_proj_thresh)
        t2_for_T = _zero_cas_t2_amplitudes(
            t2_pno_all, pno_spaces, occ_cas_idx, C_cas_vir, s1e, cas_proj_thresh)
    else:
        if C_cas_vir is None and len(occ_cas_set) > 0:
            log.warn('TCC (T): C_cas_vir not provided — T2 amplitudes not zeroed!')
        t2_for_T = t2_pno_all
    _tp("CAS T2 zeroing")

    # Fock diagonal in LMO basis for occupied orbital energies.
    # Pick up the cached F_AO the driver stored before cderi was
    # released — avoids a J/K rebuild here.
    fock_ao = getattr(mf, '_dlpno_fock_ao', None)
    if fock_ao is None:
        fock_ao = mf.get_fock()
    F_lmo = reduce(np.dot, (C_lmo.T, fock_ao, C_lmo))

    occ_list = list(range(nocc_lmo))
    _tp("Fock + LMO setup")

    # Enumerate distinct triples (i<j<k) following Psi4's triples_sparsity
    # (dlpno/triples.cc::triples_sparsity prescreening block).
    #   (1) k iterated over pair_lmo_idx[ij] (= lmopair_to_lmos_[ij] in Psi4)
    #       — the pair's interacting-LMO domain, NOT all of nocc.
    #   (2) max-2-weak-pair constraint: at least one of (ij), (ik), (jk)
    #       must be strong, else the triple is dropped.
    _negl_set = set((min(p), max(p)) for p in (negligible_pairs or []))
    _weak_set = set((min(p), max(p)) for p in (weak_pairs or []))
    # Max weak pairs allowed in a kept triple: 2 (Psi4 rule). ORCA computes
    # only triples with <= 1 weak pair; DLPNO_T_WEAKMAX=1 mimics that.
    _weak_max = int(os.environ.get('DLPNO_T_WEAKMAX', '2'))
    _tT_set = set(k for k in t2_for_T.keys() if k not in _negl_set)

    # Build pair_lmo_idx locally — m is in pair (i,j)'s domain iff both
    # (i,m) and (j,m) are non-negligible.  (Same construction as in
    # lccsd.py _run_dlpno_lccsd.)
    pair_lmo_idx = {}
    for key in _tT_set:
        i, j = key
        dom = [m for m in range(nocc_lmo)
               if (min(i, m), max(i, m)) in _tT_set
               and (min(j, m), max(j, m)) in _tT_set]
        pair_lmo_idx[key] = dom
    _tp("pair_lmo_idx build")

    valid_triples = []
    _n_before_weak_screen = 0
    for ij in _tT_set:
        i, j = ij
        if i >= j:
            continue        # strict i < j; degenerate (i,i,k) handled below
        for k in pair_lmo_idx.get(ij, []):
            if k <= j:                         # strict i < j < k
                continue
            ik = (i, k)
            jk = (j, k)
            if ik not in _tT_set or jk not in _tT_set:
                continue
            _n_before_weak_screen += 1
            weak_count = ((ij in _weak_set) + (ik in _weak_set) +
                          (jk in _weak_set))
            if weak_count > _weak_max:
                continue
            valid_triples.append((i, j, k))
    _n_all = nocc_lmo * (nocc_lmo - 1) * (nocc_lmo - 2) // 6
    print(f'  (T) triples after Psi4-style screening: '
          f'{len(valid_triples)} / {_n_all} '
          f'({100.0 * len(valid_triples) / max(_n_all, 1):.1f}%) '
          f'[pre-weak-count: {_n_before_weak_screen}]', flush=True)

    _tp("valid_triples enumerate")

    # The default Phase 3c-2 (T) path goes through _orch_full, which
    # builds its own per-aux sparse stack via build_sparse_df_arrays
    # (libcint shell-by-shell, never reads cderi).  Lpq_full is only
    # consumed by the opt-in DLPNO_TRIPLE_OMP=1 OMP-over-triples driver
    # and by the optional run_lccsd_t1_iterations path; build it lazily
    # if either is requested. At water-64 this single allocation was
    # ~51 GB (5376 aux × 1.18M nao_pair × 8 bytes) for a tensor that's
    # never touched in the default path.
    _t0 = _time.perf_counter()
    if os.environ.get('DLPNO_TRIPLE_OMP', '0') == '1':
        Lpq_full = _preload_df_integrals(mf.with_df)
        log.info('(T) preloaded DF integrals: shape=%s, %.1f MB, %.2f s',
                 Lpq_full.shape, Lpq_full.nbytes / 1e6,
                 _time.perf_counter() - _t0)
    else:
        Lpq_full = None  # _orch_full doesn't read it
    _tp("preload Lpq_full")

    # --- Build sparse-DF infrastructure for triple-local aux path ---
    # Mirrors Psi4 DLPNOCCSD_T::compute_lccsd_t0: per-triple integrals
    # are constructed by slicing pre-computed sparse arrays qij[Q],
    # qia[Q], qab[Q] by the triple's aux_ijk/LMO_ijk/PAO_ijk domains,
    # then applying a local J^{-1/2}.  _process_one_triple will use
    # this path when all of {sparse_df, screening, j2c_full,
    # lmo_aux_mask, C_pao} are available.
    from pyscf.cc.dlpno_tccsd.local_df import (
        build_screening_maps as _build_screen,
        build_sparse_df_arrays as _build_sparse,
    )
    # Psi4/Jiang defaults for the TRIPLES stage (distinct from CCSD T_CUT_MKN=1e-3):
    #   T_CUT_MKN_TRIPLES = 1e-2   (read_options.cc line 2573)
    #   T_CUT_DO_TRIPLES  = 1e-2   (read_options.cc line 2575; applied to PAO domains)
    # The triples-stage thresholds are 10× LOOSER than CCSD to keep per-triple
    # aux/PAO domains small while not hurting (T) accuracy.
    # A/B experiment knobs (default = production behavior):
    #   DLPNO_T_TCUTTNO          TNO occupation cutoff (default 1e-9)
    #   DLPNO_T_TCUTTRIPLESWEAK  SC-MP2 prescreen drop threshold; 0 disables
    #                            the prescreen (all triples at tight TNO)
    T_CutTNO = float(os.environ.get('DLPNO_T_TCUTTNO', T_CutTNO))
    T_CutTriplesWeak = float(
        os.environ.get('DLPNO_T_TCUTTRIPLESWEAK', T_CutTriplesWeak))

    # Psi4 (T) thresholds — separate defaults for tight pass and (T0) prescreen.
    # read_options.cc L2565-2575. The PRE thresholds are deliberately looser so
    # the prescreen pass runs on a SMALLER sparse-DF infrastructure (cheap (T0)),
    # while the tight pass on surviving triples uses the full-accuracy thresholds.
    _T_CUT_MKN_TRIPLES     = 1e-2
    _T_CUT_DO_TRIPLES      = 1e-2
    # Jiang JCP 2024 Table II: the (T0) prescreen loosens ONLY the TNO tolerance
    # (T_CUT_TNO_PRE); the aux/PAO domains use the same T_CUT_MKN_TRIPLES /
    # T_CUT_DO_TRIPLES = 1e-2 as the tight pass.  (Previously these were loosened
    # to 1e-1 / 2e-2 as an in-house speed optimization, which altered the
    # screened-triplet contribution vs Psi4.)
    _T_CUT_MKN_TRIPLES_PRE = 1e-2    # match Table II (was 1e-1)
    _T_CUT_DO_TRIPLES_PRE  = 1e-2    # match Table II (was 2e-2)
    _T_CUT_CLMO = 1e-3
    _auxmol = mf.with_df.auxmol if hasattr(mf.with_df, 'auxmol') else None
    _j2c_full = None
    if _auxmol is not None and C_pao is not None:
        _natm = mf.mol.natm
        _ao_labels = mf.mol.ao_labels(fmt=False)
        _atom_ids = np.array([lbl[0] for lbl in _ao_labels])
        _aux_atom_ids = np.array(
            [lbl[0] for lbl in _auxmol.ao_labels(fmt=False)])
        _atom_to_ao = [np.where(_atom_ids == a)[0] for a in range(_natm)]

        def _build_triples_infrastructure(T_CUT_MKN, T_CUT_DO, label,
                                          tight_sparse=None, tight_screen=None):
            """Build lmo_aux_mask, pao_domains, screening, and sparse_df stacks
            at the given thresholds. Mirrors Psi4 triples_sparsity(prescreening).

            If ``tight_sparse`` and ``tight_screen`` are passed in, the
            sparse-DF arrays (qij/qia/qab) are derived by slicing the tight
            build instead of recomputing the int3c2e integrals. PRESCREEN
            uses tighter Mulliken / looser DOI thresholds → its per-atom
            (lmos_ext, paos_ext) are strict subsets of TIGHT's, so the
            integrals are a no-op subset extract (~tenths of a second) vs
            the full rebuild (3-10s scaling N^2.6).
            """
            _bi_t = [_time.perf_counter()]
            def _bi(step):
                if _tp_dbg:
                    now = _time.perf_counter()
                    print(f'    [T-BI {label}] {step}: {now - _bi_t[0]:.2f}s',
                          flush=True)
                    _bi_t[0] = now

            # --- lmo_aux_mask: per-LMO Mulliken-weighted aux-atom mask ---
            # Vectorised with row/col sum + bincount (no per-atom fancy
            # indexing — that pattern blew up to ~50s at water-34, scaling
            # as N_atom × nao² of memory traffic per LMO).  Serial loop;
            # pool dispatch over this small element-wise kernel was a
            # 30× regression at large N due to GIL + memory pressure.
            def _t_mkn_per_lmo(ii):
                c_i = C_lmo[:, ii]
                P_i = s1e * c_i[:, None] * c_i[None, :]
                p_diag = np.diag(P_i)
                sum_diag = p_diag[:, None] + p_diag[None, :]
                with np.errstate(divide='ignore', invalid='ignore'):
                    w_u = np.where(sum_diag > 1e-15,
                                   p_diag[:, None] / sum_diag, 0.0)
                    w_v = np.where(sum_diag > 1e-15,
                                   p_diag[None, :] / sum_diag, 0.0)
                row_sum_u = (P_i * w_u).sum(axis=1)
                col_sum_v = (P_i * w_v).sum(axis=0)
                mkn_pop = (np.bincount(_atom_ids, weights=row_sum_u,
                                       minlength=_natm)
                         + np.bincount(_atom_ids, weights=col_sum_v,
                                       minlength=_natm))
                return np.isin(
                    _aux_atom_ids,
                    np.where(np.abs(mkn_pop) > T_CUT_MKN)[0])
            mask_rows = [_t_mkn_per_lmo(ii) for ii in range(nocc_lmo)]
            lmo_aux_mask = np.array(mask_rows)
            _bi("lmo_aux_mask (Mulliken)")

            # --- pao_domains: DOI > T_CUT_DO, atom-complete (triples.cc:316-329) ---
            pao_domains = []
            if doi_iu is not None:
                for ii in range(nocc_lmo):
                    doi = doi_iu[ii]
                    pao_inds = np.where(doi > T_CUT_DO)[0]
                    if pao_inds.size == 0:
                        pao_inds = np.array([int(np.argmax(doi))])
                    atoms_in = np.unique(_atom_ids[pao_inds])
                    domain_i = np.concatenate([_atom_to_ao[a] for a in atoms_in])
                    pao_domains.append(np.sort(domain_i))
            else:
                for ii in range(nocc_lmo):
                    key_ii = (ii, ii)
                    if (key_ii in pno_spaces
                            and pno_spaces[key_ii].get('pair_paos') is not None):
                        pao_domains.append(
                            np.asarray(pno_spaces[key_ii]['pair_paos']))
                    else:
                        pao_domains.append(np.zeros(0, dtype=int))

            _dom_sizes = [len(d) for d in pao_domains]
            log.info('(T) [%s] T_CUT_MKN=%.1e T_CUT_DO=%.1e: '
                     'avg PAOs/LMO=%.1f (min=%d, max=%d)', label,
                     T_CUT_MKN, T_CUT_DO,
                     float(np.mean(_dom_sizes)),
                     int(min(_dom_sizes)), int(max(_dom_sizes)))
            _bi("pao_domains")

            # --- screening + sparse_df + per-atom stacks ---
            strong_pair_keys = list(t2_for_T.keys())
            screening = _build_screen(
                mf.mol, _auxmol, C_lmo, pao_domains, s1e, strong_pair_keys,
                T_CUT_MKN=T_CUT_MKN, T_CUT_CLMO=_T_CUT_CLMO, C_pao=C_pao,
                _pool=_pool)
            _bi("_build_screen")
            # PRESCREEN: derive sparse-DF by slicing the TIGHT build instead
            # of recomputing integrals. PRESCREEN per-atom domains are
            # subsets of TIGHT's (looser MKN/DOI), so this is exact, not an
            # approximation. Saves the full int3c2e rebuild (3-10s wall at
            # water-22..34, scaling N^2.6).
            if tight_sparse is not None and tight_screen is not None:
                from pyscf.cc.dlpno_tccsd.local_df import derive_subset_sparse_df
                sparse_df = derive_subset_sparse_df(
                    tight_sparse, tight_screen, screening)
            else:
                # Pass the shared thread pool so the per-aux-shell loop runs
                # parallel. Without this, _build_sparse iterates serially over
                # all aux shells in Python — dominates (T) wall at large N
                # (61s of 78s on water-22).
                sparse_df = _build_sparse(
                    mf.mol, _auxmol, C_lmo, C_pao, screening, _pool=_pool)
            _bi("_build_sparse")
            aux_atom_ids_arr = screening['aux_atom_ids']
            naux_total = len(sparse_df['qij'])
            aux_at_atom_list = [np.where(aux_atom_ids_arr == A)[0]
                                for A in range(_natm)]
            qij_stack = [None] * _natm
            qia_stack = [None] * _natm
            qab_stack = [None] * _natm
            ri_lmos_ext = screening['riatom_to_lmos_ext']
            ri_paos_ext = screening['riatom_to_paos_ext']
            # Per-atom np.stack is independent per atom; the stacks become
            # ~3 s combined on water-42 if serial (~1 s on water-22 ×4).
            # Pool-parallel by atom; writes target distinct list slots.
            _qij_ref = sparse_df['qij']
            _qia_ref = sparse_df['qia']
            _qab_ref = sparse_df['qab']
            def _stack_one_atom(A):
                Qs_A = aux_at_atom_list[A]
                if len(Qs_A) == 0:
                    return None, None, None
                nl_A = len(ri_lmos_ext[A])
                np_A = len(ri_paos_ext[A])
                qij_A = (np.stack([_qij_ref[Q] for Q in Qs_A])
                         if nl_A > 0 else None)
                qia_A = (np.stack([_qia_ref[Q] for Q in Qs_A])
                         if (nl_A > 0 and np_A > 0) else None)
                qab_A = (np.stack([_qab_ref[Q] for Q in Qs_A])
                         if np_A > 0 else None)
                return qij_A, qia_A, qab_A
            if _pool is not None and _natm > 1:
                _stack_results = list(_pool.map(_stack_one_atom, range(_natm)))
            else:
                _stack_results = [_stack_one_atom(A) for A in range(_natm)]
            for A, (q_ij, q_ia, q_ab) in enumerate(_stack_results):
                qij_stack[A] = q_ij
                qia_stack[A] = q_ia
                qab_stack[A] = q_ab
            aux_pos_in_atom = -np.ones(naux_total, dtype=np.int64)
            for A in range(_natm):
                for pos, Q in enumerate(aux_at_atom_list[A]):
                    aux_pos_in_atom[Q] = pos
            sparse_df['qij_atom'] = qij_stack
            sparse_df['qia_atom'] = qia_stack
            sparse_df['qab_atom'] = qab_stack
            sparse_df['aux_at_atom'] = aux_at_atom_list
            sparse_df['aux_pos_in_atom'] = aux_pos_in_atom
            sparse_df['aux_atom_ids'] = aux_atom_ids_arr
            _bi("per-atom np.stack qij/qia/qab")

            # Flat per-atom stacks for the C kernel.
            # qij_atom[A]: (nQ_A, nl_A, nl_A); qia: (nQ_A, nl_A, np_A);
            # qab: (nQ_A, np_A, np_A). We concat each per-atom block end-to-end
            # and store offsets + dim hints so the C kernel can navigate them.
            qij_off = np.zeros(_natm + 1, dtype=np.int64)
            qia_off = np.zeros(_natm + 1, dtype=np.int64)
            qab_off = np.zeros(_natm + 1, dtype=np.int64)
            qij_n_aux = np.zeros(_natm, dtype=np.int32)
            qij_n_lmo = np.zeros(_natm, dtype=np.int32)
            qab_n_pao = np.zeros(_natm, dtype=np.int32)
            for A in range(_natm):
                if qij_stack[A] is not None:
                    nQA, nlA, _ = qij_stack[A].shape
                    qij_n_aux[A] = nQA
                    qij_n_lmo[A] = nlA
                    qij_off[A + 1] = qij_off[A] + qij_stack[A].size
                else:
                    qij_off[A + 1] = qij_off[A]
                if qia_stack[A] is not None:
                    qia_off[A + 1] = qia_off[A] + qia_stack[A].size
                else:
                    qia_off[A + 1] = qia_off[A]
                if qab_stack[A] is not None:
                    nQA2, npA, _ = qab_stack[A].shape
                    qab_n_pao[A] = npA
                    qab_off[A + 1] = qab_off[A] + qab_stack[A].size
                else:
                    qab_off[A + 1] = qab_off[A]
            # Stage 3: NVMe-back the (T) sparse-DF flats (qij/qia/qab_atom_flat)
            # when streaming is on. On a large basis the TIGHT + PRESCREEN
            # sparse-DF coexist (~2x) and are the (T)-build anon peak (rxn_12/
            # def2-tzvpp OOM'd here at anon 254). As file-backed pages they are
            # written once here, read per-triple in the energy pass, and the OS
            # can evict them under pressure. Pool workers write DISJOINT slabs
            # (safe on a memmap). flush() after fill -> clean/evictable.
            from pyscf.cc.dlpno_tccsd.pair_index import (
                stream_empty as _stream_empty, _should_spill as _shsp)
            # Whole (T) sparse-DF group decision (RAM-first; the 3 flats spill
            # together only when RAM is not enough).
            _tsp = _shsp((int(qij_off[-1]) + int(qia_off[-1])
                          + int(qab_off[-1])) * 8)
            qij_flat = _stream_empty((int(qij_off[-1]),), tag='tqij', spill=_tsp)
            qia_flat = _stream_empty((int(qia_off[-1]),), tag='tqia', spill=_tsp)
            qab_flat = _stream_empty((int(qab_off[-1]),), tag='tqab', spill=_tsp)
            # Per-atom flat copy: each atom writes a disjoint slab of each
            # flat array — independent. Pool-parallel for the same reason
            # the np.stack loop above is.
            def _flat_one_atom(A):
                if qij_stack[A] is not None:
                    qij_flat[qij_off[A]:qij_off[A + 1]] = (
                        np.ascontiguousarray(qij_stack[A]).ravel())
                if qia_stack[A] is not None:
                    qia_flat[qia_off[A]:qia_off[A + 1]] = (
                        np.ascontiguousarray(qia_stack[A]).ravel())
                if qab_stack[A] is not None:
                    qab_flat[qab_off[A]:qab_off[A + 1]] = (
                        np.ascontiguousarray(qab_stack[A]).ravel())
            if _pool is not None and _natm > 1:
                list(_pool.map(_flat_one_atom, range(_natm)))
            else:
                for A in range(_natm):
                    _flat_one_atom(A)
            for _b in (qij_flat, qia_flat, qab_flat):
                if isinstance(_b, np.memmap):
                    _b.flush()   # dirty -> clean/file-backed (evictable)
            sparse_df['qij_atom_flat'] = qij_flat
            sparse_df['qia_atom_flat'] = qia_flat
            sparse_df['qab_atom_flat'] = qab_flat
            # Free the per-atom stack LISTS (qij_atom/qia_atom/qab_atom): they
            # exist only to fill the flat buffers above.  The active per-triple
            # energy kernel (_orch -> C) reads exclusively the *_atom_flat
            # buffers; the sole reader of the stack lists,
            # _build_triple_local_DF, is dead (never called).  These are LARGE
            # arrays (qab_atom[A] ~ hundreds of MB, > mmap threshold -> returned
            # to the OS on free), so dropping them reclaims the full stack
            # footprint (~11 GiB on MOBH35-33/qzvpp) before the energy-pass peak.
            sparse_df.pop('qij_atom', None)
            sparse_df.pop('qia_atom', None)
            sparse_df.pop('qab_atom', None)
            del qij_stack, qia_stack, qab_stack
            import gc as _gc
            _gc.collect()
            sparse_df['qij_atom_off'] = qij_off
            sparse_df['qia_atom_off'] = qia_off
            sparse_df['qab_atom_off'] = qab_off
            sparse_df['qij_atom_n_aux'] = qij_n_aux
            sparse_df['qij_atom_n_lmo'] = qij_n_lmo
            sparse_df['qab_atom_n_pao'] = qab_n_pao
            # Streaming #2: the per-aux-Q lists (qij/qia/qab) are now redundant
            # — the per-atom stacks + flats above are np.stack/concat copies and
            # the triples loop consumes only qij_atom / qij_atom_flat.  Free them
            # for the PRESCREEN/subset pass (tight_sparse is not None).  For the
            # TIGHT build keep them: the prescreen derivation still slices them;
            # the caller frees them immediately after that (see below).
            if tight_sparse is not None:
                sparse_df['qij'] = sparse_df['qia'] = sparse_df['qab'] = None
            _bi("flat per-atom stacks")
            return lmo_aux_mask, pao_domains, screening, sparse_df

        # (T) entry: the CCSD cc_ints are freed by run_lccsd but glibc retains
        # the pages (no trim between CCSD and (T)).  Trim now so the (T) build
        # starts from the true live floor, not the cc_ints high-water mark.
        _tmem('T_entry')
        try:
            from pyscf.cc.dlpno_tccsd.driver import _malloc_trim as _mt0
            _mt0()
        except Exception:
            pass
        _tmem('T_entry_after_trim')
        # Tight infrastructure (used for the final (T) pass on surviving triples).
        _lmo_aux_mask, _pao_domains, _screening, _sparse_df = \
            _build_triples_infrastructure(
                _T_CUT_MKN_TRIPLES, _T_CUT_DO_TRIPLES, 'tight')
        _tp("build TIGHT sparse-DF infra")
        _tmem('after_tight_sparse')
        _j2c_full = _auxmol.intor('int2c2e')
        _S_pao_full = C_pao.T @ s1e @ C_pao
        _F_pao_full = C_pao.T @ fock_ao @ C_pao
        _tp("j2c + S_pao + F_pao")

        # Prescreen infrastructure (used by the loose (T0) pass).
        if T_CutTriplesWeak > 0.0:
            _lmo_aux_mask_pre, _pao_domains_pre, _screening_pre, _sparse_df_pre = \
                _build_triples_infrastructure(
                    _T_CUT_MKN_TRIPLES_PRE, _T_CUT_DO_TRIPLES_PRE, 'prescreen',
                    tight_sparse=_sparse_df, tight_screen=_screening)
            _tp("build PRESCREEN sparse-DF infra")
            _tmem('after_prescreen_sparse')
        else:
            _lmo_aux_mask_pre = _lmo_aux_mask
            _pao_domains_pre = _pao_domains
            _screening_pre = _screening
            _sparse_df_pre = _sparse_df
        # Streaming #2: the TIGHT per-aux-Q lists were retained only so the
        # prescreen subset above could slice them (derive_subset_sparse_df).
        # Both (T) passes consume the per-atom flats (qij_atom_flat), never the
        # per-Q lists, so drop them now — on MOBH35-12 this is the largest
        # freeable transient of the (T) phase.
        if _sparse_df is not None:
            _sparse_df['qij'] = _sparse_df['qia'] = _sparse_df['qab'] = None
        _tmem('after_free_tight_perQ')
    else:
        _lmo_aux_mask = None
        _sparse_df = None
        _screening = None
        _pao_domains = None
        _lmo_aux_mask_pre = None
        _sparse_df_pre = None
        _screening_pre = None
        _pao_domains_pre = None
        _S_pao_full = None
        _F_pao_full = None

    # Diagnostic: per-triple screening tightness. naux_ijk / naux_full and
    # len(triple_domain) / nocc tell us whether aux-Q and occupied domains
    # are O(1) per triple (good scaling) or growing with system size (bad).
    _n_tno_max_sampled = None
    if valid_triples:
        _naux_tot = _lmo_aux_mask.shape[1] if _lmo_aux_mask is not None else 0
        _ss_aux, _ss_dom, _ss_pao = [], [], []
        _p_diag = _build_partners(_tT_set, nocc_lmo)
        # Pre-convert pao_domains_triple to set-of-ints for fast union.
        _pao_doms_sets = None
        if _pao_domains is not None:
            _pao_doms_sets = [
                set(int(x) for x in np.asarray(_pao_domains[m]).tolist())
                if _pao_domains[m] is not None else set()
                for m in range(nocc_lmo)
            ]
        for _i, _j, _k in valid_triples:
            if _lmo_aux_mask is not None:
                _ss_aux.append(int((_lmo_aux_mask[_i] | _lmo_aux_mask[_j]
                                   | _lmo_aux_mask[_k]).sum()))
            _ss_dom.append(len(_p_diag[_i] & _p_diag[_j] & _p_diag[_k]))
            if _pao_doms_sets is not None:
                _ss_pao.append(len(_pao_doms_sets[_i] | _pao_doms_sets[_j]
                                   | _pao_doms_sets[_k]))
        if _ss_aux:
            print(f'  (T) naux_ijk: avg={np.mean(_ss_aux):.1f}/{_naux_tot} '
                  f'({100.0*np.mean(_ss_aux)/max(_naux_tot,1):.1f}%)  '
                  f'min={min(_ss_aux)}  max={max(_ss_aux)}',
                  flush=True)
        print(f'  (T) triple_domain: avg={np.mean(_ss_dom):.1f}/{nocc_lmo} '
              f'({100.0*np.mean(_ss_dom)/max(nocc_lmo,1):.1f}%)  '
              f'min={min(_ss_dom)}  max={max(_ss_dom)}',
              flush=True)
        if _ss_pao:
            print(f'  (T) n_pao_ijk: avg={np.mean(_ss_pao):.1f}  '
                  f'min={min(_ss_pao)}  max={max(_ss_pao)}',
                  flush=True)

        # Sample n_tno on up to 30 triples (run TNO build to get n_tno).
        if (_S_pao_full is not None and _F_pao_full is not None
                and C_pao is not None):
            import random as _rnd
            _rnd.seed(42)
            _sample = _rnd.sample(valid_triples,
                                  min(30, len(valid_triples)))
            _ss_tno = []
            for (_i, _j, _k) in _sample:
                try:
                    _r = _triple_pno_union_psi4(
                        pno_spaces, _i, _j, _k, C_pao, _S_pao_full,
                        _F_pao_full, t2_for_T=t2_for_T,
                        T_CutTNO=T_CutTNO,
                        pao_domains_triple=_pao_domains)
                    _ss_tno.append(int(_r[1]))   # n_tno
                except Exception:
                    pass
            if _ss_tno:
                _n_tno_max_sampled = max(_ss_tno)
                print(f'  (T) n_tno (sampled {len(_ss_tno)}): '
                      f'avg={np.mean(_ss_tno):.1f}  min={min(_ss_tno)}  '
                      f'max={_n_tno_max_sampled}',
                      flush=True)
    _tp("diagnostic stats + 30-tno sample")

    # ------------------------------------------------------------------
    # Memory-aware (T) worker count. The (T) energy pass is the RSS peak of
    # the whole calculation; its footprint is base + sparse_DF + W*nthreads
    # (W = per-triple C scratch). Cap the number of concurrent triples so
    # that peak fits the box. Default (no env set) = whole shared pool, i.e.
    # unchanged behaviour. See _t_worker_count for the knobs.
    # ------------------------------------------------------------------
    _pool_workers = (getattr(_pool, '_max_workers', ncores)
                     if _pool is not None else 1)
    _t_workers = _t_worker_count(_pool_workers, _n_tno_max_sampled)
    _tpool = _pool
    _tpool_owned = False
    if _pool is not None and _t_workers < _pool_workers:
        from concurrent.futures import ThreadPoolExecutor as _TPE
        _tpool = _TPE(max_workers=_t_workers)
        _tpool_owned = True
    # The OMP backend (DLPNO_TRIPLE_OMP=1) has its own thread knob and does
    # NOT use the Python pool; propagate the cap to it unless the user pinned
    # it explicitly.
    if (os.environ.get('DLPNO_TRIPLE_OMP') == '1'
            and _t_workers != _pool_workers
            and os.environ.get('DLPNO_TRIPLES_OMP_THREADS') is None):
        os.environ['DLPNO_TRIPLES_OMP_THREADS'] = str(_t_workers)
    if _t_workers != _pool_workers or os.environ.get('DLPNO_MEM_PROBE'):
        _ma = _mem_available_gb()
        print(f'  (T) energy-pass workers: {_t_workers} / pool {_pool_workers}'
              f'  (MemAvailable {_ma:.0f} GiB)' if _ma is not None
              else f'  (T) energy-pass workers: {_t_workers} / pool '
                   f'{_pool_workers}', flush=True)

    triple_kwargs = dict(
        pno_spaces=pno_spaces, t2_for_T=t2_for_T,
        Lpq_full=Lpq_full, C_lmo=C_lmo,
        fock_ao=fock_ao, F_lmo=F_lmo, s1e=s1e,
        t1_pno=t1_pno,
        T_CutTNO=T_CutTNO,
        nonneg_set=_tT_set,
        sparse_df=_sparse_df, screening=_screening,
        j2c_full=_j2c_full, lmo_aux_mask=_lmo_aux_mask,
        C_pao=C_pao,
        S_pao_full=_S_pao_full, F_pao_full=_F_pao_full,
        pao_domains_triple=_pao_domains,
    )

    def _do_triple(ijk):
        return _process_one_triple(ijk[0], ijk[1], ijk[2], **triple_kwargs)

    # ------------------------------------------------------------------
    # (3) SC-MP2 (T0) prescreen to drop low-contribution triples.
    # Psi4 (triples.cc:1296-1326) runs the prescreen on a SEPARATE sparse-DF
    # infrastructure built with LOOSER thresholds (T_CUT_MKN_TRIPLES_PRE,
    # T_CUT_DO_TRIPLES_PRE, T_CUT_TNO_PRE). Surviving triples are then
    # rerun at the tight thresholds. This makes the (T0) pass genuinely
    # cheap while preserving accuracy on the triples that matter.
    # ------------------------------------------------------------------
    _T_CUT_TNO_PRE = max(T_CutTNO, 1e-7)   # Psi4 read_options.cc:2565
    _T_CUT_TRIPLES_WEAK = T_CutTriplesWeak
    e_t_screened = 0.0   # dropped-triple contribution, ADDED BACK at end
    if (len(valid_triples) > 0 and _T_CUT_TNO_PRE > T_CutTNO
            and _T_CUT_TRIPLES_WEAK > 0.0):
        # Build kwargs for the prescreen pass: LOOSE thresholds on every
        # axis — PAO domain, aux-Q mask, sparse-DF stacks, screening maps,
        # and TNO cutoff. Everything else (triple-paos union, F/S matrices,
        # t2 amplitudes) is shared with the tight pass.
        _pre_kwargs = dict(triple_kwargs)
        _pre_kwargs['T_CutTNO'] = _T_CUT_TNO_PRE
        _pre_kwargs['sparse_df'] = _sparse_df_pre
        _pre_kwargs['screening'] = _screening_pre
        _pre_kwargs['lmo_aux_mask'] = _lmo_aux_mask_pre
        _pre_kwargs['pao_domains_triple'] = _pao_domains_pre

        def _pre_triple(ijk):
            return _process_one_triple(ijk[0], ijk[1], ijk[2], **_pre_kwargs)

        if (os.environ.get('DLPNO_TRIPLE_OMP', '0') == '1'
                and _pre_kwargs.get('sparse_df') is not None
                and _pre_kwargs.get('S_pao_full') is not None):
            _et_pre = list(_run_triples_omp(
                valid_triples,
                _pre_kwargs['pno_spaces'], _pre_kwargs['t2_for_T'],
                _pre_kwargs['C_pao'], _pre_kwargs['S_pao_full'],
                _pre_kwargs['F_pao_full'], F_lmo,
                _pre_kwargs['sparse_df'], _pre_kwargs['screening'],
                _pre_kwargs['j2c_full'], _pre_kwargs['lmo_aux_mask'],
                _pre_kwargs.get('pao_domains_triple'),
                _pre_kwargs.get('nonneg_set'),
                _pre_kwargs['T_CutTNO'],
                _pre_kwargs.get('t1_pno')))
        elif _tpool is not None:
            _et_pre = list(_tpool.map(_pre_triple, valid_triples))
        else:
            _et_pre = [_pre_triple(ijk) for ijk in valid_triples]
        _tp("PRESCREEN pass (whole call)")

        _kept = [ijk for ijk, e in zip(valid_triples, _et_pre)
                 if abs(e) >= _T_CUT_TRIPLES_WEAK]
        # Psi4 (triples.cc:235, 252): the prescreen energy of *dropped*
        # triples is tracked as de_lccsd_t_screened_ and ADDED to the
        # final (T) correction — that way the prescreen error is
        # bounded by T_CUT_TRIPLES_WEAK × n_triples rather than the
        # full per-triple contribution.
        e_t_screened = sum(e for e in _et_pre
                           if abs(e) < _T_CUT_TRIPLES_WEAK)
        print(f'  (T) SC-MP2 prescreen: kept {len(_kept)} / '
              f'{len(valid_triples)} ({100.0*len(_kept)/len(valid_triples):.1f}%)'
              f', screened-back energy = {e_t_screened:.3e} Eh',
              flush=True)
        valid_triples = _kept
        _tp("prescreen filter")
        # Streaming #2: the PRESCREEN sparse-DF infrastructure (per-atom flats
        # qij_atom_flat/etc.) is consumed only by the prescreen pass above; the
        # main (T) energy pass uses the TIGHT _sparse_df.  These are large
        # contiguous (mmap-backed) arrays, so freeing them here returns memory
        # to the OS before the energy-pass peak (~11.6 GiB on MOBH35-12/svp).
        # Guard: when no prescreen ran, _sparse_df_pre IS _sparse_df (the tight
        # set the main pass needs) — never free that.
        if (_sparse_df_pre is not None
                and _sparse_df_pre is not _sparse_df):
            _sparse_df_pre.clear()
        _sparse_df_pre = None
        _lmo_aux_mask_pre = _pao_domains_pre = _screening_pre = None
        try:
            from pyscf.cc.dlpno_tccsd.driver import _malloc_trim as _mt
            _mt()
        except Exception:
            pass
        _tmem('after_free_prescreen_infra')

    # Degenerate occupied triples: pairs (i,k) with i<k capture
    # {i,i,k} and {i,k,k} contributions (60% of canonical (T)).
    valid_pairs = [(i, k) for i in range(nocc_lmo)
                           for k in range(i + 1, nocc_lmo)
                           if (i, k) in _tT_set]

    def _do_degen(ik):
        return _process_degenerate_pair(ik[0], ik[1], **triple_kwargs)

    # Reuse the shared pool from the driver (same threads as LCCSD stage).
    # Set BLAS to single-thread during pool phase, restore after.
    _tmem('before_energy_pass')

    # (pair, RI-atom) H-cache for the q_vv build (df_qvv was 82% of (T)
    # wall; each strong pair is shared by ~33 triples).  Entries are built
    # lazily in C by the first worker that touches a (pair, atom) slot.
    # DLPNO_T_PAIRCACHE=0 disables; DLPNO_T_PAIRCACHE_GB caps memory.
    global _QVV_PAIR_ID_MAP
    _paircache_on = (os.environ.get('DLPNO_T_PAIRCACHE', '1') != '0'
                     and _sparse_df is not None)
    _qvvc_lib = None
    if _paircache_on:
        import ctypes as _ct_pc
        from pyscf import lib as _pl_pc
        _qvvc_lib = _pl_pc.load_library('libcc')
        _qvvc_lib.DLPNOqvv_cache_init.argtypes = [
            _ct_pc.c_long, _ct_pc.c_long, _ct_pc.c_double]
        _qvvc_lib.DLPNOqvv_cache_free.argtypes = []
        _qvvc_lib.DLPNOqvv_cache_stats.argtypes = [_ct_pc.c_void_p]
        _pc_budget = float(os.environ.get('DLPNO_T_PAIRCACHE_GB', '150'))
        # Priority admission: ids are handed out to pairs by descending
        # triple-use count until the ESTIMATED cache footprint reaches the
        # budget (the C side still enforces the hard cap).  Without this,
        # first-touch order fills the budget with arbitrary pairs and the
        # heavy-reuse ones get skipped.  Worst-case bytes per pair is
        # 8 * n_pno * sum_A(pages_A*np_A); actual entries only materialize
        # for atoms a triple actually touches — empirically ~half — hence
        # the 0.5 factor on the estimate.
        from collections import Counter as _Ctr
        _pc_cnt = _Ctr()
        for (_ti, _tj, _tk) in valid_triples:
            for _pk in ((min(_ti, _tj), max(_ti, _tj)),
                        (min(_tj, _tk), max(_tj, _tk)),
                        (min(_ti, _tk), max(_ti, _tk))):
                _pc_cnt[_pk] += 1
        for (_di, _dk) in valid_pairs:      # degenerate (i,i,k) + (i,k,k)
            _pc_cnt[(_di, _di)] += 1
            _pc_cnt[(_dk, _dk)] += 1
            _pc_cnt[(_di, _dk)] += 4
        _qab_off = np.asarray(_sparse_df['qab_atom_off'])
        _qab_np = np.asarray(_sparse_df['qab_atom_n_pao'], dtype=np.float64)
        _pages_np = float(np.sum(np.divide(
            np.diff(_qab_off), _qab_np, where=_qab_np > 0,
            out=np.zeros_like(_qab_np))))          # sum_A pages_A * np_A
        _budget_bytes = _pc_budget * 2**30
        _pc_map = {}
        _est = 0.0
        for _pk, _n_use in _pc_cnt.most_common():
            _pd = pno_spaces.get(_pk)
            if _pd is None or _pd.get('X_pno') is None:
                continue
            _npno = int(_pd['X_pno'].shape[1])
            if _npno == 0:
                continue
            _est_pk = 0.5 * 8.0 * _npno * _pages_np
            if _est + _est_pk > _budget_bytes:
                continue
            _est += _est_pk
            _pc_map[_pk] = len(_pc_map)
        print(f'  (T) qvv pair-cache: admitting {len(_pc_map)} / '
              f'{len(_pc_cnt)} pairs (est {_est/2**30:.1f} / '
              f'{_pc_budget:.0f} GiB)', flush=True)
        _qvvc_lib.DLPNOqvv_cache_init(max(len(_pc_map), 1),
                                      int(mf.mol.natm), _pc_budget)
        _QVV_PAIR_ID_MAP = _pc_map
    if (os.environ.get('DLPNO_TRIPLE_OMP', '0') == '1'
            and triple_kwargs.get('sparse_df') is not None
            and triple_kwargs.get('S_pao_full') is not None):
        et_values = list(_run_triples_omp(
            valid_triples,
            triple_kwargs['pno_spaces'], triple_kwargs['t2_for_T'],
            triple_kwargs['C_pao'], triple_kwargs['S_pao_full'],
            triple_kwargs['F_pao_full'], F_lmo,
            triple_kwargs['sparse_df'], triple_kwargs['screening'],
            triple_kwargs['j2c_full'], triple_kwargs['lmo_aux_mask'],
            triple_kwargs.get('pao_domains_triple'),
            triple_kwargs.get('nonneg_set'),
            triple_kwargs['T_CutTNO'],
            triple_kwargs.get('t1_pno')))
        _tp("ENERGY pass distinct (OMP)")
        if _tpool is not None:
            et_degen_values = list(_tpool.map(_do_degen, valid_pairs))
        else:
            et_degen_values = [_do_degen(ik) for ik in valid_pairs]
        _tp("ENERGY pass degenerate (pool)")
    elif _tpool is not None:
        et_values = list(_tpool.map(_do_triple, valid_triples))
        _tp("ENERGY pass distinct (pool)")
        et_degen_values = list(_tpool.map(_do_degen, valid_pairs))
        _tp("ENERGY pass degenerate (pool)")
    else:
        et_values = [_do_triple(ijk) for ijk in valid_triples]
        _tp("ENERGY pass distinct (serial)")
        et_degen_values = [_do_degen(ik) for ik in valid_pairs]
        _tp("ENERGY pass degenerate (serial)")

    if _qvvc_lib is not None:
        _st = np.zeros(5, dtype=np.float64)
        _qvvc_lib.DLPNOqvv_cache_stats(_st.ctypes.data_as(
            __import__('ctypes').c_void_p))
        print(f'  (T) qvv pair-cache: built={int(_st[0])} '
              f'hits={int(_st[1])} fallback={int(_st[2])} '
              f'skipped={int(_st[3])} mem={_st[4]/2**30:.2f} GiB', flush=True)
        _QVV_PAIR_ID_MAP = None
        _qvvc_lib.DLPNOqvv_cache_free()

    if _tpool_owned:
        _tpool.shutdown(wait=True)
        _tpool = None

    e_t_distinct = sum(et_values)
    n_triples = sum(1 for v in et_values if v != 0.0)

    e_t_degen = sum(et_degen_values)
    n_degen = sum(1 for v in et_degen_values if v != 0.0)

    # Add back the SC-MP2 prescreen energy of dropped triples — the error
    # vs a full tight-TNO calculation is bounded by T_CUT_TRIPLES_WEAK per
    # dropped triple (Psi4 de_lccsd_t_screened_ convention).
    e_t = e_t_distinct + e_t_degen + e_t_screened

    log.info('(T) correction: %d distinct triples, %d degenerate pairs '
             '(%d + %d candidates)',
             n_triples, n_degen, len(valid_triples), len(valid_pairs))
    log.info('E(T) distinct = %.15g', e_t_distinct)
    log.info('E(T) degenerate = %.15g', e_t_degen)
    log.info('E(T) screened-back = %.15g', e_t_screened)
    log.info('E(T) external = %.15g', e_t)


    return e_t


# =============================================================================
# (T1) Iterative Triples — Jiang JCP 2024 / Psi4 lccsd_t_iterations
# =============================================================================

def _build_triple_W_V_T0(i, j, k, pno_spaces, t2_for_T,
                          Lpq_full, C_lmo, fock_ao, F_lmo, s1e,
                          ooL_full, t1_pno=None, T_CutTNO=1e-9):
    """Compute W, V, T0 and TNO data for one triple.

    No occupied semicanonalization — uses diagonal LMO Fock for the denominator.
    The virtual space is semicanonalized (Fock-diagonal TNOs).

    Args:
        ooL_full: (nocc, nocc, naux) prebuilt ooL for all occupied pairs.

    Returns:
        dict with W, V, T, C_tno_sc, eps_tno, n_tno, or None if skipped.
    """
    ij = (min(i, j), max(i, j))
    ik = (min(i, k), max(i, k))
    jk = (min(j, k), max(j, k))

    if (ij not in pno_spaces or ik not in pno_spaces or jk not in pno_spaces):
        return None
    if (ij not in t2_for_T or ik not in t2_for_T or jk not in t2_for_T):
        return None

    C_tno, n_tno = _triple_pno_union(pno_spaces, i, j, k, s1e,
                                      t2_for_T=t2_for_T, T_CutTNO=T_CutTNO)
    if n_tno == 0:
        return None

    n = n_tno
    triple_lmo = [i, j, k]
    nocc = C_lmo.shape[1]

    # Virtual semicanonalization only
    F_tno_full = reduce(np.dot, (C_tno.T, fock_ao, C_tno))
    eps_tno, V_sc = np.linalg.eigh(F_tno_full)
    C_tno_sc = np.dot(C_tno, V_sc)

    # DF integrals in LMO-occ × SC-vir basis
    ovL = _build_ovL_tno(Lpq_full, C_lmo, C_tno_sc, triple_lmo)  # (3, n, naux)
    vvL = _build_vvL_tno(Lpq_full, C_tno_sc)  # (n, n, naux)

    # T2 for all occ paired with triple LMOs: t2_mr[l, r_local] = t2[l, triple[r]]
    def _proj_t2(p, q):
        pk = (min(p, q), max(p, q))
        if pk not in t2_for_T or pk not in pno_spaces:
            return np.zeros((n, n))
        C_p = pno_spaces[pk]['C_pno']
        if C_p.shape[1] == 0:
            return np.zeros((n, n))
        U = reduce(np.dot, (C_p.T, s1e, C_tno_sc))
        t2_proj = reduce(np.dot, (U.T, t2_for_T[pk], U))
        if p > q:
            t2_proj = t2_proj.T
        return t2_proj

    t2_mr = np.zeros((nocc, 3, n, n))
    for r_local, r_global in enumerate(triple_lmo):
        for m in range(nocc):
            t2_mr[m, r_local] = _proj_t2(m, r_global)

    # P_L-assembled W (corrected virtual transposes)
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    trans_axes = [(0,1,2),(0,2,1),(1,0,2),(2,0,1),(1,2,0),(2,1,0)]

    W = np.zeros((n, n, n))
    for pidx, (ip, iq, ir) in enumerate(perms):
        # vvov: sum_f (ip_a|b f) * t2[r,q,c,f]
        K_abf = np.einsum('aL,fbL->abf', ovL[ip], vvL)
        t2_rq = t2_mr[triple_lmo[ir], iq]  # t2[triple[ir], triple[iq]]
        base = np.einsum('abf,cf->abc', K_abf, t2_rq)

        # vooo: -sum_l (ip_a | iq_l) * t2[l, ir, b, c]  (Eq47 factorization)
        A_al = np.einsum('aL,lL->al', ovL[ip], ooL_full[triple_lmo[iq]])
        base -= np.einsum('al,lbc->abc', A_al, t2_mr[:, ir])

        W += base.transpose(trans_axes[pidx])

    # Denominator: diagonal LMO Fock (no occupied SC)
    eps_occ = np.array([F_lmo[ii, ii] for ii in triple_lmo])
    D_occ = eps_occ[0] + eps_occ[1] + eps_occ[2]
    D = eps_tno[:, None, None] + eps_tno[None, :, None] + eps_tno[None, None, :] - D_occ

    T = -W / D

    # V = W + T1 disconnected
    V = W.copy()
    if t1_pno is not None:
        t1_tno = np.zeros((3, n))
        for idx_local, r_global in enumerate(triple_lmo):
            key_rr = (r_global, r_global)
            if key_rr in pno_spaces and t1_pno.get(r_global) is not None:
                C_pno_rr = pno_spaces[key_rr]['C_pno']
                if C_pno_rr.shape[1] > 0 and t1_pno[r_global].size > 0:
                    U_rr = reduce(np.dot, (C_pno_rr.T, s1e, C_tno_sc))
                    t1_tno[idx_local] = U_rr.T @ t1_pno[r_global]
        K_jk = np.einsum('bL,cL->bc', ovL[1], ovL[2])
        K_ik = np.einsum('aL,cL->ac', ovL[0], ovL[2])
        K_ij = np.einsum('aL,bL->ab', ovL[0], ovL[1])
        V += (np.einsum('a,bc->abc', t1_tno[0], K_jk)
              + np.einsum('b,ac->abc', t1_tno[1], K_ik)
              + np.einsum('c,ab->abc', t1_tno[2], K_ij))

    return {
        'W': W, 'V': V, 'T': T,
        'C_tno_sc': C_tno_sc,
        'eps_tno': eps_tno,
        'D': D,
        'n_tno': n_tno,
        'i': i, 'j': j, 'k': k,
    }


def _triples_permuter(X, i_perm, j_perm, k_perm):
    """Permute virtual indices of X[a,b,c] based on occupied ordering.

    X is stored with canonical ordering (i<=j<=k). Given a target occupied
    ordering (i_perm, j_perm, k_perm), return X with virtual indices permuted
    to match. Follows Psi4's triples_permuter() logic.

    The mapping: determine which S_3 element maps the canonical ordering
    to the target ordering, then apply the same permutation to (a,b,c).
    """
    # Determine permutation index from the sorted ordering
    if i_perm <= j_perm and j_perm <= k_perm:
        return X                                        # identity
    elif i_perm <= k_perm and k_perm <= j_perm:
        return X.transpose(0, 2, 1)                     # swap b↔c
    elif j_perm <= i_perm and i_perm <= k_perm:
        return X.transpose(1, 0, 2)                     # swap a↔b
    elif j_perm <= k_perm and k_perm <= i_perm:
        # cycle: (j,k,i) stored as (i',j',k') where i'<=j'<=k'
        # perm_idx 3 → (b,c,a) in Psi4
        return X.transpose(2, 0, 1)                     # a→c, b→a, c→b
    elif k_perm <= i_perm and i_perm <= j_perm:
        # cycle: (k,i,j) stored as (i',j',k') where i'<=j'<=k'
        # perm_idx 4 → (c,a,b) in Psi4
        return X.transpose(1, 2, 0)                     # a→b, b→c, c→a
    else:
        return X.transpose(2, 1, 0)                     # swap a↔c


def _project_t3(T_src, S):
    """Project T3 amplitudes from source TNO basis to target TNO basis.

    Implements: T_tgt[a,b,c] = sum_{a',b',c'} S[a,a'] S[b,b'] S[c,c'] T_src[a',b',c']
    Uses BLAS matmul for efficiency (3 dgemm calls).

    Args:
        T_src: (n_src, n_src, n_src) T3 in source TNO basis.
        S: (n_tgt, n_src) overlap matrix between target and source TNOs.

    Returns:
        T_tgt: (n_tgt, n_tgt, n_tgt) T3 in target TNO basis.
    """
    n_tgt, n_src = S.shape
    # Contract first index (a'): result[i, b', c'] = sum_a' S[i,a'] T[a',b',c']
    tmp = np.dot(S, T_src.reshape(n_src, n_src * n_src))   # (n_tgt, n_src^2)
    tmp = tmp.reshape(n_tgt, n_src, n_src)                  # (i, b', c')
    # Contract second index (b'): put b' as column for matmul
    tmp = tmp.transpose(0, 2, 1).reshape(n_tgt * n_src, n_src)  # (i*n_src+c', b')
    tmp = np.dot(tmp, S.T)                                  # (i*n_src+c', j)
    # Contract third index (c'): put c' as column for matmul
    tmp = tmp.reshape(n_tgt, n_src, n_tgt).transpose(0, 2, 1)  # (i, j, c')
    return np.dot(tmp.reshape(n_tgt * n_tgt, n_src), S.T).reshape(n_tgt, n_tgt, n_tgt)


def _compute_t1_energy(triple_data_list, triple_idx_map, F_lmo, nocc):
    """Compute (T) energy from stored V and T using Eq53 antisymmetrizer.

    For each triple, applies: e_ijk = prefactor * (8V·T - 4V(cba)·T - ...)

    Args:
        triple_data_list: list of dicts from _build_triple_W_V_T0 (or None).
        triple_idx_map: dict (i,j,k) → index into triple_data_list.
        F_lmo: (nocc, nocc) LMO Fock matrix.
        nocc: number of occupied.

    Returns:
        e_t (float), e_ijk_list (list of per-triple energies).
    """
    e_ijk_list = [0.0] * len(triple_data_list)

    for idx, td in enumerate(triple_data_list):
        if td is None:
            continue
        i, j, k = td['i'], td['j'], td['k']
        V = td['V']
        T = td['T']

        # Degeneracy factor
        dij = int(i == j); djk = int(j == k); dik = int(i == k)
        occ_denom = 1 + dij + djk + dik + 2 * dij * djk * dik

        # Eq53 antisymmetrizer: 8V·T - 4V(kji)·T - 4V(ikj)·T - 4V(jik)·T
        #                        + 2V(jki)·T + 2V(kij)·T
        et = (8.0 * np.sum(V * T)
              - 4.0 * np.sum(V.transpose(2, 1, 0) * T)    # V(k,j,i)
              - 4.0 * np.sum(V.transpose(0, 2, 1) * T)    # V(i,k,j)
              - 4.0 * np.sum(V.transpose(1, 0, 2) * T)    # V(j,i,k)
              + 2.0 * np.sum(V.transpose(1, 2, 0) * T)    # V(j,k,i) → transpose(2,0,1) for (b,c,a)
              + 2.0 * np.sum(V.transpose(2, 0, 1) * T))   # V(k,i,j) → transpose(1,2,0) for (c,a,b)

        # Wait: the antisymmetrizer permutes V's virtual indices based on
        # occupied permutation. For V stored as V[a,b,c] (a↔i,b↔j,c↔k),
        # permute(V, k,j,i) means virtual perm matching (k,j,i):
        #   perm_idx=5 → (c,b,a) → V.transpose(2,1,0) ✓
        # permute(V, i,k,j) → perm_idx=1 → (a,c,b) → V.transpose(0,2,1) ✓
        # permute(V, j,i,k) → perm_idx=2 → (b,a,c) → V.transpose(1,0,2) ✓
        # permute(V, j,k,i) → perm_idx=3 → (b,c,a) → V.transpose(2,0,1) ✓
        # permute(V, k,i,j) → perm_idx=4 → (c,a,b) → V.transpose(1,2,0) ✓

        e_ijk_list[idx] = float(et / occ_denom)

    return sum(e_ijk_list), e_ijk_list


def run_lccsd_t1_iterations(mf, C_lmo, pno_spaces, strong_pairs,
                             t2_pno_all, occ_cas_idx,
                             t1_pno=None,
                             C_cas_vir=None,
                             T_CutTNO=1e-9,
                             ncores=1,
                             verbose=None,
                             _pool=None,
                             max_iter=50,
                             e_conv=1e-8,
                             r_conv=1e-6,
                             F_CUT_T=1e-5,
                             T_CUT_ITER=0.0):
    """Compute (T1) iterative triples correction (Jiang JCP 2024).

    Improves upon the (T0) semicanonical approximation by iteratively
    solving for T3 amplitudes including off-diagonal occupied Fock coupling
    between neighboring triples.

    The algorithm (following Psi4's lccsd_t_iterations):
    1. Compute W, V for all triples; initialize T = -W/D (T0).
    2. Iterate:
       R = W + T*D - sum_l F_lmo[l,k] * S_overlap * T3_neighbor
       T -= R/D
    3. Compute energy from converged V and T.

    Args:
        Same as run_lccsd_t_ext, plus:
        max_iter (int): Maximum number of iterations (default 50).
        e_conv (float): Energy convergence threshold (default 1e-8).
        r_conv (float): Residual convergence threshold (default 1e-6).
        F_CUT_T (float): Off-diagonal Fock cutoff for coupling (default 1e-5).
        T_CUT_ITER (float): Skip triples whose energy changed less than
            this fraction (default 0.0 = no skipping).

    Returns:
        e_t (float): Converged (T1) energy correction.
    """
    import time as _time
    log = logger.new_logger(mf, verbose)

    if not hasattr(mf, 'with_df') or mf.with_df is None:
        import warnings
        warnings.warn('mf.with_df is None — (T) correction requires DF.', UserWarning)
        return 0.0

    occ_cas_set = set(occ_cas_idx.tolist()) if occ_cas_idx is not None else set()
    nocc = C_lmo.shape[1]
    s1e = mf.get_ovlp()
    fock_ao = getattr(mf, '_dlpno_fock_ao', None)
    if fock_ao is None:
        fock_ao = mf.get_fock()
    F_lmo = reduce(np.dot, (C_lmo.T, fock_ao, C_lmo))

    # Zero CAS T2 amplitudes
    if C_cas_vir is not None and C_cas_vir.shape[1] > 0:
        t2_for_T = _zero_cas_t2_amplitudes(
            t2_pno_all, pno_spaces, occ_cas_idx, C_cas_vir, s1e)
    else:
        t2_for_T = t2_pno_all

    # Preload DF integrals
    _t0 = _time.perf_counter()
    Lpq_full = _preload_df_integrals(mf.with_df)
    log.info('(T1) preloaded DF integrals: %.1f MB, %.2f s',
             Lpq_full.nbytes / 1e6, _time.perf_counter() - _t0)

    # Build full ooL (nocc, nocc, naux) once for all triples
    _t0 = _time.perf_counter()
    ooL_full = _build_ooL_triple(Lpq_full, C_lmo, list(range(nocc)))
    log.info('(T1) built ooL_full: shape=%s, %.2f s',
             ooL_full.shape, _time.perf_counter() - _t0)

    # =========================================================================
    # Phase 1: Enumerate triples (i<=j<=k) and compute W, V, T0
    # =========================================================================
    all_triples = []
    for i in range(nocc):
        for j in range(i, nocc):
            for k in range(j, nocc):
                all_triples.append((i, j, k))

    triple_idx_map = {ijk: idx for idx, ijk in enumerate(all_triples)}
    n_triples = len(all_triples)

    _t0 = _time.perf_counter()
    triple_data = [None] * n_triples

    common_kwargs = dict(
        pno_spaces=pno_spaces, t2_for_T=t2_for_T,
        Lpq_full=Lpq_full, C_lmo=C_lmo, fock_ao=fock_ao,
        F_lmo=F_lmo, s1e=s1e, ooL_full=ooL_full,
        t1_pno=t1_pno, T_CutTNO=T_CutTNO,
    )

    def _build_one(ijk):
        return _build_triple_W_V_T0(ijk[0], ijk[1], ijk[2], **common_kwargs)

    if _pool is not None:
        results = list(_pool.map(_build_one, all_triples))
    else:
        results = [_build_one(ijk) for ijk in all_triples]

    for idx, td in enumerate(results):
        triple_data[idx] = td

    n_active = sum(1 for td in triple_data if td is not None)
    _dt = _time.perf_counter() - _t0
    log.info('(T1) Phase 1: %d/%d active triples, %.2f s', n_active, n_triples, _dt)

    if n_active == 0:
        log.info('(T1) No active triples — returning 0.')
        return 0.0

    # Compute initial T0 energy
    e_t0, e_ijk_list = _compute_t1_energy(triple_data, triple_idx_map, F_lmo, nocc)
    log.info('(T1) E(T0) = %.12f', e_t0)

    # =========================================================================
    # Phase 2: Build coupling structure and TNO overlap matrices
    # =========================================================================
    # For each active triple (i,j,k), find neighbors (i,j,l), (i,l,k), (l,j,k)
    # with F_lmo[l, replaced_idx] >= F_CUT_T.
    # Pre-compute TNO→TNO overlap S and cache.

    _t0 = _time.perf_counter()
    # Cache: coupling_info[idx] = list of (neighbor_idx, F_coupling, S_overlap, perm_occ)
    coupling_info = [[] for _ in range(n_triples)]

    for idx, td in enumerate(triple_data):
        if td is None:
            continue
        i, j, k = td['i'], td['j'], td['k']
        n_tgt = td['n_tno']
        C_tgt = td['C_tno_sc']  # (nao, n_tgt)

        for l in range(nocc):
            # Channel 1: (i,j,l) replaces k, coupling F_lmo[l,k]
            if l != k and abs(F_lmo[l, k]) >= F_CUT_T:
                # Canonical ordering of (i,j,l)
                sorted_ijl = tuple(sorted([i, j, l]))
                if sorted_ijl in triple_idx_map:
                    nbr_idx = triple_idx_map[sorted_ijl]
                    nbr_td = triple_data[nbr_idx]
                    if nbr_td is not None:
                        # TNO overlap: S(n_tgt, n_src)
                        S_ov = reduce(np.dot, (C_tgt.T, s1e, nbr_td['C_tno_sc']))
                        coupling_info[idx].append((
                            nbr_idx, -F_lmo[l, k], S_ov,
                            (i, j, l)  # target occ ordering for permuter
                        ))

            # Channel 2: (i,l,k) replaces j, coupling F_lmo[l,j]
            if l != j and abs(F_lmo[l, j]) >= F_CUT_T:
                sorted_ilk = tuple(sorted([i, l, k]))
                if sorted_ilk in triple_idx_map:
                    nbr_idx = triple_idx_map[sorted_ilk]
                    nbr_td = triple_data[nbr_idx]
                    if nbr_td is not None:
                        S_ov = reduce(np.dot, (C_tgt.T, s1e, nbr_td['C_tno_sc']))
                        coupling_info[idx].append((
                            nbr_idx, -F_lmo[l, j], S_ov,
                            (i, l, k)
                        ))

            # Channel 3: (l,j,k) replaces i, coupling F_lmo[l,i]
            if l != i and abs(F_lmo[l, i]) >= F_CUT_T:
                sorted_ljk = tuple(sorted([l, j, k]))
                if sorted_ljk in triple_idx_map:
                    nbr_idx = triple_idx_map[sorted_ljk]
                    nbr_td = triple_data[nbr_idx]
                    if nbr_td is not None:
                        S_ov = reduce(np.dot, (C_tgt.T, s1e, nbr_td['C_tno_sc']))
                        coupling_info[idx].append((
                            nbr_idx, -F_lmo[l, i], S_ov,
                            (l, j, k)
                        ))

    n_couplings = sum(len(ci) for ci in coupling_info)
    _dt = _time.perf_counter() - _t0
    log.info('(T1) Phase 2: %d coupling terms, %.2f s', n_couplings, _dt)

    # =========================================================================
    # Phase 3: Jacobi iterations
    # =========================================================================
    log.info('')
    log.info('  ==> Local CCSD(T1) Iterations <==')
    log.info('')
    log.info('  E_CONVERGENCE = %.2e', e_conv)
    log.info('  R_CONVERGENCE = %.2e', r_conv)
    log.info('  F_CUT_T       = %.2e', F_CUT_T)
    log.info('')
    log.info('  %5s %18s %12s %12s %8s',
             'Iter', 'Corr. Energy', 'Delta E', 'Max R', 'Time')

    e_prev = e_t0
    e_ijk_old = list(e_ijk_list)

    for iteration in range(1, max_iter + 1):
        _t_iter = _time.perf_counter()
        r_max_list = [0.0] * n_triples

        # Snapshot current T for Jacobi semantics (read from old, write to new)
        T_snapshot = [td['T'].copy() if td is not None else None
                      for td in triple_data]

        for idx, td in enumerate(triple_data):
            if td is None:
                continue

            if T_CUT_ITER > 0 and abs(e_ijk_list[idx] - e_ijk_old[idx]) < abs(e_ijk_old[idx] * T_CUT_ITER):
                continue

            W = td['W']
            T_old = T_snapshot[idx]
            D = td['D']

            R = W + T_old * D

            for nbr_idx, f_coupling, S_ov, perm_occ in coupling_info[idx]:
                T_nbr = T_snapshot[nbr_idx]
                if T_nbr is None:
                    continue
                T_perm = _triples_permuter(T_nbr, *perm_occ)
                T_proj = _project_t3(T_perm, S_ov)
                R += f_coupling * T_proj

            td['T'] = T_old - R / D
            r_max_list[idx] = float(np.sqrt(np.mean(R**2)))

        # Compute energy
        e_ijk_old = list(e_ijk_list)
        e_curr, e_ijk_list = _compute_t1_energy(triple_data, triple_idx_map, F_lmo, nocc)

        r_max = max(r_max_list)
        delta_e = e_curr - e_prev
        _dt_iter = _time.perf_counter() - _t_iter

        log.info('  %5d %18.12f %12.3e %12.3e %8.1f',
                 iteration, e_curr, delta_e, r_max, _dt_iter)

        e_converged = abs(delta_e) < e_conv
        r_converged = abs(r_max) < r_conv

        if e_converged and r_converged:
            log.info('')
            log.info('  (T1) converged in %d iterations.', iteration)
            break

        e_prev = e_curr
    else:
        log.warn('  (T1) NOT converged after %d iterations!', max_iter)

    e_t = e_curr
    log.info('E(T1) = %.15g', e_t)
    return e_t
