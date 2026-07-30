/* DLPNO-(T): per-pair q_vv build (Psi4-style restructure).
 *
 * Replaces the full `vvL_sc` (n_tno, n_tno, naux_ijk) tensor with three
 * thinner per-pair slices:
 *
 *   q_vv_ij[a_tno, b_pno_ij, q]   shape (n_tno, n_pno_ij, naux_ijk)
 *   q_vv_jk[a_tno, b_pno_jk, q]   shape (n_tno, n_pno_jk, naux_ijk)
 *   q_vv_ik[a_tno, b_pno_ik, q]   shape (n_tno, n_pno_ik, naux_ijk)
 *
 * Math (mirrors Psi4 triples.cc:723-754):
 *
 *   q_vv_pair[a, b, q] = Σ_{u, v} X_tno[u_pao_ijk, a]
 *                                * qab_PAO[q, u_pao_ijk, v_pao_pair]
 *                                * X_pno_pair[v_pao_pair, b]
 *
 * Per Q (per pair):
 *   1. tmp[u_pao_ijk, b] = qab_PAO_uv @ X_pno_pair      cost n_pao_ijk × n_pao_pair × n_pno_pair
 *   2. q_vv[a, b]        = X_tno.T @ tmp                cost n_tno × n_pao_ijk × n_pno_pair
 *
 * Total per Q per pair: O(n_pao_ijk × n_pao_pair × n_pno_pair) — LINEAR in n_pao_ijk
 * Versus the old vvL build: O(n_pao_ijk × n_pao_ijk × n_tno) per Q (squared in n_pao_ijk).
 *
 * This is the source of (T) scaling reduction from N^2.56 → ~N^2.2.
 *
 * SCAFFOLD STATUS: This is the FIRST atomic unit of the restructure.
 * Builds ONE pair's q_vv from per-triple sparse-DF data + X_pno_pair.
 * Validation: caller can compare q_vv_pair against `X_pno_pair.T @ vvL_sc[a,:,q] @ X_tno_to_pno_pair_inv`
 * for bit-equivalence (within FP noise).
 *
 * Inputs:
 *   n_tno                    — triple's TNO count
 *   n_pao_ijk                — triple's PAO domain size
 *   n_pao_pair               — pair's PAO domain size
 *   n_pno_pair               — pair's PNO count
 *   naux_ijk                 — triple's local aux dim
 *   n_centers                — # of aux atoms relevant to triple
 *   triple_paos              — (n_pao_ijk,) global PAO indices
 *   pair_paos                — (n_pao_pair,) global PAO indices for the pair
 *   X_tno_ijk                — (n_pao_ijk, n_tno) row-major
 *   X_pno_pair               — (n_pao_pair, n_pno_pair) row-major
 *   center_atoms             — (n_centers,) aux atoms
 *   center_off                — (n_centers+1,) offsets in local_Q/atom_pos
 *   local_Q_flat             — positions in naux_ijk
 *   atom_pos_flat            — page in atom stack
 *   qab_atom_off             — (natm+1,) offsets in qab_atom_flat
 *   qab_atom_n_pao           — (natm,) np_A per atom
 *   qab_atom_flat            — qab data
 *   riatom_to_paos_ext_dense — (natm, n_pao_global) — -1 if PAO absent
 *   n_pao_global             — full PAO count (for stride)
 *   jhi                      — (naux_ijk, naux_ijk) local J^{-1/2}
 *
 * Output:
 *   q_vv_pair_sc             — (n_tno, n_pno_pair, naux_ijk) row-major
 *
 * NOT YET IMPLEMENTED — scaffold only. See HANDOFF_TRIPLES_VVL_PER_PAIR.md.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include "vhf/fblas.h"

/* ------------------------------------------------------------------ *
 * (pair, RI-atom) H-cache for the q_vv build.
 *
 * H[pair, A] [page, p, b] = sum_v qab_A[page][p, v_in_A] * X_pno_pair[v, b]
 *
 * dgemm1 of the per-triple build depends only on (pair, atom): the same
 * product is recomputed for every triple containing the pair (~33x reuse
 * at MOBH35-12/SVP).  Entries are built lazily by the first triple worker
 * that touches a slot and consumed read-only afterwards.  A worker that
 * finds a slot mid-build falls back to the uncached path (no blocking).
 * Rows cover ALL np_A stack PAOs so any triple's row subset can gather.
 * ------------------------------------------------------------------ */
typedef struct { double *H; int state; } QvvCacheSlot;
/* state: 0 = empty, 1 = building, 2 = ready, 3 = skip (over budget) */

static struct {
    QvvCacheSlot *slots;          /* n_pairs * natm */
    long n_pairs, natm;
    size_t bytes_used, bytes_budget;
    long n_built, n_fallback, n_skip;
    long long hits;
    int enabled;
    pthread_mutex_t mu;
} _qvvc = { NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0, PTHREAD_MUTEX_INITIALIZER };

void DLPNOqvv_cache_init(long n_pairs, long natm, double budget_gb)
{
    pthread_mutex_lock(&_qvvc.mu);
    if (_qvvc.slots) {          /* stale from a previous run: drop */
        for (long s = 0; s < _qvvc.n_pairs * _qvvc.natm; s++) {
            free(_qvvc.slots[s].H);
        }
        free(_qvvc.slots);
    }
    _qvvc.slots = (QvvCacheSlot *)calloc((size_t)n_pairs * (size_t)natm,
                                         sizeof(QvvCacheSlot));
    _qvvc.n_pairs = n_pairs;
    _qvvc.natm = natm;
    _qvvc.bytes_used = 0;
    _qvvc.bytes_budget = (size_t)(budget_gb * 1073741824.0);
    _qvvc.n_built = 0; _qvvc.n_fallback = 0; _qvvc.n_skip = 0;
    _qvvc.hits = 0;
    _qvvc.enabled = (_qvvc.slots != NULL);
    pthread_mutex_unlock(&_qvvc.mu);
}

void DLPNOqvv_cache_free(void)
{
    pthread_mutex_lock(&_qvvc.mu);
    if (_qvvc.slots) {
        for (long s = 0; s < _qvvc.n_pairs * _qvvc.natm; s++) {
            free(_qvvc.slots[s].H);
        }
        free(_qvvc.slots);
    }
    _qvvc.slots = NULL;
    _qvvc.n_pairs = 0; _qvvc.natm = 0;
    _qvvc.enabled = 0;
    pthread_mutex_unlock(&_qvvc.mu);
}

void DLPNOqvv_cache_stats(double *out5)
{
    pthread_mutex_lock(&_qvvc.mu);
    out5[0] = (double)_qvvc.n_built;
    out5[1] = (double)_qvvc.hits;
    out5[2] = (double)_qvvc.n_fallback;
    out5[3] = (double)_qvvc.n_skip;
    out5[4] = (double)_qvvc.bytes_used;
    pthread_mutex_unlock(&_qvvc.mu);
}

void DLPNObuild_triple_qvv_pair(
        const int     n_tno,
        const int     n_pao_ijk,
        const int     n_pao_pair,
        const int     n_pno_pair,
        const int     naux_ijk,
        const int     n_centers,
        const int     n_pao_global,
        const long   *triple_paos,
        const long   *pair_paos,
        const double *X_tno_ijk,
        const double *X_pno_pair,
        const long   *center_atoms,
        const long   *center_off,
        const long   *local_Q_flat,
        const long   *atom_pos_flat,
        const long   *qab_atom_off,
        const int    *qab_atom_n_pao,
        const double *qab_atom_flat,
        const long   *riatom_to_paos_ext_dense,
        const double *jhi,
        const long    pair_id,
        double       *q_vv_pair_sc)
{
    if (naux_ijk <= 0 || n_tno <= 0 || n_pno_pair <= 0) return;
    const int use_cache = (_qvvc.enabled && pair_id >= 0
                           && pair_id < _qvvc.n_pairs);

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;

    /* Raw (pre-jhi) output, scattered per Q. */
    const size_t raw_sz = (size_t)n_tno * (size_t)n_pno_pair * (size_t)naux_ijk;
    /* Asymmetric-fit mode (jhi == NULL): the caller applies the FULL
     * J^-1 on the ov side instead (exact by associativity: the q_vv
     * tensor pairs only with ovL in K_ovvv).  Raw values then go
     * straight into q_vv_pair_sc — no metric dgemm on the LARGE vv side
     * (was ~84% of the whole (T) kernel).  jhi != NULL = legacy B-form. */
    double *q_vv_raw = (jhi == NULL) ? q_vv_pair_sc
        : (double *)malloc(sizeof(double)
                           * (size_t)n_tno * (size_t)n_pno_pair * naux_ijk);

    /* Scratch — sized to upper bounds. */
    const size_t max_nl = (size_t)n_pao_ijk;       /* triple-PAO at any center */
    const size_t max_nr = (size_t)n_pao_pair;      /* pair-PAO at any center */
    /* qab_cut: nl × nr per Q */
    double *qab_cut_buf = (double *)malloc(
        sizeof(double) * (max_nl * max_nr > 0 ? max_nl * max_nr : 1));
    /* tmp = qab_cut @ X_pno_pair_local: nl × n_pno_pair */
    double *tmp_buf = (double *)malloc(
        sizeof(double) * (max_nl * (size_t)n_pno_pair > 0
                          ? max_nl * (size_t)n_pno_pair : 1));
    /* q_vv_block: n_tno × n_pno_pair */
    double *qvv_blk = (double *)malloc(
        sizeof(double) * (size_t)n_tno * (size_t)n_pno_pair);
    /* per-center stack of qvv blocks: (nQc, n_tno*n_pno_pair), written
     * contiguously per Q, then transposed once per center into the aux
     * axis (strided-READ / contiguous-WRITE beats the per-element
     * strided scatter that was the top cycle consumer under perf). */
    size_t qvv_stack_cap = 0;
    double *qvv_stack = NULL;
    /* X_tno_local: nl × n_tno (gather) */
    double *X_tno_local = (double *)malloc(
        sizeof(double) * (max_nl * (size_t)n_tno > 0
                          ? max_nl * (size_t)n_tno : 1));
    /* X_pno_local: nr × n_pno_pair (gather) */
    double *X_pno_local = (double *)malloc(
        sizeof(double) * (max_nr * (size_t)n_pno_pair > 0
                          ? max_nr * (size_t)n_pno_pair : 1));

    long *valid_l_tp = (long *)malloc(sizeof(long) * (max_nl + 1));
    long *valid_r_pp = (long *)malloc(sizeof(long) * (max_nr + 1));
    /* run-length encoding of valid_r_pp (page PAO positions come in
     * per-atom contiguous stretches): gather becomes run-wise memcpy. */
    long *rr_v0 = (long *)malloc(sizeof(long) * (max_nr + 1));
    long *rr_s0 = (long *)malloc(sizeof(long) * (max_nr + 1));
    long *rr_ln = (long *)malloc(sizeof(long) * (max_nr + 1));

    for (int c = 0; c < n_centers; c++) {
        const long centerQ = center_atoms[c];
        const long c_beg = center_off[c];
        const long c_end = center_off[c + 1];
        const size_t nQc = (size_t)(c_end - c_beg);
        if (nQc == 0) continue;

        const long *local_Q  = local_Q_flat  + c_beg;
        const long *atom_pos = atom_pos_flat + c_beg;

        const int np_A = qab_atom_n_pao[centerQ];
        if (np_A == 0) continue;

        const long *paos_dense_row = riatom_to_paos_ext_dense
                                   + (size_t)centerQ * (size_t)n_pao_global;

        /* Visible triple-PAOs at this center (left axis). Each element is
         * an index into the local atom's PAO stack (np_A range). */
        int nl = 0;
        for (int u = 0; u < n_pao_ijk; u++) {
            const long u_in_A = paos_dense_row[triple_paos[u]];
            if (u_in_A >= 0) {
                /* X_tno_local[nl, t] = X_tno_ijk[u, t]   (gather rows of X_tno) */
                memcpy(X_tno_local + (size_t)nl * (size_t)n_tno,
                       X_tno_ijk + (size_t)u * (size_t)n_tno,
                       sizeof(double) * (size_t)n_tno);
                valid_l_tp[nl] = u_in_A;
                nl++;
            }
        }
        if (nl == 0) continue;

        /* Visible pair-PAOs at this center (right axis). */
        int nr = 0;
        for (int v = 0; v < n_pao_pair; v++) {
            const long v_in_A = paos_dense_row[pair_paos[v]];
            if (v_in_A >= 0) {
                memcpy(X_pno_local + (size_t)nr * (size_t)n_pno_pair,
                       X_pno_pair + (size_t)v * (size_t)n_pno_pair,
                       sizeof(double) * (size_t)n_pno_pair);
                valid_r_pp[nr] = v_in_A;
                nr++;
            }
        }
        if (nr == 0) continue;

        int n_rr = 0;
        for (int v = 0; v < nr; ) {
            int v0 = v;
            long s0 = valid_r_pp[v];
            v++;
            while (v < nr && valid_r_pp[v] == s0 + (v - v0)) v++;
            rr_v0[n_rr] = v0; rr_s0[n_rr] = s0; rr_ln[n_rr] = v - v0;
            n_rr++;
        }

        const long qab_off_A = qab_atom_off[centerQ];
        const double *qab_A = qab_atom_flat + qab_off_A;
        const size_t qab_pg = (size_t)np_A * (size_t)np_A;   /* per-Q stride */

        /* ---- (pair, atom) H-cache: skip dgemm1 + qab gather when hot ---- */
        const double *Hc = NULL;
        if (use_cache) {
            QvvCacheSlot *slot = _qvvc.slots
                + (size_t)pair_id * (size_t)_qvvc.natm + (size_t)centerQ;
            int st = __atomic_load_n(&slot->state, __ATOMIC_ACQUIRE);
            if (st == 2) {
                Hc = slot->H;
                __atomic_fetch_add(&_qvvc.hits, 1, __ATOMIC_RELAXED);
            } else if (st == 0) {
                int claimed = 0;
                pthread_mutex_lock(&_qvvc.mu);
                if (slot->state == 0) { slot->state = 1; claimed = 1; }
                pthread_mutex_unlock(&_qvvc.mu);
                if (claimed) {
                    const long n_pages =
                        (qab_atom_off[centerQ + 1] - qab_off_A) / (long)qab_pg;
                    const size_t h_row = (size_t)np_A * (size_t)n_pno_pair;
                    const size_t h_bytes =
                        (size_t)n_pages * h_row * sizeof(double);
                    int over = (n_pages <= 0);
                    if (!over) {
                        pthread_mutex_lock(&_qvvc.mu);
                        if (_qvvc.bytes_used + h_bytes > _qvvc.bytes_budget) {
                            over = 1;
                        } else {
                            _qvvc.bytes_used += h_bytes;
                        }
                        pthread_mutex_unlock(&_qvvc.mu);
                    }
                    double *H = over ? NULL : (double *)malloc(h_bytes);
                    double *gat = over ? NULL : (double *)malloc(
                        sizeof(double) * (size_t)np_A * (size_t)nr);
                    if (H == NULL || gat == NULL) {
                        free(H); free(gat);
                        if (!over) {
                            pthread_mutex_lock(&_qvvc.mu);
                            _qvvc.bytes_used -= h_bytes;
                            pthread_mutex_unlock(&_qvvc.mu);
                        }
                        __atomic_fetch_add(&_qvvc.n_skip, 1, __ATOMIC_RELAXED);
                        __atomic_store_n(&slot->state, 3, __ATOMIC_RELEASE);
                    } else {
                        int int_np_A = np_A;
                        int int_nr_b = nr;
                        int int_npno_b = n_pno_pair;
                        for (long pg2 = 0; pg2 < n_pages; pg2++) {
                            const double *src_pg = qab_A + (size_t)pg2 * qab_pg;
                            for (int u = 0; u < np_A; u++) {
                                const double *srow = src_pg
                                    + (size_t)u * (size_t)np_A;
                                double *drow = gat + (size_t)u * (size_t)nr;
                                for (int r = 0; r < n_rr; r++) {
                                    memcpy(drow + rr_v0[r], srow + rr_s0[r],
                                           sizeof(double) * rr_ln[r]);
                                }
                            }
                            dgemm_(&N_flag, &N_flag,
                                   &int_npno_b, &int_np_A, &int_nr_b,
                                   &one, X_pno_local, &int_npno_b,
                                   gat, &int_nr_b,
                                   &zero, H + (size_t)pg2 * h_row,
                                   &int_npno_b);
                        }
                        free(gat);
                        pthread_mutex_lock(&_qvvc.mu);
                        slot->H = H;
                        _qvvc.n_built++;
                        pthread_mutex_unlock(&_qvvc.mu);
                        __atomic_store_n(&slot->state, 2, __ATOMIC_RELEASE);
                        Hc = H;
                    }
                } else {
                    __atomic_fetch_add(&_qvvc.n_fallback, 1, __ATOMIC_RELAXED);
                }
            } else if (st == 1) {
                __atomic_fetch_add(&_qvvc.n_fallback, 1, __ATOMIC_RELAXED);
            }
        }

        /* Per-Q work: gather qab_cut (or cached H rows), dgemms, scatter. */
        int int_nl = nl;
        int int_nr = nr;
        int int_n_tno = n_tno;
        int int_n_pno_pair = n_pno_pair;

        const size_t NP0 = (size_t)n_tno * (size_t)n_pno_pair;
        if (qvv_stack_cap < nQc * NP0) {
            free(qvv_stack);
            qvv_stack = (double *)malloc(sizeof(double) * nQc * NP0);
            qvv_stack_cap = nQc * NP0;
        }
        int lq_contig = 1;
        for (size_t q = 1; q < nQc; q++) {
            if (local_Q[q] != local_Q[0] + (long)q) { lq_contig = 0; break; }
        }

        for (size_t q = 0; q < nQc; q++) {
            const size_t pg = (size_t)atom_pos[q];

            if (Hc != NULL) {
                /* tmp[u, b] = H[pg, valid_l[u], b] — row gather, no dgemm1 */
                const double *Hpage = Hc
                    + pg * (size_t)np_A * (size_t)n_pno_pair;
                for (int u = 0; u < nl; u++) {
                    memcpy(tmp_buf + (size_t)u * (size_t)n_pno_pair,
                           Hpage + (size_t)valid_l_tp[u] * (size_t)n_pno_pair,
                           sizeof(double) * (size_t)n_pno_pair);
                }
            } else {
            const double *src = qab_A + pg * qab_pg;

            /* qab_cut[u_local, v_local] = qab_A[atom_pos[q], valid_l[u], valid_r[v]] */
            for (int u = 0; u < nl; u++) {
                const long uA = valid_l_tp[u];
                const double *src_row = src + (size_t)uA * (size_t)np_A;
                double *dst_row = qab_cut_buf + (size_t)u * (size_t)nr;
                for (int r = 0; r < n_rr; r++) {
                    memcpy(dst_row + rr_v0[r], src_row + rr_s0[r],
                           sizeof(double) * rr_ln[r]);
                }
            }

            /* tmp[nl, n_pno_pair] = qab_cut[nl, nr] @ X_pno_local[nr, n_pno_pair]
             *   row-major C[m, n] = A[m, k] @ B[k, n]
             *   col-major dgemm: dgemm('N','N', n, m, k, B, n, A, k, 0, C, n)
             */
            dgemm_(&N_flag, &N_flag,
                   &int_n_pno_pair, &int_nl, &int_nr,
                   &one, X_pno_local, &int_n_pno_pair,
                   qab_cut_buf, &int_nr,
                   &zero, tmp_buf, &int_n_pno_pair);
            }

            /* qvv_blk[n_tno, n_pno_pair] = X_tno_local[nl, n_tno].T @ tmp[nl, n_pno_pair]
             *   row-major C[m, n] = A[k, m]^T @ B[k, n]
             *   dgemm('N', 'T', n, m, k, B, n, A, m, 0, C, n)
             */
            dgemm_(&N_flag, &T_flag,
                   &int_n_pno_pair, &int_n_tno, &int_nl,
                   &one, tmp_buf, &int_n_pno_pair,
                   X_tno_local, &int_n_tno,
                   &zero, qvv_stack + q * NP0, &int_n_pno_pair);

            /* Scatter into q_vv_raw[a, b, local_Q[q]]:
             *   row-major (n_tno, n_pno_pair, naux_ijk).
             *   q_vv_raw[a*n_pno_pair*naux + b*naux + lq] = qvv_blk[a*n_pno_pair + b]
             */
        }

        /* transpose-scatter: q_vv_raw[ab, lq] = qvv_stack[q, ab] */
        if (lq_contig) {
            const long lq0 = local_Q[0];
            for (size_t ab = 0; ab < NP0; ab++) {
                double *dst = q_vv_raw + ab * (size_t)naux_ijk + lq0;
                const double *srcc = qvv_stack + ab;
                for (size_t q = 0; q < nQc; q++) {
                    dst[q] = srcc[q * NP0];
                }
            }
        } else {
            for (size_t q = 0; q < nQc; q++) {
                const long lq = local_Q[q];
                const double *blk = qvv_stack + q * NP0;
                for (size_t ab = 0; ab < NP0; ab++) {
                    q_vv_raw[ab * (size_t)naux_ijk + lq] = blk[ab];
                }
            }
        }
    }

    free(qab_cut_buf);
    free(tmp_buf);
    free(qvv_blk);
    free(qvv_stack);
    free(X_tno_local);
    free(X_pno_local);
    free(valid_l_tp);
    free(valid_r_pp);
    free(rr_v0); free(rr_s0); free(rr_ln);

    /* Apply jhi: q_vv_pair_sc (n_tno*n_pno_pair, naux_ijk) =
     *   q_vv_raw (n_tno*n_pno_pair, naux_ijk) @ jhi (naux_ijk, naux_ijk)
     * One dgemm.
     */
    if (jhi != NULL) {
        int int_naux = naux_ijk;
        int int_rows = n_tno * n_pno_pair;
        if (int_rows > 0) {
            dgemm_(&N_flag, &N_flag,
                   &int_naux, &int_rows, &int_naux,
                   &one, jhi, &int_naux,
                   q_vv_raw, &int_naux,
                   &zero, q_vv_pair_sc, &int_naux);
        }
        free(q_vv_raw);
    }
}
