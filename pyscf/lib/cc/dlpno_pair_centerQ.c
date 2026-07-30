/* DLPNO-CCSD compute_cc_integrals_sparse: per-pair-per-centerQ inner kernel.
 *
 * BLAS port (2026-04-30): pre-gather fancy-indexed inputs to contiguous
 * tensors, then DGEMM the matmuls.  At MKL-link with JIT-GEMM, ~3-4x
 * faster than the previous fancy-index hand-rolled loops on water-10.
 *
 * Math (per pair, per centerQ; matches local_df.py:736-823 line-by-line):
 *
 *   raw_io[local_Q[q], ext_kept_lmos[k]] = qij_b[q, i_s, ext_kept_pos[k]]
 *   raw_jo[local_Q[q], ext_kept_lmos[k]] = qij_b[q, j_s, ext_kept_pos[k]]
 *   raw_pair[local_Q[q]]                 = qij_b[q, i_s, j_s]
 *
 *   raw_iv[local_Q[q], a] = sum_u qia_b[q, i_s, ij_u_in_Q[u]] * X[u, a]
 *   raw_jv[local_Q[q], a] = sum_u qia_b[q, j_s, ij_u_in_Q[u]] * X[u, a]
 *
 *   raw_ma[local_Q[q], ext_kept_lmos[k], a]
 *       = sum_u qia_b[q, ext_kept_pos[k], ij_u_in_Q[u]] * X[u, a]
 *
 *   raw_ab[local_Q[q], a, b]
 *       = sum_{u, v} X[u, a] * qab_b[q, ij_u_in_Q[u], ij_u_in_Q[v]] * X[v, b]
 *
 *   proj_ij_out[q, a, v]
 *       = sum_u X[u, a] * qab_b[q, ij_u_in_Q[u], v]
 *
 * Strategy:
 *   1. For raw_iv/jv/ma: pre-gather qia[lmo, ij_u_in_Q[*]] for the relevant
 *      LMO row → contiguous (nQp, npp_ij), then DGEMM with X.
 *   2. For raw_ab: pre-gather qab[ij_u_in_Q[*], ij_u_in_Q[*]] →
 *      contiguous (nQp, npp_ij, npp_ij); DGEMM with X twice.
 *   3. For proj_ij_out: pre-gather qab[ij_u_in_Q[*], :] → (nQp, npp_ij, np_full);
 *      DGEMM with X.
 *
 * Shapes: see header above.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"
#include <time.h>

/* In-kernel step attribution (DLPNO_CENTERQ_PROF=1): nanosecond
 * accumulators summed across all pool threads via relaxed atomics.
 * Slots: 0=s1 scatter, 1=s2 iv/jv, 2=s3 ma, 3=s5 gather, 4=s5 dgemm,
 * 5=s4 raw_ab.  Read+reset from Python via DLPNOcenterQ_prof_get. */
static long _cq_prof_ns[8];
static int _cq_prof_on = -1;

static inline double _cq_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static inline void _cq_add(int slot, double t0)
{
    if (_cq_prof_on > 0) {
        long dns = (long)((_cq_now() - t0) * 1e9);
        __atomic_fetch_add(&_cq_prof_ns[slot], dns, __ATOMIC_RELAXED);
    }
}

void DLPNOcenterQ_prof_get(double *out8)
{
    for (int i = 0; i < 8; i++) {
        out8[i] = _cq_prof_ns[i] * 1e-9;
        _cq_prof_ns[i] = 0;
    }
}

static int _dlpno_cmp_long(const void *a, const void *b) {
    long la = *(const long *)a;
    long lb = *(const long *)b;
    return (la > lb) - (la < lb);
}

/* Per-(pair, centerQ) helper to build pair_used_in_Q (sorted positions
 * in [0, np_full) for the global PAOs in pair_used_pao_global) and
 * pair_used_inv (np_full → red index, -1 elsewhere). Releases the GIL
 * via ctypes; replaces the per-centerQ Python loop, which was ~22 s
 * stalled on the GIL on water-42. */
long DLPNOcompute_pair_used(
        const long *riatom_to_paos_dense_at,
        const long *pair_used_pao_global,
        const size_t n_pair_used,
        const size_t np_full,
        long *pair_used_in_Q,
        long *pair_used_inv)
{
    for (size_t i = 0; i < np_full; i++) pair_used_inv[i] = -1;

    long n_red = 0;
    for (size_t u = 0; u < n_pair_used; u++) {
        const long pao_global = pair_used_pao_global[u];
        const long pos = riatom_to_paos_dense_at[pao_global];
        if (pos >= 0) {
            pair_used_in_Q[n_red++] = pos;
        }
    }

    if (n_red > 1)
        qsort(pair_used_in_Q, (size_t)n_red, sizeof(long), _dlpno_cmp_long);

    for (long i = 0; i < n_red; i++) {
        pair_used_inv[pair_used_in_Q[i]] = i;
    }

    return n_red;
}

void DLPNOpair_centerQ_step(
        const double *qij_atom_full,   /* (nQ_at_atom, nl, nl) */
        const double *qia_atom_full,   /* (nQ_at_atom, nl, np_full) */
        const double *qab_atom_full,   /* (nQ_at_atom, np_full, np_full) */
        const long   *local_Q,
        const long   *atom_pos,        /* (nQp,) — page index into atom stack */
        const int     i_s,
        const int     j_s,
        const long   *ij_u_in_Q,
        const long   *ext_kept_pos,
        const long   *ext_kept_lmos,
        const double *X_ij_slice,
        const long   *pair_used_in_Q,  /* (n_red,) PAO positions used by pair */
        const size_t  nQp,
        const size_t  nl,
        const size_t  np_full,
        const size_t  npno,
        const size_t  npp_ij,
        const size_t  n_kept,
        const size_t  n_local,
        const size_t  nlmo_p,
        const size_t  n_red,           /* reduced proj_ij column count */
        double       *raw_io,
        double       *raw_jo,
        double       *raw_iv,
        double       *raw_jv,
        double       *raw_pair,
        double       *raw_ma,
        double       *raw_ab,
        double       *proj_ij_out)
{
    const double *qij_b = qij_atom_full;
    const double *qia_b = qia_atom_full;
    const double *qab_b = qab_atom_full;

    const size_t qij_q = nl * nl;
    const size_t qij_l = nl;
    const size_t qia_q = nl * np_full;
    const size_t qia_l = np_full;
    const size_t qab_q = np_full * np_full;
    const size_t qab_u = np_full;
    const size_t io_row = nlmo_p;
    const size_t iv_row = npno;
    const size_t ma_row = nlmo_p * npno;
    const size_t ma_lmo = npno;
    const size_t ab_row = npno * npno;
    const size_t X_row  = npno;
    /* proj_ij_out is (nQp, npno, n_red): only PAOs needed by some
     * partner are kept. n_red ≤ np_full; with n_red=np_full and
     * pair_used_in_Q=identity, this matches the legacy behavior. */
    const size_t proj_q = npno * n_red;

    const int has_i = (i_s >= 0);
    const int has_j = (j_s >= 0);
    const int has_pair_paos = (npp_ij > 0);

    if (_cq_prof_on < 0) {
        const char *_e = getenv("DLPNO_CENTERQ_PROF");
        _cq_prof_on = (_e != NULL && _e[0] == '1') ? 1 : 0;
    }
    double _t_sec = _cq_prof_on ? _cq_now() : 0.0;

    /* Run-encode ij_u_in_Q once (pair PAO page positions come in
     * per-atom contiguous stretches): steps 2/3 gathers become run-wise
     * memcpy instead of per-element loads. */
    long *ru_u0 = NULL, *ru_s0 = NULL, *ru_ln = NULL;
    int n_ru = 0;
    if (has_pair_paos) {
        ru_u0 = (long *)malloc(sizeof(long) * npp_ij);
        ru_s0 = (long *)malloc(sizeof(long) * npp_ij);
        ru_ln = (long *)malloc(sizeof(long) * npp_ij);
        for (long u = 0; u < (long)npp_ij; ) {
            long u0 = u, s0 = ij_u_in_Q[u];
            u++;
            while (u < (long)npp_ij && ij_u_in_Q[u] == s0 + (u - u0)) u++;
            ru_u0[n_ru] = u0; ru_s0[n_ru] = s0; ru_ln[n_ru] = u - u0;
            n_ru++;
        }
    }

    /* ------------------------------------------------------------------
     * Step 1: raw_io / raw_jo / raw_pair — pure scatter, no BLAS.
     * ------------------------------------------------------------------ */
    if (n_kept > 0 && has_i) {
        for (size_t q = 0; q < nQp; q++) {
            const size_t row_lq = (size_t)local_Q[q];
            const size_t pg = (size_t)atom_pos[q];
            const double *qij_qi = qij_b + pg * qij_q + (size_t)i_s * qij_l;
            double *out_row = raw_io + row_lq * io_row;
            for (size_t k = 0; k < n_kept; k++) {
                out_row[ext_kept_lmos[k]] = qij_qi[ext_kept_pos[k]];
            }
        }
    }
    if (n_kept > 0 && has_j) {
        for (size_t q = 0; q < nQp; q++) {
            const size_t row_lq = (size_t)local_Q[q];
            const size_t pg = (size_t)atom_pos[q];
            const double *qij_qj = qij_b + pg * qij_q + (size_t)j_s * qij_l;
            double *out_row = raw_jo + row_lq * io_row;
            for (size_t k = 0; k < n_kept; k++) {
                out_row[ext_kept_lmos[k]] = qij_qj[ext_kept_pos[k]];
            }
        }
    }
    if (has_i && has_j) {
        for (size_t q = 0; q < nQp; q++) {
            const size_t row_lq = (size_t)local_Q[q];
            const size_t pg = (size_t)atom_pos[q];
            raw_pair[row_lq] = qij_b[pg * qij_q + (size_t)i_s * qij_l + (size_t)j_s];
        }
    }

    _cq_add(0, _t_sec);
    if (!has_pair_paos) return;

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;
    int int_npno = (int)npno, int_npp_ij = (int)npp_ij;
    int int_nQp = (int)nQp;

    /* ------------------------------------------------------------------
     * Step 2: raw_iv / raw_jv via DGEMM after row-gather.
     *
     * For each Q, gather qia[i_s, ij_u_in_Q[*]] to size npp_ij (contig).
     * Stack across Q's: qia_i_stack[Q, u] = qia[atom_pos[Q]][i_s][ij_u_in_Q[u]].
     * Then raw_iv_local = qia_i_stack @ X_ij_slice (nQp, npp_ij) @ (npp_ij, npno).
     * ------------------------------------------------------------------ */
    _t_sec = _cq_prof_on ? _cq_now() : 0.0;
    if (has_i || has_j) {
        const size_t stack_sz = nQp * npp_ij;
        double *qia_i_stack = (has_i) ? (double *)malloc(sizeof(double) * stack_sz) : NULL;
        double *qia_j_stack = (has_j) ? (double *)malloc(sizeof(double) * stack_sz) : NULL;
        double *iv_local    = (has_i) ? (double *)malloc(sizeof(double) * nQp * npno) : NULL;
        double *jv_local    = (has_j) ? (double *)malloc(sizeof(double) * nQp * npno) : NULL;

        for (size_t q = 0; q < nQp; q++) {
            const size_t pg = (size_t)atom_pos[q];
            const double *qia_q_ptr = qia_b + pg * qia_q;
            if (has_i) {
                const double *qia_qi = qia_q_ptr + (size_t)i_s * qia_l;
                double *out = qia_i_stack + q * npp_ij;
                for (int r = 0; r < n_ru; r++) {
                    memcpy(out + ru_u0[r], qia_qi + ru_s0[r],
                           sizeof(double) * ru_ln[r]);
                }
            }
            if (has_j) {
                const double *qia_qj = qia_q_ptr + (size_t)j_s * qia_l;
                double *out = qia_j_stack + q * npp_ij;
                for (int r = 0; r < n_ru; r++) {
                    memcpy(out + ru_u0[r], qia_qj + ru_s0[r],
                           sizeof(double) * ru_ln[r]);
                }
            }
        }

        /* iv_local[Q, a] = sum_u qia_i_stack[Q, u] * X[u, a]
         * = qia_i_stack (nQp, npp_ij) @ X_ij_slice (npp_ij, npno)
         * F: iv_F[a, Q] = sum_u X_F[a, u] * qia_F[u, Q] = X_F @ qia_F
         * dgemm('N', 'N', npno, nQp, npp_ij, 1, X, npno, qia, npp_ij, 0, iv, npno)
         */
        if (has_i) {
            dgemm_(&N_flag, &N_flag,
                   &int_npno, &int_nQp, &int_npp_ij,
                   &one, X_ij_slice, &int_npno,
                   qia_i_stack, &int_npp_ij,
                   &zero, iv_local, &int_npno);
            for (size_t q = 0; q < nQp; q++) {
                memcpy(raw_iv + (size_t)local_Q[q] * iv_row,
                       iv_local + q * npno,
                       sizeof(double) * npno);
            }
        }
        if (has_j) {
            dgemm_(&N_flag, &N_flag,
                   &int_npno, &int_nQp, &int_npp_ij,
                   &one, X_ij_slice, &int_npno,
                   qia_j_stack, &int_npp_ij,
                   &zero, jv_local, &int_npno);
            for (size_t q = 0; q < nQp; q++) {
                memcpy(raw_jv + (size_t)local_Q[q] * iv_row,
                       jv_local + q * npno,
                       sizeof(double) * npno);
            }
        }
        if (qia_i_stack) free(qia_i_stack);
        if (qia_j_stack) free(qia_j_stack);
        _cq_add(1, _t_sec);
        if (iv_local) free(iv_local);
        if (jv_local) free(jv_local);
    }

    /* ------------------------------------------------------------------
     * Step 3: raw_ma — for each kept LMO row, build (nQp, npp_ij) gather +
     * DGEMM with X.  All n_kept rows share the same X, so we can stack
     * the LMO axis: qia_k_stack[Q, k, u] then ONE DGEMM gives result
     * (Q, k, a) which is scattered into raw_ma.
     * ------------------------------------------------------------------ */
    _t_sec = _cq_prof_on ? _cq_now() : 0.0;
    if (n_kept > 0) {
        const size_t stack_sz = nQp * n_kept * npp_ij;
        double *qia_k_stack = (double *)malloc(sizeof(double) * stack_sz);
        double *ma_local    = (double *)malloc(sizeof(double) * nQp * n_kept * npno);

        for (size_t q = 0; q < nQp; q++) {
            const size_t pg = (size_t)atom_pos[q];
            const double *qia_q_ptr = qia_b + pg * qia_q;
            for (size_t k = 0; k < n_kept; k++) {
                const long lmo_pos = ext_kept_pos[k];
                const double *qia_qk = qia_q_ptr + (size_t)lmo_pos * qia_l;
                double *out = qia_k_stack + (q * n_kept + k) * npp_ij;
                for (int r = 0; r < n_ru; r++) {
                    memcpy(out + ru_u0[r], qia_qk + ru_s0[r],
                           sizeof(double) * ru_ln[r]);
                }
            }
        }

        /* ma_local[(Q*n_kept + k), a] = sum_u qia_k_stack[(Q*n_kept + k), u] * X[u, a]
         * Big DGEMM: (nQp*n_kept, npp_ij) @ (npp_ij, npno) → (nQp*n_kept, npno).
         */
        int int_M = (int)(nQp * n_kept);
        dgemm_(&N_flag, &N_flag,
               &int_npno, &int_M, &int_npp_ij,
               &one, X_ij_slice, &int_npno,
               qia_k_stack, &int_npp_ij,
               &zero, ma_local, &int_npno);

        /* Scatter ma_local[Q, k, a] → raw_ma[local_Q[Q], ext_kept_lmos[k], a] */
        for (size_t q = 0; q < nQp; q++) {
            const size_t row_lq = (size_t)local_Q[q];
            double *ma_out_row = raw_ma + row_lq * ma_row;
            const double *ma_local_q = ma_local + q * n_kept * npno;
            for (size_t k = 0; k < n_kept; k++) {
                memcpy(ma_out_row + (size_t)ext_kept_lmos[k] * ma_lmo,
                       ma_local_q + k * npno,
                       sizeof(double) * npno);
            }
        }

        free(qia_k_stack);
        free(ma_local);
        _cq_add(2, _t_sec);
    }

    /* ------------------------------------------------------------------
     * Step 5: proj_ij_out[q, a, v_red]
     *     = sum_u X[u, a] * qab[q, ij_u_in_Q[u], pair_used_in_Q[v_red]]
     * FULL-ROW path (DLPNO_S5_FULLROW, default ON): both the pair PAO
     * rows (ij_u_in_Q) and the used columns come in contiguous runs, so
     * instead of gathering an (npp x n_red) block we dgemm DIRECTLY on
     * the page's contiguous row-blocks at full width (ld = np_full;
     * zero copies, streaming reads), accumulating over row-runs, then
     * run-select the n_red columns of the small (npno x np_full)
     * result.  Same FLOPs to within np_full/n_red (~1.0-1.1 on compact
     * systems).  DLPNO_S5_FULLROW=0 restores the gather+dgemm path.
     * ------------------------------------------------------------------ */
    if (n_red > 0) {
        int int_n_red = (int)n_red;
        static int _s5_fullrow = -1;
        if (_s5_fullrow < 0) {
            const char *_e5 = getenv("DLPNO_S5_FULLROW");
            _s5_fullrow = !(_e5 != NULL && _e5[0] == '0');
        }
        double *proj_full = _s5_fullrow
            ? (double *)malloc(sizeof(double) * npno * np_full) : NULL;
        double *qab_row_gather = _s5_fullrow ? NULL
            : (double *)malloc(sizeof(double) * npp_ij * n_red);
        /* Run-encode pair_used_in_Q once (page-local PAO positions come
         * in ascending contiguous per-atom stretches). */
        long *r5_v0 = (long *)malloc(sizeof(long) * n_red);
        long *r5_s0 = (long *)malloc(sizeof(long) * n_red);
        long *r5_ln = (long *)malloc(sizeof(long) * n_red);
        int n_r5 = 0;
        for (long v = 0; v < (long)n_red; ) {
            long v0 = v, s0 = pair_used_in_Q[v];
            v++;
            while (v < (long)n_red && pair_used_in_Q[v] == s0 + (v - v0)) v++;
            r5_v0[n_r5] = v0; r5_s0[n_r5] = s0; r5_ln[n_r5] = v - v0;
            n_r5++;
        }

        for (size_t q = 0; q < nQp; q++) {
            const size_t pg = (size_t)atom_pos[q];
            const double *qab_q_ptr = qab_b + pg * qab_q;

            if (_s5_fullrow) {
                _t_sec = _cq_prof_on ? _cq_now() : 0.0;
                int int_np_full = (int)np_full;
                for (int r = 0; r < n_ru; r++) {
                    int int_ln = (int)ru_ln[r];
                    const double bet = (r == 0) ? 0.0 : 1.0;
                    dgemm_(&N_flag, &T_flag,
                           &int_np_full, &int_npno, &int_ln,
                           &one, qab_q_ptr + (size_t)ru_s0[r] * qab_u,
                           &int_np_full,
                           X_ij_slice + (size_t)ru_u0[r] * X_row,
                           &int_npno,
                           &bet, proj_full, &int_np_full);
                }
                _cq_add(4, _t_sec);
                _t_sec = _cq_prof_on ? _cq_now() : 0.0;
                double *proj_row_out = proj_ij_out + q * proj_q;
                for (size_t a = 0; a < npno; a++) {
                    const double *pf = proj_full + a * np_full;
                    double *po = proj_row_out + a * n_red;
                    for (int r = 0; r < n_r5; r++) {
                        memcpy(po + r5_v0[r], pf + r5_s0[r],
                               sizeof(double) * r5_ln[r]);
                    }
                }
                _cq_add(3, _t_sec);
            } else {
                _t_sec = _cq_prof_on ? _cq_now() : 0.0;
                for (size_t u = 0; u < npp_ij; u++) {
                    const double *src_row = qab_q_ptr
                        + (size_t)ij_u_in_Q[u] * qab_u;
                    double *dst_row = qab_row_gather + u * n_red;
                    for (int r = 0; r < n_r5; r++) {
                        memcpy(dst_row + r5_v0[r], src_row + r5_s0[r],
                               sizeof(double) * r5_ln[r]);
                    }
                }
                _cq_add(3, _t_sec);
                _t_sec = _cq_prof_on ? _cq_now() : 0.0;
                dgemm_(&N_flag, &T_flag,
                       &int_n_red, &int_npno, &int_npp_ij,
                       &one, qab_row_gather, &int_n_red,
                       X_ij_slice, &int_npno,
                       &zero, proj_ij_out + q * proj_q, &int_n_red);
                _cq_add(4, _t_sec);
            }
        }

        if (qab_row_gather) free(qab_row_gather);
        if (proj_full) free(proj_full);
        free(r5_v0); free(r5_s0); free(r5_ln);
    }

    /* ------------------------------------------------------------------
     * Step 4 (FUSED with step 5): raw_ab reuses proj.
     * The pair's own PAOs are a subset of pair_used (the union includes
     * pair_paos_ij), so step 5's proj[q][a, v_red] already contains the
     * half-transform X^T @ qab at the pair's own columns:
     *     tmp[a, u] = proj[q][a, pair_used_inv[ij_u_in_Q[u]]]
     * raw_ab = tmp @ X then needs only ONE dgemm per Q — the npp x npp
     * gather and the npp^2 x npno first dgemm are eliminated (exact).
     * Falls back to the legacy gather path when proj was not computed.
     * ------------------------------------------------------------------ */
    /* DLPNO_CENTERQ_FUSE=0 forces the legacy step-4 gather path (kept for
     * A/B isolation: the fused path won 22% at TZVPP but is suspected of
     * regressing small-basis runs). */
    static int _fuse_on = -1;
    if (_fuse_on < 0) {
        const char *_e = getenv("DLPNO_CENTERQ_FUSE");
        _fuse_on = !(_e != NULL && _e[0] == '0');
    }
    _t_sec = _cq_prof_on ? _cq_now() : 0.0;
    if (_fuse_on && n_red > 0 && proj_ij_out != NULL) {
        double *tmp = (double *)malloc(sizeof(double) * npno * npp_ij);
        long *red_cols = (long *)malloc(sizeof(long) * npp_ij);
        /* local page-position -> reduced-index inverse (pair's own PAOs
         * are guaranteed inside pair_used, so every lookup resolves) */
        long *inv_loc = (long *)malloc(sizeof(long) * np_full);
        for (size_t i = 0; i < np_full; i++) inv_loc[i] = -1;
        for (size_t vr = 0; vr < (size_t)n_red; vr++) {
            inv_loc[pair_used_in_Q[vr]] = (long)vr;
        }
        for (size_t u = 0; u < npp_ij; u++) {
            red_cols[u] = inv_loc[ij_u_in_Q[u]];
        }
        free(inv_loc);
        for (size_t q = 0; q < nQp; q++) {
            const size_t row_lq = (size_t)local_Q[q];
            const double *proj_row = proj_ij_out + q * proj_q;
            /* tmp[a, u] = proj_row[a, red_cols[u]] (npno x npp column gather) */
            for (size_t a = 0; a < npno; a++) {
                const double *pr = proj_row + a * n_red;
                double *tr = tmp + a * npp_ij;
                for (size_t u = 0; u < npp_ij; u++) {
                    tr[u] = pr[red_cols[u]];
                }
            }
            /* ab[a, b] = sum_v tmp[a, v] * X[v, b]  — same BLAS call as legacy */
            dgemm_(&N_flag, &N_flag,
                   &int_npno, &int_npno, &int_npp_ij,
                   &one, X_ij_slice, &int_npno,
                   tmp, &int_npp_ij,
                   &zero, raw_ab + row_lq * ab_row, &int_npno);
        }
        free(tmp);
        free(red_cols);
    } else {
{
        double *qab_gather = (double *)malloc(sizeof(double) * npp_ij * npp_ij);
        double *tmp = (double *)malloc(sizeof(double) * npno * npp_ij);

        for (size_t q = 0; q < nQp; q++) {
            const size_t row_lq = (size_t)local_Q[q];
            const size_t pg = (size_t)atom_pos[q];
            const double *qab_q_ptr = qab_b + pg * qab_q;

            /* qab_gather[u, v] = qab[ij_u_in_Q[u], ij_u_in_Q[v]] */
            for (size_t u = 0; u < npp_ij; u++) {
                const double *qab_qu = qab_q_ptr + (size_t)ij_u_in_Q[u] * qab_u;
                double *gather_u = qab_gather + u * npp_ij;
                for (size_t v = 0; v < npp_ij; v++) {
                    gather_u[v] = qab_qu[ij_u_in_Q[v]];
                }
            }

            /* tmp[a, v] = sum_u X[u, a] * qab_gather[u, v]   = X^T @ qab_gather
             * Row-major: (npno, npp_ij) = (npp_ij, npno)^T @ (npp_ij, npp_ij).
             * F: tmp_F[v, a] = sum_u qab_gather_F[v, u] * X_F[a, u]
             *               = qab_gather_F @ X_F^T
             * dgemm('N', 'T', npp_ij, npno, npp_ij, 1, qab_gather, npp_ij,
             *       X, npno, 0, tmp, npp_ij)
             */
            dgemm_(&N_flag, &T_flag,
                   &int_npp_ij, &int_npno, &int_npp_ij,
                   &one, qab_gather, &int_npp_ij,
                   X_ij_slice, &int_npno,
                   &zero, tmp, &int_npp_ij);

            /* ab[a, b] = sum_v tmp[a, v] * X[v, b]    = tmp @ X
             * Row-major: (npno, npno) = (npno, npp_ij) @ (npp_ij, npno).
             * F: ab_F[b, a] = sum_v X_F[b, v] * tmp_F[v, a]  =  X_F @ tmp_F
             * dgemm('N', 'N', npno, npno, npp_ij, 1, X, npno, tmp, npp_ij,
             *       0, raw_ab + row_lq*ab_row, npno)
             */
            dgemm_(&N_flag, &N_flag,
                   &int_npno, &int_npno, &int_npp_ij,
                   &one, X_ij_slice, &int_npno,
                   tmp, &int_npp_ij,
                   &zero, raw_ab + row_lq * ab_row, &int_npno);
        }

        free(qab_gather);
        free(tmp);
    }

    _cq_add(5, _t_sec);
    if (ru_u0) { free(ru_u0); free(ru_s0); free(ru_ln); }


    }
}
