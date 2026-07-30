/* DLPNO cc_ints AUX-FIRST W/M replay + partner transforms — one nogil
 * call per pair.
 *
 * The Python replay loop (local_df.py, aux-first branch) was measured at
 * 55% GIL occupancy across the 32-worker pool: per (pair, centerQ) item
 * it runs ~15 numpy glue ops (searchsorted, ascontiguousarray, fancy
 * gathers/scatters) around two dgemms, then ~2x n_partners small
 * transforms.  This kernel replicates the EXACT numpy semantics of that
 * loop in C so a worker holds the GIL only to marshal pointers.
 *
 * Per replay item (one per centerQ with proj data):
 *   cols_g[v] = lower_bound(pair_used_pao_global, used[v])
 *   Wij(2*nlmo_p, npno*nred) = T_ij[lq].T @ pj(nQp, npno*nred)
 *     W_i[:, :, cols_g] += Wij[0];  W_j[:, :, cols_g] += Wij[1]
 *   krows[t] = lda[all_k[t]]  (skip < 0);  pcols[v] = pda[used[v]]
 *   sub(nQp, nks, nred): sub[q,t,v] = page[apos[q], krows[t], pcols[v]]
 *   Mij(2*npno, nks*nred) = Z_ijv[:, lq] @ sub(nQp, nks*nred)
 *     M_i[kl[t], a, cols_g[v]] += Mij[0, a, t, v]   (same for j side)
 * Then per partner (kj side; ki side identical with W_j/M_j):
 *   cols_k[u] = lower_bound(pair_used_pao_global, pp_k[u])
 *   J_out = W_i[k_loc][:, cols_k] @ X_k     (npno, npno_k)
 *   K_out = M_i[k_loc][:, cols_k] @ X_k
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

static inline long _lb(const long *arr, long n, long v)
{
    long lo = 0, hi = n;
    while (lo < hi) {
        long mid = (lo + hi) >> 1;
        if (arr[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

void DLPNOwm_replay_pair(
        const int     n_items,
        const long   *item_nQp,        /* (n_items,) */
        const long   *item_nred,       /* (n_items,) */
        const long   *lq_flat,   const long *lq_off,
        const long   *apos_flat, const long *apos_off,
        const double **page_ptrs,      /* (n_items,) qia page base */
        const long   *page_nl,         /* (n_items,) page LMO dim */
        const long   *page_np,         /* (n_items,) page PAO dim */
        const double *pj_flat,   const long *pj_off,
        const long   *used_flat, const long *used_off,
        const long  **lda_ptrs,        /* (n_items,) global-LMO -> page row */
        const long  **pda_ptrs,        /* (n_items,) global-PAO -> page col */
        const double *T_ij,            /* (n_local, 2*nlmo_p) row-major */
        const double *Z_ijv,           /* (2*npno, n_local) row-major */
        const long   *pug,             /* pair_used_pao_global, sorted */
        const long    n_red_g,
        const long   *all_k, const int n_all_k,
        const long   *p_lmos_dense,    /* global k -> dense row in W/M */
        const int     nlmo_p, const int npno, const long n_local,
        double       *W_i, double *W_j, double *M_i, double *M_j,
        /* partners, kj side then ki side */
        const int     n_kj,
        const long   *kj_k,            /* (n_kj,) global k */
        const double **kj_X,           /* (n_kj,) X_pno base (n_pao_k, npno_k) */
        const long   *kj_npao, const long *kj_npno,
        const long   *kj_pp_flat, const long *kj_pp_off,
        const int     n_ki,
        const long   *ki_k,
        const double **ki_X,
        const long   *ki_npao, const long *ki_npno,
        const long   *ki_pp_flat, const long *ki_pp_off,
        double       *Jkj_flat, const long *Jkj_off,
        double       *Kkj_flat,
        double       *Jki_flat, const long *Jki_off,
        double       *Kki_flat)
{
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;
    const int two_nlmo = 2 * nlmo_p;
    const int two_npno = 2 * npno;

    /* Scratch maxima over items. */
    long max_nQp = 0, max_nred = 0;
    for (int it = 0; it < n_items; it++) {
        if (item_nQp[it] > max_nQp) max_nQp = item_nQp[it];
        if (item_nred[it] > max_nred) max_nred = item_nred[it];
    }
    if (max_nQp == 0) max_nQp = 1;
    if (max_nred == 0) max_nred = 1;

    long *cols_g = (long *)malloc(sizeof(long) * max_nred);
    long *krows  = (long *)malloc(sizeof(long) * (n_all_k > 0 ? n_all_k : 1));
    long *klocs  = (long *)malloc(sizeof(long) * (n_all_k > 0 ? n_all_k : 1));
    long *pcols  = (long *)malloc(sizeof(long) * max_nred);
    double *T_lq  = (double *)malloc(sizeof(double) * max_nQp * two_nlmo);
    double *Z_lq  = (double *)malloc(sizeof(double) * two_npno * max_nQp);
    double *Wij   = (double *)malloc(sizeof(double)
                                     * (size_t)two_nlmo * npno * max_nred);
    double *sub   = (double *)malloc(sizeof(double) * (size_t)max_nQp
                                     * (n_all_k > 0 ? n_all_k : 1) * max_nred);
    double *Mij   = (double *)malloc(sizeof(double) * (size_t)two_npno
                                     * (n_all_k > 0 ? n_all_k : 1) * max_nred);
    /* Run-length encodings: PAO domains are unions of per-atom contiguous
     * ranges, so cols_g / pcols come in ascending runs (typ. 8-14 long).
     * Encoding them once per item turns the per-element gathers/scatters
     * below into run-wise memcpy / contiguous vector-adds. */
    long *rg_v0  = (long *)malloc(sizeof(long) * max_nred);  /* cols_g runs */
    long *rg_d0  = (long *)malloc(sizeof(long) * max_nred);
    long *rg_len = (long *)malloc(sizeof(long) * max_nred);
    long *rp_v0  = (long *)malloc(sizeof(long) * max_nred);  /* pcols runs */
    long *rp_s0  = (long *)malloc(sizeof(long) * max_nred);
    long *rp_len = (long *)malloc(sizeof(long) * max_nred);

    for (int it = 0; it < n_items; it++) {
        const long nQp  = item_nQp[it];
        const long nred = item_nred[it];
        if (nQp == 0 || nred == 0) continue;
        const long *lq   = lq_flat + lq_off[it];
        const long *apos = apos_flat + apos_off[it];
        const long *used = used_flat + used_off[it];
        const long *lda  = lda_ptrs[it];
        const long *pda  = pda_ptrs[it];
        const double *page = page_ptrs[it];
        const long nl_pg = page_nl[it];
        const long np_pg = page_np[it];
        const double *pj = pj_flat + pj_off[it];

        for (long v = 0; v < nred; v++) {
            cols_g[v] = _lb(pug, n_red_g, used[v]);
        }

        int n_rg = 0;
        for (long v = 0; v < nred; ) {
            long v0 = v, d0 = cols_g[v];
            v++;
            while (v < nred && cols_g[v] == d0 + (v - v0)) v++;
            rg_v0[n_rg] = v0; rg_d0[n_rg] = d0; rg_len[n_rg] = v - v0;
            n_rg++;
        }

        /* ---- W side: Wij = T_ij[lq].T @ pj ---------------------------
         * T_lq gather: (nQp, 2*nlmo_p) rows of T_ij.
         * Row-major C(m,n) = A_rm(k,m)^T B_rm(k,n) with m=2*nlmo_p,
         * n=npno*nred, k=nQp:
         *   Fortran: C_cm(n,m) = B_cm(n,k) @ A_cm(k,m)^T? — use the
         *   standard recipe C_rm = A_rm^T B_rm:
         *   dgemm('N','T', n, m, k, B_rm(ld=n), A_rm(ld=m), C(ld=n)). */
        for (long q = 0; q < nQp; q++) {
            memcpy(T_lq + q * two_nlmo, T_ij + lq[q] * two_nlmo,
                   sizeof(double) * two_nlmo);
        }
        {
            int m = two_nlmo;
            int n = (int)(npno * nred);
            int k = (int)nQp;
            dgemm_(&N_flag, &T_flag, &n, &m, &k,
                   &one, pj, &n,
                   T_lq, &m,
                   &zero, Wij, &n);
        }
        /* scatter W_i / W_j: W[l, a, cols_g[v]] += Wij[side, l, a, v] */
        for (int l = 0; l < nlmo_p; l++) {
            for (int a = 0; a < npno; a++) {
                const double *src_i = Wij + ((size_t)l * npno + a) * nred;
                const double *src_j = Wij
                    + ((size_t)(nlmo_p + l) * npno + a) * nred;
                double *row_i = W_i + ((size_t)l * npno + a) * n_red_g;
                double *row_j = W_j + ((size_t)l * npno + a) * n_red_g;
                for (int r = 0; r < n_rg; r++) {
                    const long v0 = rg_v0[r], d0 = rg_d0[r], ln = rg_len[r];
                    double *ri = row_i + d0;
                    double *rj = row_j + d0;
                    const double *si = src_i + v0;
                    const double *sj = src_j + v0;
                    for (long v = 0; v < ln; v++) {
                        ri[v] += si[v];
                        rj[v] += sj[v];
                    }
                }
            }
        }

        /* ---- M side ------------------------------------------------- */
        int nks = 0;
        for (int t = 0; t < n_all_k; t++) {
            const long kr = lda[all_k[t]];
            if (kr >= 0) {
                krows[nks] = kr;
                klocs[nks] = p_lmos_dense[all_k[t]];
                nks++;
            }
        }
        if (nks == 0) continue;
        for (long v = 0; v < nred; v++) {
            pcols[v] = pda[used[v]];
        }

        int n_rp = 0;
        for (long v = 0; v < nred; ) {
            long v0 = v, s0 = pcols[v];
            v++;
            while (v < nred && pcols[v] == s0 + (v - v0)) v++;
            rp_v0[n_rp] = v0; rp_s0[n_rp] = s0; rp_len[n_rp] = v - v0;
            n_rp++;
        }
        /* sub[q, t, v] = page[apos[q], krows[t], pcols[v]] */
        for (long q = 0; q < nQp; q++) {
            const double *pq = page + apos[q] * nl_pg * np_pg;
            double *sq = sub + q * (size_t)nks * nred;
            for (int t = 0; t < nks; t++) {
                const double *pr = pq + krows[t] * np_pg;
                double *sr = sq + (size_t)t * nred;
                for (int r = 0; r < n_rp; r++) {
                    memcpy(sr + rp_v0[r], pr + rp_s0[r],
                           sizeof(double) * rp_len[r]);
                }
            }
        }
        /* Mij(2*npno, nks*nred) = Z_lq(2*npno, nQp) @ sub(nQp, nks*nred).
         * Z_lq gather: columns lq of Z_ijv. */
        for (int a = 0; a < two_npno; a++) {
            const double *zrow = Z_ijv + (size_t)a * n_local;
            double *zdst = Z_lq + (size_t)a * nQp;
            for (long q = 0; q < nQp; q++) {
                zdst[q] = zrow[lq[q]];
            }
        }
        {
            int m = two_npno;
            int n = (int)((size_t)nks * nred);
            int k = (int)nQp;
            /* C_rm(m,n) = A_rm(m,k) B_rm(k,n):
             * dgemm('N','N', n, m, k, B(ld=n), A(ld=k), C(ld=n)) */
            dgemm_(&N_flag, &N_flag, &n, &m, &k,
                   &one, sub, &n,
                   Z_lq, &k,
                   &zero, Mij, &n);
        }
        /* scatter: M[kloc[t], a, cols_g[v]] += Mij[side*npno + a, t, v] */
        for (int t = 0; t < nks; t++) {
            const long kl = klocs[t];
            for (int a = 0; a < npno; a++) {
                const double *mi = Mij + ((size_t)a * nks + t) * nred;
                const double *mj = Mij
                    + ((size_t)(npno + a) * nks + t) * nred;
                double *rid = M_i + ((size_t)kl * npno + a) * n_red_g;
                double *rjd = M_j + ((size_t)kl * npno + a) * n_red_g;
                for (int r = 0; r < n_rg; r++) {
                    const long v0 = rg_v0[r], d0 = rg_d0[r], ln = rg_len[r];
                    double *ri2 = rid + d0;
                    double *rj2 = rjd + d0;
                    const double *si = mi + v0;
                    const double *sj = mj + v0;
                    for (long v = 0; v < ln; v++) {
                        ri2[v] += si[v];
                        rj2[v] += sj[v];
                    }
                }
            }
        }
    }

    free(T_lq); free(Z_lq); free(Wij); free(sub); free(Mij);
    free(krows); free(pcols);
    free(rg_v0); free(rg_d0); free(rg_len);
    free(rp_v0); free(rp_s0); free(rp_len);

    /* ---- partner transforms ----------------------------------------- */
    long max_pp = 0;
    for (int p = 0; p < n_kj; p++) {
        if (kj_npao[p] > max_pp) max_pp = kj_npao[p];
    }
    for (int p = 0; p < n_ki; p++) {
        if (ki_npao[p] > max_pp) max_pp = ki_npao[p];
    }
    if (max_pp == 0) max_pp = 1;
    long *cols_k = (long *)malloc(sizeof(long) * max_pp);
    long *rk_u0  = (long *)malloc(sizeof(long) * max_pp);
    long *rk_c0  = (long *)malloc(sizeof(long) * max_pp);
    long *rk_len = (long *)malloc(sizeof(long) * max_pp);
    double *WK_sub = (double *)malloc(sizeof(double) * (size_t)npno * max_pp);
    int int_npno = npno;

    for (int side = 0; side < 2; side++) {
        const int n_p = side == 0 ? n_kj : n_ki;
        const long *pk = side == 0 ? kj_k : ki_k;
        const double **pX = side == 0 ? kj_X : ki_X;
        const long *pnpao = side == 0 ? kj_npao : ki_npao;
        const long *pnpno = side == 0 ? kj_npno : ki_npno;
        const long *pp_flat = side == 0 ? kj_pp_flat : ki_pp_flat;
        const long *pp_off = side == 0 ? kj_pp_off : ki_pp_off;
        const double *Wm = side == 0 ? W_i : W_j;
        const double *Mm = side == 0 ? M_i : M_j;
        double *J_flat = side == 0 ? Jkj_flat : Jki_flat;
        double *K_flat = side == 0 ? Kkj_flat : Kki_flat;
        const long *J_off = side == 0 ? Jkj_off : Jki_off;

        for (int p = 0; p < n_p; p++) {
            const long npao_k = pnpao[p];
            const int npno_k = (int)pnpno[p];
            if (npao_k == 0 || npno_k == 0) continue;
            const long kloc = p_lmos_dense[pk[p]];
            const long *pp_k = pp_flat + pp_off[p];
            for (long u = 0; u < npao_k; u++) {
                cols_k[u] = _lb(pug, n_red_g, pp_k[u]);
            }
            /* run-encode cols_k (ascending contiguous stretches) */
            int n_rk = 0;
            for (long u = 0; u < npao_k; ) {
                long u0 = u, c0 = cols_k[u];
                u++;
                while (u < npao_k && cols_k[u] == c0 + (u - u0)) u++;
                rk_u0[n_rk] = u0; rk_c0[n_rk] = c0; rk_len[n_rk] = u - u0;
                n_rk++;
            }
            const double *X_k = pX[p];
            double *J_out = J_flat + J_off[p];
            double *K_out = K_flat + J_off[p];
            int int_npao = (int)npao_k;

            /* WK_sub(npno, npao_k) = W[kloc][:, cols_k], then
             * J_out(npno, npno_k) = WK_sub @ X_k(npao_k, npno_k). */
            const double *Wrow = Wm + (size_t)kloc * npno * n_red_g;
            for (int a = 0; a < npno; a++) {
                const double *wr = Wrow + (size_t)a * n_red_g;
                double *dst = WK_sub + (size_t)a * npao_k;
                for (int r = 0; r < n_rk; r++) {
                    memcpy(dst + rk_u0[r], wr + rk_c0[r],
                           sizeof(double) * rk_len[r]);
                }
            }
            dgemm_(&N_flag, &N_flag, &npno_k, &int_npno, &int_npao,
                   &one, X_k, &npno_k,
                   WK_sub, &int_npao,
                   &zero, J_out, &npno_k);

            const double *Mrow = Mm + (size_t)kloc * npno * n_red_g;
            for (int a = 0; a < npno; a++) {
                const double *mr = Mrow + (size_t)a * n_red_g;
                double *dst = WK_sub + (size_t)a * npao_k;
                for (int r = 0; r < n_rk; r++) {
                    memcpy(dst + rk_u0[r], mr + rk_c0[r],
                           sizeof(double) * rk_len[r]);
                }
            }
            dgemm_(&N_flag, &N_flag, &npno_k, &int_npno, &int_npao,
                   &one, X_k, &npno_k,
                   WK_sub, &int_npao,
                   &zero, K_out, &npno_k);
        }
    }

    free(cols_g); free(klocs);
    free(cols_k); free(WK_sub);
    free(rk_u0); free(rk_c0); free(rk_len);
}
