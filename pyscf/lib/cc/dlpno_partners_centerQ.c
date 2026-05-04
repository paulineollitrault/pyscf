/* DLPNO-CCSD compute_cc_integrals_sparse: per-centerQ partner loop in C.
 *
 * Pre-CCSD setup port. Replaces the per-partner Python prep + ctypes
 * dispatch loop inside the centerQ inner body in
 * local_df.py:_process_pair (lines ~853-885 — kj_data + ki_data side
 * loops). Per centerQ ~30 partners × 5 small Python ops + 1 ctypes
 * call (= partner_apply) per partner. With ~5 centerQ × 820 pairs =
 * ~120k Python-loop iterations across all threads, GIL-serialized.
 * One C call per (pair, centerQ, side) replaces all of that.
 *
 * Per partner k inside this kernel:
 *   k_s          = riatom_to_lmos_ext_dense_row[k]
 *   kj_u_in_pair, kj_u_in_Q built from partner_pp_flat[p] +
 *     riatom_to_paos_ext_dense_row
 *   X_k_slice    = X_k[kj_u_in_pair, :]   (npp_kj, n_kj)
 *   raw_cross_p[local_Q[q], b, m]
 *       = sum_u proj_ij[q, b, kj_u_in_Q[u]] * X_k_slice[u, m]
 *   raw_kv_p[local_Q[q], m]
 *       = sum_u qia_b[q, k_s, kj_u_in_Q[u]] * X_k_slice[u, m]
 *
 * Two DGEMMs per partner replace the scalar (q, b, m, u) loop nest:
 *   gather proj_ij[:, :, kj_u_in_Q] → proj_gather (nQp, npno, npp_kj)
 *   DGEMM:  (nQp*npno, npp_kj) @ X_k_slice (npp_kj, n_kj)
 *           → (nQp*npno, n_kj)  scatter to raw_cross_p
 *   gather qia[atom_pos, k_s, kj_u_in_Q] → qia_gather (nQp, npp_kj)
 *   DGEMM:  (nQp, npp_kj) @ X_k_slice (npp_kj, n_kj) → (nQp, n_kj)
 *           scatter to raw_kv_p
 *
 * The kernel is serial (one pair at a time); outer parallelism stays
 * over pairs via the Python ThreadPoolExecutor.
 *
 * Skips partners with npp_kj == 0 (empty PAO intersection at this
 * centerQ). Output offsets for skipped partners are still written
 * with whatever was there — caller must zero-init the raw_cross /
 * raw_kv buffers once per pair before the centerQ loop (they are
 * accumulators across centerQ).
 *
 * Flat-buffer layout (built per-pair by Python wrapper):
 *   partner_k_arr:       (n_partners,) global LMO index per partner
 *   partner_n_kj:        (n_partners,)
 *   partner_pp_off:      (n_partners+1,) into partner_pp_flat
 *   partner_pp_flat:     concat of pp_k (global PAO indices per partner)
 *   partner_X_off:       (n_partners+1,) into partner_X_flat
 *   partner_X_flat:      concat of X_k (each (n_pao_k_total, n_kj_p))
 *   partner_cross_off:   (n_partners,) absolute offset into raw_cross_flat
 *   partner_kv_off:      (n_partners,) absolute offset into raw_kv_flat
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

void DLPNOpartners_centerQ_step(
        const double *proj_ij,                /* (nQp, npno, np_full) */
        const double *qia_atom_full,          /* (nQ_at_atom, nl, np_full) */
        const long   *atom_pos,               /* (nQp,) */
        const long   *local_Q,                /* (nQp,) */
        const long   *riatom_to_paos_dense_at,/* (nao_pao_total,) */
        const long   *riatom_to_lmos_dense_at,/* (nocc,) */
        const int     n_partners,
        const long   *partner_k_arr,
        const long   *partner_n_kj,
        const long   *partner_pp_off,
        const long   *partner_pp_flat,
        const long   *partner_X_off,
        const double *partner_X_flat,
        const long   *partner_cross_off,
        const long   *partner_kv_off,
        const size_t  nQp,
        const size_t  npno,
        const size_t  np_full,
        const size_t  nl,
        const size_t  n_local,
        const size_t  nao_pao_total,
        const size_t  nocc,
        double       *raw_cross_flat,
        double       *raw_kv_flat)
{
    if (n_partners == 0 || nQp == 0) return;

    const size_t proj_q = npno * np_full;
    const size_t proj_b = np_full;
    const size_t qia_q  = nl * np_full;
    const size_t qia_l  = np_full;

    static const char N_flag = 'N';
    static const double one = 1.0;
    static const double zero = 0.0;

    /* Reusable scratch — sized by max n_pao_k across partners (cheap upper
     * bound; alloc once per call). proj_gather/qia_gather/X_slice/cross_loc/
     * kv_loc capacities grow to fit largest partner. */
    long *kj_u_in_pair = NULL;
    long *kj_u_in_Q    = NULL;
    long  cap_pao = 0;

    double *X_slice    = NULL;
    long    cap_X      = 0;
    double *proj_gather = NULL;
    long    cap_proj    = 0;
    double *cross_loc  = NULL;
    long    cap_cross  = 0;
    double *qia_gather = NULL;
    long    cap_qiag   = 0;
    double *kv_loc     = NULL;
    long    cap_kv     = 0;

    for (int p = 0; p < n_partners; p++) {
        const long n_kj = partner_n_kj[p];
        if (n_kj <= 0) continue;
        const long n_pao_k = partner_pp_off[p + 1] - partner_pp_off[p];
        if (n_pao_k <= 0) continue;
        const long *pp_k = partner_pp_flat + partner_pp_off[p];
        const double *X_k = partner_X_flat + partner_X_off[p];
        const long k_global = partner_k_arr[p];
        const long k_s = riatom_to_lmos_dense_at[k_global];

        if (n_pao_k > cap_pao) {
            free(kj_u_in_pair); free(kj_u_in_Q);
            cap_pao = n_pao_k;
            kj_u_in_pair = (long *)malloc(sizeof(long) * (size_t)cap_pao);
            kj_u_in_Q    = (long *)malloc(sizeof(long) * (size_t)cap_pao);
        }

        long npp_kj = 0;
        for (long u = 0; u < n_pao_k; u++) {
            const long pao_global = pp_k[u];
            const long pos = riatom_to_paos_dense_at[pao_global];
            if (pos >= 0) {
                kj_u_in_pair[npp_kj] = u;
                kj_u_in_Q[npp_kj]    = pos;
                npp_kj++;
            }
        }
        if (npp_kj == 0) continue;

        /* X_k_slice (npp_kj, n_kj) — gather rows of X_k. */
        const long need_X = npp_kj * n_kj;
        if (need_X > cap_X) {
            free(X_slice);
            cap_X = need_X;
            X_slice = (double *)malloc(sizeof(double) * (size_t)cap_X);
        }
        for (long u = 0; u < npp_kj; u++) {
            memcpy(X_slice + (size_t)u * (size_t)n_kj,
                   X_k + (size_t)kj_u_in_pair[u] * (size_t)n_kj,
                   sizeof(double) * (size_t)n_kj);
        }

        double *raw_cross_p = raw_cross_flat + partner_cross_off[p];
        double *raw_kv_p    = raw_kv_flat    + partner_kv_off[p];

        /* ---- raw_cross via DGEMM ----
         * Gather proj_ij[q, b, kj_u_in_Q[u]] → proj_gather (nQp, npno, npp_kj)
         * Then cross_loc(nQp*npno, n_kj) = proj_gather(nQp*npno, npp_kj)
         *                                   @ X_slice(npp_kj, n_kj)
         * In Fortran column-major DGEMM:
         *   M=n_kj, N=nQp*npno, K=npp_kj
         *   A = X_slice(n_kj, npp_kj)  [row-major (npp_kj,n_kj) == col-major (n_kj,npp_kj)]
         *   B = proj_gather(npp_kj, nQp*npno) [row-major (nQp*npno,npp_kj)]
         *   C = cross_loc(n_kj, nQp*npno)     [row-major (nQp*npno,n_kj)]
         */
        const long need_proj = (long)nQp * (long)npno * npp_kj;
        if (need_proj > cap_proj) {
            free(proj_gather);
            cap_proj = need_proj;
            proj_gather = (double *)malloc(sizeof(double) * (size_t)cap_proj);
        }
        const long need_cross_loc = (long)nQp * (long)npno * n_kj;
        if (need_cross_loc > cap_cross) {
            free(cross_loc);
            cap_cross = need_cross_loc;
            cross_loc = (double *)malloc(sizeof(double) * (size_t)cap_cross);
        }
        /* Gather: tight inner loop over u (contiguous in dest). */
        for (size_t q = 0; q < nQp; q++) {
            const double *proj_q_ptr = proj_ij + q * proj_q;
            double *pg_q = proj_gather + q * npno * (size_t)npp_kj;
            for (size_t b = 0; b < npno; b++) {
                const double *proj_qb = proj_q_ptr + b * proj_b;
                double *pg_qb = pg_q + b * (size_t)npp_kj;
                for (long u = 0; u < npp_kj; u++) {
                    pg_qb[u] = proj_qb[kj_u_in_Q[u]];
                }
            }
        }
        {
            int M = (int)n_kj;
            int N = (int)(nQp * npno);
            int K = (int)npp_kj;
            int lda = M, ldb = K, ldc = M;
            dgemm_(&N_flag, &N_flag, &M, &N, &K,
                   &one, X_slice, &lda,
                   proj_gather, &ldb,
                   &zero, cross_loc, &ldc);
        }
        /* Scatter cross_loc[q, b, m] → raw_cross_p[local_Q[q], b, m]. */
        const size_t cross_row_stride = npno * (size_t)n_kj;
        for (size_t q = 0; q < nQp; q++) {
            const size_t row = (size_t)local_Q[q];
            memcpy(raw_cross_p + row * cross_row_stride,
                   cross_loc + q * cross_row_stride,
                   sizeof(double) * cross_row_stride);
        }

        /* ---- raw_kv via DGEMM ----
         * Gather qia[atom_pos[q], k_s, kj_u_in_Q[u]] → qia_gather (nQp, npp_kj)
         * Then kv_loc(nQp, n_kj) = qia_gather(nQp, npp_kj) @ X_slice(npp_kj, n_kj)
         */
        if (k_s >= 0) {
            const long need_qiag = (long)nQp * npp_kj;
            if (need_qiag > cap_qiag) {
                free(qia_gather);
                cap_qiag = need_qiag;
                qia_gather = (double *)malloc(sizeof(double) * (size_t)cap_qiag);
            }
            const long need_kv = (long)nQp * n_kj;
            if (need_kv > cap_kv) {
                free(kv_loc);
                cap_kv = need_kv;
                kv_loc = (double *)malloc(sizeof(double) * (size_t)cap_kv);
            }
            for (size_t q = 0; q < nQp; q++) {
                const size_t pg = (size_t)atom_pos[q];
                const double *qia_qk = qia_atom_full + pg * qia_q
                                       + (size_t)k_s * qia_l;
                double *qg_q = qia_gather + q * (size_t)npp_kj;
                for (long u = 0; u < npp_kj; u++) {
                    qg_q[u] = qia_qk[kj_u_in_Q[u]];
                }
            }
            {
                int M = (int)n_kj;
                int N = (int)nQp;
                int K = (int)npp_kj;
                int lda = M, ldb = K, ldc = M;
                dgemm_(&N_flag, &N_flag, &M, &N, &K,
                       &one, X_slice, &lda,
                       qia_gather, &ldb,
                       &zero, kv_loc, &ldc);
            }
            for (size_t q = 0; q < nQp; q++) {
                const size_t row = (size_t)local_Q[q];
                memcpy(raw_kv_p + row * (size_t)n_kj,
                       kv_loc + q * (size_t)n_kj,
                       sizeof(double) * (size_t)n_kj);
            }
        }
    }

    free(kj_u_in_pair); free(kj_u_in_Q);
    free(X_slice); free(proj_gather); free(cross_loc);
    free(qia_gather); free(kv_loc);
}
