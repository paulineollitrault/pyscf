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
 * The kernel is serial (one pair at a time); outer parallelism stays
 * over pairs via the Python ThreadPoolExecutor. partner_apply (C) was
 * already serial-per-call too, so this only collapses Python wrappers,
 * not concurrency. Same compute as
 * pyscf/lib/cc/dlpno_partner.c::DLPNOpartner_apply repeated per
 * partner.
 *
 * Skips partners with npp_kj == 0 (empty PAO intersection at this
 * centerQ). The output offsets for skipped partners are still written
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

void DLPNOpartners_centerQ_step(
        const double *proj_ij,                /* (nQp, npno, np_full) */
        const double *qia_b,                  /* (nQp, nl, np_full) */
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

    for (int p = 0; p < n_partners; p++) {
        const long n_kj = partner_n_kj[p];
        if (n_kj <= 0) continue;
        const long n_pao_k = partner_pp_off[p + 1] - partner_pp_off[p];
        if (n_pao_k <= 0) continue;
        const long *pp_k = partner_pp_flat + partner_pp_off[p];
        const double *X_k = partner_X_flat + partner_X_off[p];
        const long k_global = partner_k_arr[p];
        const long k_s = riatom_to_lmos_dense_at[k_global];

        /* Build kj_u_in_pair, kj_u_in_Q on heap (stack VLAs are problematic
         * for OMP-clean code; n_pao_k ≤ ~150 so this malloc is cheap). */
        long *kj_u_in_pair = (long *)malloc(sizeof(long) * (size_t)n_pao_k);
        long *kj_u_in_Q    = (long *)malloc(sizeof(long) * (size_t)n_pao_k);
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
        if (npp_kj == 0) {
            free(kj_u_in_pair); free(kj_u_in_Q);
            continue;
        }

        /* X_k_slice (npp_kj, n_kj) — gather rows of X_k. */
        double *X_k_slice = (double *)malloc(
            sizeof(double) * (size_t)npp_kj * (size_t)n_kj);
        for (long u = 0; u < npp_kj; u++) {
            memcpy(X_k_slice + (size_t)u * (size_t)n_kj,
                   X_k + (size_t)kj_u_in_pair[u] * (size_t)n_kj,
                   sizeof(double) * (size_t)n_kj);
        }

        double *raw_cross_p = raw_cross_flat + partner_cross_off[p];
        double *raw_kv_p    = raw_kv_flat    + partner_kv_off[p];

        /* raw_cross_p[local_Q[q], b, m] = sum_u proj_ij[q, b, kj_u_in_Q[u]]
         *                                * X_k_slice[u, m]
         *
         * Memory layout reminder:
         *   raw_cross_p has shape (n_local_total, npno, n_kj) — write only the
         *   local_Q[q] rows; other rows untouched (caller pre-zeroed once per pair).
         */
        const size_t cross_row_stride = npno * (size_t)n_kj;
        const size_t cross_b_stride   = (size_t)n_kj;
        for (size_t q = 0; q < nQp; q++) {
            const size_t row = (size_t)local_Q[q];
            const double *proj_q_ptr = proj_ij + q * proj_q;
            double *cross_row_ptr = raw_cross_p + row * cross_row_stride;
            for (size_t b = 0; b < npno; b++) {
                const double *proj_qb = proj_q_ptr + b * proj_b;
                double *cross_qb = cross_row_ptr + b * cross_b_stride;
                for (long m = 0; m < n_kj; m++) {
                    double s = 0.0;
                    for (long u = 0; u < npp_kj; u++) {
                        s += proj_qb[kj_u_in_Q[u]]
                             * X_k_slice[(size_t)u * (size_t)n_kj + (size_t)m];
                    }
                    cross_qb[m] = s;
                }
            }
        }

        /* raw_kv_p[local_Q[q], m] = sum_u qia_b[q, k_s, kj_u_in_Q[u]] * X_k_slice[u, m] */
        if (k_s >= 0) {
            for (size_t q = 0; q < nQp; q++) {
                const size_t row = (size_t)local_Q[q];
                const double *qia_qk = qia_b + q * qia_q + (size_t)k_s * qia_l;
                double *kv_row = raw_kv_p + row * (size_t)n_kj;
                for (long m = 0; m < n_kj; m++) {
                    double s = 0.0;
                    for (long u = 0; u < npp_kj; u++) {
                        s += qia_qk[kj_u_in_Q[u]]
                             * X_k_slice[(size_t)u * (size_t)n_kj + (size_t)m];
                    }
                    kv_row[m] = s;
                }
            }
        }

        free(kj_u_in_pair);
        free(kj_u_in_Q);
        free(X_k_slice);
    }
}
