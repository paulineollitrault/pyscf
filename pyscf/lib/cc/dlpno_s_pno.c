/* DLPNO-CCSD S_pno_cache build: per-pair-A all-partners-B kernel.
 *
 * Pre-CCSD setup port. Replaces ~25k Python `compute_S_pno` calls
 * during the upfront S_pno_cache build (≈4.9s wall on water-10).
 *
 * Per (key_a, key_b) overlap math (matches local_df.py:compute_S_pno):
 *
 *   S[a, b] = sum_{u, v} X_a[u, a] * S_pao_full[pp_a[u], pp_b[v]] * X_b[v, b]
 *
 * Two small matmuls + one fancy-2D index gather per partner. The C
 * kernel processes ALL partners for ONE key_a in one call, eliminating
 * the per-partner Python+numpy dispatch overhead.
 *
 * Per-partner per-pair-A scratch: T = S_pao_sub @ X_b (n_pao_a, n_pno_b)
 * Then S = X_a^T @ T (n_pno_a, n_pno_b). All BLAS via vhf/fblas.h.
 *
 * Flat-buffer layout per pair-A call:
 *   pp_a:           (n_pao_a,) int64 — global PAO indices for key_a
 *   X_a:            (n_pao_a, n_pno_a) — pair_a's X_pno
 *   partner_n_pao:  (n_partners,) int — n_pao_b per partner
 *   partner_n_pno:  (n_partners,) int — n_pno_b per partner
 *   partner_pp_off: (n_partners+1,) — offsets into partner_pp_flat
 *   partner_pp_flat: concat of pp_b arrays (int64)
 *   partner_X_off:  (n_partners+1,) — offsets into partner_X_flat
 *   partner_X_flat: concat of X_b arrays (each (n_pao_b, n_pno_b))
 *   S_out_off:      (n_partners+1,) — offsets into S_out_flat
 *   S_out_flat:     output, concat of S matrices (each (n_pno_a, n_pno_b))
 */

#include <stddef.h>
#include <stdlib.h>
#include "vhf/fblas.h"

void DLPNObuild_S_pno_for_pair(
        const long   *pp_a,                  /* (n_pao_a,) */
        const double *X_a,                   /* (n_pao_a, n_pno_a) row-major */
        const int     n_pao_a,
        const int     n_pno_a,
        const int     n_partners,
        const int    *partner_n_pao,
        const int    *partner_n_pno,
        const long   *partner_pp_off,
        const long   *partner_pp_flat,
        const long   *partner_X_off,
        const double *partner_X_flat,
        const long   *S_out_off,
        double       *S_out_flat,
        const double *S_pao_full,
        const size_t  n_pao_total)
{
    if (n_partners <= 0) return;

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;

    /* Per-partner: gather S_pao_sub (n_pao_a, n_pao_b), then S = X_a^T @ S_sub @ X_b. */
    for (int p = 0; p < n_partners; p++) {
        const int n_pao_b = partner_n_pao[p];
        const int n_pno_b = partner_n_pno[p];
        if (n_pao_b <= 0 || n_pno_b <= 0 || n_pno_a <= 0) {
            /* Output remains untouched (caller pre-zeros) — caller must
             * skip empty partners before passing if zeros are unwanted. */
            continue;
        }

        const long *pp_b = partner_pp_flat + partner_pp_off[p];
        const double *X_b = partner_X_flat + partner_X_off[p];
        double *S_out = S_out_flat + S_out_off[p];

        /* Gather S_pao_sub[u, v] = S_pao_full[pp_a[u], pp_b[v]] */
        const size_t S_sub_size = (size_t)n_pao_a * (size_t)n_pao_b;
        double *S_sub = (double *)malloc(sizeof(double) * S_sub_size);
        for (int u = 0; u < n_pao_a; u++) {
            const double *S_row = S_pao_full + (size_t)pp_a[u] * n_pao_total;
            double *S_sub_u = S_sub + (size_t)u * n_pao_b;
            for (int v = 0; v < n_pao_b; v++) {
                S_sub_u[v] = S_row[pp_b[v]];
            }
        }

        /* T = S_sub @ X_b   (n_pao_a, n_pno_b)
         * Row-major: T[u, b] = sum_v S_sub[u, v] * X_b[v, b]
         * Col-major dgemm: T_col[b, u] = sum_v X_b_col[b, v] * S_sub_col[v, u]
         *                = sum_v X_b[v, b] * S_sub[u, v]
         * dgemm('N', 'N', n_pno_b, n_pao_a, n_pao_b,
         *       1, X_b_buf, n_pno_b, S_sub_buf, n_pao_b, 0, T_buf, n_pno_b)
         */
        const size_t T_size = (size_t)n_pao_a * (size_t)n_pno_b;
        double *T = (double *)malloc(sizeof(double) * T_size);
        int int_n_pao_a = n_pao_a;
        int int_n_pao_b = n_pao_b;
        int int_n_pno_a = n_pno_a;
        int int_n_pno_b = n_pno_b;
        dgemm_(&N_flag, &N_flag,
               &int_n_pno_b, &int_n_pao_a, &int_n_pao_b,
               &one, X_b, &int_n_pno_b,
               S_sub, &int_n_pao_b,
               &zero, T, &int_n_pno_b);

        /* S_out = X_a^T @ T   (n_pno_a, n_pno_b)
         * Row-major: S[a, b] = sum_u X_a[u, a] * T[u, b]
         * Col-major: S_col[b, a] = sum_u T_col[b, u] * X_a_col[a, u]
         *                       = sum_u T[u, b] * X_a[u, a]
         * dgemm('N', 'T', n_pno_b, n_pno_a, n_pao_a,
         *       1, T_buf, n_pno_b, X_a_buf, n_pno_a, 0, S_out_buf, n_pno_b)
         */
        dgemm_(&N_flag, &T_flag,
               &int_n_pno_b, &int_n_pno_a, &int_n_pao_a,
               &one, T, &int_n_pno_b,
               X_a, &int_n_pno_a,
               &zero, S_out, &int_n_pno_b);

        free(S_sub);
        free(T);
    }
}
