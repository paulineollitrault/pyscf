/* DLPNO S_pno: one-shot build across all (canon_il, canon_lj) pairs.
 *
 * Replaces the per-canon_il pool.map dispatch in residual.py
 * build_G_tilde plan side build (~1275 Python ctypes calls, ~1052ms
 * on water-10).
 *
 * Single C entry point processes ALL canon_il at once with OMP-over-
 * canon_il and per-thread scratch buffers (sized for max n_pao*n_pno).
 *
 * For each (canon_il, partner) pair the math is the same as
 * DLPNObuild_S_pno_for_pair:
 *   T = S_pao_sub @ X_b   (n_pao_a, n_pno_b)
 *   S = X_a^T @ T         (n_pno_a, n_pno_b) → S_side_buf[S_out_off]
 *
 * Partner arrays are concatenated across all canon_il; canon_partner_start
 * gives the [start, end) slice into them for each canon_il.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

void DLPNObuild_S_pno_all(
        const long n_canon_il,
        const int *canon_n_pao_a,            /* n_canon_il */
        const int *canon_n_pno_a,            /* n_canon_il */
        const long *canon_pp_a_off,          /* n_canon_il + 1 */
        const long *canon_pp_a_flat,
        const long *canon_X_a_off,           /* n_canon_il + 1 */
        const double *canon_X_a_flat,
        const long *canon_partner_start,     /* n_canon_il + 1 — slice */

        const int *partner_n_pao,            /* n_total_partners */
        const int *partner_n_pno,            /* n_total_partners */
        const long *partner_pp_off,          /* n_total_partners + 1 */
        const long *partner_pp_flat,
        const long *partner_X_off,           /* n_total_partners + 1 */
        const double *partner_X_flat,
        const long *partner_S_out_off,       /* n_total_partners + 1 */
        double *S_side_buf,                  /* output, written at S_out_off */

        const double *S_pao_full,
        const long n_pao_total,

        const int max_n_pao_a,
        const int max_n_pao_b,
        const int max_n_pno_b,
        const int n_threads)
{
    if (n_canon_il <= 0) return;

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;

    /* Per-thread scratch sizes:
     *   S_sub: max_n_pao_a * max_n_pao_b doubles
     *   T:     max_n_pao_a * max_n_pno_b doubles
     */
    const size_t S_sub_max = (size_t)max_n_pao_a * (size_t)max_n_pao_b;
    const size_t T_max     = (size_t)max_n_pao_a * (size_t)max_n_pno_b;

#pragma omp parallel num_threads(n_threads)
    {
        double *S_sub = (double *)malloc(sizeof(double) * S_sub_max);
        double *T     = (double *)malloc(sizeof(double) * T_max);

#pragma omp for schedule(dynamic, 4)
        for (long c = 0; c < n_canon_il; c++) {
            const int n_pao_a = canon_n_pao_a[c];
            const int n_pno_a = canon_n_pno_a[c];
            if (n_pao_a <= 0 || n_pno_a <= 0) continue;
            const long *pp_a = canon_pp_a_flat + canon_pp_a_off[c];
            const double *X_a = canon_X_a_flat + canon_X_a_off[c];

            const long p_start = canon_partner_start[c];
            const long p_end   = canon_partner_start[c + 1];

            for (long p = p_start; p < p_end; p++) {
                const int n_pao_b = partner_n_pao[p];
                const int n_pno_b = partner_n_pno[p];
                if (n_pao_b <= 0 || n_pno_b <= 0) continue;

                const long *pp_b = partner_pp_flat + partner_pp_off[p];
                const double *X_b = partner_X_flat + partner_X_off[p];
                double *S_out = S_side_buf + partner_S_out_off[p];

                /* Gather S_pao_sub[u, v] = S_pao_full[pp_a[u], pp_b[v]] */
                for (int u = 0; u < n_pao_a; u++) {
                    const double *S_row = S_pao_full + (size_t)pp_a[u] * n_pao_total;
                    double *S_sub_u = S_sub + (size_t)u * n_pao_b;
                    for (int v = 0; v < n_pao_b; v++) {
                        S_sub_u[v] = S_row[pp_b[v]];
                    }
                }

                /* T = S_sub @ X_b   (n_pao_a, n_pno_b) row-major
                 * dgemm('N','N', n_pno_b, n_pao_a, n_pao_b,
                 *       1, X_b, n_pno_b, S_sub, n_pao_b, 0, T, n_pno_b) */
                int int_n_pao_a = n_pao_a;
                int int_n_pao_b = n_pao_b;
                int int_n_pno_a = n_pno_a;
                int int_n_pno_b = n_pno_b;
                dgemm_(&N_flag, &N_flag,
                       &int_n_pno_b, &int_n_pao_a, &int_n_pao_b,
                       &one, X_b, &int_n_pno_b,
                       S_sub, &int_n_pao_b,
                       &zero, T, &int_n_pno_b);

                /* S_out = X_a^T @ T   (n_pno_a, n_pno_b) row-major
                 * dgemm('N','T', n_pno_b, n_pno_a, n_pao_a,
                 *       1, T, n_pno_b, X_a, n_pno_a, 0, S_out, n_pno_b) */
                dgemm_(&N_flag, &T_flag,
                       &int_n_pno_b, &int_n_pno_a, &int_n_pao_a,
                       &one, T, &int_n_pno_b,
                       X_a, &int_n_pno_a,
                       &zero, S_out, &int_n_pno_b);
            }
        }

        free(S_sub);
        free(T);
    }
}
