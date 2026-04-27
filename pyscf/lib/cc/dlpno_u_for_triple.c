/* DLPNO-(T): per-triple batched U projection (X_pno.T @ W_pao_tno[pp]).
 *
 * Replaces the per-pair `_U_for` Python closure inside
 * lccsd_t.py:_process_one_triple. Per triple, ~60 unique pair keys
 * each contribute one U matrix of shape (n_pno_pk, n_tno). cProfile
 * showed _U_for at 10.7s tottime (the largest single hotspot in (T)
 * on water-10).
 *
 * Per-pair math (matches _U_for's PAO-basis path):
 *   U[a, t] = sum_u X_pno[u, a] * W_pao_tno[pp[u], t]     (PAO-basis path)
 *
 * Two-step:
 *   1. Gather W_sub[u, t] = W_pao_tno[pp[u], t]  (n_pao_p, n_tno)
 *   2. U = X_pno.T @ W_sub  (n_pno_p, n_tno)
 *
 * Outer parallel over pairs via #pragma omp parallel for. BLAS via
 * vhf/fblas.h. Outer parallelism over pairs in this kernel, NOT over
 * triples — the (T) driver fans out per-triple work via its own pool,
 * so this kernel is called once per triple worker on a small batch
 * of ~60 pairs. We use num_threads=1 here to avoid oversubscription.
 *
 * Flat buffer layout per kernel call (per triple):
 *   pp_off:        (n_pairs+1,) into pp_flat
 *   pp_flat:       concat of pair_paos arrays (int64 global PAO ids)
 *   X_off:         (n_pairs+1,) into X_flat
 *   X_flat:        concat of X_pno_pk row-major (each n_pao_p × n_pno_p)
 *   U_off:         (n_pairs+1,) into U_flat
 *   U_flat:        output, concat of U_pk row-major (each n_pno_p × n_tno)
 */

#include <stddef.h>
#include <stdlib.h>
#include "vhf/fblas.h"

void DLPNObuild_U_for_triple(
        const int     n_pairs,
        const double *W_pao_tno,
        const int     n_tno,
        const int     nao_pao_total,
        const int    *n_pao_arr,
        const int    *n_pno_arr,
        const long   *pp_off,
        const long   *pp_flat,
        const long   *X_off,
        const double *X_flat,
        const long   *U_off,
        double       *U_flat)
{
    if (n_pairs <= 0) return;

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;

    for (int p = 0; p < n_pairs; p++) {
        const int n_pao = n_pao_arr[p];
        const int n_pno = n_pno_arr[p];
        if (n_pao <= 0 || n_pno <= 0) continue;

        const long *pp = pp_flat + pp_off[p];
        const double *X = X_flat + X_off[p];   /* (n_pao, n_pno) row-major */
        double *U = U_flat + U_off[p];         /* (n_pno, n_tno) row-major */

        /* Gather W_sub (n_pao, n_tno): W_sub[u, t] = W_pao_tno[pp[u], t] */
        double *W_sub = (double *)malloc(
            sizeof(double) * (size_t)n_pao * (size_t)n_tno);
        for (int u = 0; u < n_pao; u++) {
            const long row = pp[u];
            const double *src = W_pao_tno + row * n_tno;
            double *dst = W_sub + (size_t)u * n_tno;
            for (int t = 0; t < n_tno; t++) {
                dst[t] = src[t];
            }
        }

        /* U (n_pno, n_tno) = X.T (n_pno, n_pao) @ W_sub (n_pao, n_tno).
         * Row-major:
         *   U[a, t] = sum_u X[u, a] * W_sub[u, t]
         * Col-major dgemm: dgemm('N', 'T', n_tno, n_pno, n_pao,
         *                       1, W_sub_buf, n_tno, X_buf, n_pno,
         *                       0, U_buf, n_tno)
         */
        int int_n_tno = n_tno;
        int int_n_pno = n_pno;
        int int_n_pao = n_pao;
        dgemm_(&N_flag, &T_flag,
               &int_n_tno, &int_n_pno, &int_n_pao,
               &one, W_sub, &int_n_tno,
               X, &int_n_pno,
               &zero, U, &int_n_tno);

        free(W_sub);
    }
}
