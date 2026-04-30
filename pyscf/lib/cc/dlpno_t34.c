/* DLPNO-CCSD compute_C_tilde / build_D_tilde Phase 2 (Terms 3 + 4):
 * per-item batched t3 and t4 kernels.
 *
 * BLAS port (2026-04-30): per-item triple/double loops -> DGEMV/DGEMM.
 * At npno~25 with MKL JIT-GEMM, ~3x faster than hand-rolled loops.
 *
 * Per-item math (one item per (ij, l) reduction step in C_tilde / D_tilde):
 *
 *   t3:
 *     Kt1[a]      = sum_b K[b, a] * t1i[b]              (DGEMV 'T')
 *     Kt1_ki[a]   = sum_b S[a, b] * Kt1[b]              (DGEMV 'N')
 *     contrib[a, c] = -T1l[a] * Kt1_ki[c]               (DGER outer prod)
 *
 *   t4:
 *     tmp1[a, b]    = sum_c S_ki_li[a, c] * t2[c, b]    (DGEMM)
 *     tmp2[a, b]    = sum_c tmp1[a, c] * S_li_kl[c, b]  (DGEMM)
 *     tmp3[a, b]    = sum_c tmp2[a, c] * K[c, b]        (DGEMM)
 *     contrib[a, b] = scale * sum_c tmp3[a, c] * S_kl_ki[c, b]  (DGEMM)
 */

#include <stddef.h>
#include <string.h>
#include "vhf/fblas.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void DLPNOt3_kernel_batched(const int     N,
                            const int    *n_kl_arr,
                            const int    *n_ki_arr,
                            const long   *K_off,
                            const long   *S_off,
                            const long   *t1i_off,
                            const long   *T1l_off,
                            const long   *tile_off,
                            const double *K_flat,
                            const double *S_flat,
                            const double *t1_flat,
                            double       *Kt1_scratch,
                            const size_t  Kt1_stride,
                            double       *Kt1_ki_scratch,
                            const size_t  Kt1_ki_stride,
                            double       *tiles_flat,
                            const int     num_threads)
{
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0, neg_one = -1.0;
    const int int_one = 1;

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int n = 0; n < N; n++) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const int n_kl = n_kl_arr[n];
        const int n_ki = n_ki_arr[n];

        const double *K   = K_flat   + K_off[n];   /* (n_kl, n_kl) */
        const double *S   = S_flat   + S_off[n];   /* (n_ki, n_kl) */
        const double *t1i = t1_flat  + t1i_off[n]; /* (n_kl,) */
        const double *T1l = t1_flat  + T1l_off[n]; /* (n_ki,) */
        double *Kt1     = Kt1_scratch     + (size_t)tid * Kt1_stride;
        double *Kt1_ki  = Kt1_ki_scratch  + (size_t)tid * Kt1_ki_stride;
        double *contrib = tiles_flat + tile_off[n];

        int int_n_kl = n_kl;
        int int_n_ki = n_ki;

        /* Kt1[a] = sum_b K[b, a] * t1i[b]    (= K^T @ t1i)
         * Row-major K (n_kl, n_kl), F view (n_kl, n_kl) (square).
         * dgemv('N', n_kl, n_kl, 1, K, n_kl, t1i, 1, 0, Kt1, 1)
         *   computes Kt1_F[a] = sum_b K_F[a, b] * t1i[b] = sum_b K[b, a] * t1i[b]
         */
        dgemv_(&N_flag, &int_n_kl, &int_n_kl,
               &one, K, &int_n_kl,
               t1i, &int_one,
               &zero, Kt1, &int_one);

        /* Kt1_ki[a] = sum_b S[a, b] * Kt1[b]   (= S @ Kt1)
         * S row-major (n_ki, n_kl), F view (n_kl, n_ki).
         * dgemv('T', n_kl, n_ki, 1, S, n_kl, Kt1, 1, 0, Kt1_ki, 1)
         *   computes Kt1_ki_F[a] = sum_b S_F[b, a] * Kt1[b] = sum_b S[a, b] * Kt1[b]
         */
        dgemv_(&T_flag, &int_n_kl, &int_n_ki,
               &one, S, &int_n_kl,
               Kt1, &int_one,
               &zero, Kt1_ki, &int_one);

        /* contrib[a, c] = -T1l[a] * Kt1_ki[c]
         * Outer product.  contrib row-major (n_ki, n_ki).
         * Use dgemm with M=1 hack OR just write the outer manually — since
         * memory access pattern is contiguous and predictable, hand-rolled
         * is usually faster than dger overhead at this size.
         */
        for (int a = 0; a < n_ki; a++) {
            const double v = -T1l[a];
            for (int c = 0; c < n_ki; c++) {
                contrib[a * n_ki + c] = v * Kt1_ki[c];
            }
        }
    }
}

void DLPNOt4_kernel_batched(const int     N,
                            const int    *n_ki_arr,
                            const int    *n_li_arr,
                            const int    *n_kl_arr,
                            const long   *S_ki_li_off,
                            const long   *t2_off,
                            const long   *S_li_kl_off,
                            const long   *K_off,
                            const long   *S_kl_ki_off,
                            const long   *tile_off,
                            const double *S_ki_li_flat,
                            const double *S_li_kl_flat,
                            const double *K_flat,
                            const double *S_kl_ki_flat,
                            const double *t2_flat,
                            double       *tmp1_scratch,
                            const size_t  tmp1_stride,
                            double       *tmp2_scratch,
                            const size_t  tmp2_stride,
                            double       *tmp3_scratch,
                            const size_t  tmp3_stride,
                            double       *tiles_flat,
                            const double  scale,
                            const int     num_threads)
{
    const char N_flag = 'N';
    const double one = 1.0, zero = 0.0;

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int n = 0; n < N; n++) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const int n_ki = n_ki_arr[n];
        const int n_li = n_li_arr[n];
        const int n_kl = n_kl_arr[n];

        const double *S_ki_li = S_ki_li_flat + S_ki_li_off[n]; /* (n_ki, n_li) */
        const double *t2      = t2_flat      + t2_off[n];      /* (n_li, n_li) */
        const double *S_li_kl = S_li_kl_flat + S_li_kl_off[n]; /* (n_li, n_kl) */
        const double *K       = K_flat       + K_off[n];       /* (n_kl, n_kl) */
        const double *S_kl_ki = S_kl_ki_flat + S_kl_ki_off[n]; /* (n_kl, n_ki) */

        double *tmp1    = tmp1_scratch + (size_t)tid * tmp1_stride;
        double *tmp2    = tmp2_scratch + (size_t)tid * tmp2_stride;
        double *tmp3    = tmp3_scratch + (size_t)tid * tmp3_stride;
        double *contrib = tiles_flat + tile_off[n];

        int int_n_ki = n_ki, int_n_li = n_li, int_n_kl = n_kl;

        /* tmp1 = S_ki_li @ t2   (n_ki, n_li) = (n_ki, n_li) @ (n_li, n_li)
         * F view: tmp1_F[b, a] = sum_c t2_F[b, c] * S_ki_li_F[c, a] = t2_F @ S_ki_li_F.
         * dgemm('N', 'N', n_li, n_ki, n_li, 1, t2, n_li, S_ki_li, n_li, 0, tmp1, n_li)
         */
        dgemm_(&N_flag, &N_flag,
               &int_n_li, &int_n_ki, &int_n_li,
               &one, t2, &int_n_li,
               S_ki_li, &int_n_li,
               &zero, tmp1, &int_n_li);

        /* tmp2 = tmp1 @ S_li_kl   (n_ki, n_kl) = (n_ki, n_li) @ (n_li, n_kl)
         * F view: tmp2_F[b, a] = sum_c S_li_kl_F[b, c] * tmp1_F[c, a]
         * dgemm('N', 'N', n_kl, n_ki, n_li, 1, S_li_kl, n_kl, tmp1, n_li, 0, tmp2, n_kl)
         */
        dgemm_(&N_flag, &N_flag,
               &int_n_kl, &int_n_ki, &int_n_li,
               &one, S_li_kl, &int_n_kl,
               tmp1, &int_n_li,
               &zero, tmp2, &int_n_kl);

        /* tmp3 = tmp2 @ K   (n_ki, n_kl) = (n_ki, n_kl) @ (n_kl, n_kl)
         * F view: tmp3_F[b, a] = sum_c K_F[b, c] * tmp2_F[c, a]
         * dgemm('N', 'N', n_kl, n_ki, n_kl, 1, K, n_kl, tmp2, n_kl, 0, tmp3, n_kl)
         */
        dgemm_(&N_flag, &N_flag,
               &int_n_kl, &int_n_ki, &int_n_kl,
               &one, K, &int_n_kl,
               tmp2, &int_n_kl,
               &zero, tmp3, &int_n_kl);

        /* contrib = scale * tmp3 @ S_kl_ki   (n_ki, n_ki) = (n_ki, n_kl) @ (n_kl, n_ki)
         * F view: contrib_F[b, a] = sum_c S_kl_ki_F[b, c] * tmp3_F[c, a]
         * dgemm('N', 'N', n_ki, n_ki, n_kl, scale, S_kl_ki, n_ki, tmp3, n_kl, 0, contrib, n_ki)
         */
        dgemm_(&N_flag, &N_flag,
               &int_n_ki, &int_n_ki, &int_n_kl,
               &scale, S_kl_ki, &int_n_ki,
               tmp3, &int_n_kl,
               &zero, contrib, &int_n_ki);
    }
}
