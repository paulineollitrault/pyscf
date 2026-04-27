/* DLPNO-CCSD compute_C_tilde / build_D_tilde Phase 2 (Terms 3 + 4):
 * per-item batched t3 and t4 kernels.
 *
 * Full-C cycle session 13 port. Mirrors the existing Cython kernels
 * _t34_batched_cy.pyx::t3_kernel_batched / t4_kernel_batched
 * byte-for-byte.
 *
 * Per-item math (one item per (ij, l) reduction step in C_tilde / D_tilde):
 *
 *   t3:
 *     Kt1[a]      = sum_b K[b, a] * t1i[b]                  (n_kl,)
 *     Kt1_ki[a]   = sum_b S[a, b] * Kt1[b]                  (n_ki,)
 *     contrib[a, c] = -T1l[a] * Kt1_ki[c]                   (n_ki, n_ki)
 *
 *   t4:
 *     tmp1[a, b]    = sum_c S_ki_li[a, c] * t2[c, b]        (n_ki, n_li)
 *     tmp2[a, b]    = sum_c tmp1[a, c]    * S_li_kl[c, b]   (n_ki, n_kl)
 *     tmp3[a, b]    = sum_c tmp2[a, c]    * K[c, b]         (n_ki, n_kl)
 *     contrib[a, b] = scale * sum_c tmp3[a, c] * S_kl_ki[c, b]  (n_ki, n_ki)
 *
 * Both kernels: outer #pragma omp parallel for over n;
 * num_threads-bounded; per-thread scratch from caller; per-item tiles
 * into tiles_flat. Caller scatters race-free into output buffers.
 */

#include <stddef.h>

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

        /* Kt1[a] = sum_b K[b, a] * t1i[b]   (= K.T @ t1i) */
        for (int a = 0; a < n_kl; a++) {
            double s = 0.0;
            for (int b = 0; b < n_kl; b++) {
                s += K[b * n_kl + a] * t1i[b];
            }
            Kt1[a] = s;
        }

        /* Kt1_ki[a] = sum_b S[a, b] * Kt1[b] */
        for (int a = 0; a < n_ki; a++) {
            double s = 0.0;
            for (int b = 0; b < n_kl; b++) {
                s += S[a * n_kl + b] * Kt1[b];
            }
            Kt1_ki[a] = s;
        }

        /* contrib[a, c] = -T1l[a] * Kt1_ki[c]  (rank-1) */
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

        /* tmp1[a, b] = sum_c S_ki_li[a, c] * t2[c, b]   (n_ki, n_li) */
        for (int a = 0; a < n_ki; a++) {
            for (int b = 0; b < n_li; b++) {
                double s = 0.0;
                for (int c = 0; c < n_li; c++) {
                    s += S_ki_li[a * n_li + c] * t2[c * n_li + b];
                }
                tmp1[a * n_li + b] = s;
            }
        }

        /* tmp2[a, b] = sum_c tmp1[a, c] * S_li_kl[c, b]   (n_ki, n_kl) */
        for (int a = 0; a < n_ki; a++) {
            for (int b = 0; b < n_kl; b++) {
                double s = 0.0;
                for (int c = 0; c < n_li; c++) {
                    s += tmp1[a * n_li + c] * S_li_kl[c * n_kl + b];
                }
                tmp2[a * n_kl + b] = s;
            }
        }

        /* tmp3[a, b] = sum_c tmp2[a, c] * K[c, b]   (n_ki, n_kl) */
        for (int a = 0; a < n_ki; a++) {
            for (int b = 0; b < n_kl; b++) {
                double s = 0.0;
                for (int c = 0; c < n_kl; c++) {
                    s += tmp2[a * n_kl + c] * K[c * n_kl + b];
                }
                tmp3[a * n_kl + b] = s;
            }
        }

        /* contrib[a, b] = scale * sum_c tmp3[a, c] * S_kl_ki[c, b]   (n_ki, n_ki) */
        for (int a = 0; a < n_ki; a++) {
            for (int b = 0; b < n_ki; b++) {
                double s = 0.0;
                for (int c = 0; c < n_kl; c++) {
                    s += tmp3[a * n_kl + c] * S_kl_ki[c * n_ki + b];
                }
                contrib[a * n_ki + b] = scale * s;
            }
        }
    }
}
