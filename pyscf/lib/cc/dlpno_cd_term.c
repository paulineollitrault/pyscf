/* DLPNO-CCSD T2 residual: per-item C-term and D-term batched kernels.
 *
 * Full-C cycle session 12 port. Mirrors the existing Cython kernels
 * _cd_batched_cy.pyx::c_kernel_batched / d_kernel_batched
 * byte-for-byte. Replaces ~4224 per-bucket numpy.matmul calls/cycle
 * with two single nogil prange calls processing ~11500 items each.
 *
 * Both kernels follow the same outer pattern: outer #pragma omp parallel
 * for over n; per-thread scratch passed in by caller. Per-item Cc/Dtile
 * tiles written into tiles_flat at tile_off[n]; caller scatters
 * race-free into the per-n_ij output buffers.
 *
 * C-term per-item math:
 *   STB[a, c]   = sum_b S_big[a, b] * ct[b, c]            (n_pno, n_ct)
 *   gamma[a, d] = J_bold[a, d] + sum_c STB[a, c] * S_mid[c, d]
 *   GT[a, e]    = sum_d gamma[a, d] * t2[e, d]
 *   Cc[a, f]    = sum_e GT[a, e] * S_outer[f, e]
 *
 * D-term per-item math:
 *   SU[a, d]   = sum_b S_a[a, b] * u[b, d]                 (n_pno, n_A)
 *   UP[a, c]   = sum_d SU[a, d] * S_b[d, c]                (n_pno, n_B)
 *   SCD[a, c]  = sum_b S_c[a, b] * dt[b, c]                (n_pno, n_B)
 *   Bint[a, d] = sum_b KJ[a, b] * u[d, b]                  (n_pno, n_A)
 *   Dtile[a, f] = sum_c SCD[a, c] * UP[f, c] + sum_d Bint[a, d] * S_a[f, d]
 */

#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void DLPNOc_term_batched(const int     N,
                         const int    *n_pno_arr,
                         const int    *n_ct_arr,
                         const int    *n_other_arr,
                         const long   *S_big_off,
                         const long   *ct_off,
                         const long   *S_mid_off,
                         const long   *J_bold_off,
                         const long   *t2_off,
                         const long   *S_outer_off,
                         const long   *tile_off,
                         const double *S_big_flat,
                         const double *S_mid_flat,
                         const double *J_bold_flat,
                         const double *S_outer_flat,
                         const double *ct_flat,
                         const double *t2_flat,
                         double       *STB_scratch,
                         const size_t  STB_stride,
                         double       *GAMMA_scratch,
                         const size_t  GAMMA_stride,
                         double       *GT_scratch,
                         const size_t  GT_stride,
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
        const int n_pno   = n_pno_arr[n];
        const int n_ct    = n_ct_arr[n];
        const int n_other = n_other_arr[n];

        const double *S_big   = S_big_flat   + S_big_off[n];
        const double *ct      = ct_flat      + ct_off[n];
        const double *S_mid   = S_mid_flat   + S_mid_off[n];
        const double *J_bold  = J_bold_flat  + J_bold_off[n];
        const double *t2      = t2_flat      + t2_off[n];
        const double *S_outer = S_outer_flat + S_outer_off[n];

        double *STB   = STB_scratch   + (size_t)tid * STB_stride;
        double *GAMMA = GAMMA_scratch + (size_t)tid * GAMMA_stride;
        double *GT    = GT_scratch    + (size_t)tid * GT_stride;
        double *Cc    = tiles_flat    + tile_off[n];

        /* STB[a, c] = sum_b S_big[a, b] * ct[b, c] */
        for (int a = 0; a < n_pno; a++) {
            for (int c = 0; c < n_ct; c++) {
                double s = 0.0;
                for (int b = 0; b < n_ct; b++) {
                    s += S_big[a * n_ct + b] * ct[b * n_ct + c];
                }
                STB[a * n_ct + c] = s;
            }
        }

        /* gamma[a, d] = J_bold[a, d] + sum_c STB[a, c] * S_mid[c, d] */
        for (int a = 0; a < n_pno; a++) {
            for (int d = 0; d < n_other; d++) {
                double s = J_bold[a * n_other + d];
                for (int c = 0; c < n_ct; c++) {
                    s += STB[a * n_ct + c] * S_mid[c * n_other + d];
                }
                GAMMA[a * n_other + d] = s;
            }
        }

        /* GT[a, e] = sum_d gamma[a, d] * t2[e, d] */
        for (int a = 0; a < n_pno; a++) {
            for (int e = 0; e < n_other; e++) {
                double s = 0.0;
                for (int d = 0; d < n_other; d++) {
                    s += GAMMA[a * n_other + d] * t2[e * n_other + d];
                }
                GT[a * n_other + e] = s;
            }
        }

        /* Cc[a, f] = sum_e GT[a, e] * S_outer[f, e] */
        for (int a = 0; a < n_pno; a++) {
            for (int f = 0; f < n_pno; f++) {
                double s = 0.0;
                for (int e = 0; e < n_other; e++) {
                    s += GT[a * n_other + e] * S_outer[f * n_other + e];
                }
                Cc[a * n_pno + f] = s;
            }
        }
    }
}

void DLPNOd_term_batched(const int     N,
                         const int    *n_pno_arr,
                         const int    *n_A_arr,
                         const int    *n_B_arr,
                         const long   *S_a_off,
                         const long   *u_off,
                         const long   *S_b_off,
                         const long   *S_c_off,
                         const long   *dt_off,
                         const long   *KJ_off,
                         const long   *tile_off,
                         const double *S_a_flat,
                         const double *S_b_flat,
                         const double *S_c_flat,
                         const double *KJ_flat,
                         const double *u_flat,
                         const double *dt_flat,
                         double       *SU_scratch,
                         const size_t  SU_stride,
                         double       *UP_scratch,
                         const size_t  UP_stride,
                         double       *SCD_scratch,
                         const size_t  SCD_stride,
                         double       *Bint_scratch,
                         const size_t  Bint_stride,
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
        const int n_pno = n_pno_arr[n];
        const int n_A   = n_A_arr[n];
        const int n_B   = n_B_arr[n];

        const double *S_a = S_a_flat + S_a_off[n];
        const double *u   = u_flat   + u_off[n];
        const double *S_b = S_b_flat + S_b_off[n];
        const double *S_c = S_c_flat + S_c_off[n];
        const double *dt  = dt_flat  + dt_off[n];
        const double *KJ  = KJ_flat  + KJ_off[n];

        double *SU    = SU_scratch    + (size_t)tid * SU_stride;
        double *UP    = UP_scratch    + (size_t)tid * UP_stride;
        double *SCD   = SCD_scratch   + (size_t)tid * SCD_stride;
        double *Bint  = Bint_scratch  + (size_t)tid * Bint_stride;
        double *Dtile = tiles_flat    + tile_off[n];

        /* SU[a, d] = sum_b S_a[a, b] * u[b, d]   (n_pno, n_A) */
        for (int a = 0; a < n_pno; a++) {
            for (int d = 0; d < n_A; d++) {
                double s = 0.0;
                for (int b = 0; b < n_A; b++) {
                    s += S_a[a * n_A + b] * u[b * n_A + d];
                }
                SU[a * n_A + d] = s;
            }
        }

        /* UP[a, c] = sum_d SU[a, d] * S_b[d, c]   (n_pno, n_B) */
        for (int a = 0; a < n_pno; a++) {
            for (int c = 0; c < n_B; c++) {
                double s = 0.0;
                for (int d = 0; d < n_A; d++) {
                    s += SU[a * n_A + d] * S_b[d * n_B + c];
                }
                UP[a * n_B + c] = s;
            }
        }

        /* SCD[a, c] = sum_b S_c[a, b] * dt[b, c]   (n_pno, n_B) */
        for (int a = 0; a < n_pno; a++) {
            for (int c = 0; c < n_B; c++) {
                double s = 0.0;
                for (int b = 0; b < n_B; b++) {
                    s += S_c[a * n_B + b] * dt[b * n_B + c];
                }
                SCD[a * n_B + c] = s;
            }
        }

        /* Bint[a, d] = sum_b KJ[a, b] * u[d, b]   (n_pno, n_A) */
        for (int a = 0; a < n_pno; a++) {
            for (int d = 0; d < n_A; d++) {
                double s = 0.0;
                for (int b = 0; b < n_A; b++) {
                    s += KJ[a * n_A + b] * u[d * n_A + b];
                }
                Bint[a * n_A + d] = s;
            }
        }

        /* Dtile[a, f] = (sum_c SCD[a, c] * UP[f, c]) + (sum_d Bint[a, d] * S_a[f, d]) */
        for (int a = 0; a < n_pno; a++) {
            for (int f = 0; f < n_pno; f++) {
                double sA = 0.0, sB = 0.0;
                for (int c = 0; c < n_B; c++) {
                    sA += SCD[a * n_B + c] * UP[f * n_B + c];
                }
                for (int d = 0; d < n_A; d++) {
                    sB += Bint[a * n_A + d] * S_a[f * n_A + d];
                }
                Dtile[a * n_pno + f] = sA + sB;
            }
        }
    }
}
