/* DLPNO-CCSD T1 residual: per-(k,l) batched B + A2 kernel.
 *
 * Full-C cycle session 8 port. Mirrors the existing Cython kernel
 * _per_kl_batched_cy.pyx::per_kl_batched line-for-line; same Psi4
 * ccsd.cc:2055-2160 reference (per-(k,l) reduction over canonical
 * pairs feeding R1[i] for i in pair_lmo_idx[(k,l)]).
 *
 * Per-task math (one task = one ordered pair (k, l) with its inner
 * i-list):
 *
 *   T_n_kl  = t1_cache[key_kl]                           (M, n_kl)
 *   Tt_kl   = 2*T2[key_kl] - T2[key_kl].T  (or swapped)  (n_kl, n_kl)
 *   K_kilc  = K_bar_kl + T_n_kl @ K_iajb_kl              (M, n_kl)
 *   B_ia    = Tt_kl @ K_kilc.T                            (n_kl, M)
 *
 *   For each inner i ∈ pair_lmo_idx[key_kl]:
 *     # B contribution
 *     contrib[a] = -B_ia[a, i]                  if key_kl == (i,i)
 *                 = -sum_c S_ii_kl[a, c] * B_ia[c, i]    otherwise
 *     # A2 contribution (if has_A2):
 *     scalar = sum_{a,b} K_iajb_kl[a, b] * Tt_ki[a, b]   if diag
 *           = sum_{c,b} Tt_ki[c, b] * Z[c, b]            else
 *       Z = (S_kl_ki.T @ K_iajb_kl) @ S_ki_kl.T
 *     contrib[a] -= scalar * T_n_l_ii[a]
 *
 * Outer #pragma omp parallel for over n_tasks (num_threads-bounded so
 * caller-allocated per-thread scratch is in-bounds). Per-task work is
 * serial; matrices are small (n_kl ~25, M ~16) so hand-rolled loops
 * vectorise well at -O3 and beat per-call BLAS dispatch.
 */

#include <stddef.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void DLPNOper_kl_batched(
        const int     n_tasks,
        const int     M,                 /* nocc */

        /* Per-task metadata */
        const int    *n_kl_arr,
        const int    *t2_swap_kl,
        const long   *K_iajb_kl_off,
        const long   *K_bar_kl_off,
        const long   *t2_kl_canon_off,
        const long   *T_n_kl_off,
        const long   *inner_off,         /* (n_tasks+1,) */

        /* Per-(task, inner_i) metadata */
        const int    *i_arr,
        const int    *n_pno_ii_arr,
        const int    *is_diag_kl_ii,
        const int    *has_S_ii_kl,
        const long   *S_ii_kl_off,
        const int    *has_A2,
        const int    *is_diag_kl_ki,
        const int    *n_ki_arr,
        const int    *t2_swap_ki,
        const long   *t2_ki_canon_off,
        const long   *S_kl_ki_off,
        const long   *S_ki_kl_off,
        const long   *T_n_l_ii_off,
        const long   *contrib_off,

        /* Static flat buffers */
        const double *K_iajb_buffer,
        const double *K_bar_kl_static,
        const double *S_pno_buffer,

        /* Dynamic flat buffers */
        const double *t2_buffer,
        const double *t1_cache_buffer,

        /* Per-thread scratch (caller pre-allocated, sized for max shape) */
        double       *Tt_kl_scratch,
        const size_t  Tt_kl_stride,
        double       *K_kilc_scratch,
        const size_t  K_kilc_stride,
        double       *B_ia_scratch,
        const size_t  B_ia_stride,
        double       *Tt_ki_scratch,
        const size_t  Tt_ki_stride,
        double       *X_scratch,
        const size_t  X_stride,
        double       *Z_scratch,
        const size_t  Z_stride,

        /* Output */
        double       *contrib_flat,

        const int     num_threads)
{
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int t = 0; t < n_tasks; t++) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const int n_kl = n_kl_arr[t];

        const double *t2_kl    = t2_buffer + t2_kl_canon_off[t];
        double       *Tt_kl    = Tt_kl_scratch + (size_t)tid * Tt_kl_stride;
        const double *T_n_kl   = t1_cache_buffer + T_n_kl_off[t];
        const double *K_iajb   = K_iajb_buffer + K_iajb_kl_off[t];
        const double *K_bar_kl = K_bar_kl_static + K_bar_kl_off[t];
        double       *K_kilc   = K_kilc_scratch + (size_t)tid * K_kilc_stride;
        double       *B_ia     = B_ia_scratch + (size_t)tid * B_ia_stride;

        /* Build Tt_kl */
        if (t2_swap_kl[t]) {
            for (int a = 0; a < n_kl; a++) {
                for (int b = 0; b < n_kl; b++) {
                    Tt_kl[a * n_kl + b] =
                        2.0 * t2_kl[b * n_kl + a] - t2_kl[a * n_kl + b];
                }
            }
        } else {
            for (int a = 0; a < n_kl; a++) {
                for (int b = 0; b < n_kl; b++) {
                    Tt_kl[a * n_kl + b] =
                        2.0 * t2_kl[a * n_kl + b] - t2_kl[b * n_kl + a];
                }
            }
        }

        /* K_kilc[m, c] = K_bar_kl[m, c] + sum_d T_n_kl[m, d] * K_iajb[d, c] */
        for (int m = 0; m < M; m++) {
            for (int c = 0; c < n_kl; c++) {
                double s = K_bar_kl[m * n_kl + c];
                for (int d = 0; d < n_kl; d++) {
                    s += T_n_kl[m * n_kl + d] * K_iajb[d * n_kl + c];
                }
                K_kilc[m * n_kl + c] = s;
            }
        }

        /* B_ia[c, m] = sum_d Tt_kl[c, d] * K_kilc[m, d] */
        for (int c = 0; c < n_kl; c++) {
            for (int m = 0; m < M; m++) {
                double s = 0.0;
                for (int d = 0; d < n_kl; d++) {
                    s += Tt_kl[c * n_kl + d] * K_kilc[m * n_kl + d];
                }
                B_ia[c * M + m] = s;
            }
        }

        /* Inner per-i loop */
        const long ti_start = inner_off[t];
        const long ti_end   = inner_off[t + 1];
        for (long ti = ti_start; ti < ti_end; ti++) {
            const int i_val    = i_arr[ti];
            const int n_pno_ii = n_pno_ii_arr[ti];
            double *contrib    = contrib_flat + contrib_off[ti];

            /* Zero contrib */
            memset(contrib, 0, sizeof(double) * (size_t)n_pno_ii);

            /* B contribution */
            if (is_diag_kl_ii[ti]) {
                /* n_pno_ii == n_kl in this branch */
                for (int a = 0; a < n_pno_ii; a++) {
                    contrib[a] = -B_ia[a * M + i_val];
                }
            } else if (has_S_ii_kl[ti]) {
                const double *S_ii_kl = S_pno_buffer + S_ii_kl_off[ti];
                for (int a = 0; a < n_pno_ii; a++) {
                    double s = 0.0;
                    for (int c = 0; c < n_kl; c++) {
                        s += S_ii_kl[a * n_kl + c] * B_ia[c * M + i_val];
                    }
                    contrib[a] = -s;
                }
            }

            /* A2 contribution */
            if (!has_A2[ti]) continue;

            const int n_ki = n_ki_arr[ti];
            const double *T_n_l_ii = t1_cache_buffer + T_n_l_ii_off[ti];

            const double *t2_ki = t2_buffer + t2_ki_canon_off[ti];
            double       *Tt_ki = Tt_ki_scratch + (size_t)tid * Tt_ki_stride;

            if (t2_swap_ki[ti]) {
                for (int a = 0; a < n_ki; a++) {
                    for (int b = 0; b < n_ki; b++) {
                        Tt_ki[a * n_ki + b] =
                            2.0 * t2_ki[b * n_ki + a] - t2_ki[a * n_ki + b];
                    }
                }
            } else {
                for (int a = 0; a < n_ki; a++) {
                    for (int b = 0; b < n_ki; b++) {
                        Tt_ki[a * n_ki + b] =
                            2.0 * t2_ki[a * n_ki + b] - t2_ki[b * n_ki + a];
                    }
                }
            }

            double scalar;
            if (is_diag_kl_ki[ti]) {
                /* n_ki == n_kl; scalar = sum_{a,b} K_iajb[a, b] * Tt_ki[a, b] */
                scalar = 0.0;
                for (int a = 0; a < n_kl; a++) {
                    for (int b = 0; b < n_kl; b++) {
                        scalar += K_iajb[a * n_kl + b] * Tt_ki[a * n_kl + b];
                    }
                }
            } else {
                const double *S_kl_ki = S_pno_buffer + S_kl_ki_off[ti];
                const double *S_ki_kl = S_pno_buffer + S_ki_kl_off[ti];
                double *X = X_scratch + (size_t)tid * X_stride;
                double *Z = Z_scratch + (size_t)tid * Z_stride;

                /* X[c, d] = sum_a S_kl_ki[a, c] * K_iajb[a, d]   (n_ki, n_kl) */
                for (int c = 0; c < n_ki; c++) {
                    for (int d = 0; d < n_kl; d++) {
                        double s = 0.0;
                        for (int a = 0; a < n_kl; a++) {
                            s += S_kl_ki[a * n_ki + c] * K_iajb[a * n_kl + d];
                        }
                        X[c * n_kl + d] = s;
                    }
                }

                /* Z[c, b] = sum_d X[c, d] * S_ki_kl[b, d]    (n_ki, n_ki) */
                for (int c = 0; c < n_ki; c++) {
                    for (int b = 0; b < n_ki; b++) {
                        double s = 0.0;
                        for (int d = 0; d < n_kl; d++) {
                            s += X[c * n_kl + d] * S_ki_kl[b * n_kl + d];
                        }
                        Z[c * n_ki + b] = s;
                    }
                }

                /* scalar = sum_{c, b} Tt_ki[c, b] * Z[c, b] */
                scalar = 0.0;
                for (int c = 0; c < n_ki; c++) {
                    for (int b = 0; b < n_ki; b++) {
                        scalar += Tt_ki[c * n_ki + b] * Z[c * n_ki + b];
                    }
                }
            }

            /* contrib[a] -= scalar * T_n_l_ii[a] */
            for (int a = 0; a < n_pno_ii; a++) {
                contrib[a] -= scalar * T_n_l_ii[a];
            }
        }
    }
}
