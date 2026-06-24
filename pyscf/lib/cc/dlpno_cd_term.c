/* DLPNO-CCSD T2 residual: per-pair C-term and D-term batched kernels.
 *
 * BLAS port (2026-04-30): per-item triple loops -> DGEMM.  At MKL-link
 * with JIT-GEMM, ~3x faster than hand-rolled at npno~25.
 *
 * c_term math (per item):
 *   STB[a, c]   = sum_b S_big[a, b] * ct[b, c]                (DGEMM)
 *   gamma[a, d] = J_bold[a, d] + sum_c STB[a, c] * S_mid[c, d] (DGEMM, beta=1)
 *   GT[a, e]    = sum_d gamma[a, d] * t2[e, d]                (DGEMM @ t2^T)
 *   Cc[a, f]    = sum_e GT[a, e] * S_outer[f, e]              (DGEMM @ S_outer^T)
 *
 * d_term math (per item):
 *   SU[a, d]   = sum_b S_a[a, b] * u[b, d]                    (DGEMM)
 *   UP[a, c]   = sum_d SU[a, d] * S_b[d, c]                   (DGEMM)
 *   SCD[a, c]  = sum_b S_c[a, b] * dt[b, c]                   (DGEMM)
 *   Bint[a, d] = sum_b KJ[a, b] * u[d, b]                     (DGEMM @ u^T)
 *   Dtile      = SCD @ UP^T + Bint @ S_a^T                    (2 DGEMMs)
 */

#include <stddef.h>
#include <string.h>
#include "vhf/fblas.h"

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
                         /* t2 offset-alias: when t2_trans != NULL, t2_flat is
                          * the CANONICAL per-pair t2 buffer and t2_off[n] is
                          * the canonical offset; the per-item transpose is
                          * applied by flipping the GT dgemm flag instead of
                          * materialising a gathered (duplicated) t2 copy. */
                         const unsigned char *t2_trans,
                         double       *STB_scratch,
                         const size_t  STB_stride,
                         double       *GAMMA_scratch,
                         const size_t  GAMMA_stride,
                         double       *GT_scratch,
                         const size_t  GT_stride,
                         double       *tiles_flat,
                         const int     num_threads)
{
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;

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

        int int_n_pno = n_pno, int_n_ct = n_ct, int_n_other = n_other;

        /* STB = S_big @ ct   (n_pno, n_ct) = (n_pno, n_ct) @ (n_ct, n_ct)
         * F: STB_F[c, a] = sum_b ct_F[c, b] * S_big_F[b, a]  =  ct_F @ S_big_F
         * dgemm('N', 'N', n_ct, n_pno, n_ct, 1, ct, n_ct, S_big, n_ct, 0, STB, n_ct)
         */
        dgemm_(&N_flag, &N_flag,
               &int_n_ct, &int_n_pno, &int_n_ct,
               &one, ct, &int_n_ct,
               S_big, &int_n_ct,
               &zero, STB, &int_n_ct);

        /* gamma = J_bold + STB @ S_mid   (n_pno, n_other) accum
         * F: gamma_F[d, a] = J_bold_F[d, a] + sum_c S_mid_F[d, c] * STB_F[c, a]
         * Init gamma <- J_bold then dgemm beta=1.
         */
        memcpy(GAMMA, J_bold, sizeof(double) * (size_t)n_pno * n_other);
        dgemm_(&N_flag, &N_flag,
               &int_n_other, &int_n_pno, &int_n_ct,
               &one, S_mid, &int_n_other,
               STB, &int_n_ct,
               &one, GAMMA, &int_n_other);

        /* GT[a, e] = sum_d gamma[a, d] * t2[e, d]   (n_pno, n_other)
         * GT = gamma @ t2^T.
         * F: GT_F[e, a] = sum_d t2_F[d, e] * gamma_F[d, a] = t2_F^T @ gamma_F
         * dgemm('T', 'N', n_other, n_pno, n_other, 1, t2, n_other, gamma, n_other, 0, GT, n_other)
         *
         * Offset-aliased t2 (t2_trans != NULL): t2 is the canonical t2[key];
         * an item flagged transpose wants t2[key].T here, which is exactly
         * t2 read with the OPPOSITE dgemm flag.  So gathered-non-transpose
         * and aliased-transpose both reduce to flipping T<->N: legacy gather
         * stored t2 (flag T) or t2.T (flag T on the pre-transposed copy);
         * aliased reads t2 always and uses flag N when the item is transpose.
         */
        const char t2_flag = (t2_trans != NULL && t2_trans[n]) ? N_flag : T_flag;
        dgemm_(&t2_flag, &N_flag,
               &int_n_other, &int_n_pno, &int_n_other,
               &one, t2, &int_n_other,
               GAMMA, &int_n_other,
               &zero, GT, &int_n_other);

        /* Cc[a, f] = sum_e GT[a, e] * S_outer[f, e]   (n_pno, n_pno)
         * Cc = GT @ S_outer^T.
         * F: Cc_F[f, a] = sum_e S_outer_F[e, f] * GT_F[e, a] = S_outer_F^T @ GT_F
         * dgemm('T', 'N', n_pno, n_pno, n_other, 1, S_outer, n_other, GT, n_other, 0, Cc, n_pno)
         */
        dgemm_(&T_flag, &N_flag,
               &int_n_pno, &int_n_pno, &int_n_other,
               &one, S_outer, &int_n_other,
               GT, &int_n_other,
               &zero, Cc, &int_n_pno);
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
                         /* u offset-alias: when u_base != NULL, u is computed
                          * per item as u = 2*t2_d - t2_d^T from the CANONICAL
                          * t2[key] at u_base + u_canon_off[n] (t2_d = t2[key]
                          * transposed iff u_trans[n]) into U_scratch, instead
                          * of materialising a gathered (duplicated) u copy. */
                         const double *u_base,
                         const long   *u_canon_off,
                         const unsigned char *u_trans,
                         double       *U_scratch,
                         const size_t  U_stride,
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
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;

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
        const double *u;
        if (u_base != NULL) {
            /* Compute u = 2*t2_d - t2_d^T into per-thread scratch, where
             * t2_d = t2[key] (transposed iff u_trans[n]).  u is n_A x n_A
             * row-major (same layout the gather produced). */
            const double *t2s = u_base + u_canon_off[n];
            double *U = U_scratch + (size_t)tid * U_stride;
            const int nA = n_A;
            const int tr = (u_trans != NULL && u_trans[n]);
            for (int a = 0; a < nA; ++a) {
                for (int b = 0; b < nA; ++b) {
                    const double t2d_ab = tr ? t2s[(size_t)b * nA + a]
                                             : t2s[(size_t)a * nA + b];
                    const double t2d_ba = tr ? t2s[(size_t)a * nA + b]
                                             : t2s[(size_t)b * nA + a];
                    U[(size_t)a * nA + b] = 2.0 * t2d_ab - t2d_ba;
                }
            }
            u = U;
        } else {
            u = u_flat + u_off[n];
        }
        const double *S_b = S_b_flat + S_b_off[n];
        const double *S_c = S_c_flat + S_c_off[n];
        const double *dt  = dt_flat  + dt_off[n];
        const double *KJ  = KJ_flat  + KJ_off[n];

        double *SU    = SU_scratch    + (size_t)tid * SU_stride;
        double *UP    = UP_scratch    + (size_t)tid * UP_stride;
        double *SCD   = SCD_scratch   + (size_t)tid * SCD_stride;
        double *Bint  = Bint_scratch  + (size_t)tid * Bint_stride;
        double *Dtile = tiles_flat    + tile_off[n];

        int int_n_pno = n_pno, int_n_A = n_A, int_n_B = n_B;

        /* SU = S_a @ u   (n_pno, n_A) = (n_pno, n_A) @ (n_A, n_A)
         * F: SU_F[d, a] = sum_b u_F[d, b] * S_a_F[b, a] = u_F @ S_a_F
         * dgemm('N', 'N', n_A, n_pno, n_A, 1, u, n_A, S_a, n_A, 0, SU, n_A)
         */
        dgemm_(&N_flag, &N_flag,
               &int_n_A, &int_n_pno, &int_n_A,
               &one, u, &int_n_A,
               S_a, &int_n_A,
               &zero, SU, &int_n_A);

        /* UP = SU @ S_b   (n_pno, n_B) = (n_pno, n_A) @ (n_A, n_B)
         * F: UP_F[c, a] = sum_d S_b_F[c, d] * SU_F[d, a] = S_b_F @ SU_F
         * dgemm('N', 'N', n_B, n_pno, n_A, 1, S_b, n_B, SU, n_A, 0, UP, n_B)
         */
        dgemm_(&N_flag, &N_flag,
               &int_n_B, &int_n_pno, &int_n_A,
               &one, S_b, &int_n_B,
               SU, &int_n_A,
               &zero, UP, &int_n_B);

        /* SCD = S_c @ dt   (n_pno, n_B) = (n_pno, n_B) @ (n_B, n_B)
         * F: SCD_F[c, a] = sum_b dt_F[c, b] * S_c_F[b, a] = dt_F @ S_c_F
         * dgemm('N', 'N', n_B, n_pno, n_B, 1, dt, n_B, S_c, n_B, 0, SCD, n_B)
         */
        dgemm_(&N_flag, &N_flag,
               &int_n_B, &int_n_pno, &int_n_B,
               &one, dt, &int_n_B,
               S_c, &int_n_B,
               &zero, SCD, &int_n_B);

        /* Bint[a, d] = sum_b KJ[a, b] * u[d, b]   (n_pno, n_A)
         * Bint = KJ @ u^T.
         * F: Bint_F[d, a] = sum_b u_F[b, d] * KJ_F[b, a] = u_F^T @ KJ_F
         * dgemm('T', 'N', n_A, n_pno, n_A, 1, u, n_A, KJ, n_A, 0, Bint, n_A)
         */
        dgemm_(&T_flag, &N_flag,
               &int_n_A, &int_n_pno, &int_n_A,
               &one, u, &int_n_A,
               KJ, &int_n_A,
               &zero, Bint, &int_n_A);

        /* Dtile[a, f] = sum_c SCD[a, c] * UP[f, c] + sum_d Bint[a, d] * S_a[f, d]
         *            = SCD @ UP^T + Bint @ S_a^T
         * Build first term: F: Dtile_F[f, a] = sum_c UP_F[c, f] * SCD_F[c, a]
         *   = UP_F^T @ SCD_F.  dgemm('T', 'N', n_pno, n_pno, n_B, 1, UP, n_B, SCD, n_B, 0, Dtile, n_pno)
         * Add second term:    F: += sum_d S_a_F[d, f] * Bint_F[d, a] = S_a_F^T @ Bint_F
         *   dgemm('T', 'N', n_pno, n_pno, n_A, 1, S_a, n_A, Bint, n_A, 1, Dtile, n_pno)
         */
        dgemm_(&T_flag, &N_flag,
               &int_n_pno, &int_n_pno, &int_n_B,
               &one, UP, &int_n_B,
               SCD, &int_n_B,
               &zero, Dtile, &int_n_pno);
        dgemm_(&T_flag, &N_flag,
               &int_n_pno, &int_n_pno, &int_n_A,
               &one, S_a, &int_n_A,
               Bint, &int_n_A,
               &one, Dtile, &int_n_pno);
    }
}
