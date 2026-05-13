/* DLPNO-CCSD T1 residual: per-i Stages 1-3 (Fai_bar, Fab_bar*t1, -T_n^T*Fia_bar*t1).
 *
 * Math (M = nlmo_pair, A = n_pno, L = n_local aux):
 *
 *   Stage 1 (any_t1):
 *     gamma[L]   = sum_{m, a} Qma[L, m, a] * T_n[m, a]      (m: M, a: A)
 *     r1[a]     += 2 * sum_L Qia[L, a] * gamma[L]
 *     y[L, a]    = sum_m Qik[L, m] * T_n[m, a]
 *     r1[a]     -= sum_{L, c} Qab[L, a, c] * y[L, c]
 *
 *   Stage 2 (do_stage_23):
 *     W[L, b, m]   = sum_c Qab[L, b, c] * T_n[m, c]
 *     Fab[a, b]    = e_pno[a] * delta(a, b)
 *     Fab[a, b]   += 2 * sum_L gamma[L] * Qab[L, a, b]
 *     Fab[a, b]   -= sum_{L, m} W[L, a, m] * Qma[L, m, b]
 *     r1[a]       += sum_b Fab[a, b] * t1_i[b]
 *
 *   Stage 3 (do_stage_23):
 *     Z[L, m, m']  = sum_c Qma[L, m, c] * T_n[m', c]
 *     Fia[m', a]   = 2 * sum_L gamma[L] * Qma[L, m', a]
 *     Fia[m', a]  -= sum_{L, m} Z[L, m, m'] * Qma[L, m, a]
 *     v[m]         = sum_b Fia[m, b] * t1_i[b]
 *     r1[a]       -= sum_m T_n[m, a] * v[m]
 *
 * BLAS port (2026-05-11): hand-rolled triple loops collapsed to dgemm/dgemv
 * calls.  Per-cycle p9 wall at water-22 was the N^4.14 outlier among CCSD
 * phases.  W and Z are never materialised — for each L we run two dgemms
 * with a single (A, M) or (M, M) scratch tile, accumulating directly into
 * Fab / Fia.  Row-major math; Fortran BLAS conventions applied via the
 * standard "swap A/B and dims" workaround (see dlpno_be.c for the pattern).
 *
 * Entry shapes (M = nlmo_pair):
 *   r1_inout: (A,)        — accumulator (caller initialises before Stage 4)
 *   Qma:      (L, M, A)
 *   Qab:      (L, A, A)
 *   Qia:      (L, A)
 *   Qik:      (L, M)
 *   T_n:      (M, A)
 *   t1_i:     (A,)
 *   e_pno:    (A,)
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

void DLPNOper_i_stages123(double *r1_inout,
                          const double *Qma,
                          const double *Qab,
                          const double *Qia,
                          const double *Qik,
                          const double *T_n,
                          const double *t1_i,
                          const double *e_pno,
                          const int do_stage_23,
                          const size_t L,
                          const size_t M,
                          const size_t A)
{
    if (A == 0 || L == 0) return;

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0, neg_one = -1.0, two = 2.0;
    int i_L = (int)L, i_M = (int)M, i_A = (int)A;
    int i_MA = (int)(M * A), i_AA = (int)(A * A);
    int ione = 1;
    /* Fortran BLAS args are const; cast away for our const-qualified inputs. */
    double *Qma_  = (double *)Qma;
    double *Qab_  = (double *)Qab;
    double *Qia_  = (double *)Qia;
    double *Qik_  = (double *)Qik;
    double *T_n_  = (double *)T_n;
    double *t1_i_ = (double *)t1_i;

    /* Scratch */
    double *gamma = (double *)malloc(sizeof(double) * L);
    double *y     = (double *)malloc(sizeof(double) * L * A);
    if (gamma == NULL || y == NULL) {
        free(gamma); free(y); return;
    }

    /* ============= Stage 1 ============= */

    /* gamma[L_q] = sum_{m, a} Qma[L_q, m, a] * T_n[m, a]
     *   Row-major view: gamma = Qma_view(L, M*A) @ T_n_flat(M*A).
     *   F view of Qma_view is (M*A, L); gamma = Qma_F.T @ T_n_flat (dgemv 'T').
     */
    if (M > 0) {
        dgemv_(&T_flag, &i_MA, &i_L,
               &one, Qma_, &i_MA, T_n_, &ione,
               &zero, gamma, &ione);
    } else {
        memset(gamma, 0, sizeof(double) * L);
    }

    /* r1[a] += 2 * sum_L Qia[L, a] * gamma[L]
     *   Row-major Qia (L, A); F view (A, L); r1 = 2 * Qia_F @ gamma (dgemv 'N').
     */
    dgemv_(&N_flag, &i_A, &i_L,
           &two, Qia_, &i_A, gamma, &ione,
           &one, r1_inout, &ione);

    /* y[L, a] = sum_m Qik[L, m] * T_n[m, a]
     *   Row-major y = Qik @ T_n.  C = A @ B row-major pattern:
     *     dgemm('N','N', N_row, M_row, K_row, alpha, B, N_row, A, K_row, beta, C, N_row).
     */
    if (M > 0) {
        dgemm_(&N_flag, &N_flag,
               &i_A, &i_L, &i_M,
               &one, T_n_, &i_A, Qik_, &i_M,
               &zero, y, &i_A);
    } else {
        memset(y, 0, sizeof(double) * L * A);
    }

    /* r1[a] -= sum_{L, c} Qab[L, a, c] * y[L, c]
     *   Per-L: r1 -= Qab_L (A, A) @ y_L (A,).  Row-major dgemv with 'T'.
     */
    for (size_t Lq = 0; Lq < L; Lq++) {
        const double *Qab_L = Qab + Lq * A * A;
        const double *y_L   = y   + Lq * A;
        dgemv_(&T_flag, &i_A, &i_A,
               &neg_one, (double *)Qab_L, &i_A, (double *)y_L, &ione,
               &one, r1_inout, &ione);
    }

    if (!do_stage_23) {
        free(gamma);
        free(y);
        return;
    }

    /* ============= Stage 2 ============= */
    double *Fab    = (double *)malloc(sizeof(double) * A * A);
    double *tmp_AM = (M > 0) ? (double *)malloc(sizeof(double) * A * M) : NULL;
    if (Fab == NULL || (M > 0 && tmp_AM == NULL)) {
        free(Fab); free(tmp_AM); free(gamma); free(y); return;
    }

    /* Fab[a, b] = 2 * sum_L gamma[L] * Qab[L, a, b]
     *   View Qab as (L, A*A) row-major; F view (A*A, L).
     *   Fab_flat = 2 * Qab_F @ gamma (dgemv 'N'). */
    dgemv_(&N_flag, &i_AA, &i_L,
           &two, Qab_, &i_AA, gamma, &ione,
           &zero, Fab, &ione);

    /* + e_pno on diagonal. */
    for (size_t a = 0; a < A; a++) {
        Fab[a * A + a] += e_pno[a];
    }

    /* For each L: Fab -= (Qab_L @ T_n.T) @ Qma_L. */
    if (M > 0) {
        for (size_t Lq = 0; Lq < L; Lq++) {
            const double *Qab_L = Qab + Lq * A * A;
            const double *Qma_L = Qma + Lq * M * A;
            /* tmp_AM (A, M) = Qab_L (A, A) @ T_n.T (A, M).
             *   Row-major C = A @ B.T pattern (B = T_n row-major (M, A)):
             *     dgemm('T','N', N_row=M, M_row=A, K_row=A,
             *           alpha, B=T_n ldb=K=A, A=Qab_L lda=K=A,
             *           beta, C ldc=N=M). */
            dgemm_(&T_flag, &N_flag,
                   &i_M, &i_A, &i_A,
                   &one, T_n_, &i_A, (double *)Qab_L, &i_A,
                   &zero, tmp_AM, &i_M);
            /* Fab -= tmp_AM (A, M) @ Qma_L (M, A).
             *   Row-major C = A @ B pattern:
             *     dgemm('N','N', N_row=A, M_row=A, K_row=M,
             *           -1, B=Qma_L ldb=N=A, A=tmp_AM lda=K=M,
             *           +1, C=Fab ldc=N=A). */
            dgemm_(&N_flag, &N_flag,
                   &i_A, &i_A, &i_M,
                   &neg_one, (double *)Qma_L, &i_A, tmp_AM, &i_M,
                   &one, Fab, &i_A);
        }
    }

    /* r1[a] += sum_b Fab[a, b] * t1_i[b]   (Fab row-major (A, A); dgemv 'T'). */
    dgemv_(&T_flag, &i_A, &i_A,
           &one, Fab, &i_A, t1_i_, &ione,
           &one, r1_inout, &ione);

    free(Fab);
    free(tmp_AM);

    /* ============= Stage 3 ============= */
    if (M == 0) {
        free(gamma);
        free(y);
        return;
    }
    double *Fia    = (double *)malloc(sizeof(double) * M * A);
    double *tmp_MM = (double *)malloc(sizeof(double) * M * M);
    double *v      = (double *)malloc(sizeof(double) * M);
    if (Fia == NULL || tmp_MM == NULL || v == NULL) {
        free(Fia); free(tmp_MM); free(v); free(gamma); free(y); return;
    }

    /* Fia[mp, a] = 2 * sum_L gamma[L] * Qma[L, mp, a]
     *   View Qma as (L, M*A) row-major; F view (M*A, L).
     *   Fia_flat = 2 * Qma_F @ gamma (dgemv 'N'). */
    dgemv_(&N_flag, &i_MA, &i_L,
           &two, Qma_, &i_MA, gamma, &ione,
           &zero, Fia, &ione);

    /* For each L: Fia -= (Qma_L @ T_n.T).T @ Qma_L. */
    for (size_t Lq = 0; Lq < L; Lq++) {
        const double *Qma_L = Qma + Lq * M * A;
        /* tmp_MM (M, M) = Qma_L (M, A) @ T_n.T (A, M).
         *   Same row-major C = A @ B.T pattern as Stage 2 tmp_AM, A→M outer. */
        dgemm_(&T_flag, &N_flag,
               &i_M, &i_M, &i_A,
               &one, T_n_, &i_A, (double *)Qma_L, &i_A,
               &zero, tmp_MM, &i_M);
        /* Fia -= tmp_MM.T (M, M) @ Qma_L (M, A).
         *
         * Row-major C[mp, a] = sum_m tmp_MM[m, mp] * Qma_L[m, a].
         * F-views: tmp_MM_F[mp, m] = tmp_MM_row[m, mp];
         *          Qma_L_F[a, m]   = Qma_L_row[m, a];
         *          Fia_F[a, mp]    = Fia_row[mp, a].
         * Then C_F[a, mp] = sum_m Qma_L_F[a, m] * tmp_MM_F[mp, m]
         *                 = (Qma_L_F @ tmp_MM_F.T)[a, mp]
         * dgemm('N','T', m=A, n=M, k=M, alpha=-1, A=Qma_L lda=A,
         *       B=tmp_MM ldb=M, beta=+1, C=Fia ldc=A). */
        dgemm_(&N_flag, &T_flag,
               &i_A, &i_M, &i_M,
               &neg_one, (double *)Qma_L, &i_A, tmp_MM, &i_M,
               &one, Fia, &i_A);
    }

    /* v[m] = sum_b Fia[m, b] * t1_i[b]  (Fia row-major (M, A); dgemv 'T'). */
    dgemv_(&T_flag, &i_A, &i_M,
           &one, Fia, &i_A, t1_i_, &ione,
           &zero, v, &ione);

    /* r1[a] -= sum_m T_n[m, a] * v[m]
     *   Row-major T_n (M, A) -> F (A, M). r1 -= T_n_F @ v (dgemv 'N'). */
    dgemv_(&N_flag, &i_A, &i_M,
           &neg_one, T_n_, &i_A, v, &ione,
           &one, r1_inout, &ione);

    free(Fia);
    free(tmp_MM);
    free(v);
    free(gamma);
    free(y);
}
