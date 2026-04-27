/* DLPNO-CCSD T1 residual: per-i Stages 1-3 (Fai_bar, Fab_bar*t1, -T_n^T*Fia_bar*t1).
 *
 * Phase II port of pyscf/cc/dlpno_tccsd/_per_i_stages_cy.pyx into PySCF
 * native style. Replaces the prior nocc-axis kernel with a pair-domain
 * (nlmo_pair) signature so the caller no longer needs to scatter-back
 * Qik and t1_cache into (n_local, nocc) / (nocc, npno) buffers.
 *
 * Math (matches the Cython reference, just M = nlmo_pair instead of nocc):
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
 * Entry shapes (M = nlmo_pair):
 *   r1_inout: (A,)        — accumulator (caller initialises before Stage 4)
 *   Qma:      (L, M, A)
 *   Qab:      (L, A, A)
 *   Qia:      (L, A)
 *   Qik:      (L, M)
 *   T_n:      (M, A)      — caller gathered: t1_cache[key_ii][ci_ii['p_lmos']]
 *   t1_i:     (A,)
 *   e_pno:    (A,)
 */

#include <stdlib.h>
#include <string.h>

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
    const size_t QMA_L = M * A;     /* Qma[L, ...] stride */
    const size_t QMA_M = A;         /* Qma[L, m, ...] stride */
    const size_t QAB_L = A * A;     /* Qab[L, ...] stride */
    const size_t QAB_A = A;         /* Qab[L, a, ...] stride */
    const size_t QIA_L = A;         /* Qia[L, ...] stride */
    const size_t QIK_L = M;         /* Qik[L, ...] stride */
    const size_t TN_M  = A;         /* T_n[m, ...] stride */

    double *gamma = (double *)malloc(sizeof(double) * L);
    double *y     = (double *)malloc(sizeof(double) * L * A);

    /* ============= Stage 1 ============= */

    /* gamma[Lq] = sum_{m, a} Qma[Lq, m, a] * T_n[m, a] */
#pragma omp parallel for schedule(static)
    for (size_t Lq = 0; Lq < L; Lq++) {
        double s = 0.0;
        const double *Qma_L = Qma + Lq * QMA_L;
        for (size_t m = 0; m < M; m++) {
            const double *Qma_Lm = Qma_L + m * QMA_M;
            const double *Tn_m   = T_n + m * TN_M;
            for (size_t a = 0; a < A; a++) {
                s += Qma_Lm[a] * Tn_m[a];
            }
        }
        gamma[Lq] = s;
    }

    /* r1[a] += 2 * sum_L Qia[L, a] * gamma[L] */
#pragma omp parallel for schedule(static)
    for (size_t a = 0; a < A; a++) {
        double s = 0.0;
        for (size_t Lq = 0; Lq < L; Lq++) {
            s += Qia[Lq * QIA_L + a] * gamma[Lq];
        }
        r1_inout[a] += 2.0 * s;
    }

    /* y[Lq, a] = sum_m Qik[Lq, m] * T_n[m, a] */
#pragma omp parallel for schedule(static)
    for (size_t Lq = 0; Lq < L; Lq++) {
        const double *Qik_L = Qik + Lq * QIK_L;
        double *y_L = y + Lq * A;
        for (size_t a = 0; a < A; a++) {
            double s = 0.0;
            for (size_t m = 0; m < M; m++) {
                s += Qik_L[m] * T_n[m * TN_M + a];
            }
            y_L[a] = s;
        }
    }

    /* r1[a] -= sum_{L, c} Qab[L, a, c] * y[L, c] */
#pragma omp parallel for schedule(static)
    for (size_t a = 0; a < A; a++) {
        double s = 0.0;
        for (size_t Lq = 0; Lq < L; Lq++) {
            const double *Qab_La = Qab + Lq * QAB_L + a * QAB_A;
            const double *y_L    = y + Lq * A;
            for (size_t c = 0; c < A; c++) {
                s += Qab_La[c] * y_L[c];
            }
        }
        r1_inout[a] -= s;
    }

    if (!do_stage_23) {
        free(gamma);
        free(y);
        return;
    }

    /* ============= Stage 2 ============= */

    /* W[L, b, m] = sum_c Qab[L, b, c] * T_n[m, c]
     * Storage: W is (L, A, M), index = Lq * A * M + b * M + m. */
    double *W = (double *)malloc(sizeof(double) * L * A * M);
#pragma omp parallel for schedule(static)
    for (size_t Lq = 0; Lq < L; Lq++) {
        const double *Qab_L = Qab + Lq * QAB_L;
        double *W_L = W + Lq * A * M;
        for (size_t b = 0; b < A; b++) {
            const double *Qab_Lb = Qab_L + b * QAB_A;
            double *W_Lb = W_L + b * M;
            for (size_t m = 0; m < M; m++) {
                double s = 0.0;
                const double *Tn_m = T_n + m * TN_M;
                for (size_t c = 0; c < A; c++) {
                    s += Qab_Lb[c] * Tn_m[c];
                }
                W_Lb[m] = s;
            }
        }
    }

    /* Fab[a, b] = e_pno*delta + 2*sum_L gamma*Qab[L,a,b] - sum_{L,m} W[L,a,m]*Qma[L,m,b] */
    double *Fab = (double *)malloc(sizeof(double) * A * A);
#pragma omp parallel for schedule(static)
    for (size_t a = 0; a < A; a++) {
        for (size_t b = 0; b < A; b++) {
            double s = (a == b) ? e_pno[a] : 0.0;
            /* +2 * sum_L gamma[L] * Qab[L, a, b] */
            double gv = 0.0;
            for (size_t Lq = 0; Lq < L; Lq++) {
                gv += gamma[Lq] * Qab[Lq * QAB_L + a * QAB_A + b];
            }
            s += 2.0 * gv;
            /* - sum_{L, m} W[L, a, m] * Qma[L, m, b] */
            gv = 0.0;
            for (size_t Lq = 0; Lq < L; Lq++) {
                const double *W_La = W + Lq * A * M + a * M;
                for (size_t m = 0; m < M; m++) {
                    gv += W_La[m] * Qma[Lq * QMA_L + m * QMA_M + b];
                }
            }
            s -= gv;
            Fab[a * A + b] = s;
        }
    }

    /* r1[a] += sum_b Fab[a, b] * t1_i[b] */
#pragma omp parallel for schedule(static)
    for (size_t a = 0; a < A; a++) {
        double s = 0.0;
        for (size_t b = 0; b < A; b++) {
            s += Fab[a * A + b] * t1_i[b];
        }
        r1_inout[a] += s;
    }

    free(W);
    free(Fab);

    /* ============= Stage 3 ============= */

    /* Z[L, m, m'] = sum_c Qma[L, m, c] * T_n[m', c] */
    double *Z = (double *)malloc(sizeof(double) * L * M * M);
#pragma omp parallel for schedule(static)
    for (size_t Lq = 0; Lq < L; Lq++) {
        const double *Qma_L = Qma + Lq * QMA_L;
        double *Z_L = Z + Lq * M * M;
        for (size_t m = 0; m < M; m++) {
            const double *Qma_Lm = Qma_L + m * QMA_M;
            double *Z_Lm = Z_L + m * M;
            for (size_t mp = 0; mp < M; mp++) {
                double s = 0.0;
                const double *Tn_mp = T_n + mp * TN_M;
                for (size_t c = 0; c < A; c++) {
                    s += Qma_Lm[c] * Tn_mp[c];
                }
                Z_Lm[mp] = s;
            }
        }
    }

    /* Fia[mp, a] = 2*sum_L gamma*Qma[L,mp,a] - sum_{L,m} Z[L,m,mp]*Qma[L,m,a] */
    double *Fia = (double *)malloc(sizeof(double) * M * A);
#pragma omp parallel for schedule(static)
    for (size_t mp = 0; mp < M; mp++) {
        for (size_t a = 0; a < A; a++) {
            double gv = 0.0;
            for (size_t Lq = 0; Lq < L; Lq++) {
                gv += gamma[Lq] * Qma[Lq * QMA_L + mp * QMA_M + a];
            }
            double s = 2.0 * gv;
            gv = 0.0;
            for (size_t Lq = 0; Lq < L; Lq++) {
                const double *Z_L = Z + Lq * M * M;
                for (size_t m = 0; m < M; m++) {
                    gv += Z_L[m * M + mp] * Qma[Lq * QMA_L + m * QMA_M + a];
                }
            }
            s -= gv;
            Fia[mp * A + a] = s;
        }
    }

    /* v[m] = sum_b Fia[m, b] * t1_i[b] */
    double *v = (double *)malloc(sizeof(double) * M);
#pragma omp parallel for schedule(static)
    for (size_t m = 0; m < M; m++) {
        double s = 0.0;
        for (size_t b = 0; b < A; b++) {
            s += Fia[m * A + b] * t1_i[b];
        }
        v[m] = s;
    }

    /* r1[a] -= sum_m T_n[m, a] * v[m] */
#pragma omp parallel for schedule(static)
    for (size_t a = 0; a < A; a++) {
        double s = 0.0;
        for (size_t m = 0; m < M; m++) {
            s += T_n[m * TN_M + a] * v[m];
        }
        r1_inout[a] -= s;
    }

    free(Z);
    free(Fia);
    free(v);
    free(gamma);
    free(y);
}
