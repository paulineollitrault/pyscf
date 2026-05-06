/* DLPNO-CCSD T2 residual: per-bucket B + E scatter-accumulator.
 *
 * Math per item (one (ij, kl) item with kl in pair_lmo_idx[ij]):
 *
 *   if same (k == l):
 *     UK[b, c]   = sum_d (2 T[b, d] - T[d, b]) * K[c, d]
 *     TB[b, c]   = beta_kl * T[b, c]
 *   else:
 *     UK[b, c]   = sum_d (2 T[b, d] - T[d, b]) * K[c, d]
 *                + sum_d (2 T[d, b] - T[b, d]) * K[d, c]
 *     TB[b, c]   = beta_kl * T[b, c] + beta_lk * T[c, b]
 *
 *   STB[a, c]  = sum_b S[a, b] * TB[b, c]
 *   SUK[a, c]  = sum_b S[a, b] * UK[b, c]
 *   Bc[a, d]   = sum_c STB[a, c] * S[d, c]   (= STB @ S^T)
 *   Ec[a, d]   = sum_c SUK[a, c] * S[d, c]
 *
 *   out_B[idx[n]] += Bc
 *   out_E[idx[n]] += Ec
 *
 * BLAS port (2026-04-30): per-item triple loops -> 4-5 DGEMM calls with the
 * MKL link.  At npno~25, MKL's JIT-compiled small-GEMM kernels are 2-3x
 * faster than hand-rolled triple loops; the Compute R2 work is the largest
 * non-BLAS phase per cycle.
 *
 * Layout:
 *   Stage 1: parallel prange over items.  Per item: build TT_minus, TB
 *            (small vec ops), then 4-5 DGEMMs into per-item Bc/Ec.
 *   Stage 2: sequential scatter-add into out_B / out_E (race-free).
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

void DLPNObe_kernel(const double      *S,           /* (N, n_ij, n_kl) */
                    const double      *T,           /* (N, n_kl, n_kl) */
                    const double      *K,           /* (N, n_kl, n_kl) */
                    const double      *beta_kl,     /* (N,) */
                    const double      *beta_lk,     /* (N,) */
                    const unsigned char *same,      /* (N,) */
                    const long        *idx,         /* (N,) */
                    double            *out_B,       /* (n_slots, n_ij, n_ij) */
                    double            *out_E,       /* (n_slots, n_ij, n_ij) */
                    const size_t       N,
                    const size_t       n_ij,
                    const size_t       n_kl,
                    const int          num_threads)
{
    if (N == 0) return;

    const size_t S_stride  = n_ij * n_kl;
    const size_t TK_stride = n_kl * n_kl;
    const size_t out_stride = n_ij * n_ij;

    /* Per-item Bc/Ec output buffers (race-free in Stage 1). */
    double *Bc = (double *)malloc(sizeof(double) * N * out_stride);
    double *Ec = (double *)malloc(sizeof(double) * N * out_stride);

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;
    int int_n_ij = (int)n_ij, int_n_kl = (int)n_kl;

#pragma omp parallel num_threads(num_threads)
    {
        /* Per-thread scratch */
        double *TT_minus = (double *)malloc(sizeof(double) * n_kl * n_kl);
        double *TT_plus  = (double *)malloc(sizeof(double) * n_kl * n_kl);
        double *TB       = (double *)malloc(sizeof(double) * n_kl * n_kl);
        double *UK       = (double *)malloc(sizeof(double) * n_kl * n_kl);
        double *STB      = (double *)malloc(sizeof(double) * n_ij * n_kl);
        double *SUK      = (double *)malloc(sizeof(double) * n_ij * n_kl);

#pragma omp for schedule(dynamic, 1)
        for (size_t n = 0; n < N; n++) {
            const double bkl = beta_kl[n];
            const double blk = beta_lk[n];
            const unsigned char sm = same[n];

            const double *Sn = S + n * S_stride;
            const double *Tn = T + n * TK_stride;
            const double *Kn = K + n * TK_stride;
            double       *Bcn = Bc + n * out_stride;
            double       *Ecn = Ec + n * out_stride;

            /* Build TT_minus[b, d] = 2 T[b, d] - T[d, b] */
            for (size_t b = 0; b < n_kl; b++) {
                for (size_t d = 0; d < n_kl; d++) {
                    TT_minus[b * n_kl + d] = 2.0 * Tn[b * n_kl + d]
                                              - Tn[d * n_kl + b];
                }
            }

            /* Build TB:
             *   sm:  TB = bkl * T
             *   !sm: TB[b, c] = bkl * T[b, c] + blk * T[c, b]
             */
            if (sm) {
                for (size_t e = 0; e < TK_stride; e++) {
                    TB[e] = bkl * Tn[e];
                }
            } else {
                for (size_t b = 0; b < n_kl; b++) {
                    for (size_t c = 0; c < n_kl; c++) {
                        TB[b * n_kl + c] = bkl * Tn[b * n_kl + c]
                                          + blk * Tn[c * n_kl + b];
                    }
                }
                /* Build TT_plus[b, d] = 2 T[d, b] - T[b, d]   (only !sm) */
                for (size_t b = 0; b < n_kl; b++) {
                    for (size_t d = 0; d < n_kl; d++) {
                        TT_plus[b * n_kl + d] = 2.0 * Tn[d * n_kl + b]
                                                 - Tn[b * n_kl + d];
                    }
                }
            }

            /* UK[b, c] = sum_d TT_minus[b, d] * K[c, d]   (= TT_minus @ K^T)
             * Row-major math: UK = TT_minus (n_kl, n_kl) @ K^T (n_kl, n_kl).
             * Fortran view: UK_F[c, b] = sum_d K_F[d, c] * TT_minus_F[d, b]
             *             = K_F^T @ TT_minus_F.  dgemm('T', 'N', n_kl, n_kl, n_kl,
             *                                          1, K, n_kl, TT_minus, n_kl,
             *                                          0, UK, n_kl).
             */
            dgemm_(&T_flag, &N_flag,
                   &int_n_kl, &int_n_kl, &int_n_kl,
                   &one, Kn, &int_n_kl,
                   TT_minus, &int_n_kl,
                   &zero, UK, &int_n_kl);

            /* If !sm: UK += TT_plus @ K
             * Math: UK[b, c] += sum_d TT_plus[b, d] * K[d, c]
             * F view: UK_F[c, b] += sum_d K_F[c, d] * TT_plus_F[d, b]
             *      = K_F @ TT_plus_F
             * dgemm('N', 'N', n_kl, n_kl, n_kl, 1, K, n_kl, TT_plus, n_kl, 1, UK, n_kl)
             */
            if (!sm) {
                dgemm_(&N_flag, &N_flag,
                       &int_n_kl, &int_n_kl, &int_n_kl,
                       &one, Kn, &int_n_kl,
                       TT_plus, &int_n_kl,
                       &one, UK, &int_n_kl);
            }

            /* STB[a, c] = sum_b S[a, b] * TB[b, c]
             * Row-major: STB (n_ij, n_kl) = S (n_ij, n_kl) @ TB (n_kl, n_kl).
             * F view: STB_F[c, a] = sum_b TB_F[c, b] * S_F[b, a] = TB_F @ S_F.
             * dgemm('N', 'N', n_kl, n_ij, n_kl, 1, TB, n_kl, S, n_kl, 0, STB, n_kl)
             */
            dgemm_(&N_flag, &N_flag,
                   &int_n_kl, &int_n_ij, &int_n_kl,
                   &one, TB, &int_n_kl,
                   Sn, &int_n_kl,
                   &zero, STB, &int_n_kl);

            /* SUK = S @ UK (same shape/op as STB) */
            dgemm_(&N_flag, &N_flag,
                   &int_n_kl, &int_n_ij, &int_n_kl,
                   &one, UK, &int_n_kl,
                   Sn, &int_n_kl,
                   &zero, SUK, &int_n_kl);

            /* Bc[a, d] = sum_c STB[a, c] * S[d, c]   (= STB @ S^T)
             * Row-major: Bc (n_ij, n_ij) = STB (n_ij, n_kl) @ S^T (n_kl, n_ij).
             * F view: Bc_F[d, a] = sum_c S_F[c, d] * STB_F[c, a] = S_F^T @ STB_F.
             * dgemm('T', 'N', n_ij, n_ij, n_kl, 1, S, n_kl, STB, n_kl, 0, Bc, n_ij)
             */
            dgemm_(&T_flag, &N_flag,
                   &int_n_ij, &int_n_ij, &int_n_kl,
                   &one, Sn, &int_n_kl,
                   STB, &int_n_kl,
                   &zero, Bcn, &int_n_ij);

            /* Ec = SUK @ S^T  (same op as Bc) */
            dgemm_(&T_flag, &N_flag,
                   &int_n_ij, &int_n_ij, &int_n_kl,
                   &one, Sn, &int_n_kl,
                   SUK, &int_n_kl,
                   &zero, Ecn, &int_n_ij);
        }

        free(TT_minus);
        free(TT_plus);
        free(TB);
        free(UK);
        free(STB);
        free(SUK);
    }

    /* Stage 2: sequential scatter-add (race-free since idx may collide). */
    for (size_t n = 0; n < N; n++) {
        const long target = idx[n];
        const double *Bcn = Bc + n * out_stride;
        const double *Ecn = Ec + n * out_stride;
        double *out_Bt = out_B + (size_t)target * out_stride;
        double *out_Et = out_E + (size_t)target * out_stride;
        for (size_t k = 0; k < out_stride; k++) {
            out_Bt[k] += Bcn[k];
            out_Et[k] += Ecn[k];
        }
    }

    free(Bc);
    free(Ec);
}

/* DLPNObe_kernel_v3 — per-target accumulation variant.
 *
 * Same math as DLPNObe_kernel (per-item TT_minus/TT_plus/TB/UK/STB/SUK build,
 * Bc = STB @ S^T, Ec = SUK @ S^T).  Difference: instead of materialising
 * per-item Bc/Ec into a global (N x out_stride) buffer and then doing a
 * sequential scatter, we group items by their target slot `idx[n]` and
 * accumulate the final dgemm directly into the destination slot with
 * BLAS beta=1.  Since slots are disjoint memory, OMP parallelism over
 * targets is race-free without locks.
 *
 * Memory savings vs v1:
 *   v1: 2 * N * n_ij^2 * 8 bytes  (Bc + Ec at once)
 *   v2: 2 * num_threads * n_ij^2 * 8 bytes  (per-thread scratch only)
 *
 * On water-22 with N≈1850 max-bucket and n_ij=24 the v1 buffer reaches
 * ~17 MB per bucket (~100 MB across all buckets per pass).  Eliminating
 * that traffic is the bulk of the win — the FLOP count is unchanged.
 *
 * Caller contract:
 *   * `idx` may collide across items; this routine groups items with the
 *     same idx and processes each group on a single thread.
 *   * `out_B` / `out_E` are caller-zeroed (or pre-existing accumulators).
 */
void DLPNObe_kernel_v3(const double      *S,
                       const double      *T,
                       const double      *K,
                       const double      *beta_kl,
                       const double      *beta_lk,
                       const unsigned char *same,
                       const long        *idx,
                       double            *out_B,
                       double            *out_E,
                       const size_t       N,
                       const size_t       n_ij,
                       const size_t       n_kl,
                       const int          num_threads,
                       const size_t       n_slots)
{
    if (N == 0) return;

    const size_t S_stride  = n_ij * n_kl;
    const size_t TK_stride = n_kl * n_kl;
    const size_t out_stride = n_ij * n_ij;

    /* Build per-target item buckets in O(N + n_slots) time.  Items with
     * the same idx[n] become contiguous in `sorted_n`. */
    long *bucket_count = (long *)calloc(n_slots, sizeof(long));
    long *bucket_off   = (long *)malloc(sizeof(long) * (n_slots + 1));
    long *sorted_n     = (long *)malloc(sizeof(long) * N);
    for (size_t n = 0; n < N; n++) {
        bucket_count[idx[n]]++;
    }
    bucket_off[0] = 0;
    for (size_t s = 0; s < n_slots; s++) {
        bucket_off[s + 1] = bucket_off[s] + bucket_count[s];
        bucket_count[s] = 0;
    }
    for (size_t n = 0; n < N; n++) {
        long s = idx[n];
        sorted_n[bucket_off[s] + bucket_count[s]] = (long)n;
        bucket_count[s]++;
    }
    free(bucket_count);

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;
    int int_n_ij = (int)n_ij, int_n_kl = (int)n_kl;

#pragma omp parallel num_threads(num_threads)
    {
        /* Per-thread scratch — sized to one item, NOT N. */
        double *TT_minus = (double *)malloc(sizeof(double) * n_kl * n_kl);
        double *TT_plus  = (double *)malloc(sizeof(double) * n_kl * n_kl);
        double *TB       = (double *)malloc(sizeof(double) * n_kl * n_kl);
        double *UK       = (double *)malloc(sizeof(double) * n_kl * n_kl);
        double *STB      = (double *)malloc(sizeof(double) * n_ij * n_kl);
        double *SUK      = (double *)malloc(sizeof(double) * n_ij * n_kl);

#pragma omp for schedule(dynamic, 1)
        for (size_t s = 0; s < n_slots; s++) {
            const long b_lo = bucket_off[s];
            const long b_hi = bucket_off[s + 1];
            if (b_lo == b_hi) continue;

            double *out_Bt = out_B + s * out_stride;
            double *out_Et = out_E + s * out_stride;

            for (long bi = b_lo; bi < b_hi; bi++) {
                const size_t n = (size_t)sorted_n[bi];
                const double bkl = beta_kl[n];
                const double blk = beta_lk[n];
                const unsigned char sm = same[n];

                const double *Sn = S + n * S_stride;
                const double *Tn = T + n * TK_stride;
                const double *Kn = K + n * TK_stride;

                /* TT_minus[b, d] = 2 T[b, d] - T[d, b] */
                for (size_t b = 0; b < n_kl; b++) {
                    for (size_t d = 0; d < n_kl; d++) {
                        TT_minus[b * n_kl + d] = 2.0 * Tn[b * n_kl + d]
                                                  - Tn[d * n_kl + b];
                    }
                }

                if (sm) {
                    for (size_t e = 0; e < TK_stride; e++) {
                        TB[e] = bkl * Tn[e];
                    }
                } else {
                    for (size_t b = 0; b < n_kl; b++) {
                        for (size_t c = 0; c < n_kl; c++) {
                            TB[b * n_kl + c] = bkl * Tn[b * n_kl + c]
                                              + blk * Tn[c * n_kl + b];
                        }
                    }
                    for (size_t b = 0; b < n_kl; b++) {
                        for (size_t d = 0; d < n_kl; d++) {
                            TT_plus[b * n_kl + d] = 2.0 * Tn[d * n_kl + b]
                                                     - Tn[b * n_kl + d];
                        }
                    }
                }

                /* UK = TT_minus @ K^T  (BLAS: K^T @ TT_minus in F-view) */
                dgemm_(&T_flag, &N_flag,
                       &int_n_kl, &int_n_kl, &int_n_kl,
                       &one, Kn, &int_n_kl,
                       TT_minus, &int_n_kl,
                       &zero, UK, &int_n_kl);

                if (!sm) {
                    /* UK += TT_plus @ K */
                    dgemm_(&N_flag, &N_flag,
                           &int_n_kl, &int_n_kl, &int_n_kl,
                           &one, Kn, &int_n_kl,
                           TT_plus, &int_n_kl,
                           &one, UK, &int_n_kl);
                }

                /* STB = S @ TB ; SUK = S @ UK */
                dgemm_(&N_flag, &N_flag,
                       &int_n_kl, &int_n_ij, &int_n_kl,
                       &one, TB, &int_n_kl,
                       Sn, &int_n_kl,
                       &zero, STB, &int_n_kl);
                dgemm_(&N_flag, &N_flag,
                       &int_n_kl, &int_n_ij, &int_n_kl,
                       &one, UK, &int_n_kl,
                       Sn, &int_n_kl,
                       &zero, SUK, &int_n_kl);

                /* Accumulate directly into out_Bt / out_Et with beta=1. */
                dgemm_(&T_flag, &N_flag,
                       &int_n_ij, &int_n_ij, &int_n_kl,
                       &one, Sn, &int_n_kl,
                       STB, &int_n_kl,
                       &one, out_Bt, &int_n_ij);
                dgemm_(&T_flag, &N_flag,
                       &int_n_ij, &int_n_ij, &int_n_kl,
                       &one, Sn, &int_n_kl,
                       SUK, &int_n_kl,
                       &one, out_Et, &int_n_ij);
            }
        }

        free(TT_minus);
        free(TT_plus);
        free(TB);
        free(UK);
        free(STB);
        free(SUK);
    }

    free(bucket_off);
    free(sorted_n);
}
