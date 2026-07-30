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

/* DLPNObe_kernel_gathered — memory-light variant of DLPNObe_kernel.
 *
 * Same math, identical kernel structure (BLAS 4-5 DGEMMs per item; Stage 2
 * sequential scatter into out_B / out_E).  The only difference is the
 * input access pattern: instead of stacked per-bucket (N, n_ij, n_kl) S
 * and (N, n_kl, n_kl) T/K buffers, S/T/K are read from caller-owned
 * master flats via per-item offsets:
 *
 *     S[n] = S_master + S_off[n]   (n_ij × n_kl tile)
 *     T[n] = T_master + T_off[n]   (n_kl × n_kl tile)
 *     K[n] = K_master + K_off[n]   (n_kl × n_kl tile)
 *
 * Eliminates the per-bucket (N, n_ij, n_kl) + (N, n_kl, n_kl) copies that
 * the legacy kernel reads from — at water-22 the BE plan stack-arrays
 * total ~1.5 GB across all buckets, growing as N² (pairs²) × n_pno²;
 * cc-pVTZ water-49 would otherwise blow the box.
 */
void DLPNObe_kernel_gathered(const double      *S_master,
                             const long        *S_off,
                             const double      *T_master,
                             const long        *T_off,
                             const double      *K_master,
                             const long        *K_off,
                             const double      *beta_kl,
                             const double      *beta_lk,
                             const unsigned char *same,
                             const long        *idx,
                             double            *out_B,
                             double            *out_E,
                             const size_t       N,
                             const size_t       n_ij,
                             const size_t       n_kl,
                             const int          uk_hoisted,
                             const int          num_threads)
{
    if (N == 0) return;

    const size_t out_stride = n_ij * n_ij;

    double *Bc = (double *)malloc(sizeof(double) * N * out_stride);
    double *Ec = (double *)malloc(sizeof(double) * N * out_stride);

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;
    int int_n_ij = (int)n_ij, int_n_kl = (int)n_kl;

#pragma omp parallel num_threads(num_threads)
    {
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

            const double *Sn = S_master + S_off[n];
            const double *Tn = T_master + T_off[n];
            const double *Kn = K_master + K_off[n];
            double       *Bcn = Bc + n * out_stride;
            double       *Ecn = Ec + n * out_stride;

            /* UK hoist: Kn already holds UK = K^T(2T-T^T) [+ K(2T^T-T)]
             * built once per pair per cycle — skip the per-item build. */
            const double *UKn = uk_hoisted ? Kn : UK;

            if (!uk_hoisted) {
            for (size_t b = 0; b < n_kl; b++) {
                for (size_t d = 0; d < n_kl; d++) {
                    TT_minus[b * n_kl + d] = 2.0 * Tn[b * n_kl + d]
                                              - Tn[d * n_kl + b];
                }
            }
            }

            if (sm) {
                for (size_t e = 0; e < n_kl * n_kl; e++) {
                    TB[e] = bkl * Tn[e];
                }
            } else {
                for (size_t b = 0; b < n_kl; b++) {
                    for (size_t c = 0; c < n_kl; c++) {
                        TB[b * n_kl + c] = bkl * Tn[b * n_kl + c]
                                          + blk * Tn[c * n_kl + b];
                    }
                }
                if (!uk_hoisted) {
                for (size_t b = 0; b < n_kl; b++) {
                    for (size_t d = 0; d < n_kl; d++) {
                        TT_plus[b * n_kl + d] = 2.0 * Tn[d * n_kl + b]
                                                 - Tn[b * n_kl + d];
                    }
                }
                }
            }

            if (!uk_hoisted) {
            dgemm_(&T_flag, &N_flag,
                   &int_n_kl, &int_n_kl, &int_n_kl,
                   &one, Kn, &int_n_kl,
                   TT_minus, &int_n_kl,
                   &zero, UK, &int_n_kl);
            if (!sm) {
                dgemm_(&N_flag, &N_flag,
                       &int_n_kl, &int_n_kl, &int_n_kl,
                       &one, Kn, &int_n_kl,
                       TT_plus, &int_n_kl,
                       &one, UK, &int_n_kl);
            }
            }

            dgemm_(&N_flag, &N_flag,
                   &int_n_kl, &int_n_ij, &int_n_kl,
                   &one, TB, &int_n_kl,
                   Sn, &int_n_kl,
                   &zero, STB, &int_n_kl);
            dgemm_(&N_flag, &N_flag,
                   &int_n_kl, &int_n_ij, &int_n_kl,
                   &one, UKn, &int_n_kl,
                   Sn, &int_n_kl,
                   &zero, SUK, &int_n_kl);
            dgemm_(&T_flag, &N_flag,
                   &int_n_ij, &int_n_ij, &int_n_kl,
                   &one, Sn, &int_n_kl,
                   STB, &int_n_kl,
                   &zero, Bcn, &int_n_ij);
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


/* DLPNObe_kernel_gathered_v3 — gathered reads (masters + per-item offsets,
 * incl. the UK hoist) combined with v3's per-target in-place accumulation:
 * items are grouped by output slot idx[n]; each slot is owned by ONE thread,
 * which accumulates the final two dgemms directly into out_B/out_E with
 * beta=1.  Eliminates the (N x n_ij^2) Bc/Ec staging buffers and the serial
 * scatter pass of DLPNObe_kernel_gathered — the FLOPs are unchanged; the
 * removed memory traffic and the parallelised reduction are the win.
 * Caller contract identical to the staged gathered kernel (out_B/out_E are
 * pre-zeroed accumulators; idx may collide across items). */
void DLPNObe_kernel_gathered_v3(const double      *S_master,
                                const long        *S_off,
                                const double      *T_master,
                                const long        *T_off,
                                const double      *K_master,
                                const long        *K_off,
                                const double      *beta_kl,
                                const double      *beta_lk,
                                const unsigned char *same,
                                const long        *idx,
                                double            *out_B,
                                double            *out_E,
                                const size_t       N,
                                const size_t       n_ij,
                                const size_t       n_kl,
                                const size_t       n_slots,
                                const int          uk_hoisted,
                                const double      *scr_t2n,
                                const double      *scr_ukn,
                                const double       scr_tau,
                                const int          num_threads)
{
    if (N == 0) return;

    const size_t out_stride = n_ij * n_ij;

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

                /* Magnitude screening with the FRESH per-cycle betas:
                 * ||B_contrib|| <= (|bkl|+|blk|) * ||T_kl||  and
                 * ||E_contrib|| <= ||UK_kl||   (PNO-overlap 2-norms <= 1).
                 * scr_* == NULL -> screening off (exact). */
                if (scr_t2n != NULL) {
                    const double _b = (bkl < 0 ? -bkl : bkl)
                                    + (blk < 0 ? -blk : blk);
                    if (_b * scr_t2n[n] + scr_ukn[n] < scr_tau) continue;
                }

                const double *Sn = S_master + S_off[n];
                const double *Tn = T_master + T_off[n];
                const double *Kn = K_master + K_off[n];
                const double *UKn;

                if (uk_hoisted) {
                    UKn = Kn;   /* per-cycle hoisted UK, T2-canonical layout */
                } else {
                    for (size_t b = 0; b < n_kl; b++) {
                        for (size_t d = 0; d < n_kl; d++) {
                            TT_minus[b * n_kl + d] = 2.0 * Tn[b * n_kl + d]
                                                      - Tn[d * n_kl + b];
                        }
                    }
                    if (!sm) {
                        for (size_t b = 0; b < n_kl; b++) {
                            for (size_t d = 0; d < n_kl; d++) {
                                TT_plus[b * n_kl + d] =
                                    2.0 * Tn[d * n_kl + b]
                                    - Tn[b * n_kl + d];
                            }
                        }
                    }
                    dgemm_(&T_flag, &N_flag,
                           &int_n_kl, &int_n_kl, &int_n_kl,
                           &one, Kn, &int_n_kl,
                           TT_minus, &int_n_kl,
                           &zero, UK, &int_n_kl);
                    if (!sm) {
                        dgemm_(&N_flag, &N_flag,
                               &int_n_kl, &int_n_kl, &int_n_kl,
                               &one, Kn, &int_n_kl,
                               TT_plus, &int_n_kl,
                               &one, UK, &int_n_kl);
                    }
                    UKn = UK;
                }

                if (sm) {
                    for (size_t e = 0; e < n_kl * n_kl; e++) {
                        TB[e] = bkl * Tn[e];
                    }
                } else {
                    for (size_t b = 0; b < n_kl; b++) {
                        for (size_t c = 0; c < n_kl; c++) {
                            TB[b * n_kl + c] = bkl * Tn[b * n_kl + c]
                                              + blk * Tn[c * n_kl + b];
                        }
                    }
                }

                dgemm_(&N_flag, &N_flag,
                       &int_n_kl, &int_n_ij, &int_n_kl,
                       &one, TB, &int_n_kl,
                       Sn, &int_n_kl,
                       &zero, STB, &int_n_kl);
                dgemm_(&N_flag, &N_flag,
                       &int_n_kl, &int_n_ij, &int_n_kl,
                       &one, UKn, &int_n_kl,
                       Sn, &int_n_kl,
                       &zero, SUK, &int_n_kl);
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

/* DLPNObe_kernel_slotcat — cross-bucket, slot-sorted BE accumulation.
 *
 * The bucketed kernels launch one OMP region per (n_ij, n_kl) shape
 * (4451 buckets / cycle at MOBH35-12 SVP, ~200 items each): threads
 * starve on the few populated slots and every region re-mallocs its
 * scratch.  Here ALL items are pre-sorted by their global output slot
 * (one slot = one ij pair) and one parallel region runs dynamic over
 * slots.  In addition, the second half-transform is batched: the
 * slot's S blocks live contiguously in S_cat as one col-major
 * (K_tot x n_ij) panel (K_tot = sum of the slot's item n_kl), so
 *   out_B += S_cat^T @ H_B,   out_E += S_cat^T @ H_E
 * run as ONE fat dgemm per slot-chunk instead of 2 tiny dgemms per
 * item.  Per item only the TB build and 2 small dgemms (into the H
 * panels, strided ldc) remain.  FLOPs and math identical to
 * DLPNObe_kernel_gathered_v3 with uk_hoisted=1; betas are read
 * directly from B_tilde_flat (dict layout mirrored from the solver's
 * per-bucket refresh), with pack-time beta0 as the p<0 fallback.
 *
 * Layout contracts (all mirrored from the v3 call conventions):
 *   T_master + T_off[n]   row-major (n_kl, n_kl) T2 block
 *   UK_master + T_off[n]  row-major (n_kl, n_kl) hoisted UK block
 *   S_cat + slot_scat_off[s] + k0[n], ld = KT  col-major (n_kl, n_ij)
 *   out_B/out_E + slot_out_off[s]  (n_ij x n_ij), pre-zeroed, beta=1
 */
void DLPNObe_kernel_slotcat(const double *S_cat,
                            const double *T_master,
                            const double *UK_master,
                            const long   *T_off,        /* (N,) sorted */
                            const int    *item_nkl,     /* (N,) sorted */
                            const long   *item_k0,      /* (N,) sorted */
                            const unsigned char *same,  /* (N,) sorted */
                            const double *beta0_kl,     /* (N,) sorted */
                            const double *beta0_lk,     /* (N,) sorted */
                            const int    *p_ij,         /* (N,) sorted */
                            const int    *dense_k,      /* (N,) sorted */
                            const int    *dense_l,      /* (N,) sorted */
                            const double *B_tilde_flat,
                            const long   *b_tilde_off,  /* per pair p */
                            const int    *nlmo_arr,     /* per pair p */
                            const long   *slot_ptr,     /* (n_slots+1,) */
                            const long   *slot_scat_off,/* (n_slots,) */
                            const long   *slot_KT,      /* (n_slots,) */
                            const int    *slot_nij,     /* (n_slots,) */
                            const long   *slot_out_off, /* (n_slots,) */
                            double       *out_B,
                            double       *out_E,
                            const size_t  n_slots,
                            const int     max_nkl,
                            const int     max_nij,
                            const long    hcap,         /* H panel rows */
                            const int     num_threads)
{
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;

#pragma omp parallel num_threads(num_threads)
    {
        double *TB  = (double *)malloc(sizeof(double) * (size_t)max_nkl * max_nkl);
        double *H_B = (double *)malloc(sizeof(double) * (size_t)hcap * max_nij);
        double *H_E = (double *)malloc(sizeof(double) * (size_t)hcap * max_nij);

#pragma omp for schedule(dynamic, 1)
        for (size_t s = 0; s < n_slots; s++) {
            const long i_lo = slot_ptr[s];
            const long i_hi = slot_ptr[s + 1];
            if (i_lo == i_hi) continue;

            const int  nij = slot_nij[s];
            const long KT  = slot_KT[s];
            const double *Scat = S_cat + slot_scat_off[s];
            double *out_Bt = out_B + slot_out_off[s];
            double *out_Et = out_E + slot_out_off[s];
            const int int_nij = nij;
            const int int_KT_ld = (int)KT;
            const int int_hcap = (int)hcap;

            long chunk_k0 = item_k0[i_lo];   /* first S_cat row of chunk */
            long cur_end  = chunk_k0;        /* rows filled so far       */

            for (long n = i_lo; n <= i_hi; n++) {
                const int nkl = (n < i_hi) ? item_nkl[n] : 0;
                const long k0 = (n < i_hi) ? item_k0[n] : cur_end;

                /* Flush the fat dgemms when the H panel would overflow
                 * (or at end of slot). */
                if (n == i_hi || k0 - chunk_k0 + nkl > hcap) {
                    const int used_k = (int)(cur_end - chunk_k0);
                    if (used_k > 0) {
                        dgemm_(&T_flag, &N_flag,
                               &int_nij, &int_nij, &used_k,
                               &one, Scat + chunk_k0, &int_KT_ld,
                               H_B, &int_hcap,
                               &one, out_Bt, &int_nij);
                        dgemm_(&T_flag, &N_flag,
                               &int_nij, &int_nij, &used_k,
                               &one, Scat + chunk_k0, &int_KT_ld,
                               H_E, &int_hcap,
                               &one, out_Et, &int_nij);
                    }
                    if (n == i_hi) break;
                    chunk_k0 = k0;
                    cur_end  = k0;
                }

                const double *Tn  = T_master  + T_off[n];
                const double *UKn = UK_master + T_off[n];
                const int int_nkl = nkl;
                const long local_k = k0 - chunk_k0;

                /* Betas from the CURRENT B_tilde (phase-5 fresh); fall
                 * back to pack-time values when the pair has no native
                 * B_tilde row (mirrors the per-bucket refresh). */
                double bkl = beta0_kl[n];
                double blk = beta0_lk[n];
                const int p = p_ij[n];
                if (p >= 0 && nlmo_arr[p] > 0) {
                    const int nlmo_p = nlmo_arr[p];
                    const long boff = b_tilde_off[p];
                    const int dk = dense_k[n], dl = dense_l[n];
                    bkl = B_tilde_flat[boff + (long)dk * nlmo_p + dl];
                    blk = (dk == dl) ? 0.0
                        : B_tilde_flat[boff + (long)dl * nlmo_p + dk];
                }

                if (same[n]) {
                    for (size_t e = 0; e < (size_t)nkl * nkl; e++) {
                        TB[e] = bkl * Tn[e];
                    }
                } else {
                    for (int b = 0; b < nkl; b++) {
                        for (int c = 0; c < nkl; c++) {
                            TB[b * nkl + c] = bkl * Tn[b * nkl + c]
                                             + blk * Tn[c * nkl + b];
                        }
                    }
                }

                dgemm_(&N_flag, &N_flag,
                       &int_nkl, &int_nij, &int_nkl,
                       &one, TB, &int_nkl,
                       Scat + k0, &int_KT_ld,
                       &zero, H_B + local_k, &int_hcap);
                dgemm_(&N_flag, &N_flag,
                       &int_nkl, &int_nij, &int_nkl,
                       &one, (double *)UKn, &int_nkl,
                       Scat + k0, &int_KT_ld,
                       &zero, H_E + local_k, &int_hcap);
                cur_end = k0 + nkl;
            }
        }

        free(TB);
        free(H_B);
        free(H_E);
    }
}
