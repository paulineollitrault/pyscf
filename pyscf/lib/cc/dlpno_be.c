/* DLPNO-CCSD T2 residual: per-bucket B + E scatter-accumulator.
 *
 * Full-C cycle session 10 port. Mirrors the existing Cython kernel
 * _be_cy.pyx::be_kernel byte-for-byte. Called once per bucketed
 * (n_ij, n_kl) shape from compute_B_E_batched_v2.
 *
 * Per-item math (one (ij, kl) item with kl in pair_lmo_idx[ij]):
 *
 *   if same (k == l):
 *     UK[b, c]   = sum_d (2 T[b, d] - T[d, b]) * K[c, d]
 *   else:
 *     UK[b, c]   = sum_d (2 T[b, d] - T[d, b]) * K[c, d]
 *                + sum_d (2 T[d, b] - T[b, d]) * K[d, c]
 *
 *   STB[a, c]  = sum_b S[a, b] * TB[b, c]
 *      TB     = beta_kl * T                      (same)
 *             = beta_kl * T + beta_lk * T.T      (!same)
 *
 *   SUK[a, c]  = sum_b S[a, b] * UK[b, c]
 *   Bc[a, d]   = sum_c STB[a, c] * S[d, c]
 *   Ec[a, d]   = sum_c SUK[a, c] * S[d, c]
 *
 *   out_B[idx[n]] += Bc
 *   out_E[idx[n]] += Ec
 *
 * Two-stage layout matching Cython:
 *   Stage 1: parallel prange over n, compute per-item Bc/Ec into N-sized
 *            scratch (no race — disjoint writes).
 *   Stage 2: sequential scatter-add into out_B / out_E (race-free; idx
 *            may collide).
 *
 * Per-item scratch sizes: UK = n_kl², STB = SUK = n_ij*n_kl,
 *                          Bc = Ec = n_ij². All malloc'd internally
 * up-front for the whole batch.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

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

    /* Per-item scratch — Bc/Ec are (n_ij²); UK/STB/SUK live within one
     * item's iteration so we keep them per-thread, not per-item. */
    double *Bc = (double *)malloc(sizeof(double) * N * out_stride);
    double *Ec = (double *)malloc(sizeof(double) * N * out_stride);

#pragma omp parallel num_threads(num_threads)
    {
        /* Per-thread per-item working buffers (sized per-call: ~few KB) */
        double *UK  = (double *)malloc(sizeof(double) * n_kl * n_kl);
        double *STB = (double *)malloc(sizeof(double) * n_ij * n_kl);
        double *SUK = (double *)malloc(sizeof(double) * n_ij * n_kl);

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

            /* UK[b, c] */
            if (sm) {
                for (size_t b = 0; b < n_kl; b++) {
                    for (size_t c = 0; c < n_kl; c++) {
                        double uk_bc = 0.0;
                        for (size_t d = 0; d < n_kl; d++) {
                            uk_bc += (2.0 * Tn[b * n_kl + d] - Tn[d * n_kl + b])
                                     * Kn[c * n_kl + d];
                        }
                        UK[b * n_kl + c] = uk_bc;
                    }
                }
            } else {
                for (size_t b = 0; b < n_kl; b++) {
                    for (size_t c = 0; c < n_kl; c++) {
                        double uk_bc = 0.0;
                        for (size_t d = 0; d < n_kl; d++) {
                            uk_bc += (2.0 * Tn[b * n_kl + d] - Tn[d * n_kl + b])
                                     * Kn[c * n_kl + d];
                            uk_bc += (2.0 * Tn[d * n_kl + b] - Tn[b * n_kl + d])
                                     * Kn[d * n_kl + c];
                        }
                        UK[b * n_kl + c] = uk_bc;
                    }
                }
            }

            /* STB[a, c] = sum_b S[a, b] * TB[b, c] */
            if (sm) {
                for (size_t a = 0; a < n_ij; a++) {
                    for (size_t c = 0; c < n_kl; c++) {
                        double s = 0.0;
                        for (size_t b = 0; b < n_kl; b++) {
                            s += Sn[a * n_kl + b] * (bkl * Tn[b * n_kl + c]);
                        }
                        STB[a * n_kl + c] = s;
                    }
                }
            } else {
                for (size_t a = 0; a < n_ij; a++) {
                    for (size_t c = 0; c < n_kl; c++) {
                        double s = 0.0;
                        for (size_t b = 0; b < n_kl; b++) {
                            s += Sn[a * n_kl + b] *
                                 (bkl * Tn[b * n_kl + c] + blk * Tn[c * n_kl + b]);
                        }
                        STB[a * n_kl + c] = s;
                    }
                }
            }

            /* SUK[a, c] = sum_b S[a, b] * UK[b, c] */
            for (size_t a = 0; a < n_ij; a++) {
                for (size_t c = 0; c < n_kl; c++) {
                    double s = 0.0;
                    for (size_t b = 0; b < n_kl; b++) {
                        s += Sn[a * n_kl + b] * UK[b * n_kl + c];
                    }
                    SUK[a * n_kl + c] = s;
                }
            }

            /* Bc[a, d] = sum_c STB[a, c] * S[d, c]
             * Ec[a, d] = sum_c SUK[a, c] * S[d, c] */
            for (size_t a = 0; a < n_ij; a++) {
                for (size_t d = 0; d < n_ij; d++) {
                    double s_bc = 0.0, s_ec = 0.0;
                    for (size_t c = 0; c < n_kl; c++) {
                        s_bc += STB[a * n_kl + c] * Sn[d * n_kl + c];
                        s_ec += SUK[a * n_kl + c] * Sn[d * n_kl + c];
                    }
                    Bcn[a * n_ij + d] = s_bc;
                    Ecn[a * n_ij + d] = s_ec;
                }
            }
        }

        free(UK);
        free(STB);
        free(SUK);
    }

    /* Stage 2: sequential scatter-add */
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
