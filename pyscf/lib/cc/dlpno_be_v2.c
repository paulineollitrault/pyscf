/* DLPNO-CCSD T2 residual: B+E kernel v2 — fold per-item gather into C.
 *
 * Session 14a port. Same compute as DLPNObe_kernel (dlpno_be.c) but
 * eliminates the per-bucket Python loop that builds T_arr from
 * t2_pno_all._buffer and beta_kl/lk from B_tilde_per_ij. The kernel
 * now reads T directly from the FlatTensorStore buffer and beta from
 * a per-cycle flattened B_tilde buffer, both via pre-built absolute
 * offset tables.
 *
 * Plan-time invariants (built once per CCSD run):
 *   t2_off[N]       — absolute offset of each item's canonical T2 in
 *                     t2_pno_all._buffer
 *   B_kl_off[N]     — absolute offset into B_flat for β_kl
 *   B_lk_off[N]     — same for β_lk, or -1 if k == l (β_lk forced to 0)
 *
 * Per-cycle dynamic input:
 *   t2_buffer       — t2_pno_all._buffer (refreshed by DIIS)
 *   B_flat          — concatenated B_local matrices for all strong pairs
 *                     (built by Python wrapper, ~one cheap loop / cycle)
 *
 * Compute math identical to DLPNObe_kernel; see dlpno_be.c for the
 * derivation.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void DLPNObe_kernel_v2(const double *S,                /* (N, n_ij, n_kl) */
                       const double *t2_buffer,        /* FlatTensorStore */
                       const long   *t2_off,           /* (N,) — into t2_buffer */
                       const double *K,                /* (N, n_kl, n_kl) */
                       const double *B_flat,           /* per-cycle */
                       const long   *B_kl_off,         /* (N,) — into B_flat */
                       const long   *B_lk_off,         /* (N,) — or -1 */
                       const unsigned char *same,      /* (N,) */
                       const long   *idx,              /* (N,) */
                       double       *out_B,            /* (n_slots, n_ij, n_ij) */
                       double       *out_E,            /* (n_slots, n_ij, n_ij) */
                       const size_t  N,
                       const size_t  n_ij,
                       const size_t  n_kl,
                       const int     num_threads)
{
    if (N == 0) return;

    const size_t S_stride   = n_ij * n_kl;
    const size_t TK_stride  = n_kl * n_kl;
    const size_t out_stride = n_ij * n_ij;

    double *Bc = (double *)malloc(sizeof(double) * N * out_stride);
    double *Ec = (double *)malloc(sizeof(double) * N * out_stride);

#pragma omp parallel num_threads(num_threads)
    {
        double *UK  = (double *)malloc(sizeof(double) * n_kl * n_kl);
        double *STB = (double *)malloc(sizeof(double) * n_ij * n_kl);
        double *SUK = (double *)malloc(sizeof(double) * n_ij * n_kl);

#pragma omp for schedule(dynamic, 1)
        for (size_t n = 0; n < N; n++) {
            const double bkl = B_flat[B_kl_off[n]];
            const long   lk_off = B_lk_off[n];
            const double blk = (lk_off >= 0) ? B_flat[lk_off] : 0.0;
            const unsigned char sm = same[n];

            const double *Sn = S + n * S_stride;
            const double *Tn = t2_buffer + t2_off[n];   /* (n_kl, n_kl) */
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
