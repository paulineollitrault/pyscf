/* DLPNO-CCSD compute_D_tilde Phase 1 (Terms 1 + 2): per-pair batched kernel.
 *
 * Full-C cycle session 5 port. Mirrors Psi4 ccsd.cc:1991 (compute_D_tilde
 * Terms 1+2 — lines 1898-1913) and the existing Cython kernel
 * _d_tilde_ph1_batched_cy.pyx::d_tilde_ph1_batched.
 *
 * Math per ordered pair (i, k) — flat-buffer-batched over all pairs:
 *
 *   Term 2  (Psi4 1898-1908: K_tilde_chem and T1 contractions, transposed):
 *     part1[j, p] = sum_q K_tilde_chem[j, p*n_pno + q] * t1_i[q]
 *     part2[a, b] = sum_c t1_i[c] * K_tilde_chem[c, a*n_pno + b]
 *     D[a, b]    +=  2 * part1[b, a] - part2[b, a]
 *
 *   Term 1  (Psi4 1910-1913: T_n_ij × L_bar_temp):
 *     M_static[l, c] = 2 * K_bar_ij_or_ji[l, c] - K_bar_chem[l, c]   (precomputed)
 *     D[a, b]       -= sum_l T1_rows[l, a] * M_static[l, b]
 *
 * Reduces to two BLAS dgemv (Term 2 part1/part2) + one small accumulate
 * loop + one BLAS dgemm (Term 1) per pair — matches the Cython kernel
 * byte-for-byte. Outer #pragma omp parallel for over ordered-pair index
 * p; per-pair work is serial small BLAS with stack scratch.
 *
 * Phase 2 (Terms 3+4) shares the t3/t4 plan-cached kernels with C_tilde
 * and is deferred to the same follow-up session.
 *
 * Entry shapes (all C-contiguous double):
 *   K_tilde_chem_flat | offsets[N+1]   per pair (n_pno, n_pno²)
 *   M_static_flat | offsets            per pair (n_domain, n_pno)
 *   t1_flat | offsets                  per pair (n_pno,)   — t1[i] in pair PNO
 *   T1_rows_flat | offsets             per pair (n_domain, n_pno)
 *   n_pno_arr[N], n_domain_arr[N]      int per pair
 *   D_flat | offsets                   output, per pair (n_pno, n_pno),
 *                                      fully overwritten by this kernel.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

void DLPNOcompute_D_tilde_ph1_batched(
        const double *K_tilde_chem_flat,
        const long   *K_tilde_chem_offsets,
        const double *M_static_flat,
        const long   *M_static_offsets,
        const double *t1_flat,
        const long   *t1_offsets,
        const double *T1_rows_flat,
        const long   *T1_rows_offsets,
        const int    *n_pno_arr,
        const int    *n_domain_arr,
        double       *D_flat,
        const long   *D_offsets,
        const size_t  N)
{
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0, neg_one = -1.0;
    const int int_one = 1;

#pragma omp parallel for schedule(dynamic, 1)
    for (size_t p = 0; p < N; p++) {
        const int n_pno    = n_pno_arr[p];
        const int n_domain = n_domain_arr[p];
        int int_npno2      = n_pno * n_pno;
        int int_n_pno      = n_pno;
        int int_n_domain   = n_domain;

        const double *K_tilde_chem =
            K_tilde_chem_flat + K_tilde_chem_offsets[p];
        const double *M_static =
            M_static_flat + M_static_offsets[p];
        const double *t1 = t1_flat + t1_offsets[p];
        const double *T1_rows = T1_rows_flat + T1_rows_offsets[p];
        double       *D_out = D_flat + D_offsets[p];

        /* Per-thread scratch for the two Term-2 partial vectors.
         * Sizes are small (n_pno² ≤ ~625 doubles ≈ 5 KB each) — malloc
         * once per pair. Total across N pairs is fine since N ≤ ~thousands
         * and the kernel runs once per CCSD cycle. */
        double *part1 = (double *)malloc(sizeof(double) * (size_t)int_npno2);
        double *part2 = (double *)malloc(sizeof(double) * (size_t)int_npno2);

        /* Init D = 0 */
        memset(D_out, 0, sizeof(double) * (size_t)int_npno2);

        /* Term 2 part 1: part1[j*n_pno + p] = sum_q K_tilde[j, p*n_pno + q] * T1[q]
         * Row-major K_tilde (n_pno, n_pno²) viewed col-major as (n_pno², n_pno)
         * with lda=n_pno. dgemv with 'T' transpose does:
         *   part1[i] = sum_j K_tilde_col[j, i] * T1[j] = sum_j K_tilde[i, j] * T1[j]?
         * No — the Cython kernel does:
         *   dgemv('T', m=n_pno, n=n_pno², 1, K_tilde, lda=n_pno, T1, 1, 0, part1, 1)
         * which interprets K_tilde col-major as (n_pno, n_pno²) with lda=n_pno;
         * with 'T', part1[k] = sum_q K_tilde_col[q, k] * T1[q] for k in [0, n_pno²).
         * In row-major, K_tilde_col[q, k] = K_tilde_row[q, k] when lda=n_pno is
         * the row stride ... matches the Cython byte-for-byte; we follow the same
         * BLAS args. */
        dgemv_(&T_flag, &int_n_pno, &int_npno2,
               &one, K_tilde_chem, &int_n_pno,
               t1, &int_one,
               &zero, part1, &int_one);

        /* Term 2 part 2: part2[a*n_pno + b] = sum_c T1[c] * K_tilde[c, a*n_pno + b]
         *   dgemv('N', m=n_pno², n=n_pno, 1, K_tilde, lda=n_pno², T1, 1, 0, part2, 1)
         */
        dgemv_(&N_flag, &int_npno2, &int_n_pno,
               &one, K_tilde_chem, &int_npno2,
               t1, &int_one,
               &zero, part2, &int_one);

        /* Combine: D[a, b] += 2 * part1[b, a] - part2[b, a]  (= transpose-add) */
        for (int a = 0; a < n_pno; a++) {
            for (int b = 0; b < n_pno; b++) {
                D_out[a * n_pno + b] +=
                    2.0 * part1[b * n_pno + a] - part2[b * n_pno + a];
            }
        }

        /* Term 1: D[a, b] -= sum_l T1_rows[l, a] * M_static[l, b]
         * dgemm('N', 'T', n_pno, n_pno, n_domain, -1,
         *       M_static, n_pno, T1_rows, n_pno, 1, D_out, n_pno)
         */
        dgemm_(&N_flag, &T_flag,
               &int_n_pno, &int_n_pno, &int_n_domain,
               &neg_one, M_static, &int_n_pno,
               T1_rows, &int_n_pno,
               &one, D_out, &int_n_pno);

        free(part1);
        free(part2);
    }
}
