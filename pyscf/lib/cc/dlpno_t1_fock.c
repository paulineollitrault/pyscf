/* DLPNO-CCSD t1_fock per-pair Fab + d_ij/d_ji + Fia kernel.
 *
 * Full-C cycle session 6 port. Mirrors Psi4 ccsd.cc:1571 (t1_fock —
 * specifically Step 1: d_ij/d_ji scalars and Step 2: Fia/Fab dressing
 * over contracted indices, lines 1585-1656) and the existing Cython
 * kernel _t1_fock_batched_cy.pyx::t1_fock_batched.
 *
 * Math per pair (matches local_df.py:_per_pair):
 *
 *   Step 1 (Psi4 1596-1597):
 *     d_ij = 2 * sum(T1 * K_chem_l) - sum(T1 * K_ji_l)
 *     d_ji = 2 * sum(T1 * K_chem_l) - sum(T1 * K_ij_l)   (if i != j)
 *
 *   Step 2 (Psi4 1601-1654) — only for canonical strong pairs (i ≤ j):
 *     Fab[a, b]      = e_pno[a] * delta(a, b)
 *     gamma[L]       = sum_{m, a} Qma[L, m, a] * T1[m, a]
 *     Fab[a, b]     += 2 * sum_L gamma[L] * Qab[L, a, b]
 *     Y[L, b, m]     = sum_c Qab[L, b, c] * T1[m, c]
 *     Fab[b, c]     -= sum_{L, m} Y[L, b, m] * Qma[L, m, c]
 *     Fia[m, a]      = 2 * sum_L gamma[L] * Qma[L, m, a]
 *     Z[L, j, i]     = sum_c Qma[L, j, c] * T1[i, c]
 *     Fia[m', a]    -= sum_{L, i} Z[L, i, m'] * Qma[L, i, a]
 *     Fab[a, b]     -= sum_m T1[m, a] * Fia[m, b]
 *
 * Six BLAS calls (2 dgemv + 4 dgemm) + 2 transposes per pair, byte-for-byte
 * the same plan as the Cython reference. Outer #pragma omp parallel for
 * over ordered-pair index p; per-thread scratch passed in by the caller.
 *
 * Step 1 computes d_ij/d_ji for ALL pairs (including i > j and weak); the
 * caller scatters them into the global Fij_bar. Step 2 only touches
 * canonical strong pairs (gated by need_dji_arr / pair-walk in caller —
 * non-strong pairs come in with degenerate shapes that this kernel skips
 * via the Fab init path; matched to Cython behavior).
 *
 * Scratch buffer sizes per pair:
 *   gamma:   n_local
 *   Y_trans: n_local * npno * nlmo
 *   Y_alt:   n_local * nlmo * npno      (same size as Y_trans, transpose copy)
 *   Fia:     nlmo * npno
 *   Z:       n_local * nlmo * nlmo
 *   Z_xxx:   n_local * nlmo * nlmo
 * Caller pre-allocates as (num_threads, max_*) and we index by tid.
 */

#include <stddef.h>
#include <string.h>
#include "vhf/fblas.h"

#ifdef _OPENMP
#include <omp.h>
#endif

void DLPNOt1_fock_batched(
        const double *T1_flat,
        const long   *T1_offsets,
        const double *K_chem_flat,
        const long   *K_chem_offsets,
        const double *K_ji_flat,
        const long   *K_ji_offsets,
        const double *K_ij_flat,
        const long   *K_ij_offsets,
        const double *Qma_flat,
        const long   *Qma_offsets,
        const double *Qab_flat,
        const long   *Qab_offsets,
        const double *e_pno_flat,
        const long   *e_pno_offsets,
        const int    *nlmo_arr,
        const int    *npno_arr,
        const int    *n_local_arr,
        const int    *need_dji_arr,
        double       *gamma_scratch,
        const size_t  gamma_sc_stride,
        double       *Y_trans_scratch,
        const size_t  Y_trans_sc_stride,
        double       *Y_alt_scratch,
        const size_t  Y_alt_sc_stride,
        double       *Fia_scratch,
        const size_t  Fia_sc_stride,
        double       *Z_stacked_scratch,
        const size_t  Z_stacked_sc_stride,
        double       *Z_xxx_scratch,
        const size_t  Z_xxx_sc_stride,
        double       *d_flat,
        double       *Fab_flat,
        const long   *Fab_offsets,
        const size_t  N,
        const int     num_threads)
{
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0, two = 2.0, neg_one = -1.0;
    const int int_one = 1;

#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (size_t p = 0; p < N; p++) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const int nlmo    = nlmo_arr[p];
        const int npno    = npno_arr[p];
        const int n_local = n_local_arr[p];
        const int need_dji = need_dji_arr[p];

        const double *T1     = T1_flat     + T1_offsets[p];
        const double *K_chem = K_chem_flat + K_chem_offsets[p];
        const double *K_ji   = K_ji_flat   + K_ji_offsets[p];
        const double *K_ij   = K_ij_flat   + K_ij_offsets[p];
        const double *Qma    = Qma_flat    + Qma_offsets[p];
        const double *Qab    = Qab_flat    + Qab_offsets[p];
        const double *e_pno  = e_pno_flat  + e_pno_offsets[p];
        double       *d_out  = d_flat      + p * 2;
        double       *Fab    = Fab_flat    + Fab_offsets[p];

        double *gamma     = gamma_scratch     + (size_t)tid * gamma_sc_stride;
        double *Y_trans   = Y_trans_scratch   + (size_t)tid * Y_trans_sc_stride;
        double *Y_alt     = Y_alt_scratch     + (size_t)tid * Y_alt_sc_stride;
        double *Fia_bar   = Fia_scratch       + (size_t)tid * Fia_sc_stride;
        double *Z_stacked = Z_stacked_scratch + (size_t)tid * Z_stacked_sc_stride;
        double *Z_xxx     = Z_xxx_scratch     + (size_t)tid * Z_xxx_sc_stride;

        int int_nlmo         = nlmo;
        int int_npno         = npno;
        int int_n_local      = n_local;
        int int_nlmo_npno    = nlmo * npno;
        int int_npno2        = npno * npno;
        int int_n_local_npno = n_local * npno;
        int int_n_local_nlmo = n_local * nlmo;

        /* Step 1: d_ij = 2 * sum(T1 * K_chem) - sum(T1 * K_ji)
         *         d_ji = 2 * sum(T1 * K_chem) - sum(T1 * K_ij)   (if i != j) */
        double acc_chem = 0.0, acc_ji = 0.0;
        const int nm = nlmo * npno;
        for (int idx = 0; idx < nm; idx++) {
            acc_chem += T1[idx] * K_chem[idx];
            acc_ji   += T1[idx] * K_ji[idx];
        }
        d_out[0] = 2.0 * acc_chem - acc_ji;
        if (need_dji) {
            double acc_ij = 0.0;
            for (int idx = 0; idx < nm; idx++) {
                acc_ij += T1[idx] * K_ij[idx];
            }
            d_out[1] = 2.0 * acc_chem - acc_ij;
        }

        /* Step 2: Fab init to diag(e_pno) */
        memset(Fab, 0, sizeof(double) * (size_t)int_npno2);
        for (int a = 0; a < npno; a++) {
            Fab[a * npno + a] = e_pno[a];
        }

        /* gamma = Qma_flat @ T1.ravel()  (dgemv 'T') */
        dgemv_(&T_flag, &int_nlmo_npno, &int_n_local,
               &one, Qma, &int_nlmo_npno,
               T1, &int_one,
               &zero, gamma, &int_one);

        /* Fab += 2 * Qab_flat.T @ gamma  (dgemv 'N', beta=1 accumulate) */
        dgemv_(&N_flag, &int_npno2, &int_n_local,
               &two, Qab, &int_npno2,
               gamma, &int_one,
               &one, Fab, &int_one);

        /* Y_trans (n_local*npno, nlmo) = Qab_flat (n_local*npno, npno) @ T1.T (npno, nlmo)
         * Col-major: Y_col(nlmo, n_local*npno) = T1_col^T @ Qab_col */
        dgemm_(&T_flag, &N_flag,
               &int_nlmo, &int_n_local_npno, &int_npno,
               &one, T1, &int_npno,
               Qab, &int_npno,
               &zero, Y_trans, &int_nlmo);

        /* Transpose Y_trans[L, b, m] → Y_alt[L, m, b] */
        for (int L = 0; L < n_local; L++) {
            for (int m = 0; m < nlmo; m++) {
                for (int b = 0; b < npno; b++) {
                    Y_alt[(size_t)L * nlmo * npno + (size_t)m * npno + b] =
                        Y_trans[(size_t)L * npno * nlmo + (size_t)b * nlmo + m];
                }
            }
        }

        /* Fab -= Y_alt_flat.T @ Qma_flat */
        dgemm_(&N_flag, &T_flag,
               &int_npno, &int_npno, &int_n_local_nlmo,
               &neg_one, Qma, &int_npno,
               Y_alt, &int_npno,
               &one, Fab, &int_npno);

        /* Fia_bar = 2 * Qma_flat.T @ gamma  (dgemv 'N') */
        dgemv_(&N_flag, &int_nlmo_npno, &int_n_local,
               &two, Qma, &int_nlmo_npno,
               gamma, &int_one,
               &zero, Fia_bar, &int_one);

        /* Z_stacked = Qma_flat @ T1.T  (dgemm) */
        dgemm_(&T_flag, &N_flag,
               &int_nlmo, &int_n_local_nlmo, &int_npno,
               &one, T1, &int_npno,
               Qma, &int_npno,
               &zero, Z_stacked, &int_nlmo);

        /* Transpose Z_stacked[L, j, i] → Z_xxx[L, i, j] */
        for (int L = 0; L < n_local; L++) {
            for (int m = 0; m < nlmo; m++) {
                for (int mp = 0; mp < nlmo; mp++) {
                    Z_xxx[(size_t)L * nlmo * nlmo + (size_t)mp * nlmo + m] =
                        Z_stacked[(size_t)L * nlmo * nlmo + (size_t)m * nlmo + mp];
                }
            }
        }

        /* Fia_bar -= Z_xxx_flat.T @ Qma_flat */
        dgemm_(&N_flag, &T_flag,
               &int_npno, &int_nlmo, &int_n_local_nlmo,
               &neg_one, Qma, &int_npno,
               Z_xxx, &int_nlmo,
               &one, Fia_bar, &int_npno);

        /* Fab -= T1.T @ Fia_bar */
        dgemm_(&N_flag, &T_flag,
               &int_npno, &int_npno, &int_nlmo,
               &neg_one, Fia_bar, &int_npno,
               T1, &int_npno,
               &one, Fab, &int_npno);
    }
}
