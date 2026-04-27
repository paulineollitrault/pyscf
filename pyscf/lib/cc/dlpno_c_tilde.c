/* DLPNO-CCSD compute_C_tilde Phase 1 (Terms 1 + 2): per-pair batched kernel.
 *
 * Full-C cycle session 4 port. Mirrors Psi4 ccsd.cc:1809 (compute_C_tilde
 * Terms 1+2) and the existing Cython kernel
 * _c_tilde_ph1_batched_cy.pyx::c_tilde_ph1_batched.
 *
 * Math per pair (k, i) — flat-buffer-batched over all ordered pairs:
 *
 *   Term 2  (Psi4 ccsd.cc:1841-1844, K_tilde_chem precomputed in cc_ints):
 *     C_out[a, b]  = sum_a' K_tilde_chem[a', a*n_pno + b] * t1_i[a']
 *
 *   Term 1  (Psi4 ccsd.cc:1847, K_bar_chem_slice = K_bar_chem[ll_idx]):
 *     C_out[a, b] -= sum_l T1_local[l, a] * K_bar_chem_slice[l, b]
 *
 * Both reduce to one BLAS dgemv (Term 2) + one dgemm (Term 1) per pair —
 * matching the Cython kernel byte-for-byte. Outer #pragma omp parallel
 * for over ordered-pair index p; per-pair work is serial small BLAS.
 *
 * Phase 2 (Terms 3+4) lives in a separate kernel
 * (DLPNOcompute_C_tilde_ph2_*) — TODO in a follow-up session.
 *
 * Entry shapes (all C-contiguous double):
 *   K_tilde_chem_flat | offsets[N+1]  — per pair (n_pno, n_pno²)
 *   K_bar_chem_slice_flat | offsets   — per pair (n_domain, n_pno)
 *   t1_ki_flat | offsets              — per pair (n_pno,)
 *   T1_local_flat | offsets           — per pair (n_domain, n_pno)
 *   n_pno_arr[N], n_domain_arr[N]     — int per pair
 *   C_flat | offsets                  — output, per pair (n_pno, n_pno),
 *                                        fully overwritten by this kernel
 */

#include <stddef.h>
#include "vhf/fblas.h"

void DLPNOcompute_C_tilde_ph1_batched(
        const double *K_tilde_chem_flat,
        const long   *K_tilde_chem_offsets,
        const double *K_bar_chem_slice_flat,
        const long   *K_bar_chem_slice_offsets,
        const double *t1_ki_flat,
        const long   *t1_ki_offsets,
        const double *T1_local_flat,
        const long   *T1_local_offsets,
        const int    *n_pno_arr,
        const int    *n_domain_arr,
        double       *C_flat,
        const long   *C_offsets,
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

        const double *K_tilde_chem     =
            K_tilde_chem_flat     + K_tilde_chem_offsets[p];
        const double *K_bar_chem_slice =
            K_bar_chem_slice_flat + K_bar_chem_slice_offsets[p];
        const double *t1               =
            t1_ki_flat            + t1_ki_offsets[p];
        const double *T1_local         =
            T1_local_flat         + T1_local_offsets[p];
        double       *C_out            = C_flat + C_offsets[p];

        /* Term 2: C[ab] = sum_a' K_tilde_chem[a', ab] * t1[a']
         * K_tilde_chem is row-major (n_pno, n_pno²). Column-major view is
         * (n_pno², n_pno) with lda = n_pno². Then the row-major C.ravel()
         * equals K_tilde.T @ t1 ≡ col-major K_col @ t1 (no trans).
         * dgemv('N', m=n_pno², n=n_pno, 1, K_tilde, lda=n_pno²,
         *       t1, 1, 0, C_out, 1)
         */
        dgemv_(&N_flag, &int_npno2, &int_n_pno,
               &one, K_tilde_chem, &int_npno2,
               t1, &int_one,
               &zero, C_out, &int_one);

        /* Term 1: C[a, b] -= sum_l T1_local[l, a] * K_bar_chem_slice[l, b]
         * Both row-major (n_domain, n_pno); result row-major (n_pno, n_pno).
         * Column-major formulation:
         *   C_col[b, a] = -K_bar_col[b, l] * T1_col[l, a]   + 1*C_col
         * dgemm('N', 'T', n_pno, n_pno, n_domain, -1,
         *       K_bar_chem_slice, n_pno, T1_local, n_pno, 1, C_out, n_pno)
         */
        dgemm_(&N_flag, &T_flag,
               &int_n_pno, &int_n_pno, &int_n_domain,
               &neg_one, K_bar_chem_slice, &int_n_pno,
               T1_local, &int_n_pno,
               &one, C_out, &int_n_pno);
    }
}
