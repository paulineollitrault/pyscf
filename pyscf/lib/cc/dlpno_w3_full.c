/* DLPNO-(T) per-triple W3 + energy kernel — port of _w3_full_cy.pyx.
 *
 * Single C function `DLPNOcompute_w3_energy` that does the complete
 * per-triple body of `_w3_intermediate` (Phases 1-5):
 *
 *   Phase 1 (K_ovvv contribution): six dgemms K_ab[ip] @ t2_T[ir, iq],
 *           W = Σ_pidx trans[pidx](base[pidx]).
 *   Phase 2 (vooo per-m subtract): per m and r, compute
 *           T_il[r] = U.T @ T2 @ U (with optional transpose),
 *           subtract six K_ooov × T_il rank-1 contributions from W.
 *   Phase 3 (T = -W / D): elementwise tensor with energy denominator.
 *   Phase 4 (V = W + T1 disconnected): three K_*[a,b] outer products.
 *   Phase 5 (antisymmetrized energy):
 *           et = 8 Σ V*T - 4 Σ V(210)*T - 4 Σ V(021)*T - 4 Σ V(102)*T
 *                + 2 Σ V(120)*T + 2 Σ V(201)*T.
 *
 * Returns et / occ_denom.  Designed to be called from inside an OMP
 * parallel-for over triples (no internal threading; small allocations
 * via malloc which is thread-safe).
 *
 * Inputs (all row-major except where noted):
 *   K_ab_cache:    (3, n, n, n)         from ovL_sc[ip] @ vvL_sc, transposed
 *   t2_T_all:      (3, 3, n, n)         t2_block.transpose(0, 1, 3, 2)
 *   K_jk, K_ik, K_ij: (n, n)            ovL pairwise
 *   K_ooov:        (3, 3, n, m_dom)
 *   U_flat / U_offsets / n_pno_arr:     per-(r, l) U blocks (n_pno × n)
 *   T2_flat / T2_offsets:               per-(r, l) T2 blocks (n_pno × n_pno)
 *   transpose_flags:                    per-(r, l) flag (1 = transpose T_il)
 *   eps_occ:       (3,)
 *   eps_vir:       (n,)
 *   t1_sc:         (3, n)                zero if has_t1 = 0
 *   has_t1, occ_denom, n, m_dom, n_pno_max
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

double DLPNOcompute_w3_energy(
        const double *K_ab_cache,         /* (3, n, n, n) */
        const double *t2_T_all,           /* (3, 3, n, n) */
        const double *K_jk,               /* (n, n) */
        const double *K_ik,
        const double *K_ij,
        const double *K_ooov,             /* (3, 3, n, m_dom) */
        const double *U_flat,
        const long   *U_offsets,
        const long   *n_pno_arr,
        const double *T2_flat,
        const long   *T2_offsets,
        const signed char *transpose_flags,
        const double *eps_occ,
        const double *eps_vir,
        const double *t1_sc,
        const int has_t1,
        const int occ_denom,
        const int n,
        const int m_dom,
        const int n_pno_max)
{
    if (n == 0 || occ_denom == 0) return 0.0;

    const char N_ = 'N', T_ = 'T';
    const double one = 1.0, zero = 0.0;
    int int_n = n;
    int int_nn = n * n;

    const double D_occ = eps_occ[0] + eps_occ[1] + eps_occ[2];

    /* Scratch */
    const size_t n3 = (size_t)n * (size_t)n * (size_t)n;
    double *W        = (double *)calloc(n3, sizeof(double));
    double *V        = (double *)malloc(sizeof(double) * n3);
    double *T_ten    = (double *)malloc(sizeof(double) * n3);
    double *base_buf = (double *)malloc(sizeof(double) * n3);
    double *T_il     = (double *)calloc((size_t)3 * (size_t)n * (size_t)n, sizeof(double));
    const size_t T2U_rows = (n_pno_max > 0) ? (size_t)n_pno_max : 1;
    double *T2U      = (double *)malloc(sizeof(double) * T2U_rows * (size_t)n);

    /* ------------------------------------------------------------------
     * Phase 1: W = Σ_pidx trans[pidx](K_ab[ip] @ t2_T[ir, iq])
     * ------------------------------------------------------------------ */
    /* p_table = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)] */
    const int p_table_ip[6] = {0, 0, 1, 1, 2, 2};
    const int p_table_iq[6] = {1, 2, 0, 2, 0, 1};
    const int p_table_ir[6] = {2, 1, 2, 0, 1, 0};
    /* For each pidx, base = K_ab[ip] @ t2_T[ir, iq], then accumulate into W
     * with virtual-axis transposition matching the pidx's permutation. */
    for (int pidx = 0; pidx < 6; pidx++) {
        const int ip = p_table_ip[pidx];
        const int ir = p_table_ir[pidx];
        const int iq = p_table_iq[pidx];

        /* base[a, b, c] = Σ_f K_ab[ip, a, b, f] * t2_T[ir, iq, c, f]
         * Cython invocation:
         *   dgemm('N','N', n, n*n, n,
         *         1, t2_T_all[ir, iq], n,
         *            K_ab_cache[ip],   n,
         *         0, base_buf,         n)
         * Reuse the exact same call. */
        const double *K = K_ab_cache + (size_t)ip * n3;
        const double *t = t2_T_all + ((size_t)ir * 3 + (size_t)iq) * (size_t)n * (size_t)n;
        dgemm_(&N_, &N_, &int_n, &int_nn, &int_n,
               &one, t, &int_n, K, &int_n,
               &zero, base_buf, &int_n);

        /* Permuted accumulate */
        switch (pidx) {
        case 0:  /* W[a,b,c] += base[a,b,c] */
            for (size_t i = 0; i < n3; i++) W[i] += base_buf[i];
            break;
        case 1: { /* W[a,b,c] += base[a,c,b] */
            for (int a = 0; a < n; a++)
                for (int b = 0; b < n; b++)
                    for (int c = 0; c < n; c++)
                        W[((size_t)a*n + b)*n + c]
                          += base_buf[((size_t)a*n + c)*n + b];
            break;
        }
        case 2: { /* W[a,b,c] += base[b,a,c] */
            for (int a = 0; a < n; a++)
                for (int b = 0; b < n; b++)
                    for (int c = 0; c < n; c++)
                        W[((size_t)a*n + b)*n + c]
                          += base_buf[((size_t)b*n + a)*n + c];
            break;
        }
        case 3: { /* W[a,b,c] += base[b,c,a] */
            for (int a = 0; a < n; a++)
                for (int b = 0; b < n; b++)
                    for (int c = 0; c < n; c++)
                        W[((size_t)a*n + b)*n + c]
                          += base_buf[((size_t)b*n + c)*n + a];
            break;
        }
        case 4: { /* W[a,b,c] += base[c,a,b] */
            for (int a = 0; a < n; a++)
                for (int b = 0; b < n; b++)
                    for (int c = 0; c < n; c++)
                        W[((size_t)a*n + b)*n + c]
                          += base_buf[((size_t)c*n + a)*n + b];
            break;
        }
        case 5: { /* W[a,b,c] += base[c,b,a] */
            for (int a = 0; a < n; a++)
                for (int b = 0; b < n; b++)
                    for (int c = 0; c < n; c++)
                        W[((size_t)a*n + b)*n + c]
                          += base_buf[((size_t)c*n + b)*n + a];
            break;
        }
        }
    }

    /* ------------------------------------------------------------------
     * Phase 2: vooo per-m subtract.
     * For each l in 0..m_dom-1:
     *   For each r in 0..2:
     *     Build T_il[r] = U.T @ T2 @ U (transpose if flag).
     *   Then for each (a,b,c) accumulate the six rank-1 K_ooov × T_il
     *   contributions into W[a,b,c] (negative sign).
     * ------------------------------------------------------------------ */
    for (int l_ijk = 0; l_ijk < m_dom; l_ijk++) {
        for (int r = 0; r < 3; r++) {
            const int flat_idx = r * m_dom + l_ijk;
            const long n_pno = n_pno_arr[flat_idx];
            double *T_il_r = T_il + (size_t)r * (size_t)n * (size_t)n;
            if (n_pno == 0) {
                memset(T_il_r, 0, sizeof(double) * (size_t)n * (size_t)n);
                continue;
            }
            const long u_off = U_offsets[flat_idx];
            const long t2_off_v = T2_offsets[flat_idx];
            int int_n_pno = (int)n_pno;
            int ld_U = int_n;
            int ld_T2 = int_n_pno;
            int ld_T2U = int_n;
            int ld_T_il = int_n;

            /* T2U (n_pno, n) = T2 (n_pno, n_pno) @ U (n_pno, n)
             * Cython call: dgemm('N','N', n, n_pno, n_pno,
             *                    1, U[u_off], n,
             *                       T2[t2_off], n_pno,
             *                    0, T2U, n)
             */
            dgemm_(&N_, &N_, &int_n, &int_n_pno, &int_n_pno,
                   &one, U_flat + u_off, &ld_U,
                   T2_flat + t2_off_v, &ld_T2,
                   &zero, T2U, &ld_T2U);

            /* T_il[r] (n, n) = U.T @ T2U
             * Cython call: dgemm('N','T', n, n, n_pno,
             *                    1, T2U, n,
             *                       U[u_off], n,
             *                    0, T_il_r, n)
             */
            dgemm_(&N_, &T_, &int_n, &int_n, &int_n_pno,
                   &one, T2U, &ld_T2U,
                   U_flat + u_off, &ld_U,
                   &zero, T_il_r, &ld_T_il);

            if (transpose_flags[flat_idx] != 0) {
                for (int a = 0; a < n; a++) {
                    for (int b = a + 1; b < n; b++) {
                        double tmp = T_il_r[(size_t)a*n + b];
                        T_il_r[(size_t)a*n + b] = T_il_r[(size_t)b*n + a];
                        T_il_r[(size_t)b*n + a] = tmp;
                    }
                }
            }
        }

        /* Six rank-1 subtracts. */
        const double *T_il0 = T_il + 0 * (size_t)n * n;
        const double *T_il1 = T_il + 1 * (size_t)n * n;
        const double *T_il2 = T_il + 2 * (size_t)n * n;
        const double *K01 = K_ooov + ((size_t)0 * 3 + 1) * (size_t)n * (size_t)m_dom;
        const double *K02 = K_ooov + ((size_t)0 * 3 + 2) * (size_t)n * (size_t)m_dom;
        const double *K10 = K_ooov + ((size_t)1 * 3 + 0) * (size_t)n * (size_t)m_dom;
        const double *K12 = K_ooov + ((size_t)1 * 3 + 2) * (size_t)n * (size_t)m_dom;
        const double *K20 = K_ooov + ((size_t)2 * 3 + 0) * (size_t)n * (size_t)m_dom;
        const double *K21 = K_ooov + ((size_t)2 * 3 + 1) * (size_t)n * (size_t)m_dom;

        for (int a = 0; a < n; a++) {
            const double K01a = K01[(size_t)a * m_dom + l_ijk];
            const double K02a = K02[(size_t)a * m_dom + l_ijk];
            for (int b = 0; b < n; b++) {
                const double K10b = K10[(size_t)b * m_dom + l_ijk];
                const double K12b = K12[(size_t)b * m_dom + l_ijk];
                const double T1ab = T_il1[(size_t)a*n + b];
                const double T0ba = T_il0[(size_t)b*n + a];
                for (int c = 0; c < n; c++) {
                    const double K20c = K20[(size_t)c * m_dom + l_ijk];
                    const double K21c = K21[(size_t)c * m_dom + l_ijk];
                    double acc = K01a * T_il2[(size_t)b*n + c]
                               + K02a * T_il1[(size_t)c*n + b]
                               + K10b * T_il2[(size_t)a*n + c]
                               + K12b * T_il0[(size_t)c*n + a]
                               + K20c * T1ab
                               + K21c * T0ba;
                    W[((size_t)a*n + b)*n + c] -= acc;
                }
            }
        }
    }

    /* ------------------------------------------------------------------
     * Phase 3+4: T = -W / D ; V = W + T1 disconnected.
     * ------------------------------------------------------------------ */
    for (int a = 0; a < n; a++) {
        const double eps_a = eps_vir[a];
        const double t1_0_a = has_t1 ? t1_sc[(size_t)0*n + a] : 0.0;
        for (int b = 0; b < n; b++) {
            const double eps_ab = eps_a + eps_vir[b];
            const double t1_1_b = has_t1 ? t1_sc[(size_t)1*n + b] : 0.0;
            for (int c = 0; c < n; c++) {
                const double D_abc = eps_ab + eps_vir[c] - D_occ;
                const size_t idx = ((size_t)a*n + b)*n + c;
                const double W_abc = W[idx];
                T_ten[idx] = -W_abc / D_abc;
                if (has_t1) {
                    const double t1_2_c = t1_sc[(size_t)2*n + c];
                    V[idx] = W_abc
                           + t1_0_a * K_jk[(size_t)b*n + c]
                           + t1_1_b * K_ik[(size_t)a*n + c]
                           + t1_2_c * K_ij[(size_t)a*n + b];
                } else {
                    V[idx] = W_abc;
                }
            }
        }
    }

    /* ------------------------------------------------------------------
     * Phase 5: antisymmetrized energy.
     * ------------------------------------------------------------------ */
    double S0 = 0.0, S1 = 0.0, S2 = 0.0, S3 = 0.0, S4 = 0.0, S5 = 0.0;
    for (int a = 0; a < n; a++) {
        for (int b = 0; b < n; b++) {
            for (int c = 0; c < n; c++) {
                const double T_abc = T_ten[((size_t)a*n + b)*n + c];
                S0 += V[((size_t)a*n + b)*n + c] * T_abc;
                S1 += V[((size_t)c*n + b)*n + a] * T_abc;
                S2 += V[((size_t)a*n + c)*n + b] * T_abc;
                S3 += V[((size_t)b*n + a)*n + c] * T_abc;
                S4 += V[((size_t)c*n + a)*n + b] * T_abc;
                S5 += V[((size_t)b*n + c)*n + a] * T_abc;
            }
        }
    }
    const double et = 8.0 * S0 - 4.0 * S1 - 4.0 * S2 - 4.0 * S3
                    + 2.0 * S4 + 2.0 * S5;

    free(W); free(V); free(T_ten); free(base_buf); free(T_il); free(T2U);
    return et / (double)occ_denom;
}
