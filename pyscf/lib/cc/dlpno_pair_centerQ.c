/* DLPNO-CCSD compute_cc_integrals_sparse: per-pair-per-centerQ inner kernel.
 *
 * BLAS port (2026-04-30): pre-gather fancy-indexed inputs to contiguous
 * tensors, then DGEMM the matmuls.  At MKL-link with JIT-GEMM, ~3-4x
 * faster than the previous fancy-index hand-rolled loops on water-10.
 *
 * Math (per pair, per centerQ; matches local_df.py:736-823 line-by-line):
 *
 *   raw_io[local_Q[q], ext_kept_lmos[k]] = qij_b[q, i_s, ext_kept_pos[k]]
 *   raw_jo[local_Q[q], ext_kept_lmos[k]] = qij_b[q, j_s, ext_kept_pos[k]]
 *   raw_pair[local_Q[q]]                 = qij_b[q, i_s, j_s]
 *
 *   raw_iv[local_Q[q], a] = sum_u qia_b[q, i_s, ij_u_in_Q[u]] * X[u, a]
 *   raw_jv[local_Q[q], a] = sum_u qia_b[q, j_s, ij_u_in_Q[u]] * X[u, a]
 *
 *   raw_ma[local_Q[q], ext_kept_lmos[k], a]
 *       = sum_u qia_b[q, ext_kept_pos[k], ij_u_in_Q[u]] * X[u, a]
 *
 *   raw_ab[local_Q[q], a, b]
 *       = sum_{u, v} X[u, a] * qab_b[q, ij_u_in_Q[u], ij_u_in_Q[v]] * X[v, b]
 *
 *   proj_ij_out[q, a, v]
 *       = sum_u X[u, a] * qab_b[q, ij_u_in_Q[u], v]
 *
 * Strategy:
 *   1. For raw_iv/jv/ma: pre-gather qia[lmo, ij_u_in_Q[*]] for the relevant
 *      LMO row → contiguous (nQp, npp_ij), then DGEMM with X.
 *   2. For raw_ab: pre-gather qab[ij_u_in_Q[*], ij_u_in_Q[*]] →
 *      contiguous (nQp, npp_ij, npp_ij); DGEMM with X twice.
 *   3. For proj_ij_out: pre-gather qab[ij_u_in_Q[*], :] → (nQp, npp_ij, np_full);
 *      DGEMM with X.
 *
 * Shapes: see header above.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

void DLPNOpair_centerQ_step(
        const double *qij_atom_full,   /* (nQ_at_atom, nl, nl) */
        const double *qia_atom_full,   /* (nQ_at_atom, nl, np_full) */
        const double *qab_atom_full,   /* (nQ_at_atom, np_full, np_full) */
        const long   *local_Q,
        const long   *atom_pos,        /* (nQp,) — page index into atom stack */
        const int     i_s,
        const int     j_s,
        const long   *ij_u_in_Q,
        const long   *ext_kept_pos,
        const long   *ext_kept_lmos,
        const double *X_ij_slice,
        const size_t  nQp,
        const size_t  nl,
        const size_t  np_full,
        const size_t  npno,
        const size_t  npp_ij,
        const size_t  n_kept,
        const size_t  n_local,
        const size_t  nlmo_p,
        double       *raw_io,
        double       *raw_jo,
        double       *raw_iv,
        double       *raw_jv,
        double       *raw_pair,
        double       *raw_ma,
        double       *raw_ab,
        double       *proj_ij_out)
{
    const double *qij_b = qij_atom_full;
    const double *qia_b = qia_atom_full;
    const double *qab_b = qab_atom_full;

    const size_t qij_q = nl * nl;
    const size_t qij_l = nl;
    const size_t qia_q = nl * np_full;
    const size_t qia_l = np_full;
    const size_t qab_q = np_full * np_full;
    const size_t qab_u = np_full;
    const size_t io_row = nlmo_p;
    const size_t iv_row = npno;
    const size_t ma_row = nlmo_p * npno;
    const size_t ma_lmo = npno;
    const size_t ab_row = npno * npno;
    const size_t X_row  = npno;
    const size_t proj_q = npno * np_full;

    const int has_i = (i_s >= 0);
    const int has_j = (j_s >= 0);
    const int has_pair_paos = (npp_ij > 0);

    /* ------------------------------------------------------------------
     * Step 1: raw_io / raw_jo / raw_pair — pure scatter, no BLAS.
     * ------------------------------------------------------------------ */
    if (n_kept > 0 && has_i) {
        for (size_t q = 0; q < nQp; q++) {
            const size_t row_lq = (size_t)local_Q[q];
            const size_t pg = (size_t)atom_pos[q];
            const double *qij_qi = qij_b + pg * qij_q + (size_t)i_s * qij_l;
            double *out_row = raw_io + row_lq * io_row;
            for (size_t k = 0; k < n_kept; k++) {
                out_row[ext_kept_lmos[k]] = qij_qi[ext_kept_pos[k]];
            }
        }
    }
    if (n_kept > 0 && has_j) {
        for (size_t q = 0; q < nQp; q++) {
            const size_t row_lq = (size_t)local_Q[q];
            const size_t pg = (size_t)atom_pos[q];
            const double *qij_qj = qij_b + pg * qij_q + (size_t)j_s * qij_l;
            double *out_row = raw_jo + row_lq * io_row;
            for (size_t k = 0; k < n_kept; k++) {
                out_row[ext_kept_lmos[k]] = qij_qj[ext_kept_pos[k]];
            }
        }
    }
    if (has_i && has_j) {
        for (size_t q = 0; q < nQp; q++) {
            const size_t row_lq = (size_t)local_Q[q];
            const size_t pg = (size_t)atom_pos[q];
            raw_pair[row_lq] = qij_b[pg * qij_q + (size_t)i_s * qij_l + (size_t)j_s];
        }
    }

    if (!has_pair_paos) return;

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;
    int int_npno = (int)npno, int_npp_ij = (int)npp_ij;
    int int_nQp = (int)nQp;

    /* ------------------------------------------------------------------
     * Step 2: raw_iv / raw_jv via DGEMM after row-gather.
     *
     * For each Q, gather qia[i_s, ij_u_in_Q[*]] to size npp_ij (contig).
     * Stack across Q's: qia_i_stack[Q, u] = qia[atom_pos[Q]][i_s][ij_u_in_Q[u]].
     * Then raw_iv_local = qia_i_stack @ X_ij_slice (nQp, npp_ij) @ (npp_ij, npno).
     * ------------------------------------------------------------------ */
    if (has_i || has_j) {
        const size_t stack_sz = nQp * npp_ij;
        double *qia_i_stack = (has_i) ? (double *)malloc(sizeof(double) * stack_sz) : NULL;
        double *qia_j_stack = (has_j) ? (double *)malloc(sizeof(double) * stack_sz) : NULL;
        double *iv_local    = (has_i) ? (double *)malloc(sizeof(double) * nQp * npno) : NULL;
        double *jv_local    = (has_j) ? (double *)malloc(sizeof(double) * nQp * npno) : NULL;

        for (size_t q = 0; q < nQp; q++) {
            const size_t pg = (size_t)atom_pos[q];
            const double *qia_q_ptr = qia_b + pg * qia_q;
            if (has_i) {
                const double *qia_qi = qia_q_ptr + (size_t)i_s * qia_l;
                double *out = qia_i_stack + q * npp_ij;
                for (size_t u = 0; u < npp_ij; u++) {
                    out[u] = qia_qi[ij_u_in_Q[u]];
                }
            }
            if (has_j) {
                const double *qia_qj = qia_q_ptr + (size_t)j_s * qia_l;
                double *out = qia_j_stack + q * npp_ij;
                for (size_t u = 0; u < npp_ij; u++) {
                    out[u] = qia_qj[ij_u_in_Q[u]];
                }
            }
        }

        /* iv_local[Q, a] = sum_u qia_i_stack[Q, u] * X[u, a]
         * = qia_i_stack (nQp, npp_ij) @ X_ij_slice (npp_ij, npno)
         * F: iv_F[a, Q] = sum_u X_F[a, u] * qia_F[u, Q] = X_F @ qia_F
         * dgemm('N', 'N', npno, nQp, npp_ij, 1, X, npno, qia, npp_ij, 0, iv, npno)
         */
        if (has_i) {
            dgemm_(&N_flag, &N_flag,
                   &int_npno, &int_nQp, &int_npp_ij,
                   &one, X_ij_slice, &int_npno,
                   qia_i_stack, &int_npp_ij,
                   &zero, iv_local, &int_npno);
            for (size_t q = 0; q < nQp; q++) {
                memcpy(raw_iv + (size_t)local_Q[q] * iv_row,
                       iv_local + q * npno,
                       sizeof(double) * npno);
            }
        }
        if (has_j) {
            dgemm_(&N_flag, &N_flag,
                   &int_npno, &int_nQp, &int_npp_ij,
                   &one, X_ij_slice, &int_npno,
                   qia_j_stack, &int_npp_ij,
                   &zero, jv_local, &int_npno);
            for (size_t q = 0; q < nQp; q++) {
                memcpy(raw_jv + (size_t)local_Q[q] * iv_row,
                       jv_local + q * npno,
                       sizeof(double) * npno);
            }
        }
        if (qia_i_stack) free(qia_i_stack);
        if (qia_j_stack) free(qia_j_stack);
        if (iv_local) free(iv_local);
        if (jv_local) free(jv_local);
    }

    /* ------------------------------------------------------------------
     * Step 3: raw_ma — for each kept LMO row, build (nQp, npp_ij) gather +
     * DGEMM with X.  All n_kept rows share the same X, so we can stack
     * the LMO axis: qia_k_stack[Q, k, u] then ONE DGEMM gives result
     * (Q, k, a) which is scattered into raw_ma.
     * ------------------------------------------------------------------ */
    if (n_kept > 0) {
        const size_t stack_sz = nQp * n_kept * npp_ij;
        double *qia_k_stack = (double *)malloc(sizeof(double) * stack_sz);
        double *ma_local    = (double *)malloc(sizeof(double) * nQp * n_kept * npno);

        for (size_t q = 0; q < nQp; q++) {
            const size_t pg = (size_t)atom_pos[q];
            const double *qia_q_ptr = qia_b + pg * qia_q;
            for (size_t k = 0; k < n_kept; k++) {
                const long lmo_pos = ext_kept_pos[k];
                const double *qia_qk = qia_q_ptr + (size_t)lmo_pos * qia_l;
                double *out = qia_k_stack + (q * n_kept + k) * npp_ij;
                for (size_t u = 0; u < npp_ij; u++) {
                    out[u] = qia_qk[ij_u_in_Q[u]];
                }
            }
        }

        /* ma_local[(Q*n_kept + k), a] = sum_u qia_k_stack[(Q*n_kept + k), u] * X[u, a]
         * Big DGEMM: (nQp*n_kept, npp_ij) @ (npp_ij, npno) → (nQp*n_kept, npno).
         */
        int int_M = (int)(nQp * n_kept);
        dgemm_(&N_flag, &N_flag,
               &int_npno, &int_M, &int_npp_ij,
               &one, X_ij_slice, &int_npno,
               qia_k_stack, &int_npp_ij,
               &zero, ma_local, &int_npno);

        /* Scatter ma_local[Q, k, a] → raw_ma[local_Q[Q], ext_kept_lmos[k], a] */
        for (size_t q = 0; q < nQp; q++) {
            const size_t row_lq = (size_t)local_Q[q];
            double *ma_out_row = raw_ma + row_lq * ma_row;
            const double *ma_local_q = ma_local + q * n_kept * npno;
            for (size_t k = 0; k < n_kept; k++) {
                memcpy(ma_out_row + (size_t)ext_kept_lmos[k] * ma_lmo,
                       ma_local_q + k * npno,
                       sizeof(double) * npno);
            }
        }

        free(qia_k_stack);
        free(ma_local);
    }

    /* ------------------------------------------------------------------
     * Step 4: raw_ab — pre-gather qab[ij_u_in_Q[*], ij_u_in_Q[*]] for each Q
     * to (nQp, npp_ij, npp_ij), then per-Q two-step:
     *   tmp = X.T @ qab_gather              (npno, npp_ij)
     *   ab  = tmp @ X                        (npno, npno)
     * Total per Q: 2 small DGEMMs.
     * ------------------------------------------------------------------ */
    {
        double *qab_gather = (double *)malloc(sizeof(double) * npp_ij * npp_ij);
        double *tmp = (double *)malloc(sizeof(double) * npno * npp_ij);

        for (size_t q = 0; q < nQp; q++) {
            const size_t row_lq = (size_t)local_Q[q];
            const size_t pg = (size_t)atom_pos[q];
            const double *qab_q_ptr = qab_b + pg * qab_q;

            /* qab_gather[u, v] = qab[ij_u_in_Q[u], ij_u_in_Q[v]] */
            for (size_t u = 0; u < npp_ij; u++) {
                const double *qab_qu = qab_q_ptr + (size_t)ij_u_in_Q[u] * qab_u;
                double *gather_u = qab_gather + u * npp_ij;
                for (size_t v = 0; v < npp_ij; v++) {
                    gather_u[v] = qab_qu[ij_u_in_Q[v]];
                }
            }

            /* tmp[a, v] = sum_u X[u, a] * qab_gather[u, v]   = X^T @ qab_gather
             * Row-major: (npno, npp_ij) = (npp_ij, npno)^T @ (npp_ij, npp_ij).
             * F: tmp_F[v, a] = sum_u qab_gather_F[v, u] * X_F[a, u]
             *               = qab_gather_F @ X_F^T
             * dgemm('N', 'T', npp_ij, npno, npp_ij, 1, qab_gather, npp_ij,
             *       X, npno, 0, tmp, npp_ij)
             */
            dgemm_(&N_flag, &T_flag,
                   &int_npp_ij, &int_npno, &int_npp_ij,
                   &one, qab_gather, &int_npp_ij,
                   X_ij_slice, &int_npno,
                   &zero, tmp, &int_npp_ij);

            /* ab[a, b] = sum_v tmp[a, v] * X[v, b]    = tmp @ X
             * Row-major: (npno, npno) = (npno, npp_ij) @ (npp_ij, npno).
             * F: ab_F[b, a] = sum_v X_F[b, v] * tmp_F[v, a]  =  X_F @ tmp_F
             * dgemm('N', 'N', npno, npno, npp_ij, 1, X, npno, tmp, npp_ij,
             *       0, raw_ab + row_lq*ab_row, npno)
             */
            dgemm_(&N_flag, &N_flag,
                   &int_npno, &int_npno, &int_npp_ij,
                   &one, X_ij_slice, &int_npno,
                   tmp, &int_npp_ij,
                   &zero, raw_ab + row_lq * ab_row, &int_npno);
        }

        free(qab_gather);
        free(tmp);
    }

    /* ------------------------------------------------------------------
     * Step 5: proj_ij_out[q, a, v] = sum_u X[u, a] * qab[q, ij_u_in_Q[u], v]
     * Per Q: gather qab[ij_u_in_Q[u], :] to (npp_ij, np_full), then
     *   proj[a, v] = X.T @ qab_gather  (npno, np_full)
     * ------------------------------------------------------------------ */
    {
        int int_np_full = (int)np_full;
        double *qab_row_gather = (double *)malloc(sizeof(double) * npp_ij * np_full);

        for (size_t q = 0; q < nQp; q++) {
            const size_t pg = (size_t)atom_pos[q];
            const double *qab_q_ptr = qab_b + pg * qab_q;

            /* qab_row_gather[u, v] = qab[ij_u_in_Q[u], v] */
            for (size_t u = 0; u < npp_ij; u++) {
                memcpy(qab_row_gather + u * np_full,
                       qab_q_ptr + (size_t)ij_u_in_Q[u] * qab_u,
                       sizeof(double) * np_full);
            }

            /* proj[a, v] = sum_u X[u, a] * qab_row_gather[u, v]   = X^T @ qab_row_gather
             * F: proj_F[v, a] = sum_u qab_row_gather_F[v, u] * X_F[a, u]
             *                 = qab_row_gather_F @ X_F^T
             * dgemm('N', 'T', np_full, npno, npp_ij, 1, qab_row_gather, np_full,
             *       X, npno, 0, proj_ij_out + q*proj_q, np_full)
             */
            dgemm_(&N_flag, &T_flag,
                   &int_np_full, &int_npno, &int_npp_ij,
                   &one, qab_row_gather, &int_np_full,
                   X_ij_slice, &int_npno,
                   &zero, proj_ij_out + q * proj_q, &int_np_full);
        }

        free(qab_row_gather);
    }
}
