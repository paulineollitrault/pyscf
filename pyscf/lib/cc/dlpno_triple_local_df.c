/* DLPNO-(T): per-triple local DF integral build.
 *
 * Replaces the Python body of `_build_triple_local_DF` in
 * pyscf/cc/dlpno_tccsd/lccsd_t.py.
 *
 * Per triple (i, j, k) and per aux atom-center, computes the raw
 * (pre-metric) integrals
 *
 *   ovL_raw[idx, a, q] = sum_u qia_atom[A][q_in_A, occ(idx), u_in_A]
 *                              * X_tno_ijk[u_in_triple, a]
 *
 *   vvL_raw[a, b, q]   = sum_{u,v} X[u, a]
 *                                 * qab_atom[A][q_in_A, u, v]
 *                                 * X[v, b]
 *
 *   ooL_raw[idx, m, q] = qij_atom[A][q_in_A, occ(idx), m_in_A]
 *
 * for occ(0..2) = (i, j, k); then applies local J^{-1/2} (= jhi) to
 * fold raw → ovL_sc / vvL_sc / ooL_sc as one BLAS-3 dgemm per output.
 *
 * Outer parallel: NONE — driver fans triples across a thread pool.
 *
 * Shapes:
 *   X_tno_ijk:                   (n_pao_ijk, n_tno)
 *   qij/qia/qab atom flat:       per atom block of (nQ_A, nl_A, nl_A)
 *                                                 (nQ_A, nl_A, np_A)
 *                                                 (nQ_A, np_A, np_A)
 *                                stacked end-to-end via qij_off etc.
 *   riatom_to_lmos_ext_dense:    (natm, n_lmo_global) int64, -1 if absent
 *   riatom_to_paos_ext_dense:    (natm, n_pao_global) int64, -1 if absent
 *   ovL_sc:  (3, n_tno, naux_ijk)
 *   vvL_sc:  (n_tno, n_tno, naux_ijk)
 *   ooL_sc:  (3, n_domain, naux_ijk)
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

void DLPNObuild_triple_local_DF(
        const int     i_idx,
        const int     j_idx,
        const int     k_idx,
        const int     n_tno,
        const int     n_domain,
        const int     n_pao_ijk,
        const int     naux_ijk,
        const int     n_centers,
        const int     n_lmo_global,
        const int     n_pao_global,
        const double *X_tno_ijk,           /* (n_pao_ijk, n_tno)   row-major */
        const long   *triple_paos,         /* (n_pao_ijk,)         */
        const long   *triple_domain,       /* (n_domain,)          */
        const long   *center_atoms,        /* (n_centers,)         */
        const long   *center_off,          /* (n_centers+1,) offsets into local_Q_flat / atom_pos_flat */
        const long   *local_Q_flat,        /* (sum nQ_c,) positions in naux_ijk */
        const long   *atom_pos_flat,       /* (sum nQ_c,) page in atom stack */
        const long   *qij_atom_off,        /* (natm+1,) starts in qij_atom_flat (elements) */
        const long   *qia_atom_off,        /* (natm+1,) */
        const long   *qab_atom_off,        /* (natm+1,) */
        const int    *qij_atom_n_aux,      /* (natm,) nQ_A */
        const int    *qij_atom_n_lmo,      /* (natm,) nl_A — same for qij rows/cols and qia rows */
        const int    *qab_atom_n_pao,      /* (natm,) np_A — same for qia cols and qab rows/cols */
        const double *qij_atom_flat,
        const double *qia_atom_flat,
        const double *qab_atom_flat,
        const long   *riatom_to_lmos_ext_dense,  /* (natm, n_lmo_global) */
        const long   *riatom_to_paos_ext_dense,  /* (natm, n_pao_global) */
        const double *jhi,                 /* (naux_ijk, naux_ijk) */
        double       *ovL_sc,              /* (3, n_tno, naux_ijk) */
        double       *vvL_sc,              /* (n_tno, n_tno, naux_ijk) */
        double       *ooL_sc)              /* (3, n_domain, naux_ijk) */
{
    if (naux_ijk <= 0) return;

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;

    /* Raw output tensors (filled to zero, then per-center scattered). */
    const size_t ov_sz = (size_t)3 * (size_t)n_tno * (size_t)naux_ijk;
    const size_t vv_sz = (size_t)n_tno * (size_t)n_tno * (size_t)naux_ijk;
    const size_t oo_sz = (size_t)3 * (size_t)n_domain * (size_t)naux_ijk;
    double *ovL_raw = (double *)calloc(ov_sz > 0 ? ov_sz : 1, sizeof(double));
    double *vvL_raw = (double *)calloc(vv_sz > 0 ? vv_sz : 1, sizeof(double));
    double *ooL_raw = (double *)calloc(oo_sz > 0 ? oo_sz : 1, sizeof(double));

    /* Scratch buffers reused across centers. We allocate to upper bounds:
     *   max nQ_c <= naux_ijk
     *   max nu   <= n_pao_ijk
     */
    const size_t max_nQc = (size_t)naux_ijk;
    const size_t max_nu  = (size_t)n_pao_ijk;
    const size_t max_nl  = (size_t)n_domain + 3;  /* generous: domain ∪ {i,j,k} */

    /* qia_stack_cut_per_occ:  (nQ_c, nu)  per occ → one dgemm against X_Q. */
    double *qia_occ_buf = NULL;
    /* qab_stack_cut: (nQ_c, nu, nu) */
    double *qab_cut_buf = NULL;
    /* tmp = qab_cut @ X_Q : (nQ_c, nu, n_tno) */
    double *qab_tmp_buf = NULL;
    /* X_Q: (nu, n_tno) */
    double *X_Q_buf = NULL;
    /* per-occ ovL block (n_tno, nQ_c) */
    double *ov_block_buf = NULL;
    /* per-q vvL slab (n_tno, n_tno) */
    double *vv_q_buf = NULL;

    if (max_nu > 0 && max_nQc > 0) {
        qia_occ_buf = (double *)malloc(sizeof(double) * max_nQc * max_nu);
        qab_cut_buf = (double *)malloc(sizeof(double) * max_nQc * max_nu * max_nu);
        qab_tmp_buf = (double *)malloc(sizeof(double) * max_nQc * max_nu * (size_t)n_tno);
        X_Q_buf     = (double *)malloc(sizeof(double) * max_nu * (size_t)n_tno);
        ov_block_buf= (double *)malloc(sizeof(double) * (size_t)n_tno * max_nQc);
        vv_q_buf    = (double *)malloc(sizeof(double) * (size_t)n_tno * (size_t)n_tno);
    }

    long *valid_tp_buf  = (long   *)malloc(sizeof(long)   * (max_nu + 1));
    long *valid_uQ_buf  = (long   *)malloc(sizeof(long)   * (max_nu + 1));
    long *dom_local_buf = (long   *)malloc(sizeof(long)   * (max_nl + 1));
    long *m_sparse_buf  = (long   *)malloc(sizeof(long)   * (max_nl + 1));

    for (int c = 0; c < n_centers; c++) {
        const long centerQ = center_atoms[c];
        const long c_beg = center_off[c];
        const long c_end = center_off[c + 1];
        const size_t nQc = (size_t)(c_end - c_beg);
        if (nQc == 0) continue;

        const long *local_Q  = local_Q_flat  + c_beg;
        const long *atom_pos = atom_pos_flat + c_beg;

        const int nl_A = qij_atom_n_lmo[centerQ];
        const int np_A = qab_atom_n_pao[centerQ];

        const long *lmos_dense_row = riatom_to_lmos_ext_dense
                                   + (size_t)centerQ * (size_t)n_lmo_global;
        const long *paos_dense_row = riatom_to_paos_ext_dense
                                   + (size_t)centerQ * (size_t)n_pao_global;

        const int i_s = (int)lmos_dense_row[i_idx];
        const int j_s = (int)lmos_dense_row[j_idx];
        const int k_s = (int)lmos_dense_row[k_idx];
        const int occ_sp[3] = {i_s, j_s, k_s};

        const long qij_off_A = qij_atom_off[centerQ];
        const long qia_off_A = qia_atom_off[centerQ];
        const long qab_off_A = qab_atom_off[centerQ];
        const int has_qij = (qij_off_A < qij_atom_off[centerQ + 1]) && (nl_A > 0);
        const int has_qia = (qia_off_A < qia_atom_off[centerQ + 1]) && (nl_A > 0) && (np_A > 0);
        const int has_qab = (qab_off_A < qab_atom_off[centerQ + 1]) && (np_A > 0);

        const double *qij_A = qij_atom_flat + qij_off_A;
        const double *qia_A = qia_atom_flat + qia_off_A;
        const double *qab_A = qab_atom_flat + qab_off_A;

        const size_t qij_pg = (size_t)nl_A * (size_t)nl_A;   /* per-Q stride */
        const size_t qia_pg = (size_t)nl_A * (size_t)np_A;
        const size_t qab_pg = (size_t)np_A * (size_t)np_A;

        /* ---------------- ooL (qij path, PAO-free) ---------------- */
        if (has_qij && n_domain > 0) {
            int dom_n = 0;
            for (int d = 0; d < n_domain; d++) {
                const long m_sp = lmos_dense_row[triple_domain[d]];
                if (m_sp >= 0) {
                    dom_local_buf[dom_n] = d;
                    m_sparse_buf[dom_n]  = m_sp;
                    dom_n++;
                }
            }
            if (dom_n > 0) {
                for (int idx = 0; idx < 3; idx++) {
                    const int occ = occ_sp[idx];
                    if (occ < 0) continue;
                    /* ooL_raw[idx, dom_local[d], local_Q[q]]
                     *   = qij_A[atom_pos[q], occ, m_sparse_buf[d]] */
                    double *out = ooL_raw + (size_t)idx * (size_t)n_domain * (size_t)naux_ijk;
                    for (size_t q = 0; q < nQc; q++) {
                        const size_t pg = (size_t)atom_pos[q];
                        const long lq = local_Q[q];
                        const double *qij_pg_ptr = qij_A + pg * qij_pg
                                                + (size_t)occ * (size_t)nl_A;
                        for (int d = 0; d < dom_n; d++) {
                            out[(size_t)dom_local_buf[d] * (size_t)naux_ijk + lq]
                                = qij_pg_ptr[m_sparse_buf[d]];
                        }
                    }
                }
            }
        }

        /* ---------------- ovL & vvL (PAO paths) ---------------- */
        if (!has_qia && !has_qab) continue;

        /* Build valid_tp / valid_uQ arrays + X_Q (nu, n_tno). */
        int nu = 0;
        for (int u = 0; u < n_pao_ijk; u++) {
            const long u_in_A = paos_dense_row[triple_paos[u]];
            if (u_in_A >= 0) {
                valid_tp_buf[nu] = u;
                valid_uQ_buf[nu] = u_in_A;
                nu++;
            }
        }
        if (nu == 0) continue;

        /* X_Q[u_local, t] = X_tno_ijk[valid_tp[u_local], t] */
        for (int u = 0; u < nu; u++) {
            const double *src = X_tno_ijk + (size_t)valid_tp_buf[u] * (size_t)n_tno;
            double *dst = X_Q_buf + (size_t)u * (size_t)n_tno;
            memcpy(dst, src, sizeof(double) * (size_t)n_tno);
        }

        /* ---- ovL: per occ_idx, build qia_stack_cut[:, occ, :] (nQ_c, nu),
         *      then dgemm against X_Q (nu, n_tno) → block (nQ_c, n_tno).
         *      Scatter into ovL_raw[idx, t, local_Q[q]]. */
        if (has_qia) {
            for (int idx = 0; idx < 3; idx++) {
                const int occ = occ_sp[idx];
                if (occ < 0) continue;

                /* qia_occ_buf[q, u_local] = qia_A[atom_pos[q], occ, valid_uQ[u_local]] */
                for (size_t q = 0; q < nQc; q++) {
                    const size_t pg = (size_t)atom_pos[q];
                    const double *src = qia_A + pg * qia_pg
                                       + (size_t)occ * (size_t)np_A;
                    double *dst = qia_occ_buf + q * (size_t)nu;
                    for (int u = 0; u < nu; u++) {
                        dst[u] = src[valid_uQ_buf[u]];
                    }
                }

                /* block (n_tno, nQ_c) = X_Q.T (n_tno, nu) @ qia_occ_buf.T (nu, nQ_c)
                 * Row-major formula:
                 *   block[t, q] = sum_u X_Q[u, t] * qia_occ[q, u]
                 * Col-major dgemm: dgemm('T', 'N', n_tno, nQc, nu,
                 *                       1, X_Q, nu, qia_occ, nu, 0, block, n_tno)
                 * gives block_col[t, q] = sum_u X_Q_col[u, t] * qia_occ_col[u, q]
                 *                       = sum_u X_Q[u, t] * qia_occ[q, u]   ✓
                 *
                 * But we want row-major block_row[t, q] = block_col[t, q] (same when
                 * we treat the buffer as (n_tno, nQc) row-major with leading dim nQc).
                 * Trick: use col-major output ld=n_tno; reading row-major as
                 * (nQc, n_tno) i.e. block_buf[t * nQc + q] requires we write
                 * block[t * nQc + q].  Instead, do dgemm with transposed X.
                 *
                 * Simpler: use 'N','T' to compute block (nQc, n_tno) row-major:
                 *   block_row[q, t] = sum_u qia_occ[q, u] * X_Q[u, t]
                 *   Col-major dgemm: dgemm('N', 'N', n_tno, nQc, nu,
                 *                         1, X_Q_col_n_tno_x_nu, n_tno,
                 *                            qia_occ_col_nu_x_nQc, nu,
                 *                         0, block_col, n_tno)
                 *   But col-major reading row-major X_Q (nu, n_tno) is X_Q^T_col (n_tno, nu).
                 *   That's the col-major X (n_tno, nu) — which is what we already have if
                 *   we just declare ld=n_tno. So just call:
                 *     dgemm('N','N', n_tno, nQc, nu, 1, X_Q, n_tno, qia_occ, nu, 0, block, n_tno)
                 *   -> block_col (n_tno, nQc) = X_Q_row_T (n_tno, nu) @ qia_occ_row_T (nu, nQc)
                 *      block[t, q] = sum_u X_Q[u, t] * qia_occ[q, u]   ✓
                 *   And ov_block_buf treated row-major as (nQc, n_tno) gives:
                 *      ov_block[q, t] = block_col[t, q] = sum_u X_Q[u, t] * qia_occ[q, u]
                 *   Same numbers, just flipped meaning. We'll treat it as (nQc, n_tno) row-major.
                 */
                int int_n_tno = n_tno;
                int int_nQc   = (int)nQc;
                int int_nu    = nu;
                dgemm_(&N_flag, &N_flag,
                       &int_n_tno, &int_nQc, &int_nu,
                       &one, X_Q_buf,    &int_n_tno,
                       qia_occ_buf,      &int_nu,
                       &zero, ov_block_buf, &int_n_tno);
                /* Now ov_block_buf, treated as col-major (n_tno, nQc) i.e. flat
                 * index t + q*n_tno, holds  block[t, q] = X_Q.T @ qia_occ.T.
                 * To scatter row-major ovL_raw[idx, t, local_Q[q]] = block[t, q],
                 * step over (t, q): */
                double *out_idx = ovL_raw
                                + (size_t)idx * (size_t)n_tno * (size_t)naux_ijk;
                for (size_t q = 0; q < nQc; q++) {
                    const long lq = local_Q[q];
                    const double *col = ov_block_buf + q * (size_t)n_tno;
                    for (int t = 0; t < n_tno; t++) {
                        out_idx[(size_t)t * (size_t)naux_ijk + lq] = col[t];
                    }
                }
            }
        }

        /* ---- vvL: build qab_stack_cut (nQ_c, nu, nu), then per Q
         *           tmp[u, t] = qab_cut[q, u, :] @ X_Q[:, t]
         *           vvL_q[a, b] = X_Q.T[a, :] @ tmp[:, b]
         *      and scatter to vvL_raw[a, b, local_Q[q]]. */
        if (has_qab) {
            /* qab_cut_buf[q, u, v] = qab_A[atom_pos[q], valid_uQ[u], valid_uQ[v]] */
            for (size_t q = 0; q < nQc; q++) {
                const size_t pg = (size_t)atom_pos[q];
                const double *src = qab_A + pg * qab_pg;
                double *dst_q = qab_cut_buf + q * (size_t)nu * (size_t)nu;
                for (int u = 0; u < nu; u++) {
                    const long uA = valid_uQ_buf[u];
                    const double *src_row = src + (size_t)uA * (size_t)np_A;
                    double *dst_row = dst_q + (size_t)u * (size_t)nu;
                    for (int v = 0; v < nu; v++) {
                        dst_row[v] = src_row[valid_uQ_buf[v]];
                    }
                }
            }

            /* Per-Q vvL: two BLAS-3 calls each, n_tno × nu and nu × n_tno.
             * tmp_q (nu, n_tno) = qab_cut_q (nu, nu) @ X_Q (nu, n_tno)   — row-major
             * vv_q (n_tno, n_tno) = X_Q.T (n_tno, nu) @ tmp_q (nu, n_tno)
             *
             * Row-major @ row-major; both equivalent to col-major dgemm with swapped args. */
            int int_nu    = nu;
            int int_n_tno = n_tno;
            for (size_t q = 0; q < nQc; q++) {
                const double *qab_cut_q = qab_cut_buf + q * (size_t)nu * (size_t)nu;
                double *tmp_q = qab_tmp_buf + q * (size_t)nu * (size_t)n_tno;
                /* tmp_q row-major (nu, n_tno) = qab_cut_q row-major (nu, nu) @ X_Q row-major (nu, n_tno)
                 * Col-major dgemm: dgemm('N','N', n_tno, nu, nu,
                 *                        1, X_Q, n_tno, qab_cut_q, nu, 0, tmp_q, n_tno)
                 * Reading row-major tmp_q as col-major (n_tno, nu): tmp_q_col[t, u] = sum_v X_Q[v, t] * qab[u, v]
                 * Then row-major view tmp_q[u, t] = tmp_q_col[t, u] = sum_v qab[u, v] * X_Q[v, t]   ✓
                 */
                dgemm_(&N_flag, &N_flag,
                       &int_n_tno, &int_nu, &int_nu,
                       &one, X_Q_buf,    &int_n_tno,
                       qab_cut_q,        &int_nu,
                       &zero, tmp_q,     &int_n_tno);
                /* vv_q row-major (n_tno, n_tno) = X_Q.T row-major (n_tno, nu) @ tmp_q row-major (nu, n_tno).
                 *   Equivalently: vv_q[a, b] = sum_u X_Q[u, a] * tmp_q[u, b]
                 * Col-major dgemm: dgemm('N','T', n_tno, n_tno, nu,
                 *                        1, tmp_q, n_tno, X_Q, n_tno, 0, vv_q, n_tno)
                 * → vv_q_col[b, a] = sum_u tmp_q_col[b, u] * X_Q_col[a, u]
                 *                  = sum_u tmp_q[u, b] * X_Q[u, a]   ✓
                 * Row-major vv_q[a, b] = vv_q_col[b, a] — we then scatter using row-major view.
                 */
                dgemm_(&N_flag, &T_flag,
                       &int_n_tno, &int_n_tno, &int_nu,
                       &one, tmp_q,    &int_n_tno,
                       X_Q_buf,        &int_n_tno,
                       &zero, vv_q_buf, &int_n_tno);

                /* Scatter to vvL_raw[a, b, local_Q[q]]:
                 *   vv_q_buf treated as col-major (n_tno, n_tno) holds vv_q_col[b, a].
                 *   So vv_q_buf[b * n_tno + a] = vv_q[a, b].
                 */
                const long lq = local_Q[q];
                for (int b = 0; b < n_tno; b++) {
                    for (int a = 0; a < n_tno; a++) {
                        vvL_raw[((size_t)a * (size_t)n_tno + (size_t)b)
                                * (size_t)naux_ijk + lq]
                            = vv_q_buf[(size_t)b * (size_t)n_tno + (size_t)a];
                    }
                }
            }
        }
    }

    free(valid_tp_buf);
    free(valid_uQ_buf);
    free(dom_local_buf);
    free(m_sparse_buf);
    free(qia_occ_buf);
    free(qab_cut_buf);
    free(qab_tmp_buf);
    free(X_Q_buf);
    free(ov_block_buf);
    free(vv_q_buf);

    /* ---------------- Apply jhi (local J^{-1/2}) ---------------- */
    /* Each output is reshaped (rows, naux_ijk) and right-multiplied by jhi
     * (naux_ijk, naux_ijk). All row-major. */

    /* ovL_sc (3*n_tno, naux_ijk) = ovL_raw (3*n_tno, naux_ijk) @ jhi (naux_ijk, naux_ijk)
     *   row-major out[r, q'] = sum_q ovL_raw[r, q] * jhi[q, q']
     * Col-major dgemm: dgemm('N','N', naux_ijk, 3*n_tno, naux_ijk,
     *                       1, jhi, naux_ijk, ovL_raw, naux_ijk,
     *                       0, ovL_sc, naux_ijk)
     */
    int int_naux = naux_ijk;
    int int_3ntno = 3 * n_tno;
    int int_ntno_sq = n_tno * n_tno;
    int int_3ndom = 3 * n_domain;
    if (int_3ntno > 0) {
        dgemm_(&N_flag, &N_flag,
               &int_naux, &int_3ntno, &int_naux,
               &one, jhi,    &int_naux,
               ovL_raw,      &int_naux,
               &zero, ovL_sc, &int_naux);
    }
    if (int_ntno_sq > 0) {
        dgemm_(&N_flag, &N_flag,
               &int_naux, &int_ntno_sq, &int_naux,
               &one, jhi,    &int_naux,
               vvL_raw,      &int_naux,
               &zero, vvL_sc, &int_naux);
    }
    if (int_3ndom > 0) {
        dgemm_(&N_flag, &N_flag,
               &int_naux, &int_3ndom, &int_naux,
               &one, jhi,    &int_naux,
               ooL_raw,      &int_naux,
               &zero, ooL_sc, &int_naux);
    }

    free(ovL_raw);
    free(vvL_raw);
    free(ooL_raw);
}
