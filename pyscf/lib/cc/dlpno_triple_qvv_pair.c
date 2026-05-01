/* DLPNO-(T): per-pair q_vv build (Psi4-style restructure).
 *
 * Replaces the full `vvL_sc` (n_tno, n_tno, naux_ijk) tensor with three
 * thinner per-pair slices:
 *
 *   q_vv_ij[a_tno, b_pno_ij, q]   shape (n_tno, n_pno_ij, naux_ijk)
 *   q_vv_jk[a_tno, b_pno_jk, q]   shape (n_tno, n_pno_jk, naux_ijk)
 *   q_vv_ik[a_tno, b_pno_ik, q]   shape (n_tno, n_pno_ik, naux_ijk)
 *
 * Math (mirrors Psi4 triples.cc:723-754):
 *
 *   q_vv_pair[a, b, q] = Σ_{u, v} X_tno[u_pao_ijk, a]
 *                                * qab_PAO[q, u_pao_ijk, v_pao_pair]
 *                                * X_pno_pair[v_pao_pair, b]
 *
 * Per Q (per pair):
 *   1. tmp[u_pao_ijk, b] = qab_PAO_uv @ X_pno_pair      cost n_pao_ijk × n_pao_pair × n_pno_pair
 *   2. q_vv[a, b]        = X_tno.T @ tmp                cost n_tno × n_pao_ijk × n_pno_pair
 *
 * Total per Q per pair: O(n_pao_ijk × n_pao_pair × n_pno_pair) — LINEAR in n_pao_ijk
 * Versus the old vvL build: O(n_pao_ijk × n_pao_ijk × n_tno) per Q (squared in n_pao_ijk).
 *
 * This is the source of (T) scaling reduction from N^2.56 → ~N^2.2.
 *
 * SCAFFOLD STATUS: This is the FIRST atomic unit of the restructure.
 * Builds ONE pair's q_vv from per-triple sparse-DF data + X_pno_pair.
 * Validation: caller can compare q_vv_pair against `X_pno_pair.T @ vvL_sc[a,:,q] @ X_tno_to_pno_pair_inv`
 * for bit-equivalence (within FP noise).
 *
 * Inputs:
 *   n_tno                    — triple's TNO count
 *   n_pao_ijk                — triple's PAO domain size
 *   n_pao_pair               — pair's PAO domain size
 *   n_pno_pair               — pair's PNO count
 *   naux_ijk                 — triple's local aux dim
 *   n_centers                — # of aux atoms relevant to triple
 *   triple_paos              — (n_pao_ijk,) global PAO indices
 *   pair_paos                — (n_pao_pair,) global PAO indices for the pair
 *   X_tno_ijk                — (n_pao_ijk, n_tno) row-major
 *   X_pno_pair               — (n_pao_pair, n_pno_pair) row-major
 *   center_atoms             — (n_centers,) aux atoms
 *   center_off                — (n_centers+1,) offsets in local_Q/atom_pos
 *   local_Q_flat             — positions in naux_ijk
 *   atom_pos_flat            — page in atom stack
 *   qab_atom_off             — (natm+1,) offsets in qab_atom_flat
 *   qab_atom_n_pao           — (natm,) np_A per atom
 *   qab_atom_flat            — qab data
 *   riatom_to_paos_ext_dense — (natm, n_pao_global) — -1 if PAO absent
 *   n_pao_global             — full PAO count (for stride)
 *   jhi                      — (naux_ijk, naux_ijk) local J^{-1/2}
 *
 * Output:
 *   q_vv_pair_sc             — (n_tno, n_pno_pair, naux_ijk) row-major
 *
 * NOT YET IMPLEMENTED — scaffold only. See HANDOFF_TRIPLES_VVL_PER_PAIR.md.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

void DLPNObuild_triple_qvv_pair(
        const int     n_tno,
        const int     n_pao_ijk,
        const int     n_pao_pair,
        const int     n_pno_pair,
        const int     naux_ijk,
        const int     n_centers,
        const int     n_pao_global,
        const long   *triple_paos,
        const long   *pair_paos,
        const double *X_tno_ijk,
        const double *X_pno_pair,
        const long   *center_atoms,
        const long   *center_off,
        const long   *local_Q_flat,
        const long   *atom_pos_flat,
        const long   *qab_atom_off,
        const int    *qab_atom_n_pao,
        const double *qab_atom_flat,
        const long   *riatom_to_paos_ext_dense,
        const double *jhi,
        double       *q_vv_pair_sc)
{
    if (naux_ijk <= 0 || n_tno <= 0 || n_pno_pair <= 0) return;

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;

    /* Raw (pre-jhi) output, scattered per Q. */
    const size_t raw_sz = (size_t)n_tno * (size_t)n_pno_pair * (size_t)naux_ijk;
    double *q_vv_raw = (double *)calloc(raw_sz > 0 ? raw_sz : 1, sizeof(double));

    /* Scratch — sized to upper bounds. */
    const size_t max_nl = (size_t)n_pao_ijk;       /* triple-PAO at any center */
    const size_t max_nr = (size_t)n_pao_pair;      /* pair-PAO at any center */
    /* qab_cut: nl × nr per Q */
    double *qab_cut_buf = (double *)malloc(
        sizeof(double) * (max_nl * max_nr > 0 ? max_nl * max_nr : 1));
    /* tmp = qab_cut @ X_pno_pair_local: nl × n_pno_pair */
    double *tmp_buf = (double *)malloc(
        sizeof(double) * (max_nl * (size_t)n_pno_pair > 0
                          ? max_nl * (size_t)n_pno_pair : 1));
    /* q_vv_block: n_tno × n_pno_pair */
    double *qvv_blk = (double *)malloc(
        sizeof(double) * (size_t)n_tno * (size_t)n_pno_pair);
    /* X_tno_local: nl × n_tno (gather) */
    double *X_tno_local = (double *)malloc(
        sizeof(double) * (max_nl * (size_t)n_tno > 0
                          ? max_nl * (size_t)n_tno : 1));
    /* X_pno_local: nr × n_pno_pair (gather) */
    double *X_pno_local = (double *)malloc(
        sizeof(double) * (max_nr * (size_t)n_pno_pair > 0
                          ? max_nr * (size_t)n_pno_pair : 1));

    long *valid_l_tp = (long *)malloc(sizeof(long) * (max_nl + 1));
    long *valid_r_pp = (long *)malloc(sizeof(long) * (max_nr + 1));

    for (int c = 0; c < n_centers; c++) {
        const long centerQ = center_atoms[c];
        const long c_beg = center_off[c];
        const long c_end = center_off[c + 1];
        const size_t nQc = (size_t)(c_end - c_beg);
        if (nQc == 0) continue;

        const long *local_Q  = local_Q_flat  + c_beg;
        const long *atom_pos = atom_pos_flat + c_beg;

        const int np_A = qab_atom_n_pao[centerQ];
        if (np_A == 0) continue;

        const long *paos_dense_row = riatom_to_paos_ext_dense
                                   + (size_t)centerQ * (size_t)n_pao_global;

        /* Visible triple-PAOs at this center (left axis). Each element is
         * an index into the local atom's PAO stack (np_A range). */
        int nl = 0;
        for (int u = 0; u < n_pao_ijk; u++) {
            const long u_in_A = paos_dense_row[triple_paos[u]];
            if (u_in_A >= 0) {
                /* X_tno_local[nl, t] = X_tno_ijk[u, t]   (gather rows of X_tno) */
                memcpy(X_tno_local + (size_t)nl * (size_t)n_tno,
                       X_tno_ijk + (size_t)u * (size_t)n_tno,
                       sizeof(double) * (size_t)n_tno);
                valid_l_tp[nl] = u_in_A;
                nl++;
            }
        }
        if (nl == 0) continue;

        /* Visible pair-PAOs at this center (right axis). */
        int nr = 0;
        for (int v = 0; v < n_pao_pair; v++) {
            const long v_in_A = paos_dense_row[pair_paos[v]];
            if (v_in_A >= 0) {
                memcpy(X_pno_local + (size_t)nr * (size_t)n_pno_pair,
                       X_pno_pair + (size_t)v * (size_t)n_pno_pair,
                       sizeof(double) * (size_t)n_pno_pair);
                valid_r_pp[nr] = v_in_A;
                nr++;
            }
        }
        if (nr == 0) continue;

        const long qab_off_A = qab_atom_off[centerQ];
        const double *qab_A = qab_atom_flat + qab_off_A;
        const size_t qab_pg = (size_t)np_A * (size_t)np_A;   /* per-Q stride */

        /* Per-Q work: gather qab_cut, two dgemms, scatter. */
        int int_nl = nl;
        int int_nr = nr;
        int int_n_tno = n_tno;
        int int_n_pno_pair = n_pno_pair;

        for (size_t q = 0; q < nQc; q++) {
            const size_t pg = (size_t)atom_pos[q];
            const double *src = qab_A + pg * qab_pg;

            /* qab_cut[u_local, v_local] = qab_A[atom_pos[q], valid_l[u], valid_r[v]] */
            for (int u = 0; u < nl; u++) {
                const long uA = valid_l_tp[u];
                const double *src_row = src + (size_t)uA * (size_t)np_A;
                double *dst_row = qab_cut_buf + (size_t)u * (size_t)nr;
                for (int v = 0; v < nr; v++) {
                    dst_row[v] = src_row[valid_r_pp[v]];
                }
            }

            /* tmp[nl, n_pno_pair] = qab_cut[nl, nr] @ X_pno_local[nr, n_pno_pair]
             *   row-major C[m, n] = A[m, k] @ B[k, n]
             *   col-major dgemm: dgemm('N','N', n, m, k, B, n, A, k, 0, C, n)
             */
            dgemm_(&N_flag, &N_flag,
                   &int_n_pno_pair, &int_nl, &int_nr,
                   &one, X_pno_local, &int_n_pno_pair,
                   qab_cut_buf, &int_nr,
                   &zero, tmp_buf, &int_n_pno_pair);

            /* qvv_blk[n_tno, n_pno_pair] = X_tno_local[nl, n_tno].T @ tmp[nl, n_pno_pair]
             *   row-major C[m, n] = A[k, m]^T @ B[k, n]
             *   dgemm('N', 'T', n, m, k, B, n, A, m, 0, C, n)
             */
            dgemm_(&N_flag, &T_flag,
                   &int_n_pno_pair, &int_n_tno, &int_nl,
                   &one, tmp_buf, &int_n_pno_pair,
                   X_tno_local, &int_n_tno,
                   &zero, qvv_blk, &int_n_pno_pair);

            /* Scatter into q_vv_raw[a, b, local_Q[q]]:
             *   row-major (n_tno, n_pno_pair, naux_ijk).
             *   q_vv_raw[a*n_pno_pair*naux + b*naux + lq] = qvv_blk[a*n_pno_pair + b]
             */
            const long lq = local_Q[q];
            for (int a = 0; a < n_tno; a++) {
                for (int b = 0; b < n_pno_pair; b++) {
                    q_vv_raw[((size_t)a * (size_t)n_pno_pair
                              + (size_t)b) * (size_t)naux_ijk + lq]
                        = qvv_blk[(size_t)a * (size_t)n_pno_pair + (size_t)b];
                }
            }
        }
    }

    free(qab_cut_buf);
    free(tmp_buf);
    free(qvv_blk);
    free(X_tno_local);
    free(X_pno_local);
    free(valid_l_tp);
    free(valid_r_pp);

    /* Apply jhi: q_vv_pair_sc (n_tno*n_pno_pair, naux_ijk) =
     *   q_vv_raw (n_tno*n_pno_pair, naux_ijk) @ jhi (naux_ijk, naux_ijk)
     * One dgemm.
     */
    int int_naux = naux_ijk;
    int int_rows = n_tno * n_pno_pair;
    if (int_rows > 0) {
        dgemm_(&N_flag, &N_flag,
               &int_naux, &int_rows, &int_naux,
               &one, jhi, &int_naux,
               q_vv_raw, &int_naux,
               &zero, q_vv_pair_sc, &int_naux);
    }

    free(q_vv_raw);
}
