/* DLPNO-CCSD compute_cc_integrals_sparse: per-pair-per-centerQ inner kernel.
 *
 * Pre-CCSD setup port. Replaces the Python body of the inner
 * `for centerQ in unique_centers:` loop in
 * pyscf/cc/dlpno_tccsd/local_df.py:_process_pair (compute_cc_integrals_sparse).
 *
 * Profile (water-10): the outer Python for-pair loop (parallelised via
 * thread pool) is bottlenecked by ~1360s CPU in this inner body (≈80%
 * of cc_ints CPU). The math is small numpy ops + a handful of
 * BLAS-friendly matmuls; the cost is Python interpreter + numpy
 * dispatch overhead at small shape. Moving the body into C with
 * hand-rolled tight loops eliminates that overhead.
 *
 * One C kernel call per (pair, centerQ). The Python wrapper still
 * iterates centerQs and dispatches the cross-pair partner_apply calls
 * (cheap; partner_apply is already in C).
 *
 * Math (per pair, per centerQ; matches local_df.py:736-823 line-by-line):
 *
 *   raw_io[local_Q[q], ext_kept_lmos[k]]
 *       = qij_b[q, i_s, ext_kept_pos[k]]                    (if i_s >= 0)
 *   raw_jo[local_Q[q], ext_kept_lmos[k]]
 *       = qij_b[q, j_s, ext_kept_pos[k]]                    (if j_s >= 0)
 *   raw_pair[local_Q[q]]
 *       = qij_b[q, i_s, j_s]                                (if both >= 0)
 *
 *   raw_iv[local_Q[q], a]
 *       = sum_u qia_b[q, i_s, ij_u_in_Q[u]] * X_ij_slice[u, a]
 *   raw_jv[local_Q[q], a]
 *       = sum_u qia_b[q, j_s, ij_u_in_Q[u]] * X_ij_slice[u, a]
 *
 *   raw_ma[local_Q[q], ext_kept_lmos[k], a]
 *       = sum_u qia_b[q, ext_kept_pos[k], ij_u_in_Q[u]]
 *               * X_ij_slice[u, a]
 *
 *   raw_ab[local_Q[q], a, b]
 *       = sum_{u, v} X_ij_slice[u, a]
 *                    * qab_b[q, ij_u_in_Q[u], ij_u_in_Q[v]]
 *                    * X_ij_slice[v, b]
 *
 *   proj_ij_out[q, a, v]
 *       = sum_u X_ij_slice[u, a] * qab_b[q, ij_u_in_Q[u], v]
 *
 * Shapes (all C-contiguous double except integer index arrays):
 *   qij_b:        (nQp, nl, nl)
 *   qia_b:        (nQp, nl, np_full)
 *   qab_b:        (nQp, np_full, np_full)
 *   local_Q:      (nQp,)        int64
 *   ij_u_in_pair: (npp_ij,)     int64  — used only via X_ij_slice rows
 *   ij_u_in_Q:    (npp_ij,)     int64
 *   ext_kept_pos: (n_kept,)     int64
 *   ext_kept_lmos:(n_kept,)     int64
 *   X_ij_slice:   (npp_ij, npno)
 *   raw_io:       (n_local, nlmo_p)   — accumulator (set, not added)
 *   raw_jo:       (n_local, nlmo_p)
 *   raw_iv:       (n_local, npno)
 *   raw_jv:       (n_local, npno)
 *   raw_pair:     (n_local,)
 *   raw_ma:       (n_local, nlmo_p, npno)
 *   raw_ab:       (n_local, npno, npno)
 *   proj_ij_out:  (nQp, npno, np_full)  — fully overwritten this call
 *
 * Notes:
 *   - All raw_* are pair-persistent buffers, accumulated across centerQ
 *     calls. Each centerQ touches a disjoint subset of local_Q rows
 *     (since local_Q comes from `centers_of_aux == centerQ` mask), so
 *     "set" semantics is race-free across centerQs.
 *   - The kernel is serial (no OMP) — outer parallelism is over pairs
 *     via the Python ThreadPoolExecutor.
 */

#include <stddef.h>

void DLPNOpair_centerQ_step(
        const double *qij_b,
        const double *qia_b,
        const double *qab_b,
        const long   *local_Q,
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
    const size_t ab_a   = npno;
    const size_t X_row  = npno;
    const size_t proj_q = npno * np_full;
    const size_t proj_a = np_full;

    const int has_i = (i_s >= 0);
    const int has_j = (j_s >= 0);
    const int has_pair_paos = (npp_ij > 0);

    for (size_t q = 0; q < nQp; q++) {
        const size_t row_lq = (size_t)local_Q[q];
        const double *qij_q_ptr = qij_b + q * qij_q;

        /* raw_io / raw_jo / raw_pair */
        if (has_i && n_kept > 0) {
            const double *qij_qi = qij_q_ptr + (size_t)i_s * qij_l;
            double *out_row = raw_io + row_lq * io_row;
            for (size_t k = 0; k < n_kept; k++) {
                out_row[ext_kept_lmos[k]] = qij_qi[ext_kept_pos[k]];
            }
        }
        if (has_j && n_kept > 0) {
            const double *qij_qj = qij_q_ptr + (size_t)j_s * qij_l;
            double *out_row = raw_jo + row_lq * io_row;
            for (size_t k = 0; k < n_kept; k++) {
                out_row[ext_kept_lmos[k]] = qij_qj[ext_kept_pos[k]];
            }
        }
        if (has_i && has_j) {
            raw_pair[row_lq] = qij_q_ptr[(size_t)i_s * qij_l + (size_t)j_s];
        }

        if (!has_pair_paos) continue;

        const double *qia_q_ptr = qia_b + q * qia_q;
        const double *qab_q_ptr = qab_b + q * qab_q;

        /* raw_iv[local_Q[q], a] = sum_u qia_b[q, i_s, ij_u_in_Q[u]] * X_ij[u, a] */
        if (has_i) {
            const double *qia_qi = qia_q_ptr + (size_t)i_s * qia_l;
            double *iv_out = raw_iv + row_lq * iv_row;
            for (size_t a = 0; a < npno; a++) {
                double s = 0.0;
                for (size_t u = 0; u < npp_ij; u++) {
                    s += qia_qi[ij_u_in_Q[u]] * X_ij_slice[u * X_row + a];
                }
                iv_out[a] = s;
            }
        }
        if (has_j) {
            const double *qia_qj = qia_q_ptr + (size_t)j_s * qia_l;
            double *jv_out = raw_jv + row_lq * iv_row;
            for (size_t a = 0; a < npno; a++) {
                double s = 0.0;
                for (size_t u = 0; u < npp_ij; u++) {
                    s += qia_qj[ij_u_in_Q[u]] * X_ij_slice[u * X_row + a];
                }
                jv_out[a] = s;
            }
        }

        /* raw_ma[local_Q[q], ext_kept_lmos[k], a]
         *   = sum_u qia_b[q, ext_kept_pos[k], ij_u_in_Q[u]] * X_ij[u, a] */
        if (n_kept > 0) {
            double *ma_out_row = raw_ma + row_lq * ma_row;
            for (size_t k = 0; k < n_kept; k++) {
                const long lmo_pos = ext_kept_pos[k];
                const double *qia_qk = qia_q_ptr + (size_t)lmo_pos * qia_l;
                double *ma_out = ma_out_row + (size_t)ext_kept_lmos[k] * ma_lmo;
                for (size_t a = 0; a < npno; a++) {
                    double s = 0.0;
                    for (size_t u = 0; u < npp_ij; u++) {
                        s += qia_qk[ij_u_in_Q[u]] * X_ij_slice[u * X_row + a];
                    }
                    ma_out[a] = s;
                }
            }
        }

        /* raw_ab[local_Q[q], a, b]
         *   = sum_{u, v} X[u, a] * qab[q, ij_u_in_Q[u], ij_u_in_Q[v]] * X[v, b]
         *
         * Two-step: tmp[u, b] = sum_v qab_q[ij_u_in_Q[u], ij_u_in_Q[v]] * X[v, b]
         *           ab[a, b]  = sum_u X[u, a] * tmp[u, b]
         *
         * Stack scratch tmp of size npp_ij * npno (≤ ~150 × 30 ≈ 4 KB).
         */
        {
            double tmp[npp_ij * npno];     /* C99 VLA on stack */
            for (size_t u = 0; u < npp_ij; u++) {
                const double *qab_qu = qab_q_ptr + (size_t)ij_u_in_Q[u] * qab_u;
                double *tmp_u = tmp + u * npno;
                for (size_t b = 0; b < npno; b++) {
                    double s = 0.0;
                    for (size_t v = 0; v < npp_ij; v++) {
                        s += qab_qu[ij_u_in_Q[v]] * X_ij_slice[v * X_row + b];
                    }
                    tmp_u[b] = s;
                }
            }
            double *ab_out = raw_ab + row_lq * ab_row;
            for (size_t a = 0; a < npno; a++) {
                for (size_t b = 0; b < npno; b++) {
                    double s = 0.0;
                    for (size_t u = 0; u < npp_ij; u++) {
                        s += X_ij_slice[u * X_row + a] * tmp[u * npno + b];
                    }
                    ab_out[a * ab_a + b] = s;
                }
            }
        }

        /* proj_ij_out[q, a, v] = sum_u X[u, a] * qab[q, ij_u_in_Q[u], v]
         * (full v range, not restricted to ij_u_in_Q[v]) */
        {
            double *proj_out = proj_ij_out + q * proj_q;
            for (size_t a = 0; a < npno; a++) {
                double *proj_out_a = proj_out + a * proj_a;
                for (size_t v = 0; v < np_full; v++) {
                    double s = 0.0;
                    for (size_t u = 0; u < npp_ij; u++) {
                        s += X_ij_slice[u * X_row + a]
                             * qab_q_ptr[(size_t)ij_u_in_Q[u] * qab_u + v];
                    }
                    proj_out_a[v] = s;
                }
            }
        }
    }
}
