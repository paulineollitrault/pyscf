/* DLPNO-CCSD compute_cc_integrals_sparse: per-partner contribution.
 *
 * Phase II port of _cc_ints_partner_cy.pyx into PySCF native style.
 * Math:
 *   raw_cross[local_Q[Q], b, m] = sum_u proj_ij[Q, b, idx[u]] * X[u, m]
 *   raw_kv   [local_Q[Q], m]    = sum_u qia_b[Q, k_s, idx[u]] * X[u, m]
 *     (only when k_s >= 0)
 *
 * Shapes (all C-contiguous):
 *   proj_ij        (nQp, npno, np_full)
 *   qia_b          (nQp, nl,   np_full)
 *   local_Q        (nQp,)         long[::1]
 *   idx            (npp,)         long[::1]
 *   X              (npp, n_kj)
 *   raw_cross_out  (n_local_total, npno, n_kj)
 *   raw_kv_out     (n_local_total, n_kj)
 */
#include <stdlib.h>

void DLPNOpartner_apply(const double *proj_ij,
                        const double *qia_b,
                        const long *local_Q,
                        const long *idx,
                        const double *X,
                        double *raw_cross_out,
                        double *raw_kv_out,
                        const int k_s,
                        const int do_proj,
                        const size_t nQp,
                        const size_t npno,
                        const size_t np_full,
                        const size_t nl,
                        const size_t n_kj,
                        const size_t npp,
                        const size_t n_local_total)
{
    const size_t proj_Q   = npno * np_full;
    const size_t proj_b   = np_full;
    const size_t qia_Q    = nl * np_full;
    const size_t qia_l    = np_full;
    const size_t cross_L  = npno * n_kj;
    const size_t cross_b  = n_kj;

    if (do_proj) {
#pragma omp parallel for schedule(static)
        for (size_t Q = 0; Q < nQp; Q++) {
            const size_t row = (size_t)local_Q[Q];
            for (size_t b = 0; b < npno; b++) {
                for (size_t m = 0; m < n_kj; m++) {
                    double s = 0.0;
                    for (size_t u = 0; u < npp; u++) {
                        const size_t col = (size_t)idx[u];
                        s += proj_ij[Q * proj_Q + b * proj_b + col]
                             * X[u * n_kj + m];
                    }
                    raw_cross_out[row * cross_L + b * cross_b + m] = s;
                }
            }
        }
    }

    if (k_s >= 0) {
#pragma omp parallel for schedule(static)
        for (size_t Q = 0; Q < nQp; Q++) {
            const size_t row = (size_t)local_Q[Q];
            for (size_t m = 0; m < n_kj; m++) {
                double s = 0.0;
                for (size_t u = 0; u < npp; u++) {
                    const size_t col = (size_t)idx[u];
                    s += qia_b[Q * qia_Q + (size_t)k_s * qia_l + col]
                         * X[u * n_kj + m];
                }
                raw_kv_out[row * n_kj + m] = s;
            }
        }
    }
}
