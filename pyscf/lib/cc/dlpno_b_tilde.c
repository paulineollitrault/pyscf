/* DLPNO-CCSD compute_B_tilde: per-pair B_tilde construction.
 *
 * Phase II/full-C cycle port of pyscf/cc/dlpno_tccsd/local_df.py:compute_B_tilde
 * into PySCF native style. Mirrors Psi4 ccsd.cc:1742 (compute_B_tilde).
 *
 * Math (per ordered strong pair ij; output is the per-pair B[k_ij, l_ij]):
 *
 *   B[k, l]  = sum_Q i_Qk_t1[Q, k] * j_Qk_t1[Q, l]
 *            + sum_{Q, a, b} Qma[Q, k, a] * T2[a, b] * Qma[Q, l, b]
 *
 * The Qk_t1 entries are the T1-dressed DF intermediates produced by
 * t1_ints (matches Psi4 i_Qk_t1_[ij] / i_Qk_t1_[ji]). All inputs are on
 * the pair-LMO domain (pair_lmo_idx, length nlmo).
 *
 * Entry shapes (all C-contiguous double):
 *   B_out:      (nlmo, nlmo)      — output, fully overwritten
 *   i_Qk_t1:    (n_local, nlmo)
 *   j_Qk_t1:    (n_local, nlmo)
 *   Qma:        (n_local, nlmo, npno)
 *   T2:         (npno, npno)
 */

#include <stdlib.h>
#include <string.h>

void DLPNOcompute_B_tilde_pair(double *B_out,
                               const double *i_Qk_t1,
                               const double *j_Qk_t1,
                               const double *Qma,
                               const double *T2,
                               const size_t n_local,
                               const size_t nlmo,
                               const size_t npno)
{
    const size_t nl = nlmo;
    const size_t np = npno;
    const size_t qma_Q = nl * np;     /* Qma[Q, ...] stride */
    const size_t qma_k = np;          /* Qma[Q, k, ...] stride */

    /* ============ Term 1 ============
     * B[k, l] = sum_Q i_Qk_t1[Q, k] * j_Qk_t1[Q, l]
     */
#pragma omp parallel for schedule(static)
    for (size_t k = 0; k < nl; k++) {
        for (size_t l = 0; l < nl; l++) {
            double s = 0.0;
            for (size_t Q = 0; Q < n_local; Q++) {
                s += i_Qk_t1[Q * nl + k] * j_Qk_t1[Q * nl + l];
            }
            B_out[k * nl + l] = s;
        }
    }

    /* ============ Term 2 ============
     * B[k, l] += sum_{Q, a, b} Qma[Q, k, a] * T2[a, b] * Qma[Q, l, b]
     *
     * Per Q (matches Psi4 ccsd.cc:1769 triplet):
     *   P[k, b]  = sum_a Qma[Q, k, a] * T2[a, b]
     *   B[k, l] += sum_b P[k, b] * Qma[Q, l, b]
     *
     * Each thread keeps its own scratch P (nl * np). The per-pair
     * accumulation into B[k, l] cannot trivially be parallelised over Q
     * (write conflict), so we parallelise over the inner (k, l) plane
     * inside each Q. This matches the size profile of nlmo (~10-30) and
     * keeps the hot Q loop serial. The arithmetic intensity per Q is
     * dominated by P-build (nl * np^2) and B-update (nl^2 * np).
     */
    double *P = (double *)malloc(sizeof(double) * nl * np);

    for (size_t Q = 0; Q < n_local; Q++) {
        const double *Qma_Q = Qma + Q * qma_Q;

        /* P[k, b] = sum_a Qma_Q[k, a] * T2[a, b] */
#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < nl; k++) {
            const double *Qma_Qk = Qma_Q + k * qma_k;
            double *Pk = P + k * np;
            for (size_t b = 0; b < np; b++) {
                double s = 0.0;
                for (size_t a = 0; a < np; a++) {
                    s += Qma_Qk[a] * T2[a * np + b];
                }
                Pk[b] = s;
            }
        }

        /* B[k, l] += sum_b P[k, b] * Qma_Q[l, b] */
#pragma omp parallel for schedule(static)
        for (size_t k = 0; k < nl; k++) {
            const double *Pk = P + k * np;
            double *Bk = B_out + k * nl;
            for (size_t l = 0; l < nl; l++) {
                const double *Qma_Ql = Qma_Q + l * qma_k;
                double s = 0.0;
                for (size_t b = 0; b < np; b++) {
                    s += Pk[b] * Qma_Ql[b];
                }
                Bk[l] += s;
            }
        }
    }

    free(P);
}
