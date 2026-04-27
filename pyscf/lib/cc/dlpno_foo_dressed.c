/* DLPNO-CCSD foo (occupied-occupied Fock) per-pair T2 dressing.
 *
 * Phase II port of pyscf/cc/dlpno_tccsd/_foo_dressed_cy.pyx into PySCF
 * native style. Replaces the prior nocc-axis kernel with a pair-domain
 * (nlmo_pair) signature so the caller no longer needs to scatter-back
 * Qma into a (n_local, nocc, n_pno) buffer.
 *
 * Math (per pair (m, q), Eq. for foo from Psi4 ccsd.cc / DePrince Eq 16):
 *
 *   X_mq[a, L]      = sum_b (2*t2[b, a] - t2[a, b]) * Qma_pair[L, m_pos, b]
 *   out_q_pair[p]   = sum_{L, a} Qma_pair[L, p, a] * X_mq[a, L]    (p in nlmo_pair)
 *
 *   if m != q:
 *     X_qm[a, L]    = sum_b (2*t2[a, b] - t2[b, a]) * Qma_pair[L, q_pos, b]
 *     out_m_pair[p] = sum_{L, a} Qma_pair[L, p, a] * X_qm[a, L]
 *
 * The caller scatters out_q_pair / out_m_pair into the global foo:
 *   foo[ci['p_lmos'], q] += out_q_pair
 *   foo[ci['p_lmos'], m] += out_m_pair
 *
 * Entry shapes:
 *   Qma_pair: (n_local, nlmo_pair, n_pno) C-contiguous
 *   t2:       (n_pno,  n_pno)              C-contiguous
 *   out_q:    (nlmo_pair,)                 C-contiguous
 *   out_m:    (nlmo_pair,)                 C-contiguous (may alias out_q if m == q)
 */

#include <stdlib.h>
#include <string.h>

void DLPNOfoo_dressed_pair(double *out_q,
                           double *out_m,
                           const double *Qma,
                           const double *t2,
                           const int m_pos,
                           const int q_pos,
                           const int need_m,
                           const size_t n_local,
                           const size_t nlmo_pair,
                           const size_t n_pno)
{
    const size_t pair_stride = nlmo_pair * n_pno;     /* Qma[L, ...] stride */
    const size_t lmo_stride  = n_pno;                 /* Qma[L, l, ...] stride */

    /* Scratch X[a, L] of shape (n_pno, n_local) — built once for q,
     * once for m (if needed). Allocated/freed per call; small relative
     * to the work. */
    double *X_mq = (double *)malloc(sizeof(double) * n_pno * n_local);
    double *X_qm = need_m ? (double *)malloc(sizeof(double) * n_pno * n_local)
                          : NULL;

    /* X_mq[a, L] = sum_b (2*t2[b, a] - t2[a, b]) * Qma[L, m_pos, b] */
#pragma omp parallel for schedule(static)
    for (size_t a = 0; a < n_pno; a++) {
        for (size_t L = 0; L < n_local; L++) {
            const double *Qma_Lm = Qma + L * pair_stride + m_pos * lmo_stride;
            double s = 0.0;
            for (size_t b = 0; b < n_pno; b++) {
                /* T_eff[b, a] = 2*t2[b, a] - t2[a, b] */
                const double t_ba = t2[b * n_pno + a];
                const double t_ab = t2[a * n_pno + b];
                s += (2.0 * t_ba - t_ab) * Qma_Lm[b];
            }
            X_mq[a * n_local + L] = s;
        }
    }

    /* out_q[p] = sum_{L, a} Qma[L, p, a] * X_mq[a, L]
     * Memory: Qma[L, p, a] indexed by p (outer), L (mid), a (inner).
     * X_mq[a, L] swap. Loop order: p outer, L middle, a inner. */
#pragma omp parallel for schedule(static)
    for (size_t p = 0; p < nlmo_pair; p++) {
        double s = 0.0;
        for (size_t L = 0; L < n_local; L++) {
            const double *Qma_Lp = Qma + L * pair_stride + p * lmo_stride;
            for (size_t a = 0; a < n_pno; a++) {
                s += Qma_Lp[a] * X_mq[a * n_local + L];
            }
        }
        out_q[p] = s;
    }

    if (need_m) {
        /* X_qm[a, L] = sum_b (2*t2[a, b] - t2[b, a]) * Qma[L, q_pos, b] */
#pragma omp parallel for schedule(static)
        for (size_t a = 0; a < n_pno; a++) {
            for (size_t L = 0; L < n_local; L++) {
                const double *Qma_Lq = Qma + L * pair_stride + q_pos * lmo_stride;
                double s = 0.0;
                for (size_t b = 0; b < n_pno; b++) {
                    const double t_ab = t2[a * n_pno + b];
                    const double t_ba = t2[b * n_pno + a];
                    s += (2.0 * t_ab - t_ba) * Qma_Lq[b];
                }
                X_qm[a * n_local + L] = s;
            }
        }

        /* out_m[p] = sum_{L, a} Qma[L, p, a] * X_qm[a, L] */
#pragma omp parallel for schedule(static)
        for (size_t p = 0; p < nlmo_pair; p++) {
            double s = 0.0;
            for (size_t L = 0; L < n_local; L++) {
                const double *Qma_Lp = Qma + L * pair_stride + p * lmo_stride;
                for (size_t a = 0; a < n_pno; a++) {
                    s += Qma_Lp[a] * X_qm[a * n_local + L];
                }
            }
            out_m[p] = s;
        }
    }

    free(X_mq);
    if (X_qm) free(X_qm);
}
