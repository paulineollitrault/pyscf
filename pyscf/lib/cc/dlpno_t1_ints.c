/* DLPNO-CCSD t1_ints: per-pair, per-side T1 dressing of DF intermediates.
 *
 * Full-C cycle session 3 port. Mirrors Psi4 ccsd.cc:1491 (t1_ints) and
 * the Python local_df.py:t1_ints `_dress_one_pair._dress` inner.
 *
 * Math (per ordered side `s` of strong pair (i, j); in our Python
 * `_dress(lmo_global, Qa_key)` is called once for s=i with Qa_key='i_Qa'
 * and once for s=j with Qa_key='j_Qa'):
 *
 *   qma_t1[Q, m]    = sum_b Qma[Q, m, b] * t1_lmo[b]
 *   Qk_t1[Q, m]     = Qk_local[Q, m] + qma_t1[Q, m]
 *   Qa_t1[Q, a]     = Qa_full[Q, a]
 *                   - sum_m Qk_t1[Q, m] * T1_local[m, a]
 *                   + sum_b Qab[Q, a, b] * t1_lmo[b]
 *
 * The first and third subtractions in Psi4's expansion fold into a
 * single -Qk_t1 @ T1_local matmul:
 *
 *   - Qk_local @ T1   - qma_t1 @ T1   ==   -(Qk_local + qma_t1) @ T1
 *                                       ==   -Qk_t1 @ T1
 *
 * which is what we emit here. Verified against the two-step Python form
 * by the energy anchor + T1INTS_DUMP cross-check.
 *
 * Outputs are scattered in pair-LMO domain (nlmo == |lmopair_to_lmos_[ij]|).
 *
 * Entry shapes (all C-contiguous double):
 *   Qa_t1_out:  (n_local, npno)             — caller-allocated, fully overwritten
 *   Qk_t1_out:  (n_local, nlmo)             — caller-allocated, fully overwritten
 *   Qa_full:    (n_local, npno)
 *   Qk_local:   (n_local, nlmo)
 *   Qma:        (n_local, nlmo, npno)
 *   Qab:        (n_local, npno, npno)
 *   t1_lmo:     (npno,)
 *   T1_local:   (nlmo, npno)
 */

#include <stddef.h>

void DLPNOt1_ints_pair_side(double *Qa_t1_out,
                            double *Qk_t1_out,
                            const double *Qa_full,
                            const double *Qk_local,
                            const double *Qma,
                            const double *Qab,
                            const double *t1_lmo,
                            const double *T1_local,
                            const size_t n_local,
                            const size_t nlmo,
                            const size_t npno)
{
    const size_t qma_Q = nlmo * npno;     /* Qma[Q, ...] stride */
    const size_t qma_m = npno;            /* Qma[Q, m, ...] stride */
    const size_t qab_Q = npno * npno;     /* Qab[Q, ...] stride */
    const size_t qab_a = npno;            /* Qab[Q, a, ...] stride */
    const size_t qa_Q  = npno;            /* Qa_full[Q, ...] / out stride */
    const size_t qk_Q  = nlmo;            /* Qk_local[Q, ...] / out stride */
    const size_t T1_m  = npno;            /* T1_local[m, ...] stride */

    /* Outer parallel over Q. Each Q's writes (Qa_t1_out[Q, *],
     * Qk_t1_out[Q, *]) are disjoint, so no critical / reduction needed.
     * Per-Q working memory: result_qk_row[nlmo] held in a stack VLA.
     */
#pragma omp parallel for schedule(static)
    for (size_t Q = 0; Q < n_local; Q++) {
        const double *Qma_Q  = Qma + Q * qma_Q;
        const double *Qab_Q  = Qab + Q * qab_Q;
        const double *Qa_Q   = Qa_full + Q * qa_Q;
        const double *Qk_Q   = Qk_local + Q * qk_Q;
        double *Qa_out_Q     = Qa_t1_out + Q * qa_Q;
        double *Qk_out_Q     = Qk_t1_out + Q * qk_Q;

        /* Step 1: qma_t1[Q, m] = sum_b Qma[Q, m, b] * t1_lmo[b]
         * and immediately fold into Qk_t1_out[Q, m] = Qk_local[Q, m] + qma_t1[Q, m]
         */
        for (size_t m = 0; m < nlmo; m++) {
            const double *Qma_Qm = Qma_Q + m * qma_m;
            double s = 0.0;
            for (size_t b = 0; b < npno; b++) {
                s += Qma_Qm[b] * t1_lmo[b];
            }
            Qk_out_Q[m] = Qk_Q[m] + s;
        }

        /* Step 2: Qa_t1_out[Q, a] = Qa_full[Q, a] - sum_m Qk_t1_out[Q, m] * T1_local[m, a]
         *                         + sum_b Qab[Q, a, b] * t1_lmo[b]
         */
        for (size_t a = 0; a < npno; a++) {
            /* term1 = sum_m Qk_t1_out[Q, m] * T1_local[m, a]
             * term2 = sum_b Qab[Q, a, b] * t1_lmo[b]
             */
            double term1 = 0.0;
            for (size_t m = 0; m < nlmo; m++) {
                term1 += Qk_out_Q[m] * T1_local[m * T1_m + a];
            }
            const double *Qab_Qa = Qab_Q + a * qab_a;
            double term2 = 0.0;
            for (size_t b = 0; b < npno; b++) {
                term2 += Qab_Qa[b] * t1_lmo[b];
            }
            Qa_out_Q[a] = Qa_Q[a] - term1 + term2;
        }
    }
}
