/* DLPNO-CCSD compute_cc_integrals_sparse: per-pair cross-partner J/K assembly.
 *
 * Pre-CCSD setup port. Replaces the Python per-partner final-assembly
 * loop in local_df.py:_process_pair (lines 1026-1049). Per pair, ~30
 * cross-pair partners; each does 4 small BLAS calls; ~120K Python
 * dispatches per CCSD run on water-10 (≈6s wall).
 *
 * Per partner k (single-side; called once per side: kj and ki):
 *   alpha[q] = sum_p jhi[q, p] * q_io[p, k_loc]   (= jhi @ q_ik, jhi sym)
 *   J_kj[a, m] = sum_p alpha[p] * raw_cross[k][p, a, m]
 *              = (raw_cross[k] viewed as (n_local, npno*n_kj))^T @ alpha
 *   K_kj[a, m] = sum_p Z_iv[a, p] * raw_kv[k][p, m]   (Z_iv = q_iv^T @ jhi)
 *
 * The kernel processes all partners on one side in one C call. Outer
 * parallelism stays over pairs via the Python ThreadPoolExecutor; this
 * kernel is serial. Per-partner work uses BLAS (dgemv/dgemm) via vhf/fblas.h.
 *
 * Hoisted optimisations vs the line-by-line Python:
 *   - Precompute Z_iv = q_iv^T @ jhi  (npno × n_local) once (caller's job).
 *     Then K_kj[k] = Z_iv @ raw_kv[k]   (one dgemm per partner, instead of
 *     two: jhi @ raw_kv  +  q_iv.T @ result).
 *   - Hoist alpha = jhi @ q_ik out of the (npno*n_kj) axis: J_kj is
 *     a single DGEMV on raw_cross_p (n_local, npno*n_kj) instead of a
 *     full DGEMM on jhi @ raw_cross_p followed by a DGEMV — saves the
 *     intermediate (n_local, npno*n_kj) materialisation and reduces
 *     FLOPs by ~npno*n_kj×.
 *
 * Flat-buffer layout (built by Python wrapper per pair, used once
 * per cc_ints build per pair):
 *   raw_cross_flat:  concat of raw_cross[k] arrays, each (n_local, npno, n_kj_k)
 *   raw_cross_off:   (n_partners+1,) offsets into raw_cross_flat
 *   raw_kv_flat:     concat of raw_kv[k], each (n_local, n_kj_k)
 *   raw_kv_off:      (n_partners+1,)
 *   J_out_flat:      output, concat of J_kj[k], each (npno, n_kj_k)
 *   J_out_off:       (n_partners+1,)
 *   K_out_flat:      output, concat of K_kj[k], each (npno, n_kj_k)
 *   K_out_off:       (n_partners+1,)
 *   partner_k_loc:   (n_partners,) — column in q_io for q_ik = q_io[:, k_loc]
 *   partner_n_kj:    (n_partners,)
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

void DLPNOcross_partner_assemble(
        const int      n_partners,
        const long    *partner_k_loc,
        const long    *partner_n_kj,
        const long    *raw_cross_off,
        const double  *raw_cross_flat,
        const long    *raw_kv_off,
        const double  *raw_kv_flat,
        const double  *jhi,           /* (n_local, n_local), symmetric */
        const double  *q_io,          /* (n_local, nlmo_p) */
        const double  *Z_iv,          /* (npno, n_local) — q_iv^T @ jhi, precomputed */
        const long    *J_out_off,
        double        *J_out_flat,
        const long    *K_out_off,
        double        *K_out_flat,
        const size_t   n_local,
        const size_t   nlmo_p,
        const size_t   npno)
{
    if (n_partners <= 0) return;

    static const char N_flag = 'N';
    static const double one = 1.0, zero = 0.0;
    static const int int_one = 1;
    const int int_n_local = (int)n_local;
    const int int_npno = (int)npno;
    const int incx_qio = (int)nlmo_p;

    /* Per-partner alpha vector (n_local). Single allocation reused; jhi is the
     * same across partners but q_ik = q_io[:, k_loc] varies per partner. */
    double *alpha = (double *)malloc(sizeof(double) * n_local);

    for (int p = 0; p < n_partners; p++) {
        const long n_kj = partner_n_kj[p];
        const long k_loc = partner_k_loc[p];
        if (n_kj <= 0) continue;
        const int int_n_kj = (int)n_kj;

        const double *raw_cross_p = raw_cross_flat + raw_cross_off[p];
        const double *raw_kv_p    = raw_kv_flat    + raw_kv_off[p];
        double       *J_out_p     = J_out_flat     + J_out_off[p];
        double       *K_out_p     = K_out_flat     + K_out_off[p];

        /* alpha = jhi @ q_ik (n_local).  q_ik is q_io[:, k_loc] — column of
         * row-major q_io (n_local, nlmo_p), accessed via vector with stride
         * nlmo_p starting at q_io[k_loc]. jhi is symmetric so jhi @ q_ik
         * == jhi^T @ q_ik; we use the row-major-friendly form
         * dgemv('N', n_local, n_local) treating jhi as col-major
         * (n_local, n_local) — same numerical result by symmetry. */
        dgemv_(&N_flag, &int_n_local, &int_n_local,
               &one, jhi, &int_n_local,
               q_io + (size_t)k_loc, &incx_qio,
               &zero, alpha, &int_one);

        /* J_kj[a*n_kj + m] = sum_p alpha[p] * raw_cross_p[p, a*n_kj + m]
         *                  = raw_cross_p^T @ alpha
         *
         * raw_cross_p is row-major (n_local, npno*n_kj). Treating it as
         * col-major (npno*n_kj, n_local), the operation y = A @ x with
         * A = raw_cross_p_col, x = alpha, y = J_out_p
         * → dgemv('N', npno*n_kj, n_local, ...).
         */
        const int N_cross = (int)((size_t)int_npno * (size_t)int_n_kj);
        dgemv_(&N_flag, &N_cross, &int_n_local,
               &one, raw_cross_p, &N_cross,
               alpha, &int_one,
               &zero, J_out_p, &int_one);

        /* K_kj (npno, n_kj) = Z_iv (npno, n_local) @ raw_kv_p (n_local, n_kj).
         * (Original Python: K = q_iv.T @ jhi @ raw_kv, fused via Z_iv.)  */
        dgemm_(&N_flag, &N_flag,
               &int_n_kj, &int_npno, &int_n_local,
               &one, raw_kv_p, &int_n_kj,
               Z_iv, &int_n_local,
               &zero, K_out_p, &int_n_kj);
    }

    free(alpha);
}
