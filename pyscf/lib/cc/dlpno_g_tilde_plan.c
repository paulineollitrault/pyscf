/* DLPNO-CCSD G_tilde plan-build inner kernel: per-triple K_proj.
 *
 * Replaces the per-triple `S^T @ K_il @ S` numpy/BLAS dispatch in
 * residual.py::build_G_tilde plan-build path. Profiled cost on
 * water-10 cycle 1: 92% of 2.5s plan-build wall is K_proj (64,000
 * matmul triples × ~30us numpy/BLAS dispatch overhead).
 *
 * Per-triple math:
 *   K_proj      = S_il_lj^T @ K_il @ S_il_lj            (n_lj × n_lj)
 *   if l <= j:  effective[d, c] = 2*K_proj[c, d] - K_proj[d, c]
 *   else:       effective[d, c] = 2*K_proj[d, c] - K_proj[c, d]
 *
 * For self-pair (S_off == -1), S_il_lj = I and K_proj = K_il (n_il
 * == n_lj guaranteed when canon_il == canon_lj).
 *
 * Inputs (all flat row-major, indexed per-triple):
 *   N_t                       — number of triples
 *   n_pno_max                 — max n_pno across pairs (scratch sizing)
 *   K_il_offsets[N_t]         — offset into K_il_pool per triple
 *   K_il_pool                 — concatenated per-(canon_il, i, l) K_il
 *                               (n_il × n_il row-major)
 *   triple_n_il[N_t]          — n_il per triple
 *   S_offsets[N_t]            — offset into S_buffer (-1 for self-pair)
 *   S_buffer                  — FlatPairPairStore flat tier
 *   triple_n_lj[N_t]          — n_lj per triple
 *   eff_offsets[N_t]          — offset into eff_flat per triple
 *   eff_flat                  — output, sum-of-(n_lj²) doubles
 *   l_le_j[N_t]               — orientation (1 if l <= j else 0)
 *
 * Output: eff_flat (overwrites in-place per triple).
 *
 * OMP parallel over triples; each thread holds 2*n_pno_max² scratch
 * for tmp + K_proj. Sized once at function entry.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "vhf/fblas.h"


void DLPNOcompute_kproj_batched(
        const long N_t,
        const int  n_pno_max,
        const long *K_il_offsets,
        const double *K_il_pool,
        const int  *triple_n_il,
        const long *S_offsets,
        const double *S_buffer,        /* main FlatPairPairStore buffer */
        const double *S_side_buffer,   /* side buffer for lazy-computed S
                                          (NULL if all S in main buffer) */
        const signed char *S_buf_sel,  /* per-triple: 0 = main, 1 = side,
                                          ignored when S_offsets[t] < 0 */
        const int  *triple_n_lj,
        const long *eff_offsets,
        double *eff_flat,
        const signed char *l_le_j,
        const int n_threads_in)
{
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;
    const size_t scratch_sz = (size_t)2 * (size_t)n_pno_max * (size_t)n_pno_max;
    int n_threads = n_threads_in;
#ifdef _OPENMP
    if (n_threads <= 0) {
        const int omp_max = omp_get_max_threads();
        n_threads = omp_max < 16 ? omp_max : 16;
    }
    omp_set_num_threads(n_threads);
#endif

#pragma omp parallel
    {
        double *scratch = (double *)malloc(sizeof(double) * scratch_sz);
        double *tmp_buf = scratch;
        double *Kp_buf  = scratch + (size_t)n_pno_max * n_pno_max;

#pragma omp for schedule(dynamic, 64)
        for (long t = 0; t < N_t; t++) {
            const int n_il = triple_n_il[t];
            const int n_lj = triple_n_lj[t];
            const long K_off = K_il_offsets[t];
            const long S_off = S_offsets[t];
            const long eff_off = eff_offsets[t];
            const double *K_il = K_il_pool + K_off;
            double *eff_out = eff_flat + eff_off;
            const double *Kp;

            if (S_off < 0) {
                /* Self-pair: K_proj = K_il (n_il == n_lj). */
                Kp = K_il;
            } else {
                const double *S = (S_buf_sel[t] != 0)
                                  ? (S_side_buffer + S_off)
                                  : (S_buffer + S_off);
                int int_n_il = n_il;
                int int_n_lj = n_lj;

                /* Step 1: tmp[n_lj, n_il] = S^T @ K_il   (row-major)
                 *
                 * Translation to Fortran column-major (dgemm sees
                 * row-major S as F-order S^T, etc.):
                 *   tmp_F (n_il, n_lj) = K_F @ S_F^T = K^T_math @ S_math
                 *   The row-major view of the same memory is
                 *   (n_lj, n_il) = S^T @ K  (math) ✓
                 *
                 * dgemm('N', 'T', m=n_il, n=n_lj, k=n_il,
                 *       1, K_il, lda=n_il, S, ldb=n_lj,
                 *       0, tmp, ldc=n_il)
                 */
                dgemm_(&N_flag, &T_flag,
                       &int_n_il, &int_n_lj, &int_n_il,
                       &one, K_il, &int_n_il,
                       S, &int_n_lj,
                       &zero, tmp_buf, &int_n_il);

                /* Step 2: K_proj[n_lj, n_lj] = tmp @ S
                 *
                 * Row-major: K_proj = tmp @ S, where tmp is row-major
                 * (n_lj, n_il), S is row-major (n_il, n_lj). In
                 * Fortran:
                 *   K_proj_col[n_lj, n_lj] = S_col @ tmp_col
                 *
                 * dgemm('N', 'N', n_lj, n_lj, n_il,
                 *       S, n_lj, tmp, n_il, 0, Kp, n_lj)
                 */
                dgemm_(&N_flag, &N_flag,
                       &int_n_lj, &int_n_lj, &int_n_il,
                       &one, S, &int_n_lj,
                       tmp_buf, &int_n_il,
                       &zero, Kp_buf, &int_n_lj);
                Kp = Kp_buf;
            }

            /* Write effective[d, c] (row-major (n_lj, n_lj)):
             *   if l <= j: eff[c, d] = 2*K_proj[d, c] - K_proj[c, d]
             *   else:      eff[c, d] = 2*K_proj[c, d] - K_proj[d, c]
             *
             * (Matches Python: case_le = 2*K_proj.T - K_proj, etc.)
             */
            if (l_le_j[t]) {
                for (int c = 0; c < n_lj; c++) {
                    for (int d = 0; d < n_lj; d++) {
                        const double v_cd = Kp[c * n_lj + d];
                        const double v_dc = Kp[d * n_lj + c];
                        eff_out[c * n_lj + d] = 2.0 * v_dc - v_cd;
                    }
                }
            } else {
                for (int c = 0; c < n_lj; c++) {
                    for (int d = 0; d < n_lj; d++) {
                        const double v_cd = Kp[c * n_lj + d];
                        const double v_dc = Kp[d * n_lj + c];
                        eff_out[c * n_lj + d] = 2.0 * v_cd - v_dc;
                    }
                }
            }
        }
        free(scratch);
    }
}
