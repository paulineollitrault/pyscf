/* DLPNO-CCSD T2 residual: per-item G-term batched kernel.
 *
 * Full-C cycle session 11 port. Mirrors the existing Cython kernel
 * _g_term_batched_cy.pyx::g_term_batched byte-for-byte.
 *
 * Per-item math (one item per (ij, k) reduction step in the T2 residual):
 *
 *   scalar = G_tilde[k_idx[n], scalar_lmo[n]]
 *   tmp[a, b]  = sum_c S[a, c] * t2[c, b]                  (n_ij, n_ik)
 *   Cc[a, d]   = scalar * sum_b tmp[a, b] * S[d, b]        (n_ij, n_ij)
 *
 * The Python wrapper scatters the per-item Cc tiles into per-n_ij
 * output buffers afterwards (race-free).
 *
 * Outer #pragma omp parallel for over n; per-thread scratch tmp passed
 * in by caller (sized for max_n_ij * max_n_ik).
 */

#include <stddef.h>

#ifdef _OPENMP
#include <omp.h>
#endif

void DLPNOg_term_batched(const int     N,
                         const int    *n_ij_arr,
                         const int    *n_ik_arr,
                         const long   *S_off,
                         const long   *t2_off,
                         const long   *tile_off,
                         const long   *k_idx,
                         const long   *scalar_lmo,
                         const double *S_flat,
                         const double *t2_flat,
                         const double *G_tilde,
                         const size_t  G_stride,        /* nocc */
                         double       *tmp_scratch,
                         const size_t  tmp_stride,
                         double       *tiles_flat,
                         const int     num_threads)
{
#pragma omp parallel for schedule(dynamic, 1) num_threads(num_threads)
    for (int n = 0; n < N; n++) {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        const int n_ij = n_ij_arr[n];
        const int n_ik = n_ik_arr[n];

        const double *S  = S_flat   + S_off[n];        /* (n_ij, n_ik) */
        const double *t2 = t2_flat  + t2_off[n];       /* (n_ik, n_ik) */
        double       *tmp = tmp_scratch + (size_t)tid * tmp_stride;
        double       *Cc  = tiles_flat  + tile_off[n];  /* (n_ij, n_ij) */

        const double scalar = G_tilde[(size_t)k_idx[n] * G_stride + scalar_lmo[n]];

        /* tmp[a, b] = sum_c S[a, c] * t2[c, b] */
        for (int a = 0; a < n_ij; a++) {
            for (int b = 0; b < n_ik; b++) {
                double s = 0.0;
                for (int c = 0; c < n_ik; c++) {
                    s += S[a * n_ik + c] * t2[c * n_ik + b];
                }
                tmp[a * n_ik + b] = s;
            }
        }

        /* Cc[a, d] = scalar * sum_b tmp[a, b] * S[d, b] */
        for (int a = 0; a < n_ij; a++) {
            for (int d = 0; d < n_ij; d++) {
                double s = 0.0;
                for (int b = 0; b < n_ik; b++) {
                    s += tmp[a * n_ik + b] * S[d * n_ik + b];
                }
                Cc[a * n_ij + d] = scalar * s;
            }
        }
    }
}
