/* DLPNO-CCSD compute_G_tilde inner kernel: per-(i,j) slot accumulator.
 *
 * Full-C cycle session 2 port. Mirrors Psi4 ccsd.cc:2085 (compute_G_tilde)
 * and the existing Cython kernel _g_tilde_batched_cy.pyx::g_tilde_batched.
 *
 * Plan-cached layout (built once per CCSD run by Python wrapper
 * residual.py:build_G_tilde):
 *
 *   For each triple (i, l, j) where (i,l) and (l,j) are stored pairs,
 *   the wrapper precomputes a static "effective" tensor folding the
 *   S_PNO projection and the Tt_lj orientation (l <= j vs l > j):
 *
 *     K_proj         = S_il_lj.T @ K_iajb[il] @ S_il_lj      (n_lj, n_lj)
 *     case_le        = 2*K_proj.T - K_proj    (used when l <= j)
 *     case_gt        = 2*K_proj   - K_proj.T  (used when l >  j)
 *     effective[t]   = case_le | case_gt depending on (l, j)
 *
 *   Per-iter contribution (single ddot per triple):
 *     G[i, j] += sum_{a, b} effective[t][a, b] * T2_canonical[lj][a, b]
 *
 * Outer parallel: prange-style loop over the (i, j) slot index. Each slot
 * writes G[i, j] disjointly, no reduction needed.
 *
 * Entry shapes:
 *   triple_eff_offset[n_triples]    — long, offset into effective_flat per t
 *   triple_T2_pair_idx[n_triples]   — long, which canonical T2 buffer to read
 *   triple_n_lj[n_triples]          — int, n_pno for the lj pair
 *   ij_triple_starts[n_ij_slots+1]  — long, slice [start, end) of triples per slot
 *   ij_i_arr[n_ij_slots]            — int, i-index per slot
 *   ij_j_arr[n_ij_slots]            — int, j-index per slot
 *   effective_flat                  — double, concatenated effective tensors (STATIC)
 *   T2_flat                         — double, concatenated canonical T2 (per-iter)
 *   T2_offsets[n_canon+1]           — long, offset per canonical pair
 *   G_addition[naocc, naocc]        — double C-order; caller initialises to Fkj
 *                                     and we accumulate into it.
 */

#include <stddef.h>

void DLPNOcompute_G_tilde_inner(const long *triple_eff_offset,
                                const long *triple_T2_pair_idx,
                                const int *triple_n_lj,
                                const long *ij_triple_starts,
                                const int *ij_i_arr,
                                const int *ij_j_arr,
                                const double *effective_flat,
                                const double *T2_flat,
                                const long *T2_offsets,
                                double *G_addition,
                                const size_t n_ij_slots,
                                const size_t naocc)
{
#pragma omp parallel for schedule(dynamic, 1)
    for (size_t slot = 0; slot < n_ij_slots; slot++) {
        const int i = ij_i_arr[slot];
        const int j = ij_j_arr[slot];
        const long t_start = ij_triple_starts[slot];
        const long t_end   = ij_triple_starts[slot + 1];

        double sum_ij = 0.0;
        for (long t = t_start; t < t_end; t++) {
            const long n_lj  = (long)triple_n_lj[t];
            const long n_lj2 = n_lj * n_lj;
            const double *eff_ptr = effective_flat + triple_eff_offset[t];
            const long pair_idx = triple_T2_pair_idx[t];
            const double *T2_ptr = T2_flat + T2_offsets[pair_idx];

            double contribution = 0.0;
            for (long k = 0; k < n_lj2; k++) {
                contribution += eff_ptr[k] * T2_ptr[k];
            }
            sum_ij += contribution;
        }

        G_addition[(size_t)i * naocc + (size_t)j] += sum_ij;
    }
}
