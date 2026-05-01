/* DLPNO G_tilde plan triple-enumeration: O(nocc^3) loop in C.
 *
 * Replaces the Python triple loop in residual.py::build_G_tilde plan
 * (PASS 1: enumerate valid triples). The loop iterates all (i, j, l)
 * tuples and records valid ones (where canon_lut[i,l] and canon_lut[l,j]
 * are canonical pairs and p_lmos_dense entries are in-domain).
 *
 * Output layout matches the Python lists 1:1:
 *   t_canon_il[N_t]      - canonical pair idx for (i, l)
 *   t_i_idx[N_t]         - global i
 *   t_l_idx[N_t]         - global l
 *   t_S_off[N_t]         - offset into S_pno_cache buffer (-1 if canon_il
 *                          == canon_lj, -3 if cache miss → side buffer)
 *   t_S_sel[N_t]         - 0 = main buffer, 1 = side buffer (lazy)
 *   t_canon_lj[N_t]      - canonical pair idx for (l, j)
 *   t_n_il[N_t]          - n_pno of (i,l) pair
 *   t_n_lj[N_t]          - n_pno of (l,j) pair
 *   t_l_le_j[N_t]        - 1 if l <= j else 0
 *   ij_triple_offsets[nocc²+1] - cumulative count per (i, j) slot
 *
 * Plus a per-(i, l) lookup populated by the kernel:
 *   kil_idx_for_il[i*nocc + l] = unique kil index in [0, kil_count)
 *                                or -1 if (i,l) has no valid pair entry.
 *
 * This makes the unique-(canon_il, i, l) numbering deterministic — every
 * valid (i, l) gets one kil entry, and the entry index is just sequential
 * over the (i, l) sweep. The downstream K_il pool build iterates this
 * array directly to fill K_il_pool.
 *
 * A pre-pass computes the global p_check[i*nocc + l] mask (1 if
 * canon_lut[i,l] >= 0 AND canon_p_dense_valid AND p_dense[i] >= 0
 * AND p_dense[l] >= 0). Caller flattens canon_p_dense into a contiguous
 * (n_canon, nocc) buffer with -2 placeholder for None entries.
 */

#include <stddef.h>

void DLPNObuild_G_tilde_plan_enum(
        const long *canon_lut,             /* (nocc, nocc), -1 if no pair */
        const long *canon_p_dense_flat,    /* (n_canon, nocc), -2 if None entry */
        const signed char *canon_p_valid,  /* n_canon, 1 if canon_p_dense != None */
        const long *canon_to_S_pi,         /* n_canon, -1 if no S_pi */
        const long *S_idx_matrix,          /* (n_pi, n_pi), -1 if no entry */
        const long *S_offsets_arr,         /* indexed by S_k */
        const int *canon_n_pno,            /* n_canon */
        long *t_canon_il, long *t_i_idx, long *t_l_idx,
        long *t_S_off, signed char *t_S_sel,
        long *t_canon_lj, int *t_n_il, int *t_n_lj,
        signed char *t_l_le_j,
        long *ij_triple_offsets,           /* (nocc² + 1) */
        long *kil_idx_for_il,              /* (nocc, nocc) */
        long *N_t_out, long *kil_count_out,
        const long nocc, const long n_pi)
{
    /* Pre-pass: compute kil_idx for each valid (i, l).
     * Valid means canon_lut[i,l] >= 0, canon_p_valid for that canon, and
     * p_dense[i] >= 0 && p_dense[l] >= 0.
     */
    long kil_count = 0;
    for (long i = 0; i < nocc; i++) {
        for (long l = 0; l < nocc; l++) {
            const long canon_il = canon_lut[i * nocc + l];
            long kil = -1;
            if (canon_il >= 0 && canon_p_valid[canon_il]) {
                const long *p_dense = canon_p_dense_flat + canon_il * nocc;
                if (p_dense[i] >= 0 && p_dense[l] >= 0) {
                    kil = kil_count++;
                }
            }
            kil_idx_for_il[i * nocc + l] = kil;
        }
    }
    *kil_count_out = kil_count;

    /* Main loop: enumerate triples (i, j, l). Outer (i, j) determines
     * the ij_slot; inner l adds triples for that slot.
     */
    long n_t = 0;
    long ij_slot = 0;
    ij_triple_offsets[0] = 0;
    for (long i = 0; i < nocc; i++) {
        for (long j = 0; j < nocc; j++) {
            for (long l = 0; l < nocc; l++) {
                const long canon_il = canon_lut[i * nocc + l];
                if (canon_il < 0) continue;
                const long canon_lj = canon_lut[l * nocc + j];
                if (canon_lj < 0) continue;
                if (!canon_p_valid[canon_il]) continue;
                const long *p_dense = canon_p_dense_flat + canon_il * nocc;
                if (p_dense[i] < 0 || p_dense[l] < 0) continue;

                long S_off;
                signed char S_sel;
                if (canon_il == canon_lj) {
                    S_off = -1;
                    S_sel = 0;
                } else {
                    const long pi_il = canon_to_S_pi[canon_il];
                    const long pi_lj = canon_to_S_pi[canon_lj];
                    if (pi_il < 0 || pi_lj < 0) continue;
                    const long S_k = S_idx_matrix[pi_il * n_pi + pi_lj];
                    if (S_k < 0) {
                        S_off = -3;   /* placeholder; caller patches via side */
                        S_sel = 1;
                    } else {
                        S_off = S_offsets_arr[S_k];
                        S_sel = 0;
                    }
                }

                t_canon_il[n_t] = canon_il;
                t_i_idx[n_t]    = i;
                t_l_idx[n_t]    = l;
                t_S_off[n_t]    = S_off;
                t_S_sel[n_t]    = S_sel;
                t_canon_lj[n_t] = canon_lj;
                t_n_il[n_t]     = canon_n_pno[canon_il];
                t_n_lj[n_t]     = canon_n_pno[canon_lj];
                t_l_le_j[n_t]   = (l <= j) ? 1 : 0;
                n_t++;
            }
            ij_slot++;
            ij_triple_offsets[ij_slot] = n_t;
        }
    }
    *N_t_out = n_t;
}
