/* DLPNO-CCSD t1_residual: OMP-over-i wrapper for per-i stages123.
 *
 * Replaces `pool.map(_per_i, range(nocc))` in
 * pyscf/cc/dlpno_tccsd/lccsd.py:_compute_t1_residual_psi4.
 *
 * Per-cycle wall on water-10 (32-pool):
 *   PER_I_DBG: wall=192ms stages_cy=1773ms (CPU sum across pool)
 * Per-thread C work cumulative = 1773ms / 32 ≈ 56ms wall.
 * Wall - C_work = 192 - 56 = 136ms = ~70% Python orchestration overhead.
 *
 * This kernel eliminates that overhead by running the per-i loop in
 * C with `#pragma omp parallel for`, calling the existing per-pair
 * Stages 1-3 kernel (DLPNOper_i_stages123) inside.
 *
 * Inputs (one entry per i, indexed 0..nocc-1):
 *   diag_pk_idx[i]: pair_idx of (i, i) in PairIndex; -1 if no key (skip i)
 *   r1_offsets[i+1] - r1_offsets[i]: n_pno of pair (i, i)
 *
 *   FlatTensorStore field buffers (cc_ints):
 *     Qma_buf, Qma_off (nocc, nlmo_p, n_pno) per pair
 *     Qab_buf, Qab_off (nocc, n_pno, n_pno)
 *     Qia_buf, Qia_off (nocc, n_pno) for diagonal
 *     Qik_buf, Qik_off (nocc, nlmo_p) for diagonal
 *
 *   Per-pair dims:
 *     L_arr[i] = n_local (aux dimension of cc_ints[(i,i)])
 *     M_arr[i] = nlmo_p
 *     A_arr[i] = n_pno of (i, i) PNO space
 *
 *   T1 cache buffer (per pair) projecting t1[m] into PNO_(i,i):
 *     T1c_buf, T1c_off  (per-pair shape (nocc, n_pno))
 *
 *   p_lmos (per-pair LMO subset for pair-domain T_n):
 *     p_lmos_off, p_lmos_buf
 *
 *   t1 (per-i flat amplitudes):
 *     t1_off, t1_buf  (per-i shape (n_pno_ii,))
 *     any_t1[i]: 1 if T_n_ii has nonzero values
 *     has_t1_i[i]: 1 if t1[i] has nonzero values
 *
 *   e_pno (per-i diagonal):
 *     epno_off, epno_buf
 *
 *   T_n_ii_pair scratch (per-i, n_local × n_pno × n_lmo_p — large, prebuilt)
 *     Tn_pair_off, Tn_pair_buf
 *
 * Output:
 *   r1_buf: (sum n_pno_ii,) flat, indexed by r1_offsets[i].
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Forward declaration — kernel in dlpno_per_i.c */
void DLPNOper_i_stages123(double *r1_inout,
                           double *Qma, double *Qab,
                           double *Qia, double *Qik,
                           double *T_n, double *t1_i,
                           double *e_pno,
                           int do_stage_23,
                           size_t L, size_t M, size_t A);

void DLPNOcompute_per_i_stages123_omp(
        const int nocc,
        /* per-i: pair-pair lookup indices */
        const long *diag_pk_idx,         /* (nocc,) — pair_idx of (i,i) or -1 */
        /* per-i shapes */
        const long *L_arr,               /* (nocc,) n_local for diagonal pair */
        const long *M_arr,               /* (nocc,) nlmo_p */
        const long *A_arr,               /* (nocc,) n_pno of (i,i) */
        const signed char *any_t1_arr,   /* (nocc,) 1 if T_n[i] has values */
        const signed char *has_t1_i_arr, /* (nocc,) 1 if t1[i] nonzero */
        /* FlatTensorStore offsets for the cc_ints fields (indexed by diag_pk_idx[i]) */
        const long *Qma_off,             /* (n_pairs+1,) */
        const long *Qab_off,
        const long *Qia_off,
        const long *Qik_off,
        /* FlatTensorStore buffers */
        double *Qma_buf,
        double *Qab_buf,
        double *Qia_buf,
        double *Qik_buf,
        /* T_n_ii_pair: per-i pre-built buffer with offsets */
        const long *Tn_pair_off,         /* (nocc+1,) */
        double *Tn_pair_buf,
        /* t1 per-i offsets/buf */
        const long *t1_off,              /* (nocc+1,) */
        double *t1_buf,
        /* e_pno per-i */
        const long *epno_off,            /* (nocc+1,) */
        double *epno_buf,
        /* output r1 per-i */
        const long *r1_off,              /* (nocc+1,) */
        double *r1_buf)
{
    /* Auto-cap OMP threads — same pattern as DLPNOcompute_E_T0_omp:
     * memory wall hits at >16 threads on this 64-physical-core box. */
#ifdef _OPENMP
    {
        const char *omp_env = getenv("DLPNO_CCSD_OMP_THREADS");
        int desired = 0;
        if (omp_env) desired = atoi(omp_env);
        if (desired <= 0) {
            const int omp_max = omp_get_max_threads();
            desired = omp_max < 16 ? omp_max : 16;
        }
        omp_set_num_threads(desired);
    }
#endif

#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < nocc; i++) {
        const long pk = diag_pk_idx[i];
        if (pk < 0) continue;
        const long n_pno = A_arr[i];
        if (n_pno == 0) continue;

        const int do_stage_23 = (any_t1_arr[i] != 0);
        const int has_t1_i    = (has_t1_i_arr[i] != 0);
        if (!do_stage_23 && !has_t1_i) continue;

        const long L = L_arr[i];
        const long M = M_arr[i];
        const long A = n_pno;

        /* Pointers into the flat buffers */
        double *Qma_i = Qma_buf + Qma_off[pk];
        double *Qab_i = Qab_buf + Qab_off[pk];
        double *Qia_i = Qia_buf + Qia_off[pk];
        double *Qik_i = Qik_buf + Qik_off[pk];
        double *Tn_i  = Tn_pair_buf + Tn_pair_off[i];
        double *t1_i  = t1_buf + t1_off[i];
        double *epno_i= epno_buf + epno_off[i];
        double *r1_i  = r1_buf + r1_off[i];

        /* If t1[i] is empty (size 0 in offsets), pass a zero stub. */
        if (!has_t1_i) {
            /* Use a stack zero buffer of size A.  A ≤ ~50, fits on stack. */
            double *t1_zero = (double *)alloca(sizeof(double) * (size_t)A);
            memset(t1_zero, 0, sizeof(double) * (size_t)A);
            t1_i = t1_zero;
        }

        /* Zero r1 output before kernel — DLPNOper_i_stages123 ACCUMULATES. */
        memset(r1_i, 0, sizeof(double) * (size_t)A);

        DLPNOper_i_stages123(
            r1_i,
            Qma_i, Qab_i, Qia_i, Qik_i,
            Tn_i, t1_i, epno_i,
            do_stage_23,
            (size_t)L, (size_t)M, (size_t)A);
    }
}
