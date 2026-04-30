/* DLPNO LMP2 inter-pair F-coupling residual: batched C kernel.
 *
 * Replaces the Python loop in pno.py::Phase 2b residual that calls
 * S @ T2 @ S.T thousands of times per LMP2 iteration. Each call
 * is small (npno~10), so Python+ctypes dispatch dominates.
 *
 * Plan-cached layout (built once per LMP2 run by Python wrapper):
 *   For each task t affecting target pair p_target with F-neighbor partner:
 *     R[p_target] -= F_coeff[t] * S[t] @ T2[partner_or_T] @ S[t].T
 *
 * Tasks for a given target pair are listed contiguously in task arrays;
 * target_task_starts[p_target..p_target+1] gives the slice.
 *
 * OMP parallel over target pairs (each writes only its own R buffer →
 * no reduction). Each task does 2 small DGEMMs via BLAS.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void dgemm_(const char*, const char*,
            const int*, const int*, const int*,
            const double*, const double*, const int*,
            const double*, const int*,
            const double*, double*, const int*);

/* Per-target-pair entry kernel: process all tasks for one pair sequentially.
 * scratch must be at least max_n_partner * max_n_pno doubles.
 */
static inline void process_one_pair(
        long p,
        const long *target_task_starts,
        const long *task_partner_idx,
        const double *task_F_coeff,
        const long *task_S_off,
        const signed char *task_transpose,
        const int *n_pno_arr,
        const double *T2_flat,
        const long *T2_offsets,
        const double *S_flat,
        double *R_flat,
        const long *R_offsets,
        double *scratch)
{
    const long t_start = target_task_starts[p];
    const long t_end   = target_task_starts[p + 1];
    if (t_start == t_end) return;

    const int n_p = n_pno_arr[p];
    if (n_p == 0) return;
    double *R_p = R_flat + R_offsets[p];

    const double done = 1.0;
    const double dzero = 0.0;
    const double dneg_one = -1.0;

    for (long t = t_start; t < t_end; t++) {
        const long partner = task_partner_idx[t];
        const int n_partner = n_pno_arr[partner];
        if (n_partner == 0) continue;

        const double F_coeff = task_F_coeff[t];
        const double *S = S_flat + task_S_off[t];               /* (n_p, n_partner), C-order */
        const double *T2_partner = T2_flat + T2_offsets[partner]; /* (n_partner, n_partner) */

        /* DGEMM is column-major. Treat C-order arrays as Fortran-order
         * with swapped operand order:
         *   C-order (M, K) * (K, N) = (M, N)
         *   ↔ Fortran (K, M) * (N, K) = (N, M)
         *   call dgemm('N','N', N, M, K, ...)
         *
         * tmp1 (n_p, n_partner) = S (n_p, n_partner) @ T2_use (n_partner, n_partner)
         *   if transpose[t]: T2_use = T2_partner.T
         *   else:            T2_use = T2_partner
         */
        const char *t2_op_F;  /* Fortran-side transpose flag for T2 */
        if (task_transpose[t]) {
            /* T2_use[a, b] = T2_partner[b, a] (C-order)
             * As Fortran (n_partner, n_partner) col-major,
             * T2_partner is its own transpose-storage swap of C-order;
             * to compute C @ Tp where C is (n_p, n_partner) and Tp is (n_partner, n_partner):
             *   we want tmp[a,b] = sum_c S[a,c] * Tp[c,b], where Tp = T2_partner.T (C-order)
             *   so tmp[a,b] = sum_c S[a,c] * T2_partner[b,c]
             * In Fortran-side: dgemm('T','N', n_partner, n_p, n_partner, 1, T2_partner_F, n_partner, S_F, n_partner, 0, tmp_F, n_partner)
             * where T2_partner_F is the col-major view of C-order T2_partner.
             * Note: C-order (M,K) presented as Fortran is (K,M). So calling with op='T'
             * on the Fortran side means we transpose T2_partner_F, which corresponds to
             * NOT transposing T2_partner in C-order.
             *
             * Easier mental model: when matrices are C-order:
             *   C = A @ B   in C-order
             *   ↔ dgemm('N','N', N_B, N_A, K, alpha, B, ldb=N_B, A, lda=K, beta, C, ldc=N_B)
             *   where A is (M, K), B is (K, N_B), C is (M, N_B).
             *
             *   For C = A @ B.T where B is (M_B, K) (so B.T is (K, M_B)):
             *     dgemm('T','N', M_B, N_A, K, ..., B, ldb=K, A, ..., C, ldc=M_B)
             *
             *   For C = A.T @ B where A is (K, M_A):
             *     dgemm('N','T', N_B, M_A, K, ..., B, ldb=N_B, A, ldb_a=M_A, ..., C, ldc=N_B)
             */
            /* tmp1 = S @ T2_partner.T : (n_p, n_partner) = (n_p, n_partner) @ (n_partner, n_partner) */
            int M = n_p, N = n_partner, K = n_partner;
            int lda_F = N, ldb_F = K, ldc_F = N;
            /* Fortran call: tmp1_F (N, M) = T2_F (N, K, op='T') @ S_F (K, M, op='N') */
            dgemm_("T", "N", &N, &M, &K, &done, T2_partner, &K, S, &K,
                   &dzero, scratch, &N);
        } else {
            /* tmp1 = S @ T2_partner : same M,N,K */
            int M = n_p, N = n_partner, K = n_partner;
            dgemm_("N", "N", &N, &M, &K, &done, T2_partner, &N, S, &K,
                   &dzero, scratch, &N);
        }

        /* tmp2 = tmp1 @ S.T : (n_p, n_p) = (n_p, n_partner) @ (n_partner, n_p)
         * We want R_p -= F_coeff * tmp2.
         * Fortran-side: dgemm to update R_p (C-order n_p × n_p):
         *   R_F (n_p, n_p) -= F_coeff * S_F (n_partner, n_p, 'T') @ tmp1_F (n_partner, n_p, 'N')
         *
         * Standard C-order: R = tmp1 @ S.T where tmp1=(M,K), S=(M2,K), so S.T=(K,M2)=(n_partner,n_p)
         *   dgemm('T','N', M2, M, K, alpha, S, ldb=K, tmp1, lda=K, beta, R, ldc=M2)
         */
        {
            const double alpha = -F_coeff;
            int M = n_p;          /* rows of tmp1 (and rows of R) */
            int N2 = n_p;          /* cols of S.T (and cols of R) */
            int K = n_partner;
            /* Fortran: R_F (N2, M) += alpha * S_F (K, N2, 'T') @ tmp1_F (K, M, 'N') */
            dgemm_("T", "N", &N2, &M, &K, &alpha, S, &K, scratch, &K,
                   &done, R_p, &N2);
        }
    }
}

/* Top-level kernel.
 *
 * Inputs:
 *   target_task_starts[n_pairs+1]
 *   task_partner_idx[n_tasks]    (long)
 *   task_F_coeff[n_tasks]        (double)
 *   task_S_off[n_tasks]          (long)
 *   task_transpose[n_tasks]      (int8: 0/1)
 *   n_pno_arr[n_pairs]           (int)
 *   T2_flat                      (double, total = sum n_pno**2)
 *   T2_offsets[n_pairs+1]
 *   S_flat                       (double, prebuilt)
 *   R_flat                       (double, must be initialized by caller
 *                                 with K_pno + D*T2 part)
 *   R_offsets[n_pairs+1]
 *   max_n_pno                    (int) for scratch sizing
 *   n_pairs                      (size_t)
 *   n_threads                    (int) — OMP threads (capped at n_pairs)
 */
void DLPNOlmp2_residual_batched(
        const long *target_task_starts,
        const long *task_partner_idx,
        const double *task_F_coeff,
        const long *task_S_off,
        const signed char *task_transpose,
        const int *n_pno_arr,
        const double *T2_flat,
        const long *T2_offsets,
        const double *S_flat,
        double *R_flat,
        const long *R_offsets,
        const int max_n_pno,
        const size_t n_pairs,
        const int n_threads)
{
    const size_t scratch_per_thread = (size_t)max_n_pno * (size_t)max_n_pno;

#pragma omp parallel num_threads(n_threads)
    {
        double scratch_local[8192];   /* fast path for small npno */
        double *scratch = scratch_local;
        double *scratch_heap = NULL;

        if (scratch_per_thread > 8192) {
            /* Fall back to heap allocation */
            scratch_heap = (double*) malloc(scratch_per_thread * sizeof(double));
            scratch = scratch_heap;
        }

#pragma omp for schedule(dynamic, 8)
        for (size_t p = 0; p < n_pairs; p++) {
            process_one_pair(
                (long)p, target_task_starts, task_partner_idx,
                task_F_coeff, task_S_off, task_transpose, n_pno_arr,
                T2_flat, T2_offsets, S_flat, R_flat, R_offsets, scratch);
        }

        if (scratch_heap) free(scratch_heap);
    }
}
