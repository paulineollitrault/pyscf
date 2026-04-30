/* DLPNO-CCSD compute_B_tilde: per-pair B_tilde construction.
 *
 * Math (per ordered strong pair ij; output is the per-pair B[k_ij, l_ij]):
 *
 *   B[k, l]  = sum_Q i_Qk_t1[Q, k] * j_Qk_t1[Q, l]                     (Term 1)
 *            + sum_{Q, a, b} Qma[Q, k, a] * T2[a, b] * Qma[Q, l, b]    (Term 2)
 *
 * Entry shapes (all C-contiguous double):
 *   B_out:      (nlmo, nlmo)      — output, fully overwritten
 *   i_Qk_t1:    (n_local, nlmo)
 *   j_Qk_t1:    (n_local, nlmo)
 *   Qma:        (n_local, nlmo, npno)
 *   T2:         (npno, npno)
 *
 * Implementation: BLAS DGEMM throughout.  No internal OpenMP — outer
 * parallelism (over pairs) is provided by the caller; nested OMP would
 * oversubscribe.
 *
 * Term 1: B = j_Qk^T_F @ i_Qk_F^T_F seen column-major, i.e. via
 *         dgemm('N', 'T', nlmo, nlmo, n_local, 1, j_Qk, nlmo, i_Qk, nlmo, 0, B, nlmo)
 *   reads i_Qk/j_Qk as (n_local, nlmo) row-major = (nlmo, n_local) col-major.
 *
 * Term 2:
 *   Step A (one big GEMM): P_flat[Q*nl+k, b] = sum_a Qma_flat[Q*nl+k, a] T2[a, b]
 *     dgemm('N', 'N', npno, n_local*nlmo, npno, 1, T2, npno, Qma, npno, 0, P, npno)
 *   Step B (per-Q small GEMMs): B[k, l] += P_Q[k, :] @ Qma_Q[l, :]^T
 *     dgemm('T', 'N', nlmo, nlmo, npno, 1, Qma_Q, npno, P_Q, npno, 1, B, nlmo)
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

void DLPNOcompute_B_tilde_pair(double *B_out,
                               const double *i_Qk_t1,
                               const double *j_Qk_t1,
                               const double *Qma,
                               const double *T2,
                               const size_t n_local,
                               const size_t nlmo,
                               const size_t npno)
{
    if (nlmo == 0 || n_local == 0) {
        if (nlmo > 0) memset(B_out, 0, sizeof(double) * nlmo * nlmo);
        return;
    }

    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;
    int int_nlmo = (int)nlmo;
    int int_npno = (int)npno;
    int int_n_local = (int)n_local;

    /* Term 1: B (nlmo, nlmo) = i_Qk^T (nlmo, n_local) @ j_Qk (n_local, nlmo)
     *
     * Row-major math: B[k, l] = sum_Q i_Qk[Q, k] * j_Qk[Q, l].
     * In Fortran column-major view of row-major (n_local, nlmo) arrays
     * shape becomes (nlmo, n_local).  Compute B_F = j_F @ i_F^T:
     *   dgemm('N', 'T', m=nlmo, n=nlmo, k=n_local, alpha=1,
     *         A=j_Qk (lda=nlmo), B=i_Qk (ldb=nlmo), beta=0,
     *         C=B (ldc=nlmo))
     */
    dgemm_(&N_flag, &T_flag,
           &int_nlmo, &int_nlmo, &int_n_local,
           &one,
           j_Qk_t1, &int_nlmo,
           i_Qk_t1, &int_nlmo,
           &zero,
           B_out, &int_nlmo);

    if (npno == 0) return;

    /* Term 2 step A: P[Q*nl+k, b] = sum_a Qma[Q*nl+k, a] * T2[a, b]
     * One big GEMM over flattened (Q, k) row-axis.
     *
     * Row-major math: P (n_local*nlmo, npno) = Qma (n_local*nlmo, npno) @ T2 (npno, npno)
     * Fortran view: P_F (npno, n_local*nlmo) = T2_F @ Qma_F.
     *   dgemm('N', 'N', m=npno, n=n_local*nlmo, k=npno, alpha=1,
     *         A=T2 (lda=npno), B=Qma (ldb=npno), beta=0,
     *         C=P (ldc=npno))
     */
    const size_t P_size = n_local * nlmo * npno;
    double *P_flat = (double *)malloc(sizeof(double) * P_size);
    if (P_flat == NULL) return;

    int int_n_local_nlmo = (int)(n_local * nlmo);
    dgemm_(&N_flag, &N_flag,
           &int_npno, &int_n_local_nlmo, &int_npno,
           &one,
           T2,  &int_npno,
           Qma, &int_npno,
           &zero,
           P_flat, &int_npno);

    /* Term 2 step B: For each Q, accumulate
     *   B[k, l] += sum_b P_Q[k, b] * Qma_Q[l, b]
     * Per-Q row-major: B (nl, nl) += P_Q (nl, np) @ Qma_Q^T (np, nl)
     * Fortran view of B (square): B_F[l, k] += sum_b Qma_Q_F[b, l] * P_Q_F[b, k]
     *   = (Qma_Q_F^T @ P_Q_F)
     *   dgemm('T', 'N', m=nlmo, n=nlmo, k=npno, alpha=1,
     *         A=Qma_Q (lda=npno), B=P_Q (ldb=npno), beta=1,
     *         C=B (ldc=nlmo))
     */
    const size_t Q_stride_qma = nlmo * npno;
    const size_t Q_stride_p   = nlmo * npno;
    for (size_t Q = 0; Q < n_local; ++Q) {
        const double *P_Q   = P_flat + Q * Q_stride_p;
        const double *Qma_Q = Qma    + Q * Q_stride_qma;
        dgemm_(&T_flag, &N_flag,
               &int_nlmo, &int_nlmo, &int_npno,
               &one,
               Qma_Q, &int_npno,
               P_Q,   &int_npno,
               &one,
               B_out, &int_nlmo);
    }

    free(P_flat);
}
