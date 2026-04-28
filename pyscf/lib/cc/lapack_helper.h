/* LAPACK Fortran-prototype declarations for DLPNO-(T) full-C kernel.
 *
 * OpenBLAS, MKL, and Netlib LAPACK all expose the same Fortran ABI.
 * We declare the routines we need explicitly here so we don't have to
 * pull in lapacke or scipy.
 */
#ifndef DLPNO_LAPACK_HELPER_H
#define DLPNO_LAPACK_HELPER_H

#ifdef __cplusplus
extern "C" {
#endif

/* Symmetric eigenproblem (divide-and-conquer): A = V * Λ * V^T.
 * Overwrites A with the eigenvectors when JOBZ='V'.
 *   JOBZ : 'N' | 'V'
 *   UPLO : 'L' | 'U'
 *   N    : order of A
 *   A    : (LDA,N) on entry, eigenvectors on exit
 *   LDA  : leading dim of A
 *   W    : (N,)   eigenvalues ascending
 *   WORK : (LWORK,) workspace
 *   IWORK: (LIWORK,) integer workspace
 *
 * Workspace query: LWORK = -1 returns optimal LWORK in WORK[0],
 *                  optimal LIWORK in IWORK[0].
 */
void dsyevd_(const char *jobz, const char *uplo,
             const int *n, double *a, const int *lda,
             double *w, double *work, const int *lwork,
             int *iwork, const int *liwork, int *info);

/* Pivoted Cholesky (partial) of a symmetric positive-semidefinite matrix.
 *   UPLO : 'L' | 'U'
 *   N    : order
 *   A    : (LDA,N) — overwritten on exit (factor in L or U)
 *   LDA  : leading dim
 *   PIV  : (N,) pivot vector (1-based)
 *   RANK : on exit, computed rank
 *   TOL  : threshold (negative → use machine default)
 *   WORK : (2*N,) workspace
 *   INFO : 0 = success
 */
void dpstrf_(const char *uplo, const int *n, double *a, const int *lda,
             int *piv, int *rank, const double *tol,
             double *work, int *info);

#ifdef __cplusplus
}
#endif

#endif  /* DLPNO_LAPACK_HELPER_H */
