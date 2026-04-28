/* DLPNO-(T) full per-triple kernel (Session 3, Phase 3a).
 *
 * Single C entry point that performs the entire
 * `_triple_pno_union_psi4` body — including LAPACK calls (pivoted
 * Cholesky via DPSTRF and symmetric eigh via DSYEVD) — in one C
 * function. Replaces the hybrid Python+C path landed in Session 2.
 *
 * Per triple (i,j,k), given:
 *   triple_paos:    union of pair_paos for the 3 pair keys (sorted, int64)
 *   F_pao_full:     (n_pao_total, n_pao_total)
 *   S_pao_full:     (n_pao_total, n_pao_total)
 *   per-key arrays: pair_paos / X_pno / T2 / same_lmo
 *
 * Computes:
 *   1. orthogonalize_pao_domain (Psi4 PartialCholesky):
 *      X_pao_ijk (n_pao_ijk, n_pao_can)
 *   2. F_orth_ijk = X_pao_ijk^T F_pao[trip,trip] X_pao_ijk
 *   3. D_ijk = (1/3) Σ_keys S^T D_pair S (Session 2 body, inlined)
 *   4. eigh(D_ijk), truncate at T_CutTNO → X_tno_initial (n_pao_can, n_tno)
 *   5. F_in_tno = X_tno_initial^T F_orth X_tno_initial
 *   6. eigh(F_in_tno) → eps_tno, tno_canon
 *   7. X_tno_canonical = X_tno_initial @ tno_canon
 *      X_tno_ijk = X_pao_ijk @ X_tno_canonical
 *
 * Outputs:
 *   X_tno_ijk_out (n_pao_ijk, n_tno)
 *   eps_tno_out   (n_tno,)
 *   F_orth_out    (n_pao_can, n_pao_can)
 *   D_ijk_out     (n_pao_can, n_pao_can)   [for diagnostics; can be NULL]
 *   X_pao_ijk_out (n_pao_ijk, n_pao_can)   [optional]
 *   n_tno_out, n_pao_can_out (int)
 *
 * Returns: 0 on success, nonzero on (LAPACK) failure.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vhf/fblas.h"
#include "lapack_helper.h"

/* --------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------- */
static void gather_submatrix_d(
        const double *src, const long *rows, const long *cols,
        const int n_rows, const int n_cols, const int n_full,
        double *dst)
{
    for (int r = 0; r < n_rows; r++) {
        const double *src_row = src + (size_t)rows[r] * (size_t)n_full;
        double *dst_row = dst + (size_t)r * (size_t)n_cols;
        for (int c = 0; c < n_cols; c++) {
            dst_row[c] = src_row[cols[c]];
        }
    }
}

/* eigh of a symmetric matrix A (in-place). On exit A holds eigenvectors,
 * w holds eigenvalues. Returns 0 on success.
 *
 * A is row-major (n, n) but symmetric, so col-major view (n, n) is the
 * same data. dsyevd_ overwrites A with eigenvectors COL-MAJOR-ordered;
 * but the matrix is symmetric, so eigenvectors are the same in both
 * orderings (just a transpose). For our purposes we treat the output
 * eigenvector matrix as col-major (n, n), with column m the m-th
 * eigenvector — when we read it back as row-major the rows become the
 * eigenvectors transposed, which is what we want for `Q.T @ … @ Q`.
 */
static int eigh_dsyevd(double *A, int n, double *w)
{
    if (n == 0) return 0;
    const char J = 'V', U = 'L';
    int info = 0;
    /* Workspace query */
    double opt_lwork = 0.0;
    int    opt_liwork = 0;
    int    lwork_q = -1;
    int    liwork_q = -1;
    dsyevd_(&J, &U, &n, A, &n, w, &opt_lwork, &lwork_q,
            &opt_liwork, &liwork_q, &info);
    if (info != 0) return info;
    int lwork = (int)opt_lwork;
    int liwork = opt_liwork;
    double *work = (double *)malloc(sizeof(double) * (size_t)lwork);
    int    *iwork = (int    *)malloc(sizeof(int)    * (size_t)liwork);
    dsyevd_(&J, &U, &n, A, &n, w, work, &lwork,
            iwork, &liwork, &info);
    free(work); free(iwork);
    return info;
}

/* qsort comparator for ascending int64 */
static int cmp_long(const void *a, const void *b)
{
    long la = *(const long *)a;
    long lb = *(const long *)b;
    return (la > lb) - (la < lb);
}

/* --------------------------------------------------------------------
 * Psi4 PartialCholesky orthogonalization (port of orthogonalize_pao_domain)
 *
 *   Inputs:
 *     S_pao_full  (n_pao_total, n_pao_total)
 *     triple_paos (n_pao_ijk,) sorted int64
 *     S_cut       Cholesky cutoff
 *
 *   Output:
 *     X_pao_ijk (n_pao_ijk, n_pao_can) row-major
 *     n_pao_can (int)
 *
 *   Steps (Psi4 BasisSetOrthogonalization::compute_partial_cholesky_orthog):
 *     1. Normalize S → S_norm[i,j] = S[i,j] / sqrt(S[i,i] * S[j,j])
 *     2. Sort columns by INCREASING off-diagonal sum
 *     3. dpstrf with tol=S_cut → pivot list (rank_c entries)
 *     4. Translate pivots back through the sort permutation
 *     5. Sort pivots ascending; build S_sub[pivots, pivots]
 *     6. eigh(S_sub) → keep eigvals > 0 → X_sub
 *     7. Pad X_sub back to (n_pao_ijk, n_keep) at pivot rows, scale by 1/sqrt(diag)
 * -------------------------------------------------------------------- */
static int orthogonalize_pao_psi4(
        const double *S_pao_full, const int n_pao_total,
        const long *triple_paos, const int n_dom,
        const double S_cut,
        double **X_out, int *n_pao_can_out)
{
    *X_out = NULL;
    *n_pao_can_out = 0;
    if (n_dom <= 0) return 0;

    /* S_dom (n_dom, n_dom) */
    double *S_dom = (double *)malloc(sizeof(double) * (size_t)n_dom * (size_t)n_dom);
    gather_submatrix_d(S_pao_full, triple_paos, triple_paos,
                       n_dom, n_dom, n_pao_total, S_dom);

    /* 1. Normalization factor norm[i] = 1/sqrt(diag[i]) */
    double *norm = (double *)malloc(sizeof(double) * (size_t)n_dom);
    for (int i = 0; i < n_dom; i++) {
        const double d = S_dom[(size_t)i * (size_t)n_dom + i];
        norm[i] = (d > 0.0) ? 1.0 / sqrt(d) : 1.0;
    }

    /* S_norm (n_dom, n_dom) = norm[i]*S[i,j]*norm[j] */
    double *S_norm = (double *)malloc(sizeof(double) * (size_t)n_dom * (size_t)n_dom);
    for (int i = 0; i < n_dom; i++) {
        for (int j = 0; j < n_dom; j++) {
            S_norm[(size_t)i * (size_t)n_dom + j]
                = S_dom[(size_t)i * (size_t)n_dom + j] * norm[i] * norm[j];
        }
    }
    free(S_dom);

    /* 2. Off-diagonal sums: od[i] = Σ_j |S_norm[i,j]| - |S_norm[i,i]| */
    double *od = (double *)malloc(sizeof(double) * (size_t)n_dom);
    for (int i = 0; i < n_dom; i++) {
        double s = 0.0;
        for (int j = 0; j < n_dom; j++) {
            s += fabs(S_norm[(size_t)i * (size_t)n_dom + j]);
        }
        od[i] = s - fabs(S_norm[(size_t)i * (size_t)n_dom + i]);
    }
    /* Sort indices by od ascending (stable). */
    int *order = (int *)malloc(sizeof(int) * (size_t)n_dom);
    for (int i = 0; i < n_dom; i++) order[i] = i;
    /* Simple stable insertion sort (n_dom is small, ≲150) */
    for (int i = 1; i < n_dom; i++) {
        int key = order[i];
        double key_v = od[key];
        int j = i - 1;
        while (j >= 0 && od[order[j]] > key_v) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }
    free(od);

    /* S_reord = S_norm[order, order] (n_dom, n_dom) col-major-friendly */
    double *S_reord = (double *)malloc(sizeof(double) * (size_t)n_dom * (size_t)n_dom);
    for (int i = 0; i < n_dom; i++) {
        for (int j = 0; j < n_dom; j++) {
            S_reord[(size_t)i * (size_t)n_dom + j]
                = S_norm[(size_t)order[i] * (size_t)n_dom + order[j]];
        }
    }

    /* 3. dpstrf(lower=1) — overwrites S_reord, returns pivots (1-based) */
    int *piv = (int *)malloc(sizeof(int) * (size_t)n_dom);
    int rank_c = 0;
    int info = 0;
    char Lc = 'L';
    double *work_chol = (double *)malloc(sizeof(double) * (size_t)(2 * n_dom));
    /* dpstrf is col-major; symmetric input → col-major view = transpose, but
     * for symmetric A it doesn't matter. */
    dpstrf_(&Lc, &n_dom, S_reord, &n_dom, piv, &rank_c, &S_cut, work_chol, &info);
    free(work_chol);
    free(S_reord);
    if (info < 0) {
        free(S_norm); free(norm); free(order); free(piv);
        return info;
    }
    if (rank_c == 0) {
        free(S_norm); free(norm); free(order); free(piv);
        return 0;  /* legitimate empty result */
    }

    /* 4. Translate pivots from reordered → original domain index */
    long *pivots_orig = (long *)malloc(sizeof(long) * (size_t)rank_c);
    for (int p = 0; p < rank_c; p++) {
        pivots_orig[p] = (long)order[piv[p] - 1];
    }
    free(order);
    free(piv);
    /* 5. sort ascending */
    qsort(pivots_orig, (size_t)rank_c, sizeof(long), cmp_long);

    /* 6. S_sub[a, b] = S_norm[pivots_orig[a], pivots_orig[b]] */
    double *S_sub = (double *)malloc(sizeof(double) * (size_t)rank_c * (size_t)rank_c);
    for (int a = 0; a < rank_c; a++) {
        for (int b = 0; b < rank_c; b++) {
            S_sub[(size_t)a * (size_t)rank_c + b]
                = S_norm[(size_t)pivots_orig[a] * (size_t)n_dom + pivots_orig[b]];
        }
    }
    free(S_norm);

    double *eigvals = (double *)malloc(sizeof(double) * (size_t)rank_c);
    int eig_info = eigh_dsyevd(S_sub, rank_c, eigvals);
    if (eig_info != 0) {
        free(eigvals); free(S_sub); free(pivots_orig); free(norm);
        return eig_info;
    }
    /* Now S_sub holds eigenvectors col-major; reading row-major it's the
     * transpose. We want X_sub[i, m] = eigvecs[i, m] / sqrt(eigvals[m]),
     * keeping eigvals > 0.
     *
     * In col-major view: eigvec_col[i, m] = element at (i + m*rank_c).
     * In row-major view of same buffer: row[i, m] = element at (i*rank_c + m).
     * These differ. The eigenvector matrix returned by LAPACK is col-major:
     *   eigvec_col[i, m] = m-th eigenvector at position i.
     *
     * We need X_sub[i, m] in row-major. Allocate fresh.
     */
    int n_keep = 0;
    for (int m = 0; m < rank_c; m++) {
        if (eigvals[m] > 0.0) n_keep++;
    }
    if (n_keep == 0) {
        free(eigvals); free(S_sub); free(pivots_orig); free(norm);
        return 0;
    }
    double *X_sub = (double *)malloc(sizeof(double) * (size_t)rank_c * (size_t)n_keep);
    int kept = 0;
    for (int m = 0; m < rank_c; m++) {
        if (eigvals[m] <= 0.0) continue;
        const double scale = 1.0 / sqrt(eigvals[m]);
        for (int i = 0; i < rank_c; i++) {
            /* col-major eigvec_col[i, m] = S_sub_col[i + m*rank_c] = same flat
             * as row-major S_sub[m, i] (since we're storing the buffer row-major
             * but LAPACK wrote it col-major). I.e., the eigenvector for index
             * m is in column m of the LAPACK output, which when read as
             * row-major is the m-th row.
             *
             * Actually: LAPACK col-major dsyevd output A_col[i, m] = i-th comp
             * of m-th eigenvector. In memory flat[i + m*rank_c] = A_col[i,m].
             * When we read same memory as row-major (rank_c, rank_c):
             *   row[r, c] at flat[r*rank_c + c]. Reading flat[r*rank_c + c] as
             *   col-major would map to A_col[c, r] (i = r*rank_c + c → solve for
             *   col-major (i', m') with i' + m'*rank_c = r*rank_c + c gives
             *   i'=c, m'=r). So row[r, c] = A_col[c, r] = c-th component of
             *   r-th eigenvector. Equivalently: row of buffer = eigenvector,
             *   col of buffer = eigenvector position component.
             *
             * So when reading row-major: m-th eigenvector = m-th ROW of buffer.
             *   eigvec[i] (i = position) = S_sub_row[m, i].
             */
            X_sub[(size_t)i * (size_t)n_keep + kept]
                = S_sub[(size_t)m * (size_t)rank_c + i] * scale;
        }
        kept++;
    }
    free(S_sub);
    free(eigvals);

    /* 7. Pad back to (n_dom, n_keep): X_orth[pivots[a], m] = X_sub[a, m] */
    double *X_orth = (double *)calloc(
        (size_t)n_dom * (size_t)n_keep, sizeof(double));
    for (int a = 0; a < rank_c; a++) {
        const long p = pivots_orig[a];
        for (int m = 0; m < n_keep; m++) {
            X_orth[(size_t)p * (size_t)n_keep + m]
                = X_sub[(size_t)a * (size_t)n_keep + m];
        }
    }
    free(X_sub);
    free(pivots_orig);

    /* Unroll normalization: X_orth[i, m] *= norm[i] */
    for (int i = 0; i < n_dom; i++) {
        for (int m = 0; m < n_keep; m++) {
            X_orth[(size_t)i * (size_t)n_keep + m] *= norm[i];
        }
    }
    free(norm);

    *X_out = X_orth;
    *n_pao_can_out = n_keep;
    return 0;
}

/* --------------------------------------------------------------------
 * Body: F_orth + D_ijk (Session 2 logic, internal)
 * -------------------------------------------------------------------- */
static void build_F_and_D(
        const int n_pao_ijk, const int n_pao_can, const int n_pao_total,
        const long *triple_paos, const double *X_pao_ijk,
        const double *F_pao_full, const double *S_pao_full,
        const int n_keys, const int *pair_paos_n,
        const long *pair_paos_off, const long *pair_paos_flat,
        const int *n_pno_arr, const long *X_pno_off,
        const double *X_pno_flat,
        const long *T2_off, const double *T2_flat,
        const int *same_lmo,
        double *F_orth, double *D_ijk)
{
    const char N = 'N', T = 'T';
    const double one = 1.0, zero = 0.0;
    int int_npc = n_pao_can;
    int int_npi = n_pao_ijk;

    /* F_dom (n_pao_ijk, n_pao_ijk) */
    double *F_dom = (double *)malloc(
        sizeof(double) * (size_t)n_pao_ijk * (size_t)n_pao_ijk);
    gather_submatrix_d(F_pao_full, triple_paos, triple_paos,
                       n_pao_ijk, n_pao_ijk, n_pao_total, F_dom);

    double *tmp = (double *)malloc(
        sizeof(double) * (size_t)n_pao_ijk * (size_t)n_pao_can);
    /* tmp = F_dom @ X */
    dgemm_(&N, &N, &int_npc, &int_npi, &int_npi,
           &one, X_pao_ijk, &int_npc, F_dom, &int_npi,
           &zero, tmp, &int_npc);
    /* F_orth = X.T @ tmp */
    dgemm_(&N, &T, &int_npc, &int_npc, &int_npi,
           &one, tmp, &int_npc, X_pao_ijk, &int_npc,
           &zero, F_orth, &int_npc);
    free(F_dom); free(tmp);

    memset(D_ijk, 0, sizeof(double) * (size_t)n_pao_can * (size_t)n_pao_can);

    int max_npp = 0, max_npno = 0;
    for (int kk = 0; kk < n_keys; kk++) {
        if (pair_paos_n[kk] > max_npp)  max_npp  = pair_paos_n[kk];
        if (n_pno_arr[kk]   > max_npno) max_npno = n_pno_arr[kk];
    }
    if (max_npp == 0 || max_npno == 0) return;

    double *S_pair_triple = (double *)malloc(
        sizeof(double) * (size_t)max_npp  * (size_t)n_pao_ijk);
    double *S_pair_X = (double *)malloc(
        sizeof(double) * (size_t)max_npp  * (size_t)n_pao_can);
    double *S_proj = (double *)malloc(
        sizeof(double) * (size_t)max_npno * (size_t)n_pao_can);
    double *Tt = (double *)malloc(
        sizeof(double) * (size_t)max_npno * (size_t)max_npno);
    double *DT = (double *)malloc(
        sizeof(double) * (size_t)max_npno * (size_t)max_npno);
    double *D_pair = (double *)malloc(
        sizeof(double) * (size_t)max_npno * (size_t)max_npno);
    double *DS = (double *)malloc(
        sizeof(double) * (size_t)max_npno * (size_t)n_pao_can);

    for (int kk = 0; kk < n_keys; kk++) {
        const int npp = pair_paos_n[kk];
        const int npno = n_pno_arr[kk];
        if (npp == 0 || npno == 0) continue;
        const long *pp = pair_paos_flat + pair_paos_off[kk];
        const double *Xpno = X_pno_flat + X_pno_off[kk];
        const double *T2 = T2_flat + T2_off[kk];

        gather_submatrix_d(S_pao_full, pp, triple_paos,
                           npp, n_pao_ijk, n_pao_total, S_pair_triple);

        int int_npp  = npp;
        int int_npno = npno;
        dgemm_(&N, &N, &int_npc, &int_npp, &int_npi,
               &one, X_pao_ijk, &int_npc, S_pair_triple, &int_npi,
               &zero, S_pair_X, &int_npc);
        dgemm_(&N, &T, &int_npc, &int_npno, &int_npp,
               &one, S_pair_X, &int_npc, Xpno, &int_npno,
               &zero, S_proj, &int_npc);

        for (int p = 0; p < npno; p++) {
            for (int q = 0; q < npno; q++) {
                Tt[(size_t)p * (size_t)npno + q]
                    = 2.0 * T2[(size_t)p * (size_t)npno + q]
                    -       T2[(size_t)q * (size_t)npno + p];
            }
        }
        dgemm_(&T, &N, &int_npno, &int_npno, &int_npno,
               &one, T2, &int_npno, Tt, &int_npno,
               &zero, DT, &int_npno);
        memcpy(D_pair, DT, sizeof(double) * (size_t)npno * (size_t)npno);
        dgemm_(&N, &T, &int_npno, &int_npno, &int_npno,
               &one, T2, &int_npno, Tt, &int_npno,
               &one, D_pair, &int_npno);

        if (same_lmo[kk]) {
            const size_t nsq = (size_t)npno * (size_t)npno;
            for (size_t ii = 0; ii < nsq; ii++) D_pair[ii] *= 0.5;
        }
        dgemm_(&N, &N, &int_npc, &int_npno, &int_npno,
               &one, S_proj, &int_npc, D_pair, &int_npno,
               &zero, DS, &int_npc);
        dgemm_(&N, &T, &int_npc, &int_npc, &int_npno,
               &one, DS, &int_npc, S_proj, &int_npc,
               &one, D_ijk, &int_npc);
    }

    const size_t nsq = (size_t)n_pao_can * (size_t)n_pao_can;
    const double inv3 = 1.0 / 3.0;
    for (size_t ii = 0; ii < nsq; ii++) D_ijk[ii] *= inv3;

    free(S_pair_triple); free(S_pair_X); free(S_proj);
    free(Tt); free(DT); free(D_pair); free(DS);
}

/* --------------------------------------------------------------------
 * Single-triple full TNO transform.
 *
 *   Returns 0 on success.  Outputs allocated and written by this fn:
 *     X_tno_ijk_out  (n_pao_ijk, n_tno) row-major  — caller must free
 *     eps_tno_out    (n_tno,)                       — caller must free
 *     X_pao_ijk_out  (n_pao_ijk, n_pao_can) row-maj — caller must free
 *     n_tno_out, n_pao_can_out
 * -------------------------------------------------------------------- */
int DLPNObuild_triple_tno_full(
        const int n_pao_ijk,
        const int n_pao_total,
        const long *triple_paos,           /* (n_pao_ijk,) */
        const double *F_pao_full,          /* (n_pao_total, n_pao_total) */
        const double *S_pao_full,          /* (n_pao_total, n_pao_total) */
        const int n_keys,
        const int *pair_paos_n,
        const long *pair_paos_off,
        const long *pair_paos_flat,
        const int *n_pno_arr,
        const long *X_pno_off,
        const double *X_pno_flat,
        const long *T2_off,
        const double *T2_flat,
        const int *same_lmo,
        const double T_CutTNO,
        const double S_cut_domain,
        double **X_tno_ijk_out,
        double **eps_tno_out,
        double **X_pao_ijk_out,
        int *n_tno_out,
        int *n_pao_can_out)
{
    *X_tno_ijk_out = NULL;
    *eps_tno_out   = NULL;
    *X_pao_ijk_out = NULL;
    *n_tno_out     = 0;
    *n_pao_can_out = 0;

    if (n_pao_ijk <= 0) return 0;

    /* Step 1: orthogonalize_pao_domain (Psi4 PartialCholesky). */
    double *X_pao_ijk = NULL;
    int n_pao_can = 0;
    int rc = orthogonalize_pao_psi4(
        S_pao_full, n_pao_total, triple_paos, n_pao_ijk,
        S_cut_domain, &X_pao_ijk, &n_pao_can);
    if (rc != 0) return rc;
    if (n_pao_can == 0) {
        if (X_pao_ijk) free(X_pao_ijk);
        return 0;
    }

    /* Step 2-3: F_orth_ijk and D_ijk via Session 2 logic */
    double *F_orth = (double *)malloc(
        sizeof(double) * (size_t)n_pao_can * (size_t)n_pao_can);
    double *D_ijk = (double *)malloc(
        sizeof(double) * (size_t)n_pao_can * (size_t)n_pao_can);
    build_F_and_D(n_pao_ijk, n_pao_can, n_pao_total,
                  triple_paos, X_pao_ijk,
                  F_pao_full, S_pao_full,
                  n_keys, pair_paos_n, pair_paos_off, pair_paos_flat,
                  n_pno_arr, X_pno_off, X_pno_flat,
                  T2_off, T2_flat, same_lmo,
                  F_orth, D_ijk);

    /* Step 4: eigh(D_ijk).  dsyevd returns ascending; we need descending
     * by |eig| to truncate at T_CutTNO.
     */
    double *D_evals = (double *)malloc(sizeof(double) * (size_t)n_pao_can);
    int eig_rc = eigh_dsyevd(D_ijk, n_pao_can, D_evals);
    if (eig_rc != 0) {
        free(F_orth); free(D_ijk); free(D_evals); free(X_pao_ijk);
        return eig_rc;
    }
    /* D_ijk now holds eigenvectors col-major. We want |eigval| descending,
     * keep those above T_CutTNO. Find selected indices. */
    int n_tno = 0;
    int *keep_idx = (int *)malloc(sizeof(int) * (size_t)n_pao_can);
    /* Sort indices by descending |eigval|. Small n_pao_can (≤150). */
    for (int m = 0; m < n_pao_can; m++) keep_idx[m] = m;
    /* Insertion sort by |D_evals[idx]| descending (stable). */
    for (int i = 1; i < n_pao_can; i++) {
        int key = keep_idx[i];
        double key_v = fabs(D_evals[key]);
        int j = i - 1;
        while (j >= 0 && fabs(D_evals[keep_idx[j]]) < key_v) {
            keep_idx[j + 1] = keep_idx[j]; j--;
        }
        keep_idx[j + 1] = key;
    }
    for (int m = 0; m < n_pao_can; m++) {
        if (fabs(D_evals[keep_idx[m]]) >= T_CutTNO) n_tno++;
        else break;
    }
    if (n_tno == 0) n_tno = 1;  /* match Python fallback */

    /* X_tno_initial (n_pao_can, n_tno) row-major:
     *   X_tno_initial[i, m] = D_evec[i, keep_idx[m]]
     * D_ijk holds col-major eigenvectors, i.e. row-major D_evec_T[m', i] =
     * D_ijk_row[m', i] = D_col[i, m']. So D_evec[i, m] = D_col[i, m] =
     * D_ijk_row[m, i]. */
    double *X_tno_initial = (double *)malloc(
        sizeof(double) * (size_t)n_pao_can * (size_t)n_tno);
    for (int i = 0; i < n_pao_can; i++) {
        for (int m = 0; m < n_tno; m++) {
            const int evec_col = keep_idx[m];
            X_tno_initial[(size_t)i * (size_t)n_tno + m]
                = D_ijk[(size_t)evec_col * (size_t)n_pao_can + i];
        }
    }
    free(D_evals); free(keep_idx); free(D_ijk);

    /* Step 5-6: F_in_tno = X_tno_initial.T @ F_orth @ X_tno_initial,
     *           eigh → eps_tno_sc, tno_canon */
    int int_npc = n_pao_can, int_ntno = n_tno;
    const char N = 'N', T = 'T';
    const double one = 1.0, zero = 0.0;
    double *Ftmp = (double *)malloc(
        sizeof(double) * (size_t)n_pao_can * (size_t)n_tno);
    /* Ftmp (n_pao_can, n_tno) = F_orth @ X_tno_initial */
    dgemm_(&N, &N, &int_ntno, &int_npc, &int_npc,
           &one, X_tno_initial, &int_ntno, F_orth, &int_npc,
           &zero, Ftmp, &int_ntno);
    free(F_orth);
    /* F_in_tno (n_tno, n_tno) = X_tno_initial.T @ Ftmp */
    double *F_in_tno = (double *)malloc(
        sizeof(double) * (size_t)n_tno * (size_t)n_tno);
    dgemm_(&N, &T, &int_ntno, &int_ntno, &int_npc,
           &one, Ftmp, &int_ntno, X_tno_initial, &int_ntno,
           &zero, F_in_tno, &int_ntno);
    free(Ftmp);

    double *eps_tno = (double *)malloc(sizeof(double) * (size_t)n_tno);
    int eig2_rc = eigh_dsyevd(F_in_tno, n_tno, eps_tno);
    if (eig2_rc != 0) {
        free(F_in_tno); free(eps_tno); free(X_tno_initial); free(X_pao_ijk);
        return eig2_rc;
    }
    /* tno_canon row-major (n_tno, m) where row i = i-th component of m-th
     * eigenvector. From col-major LAPACK output: tno_canon[i, m] =
     * F_in_tno_col[i, m] = F_in_tno_row[m, i]. So in row-major view of
     * F_in_tno buffer: row m = m-th eigenvector. Below we need
     * X_tno_canonical = X_tno_initial @ tno_canon, row-major (n_pao_can, n_tno).
     * Let's compute X_tno_canonical[i, m] = Σ_l X_tno_initial[i, l] *
     * tno_canon[l, m] = Σ_l X_tno_initial[i, l] * F_in_tno_row[m, l].
     *
     * Equivalent dgemm: row-major C(n_pao_can, n_tno) = A(n_pao_can, n_tno) @
     *   B^T(n_tno, n_tno), where A=X_tno_initial, B=F_in_tno.
     *   Col-major: C_col(n_tno, n_pao_can) = B^T_col @ A_col = (B_col)^T... messy.
     *
     * Simpler: use the same row-major matmul rule as before:
     *   row-major C = A @ B  --  dgemm('N','N', N, M, K,
     *                                  1, B (LDB=N), A (LDA=K), 0, C (LDC=N))
     * With A = X_tno_initial (n_pao_can, n_tno) and B = tno_canon (n_tno, n_tno).
     * tno_canon row-major holds eigenvectors as rows; we need it as cols.
     * That is, we need B = tno_canon^T (so tno_canon_row^T[l, m] = tno_canon_row[m, l]).
     * So row-major (X_tno_initial) @ (tno_canon^T) = X_tno_canonical.
     *
     * In dgemm terms:
     *   row-major C = A @ B^T where B = tno_canon_row.
     *   Col-major dgemm: dgemm('T','N', N, M, K, 1, B (LDB=K), A (LDA=K),
     *                          0, C (LDC=N))
     *   Here M = n_pao_can, N = n_tno, K = n_tno.
     */
    double *X_tno_canonical = (double *)malloc(
        sizeof(double) * (size_t)n_pao_can * (size_t)n_tno);
    dgemm_(&T, &N, &int_ntno, &int_npc, &int_ntno,
           &one, F_in_tno, &int_ntno, X_tno_initial, &int_ntno,
           &zero, X_tno_canonical, &int_ntno);
    free(F_in_tno); free(X_tno_initial);

    /* Step 7: X_tno_ijk = X_pao_ijk @ X_tno_canonical
     *   row-major (n_pao_ijk, n_tno) = (n_pao_ijk, n_pao_can) @
     *                                  (n_pao_can, n_tno)
     *   dgemm('N','N', n_tno, n_pao_ijk, n_pao_can,
     *         1, X_tno_canonical, n_tno, X_pao_ijk, n_pao_can,
     *         0, X_tno_ijk, n_tno)
     */
    double *X_tno_ijk = (double *)malloc(
        sizeof(double) * (size_t)n_pao_ijk * (size_t)n_tno);
    int int_npijk = n_pao_ijk;
    dgemm_(&N, &N, &int_ntno, &int_npijk, &int_npc,
           &one, X_tno_canonical, &int_ntno, X_pao_ijk, &int_npc,
           &zero, X_tno_ijk, &int_ntno);
    free(X_tno_canonical);

    *X_tno_ijk_out = X_tno_ijk;
    *eps_tno_out   = eps_tno;
    *X_pao_ijk_out = X_pao_ijk;
    *n_tno_out     = n_tno;
    *n_pao_can_out = n_pao_can;
    return 0;
}

/* --------------------------------------------------------------------
 * Python-callable wrapper: writes directly into caller-provided buffers
 * up to (n_pao_ijk × max_n_tno) and (max_n_tno).  Returns n_tno_out via
 * pointer; X_pao_ijk written to `X_pao_out` if non-NULL.  Caller passes
 * upper-bound max_n_tno = n_pao_ijk so output buffers are pre-sized.
 * -------------------------------------------------------------------- */
void DLPNObuild_triple_tno_full_py(
        const int n_pao_ijk,
        const int n_pao_total,
        const long *triple_paos,
        const double *F_pao_full,
        const double *S_pao_full,
        const int n_keys,
        const int *pair_paos_n,
        const long *pair_paos_off,
        const long *pair_paos_flat,
        const int *n_pno_arr,
        const long *X_pno_off,
        const double *X_pno_flat,
        const long *T2_off,
        const double *T2_flat,
        const int *same_lmo,
        const double T_CutTNO,
        const double S_cut_domain,
        double *X_tno_ijk_out,    /* (n_pao_ijk, n_pao_ijk) max-sized */
        double *eps_tno_out,      /* (n_pao_ijk,) max-sized */
        double *X_pao_ijk_out,    /* (n_pao_ijk, n_pao_ijk) max-sized */
        int *n_tno_out_p,
        int *n_pao_can_out_p,
        int *info_out)
{
    double *X_tno = NULL, *eps = NULL, *X_pao = NULL;
    int n_tno = 0, n_pao_can = 0;
    int rc = DLPNObuild_triple_tno_full(
        n_pao_ijk, n_pao_total, triple_paos, F_pao_full, S_pao_full,
        n_keys, pair_paos_n, pair_paos_off, pair_paos_flat,
        n_pno_arr, X_pno_off, X_pno_flat, T2_off, T2_flat, same_lmo,
        T_CutTNO, S_cut_domain,
        &X_tno, &eps, &X_pao, &n_tno, &n_pao_can);
    *info_out = rc;
    *n_tno_out_p = n_tno;
    *n_pao_can_out_p = n_pao_can;
    if (rc != 0 || n_tno == 0) {
        if (X_tno) free(X_tno);
        if (eps)   free(eps);
        if (X_pao) free(X_pao);
        return;
    }
    /* Output buffers are sized (n_pao_ijk, n_pao_ijk) row-major (max bound).
     * Internal X_tno is (n_pao_ijk, n_tno) and X_pao is (n_pao_ijk, n_pao_can);
     * we must write each row r into position [r, 0..k-1] of the full buffer
     * (leading dim = n_pao_ijk), NOT contiguously. */
    for (int r = 0; r < n_pao_ijk; r++) {
        const double *src_tno = X_tno + (size_t)r * (size_t)n_tno;
        double *dst_tno = X_tno_ijk_out + (size_t)r * (size_t)n_pao_ijk;
        memcpy(dst_tno, src_tno, sizeof(double) * (size_t)n_tno);
        if (X_pao_ijk_out) {
            const double *src_pao = X_pao + (size_t)r * (size_t)n_pao_can;
            double *dst_pao = X_pao_ijk_out + (size_t)r * (size_t)n_pao_ijk;
            memcpy(dst_pao, src_pao, sizeof(double) * (size_t)n_pao_can);
        }
    }
    memcpy(eps_tno_out, eps, sizeof(double) * (size_t)n_tno);
    free(X_tno); free(eps); free(X_pao);
}
