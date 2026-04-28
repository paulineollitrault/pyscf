/* DLPNO-(T): per-triple TNO accumulation kernel.
 *
 * Replaces the heavy-numerical body of `_triple_pno_union_psi4` in
 * pyscf/cc/dlpno_tccsd/lccsd_t.py.
 *
 * Per triple (i,j,k), given:
 *   triple_paos:   (n_pao_ijk,) raw-PAO indices in the triple union
 *   X_pao_ijk:     (n_pao_ijk, n_pao_can) raw → orth-PAO transform
 *                   (computed in Python via orthogonalize_pao_domain)
 *   F_pao_full:    (n_pao_total, n_pao_total) full-PAO Fock
 *   S_pao_full:    (n_pao_total, n_pao_total) full-PAO overlap
 *   for each of 3 pair keys (ij, jk, ik):
 *     pair_paos:   (n_pao_pair,) raw-PAO indices for this pair
 *     X_pno_pair:  (n_pao_pair, n_pno_pair) raw-PAO → PNO transform
 *     T2_pair:     (n_pno_pair, n_pno_pair) PNO-basis amplitudes
 *     same_lmo:    1 if i==j (diagonal pair), 0 otherwise
 *
 * Computes:
 *   F_orth_ijk = X_pao_ijk.T @ F_pao[triple_paos, triple_paos] @ X_pao_ijk
 *   For each key:
 *     S_proj = X_pno.T @ S_pao[pair_paos, triple_paos] @ X_pao_ijk
 *               (n_pno_pair, n_pao_can)
 *     Tt     = 2*T2 - T2.T
 *     D_pair = Tt @ T2.T + Tt.T @ T2          (n_pno_pair, n_pno_pair)
 *     if same_lmo: D_pair *= 0.5
 *     D_ijk += S_proj.T @ D_pair @ S_proj     (n_pao_can, n_pao_can)
 *   D_ijk /= 3.0
 *
 * Returns (out arguments):
 *   F_orth_ijk_out:  (n_pao_can, n_pao_can)
 *   D_ijk_out:       (n_pao_can, n_pao_can)
 *
 * Outer parallel: NONE (driver fans triples across pool).
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "vhf/fblas.h"

static void gather_submatrix(
        const double *src,                 /* (n_full, n_full) row-major */
        const long   *rows,
        const long   *cols,
        const int     n_rows,
        const int     n_cols,
        const int     n_full,
        double       *dst)                 /* (n_rows, n_cols) row-major */
{
    for (int r = 0; r < n_rows; r++) {
        const double *src_row = src + (size_t)rows[r] * (size_t)n_full;
        double *dst_row = dst + (size_t)r * (size_t)n_cols;
        for (int c = 0; c < n_cols; c++) {
            dst_row[c] = src_row[cols[c]];
        }
    }
}

void DLPNObuild_triple_tno_body(
        const int     n_pao_ijk,
        const int     n_pao_can,
        const int     n_pao_total,
        const long   *triple_paos,         /* (n_pao_ijk,) */
        const double *X_pao_ijk,           /* (n_pao_ijk, n_pao_can) row-major */
        const double *F_pao_full,          /* (n_pao_total, n_pao_total) */
        const double *S_pao_full,          /* (n_pao_total, n_pao_total) */
        const int     n_keys,              /* always 3 in practice */
        const int    *pair_paos_n,         /* (n_keys,) */
        const long   *pair_paos_off,       /* (n_keys+1,) into pair_paos_flat */
        const long   *pair_paos_flat,      /* concat of per-key pair_paos */
        const int    *n_pno_arr,           /* (n_keys,) */
        const long   *X_pno_off,           /* (n_keys+1,) into X_pno_flat */
        const double *X_pno_flat,          /* concat of per-key X_pno (n_pao_pair × n_pno) */
        const long   *T2_off,              /* (n_keys+1,) into T2_flat */
        const double *T2_flat,             /* concat of per-key T2 (n_pno × n_pno) */
        const int    *same_lmo,            /* (n_keys,) 1 for diag pair */
        double       *F_orth_ijk_out,      /* (n_pao_can, n_pao_can) */
        double       *D_ijk_out)           /* (n_pao_can, n_pao_can) */
{
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;

    /* ---------------- F_orth_ijk = X.T @ F_dom @ X ---------------- */
    /* F_dom: (n_pao_ijk, n_pao_ijk) gathered from F_pao_full. */
    double *F_dom = (double *)malloc(
        sizeof(double) * (size_t)n_pao_ijk * (size_t)n_pao_ijk);
    gather_submatrix(F_pao_full, triple_paos, triple_paos,
                     n_pao_ijk, n_pao_ijk, n_pao_total, F_dom);

    /* tmp1 (n_pao_ijk, n_pao_can) = F_dom @ X_pao_ijk
     * Row-major: tmp1[u, a] = sum_v F_dom[u, v] * X[v, a]
     * Col-major dgemm: dgemm('N', 'N', n_pao_can, n_pao_ijk, n_pao_ijk,
     *                       1, X, n_pao_can, F_dom, n_pao_ijk,
     *                       0, tmp1, n_pao_can)
     */
    double *tmp1 = (double *)malloc(
        sizeof(double) * (size_t)n_pao_ijk * (size_t)n_pao_can);
    int int_npc = n_pao_can;
    int int_npi = n_pao_ijk;
    dgemm_(&N_flag, &N_flag,
           &int_npc, &int_npi, &int_npi,
           &one, X_pao_ijk,    &int_npc,
           F_dom,              &int_npi,
           &zero, tmp1,        &int_npc);

    /* F_orth (n_pao_can, n_pao_can) = X.T @ tmp1
     * Row-major: F_orth[a, b] = sum_u X[u, a] * tmp1[u, b]
     * Col-major dgemm: dgemm('N', 'T', n_pao_can, n_pao_can, n_pao_ijk,
     *                       1, tmp1, n_pao_can, X, n_pao_can,
     *                       0, F_orth, n_pao_can)
     * Reading row-major F_orth as (n_pao_can, n_pao_can) gives
     *   F_orth_row[a, b] = X.T @ tmp1[a, b] = sum_u X[u, a] * tmp1[u, b]   ✓
     */
    dgemm_(&N_flag, &T_flag,
           &int_npc, &int_npc, &int_npi,
           &one, tmp1,                 &int_npc,
           X_pao_ijk,                  &int_npc,
           &zero, F_orth_ijk_out,      &int_npc);

    free(F_dom);
    free(tmp1);

    /* ---------------- D_ijk per-key accumulation ---------------- */
    memset(D_ijk_out, 0, sizeof(double) * (size_t)n_pao_can * (size_t)n_pao_can);

    /* Persistent scratch for the largest pair we'll see (worst-case bound). */
    int max_npp = 0, max_npno = 0;
    for (int k = 0; k < n_keys; k++) {
        if (pair_paos_n[k] > max_npp)  max_npp  = pair_paos_n[k];
        if (n_pno_arr[k]   > max_npno) max_npno = n_pno_arr[k];
    }
    double *S_pair_triple = NULL;     /* (n_pao_pair, n_pao_ijk) */
    double *S_pair_X      = NULL;     /* (n_pao_pair, n_pao_can) */
    double *S_proj        = NULL;     /* (n_pno, n_pao_can) */
    double *Tt            = NULL;     /* (n_pno, n_pno) */
    double *DT            = NULL;     /* (n_pno, n_pno) work for D_pair */
    double *D_pair        = NULL;     /* (n_pno, n_pno) */
    double *DS            = NULL;     /* (n_pno, n_pao_can) */
    if (max_npp > 0 && max_npno > 0) {
        S_pair_triple = (double *)malloc(sizeof(double) * (size_t)max_npp  * (size_t)n_pao_ijk);
        S_pair_X      = (double *)malloc(sizeof(double) * (size_t)max_npp  * (size_t)n_pao_can);
        S_proj        = (double *)malloc(sizeof(double) * (size_t)max_npno * (size_t)n_pao_can);
        Tt            = (double *)malloc(sizeof(double) * (size_t)max_npno * (size_t)max_npno);
        DT            = (double *)malloc(sizeof(double) * (size_t)max_npno * (size_t)max_npno);
        D_pair        = (double *)malloc(sizeof(double) * (size_t)max_npno * (size_t)max_npno);
        DS            = (double *)malloc(sizeof(double) * (size_t)max_npno * (size_t)n_pao_can);
    }

    for (int k = 0; k < n_keys; k++) {
        const int npp  = pair_paos_n[k];
        const int npno = n_pno_arr[k];
        if (npp == 0 || npno == 0) continue;

        const long *pp = pair_paos_flat + pair_paos_off[k];
        const double *X_pno = X_pno_flat + X_pno_off[k];   /* (npp, npno) */
        const double *T2 = T2_flat + T2_off[k];            /* (npno, npno) */

        /* S_pair_triple (npp, n_pao_ijk) = S_pao[pp, triple_paos] */
        gather_submatrix(S_pao_full, pp, triple_paos,
                         npp, n_pao_ijk, n_pao_total, S_pair_triple);

        /* S_pair_X (npp, n_pao_can) = S_pair_triple @ X_pao_ijk
         * Row-major: S_pair_X[u, a] = sum_v S_pair_triple[u, v] * X[v, a]
         * Col-major dgemm: dgemm('N','N', n_pao_can, npp, n_pao_ijk,
         *                       1, X, n_pao_can, S_pair_triple, n_pao_ijk,
         *                       0, S_pair_X, n_pao_can)
         */
        int int_npp = npp;
        dgemm_(&N_flag, &N_flag,
               &int_npc, &int_npp, &int_npi,
               &one, X_pao_ijk,         &int_npc,
               S_pair_triple,           &int_npi,
               &zero, S_pair_X,         &int_npc);

        /* S_proj (npno, n_pao_can) = X_pno.T @ S_pair_X
         * Row-major: S_proj[p, a] = sum_u X_pno[u, p] * S_pair_X[u, a]
         * Col-major dgemm: dgemm('N','T', n_pao_can, npno, npp,
         *                       1, S_pair_X, n_pao_can, X_pno, npno,
         *                       0, S_proj, n_pao_can)
         */
        int int_npno = npno;
        dgemm_(&N_flag, &T_flag,
               &int_npc, &int_npno, &int_npp,
               &one, S_pair_X,          &int_npc,
               X_pno,                   &int_npno,
               &zero, S_proj,           &int_npc);

        /* Tt = 2*T2 - T2.T  (npno, npno) */
        for (int p = 0; p < npno; p++) {
            for (int q = 0; q < npno; q++) {
                Tt[(size_t)p * (size_t)npno + q]
                    = 2.0 * T2[(size_t)p * (size_t)npno + q]
                    -       T2[(size_t)q * (size_t)npno + p];
            }
        }

        /* D_pair = Tt @ T2.T + Tt.T @ T2
         *   Term1[p,q] = sum_r Tt[p,r] * T2[q,r]
         *   Term2[p,q] = sum_r Tt[r,p] * T2[r,q]
         */
        /* Term1 → DT first via row-major: DT (npno, npno) = Tt @ T2.T
         *   col-major: dgemm('T','N', npno, npno, npno, 1, T2, npno, Tt, npno, 0, DT, npno)
         *   -> DT_col[q, p] = sum_r T2_col[r, q] * Tt_col[p, r]
         *                    = sum_r T2[q, r] * Tt[r, p] = ?
         * Hmm — let me carefully recompute via row-major.
         *
         * Row-major: DT[p, q] = sum_r Tt_row[p, r] * T2_row[q, r]
         *   col-major view: DT_col[q, p] = DT_row[p, q].
         *   So DT_col = ?
         *   sum_r Tt_row[p, r] * T2_row[q, r]
         *   = sum_r Tt_col[r, p] * T2_col[r, q]
         *   So DT_col[q, p] = sum_r T2_col[r, q] * Tt_col[r, p]
         *   This is (T2_col)^T @ (Tt_col)? No.
         *     (A @ B)_col[i, j] = sum_k A_col[i, k] * B_col[k, j]
         *   We want sum_r T2_col[r, q] * Tt_col[r, p] — that's
         *     sum_r (T2_col^T)[q, r] * Tt_col[r, p] = (T2_col^T @ Tt_col)[q, p]
         *   So DT_col = T2_col^T @ Tt_col.
         *   dgemm('T','N', npno, npno, npno,
         *         1, T2, npno, Tt, npno, 0, DT, npno).
         *
         * That gives DT_col[q, p] = (T2^T @ Tt)_col[q, p].
         * Row-major DT[p, q] = DT_col[q, p] = sum_r T2_col[r, q] * Tt_col[r, p]
         *                                   = sum_r T2_row[q, r] * Tt_row[p, r]
         *                                   = sum_r Tt_row[p, r] * T2_row[q, r]   ✓
         */
        dgemm_(&T_flag, &N_flag,
               &int_npno, &int_npno, &int_npno,
               &one, T2,        &int_npno,
               Tt,              &int_npno,
               &zero, DT,       &int_npno);

        /* Now D_pair = DT + (DT)^T  (since Term2[p,q] = sum_r Tt[r,p] * T2[r,q]
         * and DT[p, q] = sum_r Tt[p, r] * T2[q, r] gives DT_T[p, q] = DT[q, p]
         * = sum_r Tt[q, r] * T2[p, r] — that's NOT Term2.
         *
         * Term2[p,q] = sum_r Tt[r, p] * T2[r, q].
         * Let DT2[p, q] = sum_r Tt_T[p, r] * T2[r, q] = sum_r Tt[r, p] * T2[r, q] ✓
         * That's Tt^T @ T2 (row-major). So D_pair = DT + Tt^T @ T2.
         *
         * Compute D_pair = DT first (overwrites), then GEMM with beta=1 to add Tt^T @ T2.
         */
        memcpy(D_pair, DT, sizeof(double) * (size_t)npno * (size_t)npno);
        /* Tt^T @ T2 (row-major) — both row-major, result row-major.
         * Row-major: out[p, q] = sum_r Tt[r, p] * T2[r, q]
         * Col-major: out_col[q, p] = sum_r Tt_col[p, r] * T2_col[q, r]
         *                          = sum_r Tt_col[p, r] * (T2_col^T)[r, q]
         *          = (Tt_col @ T2_col^T)[p, q]?  Hmm leading dim.
         *
         * Think simpler: row-major Tt^T @ T2 can be expressed as col-major
         * with TRANSA='N', TRANSB='T' on the col-major Tt and T2... this is
         * getting confusing. Let me just define op rules carefully:
         *
         * For row-major C(M, N) = A(M, K) @ B(K, N):
         *   Use col-major dgemm('N','N', N, M, K,
         *                        1, B (LDB=N), A (LDA=K), 0, C (LDC=N))
         *
         * Here we want C = Tt^T @ T2, with A = Tt^T (M, K) = (npno, npno),
         * B = T2 (K, N) = (npno, npno), C = (M, N) = (npno, npno).
         * Tt^T row-major = transpose of Tt — but we don't physically transpose;
         * instead: A (M, K) row-major flat[i, j] = Tt^T[i, j] = Tt[j, i].
         *
         * Col-major dgemm: dgemm('N','N', N=npno, M=npno, K=npno,
         *                        beta=1 to add, B=T2 (LDB=npno),
         *                        A=Tt_T (LDA=npno), C=D_pair (LDC=npno))
         * But A = Tt^T row-major has flat[i,j] = Tt[j,i]. Col-major view of
         * that buffer = (npno, npno) with col-major flat[i + j*npno] = ?
         * Original flat[i*npno + j] = Tt[j, i], same memory address.
         *
         * Easier: instead of constructing Tt^T explicitly, use TRANSA flag on
         * the input Tt directly. We want C[p, q] = sum_r Tt[r, p] * T2[r, q].
         * That's Tt^T @ T2 for row-major C. Express via col-major:
         *
         *   Row-major C[p, q] = col-major C_col[q, p].
         *   sum_r Tt_row[r, p] * T2_row[r, q]
         *     = sum_r Tt_col[p, r] * T2_col[q, r]
         *     = sum_r (Tt_col)[p, r] * (T2_col^T)[r, q]
         *     = (Tt_col @ T2_col^T)[p, q]
         *   = col-major (Tt_col @ T2_col^T)[p, q] — but C_col[q, p] needs to equal this.
         *   So C_col[q, p] = (Tt_col @ T2_col^T)[p, q] = (T2_col @ Tt_col^T)^T_col[q, p]
         *                  = (T2_col @ Tt_col^T)_col[q, p].   Hmm nope.
         *
         * Argh. Let me just use a different decomposition: D_pair = DT + (term2),
         * where term2[p, q] = sum_r Tt[r, p] * T2[r, q] = (Tt^T @ T2)[p, q].
         * Recall row-major (A @ B) = col-major (B @ A) reading as col-major.
         * So row-major Tt^T @ T2 ⇔ col-major T2 @ Tt^T (no conjugate, same data).
         *
         * Recall my col-major-GEMM rule for row-major matmul:
         *   row-major C = A @ B, both row-major:
         *     dgemm('N','N', N, M, K, ..., B (LDB=N), A (LDA=K), C (LDC=N))
         *
         * For row-major C = (Tt^T) @ T2:
         *   A_row = Tt^T  (M,K) = (npno, npno)
         *   B_row = T2    (K,N) = (npno, npno)
         *   But A_row = Tt^T is the conjugate-transpose of Tt_row. We can pass
         *   Tt_row to dgemm and tell it "transpose A":
         *     dgemm('N', TRANSA=means transpose A_col?)
         *
         * Yet another approach: use the existing rule but with TRANSA='T' on Tt.
         *
         * Original rule I worked out for term1 (DT row-major = Tt @ T2.T):
         *   dgemm('T','N', npno, npno, npno, 1, T2, npno, Tt, npno, 0, DT, npno)
         *
         * Let me verify: TRANSA='T' on T2_col (npno, npno) gives op(A)_col[i, j]
         * = T2_col[j, i] = T2_row[i, j] (since T2 is square).
         * TRANSB='N' on Tt_col gives op(B)_col[k, l] = Tt_col[k, l] = Tt_row[l, k].
         * dgemm computes C_col[i, j] = sum_k op(A)_col[i, k] * op(B)_col[k, j]
         *                            = sum_k T2_row[i, k] * Tt_row[j, k].
         * Row-major C[p, q] = C_col[q, p] = sum_k T2_row[q, k] * Tt_row[p, k].
         * That's sum_r T2[q, r] * Tt[p, r] = sum_r Tt[p, r] * T2[q, r] = (Tt @ T2^T)[p, q] ✓
         *
         * For term2 row-major = Tt^T @ T2:
         *   term2[p, q] = sum_r Tt[r, p] * T2[r, q]
         * I'll use TRANSA='N', TRANSB='T'? let's check:
         *   dgemm('N','T', npno, npno, npno, 1, T2, npno, Tt, npno, beta=1, D_pair, npno)
         *   op(A)='N' on T2_col -> T2_col[i, k] = T2_row[k, i]
         *   op(B)='T' on Tt_col -> Tt_col[j, k] = Tt_row[k, j]
         *   C_col[i, j] = sum_k T2_row[k, i] * Tt_row[k, j]
         *   Row-major C[p, q] = C_col[q, p] = sum_k T2_row[k, q] * Tt_row[k, p]
         *   = sum_r Tt[r, p] * T2[r, q]   ✓
         */
        dgemm_(&N_flag, &T_flag,
               &int_npno, &int_npno, &int_npno,
               &one, T2,        &int_npno,
               Tt,              &int_npno,
               &one, D_pair,    &int_npno);

        if (same_lmo[k]) {
            const double half = 0.5;
            const size_t nsq = (size_t)npno * (size_t)npno;
            for (size_t ii = 0; ii < nsq; ii++) D_pair[ii] *= half;
        }

        /* DS = D_pair @ S_proj  (npno, n_pao_can)
         * Row-major: DS[p, a] = sum_q D_pair[p, q] * S_proj[q, a]
         * Col-major: dgemm('N','N', n_pao_can, npno, npno,
         *                  1, S_proj, n_pao_can, D_pair, npno,
         *                  0, DS, n_pao_can)
         */
        dgemm_(&N_flag, &N_flag,
               &int_npc, &int_npno, &int_npno,
               &one, S_proj,        &int_npc,
               D_pair,              &int_npno,
               &zero, DS,           &int_npc);

        /* D_ijk += S_proj.T @ DS  (n_pao_can, n_pao_can)
         * Row-major: D_ijk[a, b] += sum_p S_proj[p, a] * DS[p, b]
         * Col-major dgemm: dgemm('N','T', n_pao_can, n_pao_can, npno,
         *                       1, DS, n_pao_can, S_proj, n_pao_can,
         *                       beta=1, D_ijk, n_pao_can)
         * Verify: op(A)='N' on DS_col -> DS_col[i, k] = DS_row[k, i]
         *         op(B)='T' on S_proj_col -> S_proj_col[j, k] = S_proj_row[k, j]
         *         C_col[i, j] = sum_k DS_row[k, i] * S_proj_row[k, j]
         *         C_row[a, b] = C_col[b, a] = sum_k DS_row[k, b] * S_proj_row[k, a]
         *                     = sum_p S_proj[p, a] * DS[p, b]  ✓
         */
        dgemm_(&N_flag, &T_flag,
               &int_npc, &int_npc, &int_npno,
               &one, DS,            &int_npc,
               S_proj,              &int_npc,
               &one, D_ijk_out,     &int_npc);
    }

    /* D_ijk /= 3.0 */
    {
        const size_t nsq = (size_t)n_pao_can * (size_t)n_pao_can;
        const double inv3 = 1.0 / 3.0;
        for (size_t ii = 0; ii < nsq; ii++) D_ijk_out[ii] *= inv3;
    }

    free(S_pair_triple);
    free(S_pair_X);
    free(S_proj);
    free(Tt);
    free(DT);
    free(D_pair);
    free(DS);
}
