/* DLPNO-(T) per-triple orchestrator — Phase 3c-1.
 *
 * Single C entry point that does the TNO transform + per-triple local
 * DF + U cache build for one triple, eliminating ~3 ctypes round-trips
 * and ~300μs of Python orchestration per triple.
 *
 * Subsequent phases will fold in t2_block, K_ab, K_ooov, W3, and
 * energy contraction; the final phase wraps the whole loop in OMP.
 *
 * Phase 3c-1 returns the heavy intermediate buffers via output ptrs;
 * Python continues with steps 7-14 of `_process_one_triple` until the
 * full body is in C.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vhf/fblas.h"

double DLPNOcompute_w3_energy(
        const double *K_ab_cache, const double *t2_T_all,
        const double *K_jk, const double *K_ik, const double *K_ij,
        const double *K_ooov,
        const double *U_flat, const long *U_offsets, const long *n_pno_arr,
        const double *T2_flat, const long *T2_offsets,
        const signed char *transpose_flags,
        const double *eps_occ, const double *eps_vir,
        const double *t1_sc,
        const int has_t1, const int occ_denom,
        const int n, const int m_dom, const int n_pno_max);

/* ----- Forward declarations of existing public kernels ----- */
int DLPNObuild_triple_tno_full(
        const int n_pao_ijk, const int n_pao_total,
        const long *triple_paos,
        const double *F_pao_full, const double *S_pao_full,
        const int n_keys, const int *pair_paos_n,
        const long *pair_paos_off, const long *pair_paos_flat,
        const int *n_pno_arr, const long *X_pno_off,
        const double *X_pno_flat,
        const long *T2_off, const double *T2_flat,
        const int *same_lmo,
        const double T_CutTNO, const double S_cut_domain,
        double **X_tno_ijk_out, double **eps_tno_out,
        double **X_pao_ijk_out,
        int *n_tno_out, int *n_pao_can_out);

void DLPNObuild_triple_local_DF(
        const int i_idx, const int j_idx, const int k_idx,
        const int n_tno, const int n_domain,
        const int n_pao_ijk, const int naux_ijk,
        const int n_centers, const int n_lmo_global, const int n_pao_global,
        const double *X_tno_ijk,
        const long *triple_paos, const long *triple_domain,
        const long *center_atoms, const long *center_off,
        const long *local_Q_flat, const long *atom_pos_flat,
        const long *qij_atom_off, const long *qia_atom_off,
        const long *qab_atom_off,
        const int  *qij_atom_n_aux, const int  *qij_atom_n_lmo,
        const int  *qab_atom_n_pao,
        const double *qij_atom_flat, const double *qia_atom_flat,
        const double *qab_atom_flat,
        const long *riatom_to_lmos_ext_dense,
        const long *riatom_to_paos_ext_dense,
        const double *jhi,
        double *ovL_sc, double *vvL_sc, double *ooL_sc);

void DLPNObuild_U_for_triple(
        const int n_pairs,
        const double *W_pao_tno, const int n_tno, const int nao_pao_total,
        const int *n_pao_arr, const int *n_pno_arr,
        const long *pp_off, const long *pp_flat,
        const long *X_off, const double *X_flat,
        const long *U_off, double *U_flat);

/* LAPACK */
void dsyevd_(const char *jobz, const char *uplo,
             const int *n, double *a, const int *lda,
             double *w, double *work, const int *lwork,
             int *iwork, const int *liwork, int *info);

/* --------------------------------------------------------------------
 * Helper: build local J^{-1/2} (jhi) from j2c_full[aux_idx, aux_idx].
 * -------------------------------------------------------------------- */
static int build_jhi(
        const double *j2c_full, const int naux_total,
        const long *aux_idx, const int naux_ijk,
        double *jhi_out)
{
    if (naux_ijk == 0) return 0;
    const char N = 'N', T = 'T';
    const double one = 1.0, zero = 0.0;

    /* j_loc (naux_ijk × naux_ijk) symmetric */
    double *j_loc = (double *)malloc(
        sizeof(double) * (size_t)naux_ijk * (size_t)naux_ijk);
    for (int a = 0; a < naux_ijk; a++) {
        const long ra = aux_idx[a];
        for (int b = 0; b < naux_ijk; b++) {
            j_loc[(size_t)a * naux_ijk + b]
                = j2c_full[(size_t)ra * naux_total + aux_idx[b]];
        }
    }

    /* eigh: dsyevd in-place */
    char Jc = 'V', Uc = 'L';
    int n = naux_ijk, info = 0, lwork_q = -1, liwork_q = -1;
    double opt_lwork = 0.0;
    int    opt_liwork = 0;
    double *w = (double *)malloc(sizeof(double) * naux_ijk);
    dsyevd_(&Jc, &Uc, &n, j_loc, &n, w, &opt_lwork, &lwork_q,
            &opt_liwork, &liwork_q, &info);
    if (info != 0) { free(j_loc); free(w); return info; }
    int lwork = (int)opt_lwork, liwork = opt_liwork;
    double *work = (double *)malloc(sizeof(double) * lwork);
    int *iwork = (int *)malloc(sizeof(int) * liwork);
    dsyevd_(&Jc, &Uc, &n, j_loc, &n, w, work, &lwork, iwork, &liwork, &info);
    free(work); free(iwork);
    if (info != 0) { free(j_loc); free(w); return info; }

    /* Threshold and build jhi = V * D^{-1/2} * V^T (V col-major in j_loc).
     * Equivalent to M @ M^T where M[i, k] = V[i, k] / sqrt(w[k]). */
    int n_keep = 0;
    for (int k = 0; k < naux_ijk; k++) if (w[k] > 1e-14) n_keep++;
    if (n_keep == 0) {
        memset(jhi_out, 0, sizeof(double) * (size_t)naux_ijk * naux_ijk);
        free(j_loc); free(w); return 0;
    }
    double *M = (double *)malloc(
        sizeof(double) * (size_t)naux_ijk * n_keep);
    int kept = 0;
    for (int k = 0; k < naux_ijk; k++) {
        if (w[k] <= 1e-14) continue;
        /* For jhi = V @ D^{-1/2} @ V^T computed as M @ M^T, M[i, k] needs
         * scale = w[k]^{-1/4} so that sum_k M[a,k]*M[b,k] = sum_k V[a,k]*V[b,k]/sqrt(w[k]). */
        const double scale = 1.0 / sqrt(sqrt(w[k]));
        /* col-major V: j_loc[i + k*naux_ijk] = V[i, k] */
        for (int i = 0; i < naux_ijk; i++) {
            M[(size_t)i * n_keep + kept]
                = j_loc[(size_t)k * naux_ijk + i] * scale;
        }
        kept++;
    }
    /* jhi (naux_ijk, naux_ijk) = M (naux_ijk, n_keep) @ M^T (n_keep, naux_ijk).
     * Row-major C[a, b] = sum_k M[a, k] * M[b, k].
     *
     * dgemm rule for row-major C(M, N) = A(M, K) @ B(K, N):
     *   dgemm('N','N', N, M, K, 1, B (LDB=N), A (LDA=K), 0, C (LDC=N))
     * Here B = M^T row-major (n_keep, naux_ijk), but we have M row-major
     * (naux_ijk, n_keep). Pass M with TRANSB='T' so dgemm sees it transposed.
     *
     * dgemm('N','T', M=naux_ijk, N=naux_ijk, K=n_keep,
     *       A=M (LDA=n_keep, treated as transposed via TRANSA='N' on
     *       col-major view), B=M (LDB=n_keep, TRANSB='T'),
     *       C=jhi (LDC=naux_ijk))
     *
     * Actually, simpler: use the row-major rule directly. dgemm in col-major
     * computes C_col = op(A) @ op(B). If I want row-major C[i,j] = M[i,k]*M[j,k],
     * map to col-major C_col[j,i] = sum_k M_col[i,k]^T * M_col[j,k]^T...
     *
     * Cleanest: do C = M @ M^T row-major using:
     *   dgemm('T','N', n=naux_ijk, m=naux_ijk, k=n_keep,
     *         B=M (LDB=n_keep, ↓no transpose flag↓),
     *         A=M (LDA=n_keep), 0, C, ldc=naux_ijk)
     *   verify with col-major:
     *     op(A_col)='T' on M_col(K, M): A_col_T[i,k]=M_col[k,i]=M_row[i,k]
     *     op(B_col)='N' on M_col(K, N): B_col[k,j]=M_col[k,j]=M_row[j,k]
     *     C_col[i,j] = sum_k M_row[i,k] * M_row[j,k]
     *     C_row[i,j] = C_col[j,i] = sum_k M_row[j,k] * M_row[i,k] ✓ (symmetric)
     */
    int int_n = naux_ijk, int_nk = n_keep;
    dgemm_(&T, &N, &int_n, &int_n, &int_nk,
           &one, M, &int_nk, M, &int_nk,
           &zero, jhi_out, &int_n);

    free(M); free(j_loc); free(w);
    return 0;
}

/* --------------------------------------------------------------------
 * Helper: groupby sort (stable) for centers_of_aux.
 * -------------------------------------------------------------------- */
static void groupby_centers(
        const long *aux_idx, const int naux_ijk,
        const long *aux_atom_ids, const long *aux_pos_in_atom,
        long *local_Q_sorted, long *atom_pos_sorted,
        long *center_atoms, long *center_off, int *n_centers_out)
{
    long *centers_of_aux = (long *)malloc(sizeof(long) * naux_ijk);
    long *order = (long *)malloc(sizeof(long) * naux_ijk);
    for (int q = 0; q < naux_ijk; q++) {
        centers_of_aux[q] = aux_atom_ids[aux_idx[q]];
        order[q] = q;
    }
    /* Stable insertion sort by centers_of_aux ascending */
    for (int i = 1; i < naux_ijk; i++) {
        long key_pos = order[i];
        long key_v = centers_of_aux[key_pos];
        int j = i - 1;
        while (j >= 0 && centers_of_aux[order[j]] > key_v) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key_pos;
    }
    for (int q = 0; q < naux_ijk; q++) {
        local_Q_sorted[q] = order[q];
        atom_pos_sorted[q] = aux_pos_in_atom[aux_idx[order[q]]];
    }
    int n_c = 0;
    center_off[0] = 0;
    int run_start = 0;
    for (int q = 0; q < naux_ijk; q++) {
        if (q == naux_ijk - 1 ||
            centers_of_aux[order[q + 1]] != centers_of_aux[order[run_start]]) {
            center_atoms[n_c] = centers_of_aux[order[run_start]];
            center_off[n_c + 1] = q + 1;
            n_c++;
            run_start = q + 1;
        }
    }
    *n_centers_out = n_c;
    free(centers_of_aux); free(order);
}

/* --------------------------------------------------------------------
 * Public: do TNO transform + local DF + U cache for one triple.
 *
 * Outputs (caller-allocated):
 *   X_tno_ijk_out    (n_pao_ijk_max × n_pao_ijk_max)  — written at LD = n_pao_ijk
 *   eps_tno_out      (n_pao_ijk_max,)
 *   ovL_sc_out       (3 × n_tno_max × naux_max)       — written at (3, n_tno, naux_ijk)
 *   vvL_sc_out       (n_tno_max × n_tno_max × naux_max)
 *   ooL_sc_out       (3 × n_dom_max × naux_max)
 *   U_flat_out       sized to sum(n_pno_arr * n_tno) over u_pks
 *   U_off_out        (n_u_pks + 1,) — written
 *
 *   n_tno_out, naux_ijk_out
 *
 * Returns: 0 on success.
 * -------------------------------------------------------------------- */
int DLPNOprocess_one_triple_phase1(
        /* triple */
        const int i, const int j, const int k,
        const int n_pao_ijk, const long *triple_paos,
        const int n_dom, const long *triple_domain,
        /* 3-pair specific (ij, jk, ik) */
        const int *pair_paos_n_3, const long *pair_paos_off_3,
        const long *pair_paos_flat_3,
        const int *n_pno_arr_3, const long *X_pno_off_3,
        const double *X_pno_flat_3,
        const long *T2_off_3, const double *T2_flat_3,
        const int *same_lmo_3,
        /* u_pks for U cache */
        const int n_u_pks,
        const int *u_pao_n, const int *u_pno_n,
        const long *u_pp_off, const long *u_pp_flat,
        const long *u_X_off, const double *u_X_flat,
        /* PAO/aux globals */
        const int nocc_lmo, const int n_pao_total,
        const int naux_total,
        const double *F_pao_full, const double *S_pao_full,
        const double *j2c_full,
        const long *aux_atom_ids, const long *aux_pos_in_atom,
        const long *qij_atom_off, const long *qia_atom_off,
        const long *qab_atom_off,
        const int  *qij_atom_n_aux, const int  *qij_atom_n_lmo,
        const int  *qab_atom_n_pao,
        const double *qij_atom_flat, const double *qia_atom_flat,
        const double *qab_atom_flat,
        const long *riatom_to_lmos_ext_dense,
        const long *riatom_to_paos_ext_dense,
        const long *lmo_aux_mask,
        /* config */
        const double T_CutTNO, const double S_cut_domain,
        /* output buffers (max-sized by caller) */
        const int n_pao_ijk_stride,    /* leading dim of X_tno_ijk_out (= n_pao_ijk) */
        const int n_tno_max,           /* used for ovL/vvL stride */
        const int naux_max,            /* used for ovL/vvL stride */
        double *X_tno_ijk_out,         /* (n_pao_ijk, n_pao_ijk) row-major */
        double *eps_tno_out,           /* (n_pao_ijk,) */
        double *ovL_sc_out,            /* (3, n_tno, naux) — caller passes max sizes */
        double *vvL_sc_out,            /* (n_tno, n_tno, naux) */
        double *ooL_sc_out,            /* (3, n_dom, naux) */
        long   *U_off_out,             /* (n_u_pks+1,) */
        double *U_flat_out,            /* caller-sized */
        /* output scalars */
        int *n_tno_out, int *naux_ijk_out)
{
    *n_tno_out = 0;
    *naux_ijk_out = 0;
    if (n_pao_ijk == 0) return 0;

    /* --- Step 1: TNO transform (Phase 3a) --- */
    double *X_tno_ijk = NULL, *eps_tno = NULL, *X_pao_ijk = NULL;
    int n_tno = 0, n_pao_can = 0;
    int rc = DLPNObuild_triple_tno_full(
        n_pao_ijk, n_pao_total, triple_paos,
        F_pao_full, S_pao_full,
        3, pair_paos_n_3, pair_paos_off_3, pair_paos_flat_3,
        n_pno_arr_3, X_pno_off_3, X_pno_flat_3,
        T2_off_3, T2_flat_3, same_lmo_3,
        T_CutTNO, S_cut_domain,
        &X_tno_ijk, &eps_tno, &X_pao_ijk,
        &n_tno, &n_pao_can);
    if (rc != 0 || n_tno == 0) {
        if (X_tno_ijk) free(X_tno_ijk);
        if (eps_tno)   free(eps_tno);
        if (X_pao_ijk) free(X_pao_ijk);
        return rc;
    }
    *n_tno_out = n_tno;

    /* Copy X_tno_ijk (n_pao_ijk × n_tno) into output buffer with leading
     * dim = n_pao_ijk_stride.  Caller will read the first n_tno columns. */
    for (int r = 0; r < n_pao_ijk; r++) {
        memcpy(X_tno_ijk_out + (size_t)r * n_pao_ijk_stride,
               X_tno_ijk + (size_t)r * n_tno,
               sizeof(double) * (size_t)n_tno);
    }
    memcpy(eps_tno_out, eps_tno, sizeof(double) * n_tno);
    free(X_pao_ijk);

    /* --- Step 2: Build aux_idx from lmo_aux_mask[i] | [j] | [k] --- */
    int naux_ijk = 0;
    long *aux_idx = (long *)malloc(sizeof(long) * naux_total);
    for (int q = 0; q < naux_total; q++) {
        long m_i = lmo_aux_mask[(size_t)i * naux_total + q];
        long m_j = lmo_aux_mask[(size_t)j * naux_total + q];
        long m_k = lmo_aux_mask[(size_t)k * naux_total + q];
        if (m_i || m_j || m_k) aux_idx[naux_ijk++] = q;
    }
    *naux_ijk_out = naux_ijk;
    if (naux_ijk == 0) {
        free(aux_idx); free(X_tno_ijk); free(eps_tno);
        return 0;
    }

    /* --- Step 3: jhi (local J^{-1/2}) --- */
    double *jhi = (double *)malloc(
        sizeof(double) * (size_t)naux_ijk * naux_ijk);
    int jrc = build_jhi(j2c_full, naux_total, aux_idx, naux_ijk, jhi);
    if (jrc != 0) {
        free(jhi); free(aux_idx); free(X_tno_ijk); free(eps_tno);
        return jrc;
    }

    /* --- Step 4: per-center groupby for local DF --- */
    long *local_Q_sorted = (long *)malloc(sizeof(long) * naux_ijk);
    long *atom_pos_sorted = (long *)malloc(sizeof(long) * naux_ijk);
    long *center_atoms = (long *)malloc(sizeof(long) * naux_ijk);
    long *center_off = (long *)malloc(sizeof(long) * (naux_ijk + 1));
    int n_centers = 0;
    groupby_centers(aux_idx, naux_ijk, aux_atom_ids, aux_pos_in_atom,
                    local_Q_sorted, atom_pos_sorted,
                    center_atoms, center_off, &n_centers);
    free(aux_idx);

    /* --- Step 5: build local DF integrals into output buffers ---
     * Output buffer has stride matching naux_max in the last axis.
     * If naux_max > naux_ijk, write into rows 0..n_tno-1 with stride naux_max,
     * but the kernel writes contiguously assuming stride = naux_ijk. We need
     * an intermediate buffer. */
    double *ovL_local = (double *)calloc(
        (size_t)3 * n_tno * naux_ijk, sizeof(double));
    double *vvL_local = (double *)calloc(
        (size_t)n_tno * n_tno * naux_ijk, sizeof(double));
    double *ooL_local = (double *)calloc(
        (size_t)3 * n_dom * naux_ijk, sizeof(double));
    DLPNObuild_triple_local_DF(
        i, j, k, n_tno, n_dom, n_pao_ijk, naux_ijk, n_centers,
        nocc_lmo, n_pao_total,
        X_tno_ijk, triple_paos, triple_domain,
        center_atoms, center_off, local_Q_sorted, atom_pos_sorted,
        qij_atom_off, qia_atom_off, qab_atom_off,
        qij_atom_n_aux, qij_atom_n_lmo, qab_atom_n_pao,
        qij_atom_flat, qia_atom_flat, qab_atom_flat,
        riatom_to_lmos_ext_dense, riatom_to_paos_ext_dense,
        jhi, ovL_local, vvL_local, ooL_local);

    free(local_Q_sorted); free(atom_pos_sorted);
    free(center_atoms); free(center_off); free(jhi);

    /* Copy ovL/vvL/ooL into output buffers with stride naux_max in the
     * trailing axis. Each row becomes a slice of length naux_ijk, padded
     * (caller knows naux_ijk_out and slices to it). */
    for (int p = 0; p < 3; p++) {
        for (int t = 0; t < n_tno; t++) {
            memcpy(ovL_sc_out + ((size_t)p * n_tno_max + t) * naux_max,
                   ovL_local + ((size_t)p * n_tno + t) * naux_ijk,
                   sizeof(double) * naux_ijk);
        }
    }
    for (int a = 0; a < n_tno; a++) {
        for (int b = 0; b < n_tno; b++) {
            memcpy(vvL_sc_out + ((size_t)a * n_tno_max + b) * naux_max,
                   vvL_local + ((size_t)a * n_tno + b) * naux_ijk,
                   sizeof(double) * naux_ijk);
        }
    }
    for (int p = 0; p < 3; p++) {
        for (int m = 0; m < n_dom; m++) {
            memcpy(ooL_sc_out + ((size_t)p * n_dom + m) * naux_max,
                   ooL_local + ((size_t)p * n_dom + m) * naux_ijk,
                   sizeof(double) * naux_ijk);
        }
    }
    free(ovL_local); free(vvL_local); free(ooL_local);

    /* --- Step 6: U cache for u_pks --- */
    if (n_u_pks > 0) {
        /* W_pao_tno (n_pao_total, n_tno) = S_pao_full[:, triple_paos] @ X_tno_ijk */
        double *S_slice = (double *)malloc(
            sizeof(double) * (size_t)n_pao_total * n_pao_ijk);
        for (int rr = 0; rr < n_pao_total; rr++) {
            const double *src = S_pao_full + (size_t)rr * n_pao_total;
            double *dst = S_slice + (size_t)rr * n_pao_ijk;
            for (int cc = 0; cc < n_pao_ijk; cc++) {
                dst[cc] = src[triple_paos[cc]];
            }
        }
        double *W_pao_tno = (double *)malloc(
            sizeof(double) * (size_t)n_pao_total * n_tno);
        const char N = 'N';
        const double one = 1.0, zero = 0.0;
        int int_nt = n_tno, int_nao = n_pao_total, int_npi = n_pao_ijk;
        dgemm_(&N, &N,
               &int_nt, &int_nao, &int_npi,
               &one, X_tno_ijk, &int_nt,
               S_slice, &int_npi,
               &zero, W_pao_tno, &int_nt);
        free(S_slice);

        /* Build U_off cumsum */
        U_off_out[0] = 0;
        for (int p = 0; p < n_u_pks; p++) {
            U_off_out[p + 1] = U_off_out[p] + (long)u_pno_n[p] * n_tno;
        }
        DLPNObuild_U_for_triple(
            n_u_pks, W_pao_tno, n_tno, n_pao_total,
            u_pao_n, u_pno_n,
            u_pp_off, u_pp_flat, u_X_off, u_X_flat,
            U_off_out, U_flat_out);
        free(W_pao_tno);
    } else {
        U_off_out[0] = 0;
    }

    free(X_tno_ijk); free(eps_tno);
    return 0;
}
