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

/* --------------------------------------------------------------------
 * Per-thread scratch arena.  Eliminates the ~25 malloc/free pairs per
 * triple in DLPNOcompute_one_triple_E_T0 — each pthread reuses the
 * same buffers across triples (grow-only).
 *
 * The Python pool reuses worker threads, so __thread storage persists
 * across triples for the same worker.  Buffers are grown on first use
 * and never freed (cleanup via thread exit / process exit).
 * -------------------------------------------------------------------- */
typedef struct {
    long *aux_idx;          size_t aux_idx_cap;
    double *jhi;            size_t jhi_cap;
    long *local_Q_sorted;   size_t local_Q_sorted_cap;
    long *atom_pos_sorted;  size_t atom_pos_sorted_cap;
    long *center_atoms;     size_t center_atoms_cap;
    long *center_off;       size_t center_off_cap;
    double *ovL_sc;         size_t ovL_sc_cap;
    double *vvL_sc;         size_t vvL_sc_cap;
    double *ooL_sc;         size_t ooL_sc_cap;
    double *S_slice;        size_t S_slice_cap;
    double *W_pao_tno;      size_t W_pao_tno_cap;
    long *U_off_cache;      size_t U_off_cache_cap;
    double *U_flat_cache;   size_t U_flat_cache_cap;
    double *K_ab_cache;     size_t K_ab_cache_cap;
    double *t_tmp;          size_t t_tmp_cap;
    double *K_ooov;         size_t K_ooov_cap;
    double *K_pre;          size_t K_pre_cap;
    double *K_jk;           size_t K_jk_cap;
    double *K_ik;           size_t K_ik_cap;
    double *K_ij;           size_t K_ij_cap;
    double *t1_lmo;         size_t t1_lmo_cap;
    double *t2_block;       size_t t2_block_cap;
    double *T2U_buf;        size_t T2U_buf_cap;
    double *block_tmp;      size_t block_tmp_cap;
    double *t2_T_all;       size_t t2_T_all_cap;
    long *w3_n_pno_arr;     size_t w3_n_pno_arr_cap;
    long *w3_U_off;         size_t w3_U_off_cap;
    long *w3_T2_off;        size_t w3_T2_off_cap;
    signed char *w3_tflags; size_t w3_tflags_cap;
} TScratch;

static __thread TScratch tscratch = {0};

#define ENSURE(field, T, n)                                              \
    do {                                                                 \
        size_t _need = (size_t)(n);                                      \
        if (tscratch.field##_cap < _need) {                              \
            free(tscratch.field);                                        \
            tscratch.field = (T *)malloc(sizeof(T) * (_need > 0 ? _need : 1)); \
            tscratch.field##_cap = _need;                                \
        }                                                                \
    } while (0)

/* ====================================================================
 * Phase 3c-2/3: full single-call orchestrator (returns et_ijk).
 *
 * Does TNO + DF + U cache + t2_block + K_ab + K_ooov + K_*_for_V + W3
 * all in one C entry point.  Replaces the entire post-screening body
 * of `_process_one_triple` in Python.
 * ==================================================================== */
double DLPNOcompute_one_triple_E_T0(
        /* triple */
        const int i, const int j, const int k,
        const int n_pao_ijk, const long *triple_paos,
        const int n_dom, const long *triple_domain,
        /* 3-pair (TNO) */
        const int *pair_paos_n_3, const long *pair_paos_off_3,
        const long *pair_paos_flat_3,
        const int *n_pno_arr_3, const long *X_pno_off_3,
        const double *X_pno_flat_3,
        const long *T2_off_3, const double *T2_flat_3,
        const int *same_lmo_3,
        /* u_pks (X for U cache + T2 for w3 + t2_block) */
        const int n_u_pks,
        const int *u_pao_n, const int *u_pno_n,
        const long *u_pp_off, const long *u_pp_flat,
        const long *u_X_off, const double *u_X_flat,
        const long *u_T2_off, const double *u_T2_flat,
        /* t2_block 3x3 */
        const int *t2_block_u_pk_idx,    /* (9,) — u_pk index or -1 */
        const signed char *t2_block_transpose,  /* (9,) */
        /* w3 task table (r, l) flat = r*m_dom + l */
        const int *w3_u_pk_idx,          /* (3*m_dom,) — u_pk index or -1 */
        const signed char *w3_transpose, /* (3*m_dom,) */
        /* t1 path */
        const int has_t1,
        const long *t1_off, const double *t1_flat,
        const int *t1_diag_u_pk_idx,     /* (3,) — diag (r,r) u_pk per r */
        const long *t1_lmo_idx,          /* (3,) — global LMO indices [i,j,k] */
        /* eps */
        const double eps_i, const double eps_j, const double eps_k,
        const int occ_denom,
        /* PAO/aux globals */
        const int nocc_lmo, const int n_pao_total, const int naux_total,
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
        const double T_CutTNO, const double S_cut_domain)
{
    if (n_pao_ijk == 0) return 0.0;
    const char Nc = 'N', Tc = 'T';
    const double one = 1.0, zero = 0.0;

    /* === Phase 1: TNO + DF + U cache (re-implemented inline) === */
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
        return 0.0;
    }
    free(X_pao_ijk);
    int n = n_tno;

    /* aux_idx + jhi + groupby — using thread scratch */
    ENSURE(aux_idx, long, naux_total);
    long *aux_idx = tscratch.aux_idx;
    int naux_ijk = 0;
    for (int q = 0; q < naux_total; q++) {
        if (lmo_aux_mask[(size_t)i * naux_total + q]
            || lmo_aux_mask[(size_t)j * naux_total + q]
            || lmo_aux_mask[(size_t)k * naux_total + q]) {
            aux_idx[naux_ijk++] = q;
        }
    }
    if (naux_ijk == 0) {
        free(X_tno_ijk); free(eps_tno);
        return 0.0;
    }
    ENSURE(jhi, double, (size_t)naux_ijk * naux_ijk);
    double *jhi = tscratch.jhi;
    if (build_jhi(j2c_full, naux_total, aux_idx, naux_ijk, jhi) != 0) {
        free(X_tno_ijk); free(eps_tno);
        return 0.0;
    }

    ENSURE(local_Q_sorted, long, naux_ijk);
    ENSURE(atom_pos_sorted, long, naux_ijk);
    ENSURE(center_atoms, long, naux_ijk);
    ENSURE(center_off, long, naux_ijk + 1);
    long *local_Q_sorted = tscratch.local_Q_sorted;
    long *atom_pos_sorted = tscratch.atom_pos_sorted;
    long *center_atoms = tscratch.center_atoms;
    long *center_off = tscratch.center_off;
    int n_centers = 0;
    groupby_centers(aux_idx, naux_ijk, aux_atom_ids, aux_pos_in_atom,
                    local_Q_sorted, atom_pos_sorted,
                    center_atoms, center_off, &n_centers);

    ENSURE(ovL_sc, double, (size_t)3 * n * naux_ijk);
    ENSURE(vvL_sc, double, (size_t)n * n * naux_ijk);
    ENSURE(ooL_sc, double, (size_t)3 * (size_t)n_dom * naux_ijk);
    double *ovL_sc = tscratch.ovL_sc;
    double *vvL_sc = tscratch.vvL_sc;
    double *ooL_sc = tscratch.ooL_sc;
    /* DF kernel zeros what it doesn't write; explicit memset for safety */
    memset(ovL_sc, 0, sizeof(double) * (size_t)3 * n * naux_ijk);
    memset(vvL_sc, 0, sizeof(double) * (size_t)n * n * naux_ijk);
    if (n_dom > 0) {
        memset(ooL_sc, 0, sizeof(double) * (size_t)3 * n_dom * naux_ijk);
    }
    DLPNObuild_triple_local_DF(
        i, j, k, n, n_dom, n_pao_ijk, naux_ijk, n_centers,
        nocc_lmo, n_pao_total,
        X_tno_ijk, triple_paos, triple_domain,
        center_atoms, center_off, local_Q_sorted, atom_pos_sorted,
        qij_atom_off, qia_atom_off, qab_atom_off,
        qij_atom_n_aux, qij_atom_n_lmo, qab_atom_n_pao,
        qij_atom_flat, qia_atom_flat, qab_atom_flat,
        riatom_to_lmos_ext_dense, riatom_to_paos_ext_dense,
        jhi, ovL_sc, vvL_sc, ooL_sc);

    /* W_pao_tno + U cache */
    long *U_off_cache = NULL;
    double *U_flat_cache = NULL;
    if (n_u_pks > 0) {
        ENSURE(S_slice, double, (size_t)n_pao_total * n_pao_ijk);
        double *S_slice = tscratch.S_slice;
        for (int rr = 0; rr < n_pao_total; rr++) {
            const double *src = S_pao_full + (size_t)rr * n_pao_total;
            double *dst = S_slice + (size_t)rr * n_pao_ijk;
            for (int cc = 0; cc < n_pao_ijk; cc++) {
                dst[cc] = src[triple_paos[cc]];
            }
        }
        ENSURE(W_pao_tno, double, (size_t)n_pao_total * n);
        double *W_pao_tno = tscratch.W_pao_tno;
        int int_n = n, int_nao = n_pao_total, int_npi = n_pao_ijk;
        dgemm_(&Nc, &Nc, &int_n, &int_nao, &int_npi,
               &one, X_tno_ijk, &int_n,
               S_slice, &int_npi,
               &zero, W_pao_tno, &int_n);

        ENSURE(U_off_cache, long, n_u_pks + 1);
        U_off_cache = tscratch.U_off_cache;
        U_off_cache[0] = 0;
        for (int p = 0; p < n_u_pks; p++) {
            U_off_cache[p + 1] = U_off_cache[p] + (long)u_pno_n[p] * n;
        }
        ENSURE(U_flat_cache, double, (size_t)U_off_cache[n_u_pks]);
        U_flat_cache = tscratch.U_flat_cache;
        DLPNObuild_U_for_triple(
            n_u_pks, W_pao_tno, n, n_pao_total,
            u_pao_n, u_pno_n,
            u_pp_off, u_pp_flat, u_X_off, u_X_flat,
            U_off_cache, U_flat_cache);
    }

    /* === Phase 2: t2_block + K_ab + K_ooov + K_*_for_V + W3 === */
    int int_n = n;
    int int_nn = n * n;

    /* t2_block (3, 3, n, n) row-major.
     * For each (p, q) in 3x3:
     *   pk_idx = t2_block_u_pk_idx[p*3 + q]; if -1, leave zero.
     *   U_p = U_flat_cache + U_off_cache[pk_idx_for_p_diag]?  wait — need
     *   careful: t2_block[p, q] = U_p.T @ T2_pq @ U_q where U_p is for
     *   the pair (lmo_p, lmo_q) actually... no.
     *
     * Actually t2_block[p, q] is defined in Python as:
     *   _proj_t2(lmo_triple[p], lmo_triple[q]) =
     *     U_pk.T @ t2_for_T[pk] @ U_pk  with optional .T transpose
     * where pk = (min, max) of the two LMOs.
     *
     * So both U's come from the SAME u_pk (the canonical pair of (lmo_p, lmo_q)).
     * U_pk has shape (n_pno_pk, n_tno). So:
     *   tmp = T2_pq @ U_pk  (n_pno × n_tno)
     *   block = U_pk.T @ tmp  (n_tno × n_tno)
     *   if lmo_p > lmo_q: transpose block.
     */
    ENSURE(t2_block, double, (size_t)9 * n * n);
    double *t2_block = tscratch.t2_block;
    memset(t2_block, 0, sizeof(double) * (size_t)9 * n * n);
    {
        /* Find max n_pno across t2_block u_pks for scratch sizing */
        int max_npno_t2b = 0;
        for (int pq = 0; pq < 9; pq++) {
            int u_idx = t2_block_u_pk_idx[pq];
            if (u_idx >= 0 && u_pno_n[u_idx] > max_npno_t2b) {
                max_npno_t2b = u_pno_n[u_idx];
            }
        }
        if (max_npno_t2b > 0) {
            ENSURE(T2U_buf, double, (size_t)max_npno_t2b * n);
            ENSURE(block_tmp, double, (size_t)n * n);
            double *T2U_buf = tscratch.T2U_buf;
            double *block_tmp = tscratch.block_tmp;
            for (int pq = 0; pq < 9; pq++) {
                int u_idx = t2_block_u_pk_idx[pq];
                if (u_idx < 0) continue;
                int n_pno = u_pno_n[u_idx];
                if (n_pno == 0) continue;
                const double *U_pk = U_flat_cache + U_off_cache[u_idx];
                const double *T2_pk = u_T2_flat + u_T2_off[u_idx];
                int int_n_pno = n_pno;
                /* T2U = T2 @ U : (n_pno, n_tno) = (n_pno, n_pno) @ (n_pno, n_tno)
                 * Cython call:
                 *   dgemm('N','N', n_tno, n_pno, n_pno,
                 *         1, U, n_tno, T2, n_pno, 0, T2U, n_tno) */
                dgemm_(&Nc, &Nc, &int_n, &int_n_pno, &int_n_pno,
                       &one, U_pk, &int_n,
                       T2_pk, &int_n_pno,
                       &zero, T2U_buf, &int_n);
                /* block = U.T @ T2U : (n_tno, n_tno)
                 *   dgemm('N','T', n_tno, n_tno, n_pno,
                 *         1, T2U, n_tno, U, n_tno, 0, block, n_tno)
                 */
                dgemm_(&Nc, &Tc, &int_n, &int_n, &int_n_pno,
                       &one, T2U_buf, &int_n,
                       U_pk, &int_n,
                       &zero, block_tmp, &int_n);
                /* Optional transpose if non-canonical */
                double *dst = t2_block + (size_t)pq * n * n;
                if (t2_block_transpose[pq]) {
                    for (int a = 0; a < n; a++) {
                        for (int b = 0; b < n; b++) {
                            dst[(size_t)a*n + b] = block_tmp[(size_t)b*n + a];
                        }
                    }
                } else {
                    memcpy(dst, block_tmp, sizeof(double) * (size_t)n * n);
                }
            }
        }
    }

    /* K_ab_cache (3, n, n, n): K_ab[ip, a, b, f] = sum_L ovL[ip, a, L] * vvL[b, f, L]
     * Or equivalently K_ab[ip, a, b, f] = ovL[ip, a, :] @ vvL[b, f, :].T
     *
     * Computed Python-side as:
     *   t = np.tensordot(ovL_sc[ip], vvL_sc, axes=([1], [2])).transpose(0, 2, 1)
     *   shape (n, n, n) [a, b, f]
     *
     * For one ip:
     *   Reshape ovL[ip] (n, naux) and vvL (n, n, naux).
     *   tmp[a, b, f] = sum_q ovL[ip, a, q] * vvL[b, f, q]
     *   Equivalently: K_tmp (n, n*n) = ovL[ip] (n, naux) @ vvL_resh (naux, n*n)
     *     where vvL_resh[q, b*n + f] = vvL[b, f, q].
     *   Then K_ab[ip, a, b, f] = K_tmp[a, b*n + f] reshaped.
     *
     * Wait — vvL_sc has shape (n, n, naux). Treating as (n*n, naux), the
     * matmul is ovL[ip] (n, naux) @ vvL_T (naux, n*n) = (n, n*n).
     *
     * Row-major dgemm:
     *   ovL_ip (n, naux) @ vvL.T (naux, n*n) — vvL is (n*n, naux), so .T = (naux, n*n).
     *   Result (n, n*n) = ovL_ip @ vvL.T
     *   row-major: out[a, bf] = sum_q ovL[a, q] * vvL_resh[bf, q]
     *   where vvL_resh[bf, q] = vvL[b, f, q] (with bf = b*n + f).
     *
     *   So out[a, b*n + f] = sum_q ovL[ip, a, q] * vvL[b, f, q]
     *     = K_ab[ip, a, b, f]?
     *
     * From Python:
     *   t = np.tensordot(ovL_sc[ip], vvL_sc, axes=([1], [2]))
     *     → t[a, b, f] = sum_L ovL_sc[ip, a, L] * vvL_sc[b, f, L]
     *   Then K_ab_cache[ip] = t.transpose(0, 2, 1) → K_ab[ip, a, f, b]
     *   But wait, the stored shape is (3, n, n, n) and Python comment says
     *   "indexed [a, f, b]". Let me re-check.
     *
     * Actually in _w3_intermediate Python:
     *   K_ab_cache[ip][a, b, f] = ?
     *   The code:
     *     t = np.tensordot(ovL_sc[ip], vvL_sc, axes=([1], [2]))
     *     # shape (n, n, n) -- indexed [a, b, f]
     *     K_ab_cache[ip] = t.transpose(0, 2, 1)  # → [a, f, b]
     *
     *   And the Cython kernel uses K_ab_cache[ip, a, b, f] in the dgemm
     *   call as A_C (n*n, n) @ B_C (n, n) = base_buf (n*n, n).
     *   So K_ab_cache[ip] is (n, n, n) read as (n*n, n).
     *   The actual indices: K_ab_cache[ip, ?, ?, ?] = ovL[ip] @ vvL.T transposed how?
     *
     * Just match Python exactly. Python:
     *   t = np.tensordot(ovL_sc[ip], vvL_sc, axes=([1], [2]))
     *   K_ab_cache[ip] = t.transpose(0, 2, 1)
     *
     * Where ovL_sc[ip] is (n, naux), vvL_sc is (n, n, naux).
     * t[a, b, f] = sum_L ovL_sc[ip, a, L] * vvL_sc[b, f, L]  (axes=[1] on ovL, [2] on vvL)
     * Then transpose(0, 2, 1) gives K_ab_cache[ip, a, f, b] = t[a, b, f]
     * → K_ab_cache[ip, a, X, Y] where X=f, Y=b.
     *
     * So the stored layout is [ip, a, f, b] with the trailing two indices
     * "swapped" relative to what the math suggests.  We just need to
     * compute and lay it out the same way Python does.
     *
     * Easiest: compute K_temp[ip, a, b, f] via matmul, then transpose
     * trailing two axes when writing to K_ab_cache buffer.
     *
     * Or: compute directly: K_ab_cache[ip, a, f, b] = sum_L ovL[ip, a, L] * vvL[b, f, L].
     * That's still sum_L ovL @ vvL.T reshape.
     *
     * Let me just do the straightforward computation matching Python:
     *   t (n, n, n) = ovL[ip] @ vvL_resh.T   where vvL_resh[bf, L] = vvL[b, f, L]
     *   K_ab_cache[ip] (n, n, n) = t.transpose(0, 2, 1) at [a, f, b]
     */
    ENSURE(K_ab_cache, double, (size_t)3 * n * n * n);
    double *K_ab_cache = tscratch.K_ab_cache;
    {
        ENSURE(t_tmp, double, (size_t)n * n * n);
        double *t_tmp = tscratch.t_tmp;
        int int_naux = naux_ijk;
        for (int ip = 0; ip < 3; ip++) {
            const double *ovL_ip = ovL_sc + (size_t)ip * n * naux_ijk;
            /* t (n, n*n) = ovL_ip (n, naux) @ vvL_T (naux, n*n)
             * vvL is row-major (n*n, naux); .T treats it as (naux, n*n) col-major,
             * which is the same memory.
             *
             * Row-major matmul rule for C(M, N) = A(M, K) @ B^T(N, K).T = A @ B^T:
             *   dgemm('T','N', N, M, K, 1, B (LDB=K), A (LDA=K), 0, C (LDC=N))
             * Verify: op(A_col)='T' on A (n, naux): A^T_col[i,k]=A_col[k,i]=A_row[i,k] ✓
             *         op(B_col)='N' on vvL (n*n, naux): B_col[j,k]=vvL_col[j,k]=vvL_row[k,j]
             *           ... but I want vvL_row[j, k] (treating bf as flattened).
             *         Hmm. Actually vvL row-major is (n*n, naux): vvL[bf, q] =
             *         vvL[b, f, q]. So vvL_row[j, k] = vvL[bf=j, q=k].
             *         But B_col as col-major reads vvL_row transposed.
             *
             * Let me just use the row-major rule for C(M, N) = A(M, K) @ B(K, N):
             *   dgemm('N','N', N, M, K, 1, B (LDB=N), A (LDA=K), 0, C (LDC=N))
             * Here we want C(n, n*n) = ovL(n, naux) @ vvL_T(naux, n*n).
             *   But vvL is stored as (n*n, naux) row-major. To use it as
             *   (naux, n*n), I need to TRANSPOSE the storage interpretation.
             *
             * Alternative: dgemm with TRANSB='T' to transpose B.
             *   row-major C(M, N) = A(M, K) @ B^T(N, K) (where B is stored (N, K))
             *     dgemm('T','N', N, M, K, 1, B (LDB=K), A (LDA=K), 0, C (LDC=N))
             * Here M=n, N=n*n, K=naux. A=ovL_ip (n, naux), B=vvL (n*n, naux).
             *   ✓ matches the (M, N, K, A, B) pattern.
             */
            int int_n_sq = n * n;
            dgemm_(&Tc, &Nc, &int_n_sq, &int_n, &int_naux,
                   &one, vvL_sc, &int_naux,
                   ovL_ip, &int_naux,
                   &zero, t_tmp, &int_n_sq);
            /* t_tmp now row-major (n, n*n) — t_tmp[a, bf] = sum_q ovL[a, q] * vvL[bf, q]
             *   = sum_L ovL_sc[ip, a, L] * vvL_sc[b, f, L]
             *   = t_python[a, b, f].
             *
             * Now write K_ab_cache[ip] = t.transpose(0, 2, 1) i.e. K[a, f, b] = t[a, b, f].
             */
            double *K_ip = K_ab_cache + (size_t)ip * n * n * n;
            for (int a = 0; a < n; a++) {
                for (int b = 0; b < n; b++) {
                    for (int f = 0; f < n; f++) {
                        K_ip[((size_t)a * n + f) * n + b]
                            = t_tmp[((size_t)a * n * n) + b * n + f];
                    }
                }
            }
        }
    }

    /* K_ooov (3, 3, n, m_dom): K_ooov[p, q, a, m] = sum_L ovL[p, a, L] * ooL[q, m, L]
     * Python:
     *   ov_flat (3*n, naux) = ovL_ijk.reshape(3*n, naux)
     *   oo_flat (3*m_dom, naux) = ooL.reshape(3*m_dom, naux)
     *   K_ooov_pre = ov_flat @ oo_flat.T  → (3*n, 3*m_dom)
     *   K_ooov = K_ooov_pre.reshape(3, n, 3, m_dom).transpose(0, 2, 1, 3)
     *   → K_ooov[p, q, a, m]
     */
    int m_dom_size = n_dom;
    ENSURE(K_ooov, double, (size_t)3 * 3 * n * (m_dom_size > 0 ? m_dom_size : 1));
    double *K_ooov = tscratch.K_ooov;
    if (m_dom_size > 0) {
        int M = 3 * n, N = 3 * m_dom_size, K = naux_ijk;
        ENSURE(K_pre, double, (size_t)M * N);
        double *K_pre = tscratch.K_pre;
        /* ov_flat (M, K) row-major @ oo_flat^T (K, N) row-major = (M, N).
         * row-major C = A(M, K) @ B^T(N, K).T = A @ B^T:
         *   dgemm('T','N', N, M, K, 1, B (LDB=K), A (LDA=K), 0, C (LDC=N))
         */
        dgemm_(&Tc, &Nc, &N, &M, &K,
               &one, ooL_sc, &K,
               ovL_sc, &K,
               &zero, K_pre, &N);
        /* Now K_pre (3*n, 3*m_dom) row-major. Reshape to (3, n, 3, m_dom):
         *   K_pre[p*n + a, q*m_dom + m] → 4D[p, a, q, m]
         * Then transpose(0, 2, 1, 3) → K_ooov[p, q, a, m] = 4D[p, a, q, m].
         * So K_ooov[p, q, a, m] = K_pre[(p*n + a) * (3*m_dom) + q*m_dom + m].
         */
        for (int p = 0; p < 3; p++) {
            for (int q = 0; q < 3; q++) {
                for (int a = 0; a < n; a++) {
                    for (int m = 0; m < m_dom_size; m++) {
                        K_ooov[((((size_t)p * 3 + q) * n + a) * m_dom_size) + m]
                            = K_pre[((size_t)p * n + a) * 3 * m_dom_size
                                    + q * m_dom_size + m];
                    }
                }
            }
        }
    }

    /* K_jk, K_ik, K_ij (n, n) for V intermediate when has_t1.
     * Python: K_jk = ovL[1] @ ovL[2].T, etc.
     */
    ENSURE(K_jk, double, (size_t)n * n);
    ENSURE(K_ik, double, (size_t)n * n);
    ENSURE(K_ij, double, (size_t)n * n);
    double *K_jk = tscratch.K_jk;
    double *K_ik = tscratch.K_ik;
    double *K_ij = tscratch.K_ij;
    memset(K_jk, 0, sizeof(double) * (size_t)n * n);
    memset(K_ik, 0, sizeof(double) * (size_t)n * n);
    memset(K_ij, 0, sizeof(double) * (size_t)n * n);
    if (has_t1) {
        const double *ovL_0 = ovL_sc + 0 * (size_t)n * naux_ijk;
        const double *ovL_1 = ovL_sc + 1 * (size_t)n * naux_ijk;
        const double *ovL_2 = ovL_sc + 2 * (size_t)n * naux_ijk;
        int int_naux = naux_ijk;
        /* row-major C(n, n) = A(n, naux) @ B(n, naux)^T:
         *   dgemm('T','N', n, n, naux, 1, B (LDB=naux), A (LDA=naux), 0, C (LDC=n))
         */
        dgemm_(&Tc, &Nc, &int_n, &int_n, &int_naux,
               &one, ovL_2, &int_naux, ovL_1, &int_naux,
               &zero, K_jk, &int_n);
        dgemm_(&Tc, &Nc, &int_n, &int_n, &int_naux,
               &one, ovL_2, &int_naux, ovL_0, &int_naux,
               &zero, K_ik, &int_n);
        dgemm_(&Tc, &Nc, &int_n, &int_n, &int_naux,
               &one, ovL_1, &int_naux, ovL_0, &int_naux,
               &zero, K_ij, &int_n);
    }

    /* t1_lmo (3, n) — project per-LMO t1 to TNO basis via diag pair U.
     * t1_lmo[r] = U_diag_r.T @ t1_pno[lmo_r].
     */
    ENSURE(t1_lmo, double, (size_t)3 * n);
    double *t1_lmo = tscratch.t1_lmo;
    memset(t1_lmo, 0, sizeof(double) * (size_t)3 * n);
    if (has_t1) {
        for (int r = 0; r < 3; r++) {
            int u_idx = t1_diag_u_pk_idx[r];
            if (u_idx < 0) continue;
            int n_pno = u_pno_n[u_idx];
            long lmo_r = t1_lmo_idx[r];
            const double *t1_r = t1_flat + t1_off[lmo_r];
            const double *U_rr = U_flat_cache + U_off_cache[u_idx];
            /* t1_lmo[r, a] = sum_p U_rr[p, a] * t1_r[p]
             * row-major: out (n,) = U_rr.T (n, n_pno) @ t1_r (n_pno,)
             * Use dgemv_ — simpler. */
            int int_n_pno = n_pno;
            extern void dgemv_(const char *, const int *, const int *,
                                const double *, const double *, const int *,
                                const double *, const int *,
                                const double *, double *, const int *);
            int one_inc = 1;
            const char Tc_v = 'T';
            dgemv_(&Tc_v, &int_n_pno, &int_n,
                   &one, U_rr, &int_n_pno,
                   t1_r, &one_inc,
                   &zero, t1_lmo + (size_t)r * n, &one_inc);
            /* Wait — dgemv computes y = α op(A) x + β y.  The row-major
             * U_rr as col-major (n_tno, n_pno)? With LDA, etc., it's confusing.
             *
             * Simpler approach: hand-roll the matvec.
             */
            for (int a = 0; a < n; a++) {
                double s = 0.0;
                for (int p = 0; p < n_pno; p++) {
                    s += U_rr[(size_t)p * n + a] * t1_r[p];
                }
                t1_lmo[(size_t)r * n + a] = s;
            }
        }
    }

    /* t2_T_all (3, 3, n, n) = t2_block.transpose(0, 1, 3, 2):
     *   t2_T_all[ir, iq, c, f] = t2_block[ir, iq, f, c].
     */
    ENSURE(t2_T_all, double, (size_t)9 * n * n);
    double *t2_T_all = tscratch.t2_T_all;
    for (int ir = 0; ir < 3; ir++) {
        for (int iq = 0; iq < 3; iq++) {
            const double *src = t2_block + (size_t)(ir * 3 + iq) * n * n;
            double *dst = t2_T_all + (size_t)(ir * 3 + iq) * n * n;
            for (int a = 0; a < n; a++) {
                for (int b = 0; b < n; b++) {
                    dst[(size_t)a * n + b] = src[(size_t)b * n + a];
                }
            }
        }
    }

    /* Build per-task offsets into the existing U_flat_cache and u_T2_flat
     * arrays — NO data copy.  W3 kernel reads U_flat_cache[w3_U_off[t]:...]
     * and u_T2_flat[w3_T2_off[t]:...] directly.
     *
     * (Earlier version copied U/T2 into a per-task contiguous arena, which
     * doubled memory traffic per triple and dominated the wall.)
     */
    int n_w3 = 3 * m_dom_size + 1;
    ENSURE(w3_n_pno_arr, long, n_w3);
    ENSURE(w3_U_off, long, n_w3);
    ENSURE(w3_T2_off, long, n_w3);
    ENSURE(w3_tflags, signed char, n_w3);
    long *w3_n_pno_arr = tscratch.w3_n_pno_arr;
    long *w3_U_off = tscratch.w3_U_off;
    long *w3_T2_off = tscratch.w3_T2_off;
    signed char *w3_tflags = tscratch.w3_tflags;
    memset(w3_n_pno_arr, 0, sizeof(long) * n_w3);
    memset(w3_U_off, 0, sizeof(long) * n_w3);
    memset(w3_T2_off, 0, sizeof(long) * n_w3);
    memset(w3_tflags, 0, n_w3);
    int n_pno_max_w3 = 0;
    for (int t = 0; t < 3 * m_dom_size; t++) {
        int u_idx = w3_u_pk_idx[t];
        if (u_idx < 0) {
            w3_n_pno_arr[t] = 0;
            w3_U_off[t]  = 0;   /* unused when n_pno=0 */
            w3_T2_off[t] = 0;
        } else {
            w3_n_pno_arr[t] = u_pno_n[u_idx];
            w3_U_off[t]  = U_off_cache[u_idx];
            w3_T2_off[t] = u_T2_off[u_idx];
            if (u_pno_n[u_idx] > n_pno_max_w3) n_pno_max_w3 = u_pno_n[u_idx];
        }
        w3_tflags[t] = w3_transpose[t];
    }

    /* eps_occ */
    double eps_occ_arr[3] = {eps_i, eps_j, eps_k};

    /* Call W3.  Pass U_flat_cache and u_T2_flat directly with per-task
     * offsets (w3_U_off / w3_T2_off index INTO those buffers). */
    double et_ijk = DLPNOcompute_w3_energy(
        K_ab_cache, t2_T_all,
        K_jk, K_ik, K_ij, K_ooov,
        U_flat_cache, w3_U_off, w3_n_pno_arr,
        u_T2_flat,    w3_T2_off, w3_tflags,
        eps_occ_arr, eps_tno,
        t1_lmo,
        has_t1, occ_denom,
        n, m_dom_size, n_pno_max_w3);

    /* All scratch buffers persist in __thread storage — no per-triple free.
     * Only the heap-allocated TNO outputs get freed (those come from
     * DLPNObuild_triple_tno_full's internal mallocs). */
    free(X_tno_ijk); free(eps_tno);

    return et_ijk;
}
