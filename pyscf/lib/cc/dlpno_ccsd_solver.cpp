/*
 * DLPNO-CCSD monolithic C++ solver — skeleton (Step 1 / 7).
 *
 * Mission: a single C++ class that owns all CCSD state and runs the entire
 * cycle loop in C++, mirroring Psi4's DLPNOCCSD::lccsd_iterations
 * (psi4_jiang/psi4/src/psi4/dlpno/ccsd.cc:2134).  Eliminates the per-phase
 * Python/C ping-pong that bounded our piecemeal kernel ports at ~38s on
 * water-10.
 *
 * This file is the SKELETON only — phase methods are empty.  Built and loaded
 * to validate ctypes wiring before any behavioral code lands.  See
 * HANDOFF_CCSD_C_CLASS.md for the full plan.
 *
 * Single Python entry point: DLPNOcompute_lccsd_omp.  Returns -1 (sentinel
 * for "skeleton; not converged") so callers fall back to the legacy path.
 */

#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Forward decls of per-pair kernels implemented in dlpno_*.c (compiled as C;
// declared here as extern "C" to suppress C++ mangling).
// BLAS prototypes (fortran ABI).
extern "C" {
void dgemm_(const char *transa, const char *transb,
            const int *m, const int *n, const int *k,
            const double *alpha, const double *A, const int *lda,
            const double *B, const int *ldb,
            const double *beta, double *C, const int *ldc);
void dgemv_(const char *trans, const int *m, const int *n,
            const double *alpha, const double *A, const int *lda,
            const double *x, const int *incx,
            const double *beta, double *y, const int *incy);
}

extern "C" void DLPNOt1_ints_pair_side(
    double *Qa_t1_out, double *Qk_t1_out,
    const double *Qa_full, const double *Qk_local,
    const double *Qma, const double *Qab,
    const double *t1_lmo, const double *T1_local,
    size_t n_local, size_t nlmo, size_t npno);

extern "C" void DLPNOcompute_B_tilde_pair(
    double *B_out,
    const double *i_Qk_t1, const double *j_Qk_t1,
    const double *Qma, const double *T2,
    size_t n_local, size_t nlmo, size_t npno);

extern "C" void DLPNOper_i_stages123(
    double *r1_inout,
    const double *Qma, const double *Qab,
    const double *Qia, const double *Qik,
    const double *T_n, const double *t1_i,
    const double *e_pno,
    int do_stage_23,
    size_t L, size_t M, size_t A);

extern "C" void DLPNOcompute_D_tilde_ph1_batched(
    const double *K_tilde_chem_flat, const long *K_tilde_chem_offsets,
    const double *M_static_flat,     const long *M_static_offsets,
    const double *t1_flat,           const long *t1_offsets,
    const double *T1_rows_flat,      const long *T1_rows_offsets,
    const int    *n_pno_arr,
    const int    *n_domain_arr,
    double       *D_flat,            const long *D_offsets,
    size_t        N);

extern "C" void DLPNOcompute_C_tilde_ph1_batched(
    const double *K_tilde_chem_flat,       const long *K_tilde_chem_offsets,
    const double *K_bar_chem_slice_flat,   const long *K_bar_chem_slice_offsets,
    const double *t1_ki_flat,              const long *t1_ki_offsets,
    const double *T1_local_flat,           const long *T1_local_offsets,
    const int    *n_pno_arr,
    const int    *n_domain_arr,
    double       *C_flat,                  const long *C_offsets,
    size_t        N);

extern "C" void DLPNOcompute_G_tilde_inner(
    const long *triple_eff_offset,
    const long *triple_T2_pair_idx,
    const int  *triple_n_lj,
    const long *ij_triple_starts,
    const int  *ij_i_arr,
    const int  *ij_j_arr,
    const double *effective_flat,
    const double *T2_flat,
    const long  *T2_offsets,
    double      *G_addition,
    size_t n_ij_slots,
    size_t naocc);

extern "C" void DLPNObe_kernel(
    const double *S, const double *T, const double *K,
    const double *beta_kl, const double *beta_lk,
    const unsigned char *same,
    const long *idx,
    double *out_B, double *out_E,
    size_t N, size_t n_ij, size_t n_kl,
    int num_threads);

extern "C" void DLPNObe_kernel_v3(
    const double *S, const double *T, const double *K,
    const double *beta_kl, const double *beta_lk,
    const unsigned char *same,
    const long *idx,
    double *out_B, double *out_E,
    size_t N, size_t n_ij, size_t n_kl,
    int num_threads, size_t n_slots);

extern "C" void DLPNObe_kernel_gathered(
    const double *S_master, const long *S_off,
    const double *T_master, const long *T_off,
    const double *K_master, const long *K_off,
    const double *beta_kl, const double *beta_lk,
    const unsigned char *same,
    const long *idx,
    double *out_B, double *out_E,
    size_t N, size_t n_ij, size_t n_kl,
    int num_threads);

extern "C" void DLPNOc_term_batched(
    int N,
    const int  *n_pno_arr, const int *n_ct_arr, const int *n_other_arr,
    const long *S_big_off, const long *ct_off, const long *S_mid_off,
    const long *J_bold_off, const long *t2_off, const long *S_outer_off,
    const long *tile_off,
    const double *S_big_flat, const double *S_mid_flat,
    const double *J_bold_flat, const double *S_outer_flat,
    const double *ct_flat, const double *t2_flat,
    const unsigned char *t2_trans,
    double *STB_scratch,   size_t STB_stride,
    double *GAMMA_scratch, size_t GAMMA_stride,
    double *GT_scratch,    size_t GT_stride,
    double *tiles_flat,
    int num_threads);

extern "C" void DLPNOt3_kernel_batched(
    int N,
    const int  *n_kl_arr, const int *n_ki_arr,
    const long *K_off, const long *S_off,
    const long *t1i_off, const long *T1l_off, const long *tile_off,
    const double *K_flat, const double *S_flat, const double *t1_flat,
    double *Kt1_scratch,    size_t Kt1_stride,
    double *Kt1_ki_scratch, size_t Kt1_ki_stride,
    double *tiles_flat,
    int num_threads);

extern "C" void DLPNOt4_kernel_batched(
    int N,
    const int  *n_ki_arr, const int *n_li_arr, const int *n_kl_arr,
    const long *S_ki_li_off, const long *t2_off, const long *S_li_kl_off,
    const long *K_off, const long *S_kl_ki_off, const long *tile_off,
    const double *S_ki_li_flat, const double *S_li_kl_flat,
    const double *K_flat, const double *S_kl_ki_flat,
    const double *t2_flat,
    double *tmp1_scratch, size_t tmp1_stride,
    double *tmp2_scratch, size_t tmp2_stride,
    double *tmp3_scratch, size_t tmp3_stride,
    double *tiles_flat,
    double scale,
    int num_threads);

extern "C" void DLPNOg_term_batched(
    int N,
    const int  *n_ij_arr, const int *n_ik_arr,
    const long *S_off, const long *t2_off, const long *tile_off,
    const long *k_idx, const long *scalar_lmo,
    const double *S_flat, const double *t2_flat,
    const double *G_tilde, size_t G_stride,
    double *tmp_scratch, size_t tmp_stride,
    double *tiles_flat,
    int num_threads);

extern "C" void DLPNOd_term_batched(
    int N,
    const int  *n_pno_arr, const int *n_A_arr, const int *n_B_arr,
    const long *S_a_off, const long *u_off, const long *S_b_off,
    const long *S_c_off, const long *dt_off, const long *KJ_off,
    const long *tile_off,
    const double *S_a_flat, const double *S_b_flat, const double *S_c_flat,
    const double *KJ_flat, const double *u_flat, const double *dt_flat,
    const double *u_base, const long *u_canon_off, const unsigned char *u_trans,
    double *U_scratch,    size_t U_stride,
    double *SU_scratch,   size_t SU_stride,
    double *UP_scratch,   size_t UP_stride,
    double *SCD_scratch,  size_t SCD_stride,
    double *Bint_scratch, size_t Bint_stride,
    double *tiles_flat,
    int num_threads);

extern "C" void DLPNOper_kl_batched(
    int n_tasks, int M,
    const int  *n_kl_arr,
    const int  *t2_swap_kl,
    const long *K_iajb_kl_off,
    const long *K_bar_kl_off,
    const long *t2_kl_canon_off,
    const long *T_n_kl_off,
    const long *inner_off,
    const int  *i_arr,
    const int  *n_pno_ii_arr,
    const int  *is_diag_kl_ii,
    const int  *has_S_ii_kl,
    const long *S_ii_kl_off,
    const int  *has_A2,
    const int  *is_diag_kl_ki,
    const int  *n_ki_arr,
    const int  *t2_swap_ki,
    const long *t2_ki_canon_off,
    const long *S_kl_ki_off,
    const long *S_ki_kl_off,
    const long *T_n_l_ii_off,
    const long *contrib_off,
    const double *K_iajb_buffer,
    const double *K_bar_kl_static,
    const double *S_pno_buffer,
    const double *t2_buffer,
    const double *t1_cache_buffer,
    double *Tt_kl_scratch,    size_t Tt_kl_stride,
    double *K_kilc_scratch,   size_t K_kilc_stride,
    double *B_ia_scratch,     size_t B_ia_stride,
    double *Tt_ki_scratch,    size_t Tt_ki_stride,
    double *X_scratch,        size_t X_stride,
    double *Z_scratch,        size_t Z_stride,
    double *contrib_flat,
    int num_threads);

extern "C" void DLPNOt1_fock_batched(
    const double *T1_flat,        const long *T1_offsets,
    const double *K_chem_flat,    const long *K_chem_offsets,
    const double *K_ji_flat,      const long *K_ji_offsets,
    const double *K_ij_flat,      const long *K_ij_offsets,
    const double *Qma_flat,       const long *Qma_offsets,
    const double *Qab_flat,       const long *Qab_offsets,
    const double *e_pno_flat,     const long *e_pno_offsets,
    const int    *nlmo_arr,
    const int    *npno_arr,
    const int    *n_local_arr,
    const int    *need_dji_arr,
    const unsigned char *is_strong_pair,  // (N,) byte flag; nullable
    double       *gamma_scratch,    size_t gamma_sc_stride,
    double       *Y_trans_scratch,  size_t Y_trans_sc_stride,
    double       *Y_alt_scratch,    size_t Y_alt_sc_stride,
    double       *Fia_scratch,      size_t Fia_sc_stride,
    double       *Z_stacked_scratch, size_t Z_stacked_sc_stride,
    double       *Z_xxx_scratch,    size_t Z_xxx_sc_stride,
    double       *d_flat,
    double       *Fab_flat,         const long *Fab_offsets,
    size_t        N,
    int           num_threads);

namespace pyscf_dlpno_ccsd {

// -- OpenMP team size for the per-phase parallel-over-pairs regions ----------
//
// The DLPNO driver runs with OMP_NUM_THREADS=1 so the Python ThreadPool
// dispatching cc_ints / (T) is not oversubscribed by BLAS.  Around the CCSD
// cycle loop ONLY, the Python side raises the OpenMP team via threadpoolctl
// (DLPNO_CCSD_CYCLE_OMP) — the pool is idle there, so the cycle's own
// parallel regions can use the whole machine.  omp_get_max_threads() thus
// already reflects that raised team.  This helper additionally clamps it to
// DLPNO_SOLVER_MAX_THREADS when that env var is set — a benchmarking / safety
// knob.  Default cap 64 (machine has 64 physical cores; hyperthreads past
// that thrash a bandwidth-bound DGEMM kernel).
static int solver_team_size() {
#ifdef _OPENMP
    int team = omp_get_max_threads();
    int cap = 64;
    const char *env = std::getenv("DLPNO_SOLVER_MAX_THREADS");
    if (env != nullptr && env[0] != '\0') {
        int v = std::atoi(env);
        if (v > 0) cap = v;
    }
    return (team < cap) ? team : cap;
#else
    return 1;
#endif
}

// -- Pair-axis flat tensor view (non-owning) ---------------------------------
//
// Mirrors the layout introduced in the cc_ints storage refactor + Phase III
// (project_psi4_port_phase3_done.md): one contiguous data buffer plus an
// offsets array of length n_pairs+1 such that pair `p` lives at
// data[offsets[p] : offsets[p+1]].  Strides are encoded by the kernel — we
// just hand the kernel the pointer and let it interpret the per-pair shape
// using n_pno_per_pair / pair_lmo_idx_offsets.
struct FlatPairStore {
    const double *data;      // storage for all pairs (may be a SHARED buffer)
    const int64_t *offsets;  // length n_pairs + 1; offsets[p+1]-offsets[p] is
                             // the SIZE of pair p (always valid for sizing)
    // Optional: when non-NULL, pair p's data START is data[block_start[p]]
    // instead of data[offsets[p]].  This lets `data` alias a buffer whose
    // pair layout differs from this store's pack order (e.g. the cc_ints
    // flat store in canonical order vs the solver's diag-first order) — a
    // true zero-copy view, no per-pair re-copy.  NULL = contiguous (legacy).
    const int64_t *block_start;
};

// Start pointer of pair `p` in a FlatPairStore: block_start[p] when set
// (aliased/shared buffer), else offsets[p] (contiguous).  Size is always
// offsets[p+1]-offsets[p].
static inline const double *fps_ptr(const FlatPairStore &s, int64_t p) {
    return s.data + (s.block_start ? s.block_start[p] : s.offsets[p]);
}

// Position-offset array to hand a batched C kernel that does its own
// `data + offs[p]` indexing: block_start when aliased (canonical positions),
// else offsets (contiguous).  The kernel derives per-pair SIZE from
// n_local/npno, not from this array, so passing positions here is correct.
static inline const long *fps_pos(const FlatPairStore &s) {
    return (const long *)(s.block_start ? s.block_start : s.offsets);
}

// -- Inputs from Python (read-only views unless noted) -----------------------
struct SolverInputs {
    // sizes
    int nocc;
    int nlmo;
    int n_canon_pairs;
    int n_strong_pairs;
    int diis_max_vecs;
    int max_cycle;
    double e_conv;
    double r_conv;

    // sparsity (read-only views)
    const int *i_j_to_ij;            // nocc * nocc; -1 if not significant
    const int *ij_to_i_j;             // 2 * n_canon_pairs (i, j packed)
    const int *ij_to_ji;              // n_canon_pairs
    const int *pair_lmo_idx_flat;     // pair-domain LMO indices, flattened
    const int64_t *pair_lmo_idx_offsets;  // length n_canon_pairs + 1
    const int *n_pno_per_pair;        // length n_canon_pairs
    const int64_t *pno_offsets;       // length n_canon_pairs + 1 (T1/fov axis)
    const int64_t *t2_offsets;        // length n_canon_pairs + 1; per-pair npno_p^2

    // orbital data (read-only views)
    const double *F_lmo;              // nocc * nocc
    const double *eps_lmo;            // nocc
    const double *foo;                // nocc * nocc (T2-dressed)
    const double *fov_flat;           // pno_offsets[nocc] doubles (per-i)
    const double *e_pno_flat;         // pno_offsets[n_canon_pairs] doubles

    // cc_ints (read-only views, pair_lmo_idx-axis)
    FlatPairStore Qma, Qab;
    FlatPairStore i_Qk, j_Qk, i_Qa, j_Qa;
    FlatPairStore K_iajb, K_bar_ij, K_bar_chem;
    FlatPairStore K_bar_ji;
    FlatPairStore J_ij_kj, K_ij_kj;
    FlatPairStore L_iajb, L_bar;
    // Per canonical pair, two K_tilde_chem variants (Psi4 i/j orientations);
    // shape (npno_p, npno_p^2).  Used by C_tilde / D_tilde.
    FlatPairStore K_tilde_chem_i, K_tilde_chem_j;

    // pno overlap cache — cross-canonical S_PNO between any two canonical
    // pairs.  Full-table layout (sparse OK via zero-size entries):
    //   S_pno_offsets has length n_canon_pairs^2 + 1.
    //   block[(p_a, p_b)] starts at S_pno_data[ S_pno_offsets[p_a*n + p_b] ]
    //   block size       = S_pno_offsets[p_a*n + p_b + 1] - S_pno_offsets[p_a*n + p_b]
    //                    = npno[p_a] * npno[p_b]   (matrix shape (npno_a, npno_b))
    //   For canonical pair pairs not stored, block size is zero (offsets equal).
    // S_pno_index is reserved for a future sparse-tier optimisation; not yet
    // consumed by any phase.
    const double  *S_pno_data;
    const int64_t *S_pno_offsets;
    const int     *S_pno_index;

    // amplitudes (READ-WRITE views into Python buffers)
    double *T1_flat;                  // pno_offsets[nocc] doubles
    double *T2_flat;                  // sum_p npno_p^2 doubles

    // T1 projected into each pair's PNO basis, restricted to the pair's
    // pair_lmo_idx domain.  Per pair p: row-major (nlmo_p, npno_p) doubles
    // where row k is t1_pno[pair_lmo_idx[p][k]] projected via S_PNO into
    // pair p's PNO basis.  Built by Python today (build_t1_cache); will be
    // built inside the class once Step 2m wires the cycle loop.
    FlatPairStore T1_in_pair;

    // Full-nocc version of the same projection: per pair p, a (nocc, npno_p)
    // matrix where row k is t1_pno[k] projected to p's PNO basis (zero for
    // k not in domain).  Required by Stage 4 of the T1 residual
    // (R1[i] -= Fij_bar[:, i] @ T1_in_pair_full[(i,i)]).
    FlatPairStore T1_in_pair_full;

    // Ordered-pair sparsity (Psi4 all_pairs = canonical (a,b) ∪ (b,a)).
    // Used by C_tilde / D_tilde / G_tilde phases.  Each ordered pair maps
    // to a canonical pair via i_j_to_ij[i*nocc + k]; orientation drives
    // which K_tilde_chem / K_bar variant the kernel consumes.
    int n_ordered_pairs;
    const int *ordered_pair_i_idx;    // length n_ordered_pairs
    const int *ordered_pair_k_idx;    // length n_ordered_pairs

    // CAS injection (optional; n_cas_blocks==0 disables)
    int n_cas_blocks;
    const int *cas_block_pair;        // n_cas_blocks: canonical pair index
    const int *cas_block_offsets;     // n_cas_blocks + 1: into cas_block_data
    const double *cas_block_data;     // flat storage for the CAS T2 slice
    const int *cas_block_slice;       // n_cas_blocks * 2: (start, stop) PNO range

    // Optional Psi4-faithful override fields.  When set non-null, the
    // class consumes these directly instead of computing them locally.
    // Used to reach Psi4-exact match by including weak-pair contributions
    // not captured by the strong-pair-only t1_fock path.
    //
    // Fij_bar_full: (nocc, nocc) row-major.  Replaces the F_lmo + scatter(d_flat)
    //   snapshot that the class builds in t1_fock_finalize, with the FULL
    //   T1-dressed F_oo (strong + weak pair contributions, Psi4 ccsd.cc:1660).
    //   Used by R1 Stage 4: r1[i] -= Fij_bar_full[:, i] @ T_n_ii.
    const double *Fij_bar_full;
    // Fkc_per_ordered: FlatPairStore indexed by ORDERED pair (a,b).  Per
    //   ordered pair, length npno[i_j_to_ij[a, b]].  Replaces the LT1-chain
    //   inline build in run_phase_t1_residual_AC_init_into.  Built per Psi4
    //   ccsd.cc:1683-1692 with full LMO-domain scope (strong + weak).
    FlatPairStore Fkc_per_ordered;
    // R2_external: precomputed R2 contributions externally (e.g. by
    //   PySCF's full residual machinery).  Length t2_offsets[n_canon_pairs].
    //   When non-null, run_one_cycle uses this *as* R2 (instead of K+A
    //   skeleton).  Used during the R2 plan-extraction transition: lets
    //   the class consume PySCF's full R2 to validate the orchestration +
    //   update_amps + energy formula at machine precision.  Once all R2
    //   plans (BE/CD/G_term/t3/t4) are extracted natively, this becomes
    //   redundant.
    const double *R2_external;
    // is_strong_pair: byte flag per canonical pair (length n_canon_pairs).
    //   1 = strong pair (counted in correlation energy formula);
    //   0 = weak pair (T2 update applied but pair excluded from energy).
    //   Mirrors PySCF's `strong_pairs` filter at line 2770 in lccsd.py.
    //   When null, all pairs counted (legacy behavior).
    const unsigned char *is_strong_pair;
};

// S_PNO block lookup for dense pair index s_idx (= p_a*N_canon + p_b).
// When S_pno_index is provided (sparse mode), S_pno_data aliases the
// S_pno_cache buffer (slot order) and S_pno_offsets are its slot offsets:
// s_idx -> slot -> (offset, size).  slot < 0 means the block is absent
// (size 0).  When S_pno_index is NULL (legacy dense mode), s_idx indexes
// S_pno_offsets directly.  This lets the pack alias the shared cache buffer
// instead of building a second dense copy of all S_PNO overlaps.
static inline void s_pno_lookup(const SolverInputs &in, int64_t s_idx,
                                int64_t &s_off, int64_t &s_size) {
    int64_t slot = s_idx;
    if (in.S_pno_index) {
        slot = (int64_t)in.S_pno_index[s_idx];
        if (slot < 0) { s_off = 0; s_size = 0; return; }
    }
    s_off = in.S_pno_offsets[slot];
    s_size = in.S_pno_offsets[slot + 1] - s_off;
}

// -- Output struct for the t1_ints phase (Step 2b) --------------------------
//
// Mutable counterpart to FlatPairStore: caller pre-allocates four flat
// buffers with offsets indexing per-pair output slices.  Per pair p:
//   i_Qa_t1[p] : (n_local_p, npno_p) doubles
//   i_Qk_t1[p] : (n_local_p, nlmo_p) doubles
//   j_Qa_t1[p] : (n_local_p, npno_p) doubles
//   j_Qk_t1[p] : (n_local_p, nlmo_p) doubles
struct WritablePairStore {
    double *data;
    const int64_t *offsets;
};

struct T1IntsOutputs {
    WritablePairStore i_Qa_t1, j_Qa_t1, i_Qk_t1, j_Qk_t1;
};

// -- B_tilde phase (Step 2c) -------------------------------------------------
//
// Inputs: i_Qk_t1, j_Qk_t1 from phase_t1_ints output (each per pair shape
// (n_local_p, nlmo_p)).  Output per pair: (nlmo_p, nlmo_p) B_tilde matrix.
struct BTildeInputs {
    FlatPairStore i_Qk_t1;
    FlatPairStore j_Qk_t1;
};

struct BTildeOutputs {
    WritablePairStore B_tilde;
};

// -- t1_fock phase (Step 2d) -------------------------------------------------
//
// No phase-specific input struct: t1_fock reads everything from SolverInputs
// (T1_in_pair, K_bar_chem, K_bar_ij, K_bar_ji, Qma, Qab, e_pno_flat).
//
// Outputs:
//   Fab    : per pair (npno_p, npno_p)            via Fab_offsets (== t2_offsets)
//   d_flat : (n_canon_pairs * 2) — per pair (d_ij, d_ji); caller scatters into
//            Fij_bar in a later phase.
struct T1FockOutputs {
    WritablePairStore Fab;
    double *d_flat;
};

// -- t1_fock finalize (Step 2h): Fkj + Fij_bar + foo_t1 -----------------------
//
// Consumes `d_flat` from a prior `run_phase_t1_fock_into` call (Step 2d).
// Produces:
//   Fkj             : (nocc, nocc) F_lmo + scatter(d_flat) + Eq 94 contribution
//   Fij_bar_snapshot: (nocc, nocc) snapshot AFTER d-scatter, BEFORE Eq 94
//                     (used by the T1 residual to avoid recomputing d).
//   foo_t1          : (nocc, nocc) Fkj - F_lmo
struct T1FockExtraInputs {
    const double *d_flat;     // length 2 * n_canon_pairs (from Step 2d)
};

struct T1FockExtraOutputs {
    double *Fkj;
    double *Fij_bar_snapshot;
    double *foo_t1;
};

// -- T1 residual per-(k, l) batched B + A2 phase (Step 2i) -------------------
//
// Wraps DLPNOper_kl_batched.  Plan provides ALL inputs (buffers + metadata);
// class only sizes and allocates scratch arenas.  Plan-builder integration
// (and binding `t2_buffer` / `t1_cache_buffer` to SolverInputs) lands in 2l.
struct PerKlPlanInputs {
    int n_tasks;
    int M;                            // = nocc

    // Per-task arrays (length n_tasks).
    const int  *n_kl_arr;
    const int  *t2_swap_kl;
    const long *K_iajb_kl_off;
    const long *K_bar_kl_off;
    const long *t2_kl_canon_off;
    const long *T_n_kl_off;
    const long *inner_off;            // length n_tasks + 1

    // Per-(task, inner_i) arrays (length total_inner = inner_off[n_tasks]).
    const int  *i_arr;
    const int  *n_pno_ii_arr;
    const int  *is_diag_kl_ii;
    const int  *has_S_ii_kl;
    const long *S_ii_kl_off;
    const int  *has_A2;
    const int  *is_diag_kl_ki;
    const int  *n_ki_arr;
    const int  *t2_swap_ki;
    const long *t2_ki_canon_off;
    const long *S_kl_ki_off;
    const long *S_ki_kl_off;
    const long *T_n_l_ii_off;
    const long *contrib_off;          // length total_inner; [ti] = start of contrib[ti]

    // Static + dynamic buffers (plan-owned).
    const double *K_iajb_buffer;
    const double *K_bar_kl_static;
    const double *S_pno_buffer;
    const double *t2_buffer;
    const double *t1_cache_buffer;

    // Scratch sizing bounds.
    int max_n_kl;
    int max_n_ki;
};

struct PerKlOutputs {
    double *contrib_flat;
};

// -- R2 B + E term per-bucket (Step 2j-a) ------------------------------------
//
// Wraps DLPNObe_kernel.  The Python wrapper buckets items by (n_ij, n_kl)
// shape and calls the kernel once per bucket; we mirror that — phase entry
// processes one bucket per call.  Kernel mallocs its own internal scratch.
struct BEInputs {
    int N;                // items in this bucket
    int n_ij;
    int n_kl;
    int n_slots;          // for caller-allocated output buffer sizing
    const double        *S;          // (N, n_ij, n_kl) — used when S_master is null
    const double        *T;          // (N, n_kl, n_kl) — used when T_master is null
    const double        *K;          // (N, n_kl, n_kl) — used when K_master is null
    const double        *beta_kl;    // (N,) — refreshable; see p_ij_arr
    const double        *beta_lk;    // (N,)
    const unsigned char *same;       // (N,)
    const long          *idx;        // (N,)
    // Native B_tilde refresh: when p_ij_arr != nullptr, run_one_cycle
    // overwrites beta_kl[n] and beta_lk[n] from the freshly-built
    // B_tilde_flat after Phase 5.  beta_kl/lk arrays must be writable
    // by the caller; we cast away const at the refresh site.
    const int *p_ij_arr;             // (N,) source pair index
    const int *dense_k_arr;          // (N,) k row index in B_tilde_flat[p]
    const int *dense_l_arr;          // (N,) l col index in B_tilde_flat[p]
    // Memory-light "gathered" mode (DLPNO_BE_GATHERED=1): when all three
    // master pointers are non-null, BE reads S/T/K from caller-owned flats
    // via per-item element offsets instead of stacked per-bucket copies.
    // Eliminates ~1.5 GB/water-22 of redundant slice copies in _inv_cache.
    const double        *S_master;
    const double        *T_master;
    const double        *K_master;
    const long          *S_off;      // (N,) element offset into S_master
    const long          *T_off;      // (N,) element offset into T_master
    const long          *K_off;      // (N,) element offset into K_master
};

struct BEOutputs {
    double *out_B;        // (n_slots, n_ij, n_ij)
    double *out_E;        // (n_slots, n_ij, n_ij)
};

// -- R2 C-term batched (Step 2j-b1) ------------------------------------------
struct CTermInputs {
    int N;
    const int  *n_pno_arr;       // (N,)
    const int  *n_ct_arr;        // (N,)
    const int  *n_other_arr;     // (N,)
    const long *S_big_off;       // (N,)
    const long *ct_off;          // (N,)
    const long *S_mid_off;       // (N,)
    const long *J_bold_off;      // (N,)
    const long *t2_off;          // (N,)
    const long *S_outer_off;     // (N,)
    const long *tile_off;        // (N,)
    const double *S_big_flat;
    const double *S_mid_flat;
    const double *J_bold_flat;
    const double *S_outer_flat;
    const double *ct_flat;
    const double *t2_flat;
    const unsigned char *t2_trans;   // NULL => legacy gathered t2 (flag T)
    int max_n_pno;
    int max_n_ct;
    int max_n_other;
};

struct CTermOutputs {
    double *tiles_flat;  // length = sum of n_pno_arr[n]^2
};

// -- R2 D-term batched (Step 2j-b2) ------------------------------------------
struct DTermInputs {
    int N;
    const int  *n_pno_arr;       // (N,)
    const int  *n_A_arr;         // (N,)
    const int  *n_B_arr;         // (N,)
    const long *S_a_off;
    const long *u_off;
    const long *S_b_off;
    const long *S_c_off;
    const long *dt_off;
    const long *KJ_off;
    const long *tile_off;
    const double *S_a_flat;
    const double *S_b_flat;
    const double *S_c_flat;
    const double *KJ_flat;
    const double *u_flat;
    const double *dt_flat;
    const double *u_base;            // NULL => legacy gathered u_flat
    const long   *u_canon_off;       // canonical t2[key] offsets (when u_base)
    const unsigned char *u_trans;    // per-item transpose flag (when u_base)
    int max_n_pno;
    int max_n_A;
    int max_n_B;
};

struct DTermOutputs {
    double *tiles_flat;  // length = sum of n_pno_arr[n]^2
};

// -- R2 G-term batched (Step 2j-c) ------------------------------------------
//
// Wraps DLPNOg_term_batched.  Per-item math: scalar = G_tilde[k_idx[n],
// scalar_lmo[n]]; tmp = S @ t2; Cc = scalar * tmp @ S.T.  Naturally chains
// with phase 2g's output (G_tilde matrix).
struct GTermInputs {
    int N;
    const int  *n_ij_arr;        // (N,)
    const int  *n_ik_arr;        // (N,)
    const long *S_off;           // (N,)
    const long *t2_off;          // (N,)
    const long *tile_off;        // (N,)
    const long *k_idx;           // (N,) — row index into G_tilde
    const long *scalar_lmo;      // (N,) — col index into G_tilde
    const double *S_flat;
    const double *t2_flat;
    const double *G_tilde;       // (G_stride, G_stride) — typically nocc x nocc
    int G_stride;                // = nocc
    int max_n_ij;
    int max_n_ik;
};

struct GTermOutputs {
    double *tiles_flat;          // length = sum of n_ij_arr[n]^2
};

// -- C/D Phase 2 t3 + t4 batched (Step 2j-d) --------------------------------
//
// t3: contrib = -T1l ⊗ (S @ K.T @ t1i)  (rank-1 outer product per item).
// t4: contrib = scale * S_ki_li @ t2 @ S_li_kl @ K @ S_kl_ki  (4 matmuls).
struct T3Inputs {
    int N;
    const int  *n_kl_arr;        // (N,)
    const int  *n_ki_arr;        // (N,)
    const long *K_off;
    const long *S_off;
    const long *t1i_off;         // offsets into t1_flat for the t1i vectors
    const long *T1l_off;         // offsets into t1_flat for the T1l vectors
    const long *tile_off;
    const double *K_flat;
    const double *S_flat;
    const double *t1_flat;       // shared buffer for both t1i and T1l
    int max_n_kl;
    int max_n_ki;
};

struct T3Outputs {
    double *tiles_flat;          // length = sum of n_ki_arr[n]^2
};

struct T4Inputs {
    int N;
    const int  *n_ki_arr;        // (N,)
    const int  *n_li_arr;        // (N,)
    const int  *n_kl_arr;        // (N,)
    const long *S_ki_li_off;
    const long *t2_off;
    const long *S_li_kl_off;
    const long *K_off;
    const long *S_kl_ki_off;
    const long *tile_off;
    const double *S_ki_li_flat;
    const double *S_li_kl_flat;
    const double *K_flat;
    const double *S_kl_ki_flat;
    const double *t2_flat;
    double scale;
    int max_n_ki;
    int max_n_li;
    int max_n_kl;
};

struct T4Outputs {
    double *tiles_flat;          // length = sum of n_ki_arr[n]^2
};

// -- R2 K + ladder (Step 2j-e) -----------------------------------------------
//
// No existing C kernel — math is small per pair, written inline in the class.
// Per canonical pair (i, j):
//   K[a, b]      = Σ_Q i_Qa_t1[Q, a] * j_Qa_t1[Q, b]
//   For each Q:
//     Qab_t1[a, b] = Qab[Q, a, b] - Σ_n T1_in_pair[n, a] * Qma[Q, n, b]
//     A[a, b]     += Σ_{c, d} Qab_t1[a, c] * T2[c, d] * Qab_t1[b, d]
// Both K and A share the per-pair (npno, npno) layout; use t2_offsets.
struct KLadderInputs {
    FlatPairStore i_Qa_t1;       // from phase_t1_ints output
    FlatPairStore j_Qa_t1;
};

struct KLadderOutputs {
    WritablePairStore K;         // per pair (npno_p, npno_p)
    WritablePairStore A;         // per pair (npno_p, npno_p)
};

// -- Update amplitudes + energy (Step 2k) ------------------------------------
//
// Combined phase: T1 -= R1 / D_T1, T2 -= R2 / D_T2 (in-place mutation of
// T1_flat / T2_flat in SolverInputs), then compute correlation energy.
// Energy = Σ_i fov[i] · T1[i] + Σ_p (1 or 2) · K_iajb[p] : (2τ - τᵀ)
//   where τ = T2[p] + t1_i ⊗ t1_j (t1 in pair p's PNO basis), and the
//   factor is 1 for diagonal pairs (i == j), 2 for off-diagonal.
struct UpdateAmpsInputs {
    const double *R1_flat;       // pno_offsets[nocc] doubles
    const double *R2_flat;       // t2_offsets[n_canon_pairs] doubles
};

struct UpdateAmpsOutputs {
    double energy;
};

// -- t1_fock Fia_bar per pair (Step 2l-b) ------------------------------------
//
// Computes Fia_bar per canonical pair from Qma + T1_in_pair (same math as
// Eq 94 in 2h, but for ALL pairs — diagonal AND off-diagonal — so callers
// can extract Fai[i] = Fia_bar[(i,i)][i_in_p, :] and
// Fkc[(k,i)] = Fia_bar[canonical_p][k_in_p, :].
struct FiaBarOutputs {
    WritablePairStore Fia_bar;   // per pair (nlmo_p, npno_p)
};

// -- R1 init + A + C contribution (Step 2l-c) --------------------------------
//
// Combined phase covering DePrince Eq 19 Term 1 (init) + Eqs 20 (A) and 22 (C).
// Iterates ordered pairs (a_ord, b_ord) — Psi4's (i, k).  Per ordered pair:
//   A: R1[a_ord] += S_PNO(p_canon, p_ii).T @ K_chem_ki @ Tt_ki.ravel()
//   C: R1[a_ord] += S_PNO(p_canon, p_ii).T @ Tt_ik @ Fkc_ki
// Init: R1[i] = Fai[i] = Fia_bar[(i,i)][i_in_p, :].
//
// Uses cross-canonical S_pno_cache (2l-a) and Fia_bar from 2l-b.  Per-thread
// R1 buffer + reduction (multiple ordered pairs scatter to the same R1[i]).
struct R1AcInputs {
    FlatPairStore Fia_bar;       // from 2l-b output
    int do_init;                 // 1 = include Fai init step (legacy); 0 = AC only
};

struct R1AcOutputs {
    double *R1_flat;             // length pno_offsets[nocc]
};

// -- D_tilde Phase 1 (Step 2e) ----------------------------------------------
//
// Iterates ordered pairs (Psi4 all_pairs).  Per ordered pair (i, k) with
// canonical key (a, b) = (min(i,k), max(i,k)):
//   K_tilde_chem = K_tilde_chem_i if a == k else K_tilde_chem_j
//   K_bar        = K_bar_ij       if a == i else K_bar_ji
//   M_static     = 2 * K_bar - K_bar_chem
//   t1[i]        = T1_in_pair[(a,b)] row at position i_in_p[(a,b), i]
//   T1_rows      = T1_in_pair[(a,b)]
// Output per ordered pair: D_tilde[(i,k)] of shape (npno_(a,b), npno_(a,b)).
struct DTildeOutputs {
    WritablePairStore D_tilde;        // length n_ordered_pairs entries
};

// -- C_tilde Phase 1 (Step 2f) ----------------------------------------------
//
// Iterates ordered pairs (Psi4 all_pairs).  Per ordered pair (a, b) with
// canonical key (ca, cb) = (min(a, b), max(a, b)) — note Psi4 wrapper
// unpacks the ordered tuple as (k, i) with k=a, i=b:
//   K_tilde_chem = K_tilde_chem_i if ca == a else K_tilde_chem_j
//   K_bar_chem   = canonical K_bar_chem (no orientation flip)
//   t1[i]        = T1_in_pair[(ca, cb)] row at position i_in_p[(ca, cb), b]
//   T1_local     = T1_in_pair[(ca, cb)]
struct CTildeOutputs {
    WritablePairStore C_tilde;
};

// -- G_tilde inner phase (Step 2g) ------------------------------------------
//
// Plan (effective_flat + triple lists + slot lists) is provided by the
// caller — it folds S_PNO(il, lj), K_iajb[il], and orientation handling
// into a per-triple precomputed tensor.  The class runs only the inner
// reduction (per (i, j) slot, accumulate Σ_t effective[t] : T2[pair[t]]).
// Plan-building moves into the class in Step 2l along with S_pno_cache
// prewarming.
struct GTildeInputs {
    int n_ij_slots;
    const long  *triple_eff_offset;       // length sum-of-triples
    const long  *triple_T2_pair_idx;      // length sum-of-triples
    const int   *triple_n_lj;             // length sum-of-triples
    const long  *ij_triple_starts;        // length n_ij_slots + 1
    const int   *ij_i_arr;                // length n_ij_slots
    const int   *ij_j_arr;                // length n_ij_slots
    const double *effective_flat;         // length determined by triple_eff_offset[last]
};

struct GTildeOutputs {
    // (nocc, nocc) row-major.  Caller pre-initializes to Fkj; the kernel
    // accumulates into it.
    double *G_tilde;
};

// -- Cycle-collapse (c-collapse-1) -------------------------------------------
//
// Walking skeleton for one full CCSD cycle inside C++.  Plan-cached phases
// are passed as nullable plan struct pointers; null = skip the phase
// (zero contribution).  Plans get filled in incrementally as plan-build
// is hoisted out of existing Python wrappers.
struct RunCycleInputs {
    const GTildeInputs    *g_tilde_plan;
    const PerKlPlanInputs *per_kl_plan;
    const BEInputs        *be_plan;
    const CTermInputs     *c_term_plan;
    const DTermInputs     *d_term_plan;
    const GTermInputs     *g_term_plan;
    const T3Inputs        *t3_plan;
    const T4Inputs        *t4_plan;

    // -- Native R2 assembly: G_term two-sided (Step 2j-c orchestration). ---
    // ik side runs g_term_plan; jk side runs g_term_plan_jk (when both
    // non-null, R2 += flat_G_ij[p] + flat_G_ji[p].T per pair).
    const GTermInputs *g_term_plan_jk;
    // Per-G_term-item canonical-pair scatter table (length g_term_plan->N
    // OR g_term_plan_jk->N respectively).  Each entry indexes into
    // [0, n_canon_pairs); item n's tile is added to flat_G[ij/ji] at
    // pair `target_pair_idx_ik[n]` / `target_pair_idx_jk[n]`.
    const int *g_term_target_pair_idx_ik;
    const int *g_term_target_pair_idx_jk;

    // -- Native R2 assembly: BE multi-bucket (Step 2j-a orchestration). ---
    // Multiple BE buckets, one per (n_ij, n_kl) shape pair.  Class iterates
    // them, calling DLPNObe_kernel per bucket with the appropriate output
    // buffer offset (buckets sharing n_ij accumulate into the same
    // flat_B/flat_E group buffer).  After all kernels run, per canonical
    // pair p we look up its (group, slot) and add flat_B[base] + flat_E[base]
    // to R2_buf[t2_offsets[p]].
    int be_n_buckets;
    const BEInputs *be_plan_buckets;          // length be_n_buckets
    int be_n_unique_n_ij;
    const int *be_unique_n_ij;                // length be_n_unique_n_ij
    const int64_t *be_flat_off_per_n_ij;      // length be_n_unique_n_ij + 1
    const int *be_pair_n_ij_idx;              // length n_canon_pairs (-1 = no BE)
    const int *be_pair_slot;                  // length n_canon_pairs

    // -- Native R2 assembly: CD (C_term + D_term, Step 2j-b). ---
    // Per-item canonical-pair scatter index, one entry per side.  Length
    // c_term_plan->N and d_term_plan->N respectively.  Item n contributes
    // to flat_C_ij[target_pair_idx_ij[n]] (when >=0) OR
    // flat_C_ji[target_pair_idx_ji[n]] (when >=0), exactly one set per item.
    const int *c_term_target_pair_idx_ij;
    const int *c_term_target_pair_idx_ji;
    const int *d_term_target_pair_idx_ij;
    const int *d_term_target_pair_idx_ji;

    // -- Native C_tilde / D_tilde Phase 2 build (t3+t4 plans). ---
    // c_t3_plan / c_t4_plan: extend C_tilde Phase 1 (already in C++ via
    //   run_phase_c_tilde_ph1_into) with Phase 2 contributions.
    // d_t3_plan / d_t4_plan: same for D_tilde.
    // Per-item scatter table (length plan->N): ordered-pair index that
    //   the kernel's tile is added to in C_tilde_flat / D_tilde_flat.
    const T3Inputs *c_t3_plan;
    const T4Inputs *c_t4_plan;
    const T3Inputs *d_t3_plan;
    const T4Inputs *d_t4_plan;
    const int *c_t3_target_ord_idx;
    const int *c_t4_target_ord_idx;
    const int *d_t3_target_ord_idx;
    const int *d_t4_target_ord_idx;
    // CD ct_flat / dt_flat native-gather: per-CD-item ordered-pair index
    // into C_tilde_flat / D_tilde_flat.  When provided, run_one_cycle
    // builds ct_flat / dt_flat from class's natively-built C_tilde_flat /
    // D_tilde_flat at the start of CD step (replacing PySCF dict gather).
    const int *c_term_ct_ord_pair_idx;
    const int *d_term_dt_ord_pair_idx;
};

struct RunCycleOutputs {
    double *R1_flat;        // pno_offsets[nocc] doubles
    double *R2_flat;        // sum_p npno_p^2 doubles
    double  energy;
    // Optional: when non-null, run_one_cycle writes the post-g_tilde_inner
    // (nocc, nocc) G_tilde matrix here for cross-validation.
    double *G_tilde_out;
};

// -- Solver class (skeleton) -------------------------------------------------
class DLPNOCCSDSolver {
   public:
    explicit DLPNOCCSDSolver(const SolverInputs &in) : in_(in) {}

    // Returns final cycle count on success, -1 if skeleton/not implemented.
    int solve();
    double get_energy() const { return e_corr_; }

    // Step 2b helper: runs ONLY the t1_ints phase, writing into caller-
    // provided output buffers.  Cross-validation entry — when phase_t1_ints_
    // becomes class-internal in Step 2m, this entry can be retired or kept
    // for per-phase parity dumps.
    void run_phase_t1_ints_into(T1IntsOutputs *out);

    // Step 2c helper: runs ONLY the B_tilde phase (jiang).  Inputs include
    // i_Qk_t1/j_Qk_t1 from a prior phase_t1_ints call; T2 + Qma come from
    // SolverInputs.
    void run_phase_b_tilde_into(const BTildeInputs *t1_dressed,
                                BTildeOutputs *out);

    // Step 2d helper: runs the t1_fock phase (per-pair Fab + d_ij/d_ji).
    // Wraps DLPNOt1_fock_batched (single batched kernel call).  Class
    // allocates the 6 scratch arenas locally per entry call until Step 2m
    // promotes them to long-lived members.
    void run_phase_t1_fock_into(T1FockOutputs *out);

    // Step 2e helper: runs the D_tilde Phase 1 (Terms 1+2) over ordered
    // pairs.  Class builds ephemeral flat buffers (K_tilde_chem chosen per
    // orientation, M_static computed inline, t1[i] gathered) then calls the
    // existing batched kernel.
    void run_phase_d_tilde_ph1_into(DTildeOutputs *out);

    // Step 2f helper: runs the C_tilde Phase 1 (Terms 1+2) over ordered
    // pairs.  Same ordered-pair iteration as 2e; orientation rules and
    // t1 source differ — see CTildeOutputs comment.
    void run_phase_c_tilde_ph1_into(CTildeOutputs *out);

    // Step 2g helper: runs the G_tilde inner kernel given a precomputed
    // plan (effective_flat from S_pno × K_iajb × orientation folding).
    // Plan-building moves into the class in Step 2l.
    void run_phase_g_tilde_inner_into(const GTildeInputs *plan,
                                       GTildeOutputs *out);

    // Step 2h helper: t1_fock finalize — Fkj scatter, Fij_bar snapshot,
    // Eq 94 (Fia_bar_jj contribution per occupied j), foo_t1 = Fkj - F_lmo.
    void run_phase_t1_fock_finalize_into(const T1FockExtraInputs *extra,
                                          T1FockExtraOutputs *out);

    // Step 2i helper: T1 residual per-(k, l) batched B + A2 contributions.
    // Wraps DLPNOper_kl_batched.  Class allocates the 6 per-thread scratch
    // arenas; plan supplies everything else.
    void run_phase_t1_residual_per_kl_into(const PerKlPlanInputs *plan,
                                            PerKlOutputs *out);

    // Step 2j-a helper: R2 B + E term per (n_ij, n_kl) bucket.  Wraps
    // DLPNObe_kernel.  Caller pre-allocates out_B / out_E sized by
    // (n_slots, n_ij, n_ij) and passes one bucket at a time.
    void run_phase_be_into(const BEInputs *plan, BEOutputs *out);

    // Step 2j-b helpers: R2 C-term and D-term batched.  Wrap
    // DLPNOc_term_batched / DLPNOd_term_batched.  Class allocates the
    // per-thread scratch arenas (3 for C-term, 4 for D-term).
    void run_phase_c_term_into(const CTermInputs *plan, CTermOutputs *out);
    void run_phase_d_term_into(const DTermInputs *plan, DTermOutputs *out);

    // Step 2j-c helper: R2 G-term batched.  Wraps DLPNOg_term_batched.
    // Per-item: lookup scalar in G_tilde, then tmp = S @ t2, Cc = scalar *
    // tmp @ S.T.  One per-thread scratch arena.
    void run_phase_g_term_into(const GTermInputs *plan, GTermOutputs *out);

    // Step 2j-d helpers: C/D Phase 2 t3 + t4 batched.
    void run_phase_t3_into(const T3Inputs *plan, T3Outputs *out);
    void run_phase_t4_into(const T4Inputs *plan, T4Outputs *out);

    // Step 2j-e helper: R2 K + ladder.  Math is small per pair; written
    // inline (no existing C kernel).  OMP-parallel over canonical pairs.
    void run_phase_k_ladder_into(const KLadderInputs *t1_dressed,
                                  KLadderOutputs *out);

    // Step 2k helper: update T1 / T2 in place, then compute correlation
    // energy.  Mutates `in_.T1_flat` and `in_.T2_flat` (both non-const).
    void run_phase_update_amps_and_energy_into(
            const UpdateAmpsInputs *resid, UpdateAmpsOutputs *out);

    // Step 2l-b helper: compute Fia_bar per canonical pair (used by 2l-c
    // to derive Fai and Fkc on the fly via ordered-pair indexing).
    void run_phase_t1_fock_fia_bar_into(FiaBarOutputs *out);

    // Step 2l-c helper: R1 init + A + C contribution per ordered pair.
    // Consumes Fia_bar (2l-b) + cross-canonical S_pno_cache (2l-a) +
    // K_tilde_chem variants + T2 (with orientation handling).
    void run_phase_t1_residual_AC_init_into(
            const R1AcInputs *fia, R1AcOutputs *out);

    // R1 Stages 1-3 (Psi4 Fai_bar / Fab_bar*t1 / -T_n^T*Fia*t1) per
    // occupied i — wraps DLPNOper_i_stages123.  Consumes pair (i, i)'s
    // cc_ints (Qma/Qab/i_Qa/i_Qk) + T1_in_pair[(i,i)] + t1_i + e_pno.
    void run_phase_t1_residual_stages123_into(double *R1_flat);

    // c-collapse-1: run one full CCSD cycle inside C++.  Plan-cached
    // phases are passed via plans (nullable for stubs).  Outputs R1, R2,
    // energy.  T1/T2 are NOT updated by this method — the driver runs
    // DIIS externally and writes T1/T2 in place via SolverInputs.
    void run_one_cycle(const RunCycleInputs *plans, RunCycleOutputs *out);

   private:
    // -- Phase methods (all empty in the skeleton) --
    void initialize_amplitudes_();
    void prewarm_s_pno_cache_();
    void build_plan_caches_();
    void build_T_n_ij_();
    void phase_t1_ints_();
    void phase_t1_fock_();
    void phase_jiang_B_tilde_();
    void phase_jiang_C_tilde_();
    void phase_jiang_D_tilde_();
    void phase_jiang_G_tilde_();
    void phase_pairs_residual_();
    void phase_t1_residual_();
    void phase_update_amps_();
    void apply_diis_();
    double compute_iter_energy_();

    SolverInputs in_;
    double e_corr_ = 0.0;
    int final_cycle_ = -1;
};

int DLPNOCCSDSolver::solve() {
    // Skeleton: log shape, leave amplitudes untouched, return sentinel.
    std::fprintf(stderr,
        "[DLPNO-CCSD MONO skeleton] nocc=%d nlmo=%d n_canon_pairs=%d "
        "n_strong_pairs=%d max_cycle=%d e_conv=%.2e r_conv=%.2e\n",
        in_.nocc, in_.nlmo, in_.n_canon_pairs, in_.n_strong_pairs,
        in_.max_cycle, in_.e_conv, in_.r_conv);
    return -1;
}

// All phase methods are intentionally empty in the skeleton — Step 2+ fills
// them in by delegating to the existing dlpno_*.c per-pair kernels.
void DLPNOCCSDSolver::initialize_amplitudes_()    {}
void DLPNOCCSDSolver::prewarm_s_pno_cache_()      {}
void DLPNOCCSDSolver::build_plan_caches_()        {}
void DLPNOCCSDSolver::build_T_n_ij_()             {}

// Step 2b: t1_ints phase.  Loops over canonical pairs, derives per-pair
// shapes from offsets, calls DLPNOt1_ints_pair_side once per ordered side
// (i and j) of each canonical pair.  Phase-internal version (writes into
// class-owned buffers) lands in Step 2m; for now we expose the body via
// run_phase_t1_ints_into so the parity test can compare per-pair outputs.
void DLPNOCCSDSolver::phase_t1_ints_() {}

void DLPNOCCSDSolver::run_phase_t1_ints_into(T1IntsOutputs *out) {
    const int n_pairs = in_.n_canon_pairs;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < n_pairs; ++p) {
        // Skip weak pairs: PySCF's t1_ints is called for keys_sorted +
        // diagonals (strong-only); dressed Q for weak pairs is not used
        // by any downstream phase (B_tilde/K_ladder are strong-only too).
        if (in_.is_strong_pair != nullptr && in_.is_strong_pair[p] == 0) continue;
        const int i = in_.ij_to_i_j[2 * p];
        const int j = in_.ij_to_i_j[2 * p + 1];
        const int npno = in_.n_pno_per_pair[p];
        if (npno == 0) continue;

        const int64_t lmo_off = in_.pair_lmo_idx_offsets[p];
        const int nlmo_p = (int)(in_.pair_lmo_idx_offsets[p + 1] - lmo_off);
        const int *lmo_list = in_.pair_lmo_idx_flat + lmo_off;

        // Locate i and j within pair_lmo_idx[p].  For STRONG pairs the
        // invariant holds (endpoints always in their own LMO domain).  For
        // weak/extended pairs the endpoints may not appear; in that case
        // skip — the pair contributes only via T2 in per_kl, not via t1
        // dressing.
        int i_in_p = -1, j_in_p = -1;
        for (int k = 0; k < nlmo_p; ++k) {
            if (lmo_list[k] == i) i_in_p = k;
            if (lmo_list[k] == j) j_in_p = k;
        }
        if (i_in_p < 0 || j_in_p < 0) continue;

        // n_local (= naux per pair) inferred from Qma extent.
        const int64_t qma_size = in_.Qma.offsets[p + 1] - in_.Qma.offsets[p];
        const int64_t per_q = (int64_t)nlmo_p * (int64_t)npno;
        const size_t n_local = (size_t)(qma_size / per_q);

        const double *Qma_p   = fps_ptr(in_.Qma, p);
        const double *Qab_p   = fps_ptr(in_.Qab, p);
        const double *i_Qa_p  = fps_ptr(in_.i_Qa, p);
        const double *i_Qk_p  = fps_ptr(in_.i_Qk, p);
        const double *j_Qa_p  = fps_ptr(in_.j_Qa, p);
        const double *j_Qk_p  = fps_ptr(in_.j_Qk, p);

        const double *T1_local =
            fps_ptr(in_.T1_in_pair, p);
        const double *t1_lmo_i = T1_local + (int64_t)i_in_p * npno;
        const double *t1_lmo_j = T1_local + (int64_t)j_in_p * npno;

        double *i_Qa_out = out->i_Qa_t1.data + out->i_Qa_t1.offsets[p];
        double *i_Qk_out = out->i_Qk_t1.data + out->i_Qk_t1.offsets[p];
        double *j_Qa_out = out->j_Qa_t1.data + out->j_Qa_t1.offsets[p];
        double *j_Qk_out = out->j_Qk_t1.data + out->j_Qk_t1.offsets[p];

        DLPNOt1_ints_pair_side(
            i_Qa_out, i_Qk_out, i_Qa_p, i_Qk_p,
            Qma_p, Qab_p, t1_lmo_i, T1_local,
            n_local, (size_t)nlmo_p, (size_t)npno);

        DLPNOt1_ints_pair_side(
            j_Qa_out, j_Qk_out, j_Qa_p, j_Qk_p,
            Qma_p, Qab_p, t1_lmo_j, T1_local,
            n_local, (size_t)nlmo_p, (size_t)npno);
    }
}

void DLPNOCCSDSolver::run_phase_t1_fock_into(T1FockOutputs *out) {
    const int N = in_.n_canon_pairs;

    // Per-pair shape arrays (int32, kernel signature).
    std::vector<int> nlmo_arr(N), npno_arr(N), n_local_arr(N), need_dji_arr(N);
    int max_nlmo = 0, max_npno = 0, max_n_local = 0;
    for (int p = 0; p < N; ++p) {
        const int npno = in_.n_pno_per_pair[p];
        const int nlmo = (int)(in_.pair_lmo_idx_offsets[p + 1]
                                - in_.pair_lmo_idx_offsets[p]);
        const int64_t qma_size = in_.Qma.offsets[p + 1] - in_.Qma.offsets[p];
        const int64_t per_q = (int64_t)nlmo * (int64_t)npno;
        const int n_local = (per_q > 0) ? (int)(qma_size / per_q) : 0;
        nlmo_arr[p] = nlmo;
        npno_arr[p] = npno;
        n_local_arr[p] = n_local;
        const int i = in_.ij_to_i_j[2 * p];
        const int j = in_.ij_to_i_j[2 * p + 1];
        need_dji_arr[p] = (i != j) ? 1 : 0;
        if (nlmo > max_nlmo)        max_nlmo = nlmo;
        if (npno > max_npno)        max_npno = npno;
        if (n_local > max_n_local)  max_n_local = n_local;
    }

    int num_threads = 1;
    #ifdef _OPENMP
        num_threads = solver_team_size();
        if (N < num_threads) num_threads = (N > 0) ? N : 1;
    #endif

    // Scratch arenas sized to (num_threads, max_*).  Reused across
    // cycles via function-static buffers — the wrapper constructs a fresh
    // solver per call, so member caching doesn't persist.  These are
    // grown only.  Single-threaded entry (run_one_cycle is called
    // serially from Python) so static (not thread_local) is fine.
    const size_t s_gamma   = (size_t)max_n_local;
    const size_t s_Y       = (size_t)max_n_local * max_nlmo * max_npno;
    const size_t s_Fia     = (size_t)max_nlmo * max_npno;
    const size_t s_Z       = (size_t)max_n_local * max_nlmo * max_nlmo;

    static std::vector<double> gamma_sc, Y_trans_sc, Y_alt_sc;
    static std::vector<double> Fia_sc, Z_stk_sc, Z_xxx_sc;
    if (gamma_sc.size()    < (size_t)num_threads * s_gamma)
        gamma_sc.assign((size_t)num_threads * s_gamma, 0.0);
    if (Y_trans_sc.size()  < (size_t)num_threads * s_Y)
        Y_trans_sc.assign((size_t)num_threads * s_Y, 0.0);
    if (Y_alt_sc.size()    < (size_t)num_threads * s_Y)
        Y_alt_sc.assign((size_t)num_threads * s_Y, 0.0);
    if (Fia_sc.size()      < (size_t)num_threads * s_Fia)
        Fia_sc.assign((size_t)num_threads * s_Fia, 0.0);
    if (Z_stk_sc.size()    < (size_t)num_threads * s_Z)
        Z_stk_sc.assign((size_t)num_threads * s_Z, 0.0);
    if (Z_xxx_sc.size()    < (size_t)num_threads * s_Z)
        Z_xxx_sc.assign((size_t)num_threads * s_Z, 0.0);

    DLPNOt1_fock_batched(
        in_.T1_in_pair.data,    fps_pos(in_.T1_in_pair),
        in_.K_bar_chem.data,    fps_pos(in_.K_bar_chem),
        in_.K_bar_ji.data,      fps_pos(in_.K_bar_ji),
        in_.K_bar_ij.data,      fps_pos(in_.K_bar_ij),
        in_.Qma.data,           fps_pos(in_.Qma),
        in_.Qab.data,           fps_pos(in_.Qab),
        in_.e_pno_flat,         (const long *)in_.pno_offsets,
        nlmo_arr.data(), npno_arr.data(), n_local_arr.data(), need_dji_arr.data(),
        in_.is_strong_pair,
        gamma_sc.data(),  s_gamma,
        Y_trans_sc.data(), s_Y,
        Y_alt_sc.data(),   s_Y,
        Fia_sc.data(),     s_Fia,
        Z_stk_sc.data(),   s_Z,
        Z_xxx_sc.data(),   s_Z,
        out->d_flat,
        out->Fab.data, (const long *)out->Fab.offsets,
        (size_t)N, num_threads);
}

void DLPNOCCSDSolver::run_one_cycle(const RunCycleInputs *plans,
                                     RunCycleOutputs *out) {
    const int nocc   = in_.nocc;
    const int N      = in_.n_canon_pairs;
    const int N_ord  = in_.n_ordered_pairs;
    const int64_t R1_total = (N > 0) ? in_.pno_offsets[nocc] : 0;
    const int64_t R2_total = (N > 0) ? in_.t2_offsets[N] : 0;

    // ------------------------------------------------------------------
    // Per-pair shape derivation + offset tables for scratch buffers.
    // ------------------------------------------------------------------
    std::vector<int> npno_arr(N), nlmo_arr(N), n_local_arr(N);
    for (int p = 0; p < N; ++p) {
        const int npno = in_.n_pno_per_pair[p];
        const int nlmo = (int)(in_.pair_lmo_idx_offsets[p + 1]
                                - in_.pair_lmo_idx_offsets[p]);
        npno_arr[p] = npno;
        nlmo_arr[p] = nlmo;
        if (nlmo > 0 && npno > 0) {
            const int64_t qma_size = in_.Qma.offsets[p + 1] - in_.Qma.offsets[p];
            const int64_t per_q = (int64_t)nlmo * npno;
            n_local_arr[p] = (per_q > 0) ? (int)(qma_size / per_q) : 0;
        } else {
            n_local_arr[p] = 0;
        }
    }

    // Offsets for dressed Qa (n_local x npno), dressed Qk (n_local x nlmo),
    // Fia_bar (nlmo x npno), B_tilde (nlmo x nlmo).
    std::vector<int64_t> dressed_qa_off(N + 1, 0);
    std::vector<int64_t> dressed_qk_off(N + 1, 0);
    std::vector<int64_t> fia_bar_off(N + 1, 0);
    std::vector<int64_t> b_tilde_off(N + 1, 0);
    for (int p = 0; p < N; ++p) {
        const int64_t a_sz = (int64_t)n_local_arr[p] * npno_arr[p];
        const int64_t k_sz = (int64_t)n_local_arr[p] * nlmo_arr[p];
        const int64_t f_sz = (int64_t)nlmo_arr[p] * npno_arr[p];
        const int64_t b_sz = (int64_t)nlmo_arr[p] * nlmo_arr[p];
        dressed_qa_off[p + 1] = dressed_qa_off[p] + a_sz;
        dressed_qk_off[p + 1] = dressed_qk_off[p] + k_sz;
        fia_bar_off[p + 1]    = fia_bar_off[p]    + f_sz;
        b_tilde_off[p + 1]    = b_tilde_off[p]    + b_sz;
    }

    // Per-ordered-pair offsets for C_tilde / D_tilde Ph1 outputs (npno²
    // of the canonical pair).
    std::vector<int64_t> ord_npno2_off(N_ord + 1, 0);
    for (int o = 0; o < N_ord; ++o) {
        const int a = in_.ordered_pair_i_idx[o];
        const int k = in_.ordered_pair_k_idx[o];
        const int p_canon = in_.i_j_to_ij[(int64_t)a * nocc + k];
        const int npno = (p_canon >= 0) ? in_.n_pno_per_pair[p_canon] : 0;
        ord_npno2_off[o + 1] = ord_npno2_off[o] + (int64_t)npno * npno;
    }

    // One-time scratch-size report (DLPNO_CCSD_SIZE_PROBE=1): which static
    // buffers dominate run_one_cycle's anonymous footprint.  Each value is the
    // element count; ×8 bytes / 2^30 = GiB.  R2_total/t2 buffers and the
    // C/D_tilde + per-side flat_* contribution buffers are the streaming
    // targets (each ~R2_total or ~ord_npno2_off[N_ord]-sized).
    if (std::getenv("DLPNO_CCSD_SIZE_PROBE")) {
        static bool _printed_sizes = false;
        if (!_printed_sizes) {
            _printed_sizes = true;
            const double G = 8.0 / (1024.0*1024.0*1024.0);
            const int64_t R2t = in_.t2_offsets[N];
            std::fprintf(stderr,
              "[CCSD-SIZE] N=%d N_ord=%d nocc=%d | t2_off[N]=%.2fG "
              "ord_npno2_off[N_ord]=%.2fG dressed_qa=%.2fG dressed_qk=%.2fG "
              "fia_bar=%.2fG b_tilde=%.2fG\n"
              "[CCSD-SIZE] per-buffer GiB: Fab=%.2f K=%.2f A=%.2f R2_buf=%.2f "
              "C_tilde=%.2f D_tilde=%.2f | flat_{B,E}=2x%.2f "
              "flat_{C,D,G}_{ij,ji}=6x%.2f  => big-buffer subtotal ~%.1fG\n",
              N, N_ord, nocc, R2t*G, ord_npno2_off[N_ord]*G,
              dressed_qa_off[N]*G, dressed_qk_off[N]*G,
              fia_bar_off[N]*G, b_tilde_off[N]*G,
              R2t*G, R2t*G, R2t*G, R2t*G,
              ord_npno2_off[N_ord]*G, ord_npno2_off[N_ord]*G,
              R2t*G, R2t*G,
              (R2t*4 + ord_npno2_off[N_ord]*2 + R2t*2 + R2t*6)*G);
            std::fflush(stderr);
        }
    }

    // ------------------------------------------------------------------
    // Scratch buffers.  These are function-local STATIC: their sizes are
    // cycle-invariant (fixed by the pair/PNO structure), so we allocate the
    // buffer once and `.assign(size, 0.0)` each cycle to re-zero in place.
    // This is behaviourally identical to per-call `vector(size, 0.0)` but
    // avoids the ~40 GiB malloc/free churn every cycle.  That churn was the
    // real cost on large systems (e.g. MOBH35 rxn_12): each cycle's fresh
    // allocations forced the kernel to evict the cc_ints mmap page-cache,
    // which was then re-read from NVMe (~121 GiB/cycle of block I/O).  With
    // the buffers persistent the cc_ints cache stays warm.  The driver calls
    // this serially (one cycle at a time, single solver), so the statics are
    // safe; one run per process keeps the retained capacity bounded.
    static std::vector<double> dressed_iQa; dressed_iQa.assign(dressed_qa_off[N], 0.0);
    static std::vector<double> dressed_jQa; dressed_jQa.assign(dressed_qa_off[N], 0.0);
    static std::vector<double> dressed_iQk; dressed_iQk.assign(dressed_qk_off[N], 0.0);
    static std::vector<double> dressed_jQk; dressed_jQk.assign(dressed_qk_off[N], 0.0);
    static std::vector<double> Fab_flat; Fab_flat.assign(in_.t2_offsets[N], 0.0);
    static std::vector<double> d_flat; d_flat.assign((size_t)N * 2, 0.0);
    static std::vector<double> Fkj_mat; Fkj_mat.assign((size_t)nocc * nocc, 0.0);
    static std::vector<double> Fij_bar_mat; Fij_bar_mat.assign((size_t)nocc * nocc, 0.0);
    static std::vector<double> foo_t1_mat; foo_t1_mat.assign((size_t)nocc * nocc, 0.0);
    static std::vector<double> Fia_bar_flat; Fia_bar_flat.assign(fia_bar_off[N], 0.0);
    static std::vector<double> B_tilde_flat; B_tilde_flat.assign(b_tilde_off[N], 0.0);
    static std::vector<double> C_tilde_flat; C_tilde_flat.assign(ord_npno2_off[N_ord], 0.0);
    static std::vector<double> D_tilde_flat; D_tilde_flat.assign(ord_npno2_off[N_ord], 0.0);
    static std::vector<double> K_flat; K_flat.assign(in_.t2_offsets[N], 0.0);
    static std::vector<double> A_flat; A_flat.assign(in_.t2_offsets[N], 0.0);
    static std::vector<double> G_tilde_mat; G_tilde_mat.assign((size_t)nocc * nocc, 0.0);

    // Per-phase profiling (set DLPNO_CCSD_PROFILE=1 to enable).
    const bool _profile = (std::getenv("DLPNO_CCSD_PROFILE") != nullptr
                            && std::getenv("DLPNO_CCSD_PROFILE")[0] != '0');
    if (_profile) {
        int _omp_max = 1, _omp_procs = 1;
        #ifdef _OPENMP
            _omp_max   = omp_get_max_threads();
            _omp_procs = omp_get_num_procs();
        #endif
        std::fprintf(stderr,
            "[CCSD-PROFILE] run_one_cycle entry: omp_get_max_threads=%d "
            "omp_get_num_procs=%d  -> per-phase team size=%d\n",
            _omp_max, _omp_procs, solver_team_size());
        std::fflush(stderr);
    }
    using _clock = std::chrono::high_resolution_clock;
    auto _t0 = _clock::now();
    auto _tprev = _t0;
    double _t_setup = 0, _t_p1 = 0, _t_p2 = 0, _t_p3 = 0, _t_p4 = 0,
           _t_p5 = 0, _t_p6 = 0, _t_p6b = 0, _t_p7 = 0, _t_p8 = 0,
           _t_p9 = 0, _t_p10kl = 0, _t_r2_kload = 0, _t_r2_gterm = 0,
           _t_r2_be = 0, _t_r2_cd = 0, _t_r2_upd = 0;
    auto _tick = [&](double *acc) {
        auto now = _clock::now();
        *acc = std::chrono::duration<double>(now - _tprev).count();
        _tprev = now;
    };

    _tick(&_t_setup);

    // ------------------------------------------------------------------
    // Phase 1: t1_ints — produce dressed Qa/Qk per pair.
    // ------------------------------------------------------------------
    T1IntsOutputs t1_out;
    t1_out.i_Qa_t1.data    = dressed_iQa.data();
    t1_out.i_Qa_t1.offsets = dressed_qa_off.data();
    t1_out.j_Qa_t1.data    = dressed_jQa.data();
    t1_out.j_Qa_t1.offsets = dressed_qa_off.data();
    t1_out.i_Qk_t1.data    = dressed_iQk.data();
    t1_out.i_Qk_t1.offsets = dressed_qk_off.data();
    t1_out.j_Qk_t1.data    = dressed_jQk.data();
    t1_out.j_Qk_t1.offsets = dressed_qk_off.data();
    run_phase_t1_ints_into(&t1_out);
    _tick(&_t_p1);
    // ------------------------------------------------------------------
    // Phase 2: t1_fock — Fab per pair + d_flat.
    // ------------------------------------------------------------------
    T1FockOutputs t1f_out;
    t1f_out.Fab.data    = Fab_flat.data();
    t1f_out.Fab.offsets = (const int64_t *)in_.t2_offsets;
    t1f_out.d_flat      = d_flat.data();
    run_phase_t1_fock_into(&t1f_out);
    _tick(&_t_p2);
    // ------------------------------------------------------------------
    // Phase 3: t1_fock_finalize — Fkj / Fij_bar / foo_t1.
    // ------------------------------------------------------------------
    T1FockExtraInputs t1ff_in;
    t1ff_in.d_flat = d_flat.data();
    T1FockExtraOutputs t1ff_out;
    t1ff_out.Fkj              = Fkj_mat.data();
    t1ff_out.Fij_bar_snapshot = Fij_bar_mat.data();
    t1ff_out.foo_t1           = foo_t1_mat.data();
    run_phase_t1_fock_finalize_into(&t1ff_in, &t1ff_out);
    _tick(&_t_p3);
    // ------------------------------------------------------------------
    // Phase 4: Fia_bar per pair.
    // ------------------------------------------------------------------
    FiaBarOutputs fia_out;
    fia_out.Fia_bar.data    = Fia_bar_flat.data();
    fia_out.Fia_bar.offsets = fia_bar_off.data();
    run_phase_t1_fock_fia_bar_into(&fia_out);
    _tick(&_t_p4);
    // ------------------------------------------------------------------
    // Phase 5: B_tilde — consumes dressed_iQk_t1 / dressed_jQk_t1.
    // ------------------------------------------------------------------
    BTildeInputs bt_in;
    bt_in.i_Qk_t1.data    = dressed_iQk.data();
    bt_in.i_Qk_t1.offsets = dressed_qk_off.data();
    bt_in.j_Qk_t1.data    = dressed_jQk.data();
    bt_in.j_Qk_t1.offsets = dressed_qk_off.data();
    BTildeOutputs bt_out;
    bt_out.B_tilde.data    = B_tilde_flat.data();
    bt_out.B_tilde.offsets = b_tilde_off.data();
    run_phase_b_tilde_into(&bt_in, &bt_out);
    _tick(&_t_p5);
    // ------------------------------------------------------------------
    // Phase 6: C_tilde / D_tilde Phase 1 (Phase 2 from t3+t4 plans
    // is skipped in skeleton).
    // ------------------------------------------------------------------
    CTildeOutputs ct_out;
    ct_out.C_tilde.data    = C_tilde_flat.data();
    ct_out.C_tilde.offsets = ord_npno2_off.data();
    run_phase_c_tilde_ph1_into(&ct_out);
    DTildeOutputs dt_out;
    dt_out.D_tilde.data    = D_tilde_flat.data();
    dt_out.D_tilde.offsets = ord_npno2_off.data();
    run_phase_d_tilde_ph1_into(&dt_out);
    _tick(&_t_p6);

    // ------------------------------------------------------------------
    // Phase 6b: C_tilde / D_tilde Phase 2 (t3 + t4 contributions).
    // Each plan runs against C_tilde_flat (Phase 1 already in there) or
    // D_tilde_flat, accumulating per-item tiles to the targeted ordered
    // pair's slot.  PySCF residual.py:_run_t34_batched scatter is += tile
    // (sign absorbed in the kernel via -T1l for t3; t4_scale for t4).
    auto _scatter_t3_into = [&](const T3Inputs *t3p, const int *targets,
                                  std::vector<double> &dst_flat) {
        if (t3p == nullptr || targets == nullptr) return;
        const int N_t = t3p->N;
        const int64_t *tile_off = (const int64_t *)t3p->tile_off;
        const int *n_ki_arr = (const int *)t3p->n_ki_arr;
        std::vector<double> tiles((size_t)tile_off[N_t], 0.0);
        T3Outputs out_t3;
        out_t3.tiles_flat = tiles.data();
        auto _t3_t0 = _clock::now();
        run_phase_t3_into(t3p, &out_t3);
        auto _t3_t1 = _clock::now();
        // Per item, add tile to dst_flat at target ordered-pair offset.
        for (int n = 0; n < N_t; ++n) {
            const int o = targets[n];
            if (o < 0) continue;
            const int n_ki = n_ki_arr[n];
            const int64_t tile_size = (int64_t)n_ki * n_ki;
            const int64_t t_start = tile_off[n];
            const int64_t d_off = ord_npno2_off[o];
            for (int64_t e = 0; e < tile_size; ++e) {
                dst_flat[d_off + e] += tiles[t_start + e];
            }
        }
        auto _t3_t2 = _clock::now();
        if (std::getenv("DLPNO_CCSD_PROFILE_P6B") != nullptr
                && std::getenv("DLPNO_CCSD_PROFILE_P6B")[0] != '0') {
            double dt_compute = std::chrono::duration<double>(_t3_t1 - _t3_t0).count();
            double dt_scatter = std::chrono::duration<double>(_t3_t2 - _t3_t1).count();
            fprintf(stderr, "[P6B-T3] N=%d compute=%.4fs scatter=%.4fs\n",
                    N_t, dt_compute, dt_scatter);
        }
    };
    auto _scatter_t4_into = [&](const T4Inputs *t4p, const int *targets,
                                  std::vector<double> &dst_flat) {
        if (t4p == nullptr || targets == nullptr) return;
        const int N_t = t4p->N;
        const int64_t *tile_off = (const int64_t *)t4p->tile_off;
        const int *n_ki_arr = (const int *)t4p->n_ki_arr;
        std::vector<double> tiles((size_t)tile_off[N_t], 0.0);
        T4Outputs out_t4;
        out_t4.tiles_flat = tiles.data();
        auto _t4_t0 = _clock::now();
        run_phase_t4_into(t4p, &out_t4);
        auto _t4_t1 = _clock::now();
        for (int n = 0; n < N_t; ++n) {
            const int o = targets[n];
            if (o < 0) continue;
            const int n_ki = n_ki_arr[n];
            const int64_t tile_size = (int64_t)n_ki * n_ki;
            const int64_t t_start = tile_off[n];
            const int64_t d_off = ord_npno2_off[o];
            for (int64_t e = 0; e < tile_size; ++e) {
                dst_flat[d_off + e] += tiles[t_start + e];
            }
        }
        auto _t4_t2 = _clock::now();
        if (std::getenv("DLPNO_CCSD_PROFILE_P6B") != nullptr
                && std::getenv("DLPNO_CCSD_PROFILE_P6B")[0] != '0') {
            double dt_compute = std::chrono::duration<double>(_t4_t1 - _t4_t0).count();
            double dt_scatter = std::chrono::duration<double>(_t4_t2 - _t4_t1).count();
            fprintf(stderr, "[P6B-T4] N=%d compute=%.4fs scatter=%.4fs\n",
                    N_t, dt_compute, dt_scatter);
        }
    };
    // Per-sub-phase timing for p6b (set DLPNO_CCSD_PROFILE_P6B=1).
    const bool _profile_p6b = (std::getenv("DLPNO_CCSD_PROFILE_P6B") != nullptr
                                 && std::getenv("DLPNO_CCSD_PROFILE_P6B")[0] != '0');
    auto _p6b_start = _clock::now();
    auto _p6b_lap = [&](const char *label, int N_tasks, int max_n) {
        if (_profile_p6b) {
            auto now = _clock::now();
            double dt = std::chrono::duration<double>(now - _p6b_start).count();
            fprintf(stderr, "[P6B] %s N_tasks=%d max_n=%d dt=%.4fs\n",
                    label, N_tasks, max_n, dt);
            _p6b_start = now;
        }
    };
    _scatter_t3_into(plans->c_t3_plan, plans->c_t3_target_ord_idx, C_tilde_flat);
    _p6b_lap("c_t3",
             plans->c_t3_plan ? plans->c_t3_plan->N : 0,
             0);
    _scatter_t4_into(plans->c_t4_plan, plans->c_t4_target_ord_idx, C_tilde_flat);
    _p6b_lap("c_t4",
             plans->c_t4_plan ? plans->c_t4_plan->N : 0,
             0);
    _scatter_t3_into(plans->d_t3_plan, plans->d_t3_target_ord_idx, D_tilde_flat);
    _p6b_lap("d_t3",
             plans->d_t3_plan ? plans->d_t3_plan->N : 0,
             0);
    _scatter_t4_into(plans->d_t4_plan, plans->d_t4_target_ord_idx, D_tilde_flat);
    _p6b_lap("d_t4",
             plans->d_t4_plan ? plans->d_t4_plan->N : 0,
             0);
    _tick(&_t_p6b);

    // ------------------------------------------------------------------
    // Phase 7: G_tilde — initialize to Fkj (Psi4 convention); skip
    // plan-cached inner if plan absent.
    // ------------------------------------------------------------------
    std::memcpy(G_tilde_mat.data(), Fkj_mat.data(),
                (size_t)nocc * nocc * sizeof(double));
    if (plans->g_tilde_plan != nullptr) {
        GTildeOutputs g_out;
        g_out.G_tilde = G_tilde_mat.data();
        run_phase_g_tilde_inner_into(plans->g_tilde_plan, &g_out);
    }
    if (out->G_tilde_out != nullptr) {
        std::memcpy(out->G_tilde_out, G_tilde_mat.data(),
                    (size_t)nocc * nocc * sizeof(double));
    }
    _tick(&_t_p7);
    // ------------------------------------------------------------------
    // Phase 8: K + ladder — per canonical pair K and A.
    // ------------------------------------------------------------------
    KLadderInputs kl_in;
    kl_in.i_Qa_t1.data    = dressed_iQa.data();
    kl_in.i_Qa_t1.offsets = dressed_qa_off.data();
    kl_in.j_Qa_t1.data    = dressed_jQa.data();
    kl_in.j_Qa_t1.offsets = dressed_qa_off.data();
    KLadderOutputs kl_out;
    kl_out.K.data    = K_flat.data();
    kl_out.K.offsets = (const int64_t *)in_.t2_offsets;
    kl_out.A.data    = A_flat.data();
    kl_out.A.offsets = (const int64_t *)in_.t2_offsets;
    run_phase_k_ladder_into(&kl_in, &kl_out);
    _tick(&_t_p8);
    // ------------------------------------------------------------------
    // Phase 9: R1 build.
    //   R1[i] = Stages 1-3 (Psi4 Fai_bar/Fab_bar*t1/-T_n^T*Fia*t1)
    //         + Stage 4 (-Fij_bar[:, i] @ T1_in_pair_full[(i,i)])
    //         + A + C (per ordered pair, no init)
    //         + per_kl B+A2 (if plan provided).
    // ------------------------------------------------------------------
    static std::vector<double> R1_buf; R1_buf.assign((size_t)R1_total, 0.0);

    // Stages 1-3.
    run_phase_t1_residual_stages123_into(R1_buf.data());
    // Stage 4: r1[i] -= Fij_bar[:, i] @ T1_in_pair_full[(i,i)]
    // T1_in_pair_full per pair (i, i) is shape (nocc, npno_ii) row-major.
    // Use Psi4-faithful override (Fij_bar_full) when provided — it includes
    // weak-pair dressing the strong-only t1_fock snapshot misses.
    const double *Fij_bar_use = (in_.Fij_bar_full != nullptr)
                                  ? in_.Fij_bar_full : Fij_bar_mat.data();
    if (in_.T1_in_pair_full.data != nullptr) {
        for (int i = 0; i < nocc; ++i) {
            const int p_ii = in_.i_j_to_ij[(int64_t)i * nocc + i];
            if (p_ii < 0) continue;
            const int npno_ii = in_.n_pno_per_pair[p_ii];
            if (npno_ii == 0) continue;
            const double *T_full = fps_ptr(in_.T1_in_pair_full, p_ii);
            const int64_t r1_off = in_.pno_offsets[i];
            for (int a = 0; a < npno_ii; ++a) {
                double s = 0.0;
                for (int k = 0; k < nocc; ++k) {
                    s += Fij_bar_use[(int64_t)k * nocc + i]
                       * T_full[(int64_t)k * npno_ii + a];
                }
                R1_buf[r1_off + a] -= s;
            }
        }
    }

    // A + C contribution (no init — Stages 1-3 + Stage 4 already produced
    // the Fai equivalent via Psi4-style dressing).
    R1AcInputs r1ac_in;
    r1ac_in.Fia_bar.data    = Fia_bar_flat.data();
    r1ac_in.Fia_bar.offsets = fia_bar_off.data();
    r1ac_in.do_init         = 0;
    R1AcOutputs r1ac_out;
    r1ac_out.R1_flat = R1_buf.data();
    run_phase_t1_residual_AC_init_into(&r1ac_in, &r1ac_out);

    _tick(&_t_p9);
    if (plans->per_kl_plan != nullptr) {
        const PerKlPlanInputs *p_plan = plans->per_kl_plan;
        const int n_tasks = p_plan->n_tasks;
        const int64_t total_inner = p_plan->inner_off[n_tasks];

        // Total contrib_flat size = sum of n_pno_ii_arr[0..total_inner-1].
        int64_t contrib_total = 0;
        for (int64_t ti = 0; ti < total_inner; ++ti) {
            contrib_total += p_plan->n_pno_ii_arr[ti];
        }
        static std::vector<double> contrib_flat; contrib_flat.assign((size_t)contrib_total, 0.0);

        PerKlOutputs perkl_out;
        perkl_out.contrib_flat = contrib_flat.data();
        run_phase_t1_residual_per_kl_into(p_plan, &perkl_out);

        // Scatter contrib_flat into R1_buf: for each (task, inner_i),
        // R1[i_arr[ti] block] += contrib_flat[contrib_off[ti] : ...].
        for (int64_t ti = 0; ti < total_inner; ++ti) {
            const int i = p_plan->i_arr[ti];
            const int n_pno_ii = p_plan->n_pno_ii_arr[ti];
            const int64_t r1_off = in_.pno_offsets[i];
            const int64_t c_off = p_plan->contrib_off[ti];
            for (int a = 0; a < n_pno_ii; ++a) {
                R1_buf[r1_off + a] += contrib_flat[c_off + a];
            }
        }
    }
    _tick(&_t_p10kl);

    // ------------------------------------------------------------------
    // R2 orchestration.  Two paths:
    //   (a) R2_external provided → use AS-IS (Psi4-symmetrized R2
    //       precomputed by PySCF's full residual machinery).
    //   (b) Otherwise → native build: K + ladder + plan-cached R2
    //       contributions (BE/CD/G_term/t3+t4) wired here.
    // ------------------------------------------------------------------
    static std::vector<double> R2_buf; R2_buf.assign((size_t)R2_total, 0.0);
    if (in_.R2_external != nullptr) {
        std::memcpy(R2_buf.data(), in_.R2_external,
                    (size_t)R2_total * sizeof(double));
    } else {
        // ----------------------------------------------------------------
        // R2 build — Psi4-mirror monolithic per-pair fusion.
        //   Phase A: run all kernels (G_term ik/jk, BE per bucket, C/D term)
        //            populating their flat output buffers.
        //   Phase B: ONE fused per-pair loop combines K+A+B+E_tilde+G+C+D
        //            into R2[p].  Cache-locality win vs separate scatters.
        // ----------------------------------------------------------------

        // Hoisted flat buffers (empty when feature disabled).
        static std::vector<double> flat_G_ij_buf, flat_G_ji_buf;
        bool have_g_term = false;
        static std::vector<double> flat_B, flat_E;
        bool have_be = false;
        static std::vector<double> flat_C_ij_buf, flat_C_ji_buf;
        static std::vector<double> flat_D_ij_buf, flat_D_ji_buf;
        bool have_cd = false;
        _tick(&_t_r2_kload);  // K + A already populated by run_phase_k_ladder_into

        // ---- Phase A.1: G_term kernels (ik + jk) ----
        if (plans->g_term_plan != nullptr
                && plans->g_term_plan_jk != nullptr
                && plans->g_term_target_pair_idx_ik != nullptr
                && plans->g_term_target_pair_idx_jk != nullptr) {
            flat_G_ij_buf.assign((size_t)R2_total, 0.0);
            flat_G_ji_buf.assign((size_t)R2_total, 0.0);
            have_g_term = true;

            GTermInputs g_ik_local = *plans->g_term_plan;
            g_ik_local.G_tilde = G_tilde_mat.data();
            g_ik_local.G_stride = nocc;
            const int N_ik = g_ik_local.N;
            const int64_t *tile_off_ik =
                (const int64_t *)g_ik_local.tile_off;
            const int64_t total_tiles_ik = tile_off_ik[N_ik];
            static std::vector<double> tiles_ik_buf; tiles_ik_buf.assign((size_t)total_tiles_ik, 0.0);
            GTermOutputs gout_ik;
            gout_ik.tiles_flat = tiles_ik_buf.data();
            run_phase_g_term_into(&g_ik_local, &gout_ik);
            const int *n_ij_ik = (const int *)g_ik_local.n_ij_arr;
            for (int n = 0; n < N_ik; ++n) {
                const int n_ij = n_ij_ik[n];
                const int target_p = plans->g_term_target_pair_idx_ik[n];
                if (target_p < 0) continue;
                const int64_t r2_off = in_.t2_offsets[target_p];
                const int64_t tile_size = (int64_t)n_ij * n_ij;
                const int64_t t_start = tile_off_ik[n];
                for (int64_t e = 0; e < tile_size; ++e) {
                    flat_G_ij_buf[r2_off + e] -= tiles_ik_buf[t_start + e];
                }
            }

            GTermInputs g_jk_local = *plans->g_term_plan_jk;
            g_jk_local.G_tilde = G_tilde_mat.data();
            g_jk_local.G_stride = nocc;
            const int N_jk = g_jk_local.N;
            const int64_t *tile_off_jk =
                (const int64_t *)g_jk_local.tile_off;
            const int64_t total_tiles_jk = tile_off_jk[N_jk];
            static std::vector<double> tiles_jk_buf; tiles_jk_buf.assign((size_t)total_tiles_jk, 0.0);
            GTermOutputs gout_jk;
            gout_jk.tiles_flat = tiles_jk_buf.data();
            run_phase_g_term_into(&g_jk_local, &gout_jk);
            const int *n_ij_jk = (const int *)g_jk_local.n_ij_arr;
            for (int n = 0; n < N_jk; ++n) {
                const int n_ij = n_ij_jk[n];
                const int target_p = plans->g_term_target_pair_idx_jk[n];
                if (target_p < 0) continue;
                const int64_t r2_off = in_.t2_offsets[target_p];
                const int64_t tile_size = (int64_t)n_ij * n_ij;
                const int64_t t_start = tile_off_jk[n];
                for (int64_t e = 0; e < tile_size; ++e) {
                    flat_G_ji_buf[r2_off + e] -= tiles_jk_buf[t_start + e];
                }
            }
        }
        _tick(&_t_r2_gterm);

        // ---- Phase A.2: BE kernels per bucket (multi-bucket) ----
        if (plans->be_n_buckets > 0
                && plans->be_plan_buckets != nullptr
                && plans->be_pair_n_ij_idx != nullptr) {
            const int n_unique = plans->be_n_unique_n_ij;
            const int64_t total_flat = plans->be_flat_off_per_n_ij[n_unique];
            flat_B.assign((size_t)total_flat, 0.0);
            flat_E.assign((size_t)total_flat, 0.0);
            have_be = true;

            // BE-bucket beta_kl/lk refresh from native B_tilde_flat.  The
            // dict-extracted values were populated at pack time from
            // cycle-0 B_tilde and are stale for cycle 1+.  Phase 5 rebuilt
            // B_tilde_flat (sized nlmo×nlmo per pair) from current T1;
            // mirror the dict extraction using p_ij_arr / dense_k_arr /
            // dense_l_arr (LMO-domain indices into the pair's nlmo basis).
            // Each bucket's beta_kl/beta_lk arrays are distinct memory; the
            // refresh is read-only on shared B_tilde_flat → trivially parallel.
            #pragma omp parallel for schedule(dynamic, 1)
            for (int b = 0; b < plans->be_n_buckets; ++b) {
                const BEInputs *bucket = &plans->be_plan_buckets[b];
                if (bucket->p_ij_arr == nullptr) continue;
                double *bk = const_cast<double *>(bucket->beta_kl);
                double *bl = const_cast<double *>(bucket->beta_lk);
                for (int n = 0; n < bucket->N; ++n) {
                    const int p  = bucket->p_ij_arr[n];
                    if (p < 0) continue;
                    const int dk = bucket->dense_k_arr[n];
                    const int dl = bucket->dense_l_arr[n];
                    const int nlmo_p = nlmo_arr[p];
                    if (nlmo_p == 0) continue;
                    const int64_t off = b_tilde_off[p];
                    bk[n] = B_tilde_flat[off + (int64_t)dk * nlmo_p + dl];
                    bl[n] = (dk == dl) ? 0.0
                          : B_tilde_flat[off + (int64_t)dl * nlmo_p + dk];
                }
            }

            // Per bucket: find n_ij group, run kernel with offset output.
            //
            // BE-kernel-v2 (per-target accumulation): the v1 kernel
            // materialised one (n_ij × n_ij) tile per item then did a
            // serial scatter-add into the group's flat output, costing
            // ~17 MB / bucket × ~30 buckets / cycle of memory traffic on
            // water-22.  v2 groups items by their target slot idx[n] and
            // accumulates each item's STB @ S^T directly into the output
            // slot via dgemm beta=1.  Slots are disjoint so OMP runs
            // race-free over targets.  Set DLPNO_BE_V1=1 to fall back.
            const bool _be_use_v2 =
                (std::getenv("DLPNO_BE_V1") == nullptr
                 || std::getenv("DLPNO_BE_V1")[0] == '0');
            for (int b = 0; b < plans->be_n_buckets; ++b) {
                const BEInputs *bucket = &plans->be_plan_buckets[b];
                int g = -1;
                for (int gi = 0; gi < n_unique; ++gi) {
                    if (plans->be_unique_n_ij[gi] == bucket->n_ij) {
                        g = gi; break;
                    }
                }
                if (g < 0) continue;
                const int64_t group_off = plans->be_flat_off_per_n_ij[g];
                BEOutputs out;
                out.out_B = flat_B.data() + group_off;
                out.out_E = flat_E.data() + group_off;
                // Gathered mode (DLPNO_BE_GATHERED=1): bucket S/T/K stacks
                // are null; route through run_phase_be_into which dispatches
                // DLPNObe_kernel_gathered from the master flats.
                const bool _gathered = (bucket->S_master != nullptr
                                        && bucket->T_master != nullptr
                                        && bucket->K_master != nullptr);
                if (_be_use_v2 && !_gathered) {
                    int num_threads = 1;
#ifdef _OPENMP
                    num_threads = solver_team_size();
                    if (bucket->N > 0 && num_threads > bucket->N)
                        num_threads = bucket->N;
#endif
                    DLPNObe_kernel_v3(
                        bucket->S, bucket->T, bucket->K,
                        bucket->beta_kl, bucket->beta_lk,
                        bucket->same, bucket->idx,
                        out.out_B, out.out_E,
                        (size_t)bucket->N, (size_t)bucket->n_ij,
                        (size_t)bucket->n_kl, num_threads,
                        (size_t)bucket->n_slots);
                } else {
                    run_phase_be_into(bucket, &out);
                }
            }
            // BE per-pair scatter is fused below (Phase B).
        }
        _tick(&_t_r2_be);

        // Step 4: CD contribution (C_term + D_term).
        // PySCF assembly:
        //   C_term[p] = 0.5*Cij[p] + Cij[p].T + 0.5*Cji[p].T + Cji[p]
        //   D_term[p] = Dij[p] + Dji[p].T
        // where Cij/Cji are flat per-pair buffers populated by scattering
        // c-kernel tiles with sign convention (PySCF residual.py:4749 -=).
        // Dij/Dji are populated with +0.5*tile (residual.py:4874).
        // ---- Phase A.3: C_term + D_term kernels ----
        if (plans->c_term_plan != nullptr
                && plans->d_term_plan != nullptr
                && plans->c_term_target_pair_idx_ij != nullptr) {
            flat_C_ij_buf.assign((size_t)R2_total, 0.0);
            flat_C_ji_buf.assign((size_t)R2_total, 0.0);
            flat_D_ij_buf.assign((size_t)R2_total, 0.0);
            flat_D_ji_buf.assign((size_t)R2_total, 0.0);
            have_cd = true;

            // C side.
            const CTermInputs *c_plan_orig = plans->c_term_plan;
            const int N_c = c_plan_orig->N;
            const int64_t *c_tile_off = (const int64_t *)c_plan_orig->tile_off;
            static std::vector<double> c_tiles_buf; c_tiles_buf.assign((size_t)c_tile_off[N_c], 0.0);
            // If c_term_ct_ord_pair_idx is provided, build ct_flat from
            // class's native C_tilde_flat (post-Phase 2).  Otherwise use
            // the user-provided ct_flat (from PySCF dict gather).
            CTermInputs c_plan = *c_plan_orig;
            // Offset-alias ct directly into C_tilde_flat (the native per-
            // ordered-pair C_tilde) instead of gathering a per-item copy.
            // The kernel reads ct = ct_flat + ct_off[n]; pointing ct_flat at
            // C_tilde_flat and ct_off[n] at ord_npno2_off[o] reads the same
            // bytes with zero duplication.  Each item's C_tilde block is
            // (n_ct x n_ct) = (npno_o x npno_o) where o is its ordered pair,
            // so sizes match.  Eliminates the ~per-item-duplicated ct gather
            // (a multi-GiB CCSD-cycle buffer on TM complexes).
            static std::vector<int64_t> ct_src_off;
            if (plans->c_term_ct_ord_pair_idx != nullptr) {
                ct_src_off.assign((size_t)N_c, 0);
                for (int n = 0; n < N_c; ++n) {
                    const int o = plans->c_term_ct_ord_pair_idx[n];
                    ct_src_off[n] = (o >= 0) ? ord_npno2_off[o] : 0;
                }
                c_plan.ct_flat = C_tilde_flat.data();
                c_plan.ct_off  = (const long *)ct_src_off.data();
            }
            CTermOutputs c_out;
            c_out.tiles_flat = c_tiles_buf.data();
            run_phase_c_term_into(&c_plan, &c_out);
            const int *c_n_pno = (const int *)c_plan_orig->n_pno_arr;
            for (int n = 0; n < N_c; ++n) {
                const int npno_n = c_n_pno[n];
                const int p_ij = plans->c_term_target_pair_idx_ij[n];
                const int p_ji = plans->c_term_target_pair_idx_ji[n];
                const int64_t t_start = c_tile_off[n];
                const int64_t tile_size = (int64_t)npno_n * npno_n;
                if (p_ij >= 0) {
                    const int64_t r2_off = in_.t2_offsets[p_ij];
                    for (int64_t e = 0; e < tile_size; ++e) {
                        flat_C_ij_buf[r2_off + e] -= c_tiles_buf[t_start + e];
                    }
                } else if (p_ji >= 0) {
                    const int64_t r2_off = in_.t2_offsets[p_ji];
                    for (int64_t e = 0; e < tile_size; ++e) {
                        flat_C_ji_buf[r2_off + e] -= c_tiles_buf[t_start + e];
                    }
                }
            }

            // D side.
            const DTermInputs *d_plan_orig = plans->d_term_plan;
            const int N_d = d_plan_orig->N;
            const int64_t *d_tile_off = (const int64_t *)d_plan_orig->tile_off;
            static std::vector<double> d_tiles_buf; d_tiles_buf.assign((size_t)d_tile_off[N_d], 0.0);
            DTermInputs d_plan = *d_plan_orig;
            // Offset-alias dt into D_tilde_flat (per-ordered-pair D_tilde),
            // same as the c-side ct alias: read dt = dt_flat + dt_off[n] with
            // dt_flat = D_tilde_flat and dt_off[n] = ord_npno2_off[o].  No
            // per-item-duplicated copy.
            static std::vector<int64_t> dt_src_off;
            if (plans->d_term_dt_ord_pair_idx != nullptr) {
                dt_src_off.assign((size_t)N_d, 0);
                for (int n = 0; n < N_d; ++n) {
                    const int o = plans->d_term_dt_ord_pair_idx[n];
                    dt_src_off[n] = (o >= 0) ? ord_npno2_off[o] : 0;
                }
                d_plan.dt_flat = D_tilde_flat.data();
                d_plan.dt_off  = (const long *)dt_src_off.data();
            }
            DTermOutputs d_out;
            d_out.tiles_flat = d_tiles_buf.data();
            run_phase_d_term_into(&d_plan, &d_out);
            const int *d_n_pno = (const int *)d_plan_orig->n_pno_arr;
            for (int n = 0; n < N_d; ++n) {
                const int npno_n = d_n_pno[n];
                const int p_ij = plans->d_term_target_pair_idx_ij[n];
                const int p_ji = plans->d_term_target_pair_idx_ji[n];
                const int64_t t_start = d_tile_off[n];
                const int64_t tile_size = (int64_t)npno_n * npno_n;
                if (p_ij >= 0) {
                    const int64_t r2_off = in_.t2_offsets[p_ij];
                    for (int64_t e = 0; e < tile_size; ++e) {
                        flat_D_ij_buf[r2_off + e] += 0.5 * d_tiles_buf[t_start + e];
                    }
                } else if (p_ji >= 0) {
                    const int64_t r2_off = in_.t2_offsets[p_ji];
                    for (int64_t e = 0; e < tile_size; ++e) {
                        flat_D_ji_buf[r2_off + e] += 0.5 * d_tiles_buf[t_start + e];
                    }
                }
            }
            // CD per-pair assembly is fused below (Phase B).
        }

        // ------------------------------------------------------------
        // Phase B: Fused per-pair R2 assembly.  Touch each pair's
        // R2_buf[r2_off : r2_off + npno²] ONCE; combine K + A + G + B
        // + (T2*E_tilde + E_tilde*T2) + C + D in cache.  Mirrors Psi4's
        // monolithic per-pair loop in ccsd.cc:2417+.
        // ------------------------------------------------------------
        #pragma omp parallel for schedule(dynamic, 1)
        for (int p = 0; p < N; ++p) {
            if (in_.is_strong_pair != nullptr
                    && in_.is_strong_pair[p] == 0) continue;
            const int npno = npno_arr[p];
            if (npno == 0) continue;
            const int64_t r2_off = in_.t2_offsets[p];
            const int64_t sz = (int64_t)npno * npno;
            double *R = R2_buf.data() + r2_off;

            // Init from K + A (K_flat, A_flat populated by run_phase_k_ladder).
            const double *Kp = K_flat.data() + r2_off;
            const double *Ap = A_flat.data() + r2_off;
            for (int64_t e = 0; e < sz; ++e) {
                R[e] = Kp[e] + Ap[e];
            }

            // G_term: R += flat_G_ij[p] + flat_G_ji[p].T
            if (have_g_term) {
                const double *Gij = flat_G_ij_buf.data() + r2_off;
                const double *Gji = flat_G_ji_buf.data() + r2_off;
                for (int a = 0; a < npno; ++a) {
                    for (int b = 0; b < npno; ++b) {
                        R[(int64_t)a * npno + b] +=
                            Gij[(int64_t)a * npno + b]
                            + Gji[(int64_t)b * npno + a];
                    }
                }
            }

            // BE: R += B[p]; E_tilde = Fab[p] - E[p]; R += T2 @ E_tilde.T + E_tilde @ T2
            if (have_be) {
                const int g = plans->be_pair_n_ij_idx[p];
                if (g >= 0) {
                    const int slot = plans->be_pair_slot[p];
                    const int64_t group_off = plans->be_flat_off_per_n_ij[g];
                    const int64_t base = group_off + (int64_t)slot * sz;

                    // R += B[p]
                    for (int64_t e = 0; e < sz; ++e) {
                        R[e] += flat_B[base + e];
                    }

                    // E_tilde[a, b] = Fab[a, b] - E[base + (a*npno+b)]
                    std::vector<double> E_tilde((size_t)sz);
                    const double *Fab_p = Fab_flat.data() + r2_off;
                    for (int64_t e = 0; e < sz; ++e) {
                        E_tilde[(size_t)e] = Fab_p[e] - flat_E[base + e];
                    }

                    // R[a, b] += sum_c T2[a, c]*E_tilde[b, c] + E_tilde[a, c]*T2[c, b]
                    const double *T2_p = in_.T2_flat + r2_off;
                    for (int a = 0; a < npno; ++a) {
                        for (int b = 0; b < npno; ++b) {
                            double s = 0.0;
                            for (int c = 0; c < npno; ++c) {
                                s += T2_p[a * npno + c] * E_tilde[b * npno + c]
                                   + E_tilde[a * npno + c] * T2_p[c * npno + b];
                            }
                            R[(int64_t)a * npno + b] += s;
                        }
                    }
                }
            }

            // CD: R += 0.5*Cij + Cij.T + 0.5*Cji.T + Cji + Dij + Dji.T
            if (have_cd) {
                const double *Cij = flat_C_ij_buf.data() + r2_off;
                const double *Cji = flat_C_ji_buf.data() + r2_off;
                const double *Dij = flat_D_ij_buf.data() + r2_off;
                const double *Dji = flat_D_ji_buf.data() + r2_off;
                for (int a = 0; a < npno; ++a) {
                    for (int b = 0; b < npno; ++b) {
                        const int64_t e_ab = (int64_t)a * npno + b;
                        const int64_t e_ba = (int64_t)b * npno + a;
                        R[e_ab] += 0.5 * Cij[e_ab] + Cij[e_ba]
                                + 0.5 * Cji[e_ba] + Cji[e_ab]
                                + Dij[e_ab] + Dji[e_ba];
                    }
                }
            }
        }
    }
    _tick(&_t_r2_cd);

    // ------------------------------------------------------------------
    // Update T1/T2 in place + compute correlation energy.
    // ------------------------------------------------------------------
    UpdateAmpsInputs resid;
    resid.R1_flat = R1_buf.data();
    resid.R2_flat = R2_buf.data();
    UpdateAmpsOutputs upd_out;
    upd_out.energy = 0.0;
    run_phase_update_amps_and_energy_into(&resid, &upd_out);
    _tick(&_t_r2_upd);

    if (_profile) {
        const double total = std::chrono::duration<double>(_tprev - _t0).count();
        std::fprintf(stderr,
            "[CCSD-PROFILE] total=%.3fs setup=%.3f p1=%.3f p2=%.3f p3=%.3f "
            "p4=%.3f p5=%.3f p6=%.3f p6b=%.3f p7=%.3f p8=%.3f p9=%.3f "
            "p10kl=%.3f r2_K=%.3f r2_G=%.3f r2_BE=%.3f r2_CD=%.3f r2_upd=%.3f\n",
            total, _t_setup, _t_p1, _t_p2, _t_p3, _t_p4, _t_p5, _t_p6,
            _t_p6b, _t_p7, _t_p8, _t_p9, _t_p10kl, _t_r2_kload,
            _t_r2_gterm, _t_r2_be, _t_r2_cd, _t_r2_upd);
        std::fflush(stderr);
    }

    // ------------------------------------------------------------------
    // Copy R1/R2 to caller's output buffers.
    // ------------------------------------------------------------------
    std::memcpy(out->R1_flat, R1_buf.data(),
                (size_t)R1_total * sizeof(double));
    std::memcpy(out->R2_flat, R2_buf.data(),
                (size_t)R2_total * sizeof(double));
    out->energy = upd_out.energy;
}

void DLPNOCCSDSolver::run_phase_t1_residual_stages123_into(double *R1_flat) {
    const int nocc = in_.nocc;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < nocc; ++i) {
        const int p_ii = in_.i_j_to_ij[(int64_t)i * nocc + i];
        if (p_ii < 0) continue;
        const int npno_ii = in_.n_pno_per_pair[p_ii];
        if (npno_ii == 0) continue;

        const int64_t lmo_off = in_.pair_lmo_idx_offsets[p_ii];
        const int nlmo_p = (int)(in_.pair_lmo_idx_offsets[p_ii + 1] - lmo_off);
        if (nlmo_p == 0) continue;

        const int64_t qma_size = in_.Qma.offsets[p_ii + 1]
                                  - in_.Qma.offsets[p_ii];
        const int64_t per_q = (int64_t)nlmo_p * npno_ii;
        const int n_local = (per_q > 0) ? (int)(qma_size / per_q) : 0;

        const double *Qma = fps_ptr(in_.Qma, p_ii);
        const double *Qab = fps_ptr(in_.Qab, p_ii);
        const double *Qia = fps_ptr(in_.i_Qa, p_ii);
        const double *Qik = fps_ptr(in_.i_Qk, p_ii);
        const double *T_n = fps_ptr(in_.T1_in_pair, p_ii);
        const double *t1_i = in_.T1_flat + in_.pno_offsets[i];
        const double *e_pno = in_.e_pno_flat + in_.pno_offsets[p_ii];

        const int64_t r1_off = in_.pno_offsets[i];
        DLPNOper_i_stages123(
            R1_flat + r1_off,
            Qma, Qab, Qia, Qik, T_n, t1_i, e_pno,
            /*do_stage_23=*/1,
            (size_t)n_local, (size_t)nlmo_p, (size_t)npno_ii);
    }
}

void DLPNOCCSDSolver::run_phase_t1_residual_AC_init_into(
    const R1AcInputs *fia, R1AcOutputs *out) {
    const int nocc      = in_.nocc;
    const int N_canon   = in_.n_canon_pairs;
    const int N_ord     = in_.n_ordered_pairs;
    const int total_R1  = (int)in_.pno_offsets[nocc];

    // Zero R1 output ONLY when init is enabled (legacy behavior).
    // When called from run_one_cycle with do_init=0, the caller has
    // already populated R1 with Stages 1-3 + Stage 4 contributions and
    // we only ADD A + C on top.
    if (fia->do_init) {
        for (int e = 0; e < total_R1; ++e) out->R1_flat[e] = 0.0;
    }

    // ------------------------------------------------------------------
    // Init: R1[i] += Fai[i] = Fia_bar[(i,i)][i_in_p, :]   (gated)
    // ------------------------------------------------------------------
    if (fia->do_init) {
    for (int i = 0; i < nocc; ++i) {
        const int p_ii = in_.i_j_to_ij[(int64_t)i * nocc + i];
        if (p_ii < 0) continue;
        const int npno_ii = in_.n_pno_per_pair[p_ii];
        if (npno_ii == 0) continue;

        const int64_t lmo_off = in_.pair_lmo_idx_offsets[p_ii];
        const int nlmo_ii = (int)(in_.pair_lmo_idx_offsets[p_ii + 1] - lmo_off);
        const int *lmo_list = in_.pair_lmo_idx_flat + lmo_off;

        int i_in_p = -1;
        for (int kk = 0; kk < nlmo_ii; ++kk) {
            if (lmo_list[kk] == i) { i_in_p = kk; break; }
        }
        if (i_in_p < 0) continue;

        const double *Fia_bar_p =
            fia->Fia_bar.data + fia->Fia_bar.offsets[p_ii];
        const double *Fai = Fia_bar_p + (int64_t)i_in_p * npno_ii;
        const int64_t r1_off = in_.pno_offsets[i];
        for (int a = 0; a < npno_ii; ++a) {
            out->R1_flat[r1_off + a] += Fai[a];
        }
    }
    }  // end if(do_init)

    // ------------------------------------------------------------------
    // Precompute LT1 per ordered pair (i, m) for the C term inner sum.
    //   LT1[(i, m)] = (2·K_iajb[canon(i,m)] - K_iajb[canon(i,m)]ᵀ) @ t1_m_in_im
    // where t1_m_in_im = T1_in_pair_full[canon(i,m)][m, :] (length npno_canon).
    // Plus a lookup table ord_idx_lookup[i*nocc + m] = ordered-pair index.
    // ------------------------------------------------------------------
    std::vector<int> ord_idx_lookup((size_t)nocc * nocc, -1);
    for (int o = 0; o < N_ord; ++o) {
        const int i_o = in_.ordered_pair_i_idx[o];
        const int m_o = in_.ordered_pair_k_idx[o];
        ord_idx_lookup[(int64_t)i_o * nocc + m_o] = o;
    }

    std::vector<int64_t> lt1_offsets((size_t)N_ord + 1, 0);
    for (int o = 0; o < N_ord; ++o) {
        const int i_o = in_.ordered_pair_i_idx[o];
        const int m_o = in_.ordered_pair_k_idx[o];
        const int p_canon = in_.i_j_to_ij[(int64_t)i_o * nocc + m_o];
        const int npno = (p_canon >= 0) ? in_.n_pno_per_pair[p_canon] : 0;
        lt1_offsets[o + 1] = lt1_offsets[o] + npno;
    }
    std::vector<double> lt1_flat((size_t)lt1_offsets[N_ord], 0.0);

    if (in_.T1_in_pair_full.data != nullptr) {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int o = 0; o < N_ord; ++o) {
            const int i_o = in_.ordered_pair_i_idx[o];
            const int m_o = in_.ordered_pair_k_idx[o];
            const int p_im = in_.i_j_to_ij[(int64_t)i_o * nocc + m_o];
            if (p_im < 0) continue;
            const int npno_im = in_.n_pno_per_pair[p_im];
            if (npno_im == 0) continue;

            const double *K = fps_ptr(in_.K_iajb, p_im);
            const double *T_full = fps_ptr(in_.T1_in_pair_full, p_im);
            const double *t1_m = T_full + (int64_t)m_o * npno_im;

            double *lt1_o = lt1_flat.data() + lt1_offsets[o];
            for (int a = 0; a < npno_im; ++a) {
                double s = 0.0;
                for (int b = 0; b < npno_im; ++b) {
                    s += (2.0 * K[a * npno_im + b] - K[b * npno_im + a])
                         * t1_m[b];
                }
                lt1_o[a] = s;
            }
        }
    }

    // ------------------------------------------------------------------
    // A + C contribution per ordered pair.  Per-thread R1 buffer to avoid
    // races on R1[a_ord] (multiple ordered pairs share a_ord).
    // ------------------------------------------------------------------
    int num_threads = 1;
    #ifdef _OPENMP
        num_threads = solver_team_size();
        if (N_ord > 0 && num_threads > N_ord) num_threads = N_ord;
    #endif
    std::vector<double> R1_thread((size_t)num_threads * total_R1, 0.0);

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = 0;
        #ifdef _OPENMP
            tid = omp_get_thread_num();
        #endif
        double *R1_t = R1_thread.data() + (size_t)tid * total_R1;

        #pragma omp for schedule(dynamic, 1)
        for (int o = 0; o < N_ord; ++o) {
            const int a_ord = in_.ordered_pair_i_idx[o];   // Psi4 "i"
            const int b_ord = in_.ordered_pair_k_idx[o];   // Psi4 "k"

            const int p_canon = in_.i_j_to_ij[(int64_t)a_ord * nocc + b_ord];
            if (p_canon < 0) continue;
            const int npno_p = in_.n_pno_per_pair[p_canon];
            if (npno_p == 0) continue;

            const int p_ii = in_.i_j_to_ij[(int64_t)a_ord * nocc + a_ord];
            if (p_ii < 0) continue;
            const int npno_ii = in_.n_pno_per_pair[p_ii];
            if (npno_ii == 0) continue;

            const int can_first = in_.ij_to_i_j[2 * p_canon];

            // Look up S_PNO(p_canon, p_ii).  Skip if not stored.
            const int64_t s_idx = (int64_t)p_canon * N_canon + p_ii;
            int64_t s_off, s_size; s_pno_lookup(in_, s_idx, s_off, s_size);
            if (s_size != (int64_t)npno_p * (int64_t)npno_ii) continue;
            const double *S_p = in_.S_pno_data + s_off;

            const double *T2_p = in_.T2_flat + in_.t2_offsets[p_canon];

            // -- A term --------------------------------------------------
            // K_tilde_chem variant for ordered (b_ord, a_ord) ("ki"):
            //   pick "_i" if can_first == b_ord, else "_j".
            const FlatPairStore &kt = (can_first == b_ord)
                ? in_.K_tilde_chem_i : in_.K_tilde_chem_j;
            const double *K_chem_ki = kt.data + kt.offsets[p_canon];

            // T2 swap for "ki": swap iff can_first != b_ord.
            const bool swap_ki = (can_first != b_ord);

            // Build Tt_ki = 2*T2_ki - T2_ki.T as a flat (npno_p², ) array.
            // T2_ki[a, b] = swap ? T2_p[b, a] : T2_p[a, b].
            std::vector<double> Tt_ki((size_t)npno_p * npno_p);
            for (int a = 0; a < npno_p; ++a) {
                for (int b = 0; b < npno_p; ++b) {
                    double T_ab, T_ba;
                    if (swap_ki) {
                        T_ab = T2_p[b * npno_p + a];
                        T_ba = T2_p[a * npno_p + b];
                    } else {
                        T_ab = T2_p[a * npno_p + b];
                        T_ba = T2_p[b * npno_p + a];
                    }
                    Tt_ki[(int64_t)a * npno_p + b] = 2.0 * T_ab - T_ba;
                }
            }

            // Y[c] = Σ_R K_chem_ki[R, c] * Tt_ki.flat[R]
            // K_chem_ki has linear layout from (npno_p, npno_p²) row-major,
            // viewed as (npno_p², npno_p) post-reshape: K[R, c] at offset R*npno_p + c.
            std::vector<double> Y((size_t)npno_p, 0.0);
            const int npno_p2 = npno_p * npno_p;
            for (int c = 0; c < npno_p; ++c) {
                double s = 0.0;
                for (int R = 0; R < npno_p2; ++R) {
                    s += K_chem_ki[(int64_t)R * npno_p + c] * Tt_ki[R];
                }
                Y[(size_t)c] = s;
            }

            // R1[a_ord, a_ii] += Σ_c S_p[c, a_ii] * Y[c]
            const int64_t r1_off = in_.pno_offsets[a_ord];
            for (int a_ii = 0; a_ii < npno_ii; ++a_ii) {
                double s = 0.0;
                for (int c = 0; c < npno_p; ++c) {
                    s += S_p[(int64_t)c * npno_ii + a_ii] * Y[(size_t)c];
                }
                R1_t[r1_off + a_ii] += s;
            }

            // -- C term --------------------------------------------------
            // T2 swap for "ik": swap iff can_first != a_ord.
            const bool swap_ik = (can_first != a_ord);
            std::vector<double> Tt_ik((size_t)npno_p * npno_p);
            for (int a = 0; a < npno_p; ++a) {
                for (int b = 0; b < npno_p; ++b) {
                    double T_ab, T_ba;
                    if (swap_ik) {
                        T_ab = T2_p[b * npno_p + a];
                        T_ba = T2_p[a * npno_p + b];
                    } else {
                        T_ab = T2_p[a * npno_p + b];
                        T_ba = T2_p[b * npno_p + a];
                    }
                    Tt_ik[(int64_t)a * npno_p + b] = 2.0 * T_ab - T_ba;
                }
            }

            // Build fkc_dress.  Two paths:
            //   1. Fkc_per_ordered (PySCF/Psi4-faithful, FULL scope including
            //      weak pairs).  Indexed by the CURRENT ordered pair `o` =
            //      (a_ord, b_ord) = (i, k) where i is the R1 owner.  PySCF
            //      builds _LT1_cache.get((i, m)) keyed by the R1 owner i.
            //      The Fkc array is in canonical pair (k,i)'s PNO basis,
            //      length npno_p.
            //   2. Else: legacy LT1-chain inline (strong-only).
            std::vector<double> fkc_dress((size_t)npno_p, 0.0);
            if (in_.Fkc_per_ordered.data != nullptr) {
                const double *Fkc_o = fps_ptr(in_.Fkc_per_ordered, o);
                for (int a = 0; a < npno_p; ++a) fkc_dress[a] = Fkc_o[a];
            } else {
                for (int m_ = 0; m_ < nocc; ++m_) {
                    const int o_im = ord_idx_lookup[(int64_t)a_ord * nocc + m_];
                    if (o_im < 0) continue;
                    const int p_im = in_.i_j_to_ij[(int64_t)a_ord * nocc + m_];
                    if (p_im < 0) continue;
                    const int npno_im = in_.n_pno_per_pair[p_im];
                    if (npno_im == 0) continue;
                    const double *lt1_im = lt1_flat.data() + lt1_offsets[o_im];

                    if (p_canon == p_im) {
                        for (int a = 0; a < npno_p; ++a) {
                            fkc_dress[a] += lt1_im[a];
                        }
                    } else {
                        const int64_t s_idx = (int64_t)p_canon * N_canon + p_im;
                        int64_t s_off, s_size; s_pno_lookup(in_, s_idx, s_off, s_size);
                        if (s_size != (int64_t)npno_p * npno_im) continue;
                        const double *S_p_im = in_.S_pno_data + s_off;
                        for (int a = 0; a < npno_p; ++a) {
                            double s = 0.0;
                            for (int b = 0; b < npno_im; ++b) {
                                s += S_p_im[a * npno_im + b] * lt1_im[b];
                            }
                            fkc_dress[a] += s;
                        }
                    }
                }
            }

            // contrib_C_local[a] = Σ_b Tt_ik[a, b] * fkc_dress[b]
            std::vector<double> contrib_C((size_t)npno_p, 0.0);
            for (int a = 0; a < npno_p; ++a) {
                double s = 0.0;
                for (int b = 0; b < npno_p; ++b) {
                    s += Tt_ik[(int64_t)a * npno_p + b] * fkc_dress[b];
                }
                contrib_C[a] = s;
            }

            // R1[a_ord] += S_PNO((i,i)_canon, (k,i)_canon) @ contrib_C
            // (direct add when (i,i)_canon == (k,i)_canon).
            if (p_canon == p_ii) {
                for (int a = 0; a < npno_p; ++a) {
                    R1_t[r1_off + a] += contrib_C[a];
                }
            } else {
                const int64_t s_idx2 = (int64_t)p_ii * N_canon + p_canon;
                int64_t s_off2, s_size2; s_pno_lookup(in_, s_idx2, s_off2, s_size2);
                if (s_size2 != (int64_t)npno_ii * npno_p) continue;
                const double *S_ii_ki = in_.S_pno_data + s_off2;
                for (int a = 0; a < npno_ii; ++a) {
                    double s = 0.0;
                    for (int b = 0; b < npno_p; ++b) {
                        s += S_ii_ki[a * npno_p + b] * contrib_C[b];
                    }
                    R1_t[r1_off + a] += s;
                }
            }
        }
    }

    // Reduce per-thread buffers into out->R1_flat.
    for (int tid = 0; tid < num_threads; ++tid) {
        const double *R1_t = R1_thread.data() + (size_t)tid * total_R1;
        for (int e = 0; e < total_R1; ++e) {
            out->R1_flat[e] += R1_t[e];
        }
    }
}

void DLPNOCCSDSolver::run_phase_t1_fock_fia_bar_into(FiaBarOutputs *out) {
    const int N = in_.n_canon_pairs;
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0, neg_one = -1.0, two = 2.0;
    const int int_one = 1;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < N; ++p) {
        if (in_.is_strong_pair != nullptr && in_.is_strong_pair[p] == 0) continue;
        const int npno = in_.n_pno_per_pair[p];
        if (npno == 0) continue;

        const int64_t lmo_off = in_.pair_lmo_idx_offsets[p];
        const int nlmo = (int)(in_.pair_lmo_idx_offsets[p + 1] - lmo_off);
        if (nlmo == 0) continue;

        const int64_t qma_size = in_.Qma.offsets[p + 1] - in_.Qma.offsets[p];
        const int64_t per_q = (int64_t)nlmo * (int64_t)npno;
        const int n_local = (per_q > 0) ? (int)(qma_size / per_q) : 0;

        const double *Qma = fps_ptr(in_.Qma, p);
        const double *T1l = fps_ptr(in_.T1_in_pair, p);
        double *Fia_bar = out->Fia_bar.data + out->Fia_bar.offsets[p];

        // Zero output up front (covers the n_local == 0 case too).
        for (int64_t e = 0; e < per_q; ++e) Fia_bar[e] = 0.0;
        if (n_local == 0) continue;

        const int int_nlmo = nlmo;
        const int int_npno = npno;
        const int int_n_local = n_local;
        const int int_per_q = (int)per_q;

        // Step 1: gamma[Q] = sum_{m,a} Qma[Q, m, a] * T1l[m, a]
        // Treat Qma as (n_local, per_q) and T1l as (per_q,).
        // gamma = Qma_flat @ T1l   (matrix-vector).
        // In Fortran: Qma_flat row-major (n_local, per_q) → col-major (per_q, n_local).
        // dgemv('T', m=per_q, n=n_local, alpha=1, A=Qma_flat, lda=per_q,
        //       x=T1l, incx=1, beta=0, y=gamma, incy=1).
        std::vector<double> gamma((size_t)n_local, 0.0);
        dgemv_(&T_flag, &int_per_q, &int_n_local,
               &one, Qma, &int_per_q,
               T1l, &int_one,
               &zero, gamma.data(), &int_one);

        // Step 2: Z[Q, n, k] = sum_b T1l[n, b] * Qma[Q, k, b]
        // Per Q small GEMM: Z_Q (nlmo, nlmo) = T1l (nlmo, npno) @ Qma_Q^T (npno, nlmo)
        // Fortran: Z_Q_F[k, n] = sum_b Qma_Q_F[b, k] * T1l_F[b, n]
        //   = Qma_Q^T @ T1l (col-major view) → dgemm('T', 'N', nlmo, nlmo, npno,
        //                                             1, Qma_Q, npno, T1l, npno, 0, Z_Q, nlmo)
        std::vector<double> Z((size_t)n_local * nlmo * nlmo, 0.0);
        const int64_t Z_stride = (int64_t)nlmo * nlmo;
        for (int Q = 0; Q < n_local; ++Q) {
            const double *Qma_Q = Qma + (int64_t)Q * per_q;
            double *Z_Q = Z.data() + (int64_t)Q * Z_stride;
            dgemm_(&T_flag, &N_flag,
                   &int_nlmo, &int_nlmo, &int_npno,
                   &one, Qma_Q, &int_npno,
                   T1l, &int_npno,
                   &zero, Z_Q, &int_nlmo);
        }

        // Step 3a: Fia_pos[k, a] = sum_Q gamma[Q] * Qma[Q, k, a]
        // = gamma (n_local) @ Qma_flat (n_local, per_q) → result (per_q,)
        // dgemv('N', m=per_q, n=n_local, alpha=1, A=Qma, lda=per_q, x=gamma, incx=1, beta=0, y=Fia_pos)
        std::vector<double> Fia_pos((size_t)per_q, 0.0);
        dgemv_(&N_flag, &int_per_q, &int_n_local,
               &one, Qma, &int_per_q,
               gamma.data(), &int_one,
               &zero, Fia_pos.data(), &int_one);

        // Step 3b: Fia_neg[k, a] = sum_{Q, n} Qma[Q, n, a] * Z[Q, n, k]
        // View Qma_view[Q*nlmo+n, a] = Qma[Q, n, a]; Z_view[Q*nlmo+n, k] = Z[Q, n, k].
        // Fia_neg[k, a] = sum_{Qn} Z_view[Qn, k] * Qma_view[Qn, a] = Z_view^T @ Qma_view → (nlmo, npno)
        // Fortran: result_F[a, k] = sum_{Qn} Qma_F[a, Qn] * Z_F[k, Qn]
        //   = Qma_F @ Z_F^T (col-major view of (n_local*nlmo, npno) and (n_local*nlmo, nlmo))
        // dgemm('N', 'T', m=npno, n=nlmo, k=n_local*nlmo, 1, Qma, npno, Z, nlmo, 0, Fia_neg, npno)
        std::vector<double> Fia_neg((size_t)per_q, 0.0);
        const int int_nl_nlmo = (int)((int64_t)n_local * nlmo);
        dgemm_(&N_flag, &T_flag,
               &int_npno, &int_nlmo, &int_nl_nlmo,
               &one, Qma, &int_npno,
               Z.data(), &int_nlmo,
               &zero, Fia_neg.data(), &int_npno);

        // Final: Fia_bar[k, a] = 2 * Fia_pos[k, a] - Fia_neg[k, a]
        for (int64_t e = 0; e < per_q; ++e) {
            Fia_bar[e] = 2.0 * Fia_pos[(size_t)e] - Fia_neg[(size_t)e];
        }
        (void)neg_one; (void)two;
    }
}

void DLPNOCCSDSolver::run_phase_update_amps_and_energy_into(
    const UpdateAmpsInputs *resid, UpdateAmpsOutputs *out) {
    const int nocc = in_.nocc;
    const int N    = in_.n_canon_pairs;
    const double DENOM_FLOOR = 1e-12;

    // T1 update: T1[i, a] -= R1[i, a] / (e_pno_ii[a] - F_lmo[i, i]).
    // Diagonal canonical pair (i, i) holds the e_pno used here.  Convention:
    // i_j_to_ij[i*nocc + i] gives the canonical pair index for diagonal (i,i).
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < nocc; ++i) {
        const int p_ii = in_.i_j_to_ij[(int64_t)i * nocc + i];
        if (p_ii < 0) continue;
        const int npno_ii = in_.n_pno_per_pair[p_ii];
        if (npno_ii == 0) continue;

        const int64_t t1_off  = in_.pno_offsets[i];
        const int64_t epno_off = in_.pno_offsets[p_ii];
        const double F_ii = in_.F_lmo[(int64_t)i * nocc + i];

        double *T1_i       = in_.T1_flat + t1_off;
        const double *R1_i = resid->R1_flat + t1_off;
        const double *e_p  = in_.e_pno_flat + epno_off;

        for (int a = 0; a < npno_ii; ++a) {
            double denom = e_p[a] - F_ii;
            if (denom > -DENOM_FLOOR && denom < DENOM_FLOOR) denom = DENOM_FLOOR;
            T1_i[a] -= R1_i[a] / denom;
        }
    }

    // T2 update.  Weak pairs are skipped — PySCF only updates T2 for
    // strong pairs (compute_residual_v2 is only called for strong pairs;
    // R2_external for weak pairs is zero so T2_weak stays unchanged).
    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < N; ++p) {
        if (in_.is_strong_pair != nullptr && in_.is_strong_pair[p] == 0) continue;
        const int npno = in_.n_pno_per_pair[p];
        if (npno == 0) continue;
        const int i = in_.ij_to_i_j[2 * p];
        const int j = in_.ij_to_i_j[2 * p + 1];

        const int64_t t2_off = in_.t2_offsets[p];
        const double F_ii = in_.F_lmo[(int64_t)i * nocc + i];
        const double F_jj = in_.F_lmo[(int64_t)j * nocc + j];
        const double *e_p = in_.e_pno_flat + in_.pno_offsets[p];

        double *T2_p       = in_.T2_flat + t2_off;
        const double *R2_p = resid->R2_flat + t2_off;

        for (int a = 0; a < npno; ++a) {
            for (int b = 0; b < npno; ++b) {
                double denom = e_p[a] + e_p[b] - F_ii - F_jj;
                if (denom > -DENOM_FLOOR && denom < DENOM_FLOOR) denom = DENOM_FLOOR;
                T2_p[a * npno + b] -= R2_p[a * npno + b] / denom;
            }
        }
    }

    // Energy: T1 contribution.
    double e_T1 = 0.0;
    #pragma omp parallel for reduction(+:e_T1) schedule(static)
    for (int i = 0; i < nocc; ++i) {
        const int p_ii = in_.i_j_to_ij[(int64_t)i * nocc + i];
        if (p_ii < 0) continue;
        const int npno_ii = in_.n_pno_per_pair[p_ii];
        if (npno_ii == 0) continue;
        const int64_t t1_off = in_.pno_offsets[i];
        const double *T1_i  = in_.T1_flat  + t1_off;
        const double *fov_i = in_.fov_flat + t1_off;
        double s = 0.0;
        for (int a = 0; a < npno_ii; ++a) s += fov_i[a] * T1_i[a];
        e_T1 += s;
    }

    // Energy: T2 contribution.
    //   tau = T2[p] + outer(t1_i_pno, t1_j_pno);  Tt = 2*tau - tau.T
    //   e_p = K_iajb[p] : Tt;  add (1 or 2)*e_p depending on diag.
    //   Skip weak pairs when is_strong_pair[] is provided (Psi4/PySCF
    //   correlation energy formula counts only strong pairs).
    //
    // T1_in_pair was built from PRE-UPDATE t1_pno; after the T1 update
    // above it is STALE.  To form tau with NEW t1, project T1_flat[i]
    // (in pair (i,i)'s PNO basis) into pair p's basis inline via
    // S_PNO(p, (i,i)) — same projection that t1_cache encoded.
    const int N_canon = in_.n_canon_pairs;
    double e_T2 = 0.0;
    #pragma omp parallel for reduction(+:e_T2) schedule(dynamic, 1)
    for (int p = 0; p < N; ++p) {
        if (in_.is_strong_pair != nullptr && in_.is_strong_pair[p] == 0) continue;
        const int npno = in_.n_pno_per_pair[p];
        if (npno == 0) continue;
        const int i = in_.ij_to_i_j[2 * p];
        const int j = in_.ij_to_i_j[2 * p + 1];
        const int p_ii = in_.i_j_to_ij[(int64_t)i * nocc + i];
        const int p_jj = in_.i_j_to_ij[(int64_t)j * nocc + j];
        if (p_ii < 0 || p_jj < 0) continue;
        const int npno_ii = in_.n_pno_per_pair[p_ii];
        const int npno_jj = in_.n_pno_per_pair[p_jj];

        const double *T2_p =
            in_.T2_flat + in_.t2_offsets[p];
        const double *K_p = fps_ptr(in_.K_iajb, p);

        // Project T1_flat[i] from (i,i)'s PNO basis to pair p's PNO basis
        // via S_PNO(p, (i,i)) which has shape (npno_p, npno_ii).  When
        // S_pno_data is null and p != p_ii, fall back to the (stale)
        // T1_in_pair view — preserves legacy behavior for tests that
        // don't supply S_pno_cache.
        std::vector<double> t1_i_in_p((size_t)npno, 0.0);
        std::vector<double> t1_j_in_p((size_t)npno, 0.0);
        if (in_.S_pno_data == nullptr || in_.S_pno_offsets == nullptr) {
            // Legacy: read t1_i / t1_j from T1_in_pair via i_in_p / j_in_p.
            const int64_t lmo_off = in_.pair_lmo_idx_offsets[p];
            const int nlmo = (int)(in_.pair_lmo_idx_offsets[p + 1] - lmo_off);
            const int *lmo_list = in_.pair_lmo_idx_flat + lmo_off;
            int i_in_p = -1, j_in_p = -1;
            for (int k = 0; k < nlmo; ++k) {
                if (lmo_list[k] == i) i_in_p = k;
                if (lmo_list[k] == j) j_in_p = k;
            }
            if (i_in_p < 0 || j_in_p < 0) continue;
            const double *T1_pair = fps_ptr(in_.T1_in_pair, p);
            for (int a = 0; a < npno; ++a) {
                t1_i_in_p[a] = T1_pair[(int64_t)i_in_p * npno + a];
                t1_j_in_p[a] = T1_pair[(int64_t)j_in_p * npno + a];
            }
        } else {
            // S_pno-based projection (Psi4-faithful, uses NEW T1_flat).
            if (p == p_ii) {
                const double *T1_i_native = in_.T1_flat + in_.pno_offsets[i];
                for (int a = 0; a < npno; ++a) t1_i_in_p[a] = T1_i_native[a];
            } else {
                const int64_t s_idx = (int64_t)p * N_canon + p_ii;
                int64_t s_off, s_size; s_pno_lookup(in_, s_idx, s_off, s_size);
                if (s_size == (int64_t)npno * npno_ii) {
                    const double *S = in_.S_pno_data + s_off;
                    const double *T1_i_native = in_.T1_flat + in_.pno_offsets[i];
                    for (int a = 0; a < npno; ++a) {
                        double s = 0.0;
                        for (int b = 0; b < npno_ii; ++b) {
                            s += S[a * npno_ii + b] * T1_i_native[b];
                        }
                        t1_i_in_p[a] = s;
                    }
                }
            }
            if (i == j) {
                for (int a = 0; a < npno; ++a) t1_j_in_p[a] = t1_i_in_p[a];
            } else if (p == p_jj) {
                const double *T1_j_native = in_.T1_flat + in_.pno_offsets[j];
                for (int a = 0; a < npno; ++a) t1_j_in_p[a] = T1_j_native[a];
            } else {
                const int64_t s_idx = (int64_t)p * N_canon + p_jj;
                int64_t s_off, s_size; s_pno_lookup(in_, s_idx, s_off, s_size);
                if (s_size == (int64_t)npno * npno_jj) {
                    const double *S = in_.S_pno_data + s_off;
                    const double *T1_j_native = in_.T1_flat + in_.pno_offsets[j];
                    for (int a = 0; a < npno; ++a) {
                        double s = 0.0;
                        for (int b = 0; b < npno_jj; ++b) {
                            s += S[a * npno_jj + b] * T1_j_native[b];
                        }
                        t1_j_in_p[a] = s;
                    }
                }
            }
        }

        // e_p = Σ_{a, b} K[a, b] * (2*tau[a, b] - tau[b, a])
        // with tau[a, b] = T2[a, b] + t1_i[a] * t1_j[b].
        double s = 0.0;
        for (int a = 0; a < npno; ++a) {
            for (int b = 0; b < npno; ++b) {
                double tau_ab = T2_p[a * npno + b] + t1_i_in_p[a] * t1_j_in_p[b];
                double tau_ba = T2_p[b * npno + a] + t1_i_in_p[b] * t1_j_in_p[a];
                s += K_p[a * npno + b] * (2.0 * tau_ab - tau_ba);
            }
        }
        e_T2 += (i == j) ? s : 2.0 * s;
    }

    out->energy = e_T1 + e_T2;
}

void DLPNOCCSDSolver::run_phase_k_ladder_into(
    const KLadderInputs *t1_dressed, KLadderOutputs *out) {
    const int n_pairs = in_.n_canon_pairs;
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0, neg_one = -1.0;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < n_pairs; ++p) {
        if (in_.is_strong_pair != nullptr && in_.is_strong_pair[p] == 0) continue;
        const int npno = in_.n_pno_per_pair[p];
        if (npno == 0) continue;

        const int64_t lmo_off = in_.pair_lmo_idx_offsets[p];
        const int nlmo = (int)(in_.pair_lmo_idx_offsets[p + 1] - lmo_off);

        // n_local = naux per pair, derived from Qma extent.
        const int64_t qma_size = in_.Qma.offsets[p + 1] - in_.Qma.offsets[p];
        const int64_t per_q_qma = (int64_t)nlmo * (int64_t)npno;
        const int n_local = (per_q_qma > 0) ? (int)(qma_size / per_q_qma) : 0;
        if (n_local == 0) continue;

        const double *iQa = t1_dressed->i_Qa_t1.data
                            + t1_dressed->i_Qa_t1.offsets[p];
        const double *jQa = t1_dressed->j_Qa_t1.data
                            + t1_dressed->j_Qa_t1.offsets[p];
        const double *Qma = fps_ptr(in_.Qma, p);
        const double *Qab = fps_ptr(in_.Qab, p);
        const double *T1l = fps_ptr(in_.T1_in_pair, p);
        const double *T2  = in_.T2_flat + in_.t2_offsets[p];

        double *K_out = out->K.data + out->K.offsets[p];
        double *A_out = out->A.data + out->A.offsets[p];

        int int_npno = npno;
        int int_nlmo = nlmo;
        int int_n_local = n_local;

        // K[a, b] = Σ_Q iQa[Q, a] * jQa[Q, b]
        // iQa, jQa row-major (n_local, npno). Math: K = iQa^T @ jQa → (npno, npno).
        // Fortran view: K_F[b, a] = sum_Q jQa_F[b, Q] * iQa_F[a, Q] = jQa_F @ iQa_F^T.
        // dgemm('N', 'T', npno, npno, n_local, 1, jQa, npno, iQa, npno, 0, K, npno).
        dgemm_(&N_flag, &T_flag,
               &int_npno, &int_npno, &int_n_local,
               &one, jQa, &int_npno,
               iQa, &int_npno,
               &zero, K_out, &int_npno);

        // A initialised to zero.
        std::memset(A_out, 0, sizeof(double) * (size_t)npno * npno);

        // Per-Q ladder accumulation. Per-thread scratch:
        //   Qab_t1: (npno, npno)
        //   QT:     (npno, npno) = Qab_t1 @ T2
        std::vector<double> Qab_t1((size_t)npno * npno);
        std::vector<double> QT((size_t)npno * npno);

        for (int Q = 0; Q < n_local; ++Q) {
            const double *Qab_Q = Qab + (int64_t)Q * npno * npno;
            const double *Qma_Q = Qma + (int64_t)Q * nlmo * npno;

            // Qab_t1[a, b] = Qab[Q, a, b] - Σ_n T1l[n, a] * Qma[Q, n, b]
            // Start from Qab[Q] then accumulate -T1l^T @ Qma_Q via dgemm.
            std::memcpy(Qab_t1.data(), Qab_Q,
                        sizeof(double) * (size_t)npno * npno);
            // Math: Qab_t1 (npno, npno) -= T1l^T (npno, nlmo) @ Qma_Q (nlmo, npno).
            // F view: (Qab_t1)_F[b, a] -= sum_n Qma_Q_F[b, n] * T1l_F[a, n]
            //   = Qma_Q_F @ T1l_F^T.
            // dgemm('N', 'T', npno, npno, nlmo, -1, Qma_Q, npno, T1l, npno, 1, Qab_t1, npno).
            dgemm_(&N_flag, &T_flag,
                   &int_npno, &int_npno, &int_nlmo,
                   &neg_one, Qma_Q, &int_npno,
                   T1l, &int_npno,
                   &one, Qab_t1.data(), &int_npno);

            // QT[a, d] = Σ_c Qab_t1[a, c] * T2[c, d]
            // Math: QT (npno, npno) = Qab_t1 (npno, npno) @ T2 (npno, npno).
            // F view: QT_F[d, a] = sum_c T2_F[d, c] * Qab_t1_F[c, a] = T2_F @ Qab_t1_F.
            // dgemm('N', 'N', npno, npno, npno, 1, T2, npno, Qab_t1, npno, 0, QT, npno).
            dgemm_(&N_flag, &N_flag,
                   &int_npno, &int_npno, &int_npno,
                   &one, T2, &int_npno,
                   Qab_t1.data(), &int_npno,
                   &zero, QT.data(), &int_npno);

            // A[a, b] += Σ_d QT[a, d] * Qab_t1[b, d]
            // Math: A += QT (npno, npno) @ Qab_t1^T (npno, npno).
            // F view: A_F[b, a] += sum_d Qab_t1_F[d, b] * QT_F[d, a] = Qab_t1_F^T @ QT_F.
            // dgemm('T', 'N', npno, npno, npno, 1, Qab_t1, npno, QT, npno, 1, A, npno).
            dgemm_(&T_flag, &N_flag,
                   &int_npno, &int_npno, &int_npno,
                   &one, Qab_t1.data(), &int_npno,
                   QT.data(), &int_npno,
                   &one, A_out, &int_npno);
        }
    }
}

void DLPNOCCSDSolver::run_phase_t3_into(
    const T3Inputs *plan, T3Outputs *out) {
    int num_threads = 1;
    #ifdef _OPENMP
        num_threads = solver_team_size();
        if (plan->N > 0 && num_threads > plan->N) num_threads = plan->N;
    #endif
    const size_t Kt1_stride    = (size_t)plan->max_n_kl;
    const size_t Kt1_ki_stride = (size_t)plan->max_n_ki;
    std::vector<double> Kt1_sc((size_t)num_threads * Kt1_stride);
    std::vector<double> Kt1_ki_sc((size_t)num_threads * Kt1_ki_stride);

    DLPNOt3_kernel_batched(
        plan->N,
        plan->n_kl_arr, plan->n_ki_arr,
        plan->K_off, plan->S_off,
        plan->t1i_off, plan->T1l_off, plan->tile_off,
        plan->K_flat, plan->S_flat, plan->t1_flat,
        Kt1_sc.data(),    Kt1_stride,
        Kt1_ki_sc.data(), Kt1_ki_stride,
        out->tiles_flat,
        num_threads);
}

void DLPNOCCSDSolver::run_phase_t4_into(
    const T4Inputs *plan, T4Outputs *out) {
    int num_threads = 1;
    #ifdef _OPENMP
        num_threads = solver_team_size();
        if (plan->N > 0 && num_threads > plan->N) num_threads = plan->N;
    #endif
    const size_t tmp1_stride = (size_t)plan->max_n_ki * plan->max_n_li;
    const size_t tmp2_stride = (size_t)plan->max_n_ki * plan->max_n_kl;
    const size_t tmp3_stride = (size_t)plan->max_n_ki * plan->max_n_kl;
    std::vector<double> tmp1_sc((size_t)num_threads * tmp1_stride);
    std::vector<double> tmp2_sc((size_t)num_threads * tmp2_stride);
    std::vector<double> tmp3_sc((size_t)num_threads * tmp3_stride);

    DLPNOt4_kernel_batched(
        plan->N,
        plan->n_ki_arr, plan->n_li_arr, plan->n_kl_arr,
        plan->S_ki_li_off, plan->t2_off, plan->S_li_kl_off,
        plan->K_off, plan->S_kl_ki_off, plan->tile_off,
        plan->S_ki_li_flat, plan->S_li_kl_flat,
        plan->K_flat, plan->S_kl_ki_flat,
        plan->t2_flat,
        tmp1_sc.data(), tmp1_stride,
        tmp2_sc.data(), tmp2_stride,
        tmp3_sc.data(), tmp3_stride,
        out->tiles_flat,
        plan->scale,
        num_threads);
}

void DLPNOCCSDSolver::run_phase_g_term_into(
    const GTermInputs *plan, GTermOutputs *out) {
    int num_threads = 1;
    #ifdef _OPENMP
        num_threads = solver_team_size();
        if (plan->N > 0 && num_threads > plan->N) num_threads = plan->N;
    #endif
    const size_t tmp_stride = (size_t)plan->max_n_ij * plan->max_n_ik;
    std::vector<double> tmp_sc((size_t)num_threads * tmp_stride);

    DLPNOg_term_batched(
        plan->N,
        plan->n_ij_arr, plan->n_ik_arr,
        plan->S_off, plan->t2_off, plan->tile_off,
        plan->k_idx, plan->scalar_lmo,
        plan->S_flat, plan->t2_flat,
        plan->G_tilde, (size_t)plan->G_stride,
        tmp_sc.data(), tmp_stride,
        out->tiles_flat,
        num_threads);
}

void DLPNOCCSDSolver::run_phase_c_term_into(
    const CTermInputs *plan, CTermOutputs *out) {
    int num_threads = 1;
    #ifdef _OPENMP
        num_threads = solver_team_size();
        if (plan->N > 0 && num_threads > plan->N) num_threads = plan->N;
    #endif
    const size_t STB_stride   = (size_t)plan->max_n_pno * plan->max_n_ct;
    const size_t GAMMA_stride = (size_t)plan->max_n_pno * plan->max_n_other;
    const size_t GT_stride    = (size_t)plan->max_n_pno * plan->max_n_other;

    std::vector<double> STB_sc((size_t)num_threads * STB_stride);
    std::vector<double> GAMMA_sc((size_t)num_threads * GAMMA_stride);
    std::vector<double> GT_sc((size_t)num_threads * GT_stride);

    DLPNOc_term_batched(
        plan->N,
        plan->n_pno_arr, plan->n_ct_arr, plan->n_other_arr,
        plan->S_big_off, plan->ct_off, plan->S_mid_off,
        plan->J_bold_off, plan->t2_off, plan->S_outer_off, plan->tile_off,
        plan->S_big_flat, plan->S_mid_flat,
        plan->J_bold_flat, plan->S_outer_flat,
        plan->ct_flat, plan->t2_flat,
        plan->t2_trans,
        STB_sc.data(),   STB_stride,
        GAMMA_sc.data(), GAMMA_stride,
        GT_sc.data(),    GT_stride,
        out->tiles_flat,
        num_threads);
}

void DLPNOCCSDSolver::run_phase_d_term_into(
    const DTermInputs *plan, DTermOutputs *out) {
    int num_threads = 1;
    #ifdef _OPENMP
        num_threads = solver_team_size();
        if (plan->N > 0 && num_threads > plan->N) num_threads = plan->N;
    #endif
    const size_t U_stride    = (size_t)plan->max_n_A * plan->max_n_A;
    const size_t SU_stride   = (size_t)plan->max_n_pno * plan->max_n_A;
    const size_t UP_stride   = (size_t)plan->max_n_pno * plan->max_n_B;
    const size_t SCD_stride  = (size_t)plan->max_n_pno * plan->max_n_B;
    const size_t Bint_stride = (size_t)plan->max_n_pno * plan->max_n_A;

    // U scratch only needed when computing u in-kernel from aliased t2.
    std::vector<double> U_sc(plan->u_base ? (size_t)num_threads * U_stride : 0);
    std::vector<double> SU_sc((size_t)num_threads * SU_stride);
    std::vector<double> UP_sc((size_t)num_threads * UP_stride);
    std::vector<double> SCD_sc((size_t)num_threads * SCD_stride);
    std::vector<double> Bint_sc((size_t)num_threads * Bint_stride);

    DLPNOd_term_batched(
        plan->N,
        plan->n_pno_arr, plan->n_A_arr, plan->n_B_arr,
        plan->S_a_off, plan->u_off, plan->S_b_off, plan->S_c_off,
        plan->dt_off, plan->KJ_off, plan->tile_off,
        plan->S_a_flat, plan->S_b_flat, plan->S_c_flat,
        plan->KJ_flat, plan->u_flat, plan->dt_flat,
        plan->u_base, plan->u_canon_off, plan->u_trans,
        U_sc.data(), U_stride,
        SU_sc.data(),   SU_stride,
        UP_sc.data(),   UP_stride,
        SCD_sc.data(),  SCD_stride,
        Bint_sc.data(), Bint_stride,
        out->tiles_flat,
        num_threads);
}

void DLPNOCCSDSolver::run_phase_be_into(
    const BEInputs *plan, BEOutputs *out) {
    int num_threads = 1;
    #ifdef _OPENMP
        num_threads = solver_team_size();
        if (plan->N > 0 && num_threads > plan->N) num_threads = plan->N;
    #endif
    if (plan->S_master != nullptr && plan->T_master != nullptr
            && plan->K_master != nullptr) {
        DLPNObe_kernel_gathered(
            plan->S_master, plan->S_off,
            plan->T_master, plan->T_off,
            plan->K_master, plan->K_off,
            plan->beta_kl, plan->beta_lk,
            plan->same, plan->idx,
            out->out_B, out->out_E,
            (size_t)plan->N, (size_t)plan->n_ij, (size_t)plan->n_kl,
            num_threads);
    } else {
        DLPNObe_kernel(
            plan->S, plan->T, plan->K,
            plan->beta_kl, plan->beta_lk,
            plan->same, plan->idx,
            out->out_B, out->out_E,
            (size_t)plan->N, (size_t)plan->n_ij, (size_t)plan->n_kl,
            num_threads);
    }
}

void DLPNOCCSDSolver::run_phase_t1_residual_per_kl_into(
    const PerKlPlanInputs *plan, PerKlOutputs *out) {
    int num_threads = 1;
    #ifdef _OPENMP
        num_threads = solver_team_size();
        if (plan->n_tasks > 0 && num_threads > plan->n_tasks)
            num_threads = plan->n_tasks;
    #endif

    const int M       = plan->M;
    const int max_kl  = plan->max_n_kl;
    const int max_ki  = plan->max_n_ki;

    const size_t Tt_kl_stride  = (size_t)max_kl * max_kl;
    const size_t K_kilc_stride = (size_t)M * max_kl;
    const size_t B_ia_stride   = (size_t)max_kl * M;
    const size_t Tt_ki_stride  = (size_t)max_ki * max_ki;
    const size_t X_stride      = (size_t)max_ki * max_kl;
    const size_t Z_stride      = (size_t)max_ki * max_ki;

    std::vector<double> Tt_kl_sc((size_t)num_threads * Tt_kl_stride);
    std::vector<double> K_kilc_sc((size_t)num_threads * K_kilc_stride);
    std::vector<double> B_ia_sc((size_t)num_threads * B_ia_stride);
    std::vector<double> Tt_ki_sc((size_t)num_threads * Tt_ki_stride);
    std::vector<double> X_sc((size_t)num_threads * X_stride);
    std::vector<double> Z_sc((size_t)num_threads * Z_stride);

    DLPNOper_kl_batched(
        plan->n_tasks, plan->M,
        plan->n_kl_arr, plan->t2_swap_kl,
        plan->K_iajb_kl_off, plan->K_bar_kl_off,
        plan->t2_kl_canon_off, plan->T_n_kl_off,
        plan->inner_off,
        plan->i_arr, plan->n_pno_ii_arr,
        plan->is_diag_kl_ii, plan->has_S_ii_kl, plan->S_ii_kl_off,
        plan->has_A2, plan->is_diag_kl_ki,
        plan->n_ki_arr, plan->t2_swap_ki, plan->t2_ki_canon_off,
        plan->S_kl_ki_off, plan->S_ki_kl_off,
        plan->T_n_l_ii_off, plan->contrib_off,
        plan->K_iajb_buffer, plan->K_bar_kl_static, plan->S_pno_buffer,
        plan->t2_buffer, plan->t1_cache_buffer,
        Tt_kl_sc.data(),  Tt_kl_stride,
        K_kilc_sc.data(), K_kilc_stride,
        B_ia_sc.data(),   B_ia_stride,
        Tt_ki_sc.data(),  Tt_ki_stride,
        X_sc.data(),      X_stride,
        Z_sc.data(),      Z_stride,
        out->contrib_flat,
        num_threads);
}

void DLPNOCCSDSolver::run_phase_t1_fock_finalize_into(
    const T1FockExtraInputs *extra, T1FockExtraOutputs *out) {
    const int nocc   = in_.nocc;
    const int N      = in_.n_canon_pairs;
    const int64_t nn = (int64_t)nocc * nocc;

    // Step A: Fkj = F_lmo + scatter(d_flat).
    std::memcpy(out->Fkj, in_.F_lmo, (size_t)nn * sizeof(double));
    for (int p = 0; p < N; ++p) {
        const int i = in_.ij_to_i_j[2 * p];
        const int j = in_.ij_to_i_j[2 * p + 1];
        out->Fkj[(int64_t)i * nocc + j] += extra->d_flat[2 * p];
        if (i != j) {
            out->Fkj[(int64_t)j * nocc + i] += extra->d_flat[2 * p + 1];
        }
    }

    // Step B: snapshot Fij_bar before Eq 94.
    std::memcpy(out->Fij_bar_snapshot, out->Fkj, (size_t)nn * sizeof(double));

    // Step C: Eq 94 — per occupied j, build Fia_bar_jj from Qma[(j,j)] +
    // T1_in_pair[(j,j)], accumulate (Fia_bar_jj @ t1_j) into Fkj[lmo_list, j].
    // BLAS-based per-j; same identity as run_phase_t1_fock_fia_bar_into.
    {
    const char N_flag = 'N', T_flag = 'T';
    const double one = 1.0, zero = 0.0;
    const int int_one = 1;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < nocc; ++j) {
        const int p_jj = in_.i_j_to_ij[(int64_t)j * nocc + j];
        if (p_jj < 0) continue;
        const int npno = in_.n_pno_per_pair[p_jj];
        if (npno == 0) continue;

        const int64_t lmo_off = in_.pair_lmo_idx_offsets[p_jj];
        const int nlmo = (int)(in_.pair_lmo_idx_offsets[p_jj + 1] - lmo_off);
        const int *lmo_list = in_.pair_lmo_idx_flat + lmo_off;

        const int64_t qma_size = in_.Qma.offsets[p_jj + 1] - in_.Qma.offsets[p_jj];
        const int64_t per_q = (int64_t)nlmo * npno;
        const int n_local = (per_q > 0) ? (int)(qma_size / per_q) : 0;
        if (n_local == 0) continue;

        const double *Qma = fps_ptr(in_.Qma, p_jj);
        const double *T1_local =
            fps_ptr(in_.T1_in_pair, p_jj);

        // Find j's row position in this pair's lmo_list.
        int j_in_p = -1;
        for (int k = 0; k < nlmo; ++k) {
            if (lmo_list[k] == j) { j_in_p = k; break; }
        }
        if (j_in_p < 0) continue;
        const double *t1_j = T1_local + (int64_t)j_in_p * npno;

        const int int_nlmo = nlmo;
        const int int_npno = npno;
        const int int_n_local = n_local;
        const int int_per_q = (int)per_q;
        const int int_nl_nlmo = (int)((int64_t)n_local * nlmo);

        // gamma[Q] = sum_{m,a} Qma[Q, m, a] * T1_local[m, a]
        // Qma_flat row (n_local, per_q) @ T1 (per_q,) → gamma (n_local,)
        std::vector<double> gamma((size_t)n_local, 0.0);
        dgemv_(&T_flag, &int_per_q, &int_n_local,
               &one, Qma, &int_per_q,
               T1_local, &int_one,
               &zero, gamma.data(), &int_one);

        // Z[Q, n, k] = sum_b T1_local[n, b] * Qma[Q, k, b]
        // Per-Q small dgemm: Z_Q (nlmo, nlmo) = T1 (nlmo, npno) @ Qma_Q^T (npno, nlmo)
        std::vector<double> Z((size_t)n_local * nlmo * nlmo, 0.0);
        const int64_t Z_stride = (int64_t)nlmo * nlmo;
        for (int Q = 0; Q < n_local; ++Q) {
            const double *Qma_Q = Qma + (int64_t)Q * per_q;
            double *Z_Q = Z.data() + (int64_t)Q * Z_stride;
            dgemm_(&T_flag, &N_flag,
                   &int_nlmo, &int_nlmo, &int_npno,
                   &one, Qma_Q, &int_npno,
                   T1_local, &int_npno,
                   &zero, Z_Q, &int_nlmo);
        }

        // Fia_pos[k, a] = sum_Q gamma[Q] * Qma[Q, k, a]
        std::vector<double> Fia_pos((size_t)per_q, 0.0);
        dgemv_(&N_flag, &int_per_q, &int_n_local,
               &one, Qma, &int_per_q,
               gamma.data(), &int_one,
               &zero, Fia_pos.data(), &int_one);

        // Fia_neg[k, a] = sum_{Q, n} Qma[Q, n, a] * Z[Q, n, k]
        // = (npno, nlmo) result of Qma_F @ Z_F^T (col-major view).
        std::vector<double> Fia_neg((size_t)per_q, 0.0);
        dgemm_(&N_flag, &T_flag,
               &int_npno, &int_nlmo, &int_nl_nlmo,
               &one, Qma, &int_npno,
               Z.data(), &int_nlmo,
               &zero, Fia_neg.data(), &int_npno);

        // Scatter (Fia_bar[k, :] = 2*Fia_pos - Fia_neg) @ t1_j → Fkj[lmo_list[k], j]
        for (int k = 0; k < nlmo; ++k) {
            double s = 0.0;
            for (int a = 0; a < npno; ++a) {
                s += (2.0 * Fia_pos[(int64_t)k * npno + a]
                      - Fia_neg[(int64_t)k * npno + a]) * t1_j[a];
            }
            out->Fkj[(int64_t)lmo_list[k] * nocc + j] += s;
        }
    }
    }

    // foo_t1 = Fkj - F_lmo
    for (int64_t e = 0; e < nn; ++e) {
        out->foo_t1[e] = out->Fkj[e] - in_.F_lmo[e];
    }
}

void DLPNOCCSDSolver::run_phase_d_tilde_ph1_into(DTildeOutputs *out) {
    const int N = in_.n_ordered_pairs;
    const int nocc = in_.nocc;

    // Per-ordered-pair sizing + canonical mapping.
    std::vector<int> can_p_arr(N);
    std::vector<int> n_pno_arr(N), n_domain_arr(N);
    std::vector<int> i_in_p_arr(N);
    // For each ordered pair, choose K_tilde_chem (i or j variant) and
    // K_bar (ij or ji variant) per Psi4 orientation rules.
    std::vector<const double*> K_tilde_chem_src(N);
    std::vector<int64_t>       K_tilde_chem_size(N);
    std::vector<const double*> K_bar_src(N);
    std::vector<int64_t>       K_bar_size(N);
    std::vector<const double*> K_bar_chem_src(N);
    int64_t kt_total = 0, M_total = 0, t1_total = 0, T1r_total = 0, D_total = 0;

    for (int o = 0; o < N; ++o) {
        const int i = in_.ordered_pair_i_idx[o];
        const int k = in_.ordered_pair_k_idx[o];
        const int p = in_.i_j_to_ij[i * nocc + k];
        can_p_arr[o] = p;
        const int can_i = in_.ij_to_i_j[2 * p];
        const int can_j = in_.ij_to_i_j[2 * p + 1];
        const bool is_weak = (in_.is_strong_pair != nullptr
                               && p >= 0 && in_.is_strong_pair[p] == 0);

        const int npno = is_weak ? 0 : in_.n_pno_per_pair[p];
        const int64_t lmo_off = in_.pair_lmo_idx_offsets[p];
        const int nlmo = is_weak ? 0
            : (int)(in_.pair_lmo_idx_offsets[p + 1] - lmo_off);
        n_pno_arr[o] = npno;
        n_domain_arr[o] = nlmo;

        // Find i's position within pair p's lmo list.
        const int *lmo_list = in_.pair_lmo_idx_flat + lmo_off;
        int i_in_p = -1;
        for (int kk = 0; kk < nlmo; ++kk) {
            if (lmo_list[kk] == i) { i_in_p = kk; break; }
        }
        i_in_p_arr[o] = i_in_p;

        // Orientation for K_tilde_chem: pick "_i" if canonical FIRST index is k.
        const FlatPairStore &kt = (can_i == k) ? in_.K_tilde_chem_i
                                                : in_.K_tilde_chem_j;
        K_tilde_chem_src[o]  = kt.data + kt.offsets[p];
        K_tilde_chem_size[o] = kt.offsets[p + 1] - kt.offsets[p];

        // Orientation for K_bar: pick "_ij" if canonical FIRST index is i.
        const FlatPairStore &kb = (can_i == i) ? in_.K_bar_ij : in_.K_bar_ji;
        K_bar_src[o]  = fps_ptr(kb, p);
        K_bar_size[o] = kb.offsets[p + 1] - kb.offsets[p];
        K_bar_chem_src[o] = fps_ptr(in_.K_bar_chem, p);

        kt_total   += K_tilde_chem_size[o];
        M_total    += (int64_t)nlmo * (int64_t)npno;
        t1_total   += npno;
        T1r_total  += (int64_t)nlmo * (int64_t)npno;
        D_total    += (int64_t)npno * (int64_t)npno;
    }

    std::vector<int64_t> kt_off(N + 1, 0);
    std::vector<int64_t> M_off(N + 1, 0);
    std::vector<int64_t> t1_off(N + 1, 0);
    std::vector<int64_t> T1r_off(N + 1, 0);
    std::vector<int64_t> D_off(N + 1, 0);
    for (int o = 0; o < N; ++o) {
        kt_off[o + 1]   = kt_off[o]   + K_tilde_chem_size[o];
        M_off[o + 1]    = M_off[o]    + (int64_t)n_domain_arr[o] * n_pno_arr[o];
        t1_off[o + 1]   = t1_off[o]   + n_pno_arr[o];
        T1r_off[o + 1]  = T1r_off[o]  + (int64_t)n_domain_arr[o] * n_pno_arr[o];
        D_off[o + 1]    = D_off[o]    + (int64_t)n_pno_arr[o] * n_pno_arr[o];
    }

    std::vector<double> kt_flat((size_t)kt_total);
    std::vector<double> M_flat((size_t)M_total);
    std::vector<double> t1_flat((size_t)t1_total);
    std::vector<double> T1r_flat((size_t)T1r_total);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int o = 0; o < N; ++o) {
        const int p = can_p_arr[o];
        const int npno = n_pno_arr[o];
        const int nlmo = n_domain_arr[o];
        const int i_in_p = i_in_p_arr[o];

        // K_tilde_chem: copy raw bytes per orientation.
        std::memcpy(kt_flat.data() + kt_off[o], K_tilde_chem_src[o],
                    (size_t)K_tilde_chem_size[o] * sizeof(double));

        // M_static = 2 * K_bar - K_bar_chem (both shape (nlmo, npno)).
        const double *Kb = K_bar_src[o];
        const double *Kc = K_bar_chem_src[o];
        double *M = M_flat.data() + M_off[o];
        const int64_t mn = (int64_t)nlmo * npno;
        for (int64_t e = 0; e < mn; ++e) {
            M[e] = 2.0 * Kb[e] - Kc[e];
        }

        // t1[i] = T1_in_pair[p] row at i_in_p (length npno).
        const double *T1pair = fps_ptr(in_.T1_in_pair, p);
        double *t1 = t1_flat.data() + t1_off[o];
        const double *t1_row_src = T1pair + (int64_t)i_in_p * npno;
        std::memcpy(t1, t1_row_src, (size_t)npno * sizeof(double));

        // T1_rows = full T1_in_pair[p] (nlmo x npno).
        double *T1r = T1r_flat.data() + T1r_off[o];
        std::memcpy(T1r, T1pair, (size_t)mn * sizeof(double));
    }

    DLPNOcompute_D_tilde_ph1_batched(
        kt_flat.data(),  (const long *)kt_off.data(),
        M_flat.data(),   (const long *)M_off.data(),
        t1_flat.data(),  (const long *)t1_off.data(),
        T1r_flat.data(), (const long *)T1r_off.data(),
        n_pno_arr.data(), n_domain_arr.data(),
        out->D_tilde.data, (const long *)out->D_tilde.offsets,
        (size_t)N);
}

void DLPNOCCSDSolver::run_phase_g_tilde_inner_into(
    const GTildeInputs *plan, GTildeOutputs *out) {
    DLPNOcompute_G_tilde_inner(
        plan->triple_eff_offset,
        plan->triple_T2_pair_idx,
        plan->triple_n_lj,
        plan->ij_triple_starts,
        plan->ij_i_arr,
        plan->ij_j_arr,
        plan->effective_flat,
        in_.T2_flat,
        (const long *)in_.t2_offsets,
        out->G_tilde,
        (size_t)plan->n_ij_slots,
        (size_t)in_.nocc);
}

void DLPNOCCSDSolver::run_phase_c_tilde_ph1_into(CTildeOutputs *out) {
    const int N = in_.n_ordered_pairs;
    const int nocc = in_.nocc;

    std::vector<int> can_p_arr(N);
    std::vector<int> n_pno_arr(N), n_domain_arr(N);
    std::vector<int> i_for_t1_in_p_arr(N);
    std::vector<const double*> K_tilde_chem_src(N);
    std::vector<int64_t>       K_tilde_chem_size(N);
    std::vector<const double*> K_bar_chem_src(N);
    int64_t kt_total = 0, Kbc_total = 0, t1_total = 0, T1l_total = 0, C_total = 0;

    for (int o = 0; o < N; ++o) {
        // C_tilde wrapper: ordered tuple = (k, i).  We expose the ordered pair
        // as (ordered_pair_i_idx[o], ordered_pair_k_idx[o]) by our convention,
        // so map: a := ordered_pair_i_idx[o] (= "k" in C_tilde), b := ordered_pair_k_idx[o] (= "i").
        const int a = in_.ordered_pair_i_idx[o];
        const int b = in_.ordered_pair_k_idx[o];
        const int p = in_.i_j_to_ij[a * nocc + b];
        can_p_arr[o] = p;
        const int can_a = in_.ij_to_i_j[2 * p];
        const bool is_weak = (in_.is_strong_pair != nullptr
                               && p >= 0 && in_.is_strong_pair[p] == 0);

        const int npno = is_weak ? 0 : in_.n_pno_per_pair[p];
        const int64_t lmo_off = in_.pair_lmo_idx_offsets[p];
        const int nlmo = is_weak ? 0
            : (int)(in_.pair_lmo_idx_offsets[p + 1] - lmo_off);
        n_pno_arr[o] = npno;
        n_domain_arr[o] = nlmo;

        // C_tilde's "i" (index into T1) is the SECOND of the ordered pair.
        const int *lmo_list = in_.pair_lmo_idx_flat + lmo_off;
        int i_in_p = -1;
        for (int kk = 0; kk < nlmo; ++kk) {
            if (lmo_list[kk] == b) { i_in_p = kk; break; }
        }
        i_for_t1_in_p_arr[o] = i_in_p;

        // K_tilde_chem orientation: "_i" if canonical first == a (= "k").
        const FlatPairStore &kt = (can_a == a) ? in_.K_tilde_chem_i
                                                : in_.K_tilde_chem_j;
        K_tilde_chem_src[o]  = kt.data + kt.offsets[p];
        K_tilde_chem_size[o] = kt.offsets[p + 1] - kt.offsets[p];
        K_bar_chem_src[o]    = fps_ptr(in_.K_bar_chem, p);

        kt_total  += K_tilde_chem_size[o];
        Kbc_total += (int64_t)nlmo * (int64_t)npno;
        t1_total  += npno;
        T1l_total += (int64_t)nlmo * (int64_t)npno;
        C_total   += (int64_t)npno * (int64_t)npno;
    }

    std::vector<int64_t> kt_off(N + 1, 0);
    std::vector<int64_t> Kbc_off(N + 1, 0);
    std::vector<int64_t> t1_off(N + 1, 0);
    std::vector<int64_t> T1l_off(N + 1, 0);
    for (int o = 0; o < N; ++o) {
        kt_off[o + 1]  = kt_off[o]  + K_tilde_chem_size[o];
        Kbc_off[o + 1] = Kbc_off[o] + (int64_t)n_domain_arr[o] * n_pno_arr[o];
        t1_off[o + 1]  = t1_off[o]  + n_pno_arr[o];
        T1l_off[o + 1] = T1l_off[o] + (int64_t)n_domain_arr[o] * n_pno_arr[o];
    }

    std::vector<double> kt_flat((size_t)kt_total);
    std::vector<double> Kbc_flat((size_t)Kbc_total);
    std::vector<double> t1_flat((size_t)t1_total);
    std::vector<double> T1l_flat((size_t)T1l_total);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int o = 0; o < N; ++o) {
        const int p = can_p_arr[o];
        const int npno = n_pno_arr[o];
        const int nlmo = n_domain_arr[o];
        const int i_in_p = i_for_t1_in_p_arr[o];
        const int64_t mn = (int64_t)nlmo * npno;

        std::memcpy(kt_flat.data() + kt_off[o], K_tilde_chem_src[o],
                    (size_t)K_tilde_chem_size[o] * sizeof(double));
        std::memcpy(Kbc_flat.data() + Kbc_off[o], K_bar_chem_src[o],
                    (size_t)mn * sizeof(double));

        const double *T1pair = fps_ptr(in_.T1_in_pair, p);
        const double *t1_row = T1pair + (int64_t)i_in_p * npno;
        std::memcpy(t1_flat.data() + t1_off[o], t1_row,
                    (size_t)npno * sizeof(double));
        std::memcpy(T1l_flat.data() + T1l_off[o], T1pair,
                    (size_t)mn * sizeof(double));
    }

    DLPNOcompute_C_tilde_ph1_batched(
        kt_flat.data(),  (const long *)kt_off.data(),
        Kbc_flat.data(), (const long *)Kbc_off.data(),
        t1_flat.data(),  (const long *)t1_off.data(),
        T1l_flat.data(), (const long *)T1l_off.data(),
        n_pno_arr.data(), n_domain_arr.data(),
        out->C_tilde.data, (const long *)out->C_tilde.offsets,
        (size_t)N);
}

void DLPNOCCSDSolver::run_phase_b_tilde_into(
    const BTildeInputs *t1_dressed, BTildeOutputs *out) {
    const int n_pairs = in_.n_canon_pairs;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int p = 0; p < n_pairs; ++p) {
        if (in_.is_strong_pair != nullptr && in_.is_strong_pair[p] == 0) continue;
        const int npno = in_.n_pno_per_pair[p];
        if (npno == 0) continue;

        const int64_t lmo_off = in_.pair_lmo_idx_offsets[p];
        const int nlmo_p = (int)(in_.pair_lmo_idx_offsets[p + 1] - lmo_off);
        if (nlmo_p == 0) continue;

        const int64_t qma_size = in_.Qma.offsets[p + 1] - in_.Qma.offsets[p];
        const int64_t per_q = (int64_t)nlmo_p * (int64_t)npno;
        if (per_q == 0) continue;
        const size_t n_local = (size_t)(qma_size / per_q);
        if (n_local == 0) continue;

        const double *Qma_p = fps_ptr(in_.Qma, p);
        const double *T2_p  = in_.T2_flat + in_.t2_offsets[p];
        const double *iQk   = t1_dressed->i_Qk_t1.data
                              + t1_dressed->i_Qk_t1.offsets[p];
        const double *jQk   = t1_dressed->j_Qk_t1.data
                              + t1_dressed->j_Qk_t1.offsets[p];

        double *B_p = out->B_tilde.data + out->B_tilde.offsets[p];

        DLPNOcompute_B_tilde_pair(
            B_p, iQk, jQk, Qma_p, T2_p,
            n_local, (size_t)nlmo_p, (size_t)npno);
    }
}
void DLPNOCCSDSolver::phase_t1_fock_()            {}
void DLPNOCCSDSolver::phase_jiang_B_tilde_()      {}
void DLPNOCCSDSolver::phase_jiang_C_tilde_()      {}
void DLPNOCCSDSolver::phase_jiang_D_tilde_()      {}
void DLPNOCCSDSolver::phase_jiang_G_tilde_()      {}
void DLPNOCCSDSolver::phase_pairs_residual_()     {}
void DLPNOCCSDSolver::phase_t1_residual_()        {}
void DLPNOCCSDSolver::phase_update_amps_()        {}
void DLPNOCCSDSolver::apply_diis_()               {}
double DLPNOCCSDSolver::compute_iter_energy_()    { return 0.0; }

}  // namespace pyscf_dlpno_ccsd

// -- C entry point exported to Python (ctypes) -------------------------------
extern "C" {

int DLPNOcompute_lccsd_omp(const pyscf_dlpno_ccsd::SolverInputs *in,
                           double *e_out) {
    if (in == nullptr) return -2;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    int rc = solver.solve();
    if (e_out != nullptr) *e_out = solver.get_energy();
    return rc;
}

// Lightweight smoke-test entry: validates that the library loaded and that
// ctypes can call into it.  Returns the size of SolverInputs in bytes — used
// by the Python smoke test to confirm the struct layouts agree.
int DLPNOcompute_lccsd_solver_inputs_size(void) {
    return (int)sizeof(pyscf_dlpno_ccsd::SolverInputs);
}

}  // extern "C"

// ============================================================================
// Step 2a: dump-and-compare parity test entry.  Reads SolverInputs and writes
// per-field checksums (sum of all values) into checksums_out[0:N].  Python
// computes the same checksums against the same numpy buffers and compares.
// This validates: (a) ctypes Structure layout matches the C++ struct;
// (b) FlatPairStore offsets are interpreted identically; (c) buffer pointer
// arithmetic agrees on both sides.
// ============================================================================

namespace pyscf_dlpno_ccsd {

static double sum_doubles_(const double *p, size_t n) {
    if (p == nullptr) return std::nan("");
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += p[i];
    return s;
}

static double sum_ints_(const int *p, size_t n) {
    if (p == nullptr) return std::nan("");
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += (double)p[i];
    return s;
}

static double sum_int64s_(const int64_t *p, size_t n) {
    if (p == nullptr) return std::nan("");
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += (double)p[i];
    return s;
}

static double sum_pair_store_(const FlatPairStore &fps, int n_pairs) {
    if (fps.data == nullptr || fps.offsets == nullptr) return std::nan("");
    const int64_t total = fps.offsets[n_pairs];
    return sum_doubles_(fps.data, (size_t)total);
}

}  // namespace pyscf_dlpno_ccsd

extern "C" {

// Field index → meaning (Python-side mirrors this exact order).  Adding new
// fields: APPEND only; never reorder.  NaN means "buffer was null" — Python
// must handle it as expected when the caller didn't populate that field.
enum DumpField {
    F_NOCC = 0,             F_NLMO,
    F_N_CANON_PAIRS,        F_N_STRONG_PAIRS,
    F_DIIS_MAX_VECS,        F_MAX_CYCLE,
    F_E_CONV,               F_R_CONV,
    F_F_LMO,                F_EPS_LMO,
    F_FOO,                  F_FOV_FLAT,
    F_E_PNO_FLAT,
    F_T1_FLAT,              F_T2_FLAT,
    F_PAIR_LMO_IDX_FLAT,    F_PAIR_LMO_IDX_OFFSETS_LAST,
    F_N_PNO_PER_PAIR,       F_PNO_OFFSETS_LAST,
    F_QMA,                  F_QAB,
    F_I_QK,                 F_J_QK,
    F_I_QA,                 F_J_QA,
    F_K_IAJB,               F_K_BAR_IJ,        F_K_BAR_CHEM,
    F_J_IJ_KJ,              F_K_IJ_KJ,
    F_L_IAJB,               F_L_BAR,
    F_T1_IN_PAIR,
    F_T2_OFFSETS_LAST,
    F_K_BAR_JI,
    F_T1_IN_PAIR_FULL,
    F_K_TILDE_CHEM_I,
    F_K_TILDE_CHEM_J,
    F_N_ORDERED_PAIRS,
    F_ORDERED_PAIR_I_IDX,
    F_ORDERED_PAIR_K_IDX,
    F_S_PNO_DATA,
    F_S_PNO_OFFSETS_LAST,
    DUMP_N_FIELDS,
};

int DLPNOcompute_lccsd_dump_inputs(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    double *checksums_out,
    int checksums_out_n) {
    using namespace pyscf_dlpno_ccsd;
    if (in == nullptr || checksums_out == nullptr) return -2;
    if (checksums_out_n < (int)DUMP_N_FIELDS) return -3;

    for (int i = 0; i < (int)DUMP_N_FIELDS; ++i) {
        checksums_out[i] = std::nan("");
    }

    checksums_out[F_NOCC]            = (double)in->nocc;
    checksums_out[F_NLMO]            = (double)in->nlmo;
    checksums_out[F_N_CANON_PAIRS]   = (double)in->n_canon_pairs;
    checksums_out[F_N_STRONG_PAIRS]  = (double)in->n_strong_pairs;
    checksums_out[F_DIIS_MAX_VECS]   = (double)in->diis_max_vecs;
    checksums_out[F_MAX_CYCLE]       = (double)in->max_cycle;
    checksums_out[F_E_CONV]          = in->e_conv;
    checksums_out[F_R_CONV]          = in->r_conv;

    const int n_pairs = in->n_canon_pairs;
    const int nocc    = in->nocc;

    checksums_out[F_F_LMO]    = sum_doubles_(in->F_lmo, (size_t)nocc * nocc);
    checksums_out[F_EPS_LMO]  = sum_doubles_(in->eps_lmo, (size_t)nocc);
    checksums_out[F_FOO]      = sum_doubles_(in->foo, (size_t)nocc * nocc);

    if (in->pno_offsets != nullptr) {
        const int64_t fov_total = in->pno_offsets[nocc];
        const int64_t e_pno_total = in->pno_offsets[n_pairs];
        checksums_out[F_FOV_FLAT]   = sum_doubles_(in->fov_flat, (size_t)fov_total);
        checksums_out[F_E_PNO_FLAT] = sum_doubles_(in->e_pno_flat, (size_t)e_pno_total);

        // T1: one (npno_ii) block per occupied i (npno_ii == n_pno_per_pair[ii])
        // T2: per-pair npno×npno; total is sum(n_pno_per_pair[p]^2)
        if (in->n_pno_per_pair != nullptr) {
            int64_t t2_total = 0;
            for (int p = 0; p < n_pairs; ++p) {
                int64_t n = in->n_pno_per_pair[p];
                t2_total += n * n;
            }
            checksums_out[F_T1_FLAT] = sum_doubles_(in->T1_flat, (size_t)fov_total);
            checksums_out[F_T2_FLAT] = sum_doubles_(in->T2_flat, (size_t)t2_total);
        }
        checksums_out[F_PNO_OFFSETS_LAST] = (double)in->pno_offsets[n_pairs];
    }

    if (in->pair_lmo_idx_offsets != nullptr) {
        const int64_t total = in->pair_lmo_idx_offsets[n_pairs];
        checksums_out[F_PAIR_LMO_IDX_FLAT] = sum_ints_(in->pair_lmo_idx_flat, (size_t)total);
        checksums_out[F_PAIR_LMO_IDX_OFFSETS_LAST] = (double)total;
    }
    if (in->n_pno_per_pair != nullptr) {
        checksums_out[F_N_PNO_PER_PAIR] = sum_ints_(in->n_pno_per_pair, (size_t)n_pairs);
    }

    checksums_out[F_QMA]        = sum_pair_store_(in->Qma, n_pairs);
    checksums_out[F_QAB]        = sum_pair_store_(in->Qab, n_pairs);
    checksums_out[F_I_QK]       = sum_pair_store_(in->i_Qk, n_pairs);
    checksums_out[F_J_QK]       = sum_pair_store_(in->j_Qk, n_pairs);
    checksums_out[F_I_QA]       = sum_pair_store_(in->i_Qa, n_pairs);
    checksums_out[F_J_QA]       = sum_pair_store_(in->j_Qa, n_pairs);
    checksums_out[F_K_IAJB]     = sum_pair_store_(in->K_iajb, n_pairs);
    checksums_out[F_K_BAR_IJ]   = sum_pair_store_(in->K_bar_ij, n_pairs);
    checksums_out[F_K_BAR_CHEM] = sum_pair_store_(in->K_bar_chem, n_pairs);
    checksums_out[F_J_IJ_KJ]    = sum_pair_store_(in->J_ij_kj, n_pairs);
    checksums_out[F_K_IJ_KJ]    = sum_pair_store_(in->K_ij_kj, n_pairs);
    checksums_out[F_L_IAJB]     = sum_pair_store_(in->L_iajb, n_pairs);
    checksums_out[F_L_BAR]      = sum_pair_store_(in->L_bar, n_pairs);
    checksums_out[F_T1_IN_PAIR] = sum_pair_store_(in->T1_in_pair, n_pairs);
    if (in->t2_offsets != nullptr) {
        checksums_out[F_T2_OFFSETS_LAST] = (double)in->t2_offsets[n_pairs];
    }
    checksums_out[F_K_BAR_JI]      = sum_pair_store_(in->K_bar_ji, n_pairs);
    checksums_out[F_T1_IN_PAIR_FULL] = sum_pair_store_(in->T1_in_pair_full, n_pairs);
    checksums_out[F_K_TILDE_CHEM_I] = sum_pair_store_(in->K_tilde_chem_i, n_pairs);
    checksums_out[F_K_TILDE_CHEM_J] = sum_pair_store_(in->K_tilde_chem_j, n_pairs);
    checksums_out[F_N_ORDERED_PAIRS] = (double)in->n_ordered_pairs;
    checksums_out[F_ORDERED_PAIR_I_IDX] = sum_ints_(in->ordered_pair_i_idx,
                                                    (size_t)in->n_ordered_pairs);
    checksums_out[F_ORDERED_PAIR_K_IDX] = sum_ints_(in->ordered_pair_k_idx,
                                                    (size_t)in->n_ordered_pairs);

    if (in->S_pno_offsets != nullptr) {
        const int64_t total =
            in->S_pno_offsets[(int64_t)n_pairs * n_pairs];
        checksums_out[F_S_PNO_OFFSETS_LAST] = (double)total;
        checksums_out[F_S_PNO_DATA] =
            sum_doubles_(in->S_pno_data, (size_t)total);
    }

    return (int)DUMP_N_FIELDS;
}

int DLPNOcompute_lccsd_dump_n_fields(void) {
    return (int)DUMP_N_FIELDS;
}

// Step 2b entry: run only the t1_ints phase, writing into caller-provided
// output buffers.  Returns 0 on success, negative on argument errors.
int DLPNOcompute_lccsd_phase_t1_ints(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    pyscf_dlpno_ccsd::T1IntsOutputs *out) {
    if (in == nullptr || out == nullptr) return -2;
    if (out->i_Qa_t1.data == nullptr || out->j_Qa_t1.data == nullptr
        || out->i_Qk_t1.data == nullptr || out->j_Qk_t1.data == nullptr) {
        return -3;
    }
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_t1_ints_into(out);
    return 0;
}

// Step 2c entry: run only the B_tilde phase given previously-computed
// t1-dressed intermediates.
int DLPNOcompute_lccsd_phase_b_tilde(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::BTildeInputs *t1_dressed,
    pyscf_dlpno_ccsd::BTildeOutputs *out) {
    if (in == nullptr || t1_dressed == nullptr || out == nullptr) return -2;
    if (out->B_tilde.data == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_b_tilde_into(t1_dressed, out);
    return 0;
}

// Step 2d entry: run only the t1_fock phase (per-pair Fab + d_ij/d_ji).
int DLPNOcompute_lccsd_phase_t1_fock(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    pyscf_dlpno_ccsd::T1FockOutputs *out) {
    if (in == nullptr || out == nullptr) return -2;
    if (out->Fab.data == nullptr || out->d_flat == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_t1_fock_into(out);
    return 0;
}

// Step 2e entry: run only the D_tilde Phase 1 (Terms 1+2) over ordered pairs.
int DLPNOcompute_lccsd_phase_d_tilde_ph1(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    pyscf_dlpno_ccsd::DTildeOutputs *out) {
    if (in == nullptr || out == nullptr) return -2;
    if (out->D_tilde.data == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_d_tilde_ph1_into(out);
    return 0;
}

// Step 2f entry: run only the C_tilde Phase 1 (Terms 1+2) over ordered pairs.
int DLPNOcompute_lccsd_phase_c_tilde_ph1(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    pyscf_dlpno_ccsd::CTildeOutputs *out) {
    if (in == nullptr || out == nullptr) return -2;
    if (out->C_tilde.data == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_c_tilde_ph1_into(out);
    return 0;
}

// Step 2g entry: run the G_tilde inner kernel given a precomputed plan.
int DLPNOcompute_lccsd_phase_g_tilde_inner(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::GTildeInputs *plan,
    pyscf_dlpno_ccsd::GTildeOutputs *out) {
    if (in == nullptr || plan == nullptr || out == nullptr) return -2;
    if (out->G_tilde == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_g_tilde_inner_into(plan, out);
    return 0;
}

// Step 2h entry: t1_fock finalize (Fkj scatter, Fij_bar snapshot, Eq 94).
int DLPNOcompute_lccsd_phase_t1_fock_finalize(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::T1FockExtraInputs *extra,
    pyscf_dlpno_ccsd::T1FockExtraOutputs *out) {
    if (in == nullptr || extra == nullptr || out == nullptr) return -2;
    if (out->Fkj == nullptr || out->Fij_bar_snapshot == nullptr
        || out->foo_t1 == nullptr) return -3;
    if (extra->d_flat == nullptr) return -4;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_t1_fock_finalize_into(extra, out);
    return 0;
}

// Step 2i entry: T1 residual per-(k, l) batched B + A2 phase.
int DLPNOcompute_lccsd_phase_t1_residual_per_kl(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::PerKlPlanInputs *plan,
    pyscf_dlpno_ccsd::PerKlOutputs *out) {
    if (in == nullptr || plan == nullptr || out == nullptr) return -2;
    if (out->contrib_flat == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_t1_residual_per_kl_into(plan, out);
    return 0;
}

// Step 2j-a entry: R2 B + E term per bucket.
int DLPNOcompute_lccsd_phase_be(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::BEInputs *plan,
    pyscf_dlpno_ccsd::BEOutputs *out) {
    if (in == nullptr || plan == nullptr || out == nullptr) return -2;
    if (out->out_B == nullptr || out->out_E == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_be_into(plan, out);
    return 0;
}

// Step 2j-b1 entry: R2 C-term batched.
int DLPNOcompute_lccsd_phase_c_term(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::CTermInputs *plan,
    pyscf_dlpno_ccsd::CTermOutputs *out) {
    if (in == nullptr || plan == nullptr || out == nullptr) return -2;
    if (out->tiles_flat == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_c_term_into(plan, out);
    return 0;
}

// Step 2j-b2 entry: R2 D-term batched.
int DLPNOcompute_lccsd_phase_d_term(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::DTermInputs *plan,
    pyscf_dlpno_ccsd::DTermOutputs *out) {
    if (in == nullptr || plan == nullptr || out == nullptr) return -2;
    if (out->tiles_flat == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_d_term_into(plan, out);
    return 0;
}

// Step 2j-c entry: R2 G-term batched.
int DLPNOcompute_lccsd_phase_g_term(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::GTermInputs *plan,
    pyscf_dlpno_ccsd::GTermOutputs *out) {
    if (in == nullptr || plan == nullptr || out == nullptr) return -2;
    if (out->tiles_flat == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_g_term_into(plan, out);
    return 0;
}

// Step 2j-d1 entry: C/D Phase 2 t3 batched.
int DLPNOcompute_lccsd_phase_t3(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::T3Inputs *plan,
    pyscf_dlpno_ccsd::T3Outputs *out) {
    if (in == nullptr || plan == nullptr || out == nullptr) return -2;
    if (out->tiles_flat == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_t3_into(plan, out);
    return 0;
}

// Step 2j-d2 entry: C/D Phase 2 t4 batched.
int DLPNOcompute_lccsd_phase_t4(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::T4Inputs *plan,
    pyscf_dlpno_ccsd::T4Outputs *out) {
    if (in == nullptr || plan == nullptr || out == nullptr) return -2;
    if (out->tiles_flat == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_t4_into(plan, out);
    return 0;
}

// Step 2j-e entry: R2 K + ladder.
int DLPNOcompute_lccsd_phase_k_ladder(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::KLadderInputs *t1_dressed,
    pyscf_dlpno_ccsd::KLadderOutputs *out) {
    if (in == nullptr || t1_dressed == nullptr || out == nullptr) return -2;
    if (out->K.data == nullptr || out->A.data == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_k_ladder_into(t1_dressed, out);
    return 0;
}

// Step 2k entry: update T1/T2 in place + compute correlation energy.
int DLPNOcompute_lccsd_phase_update_amps_and_energy(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::UpdateAmpsInputs *resid,
    pyscf_dlpno_ccsd::UpdateAmpsOutputs *out) {
    if (in == nullptr || resid == nullptr || out == nullptr) return -2;
    if (resid->R1_flat == nullptr || resid->R2_flat == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_update_amps_and_energy_into(resid, out);
    return 0;
}

// Step 2l-b entry: compute Fia_bar per canonical pair.
int DLPNOcompute_lccsd_phase_t1_fock_fia_bar(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    pyscf_dlpno_ccsd::FiaBarOutputs *out) {
    if (in == nullptr || out == nullptr) return -2;
    if (out->Fia_bar.data == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_t1_fock_fia_bar_into(out);
    return 0;
}

// Step 2l-c entry: R1 init + A + C contribution per ordered pair.
int DLPNOcompute_lccsd_phase_t1_residual_AC_init(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::R1AcInputs *fia,
    pyscf_dlpno_ccsd::R1AcOutputs *out) {
    if (in == nullptr || fia == nullptr || out == nullptr) return -2;
    if (out->R1_flat == nullptr) return -3;
    if (fia->Fia_bar.data == nullptr) return -4;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_phase_t1_residual_AC_init_into(fia, out);
    return 0;
}

// c-collapse-1: run one full CCSD cycle.
int DLPNOcompute_lccsd_run_one_cycle(
    const pyscf_dlpno_ccsd::SolverInputs *in,
    const pyscf_dlpno_ccsd::RunCycleInputs *plans,
    pyscf_dlpno_ccsd::RunCycleOutputs *out) {
    if (in == nullptr || plans == nullptr || out == nullptr) return -2;
    if (out->R1_flat == nullptr || out->R2_flat == nullptr) return -3;
    pyscf_dlpno_ccsd::DLPNOCCSDSolver solver(*in);
    solver.run_one_cycle(plans, out);
    return 0;
}

}  // extern "C"
