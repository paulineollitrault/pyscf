"""DLPNO-CCSD monolithic solver — Python ctypes bridge.

Step 2a: marshalling layer.

Defines ``PyFlatPairStore`` and ``PySolverInputs`` ctypes Structures whose
byte layout matches ``FlatPairStore`` / ``SolverInputs`` in
``pyscf/lib/cc/dlpno_ccsd_solver.cpp``.  Provides ``build_synthetic_inputs``
which constructs flat numpy buffers + offsets and populates a
``PySolverInputs`` instance, plus ``parity_test_dump_inputs`` which calls
the C++ ``DLPNOcompute_lccsd_dump_inputs`` and verifies the per-field
checksums match Python's numpy-computed checksums on the same buffers.

No CCSD math runs at this layer.  Step 2b adds the first real phase.
"""

import ctypes
import math
import os

import numpy as np

from pyscf import lib as _pyscf_lib


_libcc = _pyscf_lib.load_library('libcc')


# ----------------------------------------------------------------------------
# ctypes structs — must match dlpno_ccsd_solver.cpp byte-for-byte.
# ----------------------------------------------------------------------------

class PyFlatPairStore(ctypes.Structure):
    """Mirrors C++ ``pyscf_dlpno_ccsd::FlatPairStore``."""
    _fields_ = [
        ('data',        ctypes.c_void_p),  # const double *
        ('offsets',     ctypes.c_void_p),  # const int64_t * (sizes)
        ('block_start', ctypes.c_void_p),  # const int64_t * or NULL (aliased pos)
    ]


class PySolverInputs(ctypes.Structure):
    """Mirrors C++ ``pyscf_dlpno_ccsd::SolverInputs``.

    Field order MUST match the C++ definition exactly — natural alignment
    on both sides; ctypes inserts the same 4-byte padding after
    ``n_cas_blocks``.
    """
    _fields_ = [
        # sizes
        ('nocc',                   ctypes.c_int),
        ('nlmo',                   ctypes.c_int),
        ('n_canon_pairs',          ctypes.c_int),
        ('n_strong_pairs',         ctypes.c_int),
        ('diis_max_vecs',          ctypes.c_int),
        ('max_cycle',              ctypes.c_int),
        ('e_conv',                 ctypes.c_double),
        ('r_conv',                 ctypes.c_double),

        # sparsity (read-only views)
        ('i_j_to_ij',              ctypes.c_void_p),
        ('ij_to_i_j',              ctypes.c_void_p),
        ('ij_to_ji',               ctypes.c_void_p),
        ('pair_lmo_idx_flat',      ctypes.c_void_p),
        ('pair_lmo_idx_offsets',   ctypes.c_void_p),
        ('n_pno_per_pair',         ctypes.c_void_p),
        ('pno_offsets',            ctypes.c_void_p),
        ('t2_offsets',             ctypes.c_void_p),

        # orbital data (read-only views)
        ('F_lmo',                  ctypes.c_void_p),
        ('eps_lmo',                ctypes.c_void_p),
        ('foo',                    ctypes.c_void_p),
        ('fov_flat',               ctypes.c_void_p),
        ('e_pno_flat',             ctypes.c_void_p),

        # cc_ints (read-only views, pair_lmo_idx-axis)
        ('Qma',                    PyFlatPairStore),
        ('Qab',                    PyFlatPairStore),
        ('i_Qk',                   PyFlatPairStore),
        ('j_Qk',                   PyFlatPairStore),
        ('i_Qa',                   PyFlatPairStore),
        ('j_Qa',                   PyFlatPairStore),
        ('K_iajb',                 PyFlatPairStore),
        ('K_bar_ij',               PyFlatPairStore),
        ('K_bar_chem',             PyFlatPairStore),
        ('K_bar_ji',               PyFlatPairStore),
        ('J_ij_kj',                PyFlatPairStore),
        ('K_ij_kj',                PyFlatPairStore),
        ('L_iajb',                 PyFlatPairStore),
        ('L_bar',                  PyFlatPairStore),
        ('K_tilde_chem_i',         PyFlatPairStore),
        ('K_tilde_chem_j',         PyFlatPairStore),

        # pno overlap cache
        ('S_pno_data',             ctypes.c_void_p),
        ('S_pno_offsets',          ctypes.c_void_p),
        ('S_pno_index',            ctypes.c_void_p),

        # amplitudes (READ-WRITE)
        ('T1_flat',                ctypes.c_void_p),
        ('T2_flat',                ctypes.c_void_p),

        # T1 projected into each pair's PNO basis (per-pair (nlmo_p, npno_p))
        ('T1_in_pair',             PyFlatPairStore),

        # Full-nocc version (per-pair (nocc, npno_p)) — needed for Stage 4
        # of the T1 residual (R1 -= Fij_bar @ T_n_full).
        ('T1_in_pair_full',        PyFlatPairStore),

        # Ordered-pair sparsity (Psi4 all_pairs)
        ('n_ordered_pairs',        ctypes.c_int),
        ('ordered_pair_i_idx',     ctypes.c_void_p),
        ('ordered_pair_k_idx',     ctypes.c_void_p),

        # CAS injection (optional)
        ('n_cas_blocks',           ctypes.c_int),
        ('cas_block_pair',         ctypes.c_void_p),
        ('cas_block_offsets',      ctypes.c_void_p),
        ('cas_block_data',         ctypes.c_void_p),
        ('cas_block_slice',        ctypes.c_void_p),

        # Optional Psi4-faithful overrides (full strong+weak scope).
        # Set to NULL to disable (legacy strong-only paths apply).
        ('Fij_bar_full',           ctypes.c_void_p),
        ('Fkc_per_ordered',        PyFlatPairStore),
        # External R2 (precomputed by PySCF's full residual machinery,
        # used to validate orchestration + update_amps + energy formula
        # while R2 plan extraction is incremental).  NULL = use K+A.
        ('R2_external',            ctypes.c_void_p),
        # is_strong_pair byte flag per canonical pair.  1 = strong (in
        # correlation energy), 0 = weak (skip in energy formula).  NULL
        # = all pairs counted.
        ('is_strong_pair',         ctypes.c_void_p),
    ]


# ----------------------------------------------------------------------------
# C entry-point declarations.
# ----------------------------------------------------------------------------

_libcc.DLPNOcompute_lccsd_solver_inputs_size.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_solver_inputs_size.argtypes = []

_libcc.DLPNOcompute_lccsd_omp.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_omp.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(ctypes.c_double),
]

_libcc.DLPNOcompute_lccsd_dump_inputs.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_dump_inputs.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
]

_libcc.DLPNOcompute_lccsd_dump_n_fields.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_dump_n_fields.argtypes = []


# Field-index mirror (must match enum DumpField in dlpno_ccsd_solver.cpp).
DUMP_FIELDS = [
    'nocc', 'nlmo', 'n_canon_pairs', 'n_strong_pairs',
    'diis_max_vecs', 'max_cycle', 'e_conv', 'r_conv',
    'F_lmo', 'eps_lmo', 'foo', 'fov_flat', 'e_pno_flat',
    'T1_flat', 'T2_flat',
    'pair_lmo_idx_flat', 'pair_lmo_idx_offsets_last',
    'n_pno_per_pair', 'pno_offsets_last',
    'Qma', 'Qab', 'i_Qk', 'j_Qk', 'i_Qa', 'j_Qa',
    'K_iajb', 'K_bar_ij', 'K_bar_chem', 'J_ij_kj', 'K_ij_kj',
    'L_iajb', 'L_bar',
    'T1_in_pair',
    't2_offsets_last',
    'K_bar_ji',
    'T1_in_pair_full',
    'K_tilde_chem_i', 'K_tilde_chem_j',
    'n_ordered_pairs',
    'ordered_pair_i_idx', 'ordered_pair_k_idx',
    'S_pno_data', 'S_pno_offsets_last',
]


# ----------------------------------------------------------------------------
# Output struct for the t1_ints phase.
# ----------------------------------------------------------------------------

class PyWritablePairStore(ctypes.Structure):
    _fields_ = [
        ('data',    ctypes.c_void_p),
        ('offsets', ctypes.c_void_p),
    ]


class PyT1IntsOutputs(ctypes.Structure):
    _fields_ = [
        ('i_Qa_t1', PyWritablePairStore),
        ('j_Qa_t1', PyWritablePairStore),
        ('i_Qk_t1', PyWritablePairStore),
        ('j_Qk_t1', PyWritablePairStore),
    ]


_libcc.DLPNOcompute_lccsd_phase_t1_ints.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_t1_ints.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyT1IntsOutputs),
]


class PyBTildeInputs(ctypes.Structure):
    _fields_ = [
        ('i_Qk_t1', PyFlatPairStore),
        ('j_Qk_t1', PyFlatPairStore),
    ]


class PyBTildeOutputs(ctypes.Structure):
    _fields_ = [
        ('B_tilde', PyWritablePairStore),
    ]


_libcc.DLPNOcompute_lccsd_phase_b_tilde.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_b_tilde.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyBTildeInputs),
    ctypes.POINTER(PyBTildeOutputs),
]


_libcc.DLPNOcompute_B_tilde_pair.restype = None
_libcc.DLPNOcompute_B_tilde_pair.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
]


class PyT1FockOutputs(ctypes.Structure):
    _fields_ = [
        ('Fab',    PyWritablePairStore),
        ('d_flat', ctypes.c_void_p),
    ]


_libcc.DLPNOcompute_lccsd_phase_t1_fock.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_t1_fock.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyT1FockOutputs),
]


_libcc.DLPNOt1_fock_batched.restype = None
_libcc.DLPNOt1_fock_batched.argtypes = (
    [ctypes.c_void_p] * 18                                # 7 (ptr/off) + 4 shape arrays
    + [ctypes.c_void_p]                                   # is_strong_pair (nullable)
    + [ctypes.c_void_p, ctypes.c_size_t] * 6              # 6 scratch (ptr + stride)
    + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # d_flat, Fab, Fab_off
       ctypes.c_size_t, ctypes.c_int])                    # N, num_threads


class PyDTildeOutputs(ctypes.Structure):
    _fields_ = [
        ('D_tilde', PyWritablePairStore),
    ]


_libcc.DLPNOcompute_lccsd_phase_d_tilde_ph1.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_d_tilde_ph1.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyDTildeOutputs),
]


_libcc.DLPNOcompute_D_tilde_ph1_batched.restype = None
_libcc.DLPNOcompute_D_tilde_ph1_batched.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,    # K_tilde_chem
    ctypes.c_void_p, ctypes.c_void_p,    # M_static
    ctypes.c_void_p, ctypes.c_void_p,    # t1
    ctypes.c_void_p, ctypes.c_void_p,    # T1_rows
    ctypes.c_void_p, ctypes.c_void_p,    # n_pno_arr, n_domain_arr
    ctypes.c_void_p, ctypes.c_void_p,    # D_flat, D_offsets
    ctypes.c_size_t,                      # N
]


class PyCTildeOutputs(ctypes.Structure):
    _fields_ = [
        ('C_tilde', PyWritablePairStore),
    ]


_libcc.DLPNOcompute_lccsd_phase_c_tilde_ph1.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_c_tilde_ph1.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyCTildeOutputs),
]


_libcc.DLPNOcompute_C_tilde_ph1_batched.restype = None
_libcc.DLPNOcompute_C_tilde_ph1_batched.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,    # K_tilde_chem
    ctypes.c_void_p, ctypes.c_void_p,    # K_bar_chem_slice
    ctypes.c_void_p, ctypes.c_void_p,    # t1
    ctypes.c_void_p, ctypes.c_void_p,    # T1_local
    ctypes.c_void_p, ctypes.c_void_p,    # n_pno_arr, n_domain_arr
    ctypes.c_void_p, ctypes.c_void_p,    # C_flat, C_offsets
    ctypes.c_size_t,                      # N
]


class PyGTildeInputs(ctypes.Structure):
    _fields_ = [
        ('n_ij_slots',          ctypes.c_int),
        ('triple_eff_offset',   ctypes.c_void_p),
        ('triple_T2_pair_idx',  ctypes.c_void_p),
        ('triple_n_lj',         ctypes.c_void_p),
        ('ij_triple_starts',    ctypes.c_void_p),
        ('ij_i_arr',            ctypes.c_void_p),
        ('ij_j_arr',            ctypes.c_void_p),
        ('effective_flat',      ctypes.c_void_p),
    ]


class PyGTildeOutputs(ctypes.Structure):
    _fields_ = [
        ('G_tilde', ctypes.c_void_p),     # (nocc, nocc) row-major
    ]


_libcc.DLPNOcompute_lccsd_phase_g_tilde_inner.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_g_tilde_inner.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyGTildeInputs),
    ctypes.POINTER(PyGTildeOutputs),
]


_libcc.DLPNOcompute_G_tilde_inner.restype = None
_libcc.DLPNOcompute_G_tilde_inner.argtypes = [
    ctypes.c_void_p,        # triple_eff_offset
    ctypes.c_void_p,        # triple_T2_pair_idx
    ctypes.c_void_p,        # triple_n_lj
    ctypes.c_void_p,        # ij_triple_starts
    ctypes.c_void_p,        # ij_i_arr
    ctypes.c_void_p,        # ij_j_arr
    ctypes.c_void_p,        # effective_flat
    ctypes.c_void_p,        # T2_flat
    ctypes.c_void_p,        # T2_offsets
    ctypes.c_void_p,        # G_addition
    ctypes.c_size_t,        # n_ij_slots
    ctypes.c_size_t,        # naocc
]


class PyT1FockExtraInputs(ctypes.Structure):
    _fields_ = [
        ('d_flat', ctypes.c_void_p),
    ]


class PyT1FockExtraOutputs(ctypes.Structure):
    _fields_ = [
        ('Fkj',              ctypes.c_void_p),
        ('Fij_bar_snapshot', ctypes.c_void_p),
        ('foo_t1',           ctypes.c_void_p),
    ]


_libcc.DLPNOcompute_lccsd_phase_t1_fock_finalize.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_t1_fock_finalize.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyT1FockExtraInputs),
    ctypes.POINTER(PyT1FockExtraOutputs),
]


class PyPerKlPlanInputs(ctypes.Structure):
    _fields_ = [
        ('n_tasks',           ctypes.c_int),
        ('M',                 ctypes.c_int),
        ('n_kl_arr',          ctypes.c_void_p),
        ('t2_swap_kl',        ctypes.c_void_p),
        ('K_iajb_kl_off',     ctypes.c_void_p),
        ('K_bar_kl_off',      ctypes.c_void_p),
        ('t2_kl_canon_off',   ctypes.c_void_p),
        ('T_n_kl_off',        ctypes.c_void_p),
        ('inner_off',         ctypes.c_void_p),
        ('i_arr',             ctypes.c_void_p),
        ('n_pno_ii_arr',      ctypes.c_void_p),
        ('is_diag_kl_ii',     ctypes.c_void_p),
        ('has_S_ii_kl',       ctypes.c_void_p),
        ('S_ii_kl_off',       ctypes.c_void_p),
        ('has_A2',            ctypes.c_void_p),
        ('is_diag_kl_ki',     ctypes.c_void_p),
        ('n_ki_arr',          ctypes.c_void_p),
        ('t2_swap_ki',        ctypes.c_void_p),
        ('t2_ki_canon_off',   ctypes.c_void_p),
        ('S_kl_ki_off',       ctypes.c_void_p),
        ('S_ki_kl_off',       ctypes.c_void_p),
        ('T_n_l_ii_off',      ctypes.c_void_p),
        ('contrib_off',       ctypes.c_void_p),
        ('K_iajb_buffer',     ctypes.c_void_p),
        ('K_bar_kl_static',   ctypes.c_void_p),
        ('S_pno_buffer',      ctypes.c_void_p),
        ('t2_buffer',         ctypes.c_void_p),
        ('t1_cache_buffer',   ctypes.c_void_p),
        ('max_n_kl',          ctypes.c_int),
        ('max_n_ki',          ctypes.c_int),
    ]


class PyPerKlOutputs(ctypes.Structure):
    _fields_ = [
        ('contrib_flat', ctypes.c_void_p),
    ]


_libcc.DLPNOcompute_lccsd_phase_t1_residual_per_kl.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_t1_residual_per_kl.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyPerKlPlanInputs),
    ctypes.POINTER(PyPerKlOutputs),
]


_libcc.DLPNOper_kl_batched.restype = None
_libcc.DLPNOper_kl_batched.argtypes = (
    [ctypes.c_int, ctypes.c_int]
    + [ctypes.c_void_p] * 21              # 7 per-task + 14 per-(task, inner_i)
    + [ctypes.c_void_p] * 5               # 5 buffers
    + [ctypes.c_void_p, ctypes.c_size_t] * 6   # 6 scratch (ptr, stride)
    + [ctypes.c_void_p, ctypes.c_int])    # contrib_flat, num_threads


class PyBEInputs(ctypes.Structure):
    _fields_ = [
        ('N',          ctypes.c_int),
        ('n_ij',       ctypes.c_int),
        ('n_kl',       ctypes.c_int),
        ('n_slots',    ctypes.c_int),
        ('S',          ctypes.c_void_p),
        ('T',          ctypes.c_void_p),
        ('K',          ctypes.c_void_p),
        ('beta_kl',    ctypes.c_void_p),
        ('beta_lk',    ctypes.c_void_p),
        ('same',       ctypes.c_void_p),
        ('idx',        ctypes.c_void_p),
        # Native B_tilde refresh: per bucket-entry index of source pair
        # in B_tilde_flat plus dense k/l within that pair's PNO basis.
        # When set on every bucket, run_one_cycle refreshes beta_kl/lk
        # from B_tilde_flat after Phase 5, before BE step.
        ('p_ij_arr',    ctypes.c_void_p),
        ('dense_k_arr', ctypes.c_void_p),
        ('dense_l_arr', ctypes.c_void_p),
        # Gathered mode (DLPNO_BE_GATHERED=1): when all three masters are
        # non-null the BE C kernel reads S/T/K from these caller-owned
        # flats via per-item element offsets, eliminating the redundant
        # (N, n_ij, n_kl) + 2 × (N, n_kl, n_kl) per-bucket stack copies.
        ('S_master',    ctypes.c_void_p),
        ('T_master',    ctypes.c_void_p),
        ('K_master',    ctypes.c_void_p),
        ('S_off',       ctypes.c_void_p),
        ('T_off',       ctypes.c_void_p),
        ('K_off',       ctypes.c_void_p),
        # UK hoist: K_master/K_off point at the per-cycle UK master
        # (T2-canonical layout); kernel skips its internal UK build.
        ('uk_hoisted',  ctypes.c_int),
        # BE screening (v3): per-item ||T_kl||/||UK_kl||; in-kernel test
        # (|bkl|+|blk|)*t2n + ukn < tau with the FRESH per-cycle betas.
        ('scr_t2n',     ctypes.c_void_p),
        ('scr_ukn',     ctypes.c_void_p),
        ('scr_tau',     ctypes.c_double),
    ]


class PyBEOutputs(ctypes.Structure):
    _fields_ = [
        ('out_B', ctypes.c_void_p),
        ('out_E', ctypes.c_void_p),
    ]


_libcc.DLPNOcompute_lccsd_phase_be.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_be.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyBEInputs),
    ctypes.POINTER(PyBEOutputs),
]


_libcc.DLPNObe_kernel.restype = None
_libcc.DLPNObe_kernel.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,    # S, T, K
    ctypes.c_void_p, ctypes.c_void_p,                      # beta_kl, beta_lk
    ctypes.c_void_p,                                       # same
    ctypes.c_void_p,                                       # idx
    ctypes.c_void_p, ctypes.c_void_p,                      # out_B, out_E
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,    # N, n_ij, n_kl
    ctypes.c_int,                                          # num_threads
]


class PyCTermInputs(ctypes.Structure):
    _fields_ = [
        ('N',             ctypes.c_int),
        ('n_pno_arr',     ctypes.c_void_p),
        ('n_ct_arr',      ctypes.c_void_p),
        ('n_other_arr',   ctypes.c_void_p),
        ('S_big_off',     ctypes.c_void_p),
        ('ct_off',        ctypes.c_void_p),
        ('S_mid_off',     ctypes.c_void_p),
        ('J_bold_off',    ctypes.c_void_p),
        ('t2_off',        ctypes.c_void_p),
        ('S_outer_off',   ctypes.c_void_p),
        ('tile_off',      ctypes.c_void_p),
        ('S_big_flat',    ctypes.c_void_p),
        ('S_mid_flat',    ctypes.c_void_p),
        ('J_bold_flat',   ctypes.c_void_p),
        ('S_outer_flat',  ctypes.c_void_p),
        ('ct_flat',       ctypes.c_void_p),
        ('t2_flat',       ctypes.c_void_p),
        ('t2_trans',      ctypes.c_void_p),
        ('max_n_pno',     ctypes.c_int),
        ('max_n_ct',      ctypes.c_int),
        ('max_n_other',   ctypes.c_int),
    ]


class PyCTermOutputs(ctypes.Structure):
    _fields_ = [
        ('tiles_flat', ctypes.c_void_p),
    ]


_libcc.DLPNOcompute_lccsd_phase_c_term.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_c_term.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyCTermInputs),
    ctypes.POINTER(PyCTermOutputs),
]


_libcc.DLPNOc_term_batched.restype = None
_libcc.DLPNOc_term_batched.argtypes = (
    [ctypes.c_int]                            # N
    + [ctypes.c_void_p] * 16                  # 3 shape + 7 offsets + 6 buffers
    + [ctypes.c_void_p, ctypes.c_size_t] * 3  # 3 scratch (ptr, stride)
    + [ctypes.c_void_p, ctypes.c_int])        # tiles_flat, num_threads


class PyDTermInputs(ctypes.Structure):
    _fields_ = [
        ('N',           ctypes.c_int),
        ('n_pno_arr',   ctypes.c_void_p),
        ('n_A_arr',     ctypes.c_void_p),
        ('n_B_arr',     ctypes.c_void_p),
        ('S_a_off',     ctypes.c_void_p),
        ('u_off',       ctypes.c_void_p),
        ('S_b_off',     ctypes.c_void_p),
        ('S_c_off',     ctypes.c_void_p),
        ('dt_off',      ctypes.c_void_p),
        ('KJ_off',      ctypes.c_void_p),
        ('tile_off',    ctypes.c_void_p),
        ('S_a_flat',    ctypes.c_void_p),
        ('S_b_flat',    ctypes.c_void_p),
        ('S_c_flat',    ctypes.c_void_p),
        ('KJ_flat',     ctypes.c_void_p),
        ('u_flat',      ctypes.c_void_p),
        ('dt_flat',     ctypes.c_void_p),
        ('u_base',      ctypes.c_void_p),
        ('u_canon_off', ctypes.c_void_p),
        ('u_trans',     ctypes.c_void_p),
        ('max_n_pno',   ctypes.c_int),
        ('max_n_A',     ctypes.c_int),
        ('max_n_B',     ctypes.c_int),
    ]


class PyDTermOutputs(ctypes.Structure):
    _fields_ = [
        ('tiles_flat', ctypes.c_void_p),
    ]


_libcc.DLPNOcompute_lccsd_phase_d_term.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_d_term.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyDTermInputs),
    ctypes.POINTER(PyDTermOutputs),
]


_libcc.DLPNOd_term_batched.restype = None
_libcc.DLPNOd_term_batched.argtypes = (
    [ctypes.c_int]                            # N
    + [ctypes.c_void_p] * 16                  # 3 shape + 7 offsets + 6 buffers
    + [ctypes.c_void_p, ctypes.c_size_t] * 4  # 4 scratch (ptr, stride)
    + [ctypes.c_void_p, ctypes.c_int])        # tiles_flat, num_threads


class PyGTermInputs(ctypes.Structure):
    _fields_ = [
        ('N',           ctypes.c_int),
        ('n_ij_arr',    ctypes.c_void_p),
        ('n_ik_arr',    ctypes.c_void_p),
        ('S_off',       ctypes.c_void_p),
        ('t2_off',      ctypes.c_void_p),
        ('tile_off',    ctypes.c_void_p),
        ('k_idx',       ctypes.c_void_p),
        ('scalar_lmo',  ctypes.c_void_p),
        ('S_flat',      ctypes.c_void_p),
        ('t2_flat',     ctypes.c_void_p),
        ('G_tilde',     ctypes.c_void_p),
        ('G_stride',    ctypes.c_int),
        ('max_n_ij',    ctypes.c_int),
        ('max_n_ik',    ctypes.c_int),
    ]


class PyGTermOutputs(ctypes.Structure):
    _fields_ = [
        ('tiles_flat', ctypes.c_void_p),
    ]


_libcc.DLPNOcompute_lccsd_phase_g_term.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_g_term.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyGTermInputs),
    ctypes.POINTER(PyGTermOutputs),
]


_libcc.DLPNOg_term_batched.restype = None
_libcc.DLPNOg_term_batched.argtypes = (
    [ctypes.c_int]                            # N
    + [ctypes.c_void_p] * 9                   # 2 shape + 3 offsets + 2 idx + S_flat + t2_flat
    + [ctypes.c_void_p, ctypes.c_size_t]      # G_tilde, G_stride
    + [ctypes.c_void_p, ctypes.c_size_t]      # tmp scratch (ptr, stride)
    + [ctypes.c_void_p, ctypes.c_int])        # tiles_flat, num_threads


class PyT3Inputs(ctypes.Structure):
    _fields_ = [
        ('N',          ctypes.c_int),
        ('n_kl_arr',   ctypes.c_void_p),
        ('n_ki_arr',   ctypes.c_void_p),
        ('K_off',      ctypes.c_void_p),
        ('S_off',      ctypes.c_void_p),
        ('t1i_off',    ctypes.c_void_p),
        ('T1l_off',    ctypes.c_void_p),
        ('tile_off',   ctypes.c_void_p),
        ('K_flat',     ctypes.c_void_p),
        ('S_flat',     ctypes.c_void_p),
        ('t1_flat',    ctypes.c_void_p),
        ('max_n_kl',   ctypes.c_int),
        ('max_n_ki',   ctypes.c_int),
    ]


class PyT3Outputs(ctypes.Structure):
    _fields_ = [
        ('tiles_flat', ctypes.c_void_p),
    ]


_libcc.DLPNOcompute_lccsd_phase_t3.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_t3.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyT3Inputs),
    ctypes.POINTER(PyT3Outputs),
]


_libcc.DLPNOt3_kernel_batched.restype = None
_libcc.DLPNOt3_kernel_batched.argtypes = (
    [ctypes.c_int]                            # N
    + [ctypes.c_void_p] * 10                  # 2 shape + 5 offsets + K_flat + S_flat + t1_flat
    + [ctypes.c_void_p, ctypes.c_size_t] * 2  # 2 scratch
    + [ctypes.c_void_p, ctypes.c_int])        # tiles_flat, num_threads


class PyT4Inputs(ctypes.Structure):
    _fields_ = [
        ('N',              ctypes.c_int),
        ('n_ki_arr',       ctypes.c_void_p),
        ('n_li_arr',       ctypes.c_void_p),
        ('n_kl_arr',       ctypes.c_void_p),
        ('S_ki_li_off',    ctypes.c_void_p),
        ('t2_off',         ctypes.c_void_p),
        ('S_li_kl_off',    ctypes.c_void_p),
        ('K_off',          ctypes.c_void_p),
        ('S_kl_ki_off',    ctypes.c_void_p),
        ('tile_off',       ctypes.c_void_p),
        ('S_ki_li_flat',   ctypes.c_void_p),
        ('S_li_kl_flat',   ctypes.c_void_p),
        ('K_flat',         ctypes.c_void_p),
        ('S_kl_ki_flat',   ctypes.c_void_p),
        ('t2_flat',        ctypes.c_void_p),
        ('scale',          ctypes.c_double),
        ('max_n_ki',       ctypes.c_int),
        ('max_n_li',       ctypes.c_int),
        ('max_n_kl',       ctypes.c_int),
        # magnitude screening: per-item bound ptr (NULL = off) + threshold
        ('bound',          ctypes.c_void_p),
        ('tau',            ctypes.c_double),
    ]


class PyT4Outputs(ctypes.Structure):
    _fields_ = [
        ('tiles_flat', ctypes.c_void_p),
    ]


_libcc.DLPNOcompute_lccsd_phase_t4.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_t4.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyT4Inputs),
    ctypes.POINTER(PyT4Outputs),
]


_libcc.DLPNOt4_kernel_batched.restype = None
# NOTE: signature includes trailing (bound_ptr, tau) screening args.
_libcc.DLPNOt4_kernel_batched.argtypes = (
    [ctypes.c_int]                            # N
    + [ctypes.c_void_p] * 14                  # 3 shape + 6 offsets + 5 buffers
    + [ctypes.c_void_p, ctypes.c_size_t] * 3  # 3 scratch
    + [ctypes.c_void_p, ctypes.c_double,       # tiles_flat, scale
       ctypes.c_void_p, ctypes.c_double,       # bound (NULL=off), tau
       ctypes.c_int])                          # num_threads


class PyKLadderInputs(ctypes.Structure):
    _fields_ = [
        ('i_Qa_t1', PyFlatPairStore),
        ('j_Qa_t1', PyFlatPairStore),
    ]


class PyKLadderOutputs(ctypes.Structure):
    _fields_ = [
        ('K', PyWritablePairStore),
        ('A', PyWritablePairStore),
    ]


_libcc.DLPNOcompute_lccsd_phase_k_ladder.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_k_ladder.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyKLadderInputs),
    ctypes.POINTER(PyKLadderOutputs),
]


class PyUpdateAmpsInputs(ctypes.Structure):
    _fields_ = [
        ('R1_flat', ctypes.c_void_p),
        ('R2_flat', ctypes.c_void_p),
    ]


class PyUpdateAmpsOutputs(ctypes.Structure):
    _fields_ = [
        ('energy', ctypes.c_double),
    ]


_libcc.DLPNOcompute_lccsd_phase_update_amps_and_energy.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_update_amps_and_energy.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyUpdateAmpsInputs),
    ctypes.POINTER(PyUpdateAmpsOutputs),
]


class PyFiaBarOutputs(ctypes.Structure):
    _fields_ = [
        ('Fia_bar', PyWritablePairStore),
    ]


_libcc.DLPNOcompute_lccsd_phase_t1_fock_fia_bar.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_t1_fock_fia_bar.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyFiaBarOutputs),
]


class PyR1AcInputs(ctypes.Structure):
    _fields_ = [
        ('Fia_bar', PyFlatPairStore),
        ('do_init', ctypes.c_int),
    ]


class PyR1AcOutputs(ctypes.Structure):
    _fields_ = [
        ('R1_flat', ctypes.c_void_p),
    ]


_libcc.DLPNOcompute_lccsd_phase_t1_residual_AC_init.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_phase_t1_residual_AC_init.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyR1AcInputs),
    ctypes.POINTER(PyR1AcOutputs),
]


# c-collapse-1: one-cycle entry.  Plan-cached phases are passed as nullable
# struct pointers; null = skip phase (zero contribution).
class PyRunCycleInputs(ctypes.Structure):
    _fields_ = [
        ('g_tilde_plan',  ctypes.POINTER(PyGTildeInputs)),
        ('per_kl_plan',   ctypes.POINTER(PyPerKlPlanInputs)),
        ('be_plan',       ctypes.POINTER(PyBEInputs)),
        ('c_term_plan',   ctypes.POINTER(PyCTermInputs)),
        ('d_term_plan',   ctypes.POINTER(PyDTermInputs)),
        ('g_term_plan',   ctypes.POINTER(PyGTermInputs)),
        ('t3_plan',       ctypes.POINTER(PyT3Inputs)),
        ('t4_plan',       ctypes.POINTER(PyT4Inputs)),
        # Native R2 assembly: G_term jk side + per-item canonical pair
        # scatter tables (length g_term_plan->N / g_term_plan_jk->N).
        ('g_term_plan_jk',           ctypes.POINTER(PyGTermInputs)),
        ('g_term_target_pair_idx_ik', ctypes.c_void_p),
        ('g_term_target_pair_idx_jk', ctypes.c_void_p),
        # Native R2 assembly: BE multi-bucket dispatch + per-pair scatter.
        ('be_n_buckets',              ctypes.c_int),
        ('be_plan_buckets',           ctypes.c_void_p),  # array of BEInputs
        ('be_n_unique_n_ij',          ctypes.c_int),
        ('be_unique_n_ij',            ctypes.c_void_p),
        ('be_flat_off_per_n_ij',      ctypes.c_void_p),
        ('be_pair_n_ij_idx',          ctypes.c_void_p),
        ('be_pair_slot',              ctypes.c_void_p),
        # Native R2 assembly: CD per-item canonical-pair scatter indices.
        ('c_term_target_pair_idx_ij', ctypes.c_void_p),
        ('c_term_target_pair_idx_ji', ctypes.c_void_p),
        ('d_term_target_pair_idx_ij', ctypes.c_void_p),
        ('d_term_target_pair_idx_ji', ctypes.c_void_p),
        # Native C_tilde / D_tilde Phase 2 build (t3 + t4 plans).
        ('c_t3_plan',                 ctypes.POINTER(PyT3Inputs)),
        ('c_t4_plan',                 ctypes.POINTER(PyT4Inputs)),
        ('d_t3_plan',                 ctypes.POINTER(PyT3Inputs)),
        ('d_t4_plan',                 ctypes.POINTER(PyT4Inputs)),
        ('c_t3_target_ord_idx',       ctypes.c_void_p),
        ('c_t4_target_ord_idx',       ctypes.c_void_p),
        ('d_t3_target_ord_idx',       ctypes.c_void_p),
        ('d_t4_target_ord_idx',       ctypes.c_void_p),
        # Per-CD-item ordered-pair index for native ct_flat/dt_flat gather.
        ('c_term_ct_ord_pair_idx',    ctypes.c_void_p),
        ('d_term_dt_ord_pair_idx',    ctypes.c_void_p),
        # BE slot-cat (cross-bucket slot-sorted kernel; DLPNO_BE_SLOTCAT).
        ('bes_S_cat',         ctypes.c_void_p),
        ('bes_T_master',      ctypes.c_void_p),
        ('bes_UK_master',     ctypes.c_void_p),
        ('bes_T_off',         ctypes.c_void_p),
        ('bes_item_nkl',      ctypes.c_void_p),
        ('bes_item_k0',       ctypes.c_void_p),
        ('bes_same',          ctypes.c_void_p),
        ('bes_beta0_kl',      ctypes.c_void_p),
        ('bes_beta0_lk',      ctypes.c_void_p),
        ('bes_p_ij',          ctypes.c_void_p),
        ('bes_dense_k',       ctypes.c_void_p),
        ('bes_dense_l',       ctypes.c_void_p),
        ('bes_slot_ptr',      ctypes.c_void_p),
        ('bes_slot_scat_off', ctypes.c_void_p),
        ('bes_slot_KT',       ctypes.c_void_p),
        ('bes_slot_nij',      ctypes.c_void_p),
        ('bes_slot_out_off',  ctypes.c_void_p),
        ('bes_n_slots',       ctypes.c_int64),
        ('bes_max_nkl',       ctypes.c_int),
        ('bes_max_nij',       ctypes.c_int),
        ('bes_hcap',          ctypes.c_int64),
    ]


class PyRunCycleOutputs(ctypes.Structure):
    _fields_ = [
        ('R1_flat', ctypes.c_void_p),
        ('R2_flat', ctypes.c_void_p),
        ('energy',  ctypes.c_double),
        ('G_tilde_out', ctypes.c_void_p),
    ]


_libcc.DLPNOcompute_lccsd_run_one_cycle.restype = ctypes.c_int
_libcc.DLPNOcompute_lccsd_run_one_cycle.argtypes = [
    ctypes.POINTER(PySolverInputs),
    ctypes.POINTER(PyRunCycleInputs),
    ctypes.POINTER(PyRunCycleOutputs),
]


# Forward decl of the per-pair kernel; we call it directly from Python in
# the parity reference path.  Existing wiring exists in local_df.py; we
# repeat it here to keep this module self-contained.
_libcc.DLPNOt1_ints_pair_side.restype = None
_libcc.DLPNOt1_ints_pair_side.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
]


# ----------------------------------------------------------------------------
# Synthetic input builder.
# ----------------------------------------------------------------------------

def _ptr(arr):
    """Pointer-as-int into a numpy array for ctypes c_void_p."""
    if arr is None:
        return None
    return arr.ctypes.data


def _build_flat_pair_store(per_pair_arrays, ownership):
    """Coalesce a list of per-pair arrays into one flat buffer + offsets.

    Returns (PyFlatPairStore, total_len). ``ownership`` is a list the caller
    appends the flat numpy arrays to — the C side reads through their
    pointers, so they must outlive the call.
    """
    n_pairs = len(per_pair_arrays)
    offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    for p in range(n_pairs):
        offsets[p + 1] = offsets[p] + per_pair_arrays[p].size
    total = int(offsets[-1])
    flat = np.empty(total, dtype=np.float64)
    for p in range(n_pairs):
        a = per_pair_arrays[p]
        flat[offsets[p]:offsets[p + 1]] = a.ravel()
    ownership.append(flat)
    ownership.append(offsets)
    fps = PyFlatPairStore()
    fps.data = _ptr(flat)
    fps.offsets = _ptr(offsets)
    return fps, total, flat, offsets


def _build_Fij_bar_full(F_lmo, t2_pno_all, cc_ints, pno_spaces,
                         t1_cache, pair_lmo_idx, nocc):
    """Mirror PySCF _compute_t1_residual's Fij_bar dressing (lines
    681-709).  Builds the FULL T1-dressed F_oo (strong + weak pair
    contributions).  Returns a (nocc, nocc) numpy array.
    """
    Fij_bar = np.ascontiguousarray(F_lmo, dtype=np.float64).copy()
    for key_ij, _T2 in t2_pno_all.items():
        ci_ij = cc_ints.get(key_ij)
        if ci_ij is None:
            continue
        i0, j0 = key_ij
        if pno_spaces[key_ij]['n_pno'] == 0:
            continue
        T_n_ij_mat = t1_cache[key_ij]
        T_n_red = T_n_ij_mat[ci_ij['p_lmos']]
        Fij_bar[i0, j0] += (
            2.0 * np.sum(T_n_red * ci_ij['K_bar_chem'])
            - np.sum(T_n_red * ci_ij['K_bar_ji']))
        if i0 != j0:
            Fij_bar[j0, i0] += (
                2.0 * np.sum(T_n_red * ci_ij['K_bar_chem'])
                - np.sum(T_n_red * ci_ij['K_bar_ij']))
    return np.ascontiguousarray(Fij_bar)


def _build_Fkc_per_ordered(cc_ints, t1_pno, t1_cache, S_pno_cache,
                            pno_spaces, keys_sorted,
                            ordered_pair_i_idx, ordered_pair_k_idx,
                            n_pno_per_pair, i_j_to_ij_2d, nocc):
    """Build Fkc per ORDERED pair (a_ord=i, b_ord=k), mirroring PySCF's
    fkc_dress inner sum in `_compute_t1_residual` C term:
        Fkc[(i, k)] = sum_m S(canon(k,i), canon(i,m)) @ L_iajb[canon(i,m)]
                              @ t1[m]_in_(i,m)
    where i is the R1 owner, k is the partner.  Length npno[canon(k,i)].
    Returns (Fkc_flat, Fkc_offsets) with offsets[o+1] - offsets[o] =
    npno[canon(k,i)] of ordered pair o.
    """
    n_ord = ordered_pair_i_idx.size
    # Precompute LT1[(i, m)] = (key_im, L_im @ t1_m_in_im) for all valid (i, m).
    _LT1_cache = {}
    for m in range(nocc):
        t1_m = t1_pno.get(m)
        if t1_m is None or t1_m.size == 0:
            continue
        if np.max(np.abs(t1_m)) < 1e-15:
            continue
        for i_out in range(nocc):
            key_im = (min(i_out, m), max(i_out, m))
            if key_im not in pno_spaces:
                continue
            if pno_spaces[key_im]['n_pno'] == 0:
                continue
            ci_im = cc_ints.get(key_im)
            if ci_im is None:
                continue
            K_im = ci_im['K_iajb']
            L_im = 2.0 * K_im - K_im.T
            t1_m_in_im = t1_cache[key_im][m]
            _LT1_cache[(i_out, m)] = (key_im, L_im @ t1_m_in_im)

    # Build per-ordered-pair Fkc.
    Fkc_offsets = np.zeros(n_ord + 1, dtype=np.int64)
    Fkc_blocks = []
    for o in range(n_ord):
        a_ord = int(ordered_pair_i_idx[o])  # i (R1 owner)
        b_ord = int(ordered_pair_k_idx[o])  # k (partner)
        p_canon = int(i_j_to_ij_2d[b_ord, a_ord])  # canon(k, i) == canon(i, k)
        if p_canon < 0:
            Fkc_offsets[o + 1] = Fkc_offsets[o]
            continue
        npno_p = int(n_pno_per_pair[p_canon])
        if npno_p == 0:
            Fkc_offsets[o + 1] = Fkc_offsets[o]
            continue
        key_ki = keys_sorted[p_canon]
        Fkc = np.zeros(npno_p, dtype=np.float64)
        for m in range(nocc):
            entry = _LT1_cache.get((a_ord, m))
            if entry is None:
                continue
            key_im, LT1 = entry
            if key_ki == key_im:
                Fkc += LT1
            else:
                S_ki_im = S_pno_cache.get((key_ki, key_im))
                if S_ki_im is not None:
                    Fkc += S_ki_im @ LT1
        Fkc_blocks.append(Fkc)
        Fkc_offsets[o + 1] = Fkc_offsets[o] + npno_p
    Fkc_flat = (np.concatenate(Fkc_blocks)
                if Fkc_blocks else np.zeros(0, dtype=np.float64))
    return np.ascontiguousarray(Fkc_flat), Fkc_offsets


def _build_t34_plan(plan_obj, side, t1_cache, t2_pno_all, ord_idx_lookup,
                      nocc):
    """Extract t3+t4 plans for a given Phase 2 plan object (either
    compute_C_tilde_batched or build_D_tilde_batched cache value).

    Returns dict with t3_struct, t4_struct, t3_target_ord, t4_target_ord
    + ownership list.  side: 'c' or 'd' (purely for naming).
    """
    from pyscf.cc.dlpno_tccsd.residual import (
        _get_or_build_t34_batched_view)
    from pyscf.cc.dlpno_tccsd._cd_gather_cy import (
        gather_t2_with_transpose, gather_u_from_t2)

    bv = _get_or_build_t34_batched_view(plan_obj, t1_cache, t2_pno_all)
    own = []
    pairs_by_n_ki = plan_obj['pairs_by_n_ki']

    def _slot_to_ord(n_ki_n, slot_n):
        if n_ki_n not in pairs_by_n_ki:
            return -1
        if slot_n >= len(pairs_by_n_ki[n_ki_n]):
            return -1
        k, i = pairs_by_n_ki[n_ki_n][slot_n]
        # Class's ord index = ord_idx_lookup[a_ord * nocc + b_ord] where
        # the ordered pair convention is (a, b) = i, k (R1 owner first).
        # But t34 plans iterate over Psi4 'all_pairs' as ORDERED (k, i)
        # with k being the FIRST in the tuple from all_pairs.  Our class's
        # ordered_pair_i_idx is the FIRST index per ordered pair.
        return ord_idx_lookup.get((k, i), -1)

    # ----- t3 plan struct -----
    t3_N = bv['t3_N']
    t3_struct = None
    t3_target_ord = None
    if t3_N > 0:
        t3_struct = PyT3Inputs()
        t3_struct.N         = int(t3_N)
        t3_struct.n_kl_arr  = bv['t3_n_kl'].ctypes.data
        t3_struct.n_ki_arr  = bv['t3_n_ki'].ctypes.data
        t3_struct.K_off     = bv['t3_K_off'].ctypes.data
        t3_struct.S_off     = bv['t3_S_off'].ctypes.data
        t3_struct.t1i_off   = bv['t3_t1i_off'].ctypes.data
        t3_struct.T1l_off   = bv['t3_T1l_off'].ctypes.data
        t3_struct.tile_off  = bv['t3_tile_off'].ctypes.data
        t3_struct.K_flat    = bv['t3_K_flat'].ctypes.data
        t3_struct.S_flat    = bv['t3_S_flat'].ctypes.data
        t3_struct.t1_flat   = t1_cache._buffer.ctypes.data
        t3_struct.max_n_kl  = int(bv['t3_n_kl'].max(initial=1))
        t3_struct.max_n_ki  = int(bv['t3_n_ki'].max(initial=1))
        # t3_target_ord is cycle-invariant (depends only on plan structure
        # + ord_idx_lookup). Cache on bv.
        if 't3_target_ord_cached' in bv:
            t3_target_ord = bv['t3_target_ord_cached']
        else:
            t3_target_ord = np.empty(t3_N, dtype=np.int32)
            for n in range(t3_N):
                n_ki_n, slot_n = bv['t3_target_slot'][n]
                t3_target_ord[n] = _slot_to_ord(n_ki_n, slot_n)
            bv['t3_target_ord_cached'] = t3_target_ord
        own.append(t3_target_ord)
        own.extend([bv['t3_n_kl'], bv['t3_n_ki'], bv['t3_K_off'],
                    bv['t3_S_off'], bv['t3_t1i_off'], bv['t3_T1l_off'],
                    bv['t3_tile_off'], bv['t3_K_flat'], bv['t3_S_flat']])

    # ----- t4 plan struct -----
    t4_N = bv['t4_N']
    t4_struct = None
    t4_target_ord = None
    if t4_N > 0:
        # Per-iter t2_flat / u_flat gather.  side='d' uses u (anti-sym),
        # side='c' uses raw t2.  The plan_obj has 't4_use_u' flag.
        t4_use_u = (side == 'd')
        # Cache t2_flat buffer on bv (size cycle-invariant).
        if '_t4_t2_flat_buf' in bv:
            t2_flat = bv['_t4_t2_flat_buf']
        else:
            t2_flat = np.empty(int(bv['t4_t2_off'][-1]))
            bv['_t4_t2_flat_buf'] = t2_flat
        if t4_use_u:
            gather_u_from_t2(
                t4_N, bv['t4_n_li'],
                bv['t4_t2_canon_off'], bv['t4_t2_trans_arr'],
                bv['t4_t2_off'], t2_pno_all._buffer, t2_flat,
                min(64, t4_N))
        else:
            gather_t2_with_transpose(
                t4_N, bv['t4_n_li'],
                bv['t4_t2_canon_off'], bv['t4_t2_trans_arr'],
                bv['t4_t2_off'], t2_pno_all._buffer, t2_flat,
                min(64, t4_N))
        own.append(t2_flat)

        # t4_scale: -0.5 for C_tilde, +0.5 for D_tilde.
        t4_scale = +0.5 if side == 'd' else -0.5

        t4_struct = PyT4Inputs()
        t4_struct.N             = int(t4_N)
        t4_struct.n_ki_arr      = bv['t4_n_ki'].ctypes.data
        t4_struct.n_li_arr      = bv['t4_n_li'].ctypes.data
        t4_struct.n_kl_arr      = bv['t4_n_kl'].ctypes.data
        t4_struct.S_ki_li_off   = bv['t4_S_ki_li_off'].ctypes.data
        t4_struct.t2_off        = bv['t4_t2_off'].ctypes.data
        t4_struct.S_li_kl_off   = bv['t4_S_li_kl_off'].ctypes.data
        t4_struct.K_off         = bv['t4_K_off'].ctypes.data
        t4_struct.S_kl_ki_off   = bv['t4_S_kl_ki_off'].ctypes.data
        t4_struct.tile_off      = bv['t4_tile_off'].ctypes.data
        t4_struct.S_ki_li_flat  = bv['t4_S_ki_li_flat'].ctypes.data
        t4_struct.S_li_kl_flat  = bv['t4_S_li_kl_flat'].ctypes.data
        t4_struct.K_flat        = bv['t4_K_flat'].ctypes.data
        t4_struct.S_kl_ki_flat  = bv['t4_S_kl_ki_flat'].ctypes.data
        t4_struct.t2_flat       = t2_flat.ctypes.data
        t4_struct.scale         = float(t4_scale)
        # Magnitude screening (DLPNO_T4_SCREEN_TAU > 0): rigorous per-item
        # bound  ||contrib||_F <= |scale| * ||t2_li||_F * ||K_kl||_F
        # (PNO-overlap 2-norms <= 1).  K norms are cycle-invariant (cached
        # on bv); t2 norms recomputed each cycle from the freshly gathered
        # t2_flat.  tau=0 (default) -> bound=NULL -> bit-exact legacy path.
        _tau_t4 = float(os.environ.get('DLPNO_T4_SCREEN_TAU', '0') or 0)
        t4_struct.tau = _tau_t4
        t4_struct.bound = None
        if _tau_t4 > 0.0:
            from pyscf.cc.dlpno_tccsd._wm_scatter_cy import norms_by_offsets
            _Kn = bv.get('_t4_K_norms')
            if _Kn is None:
                _Kn = np.empty(t4_N)
                norms_by_offsets(bv['t4_K_flat'],
                                 np.ascontiguousarray(bv['t4_K_off']),
                                 bv['t4_n_kl'], _Kn)
                bv['_t4_K_norms'] = _Kn
            _t2n = bv.get('_t4_t2_norms_buf')
            if _t2n is None:
                _t2n = np.empty(t4_N)
                bv['_t4_t2_norms_buf'] = _t2n
            norms_by_offsets(t2_flat,
                             np.ascontiguousarray(bv['t4_t2_off'][:t4_N]),
                             bv['t4_n_li'], _t2n)
            _t4_bound = bv.get('_t4_bound_buf')
            if _t4_bound is None:
                _t4_bound = np.empty(t4_N)
                bv['_t4_bound_buf'] = _t4_bound
            np.multiply(_Kn, _t2n, out=_t4_bound)
            _t4_bound *= abs(float(t4_scale))
            t4_struct.bound = _t4_bound.ctypes.data
            own.extend([_Kn, _t2n, _t4_bound])
        t4_struct.max_n_ki      = int(bv['t4_n_ki'].max(initial=1))
        t4_struct.max_n_li      = int(bv['t4_n_li'].max(initial=1))
        t4_struct.max_n_kl      = int(bv['t4_n_kl'].max(initial=1))
        # t4_target_ord cycle-invariant — cache on bv.
        if 't4_target_ord_cached' in bv:
            t4_target_ord = bv['t4_target_ord_cached']
        else:
            t4_target_ord = np.empty(t4_N, dtype=np.int32)
            for n in range(t4_N):
                n_ki_n, slot_n = bv['t4_target_slot'][n]
                t4_target_ord[n] = _slot_to_ord(n_ki_n, slot_n)
            bv['t4_target_ord_cached'] = t4_target_ord
        own.append(t4_target_ord)
        own.extend([bv['t4_n_ki'], bv['t4_n_li'], bv['t4_n_kl'],
                    bv['t4_S_ki_li_off'], bv['t4_S_li_kl_off'],
                    bv['t4_K_off'], bv['t4_S_kl_ki_off'],
                    bv['t4_tile_off'], bv['t4_S_ki_li_flat'],
                    bv['t4_S_li_kl_flat'], bv['t4_K_flat'],
                    bv['t4_S_kl_ki_flat']])

    return {
        't3_struct': t3_struct, 't4_struct': t4_struct,
        't3_target_ord': t3_target_ord, 't4_target_ord': t4_target_ord,
    }, own


def _build_native_r2_plans(t2_pno_all, key_to_p, keys_reorder,
                              pno_spaces, b_tilde_per_ij, C_tilde_cache,
                              D_tilde_cache, n_canon_pairs):
    """Build all scatter tables + plan structs needed to drive
    run_one_cycle's native R2 path (R2_external=None).  Returns a dict
    of arrays + the ownership list to keep them alive.

    Wires:
      G_term: g_term_plan, g_term_plan_jk, target_pair_idx_ik/jk
      BE:     be_plan_buckets array, be_unique_n_ij, be_flat_off_per_n_ij,
              be_pair_n_ij_idx, be_pair_slot
      CD:     c_term_plan, d_term_plan, target_pair_idx_ij/ji per side
    """
    from pyscf.cc.dlpno_tccsd.residual import (
        compute_G_term_batched, compute_B_E_batched,
        compute_CD_terms_batched, _get_or_build_g_term_batched_view,
        _get_or_build_cd_batched_view)
    from pyscf.cc.dlpno_tccsd._cd_gather_cy import (
        gather_t2_with_transpose, gather_u_from_t2)
    own = []

    # ----------------------- G_term -----------------------
    g_term_cache = getattr(compute_G_term_batched, '_plan_cache', None)
    g_plan = next(iter(g_term_cache.values())) if g_term_cache else None
    g_plan_struct_ik = g_plan_struct_jk = None
    g_target_ik_arr = g_target_jk_arr = None
    if g_plan is not None:
        for side, suffix in [('ik', 'ik'), ('jk', 'jk')]:
            ps, po, target_slots, _ = _extract_g_term_plan(
                t2_pno_all, key_to_p, side=side)
            if ps is None:
                continue
            # Build per-item target canonical pair index.
            # Target array is cycle-invariant (depends only on plan
            # structure + key_to_p). Cache on g_plan dict.
            cache_k = f'_g_target_{suffix}_cached'
            if cache_k in g_plan:
                tgt = g_plan[cache_k]
            else:
                tgt = np.full(ps.N, -1, dtype=np.int32)
                pairs_by_n_ij = g_plan['pairs_by_n_ij']
                for n in range(ps.N):
                    n_ij_n, slot_n = target_slots[n]
                    if n_ij_n in pairs_by_n_ij:
                        pyscf_pair = pairs_by_n_ij[n_ij_n][slot_n]
                        tgt[n] = key_to_p.get(pyscf_pair, -1)
                g_plan[cache_k] = tgt
            own.append(tgt)
            own.extend(po)
            if side == 'ik':
                g_plan_struct_ik = ps
                g_target_ik_arr = tgt
            else:
                g_plan_struct_jk = ps
                g_target_jk_arr = tgt

    # ----------------------- BE -----------------------
    be_cache = getattr(compute_B_E_batched, '_plan_cache', None)
    be_plan = next(iter(be_cache.values())) if be_cache else None
    be_plan_buckets_arr = be_unique_n_ij = be_flat_off = None
    be_pair_n_ij_idx_arr = be_pair_slot_arr = None
    be_n_buckets = 0
    be_n_unique = 0
    be_owned_buckets = []
    bes = None
    # UK hoist (DLPNO_BE_UK_HOIST=1): build UK = K^T(2T-T^T)[+K(2T^T-T)]
    # once per canonical pair per cycle instead of per (ij,kl) item —
    # exact reordering (neither we nor Psi4 exploited this); drops the BE
    # item from 6 to 4 dgemms and its TT/UK scratch builds.
    _uk_master_buf = None
    if (be_plan is not None
            and os.environ.get('DLPNO_BE_UK_HOIST', '1') == '1'
            and be_plan.get('uk_k_off') is not None
            and be_plan.get('K_iajb_master') is not None
            and hasattr(t2_pno_all, '_buffer')):
        from pyscf.cc.dlpno_tccsd._wm_scatter_cy import build_uk_master
        _uk_n_arr = be_plan['uk_n']
        _uk_t2off = be_plan.get('_uk_t2_off_c')
        if _uk_t2off is None:
            _uk_t2off = np.ascontiguousarray(
                np.asarray(t2_pno_all._offsets, dtype=np.int64)[
                    :_uk_n_arr.shape[0]])
            be_plan['_uk_t2_off_c'] = _uk_t2off
        _uk_master_buf = be_plan.get('_uk_master_buf')
        if _uk_master_buf is None:
            _uk_master_buf = np.empty(int(t2_pno_all._buffer.shape[0]))
            be_plan['_uk_master_buf'] = _uk_master_buf
        _uk_maxn = int(_uk_n_arr.max(initial=1))
        _uk_scr = be_plan.get('_uk_scratch')
        if _uk_scr is None or _uk_scr.shape[1] < 2 * _uk_maxn * _uk_maxn:
            _uk_scr = np.empty((32, 2 * _uk_maxn * _uk_maxn))
            be_plan['_uk_scratch'] = _uk_scr
        build_uk_master(
            np.asarray(t2_pno_all._buffer), _uk_t2off,
            np.asarray(be_plan['K_iajb_master']), be_plan['uk_k_off'],
            _uk_n_arr, be_plan['uk_same'], _uk_scr, _uk_master_buf, 32)
        be_owned_buckets.append(_uk_master_buf)
    # BE magnitude screening (DLPNO_BE_SCREEN_TAU > 0; needs the UK hoist
    # for ||UK|| norms): per-canonical-pair Frobenius norms of T2 and UK,
    # gathered per item at bucket-wiring time below.
    _be_scr_tau = 0.0
    _be_t2norm_pair = None
    _be_uknorm_pair = None
    if _uk_master_buf is not None:
        _be_scr_tau = float(os.environ.get('DLPNO_BE_SCREEN_TAU', '0') or 0)
        if _be_scr_tau > 0.0:
            from pyscf.cc.dlpno_tccsd._wm_scatter_cy import norms_by_offsets
            _uk_n_arr = be_plan['uk_n']
            _uk_t2off = be_plan['_uk_t2_off_c']
            _be_t2norm_pair = be_plan.get('_be_t2norm_buf')
            if _be_t2norm_pair is None:
                _be_t2norm_pair = np.empty(_uk_n_arr.shape[0])
                be_plan['_be_t2norm_buf'] = _be_t2norm_pair
            _be_uknorm_pair = be_plan.get('_be_uknorm_buf')
            if _be_uknorm_pair is None:
                _be_uknorm_pair = np.empty(_uk_n_arr.shape[0])
                be_plan['_be_uknorm_buf'] = _be_uknorm_pair
            norms_by_offsets(np.asarray(t2_pno_all._buffer), _uk_t2off,
                             _uk_n_arr, _be_t2norm_pair)
            norms_by_offsets(_uk_master_buf, _uk_t2off,
                             _uk_n_arr, _be_uknorm_pair)
    if be_plan is not None:
        if os.environ.get('DLPNO_BE_STATS') and not be_plan.get('_stats_done'):
            be_plan['_stats_done'] = True
            _tot_N = _tot_fl = _tot_by = 0
            _hist = {}
            for _bk in be_plan['buckets']:
                _ni, _nk = _bk['n_ij'], _bk['n_kl']
                _Nb = len(_bk['kl_keys'])
                _tot_N += _Nb
                # 4 dgemms: 2x (nk^2 * ni) + 2x (ni^2 * nk), 2 flops/MAC
                _tot_fl += _Nb * (4 * _nk * _nk * _ni + 4 * _ni * _ni * _nk)
                # gathered reads: S (ni*nk) + T (nk^2) + UK (nk^2), 8 B each
                _tot_by += _Nb * 8 * (_ni * _nk + 2 * _nk * _nk)
                _hist[(_ni, _nk)] = _hist.get((_ni, _nk), 0) + _Nb
            print(f'[BE_STATS] buckets={len(be_plan["buckets"])} items={_tot_N} '
                  f'flops={_tot_fl/1e9:.2f}G read={_tot_by/1e9:.2f}GB', flush=True)
            _top = sorted(_hist.items(), key=lambda kv: -kv[1] * (kv[0][0] * kv[0][1]))[:12]
            for (_ni, _nk), _Nb in _top:
                print(f'[BE_STATS]   n_ij={_ni:4d} n_kl={_nk:4d} N={_Nb}', flush=True)
        # BE slot-cat eligibility (DLPNO_BE_SLOTCAT, default ON): needs the
        # master flats + per-item offsets (mode-independent, present
        # whenever the S_pno/K_iajb stores share base buffers), the UK
        # hoist master, and no BE screening.  When eligible, the bucket
        # loop below builds the LIGHT gathered-style inv (no stacked
        # S_c/K_c copies, no per-cycle T_buf refill) even outside
        # DLPNO_BE_GATHERED mode.
        bes = be_plan.get('_bes_cache')
        _bes_env_on = (os.environ.get('DLPNO_BE_SLOTCAT', '1') != '0')
        _bes_eligible = (
            bes is not None
            or (_bes_env_on
                and _uk_master_buf is not None
                and _be_t2norm_pair is None
                and be_plan.get('S_pno_master') is not None
                and be_plan.get('K_iajb_master') is not None
                and hasattr(t2_pno_all, '_buffer')
                and all(bk.get('S_off') is not None
                        and bk.get('K_off') is not None
                        and bk.get('t2_off') is not None
                        for bk in be_plan['buckets'])))
        unique_n_ij = sorted(be_plan['pairs_by_n_ij'].keys())
        n_unique = len(unique_n_ij)
        flat_off = np.zeros(n_unique + 1, dtype=np.int64)
        for gi, n_ij in enumerate(unique_n_ij):
            n_pairs_in_group = len(be_plan['pairs_by_n_ij'][n_ij])
            flat_off[gi + 1] = flat_off[gi] + n_pairs_in_group * n_ij * n_ij

        # Build buckets array (PyBEInputs[N]).
        buckets = be_plan['buckets']
        BUCK_T = PyBEInputs * len(buckets)
        buckets_arr = BUCK_T()
        for b_idx, bucket in enumerate(buckets):
            n_ij = bucket['n_ij']
            n_kl = bucket['n_kl']
            N_b = len(bucket['kl_keys'])
            # Cache cycle-invariant arrays on the bucket dict (built once
            # per CCSD run; bucket survives across cycles via plan_cache).
            # Per-cycle work is reduced to: T_arr refill + scalar wiring.
            # Beta arrays are intentionally NOT computed here — the C++
            # solver refreshes them from B_tilde_flat at the top of
            # run_one_cycle (lines ~1601-1612 of dlpno_ccsd_solver.cpp),
            # so any work here is dead.
            # Gathered mode: BE plan kept only S_off/K_off into master flats
            # (S_pno_master, K_iajb_master) instead of stacked per-bucket
            # copies.  C kernel reads S/T/K directly via offsets — eliminate
            # T_buf, S_c, K_c entirely.  Saves ~1.5 GB at water-22, scales
            # as N_pair² × n_pno² (cc-pVTZ water-49+ requires this).
            # Gathered (memory-light) when either: full gathered_mode, OR the
            # cycle-0 deferred-stack path left only ref-lists but the master
            # flats + per-bucket offsets are available (class_gathered).  This
            # lets the class read S/T/K via offsets instead of rebuilding and
            # caching the +55 GiB S_c/K_c stacks for the whole class phase.
            _be_gathered = (
                (be_plan.get('gathered_mode') or be_plan.get('class_gathered'))
                and bucket.get('S_off') is not None
                and bucket.get('K_off') is not None
                and bucket.get('t2_off') is not None
                and be_plan.get('S_pno_master') is not None
                and be_plan.get('K_iajb_master') is not None)
            inv = bucket.get('_inv_cache')
            if inv is None:
                p_ij_arr    = np.empty(N_b, dtype=np.int32)
                dense_k_arr = np.empty(N_b, dtype=np.int32)
                dense_l_arr = np.empty(N_b, dtype=np.int32)
                for n in range(N_b):
                    key_ij_n, k_n, l_n = bucket['beta_coords'][n]
                    B_tilde = b_tilde_per_ij[key_ij_n]
                    if isinstance(B_tilde, tuple):
                        _, p_dense = B_tilde
                        dk = int(p_dense[k_n]); dl = int(p_dense[l_n])
                    else:
                        dk = int(k_n); dl = int(l_n)
                    p_ij_arr[n]    = key_to_p.get(key_ij_n, -1)
                    dense_k_arr[n] = dk
                    dense_l_arr[n] = dl
                # zeros, not empty: pack-time betas are refreshed in-class
                # for p>=0 items, but the slot-cat beta0 fallback READS
                # these for p<0/nlmo==0 items — garbage here would be a
                # silent landmine (none exist at current systems, but).
                beta_kl_arr = np.zeros(N_b)
                beta_lk_arr = np.zeros(N_b)
                same_c = np.ascontiguousarray(bucket['same']).astype(
                    np.uint8, copy=False)
                idx_c = np.ascontiguousarray(bucket['item_idx']).astype(
                    np.int64, copy=False)
                if _be_gathered or _bes_eligible:
                    # No stacked arrays; cache the offset arrays + master
                    # pointers used by every cycle.
                    S_off_c = np.ascontiguousarray(
                        bucket['S_off']).astype(np.int64, copy=False)
                    K_off_c = np.ascontiguousarray(
                        bucket['K_off']).astype(np.int64, copy=False)
                    T_off_c = np.ascontiguousarray(
                        bucket['t2_off']).astype(np.int64, copy=False)
                    inv = {
                        'beta_kl': beta_kl_arr, 'beta_lk': beta_lk_arr,
                        'p_ij_arr': p_ij_arr, 'dense_k_arr': dense_k_arr,
                        'dense_l_arr': dense_l_arr,
                        'same_c': same_c, 'idx_c': idx_c,
                        'S_off_c': S_off_c, 'T_off_c': T_off_c,
                        'K_off_c': K_off_c,
                        'gathered': True,
                    }
                else:
                    T_buf = np.empty((N_b, n_kl, n_kl))
                    # Deferred-stack BE plan (DLPNO_BE_DEFER_STACK): the plan
                    # holds light ref-lists, not stacked S/K — rebuild this
                    # bucket's stacks here (same as compute_B_E_batched).
                    _Sb = bucket['S']
                    _Kb = bucket['K']
                    if _Sb is None and bucket.get('S_list') is not None:
                        _Sb = np.empty((N_b, n_ij, n_kl))
                        _Kb = np.empty((N_b, n_kl, n_kl))
                        _Sl = bucket['S_list']
                        _Kl = bucket['K_list']
                        for _n in range(N_b):
                            _Sb[_n] = _Sl[_n]
                            _Kb[_n] = _Kl[_n]
                    S_c = np.ascontiguousarray(_Sb)
                    K_c = np.ascontiguousarray(_Kb)
                    kl_refs = [t2_pno_all[k] for k in bucket['kl_keys']]
                    inv = {
                        'T_buf': T_buf, 'beta_kl': beta_kl_arr,
                        'beta_lk': beta_lk_arr,
                        'p_ij_arr': p_ij_arr, 'dense_k_arr': dense_k_arr,
                        'dense_l_arr': dense_l_arr,
                        'S_c': S_c, 'K_c': K_c, 'same_c': same_c,
                        'idx_c': idx_c,
                        'kl_refs': kl_refs,
                        'gathered': False,
                    }
                bucket['_inv_cache'] = inv

            beta_kl_arr = inv['beta_kl']
            beta_lk_arr = inv['beta_lk']
            p_ij_arr    = inv['p_ij_arr']
            dense_k_arr = inv['dense_k_arr']
            dense_l_arr = inv['dense_l_arr']
            same_c      = inv['same_c']
            idx_c       = inv['idx_c']
            n_pairs_in_group = len(be_plan['pairs_by_n_ij'][n_ij])
            buckets_arr[b_idx].N = int(N_b)
            buckets_arr[b_idx].n_ij = int(n_ij)
            buckets_arr[b_idx].n_kl = int(n_kl)
            buckets_arr[b_idx].n_slots = int(n_pairs_in_group)
            buckets_arr[b_idx].beta_kl     = beta_kl_arr.ctypes.data
            buckets_arr[b_idx].beta_lk     = beta_lk_arr.ctypes.data
            buckets_arr[b_idx].same        = same_c.ctypes.data
            buckets_arr[b_idx].idx         = idx_c.ctypes.data
            buckets_arr[b_idx].p_ij_arr    = p_ij_arr.ctypes.data
            buckets_arr[b_idx].dense_k_arr = dense_k_arr.ctypes.data
            buckets_arr[b_idx].dense_l_arr = dense_l_arr.ctypes.data
            if inv.get('gathered'):
                buckets_arr[b_idx].S        = 0
                buckets_arr[b_idx].T        = 0
                buckets_arr[b_idx].K        = 0
                buckets_arr[b_idx].S_master = be_plan[
                    'S_pno_master'].ctypes.data
                buckets_arr[b_idx].T_master = t2_pno_all._buffer.ctypes.data
                if _uk_master_buf is not None:
                    # UK hoist: K slot carries the per-cycle UK master;
                    # per-item offsets are the T2-canonical ones (UK has
                    # T2's layout).  Kernel skips its internal UK build.
                    buckets_arr[b_idx].K_master = _uk_master_buf.ctypes.data
                    buckets_arr[b_idx].K_off   = inv['T_off_c'].ctypes.data
                    buckets_arr[b_idx].uk_hoisted = 1
                else:
                    buckets_arr[b_idx].K_master = be_plan[
                        'K_iajb_master'].ctypes.data
                    buckets_arr[b_idx].K_off   = inv['K_off_c'].ctypes.data
                    buckets_arr[b_idx].uk_hoisted = 0
                # BE screening: per-item ||T_kl|| / ||UK_kl|| gathered from
                # the per-pair norms (item -> pair via its T2-canonical
                # offset; the index map is cycle-invariant and cached).
                buckets_arr[b_idx].scr_t2n = None
                buckets_arr[b_idx].scr_ukn = None
                buckets_arr[b_idx].scr_tau = 0.0
                if _be_t2norm_pair is not None:
                    _pi_map = inv.get('_scr_pair_idx')
                    if _pi_map is None:
                        _pi_map = np.searchsorted(
                            _uk_t2off, np.asarray(inv['T_off_c']))
                        inv['_scr_pair_idx'] = _pi_map
                    # Static per-item ||S_ij,kl||^2 — the LOCALITY factor
                    # (distant items decay through S, not beta/T).  Folded
                    # into the norm arrays so the kernel test is unchanged.
                    _s2 = inv.get('_scr_s2')
                    if _s2 is None:
                        from pyscf.cc.dlpno_tccsd._wm_scatter_cy import (
                            norms_by_offsets_rect)
                        _Nb_i = _pi_map.shape[0]
                        _s2 = np.empty(_Nb_i)
                        _nr = np.full(_Nb_i, int(n_ij), dtype=np.int32)
                        _nc = np.full(_Nb_i, int(n_kl), dtype=np.int32)
                        norms_by_offsets_rect(
                            np.asarray(be_plan['S_pno_master']),
                            np.ascontiguousarray(inv['S_off_c']),
                            _nr, _nc, _s2)
                        np.square(_s2, out=_s2)
                        inv['_scr_s2'] = _s2
                    _s_t2n = inv.get('_scr_t2n_buf')
                    if _s_t2n is None:
                        _s_t2n = np.empty(_pi_map.shape[0])
                        inv['_scr_t2n_buf'] = _s_t2n
                        inv['_scr_ukn_buf'] = np.empty(_pi_map.shape[0])
                    _s_ukn = inv['_scr_ukn_buf']
                    np.take(_be_t2norm_pair, _pi_map, out=_s_t2n)
                    np.take(_be_uknorm_pair, _pi_map, out=_s_ukn)
                    _s_t2n *= _s2
                    _s_ukn *= _s2
                    buckets_arr[b_idx].scr_t2n = _s_t2n.ctypes.data
                    buckets_arr[b_idx].scr_ukn = _s_ukn.ctypes.data
                    buckets_arr[b_idx].scr_tau = _be_scr_tau
                    be_owned_buckets.extend([_s_t2n, _s_ukn, _s2])
                buckets_arr[b_idx].S_off    = inv['S_off_c'].ctypes.data
                buckets_arr[b_idx].T_off    = inv['T_off_c'].ctypes.data
                be_owned_buckets.extend([
                    beta_kl_arr, beta_lk_arr, same_c, idx_c,
                    p_ij_arr, dense_k_arr, dense_l_arr,
                    inv['S_off_c'], inv['T_off_c'], inv['K_off_c'],
                    be_plan['S_pno_master'], be_plan['K_iajb_master'],
                    t2_pno_all._buffer,
                ])
            else:
                T_buf       = inv['T_buf']
                S_c         = inv['S_c']
                K_c         = inv['K_c']
                kl_refs     = inv['kl_refs']
                # Per-cycle refill of T_buf in place from current t2_pno_all.
                for n, src in enumerate(kl_refs):
                    np.copyto(T_buf[n], src)
                T_c = T_buf
                buckets_arr[b_idx].S        = S_c.ctypes.data
                buckets_arr[b_idx].T        = T_c.ctypes.data
                buckets_arr[b_idx].K        = K_c.ctypes.data
                buckets_arr[b_idx].S_master = 0
                buckets_arr[b_idx].T_master = 0
                buckets_arr[b_idx].K_master = 0
                buckets_arr[b_idx].S_off    = 0
                buckets_arr[b_idx].T_off    = 0
                buckets_arr[b_idx].K_off    = 0
                be_owned_buckets.extend([S_c, T_c, K_c, beta_kl_arr,
                                         beta_lk_arr, same_c, idx_c,
                                         p_ij_arr, dense_k_arr, dense_l_arr])

        unique_arr = np.asarray(unique_n_ij, dtype=np.int32)
        own.extend([buckets_arr, unique_arr, flat_off])
        own.extend(be_owned_buckets)
        be_plan_buckets_arr = buckets_arr
        be_unique_n_ij = unique_arr
        be_flat_off = flat_off
        be_n_buckets = len(buckets)
        be_n_unique = n_unique

        # Per-pair (group, slot) lookup.
        be_pair_n_ij_idx_arr = np.full(n_canon_pairs, -1, dtype=np.int32)
        be_pair_slot_arr = np.zeros(n_canon_pairs, dtype=np.int32)
        for key, slot in be_plan['pair_to_slot'].items():
            p = key_to_p.get(key, -1)
            if p < 0:
                continue
            n_ij = pno_spaces[key]['n_pno']
            if n_ij in unique_n_ij:
                be_pair_n_ij_idx_arr[p] = unique_n_ij.index(n_ij)
                be_pair_slot_arr[p] = slot
        own.extend([be_pair_n_ij_idx_arr, be_pair_slot_arr])

        # ---- BE slot-cat plan (DLPNO_BE_SLOTCAT, default ON) ----
        # Cross-bucket item sort by global output slot + a slot-sorted
        # contiguous copy of the S blocks (S_cat) so the second
        # half-transform runs as one fat dgemm per slot.  Cycle-invariant;
        # built once and cached on be_plan.  Requires gathered mode with
        # the UK hoist and no BE screening (falls back to buckets else).
        if (bes is None and _bes_eligible
                and all(bk['n_ij'] in unique_n_ij for bk in buckets)):
            _gi_of = {v: i for i, v in enumerate(unique_n_ij)}
            _grp_np = np.array(
                [len(be_plan['pairs_by_n_ij'][v]) for v in unique_n_ij],
                dtype=np.int64)
            _slot_base = np.zeros(n_unique + 1, dtype=np.int64)
            np.cumsum(_grp_np, out=_slot_base[1:])
            n_slots_tot = int(_slot_base[-1])

            _gs, _toff, _soff, _nkl, _same = [], [], [], [], []
            _b0k, _b0l, _pij, _dk, _dl = [], [], [], [], []
            for bk in buckets:
                inv_b = bk['_inv_cache']
                gi = _gi_of[bk['n_ij']]
                _gs.append(_slot_base[gi]
                           + np.asarray(inv_b['idx_c'], dtype=np.int64))
                _toff.append(np.asarray(inv_b['T_off_c'], dtype=np.int64))
                _soff.append(np.asarray(inv_b['S_off_c'], dtype=np.int64))
                _nkl.append(np.full(len(bk['kl_keys']), bk['n_kl'],
                                    dtype=np.int32))
                _same.append(np.asarray(inv_b['same_c'], dtype=np.uint8))
                _b0k.append(np.asarray(inv_b['beta_kl'], dtype=np.float64))
                _b0l.append(np.asarray(inv_b['beta_lk'], dtype=np.float64))
                _pij.append(np.asarray(inv_b['p_ij_arr'], dtype=np.int32))
                _dk.append(np.asarray(inv_b['dense_k_arr'], dtype=np.int32))
                _dl.append(np.asarray(inv_b['dense_l_arr'], dtype=np.int32))
            gslot = np.concatenate(_gs)
            order = np.argsort(gslot, kind='stable')
            gslot = np.ascontiguousarray(gslot[order])
            s_T_off = np.ascontiguousarray(np.concatenate(_toff)[order])
            s_S_off = np.ascontiguousarray(np.concatenate(_soff)[order])
            s_nkl = np.ascontiguousarray(np.concatenate(_nkl)[order])
            s_same = np.ascontiguousarray(np.concatenate(_same)[order])
            s_b0k = np.ascontiguousarray(np.concatenate(_b0k)[order])
            s_b0l = np.ascontiguousarray(np.concatenate(_b0l)[order])
            s_pij = np.ascontiguousarray(np.concatenate(_pij)[order])
            s_dk = np.ascontiguousarray(np.concatenate(_dk)[order])
            s_dl = np.ascontiguousarray(np.concatenate(_dl)[order])
            del _gs, _toff, _soff, _nkl, _same, _b0k, _b0l, _pij, _dk, _dl

            counts = np.bincount(gslot, minlength=n_slots_tot)
            slot_ptr = np.zeros(n_slots_tot + 1, dtype=np.int64)
            np.cumsum(counts, out=slot_ptr[1:])
            _kend = np.cumsum(s_nkl.astype(np.int64))
            _k0g = _kend - s_nkl               # global row index (unsplit)
            _slot_k_start = np.zeros(n_slots_tot, dtype=np.int64)
            _nz = counts > 0
            _slot_k_start[_nz] = _k0g[slot_ptr[:-1][_nz]]
            item_k0 = np.ascontiguousarray(
                _k0g - np.repeat(_slot_k_start, counts))
            slot_KT = np.zeros(n_slots_tot, dtype=np.int64)
            np.add.at(slot_KT, gslot, s_nkl.astype(np.int64))
            slot_nij = np.zeros(n_slots_tot, dtype=np.int32)
            slot_out_off = np.zeros(n_slots_tot, dtype=np.int64)
            for gi, v in enumerate(unique_n_ij):
                b0, b1 = int(_slot_base[gi]), int(_slot_base[gi + 1])
                slot_nij[b0:b1] = v
                slot_out_off[b0:b1] = (int(flat_off[gi])
                                       + np.arange(b1 - b0, dtype=np.int64)
                                       * (v * v))
            _panel = slot_nij.astype(np.int64) * slot_KT
            slot_scat_off = np.zeros(n_slots_tot, dtype=np.int64)
            np.cumsum(_panel[:-1], out=slot_scat_off[1:])
            _scat_total = int(_panel.sum())
            S_cat = np.empty(_scat_total)
            from pyscf.cc.dlpno_tccsd._wm_scatter_cy import build_scat_master
            build_scat_master(
                np.asarray(be_plan['S_pno_master']).ravel(), s_S_off,
                np.ascontiguousarray(slot_nij[gslot]), s_nkl,
                np.ascontiguousarray(slot_scat_off[gslot]), item_k0,
                np.ascontiguousarray(slot_KT[gslot]), S_cat)
            max_nkl = int(s_nkl.max(initial=1))
            max_nij = int(slot_nij.max(initial=1))
            hcap = int(min(max(int(slot_KT.max(initial=1)), max_nkl), 8192))
            hcap = max(hcap, max_nkl)
            bes = dict(S_cat=S_cat, T_off=s_T_off, item_nkl=s_nkl,
                       item_k0=item_k0, same=s_same, beta0_kl=s_b0k,
                       beta0_lk=s_b0l, p_ij=s_pij, dense_k=s_dk,
                       dense_l=s_dl, slot_ptr=slot_ptr,
                       slot_scat_off=slot_scat_off, slot_KT=slot_KT,
                       slot_nij=slot_nij, slot_out_off=slot_out_off,
                       n_slots=n_slots_tot, max_nkl=max_nkl,
                       max_nij=max_nij, hcap=hcap,
                       T_master_buf=t2_pno_all._buffer,
                       UK_master_buf=_uk_master_buf)
            be_plan['_bes_cache'] = bes
            print(f'[BE-SLOTCAT] plan built: items={len(gslot)} '
                  f'slots={n_slots_tot} S_cat={_scat_total*8/1e9:.2f}GB '
                  f'hcap={hcap}', flush=True)
        if bes is not None and (not _bes_env_on
                                or _uk_master_buf is None
                                or _be_t2norm_pair is not None):
            bes = None    # config changed mid-run: fall back to buckets
        if bes is not None:
            own.extend(list(bes.values()))

    # ----------------------- CD -----------------------
    cd_cache = getattr(compute_CD_terms_batched, '_plan_cache', None)
    cd_plan = next(iter(cd_cache.values())) if cd_cache else None
    c_plan_struct = d_plan_struct = None
    c_target_ij_arr = c_target_ji_arr = None
    d_target_ij_arr = d_target_ji_arr = None
    if cd_plan is not None:
        bv = _get_or_build_cd_batched_view(cd_plan, pno_spaces, t2_pno_all)
        n_pno_offsets = bv['n_pno_offsets']

        # Build per-item canonical pair maps.  Walk back target_off → (n_pno, slot).
        # n_pno_offsets[n_pno] is the global flat-offset start.  slot = (off-start)//(n_pno²).
        def _off_to_pair(off, n_pno):
            if off < 0:
                return -1
            slot = (int(off) - int(n_pno_offsets[n_pno])) // (n_pno * n_pno)
            pairs_in_group = cd_plan['pairs_by_n_pno'][n_pno]
            if slot < 0 or slot >= len(pairs_in_group):
                return -1
            return key_to_p.get(pairs_in_group[slot], -1)

        # C side.
        c_N = bv['c_N']
        if c_N > 0:
            # ct_flat is NOT built here: when c_N>0 the class always wires
            # c_term_ct_ord_pair_idx, so the C++ run_one_cycle overrides
            # c_plan.ct_flat to point at the native (current-cycle) C_tilde_flat
            # via per-item ord offsets.  The old Python gather copied the
            # cycle-0 C_tilde_cache (stale AND duplicated, ~GiB) and was then
            # discarded — pure dead weight.  Skip it entirely.
            ct_flat = None
            # t2: offset-alias into the canonical t2_pno_all buffer (read
            # t2[key] directly; the kernel applies the per-item transpose by
            # flipping its GT dgemm flag) instead of gathering a per-item-
            # DUPLICATED t2_flat copy each cycle.  The unique t2 buffer is tiny
            # (~0.05 GiB) but the gather inflates it to GiBs (same t2[key]
            # copied for every item sharing the key) — the dominant CCSD-cycle
            # anon on TM complexes.  Falls back to the gather when canonical
            # offsets are unavailable.
            _t2_alias = bv.get('c_t2_canon_off') is not None
            if _t2_alias:
                _c_t2_base_ptr = t2_pno_all._buffer.ctypes.data
                _c_t2_off_ptr  = bv['c_t2_canon_off'].ctypes.data
                _c_t2_tr_ptr   = bv['c_t2_trans'].ctypes.data  # bool/uint8
            else:
                if '_c_t2_flat_buf' in bv:
                    t2_flat = bv['_c_t2_flat_buf']
                else:
                    t2_flat = np.empty(int(bv['c_t2_off'][-1]))
                    bv['_c_t2_flat_buf'] = t2_flat
                gather_t2_with_transpose(
                    c_N, bv['c_n_other'],
                    bv['c_t2_canon_off'], bv['c_t2_trans_arr'],
                    bv['c_t2_off'], t2_pno_all._buffer, t2_flat,
                    min(64, c_N))
                _c_t2_base_ptr = t2_flat.ctypes.data
                _c_t2_off_ptr  = bv['c_t2_off'].ctypes.data
                _c_t2_tr_ptr   = 0
            c_plan_struct = PyCTermInputs()
            c_plan_struct.N           = int(c_N)
            c_plan_struct.n_pno_arr   = bv['c_n_pno'].ctypes.data
            c_plan_struct.n_ct_arr    = bv['c_n_ct'].ctypes.data
            c_plan_struct.n_other_arr = bv['c_n_other'].ctypes.data
            c_plan_struct.S_big_off   = bv['c_S_big_off'].ctypes.data
            c_plan_struct.ct_off      = bv['c_ct_off'].ctypes.data
            c_plan_struct.S_mid_off   = bv['c_S_mid_off'].ctypes.data
            c_plan_struct.J_bold_off  = bv['c_J_bold_off'].ctypes.data
            c_plan_struct.t2_off      = _c_t2_off_ptr
            c_plan_struct.S_outer_off = bv['c_S_outer_off'].ctypes.data
            c_plan_struct.tile_off    = bv['c_tile_off'].ctypes.data
            c_plan_struct.S_big_flat  = bv['c_S_big_flat'].ctypes.data
            c_plan_struct.S_mid_flat  = bv['c_S_mid_flat'].ctypes.data
            c_plan_struct.J_bold_flat = bv['c_J_bold_flat'].ctypes.data
            c_plan_struct.S_outer_flat= bv['c_S_outer_flat'].ctypes.data
            c_plan_struct.ct_flat     = 0   # overridden by C++ (C_tilde_flat)
            c_plan_struct.t2_flat     = _c_t2_base_ptr
            c_plan_struct.t2_trans    = _c_t2_tr_ptr
            c_plan_struct.max_n_pno   = int(bv['c_n_pno'].max(initial=1))
            c_plan_struct.max_n_ct    = int(bv['c_n_ct'].max(initial=1))
            c_plan_struct.max_n_other = int(bv['c_n_other'].max(initial=1))
            # Target arrays are cycle-invariant (depend only on plan
            # structure). Cache on bv after first build.
            if 'c_target_ij_arr_cached' in bv:
                c_target_ij_arr = bv['c_target_ij_arr_cached']
                c_target_ji_arr = bv['c_target_ji_arr_cached']
            else:
                c_target_ij_arr = np.full(c_N, -1, dtype=np.int32)
                c_target_ji_arr = np.full(c_N, -1, dtype=np.int32)
                c_n_pno_arr = bv['c_n_pno']
                c_target_ij_off = bv['c_target_off_ij']
                c_target_ji_off = bv['c_target_off_ji']
                for n in range(c_N):
                    np_n = int(c_n_pno_arr[n])
                    if c_target_ij_off[n] >= 0:
                        c_target_ij_arr[n] = _off_to_pair(c_target_ij_off[n], np_n)
                    if c_target_ji_off[n] >= 0:
                        c_target_ji_arr[n] = _off_to_pair(c_target_ji_off[n], np_n)
                bv['c_target_ij_arr_cached'] = c_target_ij_arr
                bv['c_target_ji_arr_cached'] = c_target_ji_arr
            own.extend([c_target_ij_arr, c_target_ji_arr])
            if not _t2_alias:
                own.append(t2_flat)   # aliased t2 stays alive via t2_pno_all

        # D side.
        d_N = bv['d_N']
        if d_N > 0:
            # u = 2*t2 - t2.T: offset-alias the canonical t2_pno_all buffer and
            # compute u in-kernel (per item) instead of gathering a per-item-
            # DUPLICATED u_flat copy each cycle.  Falls back to the gather when
            # canonical offsets are unavailable.
            _d_u_alias = bv.get('d_t2_canon_off') is not None
            if _d_u_alias:
                _d_u_base_ptr = t2_pno_all._buffer.ctypes.data
                _d_u_off_ptr  = bv['d_t2_canon_off'].ctypes.data
                _d_u_tr_ptr   = bv['d_t2_trans'].ctypes.data  # bool/uint8
                _d_u_flat_ptr = 0          # kernel computes u in-scratch
                u_flat = None
            else:
                if '_d_u_flat_buf' in bv:
                    u_flat = bv['_d_u_flat_buf']
                else:
                    u_flat = np.empty(int(bv['d_u_off'][-1]))
                    bv['_d_u_flat_buf'] = u_flat
                gather_u_from_t2(
                    d_N, bv['d_n_A'],
                    bv['d_t2_canon_off'], bv['d_t2_trans_arr'],
                    bv['d_u_off'], t2_pno_all._buffer, u_flat,
                    min(64, d_N))
                _d_u_base_ptr = 0
                _d_u_off_ptr  = bv['d_u_off'].ctypes.data
                _d_u_tr_ptr   = 0
                _d_u_flat_ptr = u_flat.ctypes.data
            # dt_flat NOT built here (mirrors ct_flat): the C++ overrides
            # d_plan.dt_flat to the native current-cycle D_tilde_flat via ord
            # offsets, so the cycle-0 D_tilde_cache gather was stale dead
            # weight (~GiB).  Skip it.
            dt_flat = None
            d_plan_struct = PyDTermInputs()
            d_plan_struct.N         = int(d_N)
            d_plan_struct.n_pno_arr = bv['d_n_pno'].ctypes.data
            d_plan_struct.n_A_arr   = bv['d_n_A'].ctypes.data
            d_plan_struct.n_B_arr   = bv['d_n_B'].ctypes.data
            d_plan_struct.S_a_off   = bv['d_S_a_off'].ctypes.data
            d_plan_struct.u_off     = _d_u_off_ptr
            d_plan_struct.S_b_off   = bv['d_S_b_off'].ctypes.data
            d_plan_struct.S_c_off   = bv['d_S_c_off'].ctypes.data
            d_plan_struct.dt_off    = bv['d_dt_off'].ctypes.data
            d_plan_struct.KJ_off    = bv['d_KJ_off'].ctypes.data
            d_plan_struct.tile_off  = bv['d_tile_off'].ctypes.data
            d_plan_struct.S_a_flat  = bv['d_S_a_flat'].ctypes.data
            d_plan_struct.S_b_flat  = bv['d_S_b_flat'].ctypes.data
            d_plan_struct.S_c_flat  = bv['d_S_c_flat'].ctypes.data
            d_plan_struct.KJ_flat   = bv['d_KJ_flat'].ctypes.data
            d_plan_struct.u_flat    = _d_u_flat_ptr
            d_plan_struct.dt_flat   = 0   # overridden by C++ (D_tilde_flat)
            d_plan_struct.u_base    = _d_u_base_ptr
            d_plan_struct.u_canon_off = _d_u_off_ptr if _d_u_alias else 0
            d_plan_struct.u_trans   = _d_u_tr_ptr
            d_plan_struct.max_n_pno = int(bv['d_n_pno'].max(initial=1))
            d_plan_struct.max_n_A   = int(bv['d_n_A'].max(initial=1))
            d_plan_struct.max_n_B   = int(bv['d_n_B'].max(initial=1))
            if 'd_target_ij_arr_cached' in bv:
                d_target_ij_arr = bv['d_target_ij_arr_cached']
                d_target_ji_arr = bv['d_target_ji_arr_cached']
            else:
                d_target_ij_arr = np.full(d_N, -1, dtype=np.int32)
                d_target_ji_arr = np.full(d_N, -1, dtype=np.int32)
                d_n_pno_arr = bv['d_n_pno']
                d_target_ij_off = bv['d_target_off_ij']
                d_target_ji_off = bv['d_target_off_ji']
                for n in range(d_N):
                    np_n = int(d_n_pno_arr[n])
                    if d_target_ij_off[n] >= 0:
                        d_target_ij_arr[n] = _off_to_pair(d_target_ij_off[n], np_n)
                    if d_target_ji_off[n] >= 0:
                        d_target_ji_arr[n] = _off_to_pair(d_target_ji_off[n], np_n)
                bv['d_target_ij_arr_cached'] = d_target_ij_arr
                bv['d_target_ji_arr_cached'] = d_target_ji_arr
            own.extend([d_target_ij_arr, d_target_ji_arr])
            if not _d_u_alias:
                own.append(u_flat)   # aliased u computed in-kernel from t2

    return {
        'g_plan_ik': g_plan_struct_ik, 'g_plan_jk': g_plan_struct_jk,
        'g_target_ik': g_target_ik_arr, 'g_target_jk': g_target_jk_arr,
        'be_n_buckets': be_n_buckets,
        'be_plan_buckets': be_plan_buckets_arr,
        'be_n_unique': be_n_unique,
        'be_unique_n_ij': be_unique_n_ij,
        'be_flat_off_per_n_ij': be_flat_off,
        'be_pair_n_ij_idx': be_pair_n_ij_idx_arr,
        'be_pair_slot': be_pair_slot_arr,
        'bes': bes,
        'c_plan': c_plan_struct, 'd_plan': d_plan_struct,
        'c_target_ij': c_target_ij_arr, 'c_target_ji': c_target_ji_arr,
        'd_target_ij': d_target_ij_arr, 'd_target_ji': d_target_ji_arr,
        'cd_bv': bv if cd_plan is not None else None,
    }, own


def _add_t34_plans_to_natives(natives, native_own, t1_cache, t2_pno_all,
                                ord_idx_lookup, key_to_p, nocc):
    """Augment natives dict + ownership with c_t3/c_t4/d_t3/d_t4 plans
    + per-CD-item ord_pair_idx for native ct_flat/dt_flat gather.

    Requires natives['cd_bv'] (the CD batched view, used to map
    c_ct_keys / d_dt_keys to ordered pair indices).
    """
    from pyscf.cc.dlpno_tccsd.residual import (
        compute_C_tilde_batched, build_D_tilde_batched)

    # ----- C_tilde t3+t4 -----
    c_cache = getattr(compute_C_tilde_batched, '_plan_cache', None)
    if c_cache:
        c_plan_obj = next(iter(c_cache.values()))
        c_t34, c_t34_own = _build_t34_plan(
            c_plan_obj, 'c', t1_cache, t2_pno_all, ord_idx_lookup, nocc)
        native_own.extend(c_t34_own)
        natives['c_t3_struct'] = c_t34['t3_struct']
        natives['c_t4_struct'] = c_t34['t4_struct']
        natives['c_t3_target_ord'] = c_t34['t3_target_ord']
        natives['c_t4_target_ord'] = c_t34['t4_target_ord']

    # ----- D_tilde t3+t4 -----
    d_cache = getattr(build_D_tilde_batched, '_plan_cache', None)
    if d_cache:
        d_plan_obj = next(iter(d_cache.values()))
        d_t34, d_t34_own = _build_t34_plan(
            d_plan_obj, 'd', t1_cache, t2_pno_all, ord_idx_lookup, nocc)
        native_own.extend(d_t34_own)
        natives['d_t3_struct'] = d_t34['t3_struct']
        natives['d_t4_struct'] = d_t34['t4_struct']
        natives['d_t3_target_ord'] = d_t34['t3_target_ord']
        natives['d_t4_target_ord'] = d_t34['t4_target_ord']

    # ----- Per-CD-item ord_pair_idx for native ct_flat/dt_flat gather. ---
    # Both arrays are cycle-invariant (depend only on bv keys + ord_idx_lookup);
    # cache on bv.
    bv = natives.get('cd_bv')
    if bv is not None:
        c_N = bv['c_N']
        if c_N > 0:
            if 'c_ct_ord_pair_idx_cached' in bv:
                c_ct_ord_pair_idx = bv['c_ct_ord_pair_idx_cached']
            else:
                c_ct_ord_pair_idx = np.empty(c_N, dtype=np.int32)
                for n in range(c_N):
                    k, i = bv['c_ct_keys'][n]  # ordered pair (k, i)
                    c_ct_ord_pair_idx[n] = ord_idx_lookup.get((k, i), -1)
                bv['c_ct_ord_pair_idx_cached'] = c_ct_ord_pair_idx
            native_own.append(c_ct_ord_pair_idx)
            natives['c_ct_ord_pair_idx'] = c_ct_ord_pair_idx

        d_N = bv['d_N']
        if d_N > 0:
            if 'd_dt_ord_pair_idx_cached' in bv:
                d_dt_ord_pair_idx = bv['d_dt_ord_pair_idx_cached']
            else:
                d_dt_ord_pair_idx = np.empty(d_N, dtype=np.int32)
                for n in range(d_N):
                    # bv['d_dt_keys'] holds the ordered pair (k, i) keys for
                    # D_tilde lookups (matches bv['c_ct_keys'] semantics).
                    k, i = bv['d_dt_keys'][n]
                    d_dt_ord_pair_idx[n] = ord_idx_lookup.get((k, i), -1)
                bv['d_dt_ord_pair_idx_cached'] = d_dt_ord_pair_idx
            native_own.append(d_dt_ord_pair_idx)
            natives['d_dt_ord_pair_idx'] = d_dt_ord_pair_idx

    return natives, native_own


def run_remaining_cycles_via_class(
        cycle_start, max_cycle, this_tol,
        t1_pno, t2_pno_all,
        cc_ints, pno_spaces, pair_lmo_idx, F_lmo, eps_lmo, fov_pno,
        nocc, keys_sorted, S_pno_cache,
        cc_ints_flat, pair_index, ovL_pno_cache, K_pno_cache,
        # PySCF caches/state needed for plan extractors:
        b_tilde_per_ij_pyscf, jiang_C_pyscf, jiang_D_pyscf,
        # DIIS state:
        mydiis, diis_start_cycle, strong_pairs, cas_blocks,
        verbose=True, _pool=None, ktc_store=None):
    """Pack-once optimized drop-in cycle driver.

    SolverInputs and per_kl plan are packed ONCE at function entry; per
    cycle only T1_flat / T2_flat / T1_in_pair are refreshed (not the
    full 210-pair flat buffers).  DIIS operates on flat buffers
    directly (no dict <-> array conversion).
    """
    import time as _time
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints
    from pyscf.cc.dlpno_tccsd.pair_index import build_t1_cache, PairIndex
    from pyscf.cc.dlpno_tccsd.lccsd import _compute_t1_residual

    _pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)

    if os.environ.get('DLPNO_CLASS_INPUT_DEBUG') == '1':
        import numpy as _np
        _fields = ('Qab', 'Qma', 'K_iajb', 'K_tilde_chem_i', 'K_tilde_chem_j',
                   'K_bar_chem', 'K_bar_ij', 'K_bar_ji', 'i_Qk', 'j_Qk',
                   'i_Qa', 'j_Qa')
        _sums = {f: 0.0 for f in _fields}
        for _k in sorted(cc_ints.keys()):
            _e = cc_ints.get(_k)
            if not isinstance(_e, dict):
                continue
            for f in _fields:
                a = _e.get(f)
                if a is not None:
                    _sums[f] += float(_np.abs(a).sum())
        _spno = (float(_np.abs(S_pno_cache._buffer).sum())
                 if hasattr(S_pno_cache, '_buffer') else -1.0)
        print('[CLASS_INPUT] ' + ' '.join(f'{f}={_sums[f]:.4f}'
              for f in _fields) + f' Spno={_spno:.4f}', flush=True)

    # ---- ONE-TIME setup ----
    _t_setup0 = _time.perf_counter()
    def _pmark(label, t0):
        pass
    _t = _time.perf_counter()
    if hasattr(_compute_t1_residual, '_per_kl_plan_cache'):
        _compute_t1_residual._per_kl_plan_cache.clear()
    # Plan-only call: builds + caches _per_kl_plan_cache without running
    # the residual computation itself (we discard the result anyway).
    # Saves ~1.0s of one-time setup on water-15.
    _compute_t1_residual(
        t1_pno, t2_pno_all, pno_spaces, fov_pno, F_lmo, eps_lmo, nocc,
        S_pno_cache, cc_ints, ovL_pno_cache=ovL_pno_cache,
        pair_lmo_idx=pair_lmo_idx, t1_cache=None, _pool=None,
        cc_ints_flat=cc_ints_flat, pair_index=pair_index,
        _plan_only=True)
    _pmark('t1_residual plan_only', _t)
    _t = _time.perf_counter()
    t1_cache = build_t1_cache(t1_pno, _pi, S_pno_cache, pno_spaces)
    _pmark('build_t1_cache', _t)
    _t = _time.perf_counter()
    _all_keys = sorted(t2_pno_all.keys())
    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, _all_keys,
        t2_pno_all=t2_pno_all, S_pno_cache=S_pno_cache, _pool=_pool,
        cc_ints_flat=cc_ints_flat, pair_index=pair_index)
    _pmark('pack_for_t1_ints', _t)

    # K_tilde_chem_{i,j} were just copied into the pack's own flat store
    # (inputs.K_tilde_chem_{i,j}); the C++ class reads them from there for the
    # whole cycle loop.  The combined-ktilde buffer (ktc_store['buf'], ~8 GiB
    # on a TM complex) and the per-pair cc_ints views into it are now dead
    # weight — drop them so the ~8 GiB is reclaimed before the cycle loop.
    if ktc_store is not None:
        for _k in (list(ktc_store.get('off_i', {}).keys())
                   + list(ktc_store.get('off_j', {}).keys())):
            _e = cc_ints.get(_k)
            if _e is not None:
                _e.pop('K_tilde_chem_i', None)
                _e.pop('K_tilde_chem_j', None)
        ktc_store['buf'] = None
        ktc_store['off_i'] = {}
        ktc_store['off_j'] = {}
        import gc as _gc
        _gc.collect()

    keys_reorder = aux['keys_sorted']
    n_pairs = len(keys_reorder)
    npno = aux['n_pno_per_pair']
    pno_offsets = aux['pno_offsets']
    t2_offsets = aux['t2_offsets']
    pair_lmo_lists = aux['pair_lmo_lists']
    T1_flat_arr = aux['T1_flat']
    T2_flat_arr = aux['T2_flat']

    # Strong-pair flag for energy formula.
    is_strong_arr = np.zeros(n_pairs, dtype=np.uint8)
    _strong_keys_set = set(strong_pairs) | {(i, i) for i in range(nocc)}
    for p, key in enumerate(keys_reorder):
        if key in _strong_keys_set:
            is_strong_arr[p] = 1
    ownership.append(is_strong_arr)
    inputs.is_strong_pair = is_strong_arr.ctypes.data

    # per_kl plan extracted once.
    _t = _time.perf_counter()
    from pyscf.cc.dlpno_tccsd._ccsd_solver import (
        _extract_per_kl_plan, _extract_g_tilde_plan)
    plan_struct, plan_own = _extract_per_kl_plan(_compute_t1_residual)
    if plan_struct is None:
        raise RuntimeError('per_kl plan unavailable')
    plan_struct.t2_buffer       = t2_pno_all._buffer.ctypes.data
    plan_struct.t1_cache_buffer = t1_cache._buffer.ctypes.data
    g_plan_struct, g_plan_own = _extract_g_tilde_plan(key_to_p)
    _pmark('extract per_kl + g_tilde plans', _t)
    if os.environ.get('DLPNO_PACK_PROBE'):
        def _gb(lst):
            return sum(getattr(a, 'nbytes', 0) for a in lst) / 2**30
        print(f'  [class_probe] pack_own={_gb(ownership):.1f} GiB '
              f'per_kl_plan={_gb(plan_own):.1f} GiB '
              f'g_tilde_plan={_gb(g_plan_own):.1f} GiB', flush=True)
    # Note: the class drop-in keeps reading per-cycle gather metadata
    # (`bv['c_t2_canon_off']`, `bv['c_t2_trans_arr']`, etc.) from inside
    # the python plan dicts during the cycle loop. Clearing the plan
    # caches here corrupted the residual at S22-1 (E_int went from
    # -1.65 to -26.6 kcal/mol) — not safe. The plan caches stay alive
    # until run_lccsd returns, and `_free_ccsd_plan_caches()` clears
    # them in the driver between Stage 5 and Stage 6.

    # Strong-pair mask (cycle-invariant).  For DIIS we emit T1 (nocc slots)
    # + T2 over STRONG pairs (matching PySCF DIIS vector layout).
    strong_pair_indices = [p for p, key in enumerate(keys_reorder)
                           if key in _strong_keys_set]
    R1_size = sum(int(npno[i]) for i in range(nocc))
    R2_size = int(t2_offsets[n_pairs])
    R1_flat = np.zeros(R1_size, dtype=np.float64)
    R2_flat = np.zeros(R2_size, dtype=np.float64)

    # DIIS amplitude/error vector layout.
    # T1: pno_offsets[nocc] doubles.  T2: sum over strong pairs of npno^2.
    # CAS blocks: t2[cas_sl, cas_sl] zeroed before vectorization.
    diis_t2_size = sum(int(npno[p]) ** 2 for p in strong_pair_indices)
    diis_amp_size = int(pno_offsets[nocc]) + diis_t2_size

    def _amp_to_vec(T1_flat, T2_flat):
        """Build DIIS amp vector from current flat T1/T2 (with CAS mask)."""
        vec = np.empty(diis_amp_size, dtype=np.float64)
        vec[:pno_offsets[nocc]] = T1_flat[:pno_offsets[nocc]]
        off = int(pno_offsets[nocc])
        for p in strong_pair_indices:
            n_p = int(npno[p])
            if n_p == 0:
                continue
            sl = slice(int(t2_offsets[p]), int(t2_offsets[p + 1]))
            t2k = T2_flat[sl].reshape(n_p, n_p).copy()
            key = keys_reorder[p]
            if key in cas_blocks:
                cas_sl = cas_blocks[key][0]
                t2k[cas_sl, cas_sl] = 0.0
            vec[off:off + n_p * n_p] = t2k.ravel()
            off += n_p * n_p
        return vec

    def _vec_to_amp(vec, T1_flat, T2_flat):
        """Unpack DIIS-output amp vector back to T1_flat / T2_flat."""
        T1_flat[:pno_offsets[nocc]] = vec[:pno_offsets[nocc]]
        off = int(pno_offsets[nocc])
        for p in strong_pair_indices:
            n_p = int(npno[p])
            if n_p == 0:
                continue
            sl = slice(int(t2_offsets[p]), int(t2_offsets[p + 1]))
            T2_flat[sl] = vec[off:off + n_p * n_p]
            off += n_p * n_p
        # Restore CAS blocks (post-DIIS).
        for key, cb in cas_blocks.items():
            if key not in key_to_p:
                continue
            p = key_to_p[key]
            n_p = int(npno[p])
            if n_p == 0:
                continue
            sl_full = slice(int(t2_offsets[p]), int(t2_offsets[p + 1]))
            t2_block = T2_flat[sl_full].reshape(n_p, n_p)
            cas_sl = cb[0]
            if len(cb) == 3:
                _, t2c_dmrg_ref, t2c_mp2 = cb
                t2_block[cas_sl, cas_sl] = t2c_mp2
            else:
                t2_block[cas_sl, cas_sl] = cb[1]

    T1_in_pair_flat = aux['T1_in_pair_flat']
    T1_in_pair_offs = aux['T1_in_pair_offs']
    T1_in_pair_full_flat = aux['T1_in_pair_full_flat']
    T1_in_pair_full_offs = aux['T1_in_pair_full_offs']

    def _refresh_T1_in_pair(t1_cache_obj):
        """Update T1_in_pair / T1_in_pair_full FLAT buffers from t1_cache.
        The class kernels read from the flat buffer; per-pair list views
        are stale until we re-flatten."""
        for p, key in enumerate(keys_reorder):
            n_p = int(npno[p])
            nlmo_p = int(pair_lmo_lists[p].size)
            if n_p == 0:
                continue
            full_view = t1_cache_obj[key]  # (nocc, n_p)
            # T1_in_pair: pair-domain view (nlmo_p, n_p).
            if nlmo_p > 0:
                lmo_idx = np.asarray(pair_lmo_lists[p], dtype=np.intp)
                T1_in_pair_flat[T1_in_pair_offs[p]:T1_in_pair_offs[p + 1]] = (
                    full_view[lmo_idx].ravel())
            # T1_in_pair_full: full (nocc, n_p).
            T1_in_pair_full_flat[
                T1_in_pair_full_offs[p]:T1_in_pair_full_offs[p + 1]] = (
                full_view.ravel())

    _t_setup = _time.perf_counter() - _t_setup0
    print(f'[CCSD MONO PACK-ONCE] setup: {_t_setup:.2f}s', flush=True)

    # Per-cycle refresh helpers for T1-dependent intermediates.  PySCF's
    # cycle loop rebuilds these each iteration; in our drop-in we must
    # do the same so BE plan's beta_kl/lk are in sync with current t1.
    # (CD plan's C_tilde / D_tilde are built natively by the class via
    #  c_term_ct_ord_pair_idx / d_term_dt_ord_pair_idx — no Python
    #  refresh needed.)
    e_prev = float('-inf')

    # Inside `run_one_cycle` every C-kernel reads `omp_get_max_threads()`
    # (clamped by solver_team_size() in dlpno_ccsd_solver.cpp) to set its
    # team size.  When the driver runs with `OMP_NUM_THREADS=1` (so the
    # Python pool can dispatch cc_ints / S_pno builds without OMP
    # oversubscription), this returns 1 and every class kernel runs
    # SINGLE-THREADED.  The pool is idle while the class iterates, so it's
    # safe (and a big win) to bump the OMP team inside the cycle loop only.
    #
    # Default 32: an MOBH35-33 def2-tzvpp sweep (2026-05-19) gave per-cycle
    # wall 3.80 s @16, 3.50 s @32, 3.55 s @48, 3.72 s @64 — 32 is the sweet
    # spot.  Past 32 the dominant phases (BE kernel, C/D-tilde) stop scaling
    # and thread contention regresses the total.  Override via
    # DLPNO_CCSD_CYCLE_OMP; the C side honours up to DLPNO_SOLVER_MAX_THREADS.
    try:
        from threadpoolctl import threadpool_limits as _tpl
    except ImportError:
        _tpl = None
    _omp_n = int(os.environ.get('DLPNO_CCSD_CYCLE_OMP', '32'))
    _omp_ctx = (_tpl(limits=_omp_n, user_api='openmp')
                 if _tpl is not None and _omp_n > 1 else None)
    if _omp_ctx is not None:
        _omp_ctx.__enter__()

    # ORCA-parity dual stopping criterion (see the check in the loop).
    _conv_orca = os.environ.get('DLPNO_CONV_ORCA', '1') != '0'
    _tol_e = float(os.environ.get('DLPNO_TOL_E', '1e-8'))
    _tol_err = float(os.environ.get('DLPNO_TOL_ERR', '5e-7'))

    for cycle in range(cycle_start, max_cycle):
        _t_cyc_start = _time.perf_counter()

        # Per-iter: rebuild t1_cache from current t1_pno (DIIS may have
        # mixed it; t1_pno is the reference).  Sync T1_flat from t1_pno.
        # All T1-dressed intermediates (B_tilde, C_tilde, D_tilde,
        # G_tilde, t1_ints) are rebuilt natively by the class on each
        # cycle — Python passes through cycle-0 dicts unchanged.
        _t_t1cache0 = _time.perf_counter()
        t1_cache = build_t1_cache(t1_pno, _pi, S_pno_cache, pno_spaces)
        plan_struct.t1_cache_buffer = t1_cache._buffer.ctypes.data
        # Sync T1_flat (per occupied i) from t1_pno.
        for ii in range(nocc):
            n_ii = int(npno[ii])
            if n_ii == 0:
                continue
            T1_flat_arr[pno_offsets[ii]:pno_offsets[ii] + n_ii] = t1_pno[ii]
        # Sync T2_flat from t2_pno_all (DIIS may have mixed).
        for p, key in enumerate(keys_reorder):
            n_p = int(npno[p])
            if n_p == 0:
                continue
            if key in t2_pno_all:
                T2_flat_arr[t2_offsets[p]:t2_offsets[p + 1]] = (
                    t2_pno_all[key].ravel())
        _refresh_T1_in_pair(t1_cache)
        _t_t1cache = _time.perf_counter() - _t_t1cache0

        # Per-iter: rebuild plan inputs (T_arr/beta in BE, t2_flat/u_flat
        # in CD/G_term/t34 — depend on T2 / T1).
        _t_plans0 = _time.perf_counter()
        from pyscf.cc.dlpno_tccsd._ccsd_solver import (
            _build_native_r2_plans, _add_t34_plans_to_natives)
        natives, native_own = _build_native_r2_plans(
            t2_pno_all, key_to_p, keys_reorder, pno_spaces,
            b_tilde_per_ij_pyscf, jiang_C_pyscf, jiang_D_pyscf, n_pairs)
        ord_idx_lookup = {}
        for o in range(int(aux['ordered_pair_i_idx'].size)):
            a = int(aux['ordered_pair_i_idx'][o])
            b = int(aux['ordered_pair_k_idx'][o])
            ord_idx_lookup[(a, b)] = o
        natives, native_own = _add_t34_plans_to_natives(
            natives, native_own, t1_cache, t2_pno_all,
            ord_idx_lookup, key_to_p, nocc)
        _t_plans = _time.perf_counter() - _t_plans0
        if os.environ.get('DLPNO_PACK_PROBE') and cycle == cycle_start:
            _nat_gb = sum(getattr(a, 'nbytes', 0)
                          for a in native_own) / 2**30
            print(f'  [class_probe] native_r2_plans(t34/c/d/g)='
                  f'{_nat_gb:.1f} GiB', flush=True)
        if os.environ.get('DLPNO_MEM_PROBE'):
            try:
                from pyscf.cc.dlpno_tccsd.driver import _log_mem as _lm
                _lm(f'class_after_native_plans_cyc{cycle}')
            except Exception:
                pass
        if cycle <= cycle_start + 1:   # first two class cycles only
            try:
                from pyscf.cc.dlpno_tccsd.driver import _dump_anon_profile
                _dump_anon_profile(f'cyc{cycle}', extra_roots={
                    'jiang_C': jiang_C_pyscf, 'jiang_D': jiang_D_pyscf,
                    'native_own': native_own, 'natives': natives,
                    't2_pno_all': t2_pno_all, 't1_cache': t1_cache,
                    'ownership': ownership})
            except Exception:
                pass

        # Wire plans (most fields cycle-invariant; ptrs may rebind).
        plans = PyRunCycleInputs()
        for fname in ('g_tilde_plan', 'be_plan', 'c_term_plan',
                      'd_term_plan', 'g_term_plan', 't3_plan', 't4_plan',
                      'g_term_plan_jk',
                      'c_t3_plan', 'c_t4_plan', 'd_t3_plan', 'd_t4_plan'):
            setattr(plans, fname, None)
        plans.per_kl_plan = ctypes.pointer(plan_struct)
        if g_plan_struct is not None:
            plans.g_tilde_plan = ctypes.pointer(g_plan_struct)
        if natives['g_plan_ik'] is not None:
            plans.g_term_plan = ctypes.pointer(natives['g_plan_ik'])
            plans.g_term_plan_jk = ctypes.pointer(natives['g_plan_jk'])
            plans.g_term_target_pair_idx_ik = natives['g_target_ik'].ctypes.data
            plans.g_term_target_pair_idx_jk = natives['g_target_jk'].ctypes.data
        plans.be_n_buckets    = natives['be_n_buckets']
        plans.be_plan_buckets = (
            ctypes.addressof(natives['be_plan_buckets'])
            if natives['be_plan_buckets'] is not None else 0)
        plans.be_n_unique_n_ij = natives['be_n_unique']
        plans.be_unique_n_ij = (natives['be_unique_n_ij'].ctypes.data
            if natives['be_unique_n_ij'] is not None else None)
        plans.be_flat_off_per_n_ij = (natives['be_flat_off_per_n_ij'].ctypes.data
            if natives['be_flat_off_per_n_ij'] is not None else None)
        plans.be_pair_n_ij_idx = (natives['be_pair_n_ij_idx'].ctypes.data
            if natives['be_pair_n_ij_idx'] is not None else None)
        plans.be_pair_slot = (natives['be_pair_slot'].ctypes.data
            if natives['be_pair_slot'] is not None else None)
        _bes = natives.get('bes')
        if _bes is not None:
            plans.bes_S_cat = _bes['S_cat'].ctypes.data
            plans.bes_T_master = _bes['T_master_buf'].ctypes.data
            plans.bes_UK_master = _bes['UK_master_buf'].ctypes.data
            plans.bes_T_off = _bes['T_off'].ctypes.data
            plans.bes_item_nkl = _bes['item_nkl'].ctypes.data
            plans.bes_item_k0 = _bes['item_k0'].ctypes.data
            plans.bes_same = _bes['same'].ctypes.data
            plans.bes_beta0_kl = _bes['beta0_kl'].ctypes.data
            plans.bes_beta0_lk = _bes['beta0_lk'].ctypes.data
            plans.bes_p_ij = _bes['p_ij'].ctypes.data
            plans.bes_dense_k = _bes['dense_k'].ctypes.data
            plans.bes_dense_l = _bes['dense_l'].ctypes.data
            plans.bes_slot_ptr = _bes['slot_ptr'].ctypes.data
            plans.bes_slot_scat_off = _bes['slot_scat_off'].ctypes.data
            plans.bes_slot_KT = _bes['slot_KT'].ctypes.data
            plans.bes_slot_nij = _bes['slot_nij'].ctypes.data
            plans.bes_slot_out_off = _bes['slot_out_off'].ctypes.data
            plans.bes_n_slots = int(_bes['n_slots'])
            plans.bes_max_nkl = int(_bes['max_nkl'])
            plans.bes_max_nij = int(_bes['max_nij'])
            plans.bes_hcap = int(_bes['hcap'])
        if natives['c_plan'] is not None:
            plans.c_term_plan = ctypes.pointer(natives['c_plan'])
            plans.c_term_target_pair_idx_ij = natives['c_target_ij'].ctypes.data
            plans.c_term_target_pair_idx_ji = natives['c_target_ji'].ctypes.data
        if natives['d_plan'] is not None:
            plans.d_term_plan = ctypes.pointer(natives['d_plan'])
            plans.d_term_target_pair_idx_ij = natives['d_target_ij'].ctypes.data
            plans.d_term_target_pair_idx_ji = natives['d_target_ji'].ctypes.data
        for k_struct, k_target, attr in [
                ('c_t3_struct', 'c_t3_target_ord', 'c_t3'),
                ('c_t4_struct', 'c_t4_target_ord', 'c_t4'),
                ('d_t3_struct', 'd_t3_target_ord', 'd_t3'),
                ('d_t4_struct', 'd_t4_target_ord', 'd_t4')]:
            s = natives.get(k_struct)
            t = natives.get(k_target)
            if s is not None:
                setattr(plans, f'{attr}_plan', ctypes.pointer(s))
                setattr(plans, f'{attr}_target_ord_idx',
                        t.ctypes.data if t is not None else None)
        plans.c_term_ct_ord_pair_idx = (
            natives['c_ct_ord_pair_idx'].ctypes.data
            if 'c_ct_ord_pair_idx' in natives else None)
        plans.d_term_dt_ord_pair_idx = (
            natives['d_dt_ord_pair_idx'].ctypes.data
            if 'd_dt_ord_pair_idx' in natives else None)

        # Snapshot pre-update amp.
        amp_old = _amp_to_vec(T1_flat_arr, T2_flat_arr)

        # Run class one cycle.
        _t_run0 = _time.perf_counter()
        out = PyRunCycleOutputs()
        out.R1_flat = R1_flat.ctypes.data
        out.R2_flat = R2_flat.ctypes.data
        out.energy = 0.0
        out.G_tilde_out = 0
        rc = _libcc.DLPNOcompute_lccsd_run_one_cycle(
            ctypes.byref(inputs), ctypes.byref(plans), ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f'class run_one_cycle rc={rc}')
        _t_run = _time.perf_counter() - _t_run0

        # Build amp_new from updated T1_flat / T2_flat (mutated in place).
        # Build err vec from R1, R2 (over strong pairs).
        amp_new = _amp_to_vec(T1_flat_arr, T2_flat_arr)
        err_vec = np.empty(diis_amp_size, dtype=np.float64)
        err_vec[:pno_offsets[nocc]] = R1_flat[:pno_offsets[nocc]]
        off = int(pno_offsets[nocc])
        for p in strong_pair_indices:
            n_p = int(npno[p])
            if n_p == 0:
                continue
            sl = slice(int(t2_offsets[p]), int(t2_offsets[p + 1]))
            r2k = R2_flat[sl].reshape(n_p, n_p).copy()
            key = keys_reorder[p]
            if key in cas_blocks:
                cas_sl = cas_blocks[key][0]
                r2k[cas_sl, cas_sl] = 0.0
            err_vec[off:off + n_p * n_p] = r2k.ravel()
            off += n_p * n_p
        dT = float(np.max(np.abs(amp_new - amp_old)))
        r_max = float(np.max(np.abs(err_vec))) if err_vec.size else 0.0

        _t_diis0 = _time.perf_counter()
        if cycle >= diis_start_cycle and err_vec.size > 0:
            amp_new = mydiis.update(amp_new, err_vec)
        _vec_to_amp(amp_new, T1_flat_arr, T2_flat_arr)
        _t_diis = _time.perf_counter() - _t_diis0

        # Sync T1_flat / T2_flat back to t1_pno / t2_pno_all (for next
        # cycle's t1_cache build + post-CCSD callers).
        _t_sync0 = _time.perf_counter()
        for ii in range(nocc):
            n_ii = int(npno[ii])
            if n_ii == 0:
                continue
            t1_pno[ii] = T1_flat_arr[pno_offsets[ii]:pno_offsets[ii] + n_ii].copy()
        for p, key in enumerate(keys_reorder):
            if key not in t2_pno_all:
                continue
            n_p = int(npno[p])
            if n_p == 0:
                continue
            t2_pno_all[key] = T2_flat_arr[
                t2_offsets[p]:t2_offsets[p + 1]].reshape(n_p, n_p).copy()
        _t_sync = _time.perf_counter() - _t_sync0

        e_cyc = float(out.energy)
        dE = abs(e_cyc - e_prev) if cycle > cycle_start else float('inf')
        e_prev = e_cyc

        _dt = _time.perf_counter() - _t_cyc_start
        print(f'  Cycle {cycle + 1:3d} [class]: dT={dT:.3e}  '
              f'E_corr={e_cyc:.10f}  dE={dE:.2e}  Rmax={r_max:.2e}  '
              f'[{_dt:.2f}s: t1cache={_t_t1cache:.2f} plans={_t_plans:.2f} '
              f'run_cyc={_t_run:.2f} diis={_t_diis:.2f} sync={_t_sync:.2f}]',
              flush=True)

        del native_own

        # ORCA-style dual stopping criterion (DLPNO_CONV_ORCA, default ON):
        # converge only when BOTH the energy change AND the max raw
        # residual are below tolerance — mirrors ORCA MDCI (TolE=1e-8 Eh,
        # TolErr=5e-7) and removes the ±2-4-cycle luck of the
        # dE-plateau-only rule.  DLPNO_CONV_ORCA=0 restores the legacy
        # dT/dE stopping.  Tolerances: DLPNO_TOL_E / DLPNO_TOL_ERR.
        if _conv_orca:
            if cycle > cycle_start and dE < _tol_e and r_max < _tol_err:
                print(f'  DLPNO-CCSD converged in {cycle + 1} cycles '
                      f'(dual, dE={dE:.2e} Rmax={r_max:.2e}, class).',
                      flush=True)
                if _omp_ctx is not None:
                    _omp_ctx.__exit__(None, None, None)
                return cycle, e_cyc
        else:
            if dT < this_tol:
                print(f'  DLPNO-CCSD converged in {cycle + 1} cycles '
                      f'(amplitude, class).', flush=True)
                if _omp_ctx is not None:
                    _omp_ctx.__exit__(None, None, None)
                return cycle, e_cyc
            if cycle > 5 and dE < this_tol:
                print(f'  DLPNO-CCSD converged in {cycle + 1} cycles '
                      f'(energy, dE={dE:.2e}, class).', flush=True)
                if _omp_ctx is not None:
                    _omp_ctx.__exit__(None, None, None)
                return cycle, e_cyc

    if _omp_ctx is not None:
        _omp_ctx.__exit__(None, None, None)
    return max_cycle - 1, e_prev


def _extract_g_term_plan(t2_pno_all, key_to_p, side='ik'):
    """Extract one side ('ik' or 'jk') of the G_term plan from
    PySCF's `compute_G_term_batched._plan_cache`.

    Translates t2_canon_off (PySCF FlatTensorStore offsets into
    `t2_pno_all._buffer`) to absolute offsets into our class's T2_flat.

    Returns (plan_struct, ownership, target_slot_list, n_ij_to_n_pairs_in_bucket).
    target_slot_list is a list of (n_ij, slot) pairs needed for tile
    scatter post-kernel.
    """
    from pyscf.cc.dlpno_tccsd.residual import (
        compute_G_term_batched, _get_or_build_g_term_batched_view,
        _build_g_term_plan)
    cache = getattr(compute_G_term_batched, '_plan_cache', None)
    if not cache:
        return None, [], [], {}
    plan = next(iter(cache.values()))
    bv = _get_or_build_g_term_batched_view(plan, t2_pno_all)
    side_bv = bv[side]
    N = side_bv['N']
    if N == 0:
        return None, [], [], {}

    # PySCF's t2_canon_off is offset into t2_pno_all._buffer (in PySCF
    # FTS canonical order).  Our class's T2_flat is in our diag-first
    # canonical order — they're DIFFERENT.  PySCF's _canon_to_idx maps
    # key → PySCF idx; our key_to_p maps key → our idx.  Since the BV
    # already encodes absolute byte offsets through PySCF's FTS, we
    # need to instead point directly at PySCF's t2_pno_all._buffer
    # (not our T2_flat).  This still uses class's kernel + plan; just
    # the T2 buffer comes from PySCF.  When we later sync T2 between
    # PySCF and class, this will be a no-op.
    pyscf_t2_buffer = t2_pno_all._buffer

    # Build per-iter t2_flat by gather (mimics _run_g_term_batched).
    # Cache the t2_flat buffer on side_bv (same size every cycle).
    from pyscf.cc.dlpno_tccsd._cd_gather_cy import gather_t2_with_transpose
    if '_t2_flat_buf' in side_bv:
        t2_flat = side_bv['_t2_flat_buf']
    else:
        t2_flat = np.empty(int(side_bv['t2_off'][-1]))
        side_bv['_t2_flat_buf'] = t2_flat
    gather_t2_with_transpose(
        N, side_bv['n_ik'],
        side_bv['t2_canon_off'], side_bv['t2_trans_arr'],
        side_bv['t2_off'], pyscf_t2_buffer, t2_flat,
        min(64, N),
    )

    plan_struct = PyGTermInputs()
    plan_struct.N           = int(N)
    plan_struct.n_ij_arr    = side_bv['n_ij'].ctypes.data
    plan_struct.n_ik_arr    = side_bv['n_ik'].ctypes.data
    plan_struct.S_off       = side_bv['S_off'].ctypes.data
    plan_struct.t2_off      = side_bv['t2_off'].ctypes.data
    plan_struct.tile_off    = side_bv['tile_off'].ctypes.data
    plan_struct.k_idx       = side_bv['k_idx'].ctypes.data
    plan_struct.scalar_lmo  = side_bv['scalar_lmo'].ctypes.data
    plan_struct.S_flat      = side_bv['S_flat'].ctypes.data
    plan_struct.t2_flat     = t2_flat.ctypes.data
    # G_tilde and G_stride are filled at run-time by the caller.
    plan_struct.max_n_ij    = int(side_bv['n_ij'].max(initial=1))
    plan_struct.max_n_ik    = int(side_bv['n_ik'].max(initial=1))

    # target_slot_list: for tile scatter back to flat_G_ij[n_ij].
    target_slots = list(plan['_g_batched_view'][side].get('target_slot', []))
    if not target_slots:
        # Reconstruct from buckets if not stored.
        target_slots = []
        for bucket in plan[f'{side}_buckets']:
            n_ij = bucket['n_ij']
            for slot in bucket['item_idx']:
                target_slots.append((n_ij, int(slot)))

    n_ij_to_n_pairs = {n_ij: len(pairs)
                        for n_ij, pairs in plan['pairs_by_n_ij'].items()}

    ownership = [
        side_bv['n_ij'], side_bv['n_ik'], side_bv['S_off'],
        side_bv['t2_off'], side_bv['tile_off'],
        side_bv['k_idx'], side_bv['scalar_lmo'],
        side_bv['S_flat'], t2_flat,
    ]
    return plan_struct, ownership, target_slots, n_ij_to_n_pairs


def _extract_g_tilde_plan(key_to_p):
    """Extract the G_tilde inner plan from PySCF's
    `build_G_tilde._batched_plan` cache.  Translates
    `triple_T2_pair_idx` from PySCF's canonical_pairs ordering to our
    class's pair ordering via `key_to_p`.

    Returns (plan_struct, ownership_arrays) — ownership keeps the numpy
    arrays alive past the C call.  Returns (None, []) if no plan cached
    or if the plan is empty.
    """
    from pyscf.cc.dlpno_tccsd.residual import build_G_tilde
    bp = getattr(build_G_tilde, '_batched_plan', None)
    if bp is None or bp.get('empty'):
        return None, []

    canonical_pairs = bp['canonical_pairs']
    # Translate triple_T2_pair_idx from PySCF idx → our class idx.
    pyscf_to_class = np.zeros(len(canonical_pairs), dtype=np.int64)
    for pyscf_idx, key in enumerate(canonical_pairs):
        cls_idx = key_to_p.get(key)
        if cls_idx is None:
            # Pair not in our class schema — should not happen if the
            # validator passed all of t2_pno_all.keys() to the packer.
            return None, []
        pyscf_to_class[pyscf_idx] = cls_idx

    triple_T2_pair_idx_class = pyscf_to_class[bp['triple_T2_pair_idx']]
    triple_T2_pair_idx_class = np.ascontiguousarray(
        triple_T2_pair_idx_class, dtype=np.int64)

    plan = PyGTildeInputs()
    plan.n_ij_slots         = int(bp['ij_i_arr'].shape[0])
    plan.triple_eff_offset  = bp['triple_eff_offset'].ctypes.data
    plan.triple_T2_pair_idx = triple_T2_pair_idx_class.ctypes.data
    plan.triple_n_lj        = bp['triple_n_lj'].ctypes.data
    plan.ij_triple_starts   = bp['ij_triple_starts'].ctypes.data
    plan.ij_i_arr           = bp['ij_i_arr'].ctypes.data
    plan.ij_j_arr           = bp['ij_j_arr'].ctypes.data
    plan.effective_flat     = bp['effective_flat'].ctypes.data

    ownership = [
        triple_T2_pair_idx_class,
        bp['triple_eff_offset'],
        bp['triple_n_lj'],
        bp['ij_triple_starts'],
        bp['ij_i_arr'], bp['ij_j_arr'],
        bp['effective_flat'],
    ]
    return plan, ownership


def _extract_per_kl_plan(t1_residual_func):
    """Extract a PyPerKlPlanInputs from the cached `_batched_plan` in
    `_compute_t1_residual`.  Caller must have already invoked the
    function once (so the cache is populated).  Returns (plan_struct,
    ownership) — ownership keeps the cached numpy arrays alive.
    """
    cache = getattr(t1_residual_func, '_per_kl_plan_cache', None)
    if cache is None or not cache:
        return None, []
    # Take the first (and typically only) cached plan.
    pkl_plan = next(iter(cache.values()))
    bp = pkl_plan.get('_batched_plan')
    if bp is None or bp.get('n_tasks', 0) == 0:
        return None, []

    # The cached plan's static buffers (K_iajb_static, K_bar_static,
    # S_consolidated) are contiguous numpy arrays kept alive in the
    # cache; t2_buffer / t1_cache_buffer are FlatTensorStore _buffer
    # arrays held by t2_pno_all and t1_cache (also alive through the
    # caller's scope).
    plan = PyPerKlPlanInputs()
    plan.n_tasks  = int(bp['n_tasks'])
    plan.M        = int(bp['M'])
    plan.max_n_kl = int(bp['max_n_kl'])
    plan.max_n_ki = int(bp['max_n_ki'])
    for name in ('n_kl_arr', 't2_swap_kl', 'inner_off',
                 'i_arr', 'n_pno_ii_arr',
                 'is_diag_kl_ii', 'has_S_ii_kl', 'has_A2',
                 'is_diag_kl_ki', 'n_ki_arr', 't2_swap_ki',
                 'S_ii_kl_off', 'S_kl_ki_off', 'S_ki_kl_off',
                 'T_n_l_ii_off', 'contrib_off',
                 't2_kl_canon_off', 'T_n_kl_off',
                 't2_ki_canon_off'):
        # Map cached-name to our struct-name (most match exactly).
        struct_name = name
        cache_name = name
        # Special: K_iajb_off / K_bar_off in cache → K_iajb_kl_off / K_bar_kl_off in struct.
        setattr(plan, struct_name, bp[cache_name].ctypes.data)
    plan.K_iajb_kl_off = bp['K_iajb_off'].ctypes.data
    plan.K_bar_kl_off  = bp['K_bar_off'].ctypes.data
    plan.K_iajb_buffer    = bp['K_iajb_static'].ctypes.data
    plan.K_bar_kl_static  = bp['K_bar_static'].ctypes.data
    plan.S_pno_buffer     = bp['S_consolidated'].ctypes.data
    # t2_buffer / t1_cache_buffer are filled in by caller from
    # t2_pno_all._buffer / t1_cache._buffer (FlatTensorStore).
    plan.t2_buffer       = 0
    plan.t1_cache_buffer = 0
    # Ownership references all the numpy arrays we depend on.
    own = list(bp.values())
    return plan, own


