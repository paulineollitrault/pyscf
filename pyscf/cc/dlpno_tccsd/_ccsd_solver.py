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
        ('data',    ctypes.c_void_p),     # const double *
        ('offsets', ctypes.c_void_p),     # const int64_t *
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
_libcc.DLPNOt4_kernel_batched.argtypes = (
    [ctypes.c_int]                            # N
    + [ctypes.c_void_p] * 14                  # 3 shape + 6 offsets + 5 buffers
    + [ctypes.c_void_p, ctypes.c_size_t] * 3  # 3 scratch
    + [ctypes.c_void_p, ctypes.c_double, ctypes.c_int])  # tiles_flat, scale, num_threads


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


def validate_t1_ints_real(cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
                           F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
                           S_pno_cache, verbose=True):
    """Real-data validation of phase_t1_ints against the existing
    Python `t1_ints(...)` wrapper.  Called from a hook in lccsd.py at
    cycle 0 (after t1_cache is built).  Compares per-pair output dicts.
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints
    from pyscf.cc.dlpno_tccsd.local_df import t1_ints as _ref_t1_ints

    # Reference path (existing Python wrapper).
    ref_dict = _ref_t1_ints(
        cc_ints, t1_pno, pno_spaces, S_pno_cache, list(keys_sorted), nocc,
        pair_lmo_idx=pair_lmo_idx, t1_cache=t1_cache, _pool=None)

    # Pack SolverInputs.
    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, list(keys_sorted))

    # Allocate outputs.
    keys_reorder = aux['keys_sorted']
    out, own, bufs = _allocate_t1_ints_outputs({
        'ij_pairs': keys_reorder,
        'pair_lmo_idx_offsets': np.cumsum(
            np.concatenate(([0],
                np.array([pl.size for pl in aux['pair_lmo_lists']], dtype=np.int64)))),
        'n_pno_per_pair': aux['n_pno_per_pair'],
        'naux_per_pair': np.array([
            aux['Qma_list'][p].shape[0] for p in range(len(keys_reorder))
        ], dtype=np.int32),
    })

    rc = _libcc.DLPNOcompute_lccsd_phase_t1_ints(
        ctypes.byref(inputs), ctypes.byref(out))
    if rc != 0:
        raise RuntimeError(f'phase_t1_ints rc={rc}')

    # Compare per-pair outputs.
    qa_off = bufs['i_Qa_t1'][1]
    qk_off = bufs['i_Qk_t1'][1]
    iQa_flat = bufs['i_Qa_t1'][0]
    jQa_flat = bufs['j_Qa_t1'][0]
    iQk_flat = bufs['i_Qk_t1'][0]
    jQk_flat = bufs['j_Qk_t1'][0]

    n_compared = 0
    n_skipped = 0
    max_iQa = max_jQa = max_iQk = max_jQk = 0.0
    failures = []

    for key in keys_sorted:
        if key not in ref_dict:
            n_skipped += 1
            continue
        if key not in key_to_p:
            n_skipped += 1
            continue
        p = key_to_p[key]
        ref = ref_dict[key]
        nlmo_p = aux['pair_lmo_lists'][p].size
        npno_p = int(aux['n_pno_per_pair'][p])
        n_local = aux['Qma_list'][p].shape[0]

        c_iQa = iQa_flat[qa_off[p]:qa_off[p + 1]].reshape(n_local, npno_p)
        c_jQa = jQa_flat[qa_off[p]:qa_off[p + 1]].reshape(n_local, npno_p)
        c_iQk = iQk_flat[qk_off[p]:qk_off[p + 1]].reshape(n_local, nlmo_p)
        c_jQk = jQk_flat[qk_off[p]:qk_off[p + 1]].reshape(n_local, nlmo_p)

        d_iQa = float(np.max(np.abs(c_iQa - ref['i_Qa_t1'])))
        d_jQa = float(np.max(np.abs(c_jQa - ref['j_Qa_t1'])))
        d_iQk = float(np.max(np.abs(c_iQk - ref['i_Qk_t1'])))
        d_jQk = float(np.max(np.abs(c_jQk - ref['j_Qk_t1'])))
        max_iQa = max(max_iQa, d_iQa)
        max_jQa = max(max_jQa, d_jQa)
        max_iQk = max(max_iQk, d_iQk)
        max_jQk = max(max_jQk, d_jQk)
        if max(d_iQa, d_jQa, d_iQk, d_jQk) > 1e-10:
            failures.append((p, key, d_iQa, d_jQa, d_iQk, d_jQk))
        n_compared += 1
    _ = n_compared  # silence unused-warning if any

    print(f'[CCSD MONO] phase_t1_ints REAL-DATA parity over {n_compared} '
          f'pairs (skipped {n_skipped}):', flush=True)
    print(f'  max |di_Qa_t1| = {max_iQa:.3e}', flush=True)
    print(f'  max |dj_Qa_t1| = {max_jQa:.3e}', flush=True)
    print(f'  max |di_Qk_t1| = {max_iQk:.3e}', flush=True)
    print(f'  max |dj_Qk_t1| = {max_jQk:.3e}', flush=True)
    if failures:
        print(f'  FAILURES on {len(failures)} pairs (showing first 5):', flush=True)
        for f in failures[:5]:
            print(f'    pair p={f[0]} key={f[1]}: '
                  f'iQa={f[2]:.3e} jQa={f[3]:.3e} '
                  f'iQk={f[4]:.3e} jQk={f[5]:.3e}', flush=True)
    del own, ownership
    return max(max_iQa, max_jQa, max_iQk, max_jQk)


def validate_b_tilde_real(cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
                           F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
                           S_pno_cache, t2_pno_all, verbose=True):
    """Real-data validation of phase_b_tilde.  Chains C++ phase_t1_ints
    output → phase_b_tilde, compares to existing PySCF compute_B_tilde.
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints
    from pyscf.cc.dlpno_tccsd.local_df import (
        t1_ints as _ref_t1_ints, compute_B_tilde as _ref_compute_B_tilde)

    # Reference path: existing dressed dict → compute_B_tilde per pair.
    ref_dressed = _ref_t1_ints(
        cc_ints, t1_pno, pno_spaces, S_pno_cache, list(keys_sorted), nocc,
        pair_lmo_idx=pair_lmo_idx, t1_cache=t1_cache, _pool=None)
    ref_B_per_key = {}
    for key in keys_sorted:
        if key not in cc_ints or cc_ints[key] is None:
            continue
        bt = _ref_compute_B_tilde(
            cc_ints, ref_dressed, t2_pno_all, t1_pno,
            pno_spaces, S_pno_cache, key, nocc,
            pair_lmo_idx=pair_lmo_idx, t1_cache=t1_cache)
        if bt is None:
            continue
        # compute_B_tilde returns (B_local, p_lmos_dense) — take B_local.
        ref_B_per_key[key] = bt[0]

    # Pack SolverInputs (with real T2_pno_all).
    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, list(keys_sorted),
        t2_pno_all=t2_pno_all)

    # Run phase_t1_ints to produce dressed Qa/Qk.
    keys_reorder = aux['keys_sorted']
    aux_for_alloc = {
        'ij_pairs': keys_reorder,
        'pair_lmo_idx_offsets': np.cumsum(
            np.concatenate(([0],
                np.array([pl.size for pl in aux['pair_lmo_lists']], dtype=np.int64)))),
        'n_pno_per_pair': aux['n_pno_per_pair'],
        'naux_per_pair': np.array([
            aux['Qma_list'][p].shape[0] for p in range(len(keys_reorder))
        ], dtype=np.int32),
    }
    t1_out, t1_own, t1_bufs = _allocate_t1_ints_outputs(aux_for_alloc)
    rc = _libcc.DLPNOcompute_lccsd_phase_t1_ints(
        ctypes.byref(inputs), ctypes.byref(t1_out))
    if rc != 0:
        raise RuntimeError(f'phase_t1_ints rc={rc}')

    # Wrap dressed outputs as BTildeInputs (FlatPairStore views).
    bt_in = PyBTildeInputs()
    for name in ('i_Qk_t1', 'j_Qk_t1'):
        wps = getattr(t1_out, name)
        fps = PyFlatPairStore()
        fps.data = wps.data
        fps.offsets = wps.offsets
        setattr(bt_in, name, fps)

    # Allocate B_tilde outputs (per pair (nlmo_p, nlmo_p)).
    bt_out, bt_own, (B_flat, B_offsets) = _allocate_b_tilde_outputs(aux_for_alloc)
    rc = _libcc.DLPNOcompute_lccsd_phase_b_tilde(
        ctypes.byref(inputs), ctypes.byref(bt_in), ctypes.byref(bt_out))
    if rc != 0:
        raise RuntimeError(f'phase_b_tilde rc={rc}')

    # Compare per-pair B_local against reference.
    n_compared = 0
    n_skipped = 0
    max_abs = 0.0
    failures = []
    for key in keys_sorted:
        if key not in ref_B_per_key:
            n_skipped += 1
            continue
        if key not in key_to_p:
            n_skipped += 1
            continue
        p = key_to_p[key]
        nlmo_p = aux['pair_lmo_lists'][p].size
        c_B = B_flat[B_offsets[p]:B_offsets[p + 1]].reshape(nlmo_p, nlmo_p)
        ref_B = ref_B_per_key[key]
        d = float(np.max(np.abs(c_B - ref_B)))
        max_abs = max(max_abs, d)
        if d > 1e-10:
            failures.append((p, key, d))
        n_compared += 1

    print(f'[CCSD MONO] phase_b_tilde REAL-DATA parity over {n_compared} '
          f'pairs (skipped {n_skipped}): max abs diff = {max_abs:.3e}', flush=True)
    if failures:
        print(f'  FAILURES on {len(failures)} pairs (first 5):', flush=True)
        for f in failures[:5]:
            print(f'    p={f[0]} key={f[1]}: max abs = {f[2]:.3e}',
                  flush=True)
    del t1_own, bt_own, ownership
    return max_abs


def validate_update_amps_and_energy_real(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
        S_pno_cache, t2_pno_all, K_pno_cache=None, verbose=True):
    """Real-data validation of update_amps_and_energy with R1 = R2 = 0.

    With zero residuals, T1/T2 mutation is a no-op (T -= 0/D = T).  We
    just compare the cycle correlation energy against a numpy reference
    formed from the same state.
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints

    # Pack SolverInputs.
    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, list(keys_sorted),
        t2_pno_all=t2_pno_all)

    # Snapshot T1/T2 BEFORE the call (to verify they don't change with
    # zero residual).
    pno_offsets = aux.get('pno_offsets')
    t2_offsets = aux.get('t2_offsets')
    # T1_flat / T2_flat live in `ownership` — find them by exact identity.
    # Easier: re-derive from inputs pointers via a temporary numpy view.
    # The packer keeps them in ownership; simplest is to read via the
    # SolverInputs pointers (caller-allocated arrays still in ownership).
    keys_reorder = aux['keys_sorted']

    # Build R1 = R2 = 0.
    R1_size = sum(int(aux['n_pno_per_pair'][i]) for i in range(nocc))
    R2_size = sum(int(aux['n_pno_per_pair'][p]) ** 2
                  for p in range(len(keys_reorder)))
    R1_zero = np.zeros(R1_size, dtype=np.float64)
    R2_zero = np.zeros(R2_size, dtype=np.float64)

    resid_struct = PyUpdateAmpsInputs()
    resid_struct.R1_flat = R1_zero.ctypes.data
    resid_struct.R2_flat = R2_zero.ctypes.data
    out_struct = PyUpdateAmpsOutputs()
    out_struct.energy = 0.0

    rc = _libcc.DLPNOcompute_lccsd_phase_update_amps_and_energy(
        ctypes.byref(inputs), ctypes.byref(resid_struct),
        ctypes.byref(out_struct))
    if rc != 0:
        raise RuntimeError(f'phase_update_amps_and_energy rc={rc}')
    e_class = float(out_struct.energy)

    # Reference: numpy reproduction of the same energy formula on the
    # original Python state.  E = Σ_i fov[i]·t1[i] +
    #                           Σ_p (1 or 2) K_iajb[p]:(2τ - τᵀ) with
    #                           τ = T2[p] + t1_i ⊗ t1_j (in pair PNO basis).
    e_ref_T1 = 0.0
    for i in range(nocc):
        if i not in t1_pno or t1_pno[i].size == 0:
            continue
        if i not in fov_pno or fov_pno[i].size == 0:
            continue
        e_ref_T1 += float(np.dot(fov_pno[i], t1_pno[i]))

    e_ref_T2 = 0.0
    for key in keys_sorted:
        if key not in t2_pno_all:
            continue
        T2_p = t2_pno_all[key]
        if T2_p.shape[0] == 0:
            continue
        i, j = key
        # K_iajb: prefer cc_ints['K_iajb']; else K_pno_cache; else
        # pno_spaces[key].get('K_pno').
        K_p = None
        if cc_ints.get(key) is not None and 'K_iajb' in cc_ints[key]:
            K_p = cc_ints[key]['K_iajb']
        elif K_pno_cache is not None and key in K_pno_cache:
            K_p = K_pno_cache[key]
        elif key in pno_spaces and 'K_pno' in pno_spaces[key]:
            K_p = pno_spaces[key]['K_pno']
        if K_p is None:
            continue
        # t1_i / t1_j projected into pair (i, j) PNO basis = t1_cache row.
        t1_i_pno = t1_cache[key][i]
        t1_j_pno = t1_cache[key][j]
        tau = T2_p + np.outer(t1_i_pno, t1_j_pno)
        Tt = 2.0 * tau - tau.T
        e_p = float(np.einsum('ab,ab->', K_p, Tt))
        e_ref_T2 += e_p if i == j else 2.0 * e_p

    e_ref = e_ref_T1 + e_ref_T2
    diff = abs(e_class - e_ref)
    print(f'[CCSD MONO] phase_update_amps_and_energy REAL-DATA energy: '
          f'class={e_class:.10f}  ref={e_ref:.10f}  '
          f'|d|={diff:.3e}', flush=True)
    del ownership
    return diff


def validate_t1_fock_fia_bar_real(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
        S_pno_cache, t2_pno_all, verbose=True):
    """Real-data validation of phase_t1_fock_fia_bar.  No existing Python
    counterpart for this exact tensor — reference is computed inline via
    numpy from the same Qma + T1_in_pair the packer feeds the C++ class.
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints

    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, list(keys_sorted),
        t2_pno_all=t2_pno_all)

    # Allocate Fia_bar outputs (per pair (nlmo_p, npno_p) doubles).
    keys_reorder = aux['keys_sorted']
    n_pairs = len(keys_reorder)
    pair_lmo_offs = np.cumsum(np.concatenate(([0],
        np.array([pl.size for pl in aux['pair_lmo_lists']],
                 dtype=np.int64))))
    npno = aux['n_pno_per_pair']
    sizes = np.empty(n_pairs, dtype=np.int64)
    for p in range(n_pairs):
        nlmo_p = int(pair_lmo_offs[p+1] - pair_lmo_offs[p])
        sizes[p] = nlmo_p * int(npno[p])
    offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    flat = np.full(int(offsets[-1]), np.nan, dtype=np.float64)

    out_struct = PyFiaBarOutputs()
    wps = PyWritablePairStore()
    wps.data = flat.ctypes.data
    wps.offsets = offsets.ctypes.data
    out_struct.Fia_bar = wps

    rc = _libcc.DLPNOcompute_lccsd_phase_t1_fock_fia_bar(
        ctypes.byref(inputs), ctypes.byref(out_struct))
    if rc != 0:
        raise RuntimeError(f'phase_t1_fock_fia_bar rc={rc}')

    # Reference: per canonical pair, compute Fia_bar with numpy.
    n_compared = 0
    n_skipped = 0
    max_abs = 0.0
    failures = []
    for p in range(n_pairs):
        npno_p = int(aux['n_pno_per_pair'][p])
        nlmo_p = int(aux['pair_lmo_lists'][p].size)
        if npno_p == 0 or nlmo_p == 0:
            n_skipped += 1
            continue
        Qma = aux['Qma_list'][p]                              # (n_local, nlmo_p, npno_p)
        T1l = aux['T1_in_pair_list'][p]                       # (nlmo_p, npno_p)
        n_local = Qma.shape[0]
        gamma = Qma.reshape(n_local, -1) @ T1l.ravel()        # (n_local,)
        Z = T1l @ Qma.transpose(0, 2, 1)                      # (n_local, nlmo_p, nlmo_p)
        Fia_bar_ref = 2.0 * np.tensordot(gamma, Qma, axes=(0, 0))
        Fia_bar_ref -= np.tensordot(Z, Qma, axes=((0, 1), (0, 1)))

        c_block = flat[offsets[p]:offsets[p + 1]].reshape(nlmo_p, npno_p)
        d = float(np.max(np.abs(c_block - Fia_bar_ref)))
        max_abs = max(max_abs, d)
        if d > 1e-10:
            failures.append((p, d))
        n_compared += 1

    print(f'[CCSD MONO] phase_t1_fock_fia_bar REAL-DATA parity over {n_compared} '
          f'pairs (skipped {n_skipped}): max abs diff = {max_abs:.3e}',
          flush=True)
    if failures:
        print(f'  FAILURES on {len(failures)} pairs (first 5):', flush=True)
        for f in failures[:5]:
            print(f'    p={f[0]}: max abs = {f[1]:.3e}', flush=True)
    del ownership
    return max_abs


def validate_t1_fock_real(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
        S_pno_cache, t2_pno_all, foo_total, verbose=True):
    """Real-data validation of phase_t1_fock (Fab + d_ij/d_ji).  Compares
    per-pair Fab against the existing Python `t1_fock(...)` wrapper.
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints
    from pyscf.cc.dlpno_tccsd.local_df import t1_fock as _ref_t1_fock

    # Reference: existing Python t1_fock returns (Fkj, Fab_all, foo_t1, Fij_bar).
    Fkj_ref, Fab_ref, foo_t1_ref, Fij_bar_ref = _ref_t1_fock(
        cc_ints, None, t1_pno, fov_pno, pno_spaces, S_pno_cache,
        F_lmo, eps_lmo, foo_total, list(keys_sorted), nocc,
        pair_lmo_idx=pair_lmo_idx, t1_cache=t1_cache, _pool=None)

    # Pack SolverInputs.
    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, list(keys_sorted),
        t2_pno_all=t2_pno_all)

    keys_reorder = aux['keys_sorted']
    n_pairs = len(keys_reorder)
    npno = aux['n_pno_per_pair']

    # Allocate Fab + d outputs.
    sizes = (npno.astype(np.int64) ** 2)
    Fab_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    Fab_offsets[1:] = np.cumsum(sizes)
    Fab_flat = np.full(int(Fab_offsets[-1]), np.nan, dtype=np.float64)
    d_flat   = np.zeros(n_pairs * 2, dtype=np.float64)

    out = PyT1FockOutputs()
    wps = PyWritablePairStore()
    wps.data    = Fab_flat.ctypes.data
    wps.offsets = Fab_offsets.ctypes.data
    out.Fab    = wps
    out.d_flat = d_flat.ctypes.data

    rc = _libcc.DLPNOcompute_lccsd_phase_t1_fock(
        ctypes.byref(inputs), ctypes.byref(out))
    if rc != 0:
        raise RuntimeError(f'phase_t1_fock rc={rc}')

    # Compare per-pair Fab against reference.
    n_compared = 0
    n_skipped = 0
    max_Fab = 0.0
    failures = []
    for key in keys_sorted:
        if key not in Fab_ref:
            n_skipped += 1
            continue
        if key not in key_to_p:
            n_skipped += 1
            continue
        p = key_to_p[key]
        npno_p = int(npno[p])
        c_Fab = Fab_flat[Fab_offsets[p]:Fab_offsets[p + 1]].reshape(npno_p, npno_p)
        d = float(np.max(np.abs(c_Fab - Fab_ref[key])))
        max_Fab = max(max_Fab, d)
        if d > 1e-10:
            failures.append((p, key, d))
        n_compared += 1

    print(f'[CCSD MONO] phase_t1_fock REAL-DATA Fab parity over {n_compared} '
          f'pairs (skipped {n_skipped}): max abs diff = {max_Fab:.3e}',
          flush=True)
    if failures:
        print(f'  FAILURES on {len(failures)} pairs (first 5):', flush=True)
        for f in failures[:5]:
            print(f'    p={f[0]} key={f[1]}: {f[2]:.3e}', flush=True)
    del ownership
    return max_Fab


def validate_t1_fock_finalize_real(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
        S_pno_cache, t2_pno_all, foo_total, verbose=True):
    """Real-data validation of phase_t1_fock_finalize: chains 2d → 2h
    and compares Fkj / Fij_bar / foo_t1 against existing Python t1_fock.
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints
    from pyscf.cc.dlpno_tccsd.local_df import t1_fock as _ref_t1_fock

    Fkj_ref, Fab_ref, foo_t1_ref, Fij_bar_ref = _ref_t1_fock(
        cc_ints, None, t1_pno, fov_pno, pno_spaces, S_pno_cache,
        F_lmo, eps_lmo, foo_total, list(keys_sorted), nocc,
        pair_lmo_idx=pair_lmo_idx, t1_cache=t1_cache, _pool=None)

    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, list(keys_sorted),
        t2_pno_all=t2_pno_all)

    keys_reorder = aux['keys_sorted']
    n_pairs = len(keys_reorder)
    npno = aux['n_pno_per_pair']

    # Run 2d to produce d_flat.
    sizes = (npno.astype(np.int64) ** 2)
    Fab_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    Fab_offsets[1:] = np.cumsum(sizes)
    Fab_flat = np.full(int(Fab_offsets[-1]), np.nan, dtype=np.float64)
    d_flat   = np.zeros(n_pairs * 2, dtype=np.float64)
    out_d = PyT1FockOutputs()
    wps = PyWritablePairStore()
    wps.data    = Fab_flat.ctypes.data
    wps.offsets = Fab_offsets.ctypes.data
    out_d.Fab    = wps
    out_d.d_flat = d_flat.ctypes.data
    rc = _libcc.DLPNOcompute_lccsd_phase_t1_fock(
        ctypes.byref(inputs), ctypes.byref(out_d))
    if rc != 0:
        raise RuntimeError(f'phase_t1_fock rc={rc}')

    # Run 2h to produce Fkj / Fij_bar / foo_t1.
    Fkj_class       = np.zeros((nocc, nocc), dtype=np.float64)
    Fij_bar_class   = np.zeros((nocc, nocc), dtype=np.float64)
    foo_t1_class    = np.zeros((nocc, nocc), dtype=np.float64)
    extra_in = PyT1FockExtraInputs()
    extra_in.d_flat = d_flat.ctypes.data
    extra_out = PyT1FockExtraOutputs()
    extra_out.Fkj              = Fkj_class.ctypes.data
    extra_out.Fij_bar_snapshot = Fij_bar_class.ctypes.data
    extra_out.foo_t1           = foo_t1_class.ctypes.data
    rc = _libcc.DLPNOcompute_lccsd_phase_t1_fock_finalize(
        ctypes.byref(inputs), ctypes.byref(extra_in),
        ctypes.byref(extra_out))
    if rc != 0:
        raise RuntimeError(f'phase_t1_fock_finalize rc={rc}')

    d_Fkj  = float(np.max(np.abs(Fkj_class       - Fkj_ref)))
    d_Fij  = float(np.max(np.abs(Fij_bar_class   - Fij_bar_ref)))
    d_foo  = float(np.max(np.abs(foo_t1_class    - foo_t1_ref)))
    print(f'[CCSD MONO] phase_t1_fock_finalize REAL-DATA: '
          f'|dFkj|={d_Fkj:.3e}  |dFij|={d_Fij:.3e}  |dfoo|={d_foo:.3e}',
          flush=True)
    del ownership
    return max(d_Fkj, d_Fij, d_foo)


def validate_d_tilde_ph1_real(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
        S_pno_cache, t2_pno_all, ovL_pno_cache=None, ooL_3idx=None,
        with_df=None, S_pao_full=None, s1e=None, K_coul_cache=None,
        verbose=True):
    """Real-data validation of phase_d_tilde_ph1 (ordered-pair).

    Reference: existing Python build_D_tilde_batched (Phase 1 portion).
    The reference returns a per-ordered-pair dict keyed by (i, k) → (n_pno_ki, n_pno_ki) D tile.
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints
    from pyscf.cc.dlpno_tccsd.residual import build_D_tilde_batched

    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, list(keys_sorted),
        t2_pno_all=t2_pno_all)

    # Reference path.  build_D_tilde_batched needs a list of ordered pairs;
    # the function builds (a, b) and (b, a) internally as `all_pairs`.
    ref = build_D_tilde_batched(
        t1_pno, t2_pno_all, pno_spaces, nocc,
        ovL_pno_cache, ooL_3idx, S_pno_cache, with_df,
        cc_ints=cc_ints,
        pair_lmo_idx=pair_lmo_idx, _pool=None,
        S_pao_full=S_pao_full, s1e=s1e, t1_cache=t1_cache,
        omp_threads=1)

    # Allocate D_tilde outputs (per ordered pair, (npno_p, npno_p)).
    n_ord = inputs.n_ordered_pairs
    keys_reorder = aux['keys_sorted']
    npno = aux['n_pno_per_pair']
    o_i = np.frombuffer(
        (ctypes.c_int * n_ord).from_address(inputs.ordered_pair_i_idx),
        dtype=np.int32, count=n_ord)
    o_k = np.frombuffer(
        (ctypes.c_int * n_ord).from_address(inputs.ordered_pair_k_idx),
        dtype=np.int32, count=n_ord)

    sizes = np.empty(n_ord, dtype=np.int64)
    for o in range(n_ord):
        i_, k_ = int(o_i[o]), int(o_k[o])
        p_canon = key_to_p[(min(i_, k_), max(i_, k_))]
        sizes[o] = int(npno[p_canon]) ** 2
    offsets = np.zeros(n_ord + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(sizes)
    D_flat = np.full(int(offsets[-1]), np.nan, dtype=np.float64)

    out = PyDTildeOutputs()
    wps = PyWritablePairStore()
    wps.data = D_flat.ctypes.data
    wps.offsets = offsets.ctypes.data
    out.D_tilde = wps

    rc = _libcc.DLPNOcompute_lccsd_phase_d_tilde_ph1(
        ctypes.byref(inputs), ctypes.byref(out))
    if rc != 0:
        raise RuntimeError(f'phase_d_tilde_ph1 rc={rc}')

    # Compare per-ordered-pair: ref[(i, k)] vs our flat output.
    n_compared = 0
    n_skipped = 0
    max_abs = 0.0
    for o in range(n_ord):
        i_, k_ = int(o_i[o]), int(o_k[o])
        ref_key = (i_, k_)
        if ref_key not in ref:
            n_skipped += 1
            continue
        p_canon = key_to_p[(min(i_, k_), max(i_, k_))]
        npno_p = int(npno[p_canon])
        c_D = D_flat[offsets[o]:offsets[o + 1]].reshape(npno_p, npno_p)
        d = float(np.max(np.abs(c_D - ref[ref_key])))
        max_abs = max(max_abs, d)
        n_compared += 1

    print(f'[CCSD MONO] phase_d_tilde_ph1 REAL-DATA parity over {n_compared} '
          f'ordered pairs (skipped {n_skipped}): max abs diff = {max_abs:.3e}',
          flush=True)
    del ownership
    return max_abs


def validate_t1_residual_AC_init_real(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
        S_pno_cache, t2_pno_all, verbose=True):
    """Real-data validation of phase_t1_residual_AC_init.  Reference is
    a numpy reproduction of the same math (init + A + C per ordered pair),
    using the same packed-from-real-data state.  Validates the cross-
    canonical S_pno_cache packer + the R1 init/A/C math on real data.
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints

    # Pack with S_pno_cache.
    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, list(keys_sorted),
        t2_pno_all=t2_pno_all, S_pno_cache=S_pno_cache)

    keys_reorder = aux['keys_sorted']
    n_pairs = len(keys_reorder)
    npno = aux['n_pno_per_pair']

    # Run 2l-b to get Fia_bar per pair.
    pair_lmo_offs = np.cumsum(np.concatenate(([0],
        np.array([pl.size for pl in aux['pair_lmo_lists']], dtype=np.int64))))
    sizes = np.empty(n_pairs, dtype=np.int64)
    for p in range(n_pairs):
        nlmo_p = int(pair_lmo_offs[p+1] - pair_lmo_offs[p])
        sizes[p] = nlmo_p * int(npno[p])
    fia_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    fia_offsets[1:] = np.cumsum(sizes)
    fia_flat = np.full(int(fia_offsets[-1]), np.nan, dtype=np.float64)
    fia_out = PyFiaBarOutputs()
    wps = PyWritablePairStore()
    wps.data    = fia_flat.ctypes.data
    wps.offsets = fia_offsets.ctypes.data
    fia_out.Fia_bar = wps
    rc = _libcc.DLPNOcompute_lccsd_phase_t1_fock_fia_bar(
        ctypes.byref(inputs), ctypes.byref(fia_out))
    if rc != 0:
        raise RuntimeError(f'phase_t1_fock_fia_bar rc={rc}')

    # Wrap as R1AcInputs (FlatPairStore over the writable buffer).
    r1ac_in = PyR1AcInputs()
    fps = PyFlatPairStore()
    fps.data    = fia_out.Fia_bar.data
    fps.offsets = fia_out.Fia_bar.offsets
    r1ac_in.Fia_bar = fps
    r1ac_in.do_init = 1

    # Run 2l-c.
    R1_size = sum(int(npno[i]) for i in range(nocc))
    R1_class = np.full(R1_size, np.nan, dtype=np.float64)
    out_struct = PyR1AcOutputs()
    out_struct.R1_flat = R1_class.ctypes.data
    rc = _libcc.DLPNOcompute_lccsd_phase_t1_residual_AC_init(
        ctypes.byref(inputs), ctypes.byref(r1ac_in),
        ctypes.byref(out_struct))
    if rc != 0:
        raise RuntimeError(f'phase_t1_residual_AC_init rc={rc}')

    # Reference: numpy reproduction of init + A + C using the SAME packed
    # state.  We reuse aux fields directly.
    pno_offsets = aux['pno_offsets']
    t2_offsets = aux['t2_offsets']
    T2_flat = aux['T2_flat']
    fps_arr = aux  # has Qma_list, T1_in_pair_list, pair_lmo_lists, etc.

    # First compute per-pair Fia_bar reference (numpy).
    Fia_bar_ref_list = []
    for p in range(n_pairs):
        nlmo_p = int(aux['pair_lmo_lists'][p].size)
        npno_p = int(npno[p])
        if nlmo_p == 0 or npno_p == 0:
            Fia_bar_ref_list.append(np.zeros((nlmo_p, npno_p)))
            continue
        Qma = aux['Qma_list'][p]
        T1l = aux['T1_in_pair_list'][p]
        n_local = Qma.shape[0]
        gamma = Qma.reshape(n_local, -1) @ T1l.ravel()
        Z = T1l @ Qma.transpose(0, 2, 1)
        F = 2.0 * np.tensordot(gamma, Qma, axes=(0, 0))
        F -= np.tensordot(Z, Qma, axes=((0, 1), (0, 1)))
        Fia_bar_ref_list.append(F)

    # Now build R1 reference.
    R1_ref = np.zeros(R1_size, dtype=np.float64)

    # Init: R1[i] += Fia_bar[(i,i)][i_in_p, :]  (diag-first: p_ii = i)
    for i in range(nocc):
        if i >= n_pairs:
            continue
        npno_ii = int(npno[i])
        if npno_ii == 0:
            continue
        lmo_list_ii = aux['pair_lmo_lists'][i]
        i_in_p_arr = np.where(lmo_list_ii == i)[0]
        if i_in_p_arr.size == 0:
            continue
        i_in_p = int(i_in_p_arr[0])
        Fai = Fia_bar_ref_list[i][i_in_p]
        R1_ref[pno_offsets[i]:pno_offsets[i] + npno_ii] += Fai

    # A + C per ordered pair.  Read S_pno from the packed flat buffer
    # we stored in ownership (so we test the SAME data the C++ reads).
    n_ord = inputs.n_ordered_pairs
    o_i = np.frombuffer(
        (ctypes.c_int * n_ord).from_address(inputs.ordered_pair_i_idx),
        dtype=np.int32, count=n_ord)
    o_k = np.frombuffer(
        (ctypes.c_int * n_ord).from_address(inputs.ordered_pair_k_idx),
        dtype=np.int32, count=n_ord)
    # S_pno_data + S_pno_offsets pointers from inputs.
    S_pno_offsets_arr = None
    S_pno_data_arr = None
    for arr in ownership:
        if (isinstance(arr, np.ndarray)
                and arr.dtype == np.int64
                and arr.size == n_pairs * n_pairs + 1
                and arr.ctypes.data == inputs.S_pno_offsets):
            S_pno_offsets_arr = arr
        elif (isinstance(arr, np.ndarray)
                and arr.dtype == np.float64
                and arr.ctypes.data == inputs.S_pno_data):
            S_pno_data_arr = arr

    i_j_to_ij_2d = np.frombuffer(
        (ctypes.c_int * (nocc * nocc)).from_address(inputs.i_j_to_ij),
        dtype=np.int32, count=nocc * nocc).reshape(nocc, nocc).copy()

    for o_idx in range(n_ord):
        a_ord = int(o_i[o_idx])
        b_ord = int(o_k[o_idx])
        p_canon = int(i_j_to_ij_2d[a_ord, b_ord])
        if p_canon < 0:
            continue
        npno_p = int(npno[p_canon])
        p_ii = int(i_j_to_ij_2d[a_ord, a_ord])
        if p_ii < 0:
            continue
        npno_ii = int(npno[p_ii])
        if npno_p == 0 or npno_ii == 0:
            continue

        can_first = keys_reorder[p_canon][0]

        s_idx = p_canon * n_pairs + p_ii
        s_off = int(S_pno_offsets_arr[s_idx])
        s_size = int(S_pno_offsets_arr[s_idx + 1] - s_off)
        if s_size != npno_p * npno_ii:
            continue
        S_p = S_pno_data_arr[s_off:s_off + s_size].reshape(npno_p, npno_ii)

        T2_p = T2_flat[t2_offsets[p_canon]:t2_offsets[p_canon+1]] \
            .reshape(npno_p, npno_p)

        # A term.
        # Get K_tilde_chem variant from cc_ints (we didn't expose it via
        # aux explicitly).  For ordered (b_ord, a_ord), pick "_i" if
        # can_first == b_ord else "_j".
        ci = cc_ints.get(keys_reorder[p_canon])
        if ci is None:
            continue
        if can_first == b_ord:
            K_chem_ki = np.ascontiguousarray(ci['K_tilde_chem_i'])
        else:
            K_chem_ki = np.ascontiguousarray(ci['K_tilde_chem_j'])

        swap_ki = (can_first != b_ord)
        T2_ki = T2_p.T if swap_ki else T2_p
        Tt_ki = 2.0 * T2_ki - T2_ki.T

        K_R = K_chem_ki.reshape(npno_p * npno_p, npno_p)
        Y = K_R.T @ Tt_ki.ravel()
        A_contrib = S_p.T @ Y
        R1_ref[pno_offsets[a_ord]:pno_offsets[a_ord] + npno_ii] += A_contrib

        # C term.
        swap_ik = (can_first != a_ord)
        T2_ik = T2_p.T if swap_ik else T2_p
        Tt_ik = 2.0 * T2_ik - T2_ik.T

        lmo_list_p = aux['pair_lmo_lists'][p_canon]
        k_in_p_arr = np.where(lmo_list_p == b_ord)[0]
        if k_in_p_arr.size == 0:
            continue
        k_in_p = int(k_in_p_arr[0])
        Fkc_ki = Fia_bar_ref_list[p_canon][k_in_p]
        C_contrib = S_p.T @ (Tt_ik @ Fkc_ki)
        R1_ref[pno_offsets[a_ord]:pno_offsets[a_ord] + npno_ii] += C_contrib

    max_abs = float(np.max(np.abs(R1_class - R1_ref)))
    print(f'[CCSD MONO] phase_t1_residual_AC_init REAL-DATA parity over '
          f'{n_ord} ordered pairs: max abs diff = {max_abs:.3e}',
          flush=True)
    del ownership
    return max_abs


def run_one_cycle_via_class(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
        S_pno_cache, t2_pno_all):
    """c-collapse-1 driver: pack SolverInputs, call DLPNOcompute_lccsd_run_one_cycle
    (currently a no-op skeleton), return (R1_flat, R2_flat, energy).

    Walking-skeleton scope: plan-cached phase inputs are all NULL, so the
    C++ method zeros R1/R2 and returns 0 energy.  Subsequent pushes fill
    in plans + phase calls inside C++.
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints

    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, list(keys_sorted),
        t2_pno_all=t2_pno_all, S_pno_cache=S_pno_cache)

    keys_reorder = aux['keys_sorted']
    n_pairs = len(keys_reorder)
    npno = aux['n_pno_per_pair']

    # Allocate output buffers.
    R1_size = sum(int(npno[i]) for i in range(nocc))
    R2_size = sum(int(npno[p]) ** 2 for p in range(n_pairs))
    R1_flat = np.zeros(R1_size, dtype=np.float64)
    R2_flat = np.zeros(R2_size, dtype=np.float64)

    # Walking-skeleton: all plans null.
    plans = PyRunCycleInputs()
    plans.g_tilde_plan = None
    plans.per_kl_plan  = None
    plans.be_plan      = None
    plans.c_term_plan  = None
    plans.d_term_plan  = None
    plans.g_term_plan  = None
    plans.t3_plan      = None
    plans.t4_plan      = None

    out = PyRunCycleOutputs()
    out.R1_flat = R1_flat.ctypes.data
    out.R2_flat = R2_flat.ctypes.data
    out.energy  = 0.0

    rc = _libcc.DLPNOcompute_lccsd_run_one_cycle(
        ctypes.byref(inputs), ctypes.byref(plans), ctypes.byref(out))
    if rc != 0:
        raise RuntimeError(f'run_one_cycle rc={rc}')

    energy = float(out.energy)
    del ownership
    return R1_flat, R2_flat, energy


def validate_run_one_cycle_real(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
        S_pno_cache, t2_pno_all, verbose=True):
    """Real-data validation of run_one_cycle's orchestration.

    Strategy: run_one_cycle calls a known sequence of phases inside C++.
    Reference: call the SAME phases via individual ctypes wrappers from
    Python, assemble R1 / R2 the same way, run update_amps_and_energy
    independently.  Bit-for-bit comparison validates the orchestration
    glue (since each phase is already validated separately).

    Skeleton-mode caveat: only K+A contribute to R2 (plans absent for
    BE/CD/G_term/t3/t4); only init+A+C contribute to R1 (per_kl plan
    absent).  Reference path mirrors this subset exactly.
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints

    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, list(keys_sorted),
        t2_pno_all=t2_pno_all, S_pno_cache=S_pno_cache)

    keys_reorder = aux['keys_sorted']
    n_pairs = len(keys_reorder)
    npno = aux['n_pno_per_pair']

    # Snapshot T1 / T2 before mutation (run_one_cycle mutates in place).
    T1_flat_arr = aux['T1_flat']
    T2_flat_arr = aux['T2_flat']
    T1_snapshot = T1_flat_arr.copy()
    T2_snapshot = T2_flat_arr.copy()

    R1_size = sum(int(npno[i]) for i in range(nocc))
    R2_size = sum(int(npno[p]) ** 2 for p in range(n_pairs))

    # ---- Path A: run_one_cycle ----
    R1_class = np.zeros(R1_size, dtype=np.float64)
    R2_class = np.zeros(R2_size, dtype=np.float64)
    plans = PyRunCycleInputs()
    for fname in ('g_tilde_plan', 'per_kl_plan', 'be_plan', 'c_term_plan',
                  'd_term_plan', 'g_term_plan', 't3_plan', 't4_plan'):
        setattr(plans, fname, None)
    out_struct = PyRunCycleOutputs()
    out_struct.R1_flat = R1_class.ctypes.data
    out_struct.R2_flat = R2_class.ctypes.data
    out_struct.energy  = 0.0
    rc = _libcc.DLPNOcompute_lccsd_run_one_cycle(
        ctypes.byref(inputs), ctypes.byref(plans), ctypes.byref(out_struct))
    if rc != 0:
        raise RuntimeError(f'run_one_cycle rc={rc}')
    e_class = float(out_struct.energy)

    # Restore T1/T2 to snapshot before path B.
    T1_flat_arr[:] = T1_snapshot
    T2_flat_arr[:] = T2_snapshot

    # ---- Path B: same phases via individual ctypes calls ----
    pair_lmo_offs = np.cumsum(np.concatenate(([0],
        np.array([pl.size for pl in aux['pair_lmo_lists']],
                 dtype=np.int64))))
    aux_for_alloc = {
        'ij_pairs': keys_reorder,
        'pair_lmo_idx_offsets': pair_lmo_offs,
        'n_pno_per_pair': npno,
        'naux_per_pair': np.array([
            aux['Qma_list'][p].shape[0] for p in range(n_pairs)
        ], dtype=np.int32),
    }

    # Phase 1: t1_ints.
    t1_out, t1_own, t1_bufs = _allocate_t1_ints_outputs(aux_for_alloc)
    rc = _libcc.DLPNOcompute_lccsd_phase_t1_ints(
        ctypes.byref(inputs), ctypes.byref(t1_out))
    assert rc == 0

    # Phase 2: t1_fock (Fab + d).
    t2_offsets = aux['t2_offsets']
    Fab_flat = np.zeros(int(t2_offsets[-1]), dtype=np.float64)
    d_flat   = np.zeros(n_pairs * 2, dtype=np.float64)
    t1f_out = PyT1FockOutputs()
    wps = PyWritablePairStore()
    wps.data    = Fab_flat.ctypes.data
    wps.offsets = t2_offsets.ctypes.data
    t1f_out.Fab    = wps
    t1f_out.d_flat = d_flat.ctypes.data
    rc = _libcc.DLPNOcompute_lccsd_phase_t1_fock(
        ctypes.byref(inputs), ctypes.byref(t1f_out))
    assert rc == 0

    # Phase 3: Fia_bar.
    fia_sizes = np.array([
        int(pair_lmo_offs[p+1] - pair_lmo_offs[p]) * int(npno[p])
        for p in range(n_pairs)], dtype=np.int64)
    fia_offsets = np.zeros(n_pairs + 1, dtype=np.int64)
    fia_offsets[1:] = np.cumsum(fia_sizes)
    fia_flat = np.zeros(int(fia_offsets[-1]), dtype=np.float64)
    fia_out = PyFiaBarOutputs()
    fia_wps = PyWritablePairStore()
    fia_wps.data    = fia_flat.ctypes.data
    fia_wps.offsets = fia_offsets.ctypes.data
    fia_out.Fia_bar = fia_wps
    rc = _libcc.DLPNOcompute_lccsd_phase_t1_fock_fia_bar(
        ctypes.byref(inputs), ctypes.byref(fia_out))
    assert rc == 0

    # Phase 4: K + ladder (consumes dressed Qa).
    kl_in = PyKLadderInputs()
    for name in ('i_Qa_t1', 'j_Qa_t1'):
        wps = getattr(t1_out, name)
        fps = PyFlatPairStore()
        fps.data    = wps.data
        fps.offsets = wps.offsets
        setattr(kl_in, name, fps)
    K_flat = np.zeros(int(t2_offsets[-1]), dtype=np.float64)
    A_flat = np.zeros(int(t2_offsets[-1]), dtype=np.float64)
    kl_out = PyKLadderOutputs()
    K_wps = PyWritablePairStore()
    K_wps.data = K_flat.ctypes.data; K_wps.offsets = t2_offsets.ctypes.data
    kl_out.K = K_wps
    A_wps = PyWritablePairStore()
    A_wps.data = A_flat.ctypes.data; A_wps.offsets = t2_offsets.ctypes.data
    kl_out.A = A_wps
    rc = _libcc.DLPNOcompute_lccsd_phase_k_ladder(
        ctypes.byref(inputs), ctypes.byref(kl_in), ctypes.byref(kl_out))
    assert rc == 0

    # Phase 5: R1 init + A + C (uses Fia_bar).
    r1ac_in = PyR1AcInputs()
    fps = PyFlatPairStore()
    fps.data    = fia_out.Fia_bar.data
    fps.offsets = fia_out.Fia_bar.offsets
    r1ac_in.Fia_bar = fps
    r1ac_in.do_init = 1
    R1_ref = np.zeros(R1_size, dtype=np.float64)
    r1ac_out = PyR1AcOutputs()
    r1ac_out.R1_flat = R1_ref.ctypes.data
    rc = _libcc.DLPNOcompute_lccsd_phase_t1_residual_AC_init(
        ctypes.byref(inputs), ctypes.byref(r1ac_in), ctypes.byref(r1ac_out))
    assert rc == 0

    # Phase 6: R2 = K + A per canonical pair.
    R2_ref = np.zeros(R2_size, dtype=np.float64)
    for p in range(n_pairs):
        npno_p = int(npno[p])
        if npno_p == 0:
            continue
        sl = slice(int(t2_offsets[p]), int(t2_offsets[p+1]))
        R2_ref[sl] = K_flat[sl] + A_flat[sl]

    # Phase 7: update_amps_and_energy.
    resid = PyUpdateAmpsInputs()
    resid.R1_flat = R1_ref.ctypes.data
    resid.R2_flat = R2_ref.ctypes.data
    upd_out = PyUpdateAmpsOutputs()
    upd_out.energy = 0.0
    rc = _libcc.DLPNOcompute_lccsd_phase_update_amps_and_energy(
        ctypes.byref(inputs), ctypes.byref(resid), ctypes.byref(upd_out))
    assert rc == 0
    e_ref = float(upd_out.energy)

    d_R1 = float(np.max(np.abs(R1_class - R1_ref)))
    d_R2 = float(np.max(np.abs(R2_class - R2_ref)))
    d_e  = float(abs(e_class - e_ref))
    print(f'[CCSD MONO] run_one_cycle orchestration parity: '
          f'|dR1|={d_R1:.3e}  |dR2|={d_R2:.3e}  |dE|={d_e:.3e}  '
          f'(class energy = {e_class:.6e})', flush=True)

    del t1_own, ownership
    return max(d_R1, d_R2, d_e)


def _build_Fij_bar_full(F_lmo, t2_pno_all, cc_ints, pno_spaces,
                         t1_cache, pair_lmo_idx, nocc):
    """Mirror PySCF _compute_t1_residual_psi4's Fij_bar dressing (lines
    681-709).  Builds the FULL T1-dressed F_oo (strong + weak pair
    contributions).  Returns a (nocc, nocc) numpy array.
    """
    Fij_bar = np.ascontiguousarray(F_lmo, dtype=np.float64).copy()
    for key_ij, _T2 in t2_pno_all.items():
        ci_ij = cc_ints.get(key_ij)
        if ci_ij is None:
            continue
        i0, j0 = key_ij
        if pno_spaces[key_ij]['C_pno'].shape[1] == 0:
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
    fkc_dress inner sum in `_compute_t1_residual_psi4` C term:
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
            if pno_spaces[key_im]['C_pno'].shape[1] == 0:
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


def validate_run_one_cycle_with_per_kl(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, keys_sorted,
        S_pno_cache, t2_pno_all, cc_ints_flat, pair_index, ovL_pno_cache,
        verbose=True):
    """Validate run_one_cycle with the per_kl plan extracted from
    `_compute_t1_residual_psi4`'s cache.  R1 should equal the Python
    wrapper's R1 (init + A + C + B + A2).  Skeleton mode for R2 (only
    K + A) — energy not compared (incomplete R2 → wrong T2 update).
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints
    from pyscf.cc.dlpno_tccsd.lccsd import _compute_t1_residual_psi4

    # ---- Reference: full Python R1 computation. ----
    # This populates _compute_t1_residual_psi4._per_kl_plan_cache.
    r1_pno_ref = _compute_t1_residual_psi4(
        t1_pno, t2_pno_all, pno_spaces, fov_pno, F_lmo, eps_lmo, nocc,
        S_pno_cache, cc_ints, ovL_pno_cache=ovL_pno_cache,
        pair_lmo_idx=pair_lmo_idx, t1_cache=t1_cache, _pool=None,
        cc_ints_flat=cc_ints_flat, pair_index=pair_index)

    # Extract plan from cache.
    plan_struct, plan_own = _extract_per_kl_plan(_compute_t1_residual_psi4)
    if plan_struct is None:
        print('[CCSD MONO] no per_kl plan cached; skipping', flush=True)
        return None
    # Wire t2_buffer / t1_cache_buffer to the FlatTensorStore _buffer arrays.
    plan_struct.t2_buffer       = t2_pno_all._buffer.ctypes.data
    plan_struct.t1_cache_buffer = t1_cache._buffer.ctypes.data

    # ---- Pack SolverInputs and call run_one_cycle. ----
    # Use ALL surviving pairs (strong + weak) so AC iteration spans every
    # partner k, not just strong ones.  PySCF's _compute_t1_residual_psi4
    # iterates over t2_pno_all (210 pairs) — we must match.
    _all_keys = sorted(t2_pno_all.keys())
    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, _all_keys,
        t2_pno_all=t2_pno_all, S_pno_cache=S_pno_cache)

    keys_reorder = aux['keys_sorted']
    n_pairs = len(keys_reorder)
    npno = aux['n_pno_per_pair']



    # Snapshot T1/T2 (run_one_cycle mutates).
    T1_snapshot = aux['T1_flat'].copy()
    T2_snapshot = aux['T2_flat'].copy()

    R1_size = sum(int(npno[i]) for i in range(nocc))
    R2_size = sum(int(npno[p]) ** 2 for p in range(n_pairs))
    R1_class = np.zeros(R1_size, dtype=np.float64)
    R2_class = np.zeros(R2_size, dtype=np.float64)

    plans = PyRunCycleInputs()
    for fname in ('g_tilde_plan', 'be_plan', 'c_term_plan',
                  'd_term_plan', 'g_term_plan', 't3_plan', 't4_plan'):
        setattr(plans, fname, None)
    plans.per_kl_plan = ctypes.pointer(plan_struct)

    # G_tilde plan extraction (the only plan currently extracted from
    # PySCF's caches; the remaining R2 plans — BE/CD/G_term/t3/t4 — are
    # still null, so R2 = K + ladder only).
    g_plan_struct, g_plan_own = _extract_g_tilde_plan(key_to_p)
    if g_plan_struct is not None:
        plans.g_tilde_plan = ctypes.pointer(g_plan_struct)

    out_struct = PyRunCycleOutputs()
    out_struct.R1_flat = R1_class.ctypes.data
    out_struct.R2_flat = R2_class.ctypes.data
    out_struct.energy  = 0.0
    rc = _libcc.DLPNOcompute_lccsd_run_one_cycle(
        ctypes.byref(inputs), ctypes.byref(plans), ctypes.byref(out_struct))
    if rc != 0:
        raise RuntimeError(f'run_one_cycle rc={rc}')

    # Restore T1/T2 (so subsequent validators see clean state).
    aux['T1_flat'][:] = T1_snapshot
    aux['T2_flat'][:] = T2_snapshot

    # ---- Compare R1: flatten r1_pno_ref to per-occupied layout. ----
    pno_offsets = aux['pno_offsets']
    R1_ref = np.zeros(R1_size, dtype=np.float64)
    for i in range(nocc):
        if i not in r1_pno_ref:
            continue
        npno_ii = int(npno[i])
        if npno_ii == 0:
            continue
        R1_ref[pno_offsets[i]:pno_offsets[i] + npno_ii] = r1_pno_ref[i]

    d_R1 = float(np.max(np.abs(R1_class - R1_ref)))
    print(f'[CCSD MONO] run_one_cycle WITH per_kl plan: '
          f'|dR1| = {d_R1:.3e} vs Python full R1 ref '
          f'(packed n_canon_pairs={n_pairs})',
          flush=True)

    del plan_own, ownership
    return d_R1


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
        compute_G_term_batched, compute_B_E_batched_v2,
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
    be_cache = getattr(compute_B_E_batched_v2, '_plan_cache', None)
    be_plan = next(iter(be_cache.values())) if be_cache else None
    be_plan_buckets_arr = be_unique_n_ij = be_flat_off = None
    be_pair_n_ij_idx_arr = be_pair_slot_arr = None
    be_n_buckets = 0
    be_n_unique = 0
    be_owned_buckets = []
    if be_plan is not None:
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
                # T_buf and beta arrays — allocated once, refilled in place.
                T_buf       = np.empty((N_b, n_kl, n_kl))
                beta_kl_arr = np.empty(N_b)
                beta_lk_arr = np.empty(N_b)
                S_c = np.ascontiguousarray(bucket['S'])
                K_c = np.ascontiguousarray(bucket['K'])
                same_c = np.ascontiguousarray(bucket['same']).astype(
                    np.uint8, copy=False)
                idx_c = np.ascontiguousarray(bucket['item_idx']).astype(
                    np.int64, copy=False)
                # Pre-resolve the t2_pno_all references for fast per-cycle
                # refill (saves dict lookups in the hot loop).
                kl_refs = [t2_pno_all[k] for k in bucket['kl_keys']]
                inv = {
                    'T_buf': T_buf, 'beta_kl': beta_kl_arr,
                    'beta_lk': beta_lk_arr,
                    'p_ij_arr': p_ij_arr, 'dense_k_arr': dense_k_arr,
                    'dense_l_arr': dense_l_arr,
                    'S_c': S_c, 'K_c': K_c, 'same_c': same_c, 'idx_c': idx_c,
                    'kl_refs': kl_refs,
                }
                bucket['_inv_cache'] = inv

            T_buf       = inv['T_buf']
            beta_kl_arr = inv['beta_kl']
            beta_lk_arr = inv['beta_lk']
            p_ij_arr    = inv['p_ij_arr']
            dense_k_arr = inv['dense_k_arr']
            dense_l_arr = inv['dense_l_arr']
            S_c         = inv['S_c']
            K_c         = inv['K_c']
            same_c      = inv['same_c']
            idx_c       = inv['idx_c']
            # Per-cycle: refill T_buf in place from current t2_pno_all values.
            kl_refs = inv['kl_refs']
            for n, src in enumerate(kl_refs):
                np.copyto(T_buf[n], src)
            T_c = T_buf  # alias; already contiguous (np.empty default)
            n_pairs_in_group = len(be_plan['pairs_by_n_ij'][n_ij])
            buckets_arr[b_idx].N = int(N_b)
            buckets_arr[b_idx].n_ij = int(n_ij)
            buckets_arr[b_idx].n_kl = int(n_kl)
            buckets_arr[b_idx].n_slots = int(n_pairs_in_group)
            buckets_arr[b_idx].S           = S_c.ctypes.data
            buckets_arr[b_idx].T           = T_c.ctypes.data
            buckets_arr[b_idx].K           = K_c.ctypes.data
            buckets_arr[b_idx].beta_kl     = beta_kl_arr.ctypes.data
            buckets_arr[b_idx].beta_lk     = beta_lk_arr.ctypes.data
            buckets_arr[b_idx].same        = same_c.ctypes.data
            buckets_arr[b_idx].idx         = idx_c.ctypes.data
            buckets_arr[b_idx].p_ij_arr    = p_ij_arr.ctypes.data
            buckets_arr[b_idx].dense_k_arr = dense_k_arr.ctypes.data
            buckets_arr[b_idx].dense_l_arr = dense_l_arr.ctypes.data
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
            n_ij = pno_spaces[key]['C_pno'].shape[1]
            if n_ij in unique_n_ij:
                be_pair_n_ij_idx_arr[p] = unique_n_ij.index(n_ij)
                be_pair_slot_arr[p] = slot
        own.extend([be_pair_n_ij_idx_arr, be_pair_slot_arr])

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
            # Gather ct_flat + t2_flat per cycle.
            # ct_flat scratch buffer is cycle-invariant in size; keep it in
            # bv so we don't reallocate every cycle. Likewise cache the
            # offsets/sizes as Python ints so the inner refill loop avoids
            # numpy.int64 box-unbox overhead.
            if 'c_ct_flat_buf' not in bv:
                bv['c_ct_flat_buf'] = np.zeros(int(bv['c_ct_off'][-1]))
                bv['_c_ct_off_int'] = [int(x) for x in bv['c_ct_off']]
                bv['_c_n_ct_int']   = [int(x) for x in bv['c_n_ct']]
            ct_flat = bv['c_ct_flat_buf']
            ct_flat.fill(0.0)
            _c_ct_off_int = bv['_c_ct_off_int']
            _c_n_ct_int   = bv['_c_n_ct_int']
            _c_ct_keys    = bv['c_ct_keys']
            if C_tilde_cache is not None:
                _ctc_get = C_tilde_cache.get
                for n in range(c_N):
                    ct_val = _ctc_get(_c_ct_keys[n])
                    if ct_val is not None and ct_val.shape[0] == _c_n_ct_int[n]:
                        ct_flat[_c_ct_off_int[n]:_c_ct_off_int[n + 1]] = (
                            ct_val.ravel())
            # Cache t2_flat scratch buffer on bv (size cycle-invariant).
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
            c_plan_struct = PyCTermInputs()
            c_plan_struct.N           = int(c_N)
            c_plan_struct.n_pno_arr   = bv['c_n_pno'].ctypes.data
            c_plan_struct.n_ct_arr    = bv['c_n_ct'].ctypes.data
            c_plan_struct.n_other_arr = bv['c_n_other'].ctypes.data
            c_plan_struct.S_big_off   = bv['c_S_big_off'].ctypes.data
            c_plan_struct.ct_off      = bv['c_ct_off'].ctypes.data
            c_plan_struct.S_mid_off   = bv['c_S_mid_off'].ctypes.data
            c_plan_struct.J_bold_off  = bv['c_J_bold_off'].ctypes.data
            c_plan_struct.t2_off      = bv['c_t2_off'].ctypes.data
            c_plan_struct.S_outer_off = bv['c_S_outer_off'].ctypes.data
            c_plan_struct.tile_off    = bv['c_tile_off'].ctypes.data
            c_plan_struct.S_big_flat  = bv['c_S_big_flat'].ctypes.data
            c_plan_struct.S_mid_flat  = bv['c_S_mid_flat'].ctypes.data
            c_plan_struct.J_bold_flat = bv['c_J_bold_flat'].ctypes.data
            c_plan_struct.S_outer_flat= bv['c_S_outer_flat'].ctypes.data
            c_plan_struct.ct_flat     = ct_flat.ctypes.data
            c_plan_struct.t2_flat     = t2_flat.ctypes.data
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
            own.extend([ct_flat, t2_flat, c_target_ij_arr, c_target_ji_arr])

        # D side.
        d_N = bv['d_N']
        if d_N > 0:
            # Cache u_flat scratch buffer (size cycle-invariant).
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
            # Same caching pattern as ct_flat: persistent buffer + Python-int
            # offset list to skip numpy box-unbox per item.
            if 'd_dt_flat_buf' not in bv:
                bv['d_dt_flat_buf'] = np.zeros(int(bv['d_dt_off'][-1]))
                bv['_d_dt_off_int'] = [int(x) for x in bv['d_dt_off']]
            dt_flat = bv['d_dt_flat_buf']
            dt_flat.fill(0.0)
            _d_dt_off_int = bv['_d_dt_off_int']
            _d_dt_keys    = bv['d_dt_keys']
            if D_tilde_cache is not None:
                _dtc_get = D_tilde_cache.get
                for n in range(d_N):
                    dt_val = _dtc_get(_d_dt_keys[n])
                    if dt_val is not None:
                        dt_flat[_d_dt_off_int[n]:_d_dt_off_int[n + 1]] = (
                            dt_val.ravel())
            d_plan_struct = PyDTermInputs()
            d_plan_struct.N         = int(d_N)
            d_plan_struct.n_pno_arr = bv['d_n_pno'].ctypes.data
            d_plan_struct.n_A_arr   = bv['d_n_A'].ctypes.data
            d_plan_struct.n_B_arr   = bv['d_n_B'].ctypes.data
            d_plan_struct.S_a_off   = bv['d_S_a_off'].ctypes.data
            d_plan_struct.u_off     = bv['d_u_off'].ctypes.data
            d_plan_struct.S_b_off   = bv['d_S_b_off'].ctypes.data
            d_plan_struct.S_c_off   = bv['d_S_c_off'].ctypes.data
            d_plan_struct.dt_off    = bv['d_dt_off'].ctypes.data
            d_plan_struct.KJ_off    = bv['d_KJ_off'].ctypes.data
            d_plan_struct.tile_off  = bv['d_tile_off'].ctypes.data
            d_plan_struct.S_a_flat  = bv['d_S_a_flat'].ctypes.data
            d_plan_struct.S_b_flat  = bv['d_S_b_flat'].ctypes.data
            d_plan_struct.S_c_flat  = bv['d_S_c_flat'].ctypes.data
            d_plan_struct.KJ_flat   = bv['d_KJ_flat'].ctypes.data
            d_plan_struct.u_flat    = u_flat.ctypes.data
            d_plan_struct.dt_flat   = dt_flat.ctypes.data
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
            own.extend([u_flat, dt_flat, d_target_ij_arr, d_target_ji_arr])

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
        verbose=True, _pool=None):
    """Pack-once optimized drop-in cycle driver.

    SolverInputs and per_kl plan are packed ONCE at function entry; per
    cycle only T1_flat / T2_flat / T1_in_pair are refreshed (not the
    full 210-pair flat buffers).  DIIS operates on flat buffers
    directly (no dict <-> array conversion).
    """
    import time as _time
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints
    from pyscf.cc.dlpno_tccsd.pair_index import build_t1_cache, PairIndex
    from pyscf.cc.dlpno_tccsd.lccsd import _compute_t1_residual_psi4

    _pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)

    # ---- ONE-TIME setup ----
    _t_setup0 = _time.perf_counter()
    _pack_prof = bool(int(os.environ.get('DLPNO_PACK_PROF', '0')))
    def _pmark(label, t0):
        if _pack_prof:
            print(f'  [PACK-PROF] {label}: '
                  f'{_time.perf_counter() - t0:.3f}s', flush=True)
    _t = _time.perf_counter()
    if hasattr(_compute_t1_residual_psi4, '_per_kl_plan_cache'):
        _compute_t1_residual_psi4._per_kl_plan_cache.clear()
    # Plan-only call: builds + caches _per_kl_plan_cache without running
    # the residual computation itself (we discard the result anyway).
    # Saves ~1.0s of one-time setup on water-15.
    _compute_t1_residual_psi4(
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
        t2_pno_all=t2_pno_all, S_pno_cache=S_pno_cache, _pool=_pool)
    _pmark('pack_for_t1_ints', _t)
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
    plan_struct, plan_own = _extract_per_kl_plan(_compute_t1_residual_psi4)
    if plan_struct is None:
        raise RuntimeError('per_kl plan unavailable')
    plan_struct.t2_buffer       = t2_pno_all._buffer.ctypes.data
    plan_struct.t1_cache_buffer = t1_cache._buffer.ctypes.data
    g_plan_struct, g_plan_own = _extract_g_tilde_plan(key_to_p)
    _pmark('extract per_kl + g_tilde plans', _t)

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
    # to set its team size.  When the driver runs with
    # `OMP_NUM_THREADS=1` (so the Python pool can dispatch cc_ints /
    # S_pno builds without OMP oversubscription), this returns 1 and
    # every class kernel runs SINGLE-THREADED.  The pool is idle while
    # the class iterates, so it's safe (and a big win) to bump the OMP
    # team inside the cycle loop only.  On water-22 baseline per-cycle
    # wall = 5.0 s; with OMP=16 each phase's BLAS dgemms parallelise
    # and per-cycle drops by O(2-4×).
    try:
        from threadpoolctl import threadpool_limits as _tpl
    except ImportError:
        _tpl = None
    _omp_n = int(os.environ.get('DLPNO_CCSD_CYCLE_OMP', '16'))
    _omp_ctx = (_tpl(limits=_omp_n, user_api='openmp')
                 if _tpl is not None and _omp_n > 1 else None)
    if _omp_ctx is not None:
        _omp_ctx.__enter__()

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
              f'E_corr={e_cyc:.10f}  dE={dE:.2e}  '
              f'[{_dt:.2f}s: t1cache={_t_t1cache:.2f} plans={_t_plans:.2f} '
              f'run_cyc={_t_run:.2f} diis={_t_diis:.2f} sync={_t_sync:.2f}]',
              flush=True)

        del native_own

        if dT < this_tol:
            print(f'  DLPNO-CCSD converged in {cycle + 1} cycles (amplitude, class).',
                  flush=True)
            if _omp_ctx is not None:
                _omp_ctx.__exit__(None, None, None)
            return cycle, e_cyc
        if cycle > 5 and dE < this_tol:
            print(f'  DLPNO-CCSD converged in {cycle + 1} cycles (energy, dE={dE:.2e}, class).',
                  flush=True)
            if _omp_ctx is not None:
                _omp_ctx.__exit__(None, None, None)
            return cycle, e_cyc

    if _omp_ctx is not None:
        _omp_ctx.__exit__(None, None, None)
    return max_cycle - 1, e_prev


def validate_run_one_cycle_full_with_external_R2(
        cc_ints, t1_pno_old, t1_pno_new, t2_pno_all_old, t2_new_dict,
        r1_pno, r2_all,
        pno_spaces, pair_lmo_idx, F_lmo, eps_lmo, fov_pno, nocc,
        keys_sorted, S_pno_cache, cc_ints_flat, pair_index, ovL_pno_cache,
        K_pno_cache, g_tilde_pyscf=None, g_term_pyscf=None,
        be_pyscf=None, b_tilde_per_ij_pyscf=None,
        c_term_pyscf=None, d_term_pyscf=None,
        jiang_C_pyscf=None, jiang_D_pyscf=None, verbose=True):
    """End-to-end validator: compares class's run_one_cycle output to
    PySCF's BEFORE-DIIS state.  Uses PySCF's r2_all as ``R2_external``
    while the native plan extraction (BE/CD/G_term/t3/t4) is incremental.

    Inputs (captured by the lccsd.py hook AFTER R1+T1 update +
    R2+T2_new computation but BEFORE DIIS):
      t1_pno_old: snapshot of t1_pno BEFORE this cycle's T1 update
      t1_pno_new: t1_pno AFTER Psi4 increment (== t1_pno at hook point)
      t2_pno_all_old: snapshot of t2_pno_all BEFORE this cycle's T2 update
                     (still the input t2 since DIIS hasn't run yet)
      t2_new_dict: dict of T2_new per pair (Psi4 increment applied)
      r1_pno: dict of full R1 per occupied i
      r2_all: dict of full R2 per pair (from compute_residual_v2)

    Compares T1_new, T2_new, and energy against PySCF's reference.
    """
    from pyscf.cc.dlpno_tccsd._ccsd_solver_pack_real import pack_for_t1_ints
    from pyscf.cc.dlpno_tccsd.lccsd import _compute_t1_residual_psi4

    # ---- Pre-flight: populate per_kl plan cache by calling PySCF R1. ----
    if hasattr(_compute_t1_residual_psi4, '_per_kl_plan_cache'):
        _compute_t1_residual_psi4._per_kl_plan_cache.clear()
    _compute_t1_residual_psi4(
        t1_pno_old, t2_pno_all_old, pno_spaces, fov_pno,
        F_lmo, eps_lmo, nocc, S_pno_cache, cc_ints,
        ovL_pno_cache=ovL_pno_cache,
        pair_lmo_idx=pair_lmo_idx, t1_cache=None, _pool=None,
        cc_ints_flat=cc_ints_flat, pair_index=pair_index)
    plan_struct, plan_own = _extract_per_kl_plan(_compute_t1_residual_psi4)
    if plan_struct is None:
        print('[CCSD MONO] no per_kl plan; skipping full-cycle validation',
              flush=True)
        return None
    plan_struct.t2_buffer       = t2_pno_all_old._buffer.ctypes.data
    plan_struct.t1_cache_buffer = None  # populated below

    # ---- Pack inputs (210 pairs) using OLD T1/T2 state. ----
    from pyscf.cc.dlpno_tccsd.pair_index import build_t1_cache, PairIndex
    _pi = PairIndex(pno_spaces.keys(), pno_spaces, pair_lmo_idx, nocc)
    t1_cache = build_t1_cache(t1_pno_old, _pi, S_pno_cache, pno_spaces)
    plan_struct.t1_cache_buffer = t1_cache._buffer.ctypes.data

    _all_keys = sorted(t2_pno_all_old.keys())
    inputs, ownership, key_to_p, aux = pack_for_t1_ints(
        cc_ints, t1_pno_old, t1_cache, pno_spaces, pair_lmo_idx,
        F_lmo, eps_lmo, fov_pno, nocc, _all_keys,
        t2_pno_all=t2_pno_all_old, S_pno_cache=S_pno_cache)

    keys_reorder = aux['keys_sorted']
    n_pairs = len(keys_reorder)
    npno = aux['n_pno_per_pair']
    pno_offsets = aux['pno_offsets']
    t2_offsets = aux['t2_offsets']

    # ---- Build R2_external from PySCF's r2_all (in our packed order). ----
    R2_total = int(t2_offsets[n_pairs])
    R2_external = np.zeros(R2_total, dtype=np.float64)
    for p, key in enumerate(keys_reorder):
        if key not in r2_all:
            continue
        npno_p = int(npno[p])
        if npno_p == 0:
            continue
        R2_external[t2_offsets[p]:t2_offsets[p + 1]] = (
            r2_all[key].ravel())
    ownership.append(R2_external)
    inputs.R2_external = R2_external.ctypes.data

    # is_strong_pair: 1 if pair was in PySCF's r2_all (i.e. iterated as
    # a strong pair), 0 otherwise.  Used to scope the energy formula.
    is_strong_arr = np.zeros(n_pairs, dtype=np.uint8)
    for p, key in enumerate(keys_reorder):
        if key in r2_all:
            is_strong_arr[p] = 1
    ownership.append(is_strong_arr)
    inputs.is_strong_pair = is_strong_arr.ctypes.data

    # ---- Invoke run_one_cycle. ----
    R1_size = sum(int(npno[i]) for i in range(nocc))
    R1_class = np.zeros(R1_size, dtype=np.float64)
    R2_class = np.zeros(R2_total, dtype=np.float64)

    # ---- G_tilde plan: validate G_tilde matrix matches PySCF. ----
    # build_G_tilde was called during cycle 0 → its _batched_plan cache
    # is populated.  Extract it, run the class's g_tilde_inner kernel
    # directly with Fkj as the initial G, and compare to PySCF's
    # _local_df_G (passed as g_tilde_pyscf).
    g_plan_struct, g_plan_own = _extract_g_tilde_plan(key_to_p)
    if g_plan_struct is not None and g_tilde_pyscf is not None:
        # Build Fkj initial state via t1_fock_finalize (run separately).
        # We need to actually invoke the class's t1_fock pipeline to get
        # Fkj — but for a self-contained test, just initialize G to
        # Fkj_pyscf (PySCF's _local_Fkj is in scope at the hook but not
        # passed; reuse g_tilde_pyscf - increment from its initial Fkj
        # is what we want).  Simpler: take G_init = g_tilde_pyscf - delta
        # is hard.  Instead: re-init G to zero, run kernel, compare to
        # (g_tilde_pyscf - Fkj_initial).  But Fkj_initial is also not
        # passed.  So skip the matrix-level check for now; once we have
        # native run_one_cycle producing G_tilde, that path validates it.
        pass
    plans = PyRunCycleInputs()
    for fname in ('g_tilde_plan', 'be_plan', 'c_term_plan',
                  'd_term_plan', 'g_term_plan', 't3_plan', 't4_plan'):
        setattr(plans, fname, None)
    plans.per_kl_plan = ctypes.pointer(plan_struct)
    if g_plan_struct is not None:
        plans.g_tilde_plan = ctypes.pointer(g_plan_struct)

    # Snapshot T1/T2 BEFORE first run_one_cycle (so we can reset for the
    # native-R2 second pass).
    T1_snapshot = aux['T1_flat'].copy()
    T2_snapshot = aux['T2_flat'].copy()

    G_tilde_class = np.zeros((nocc, nocc), dtype=np.float64)
    out = PyRunCycleOutputs()
    out.R1_flat = R1_class.ctypes.data
    out.R2_flat = R2_class.ctypes.data
    out.energy = 0.0
    out.G_tilde_out = G_tilde_class.ctypes.data
    rc = _libcc.DLPNOcompute_lccsd_run_one_cycle(
        ctypes.byref(inputs), ctypes.byref(plans), ctypes.byref(out))
    if rc != 0:
        raise RuntimeError(f'run_one_cycle rc={rc}')

    # After run_one_cycle, T1/T2 in inputs (pointing at aux['T1_flat'] /
    # aux['T2_flat']) have been updated in place.  T1_flat is sized for
    # per-canonical-pair pno_offsets; the first sum_i(npno_ii) entries
    # are the per-occupied t1 slab (diag-first invariant).
    T1_class = aux['T1_flat'][:int(pno_offsets[nocc])].copy()
    T2_class = aux['T2_flat'].copy()
    energy_class = float(out.energy)

    # ---- Compare R1. ----
    R1_ref = np.zeros(R1_size, dtype=np.float64)
    for i in range(nocc):
        if i not in r1_pno:
            continue
        npno_ii = int(npno[i])
        if npno_ii == 0:
            continue
        R1_ref[pno_offsets[i]:pno_offsets[i] + npno_ii] = r1_pno[i]
    d_R1 = float(np.max(np.abs(R1_class - R1_ref)))

    # ---- Compare T1_new. ----
    T1_ref = np.zeros(R1_size, dtype=np.float64)
    for i in range(nocc):
        if i not in t1_pno_new:
            continue
        npno_ii = int(npno[i])
        if npno_ii == 0:
            continue
        T1_ref[pno_offsets[i]:pno_offsets[i] + npno_ii] = t1_pno_new[i]
    d_T1 = float(np.max(np.abs(T1_class - T1_ref)))

    # ---- Compare T2_new. ----
    # Strong pairs: from t2_new_dict (Psi4 increment applied).  Weak
    # pairs: from t2_pno_all_old (unchanged — PySCF doesn't iterate
    # weak; class also leaves them since R[weak] = 0).
    T2_ref = np.zeros(R2_total, dtype=np.float64)
    for p, key in enumerate(keys_reorder):
        npno_p = int(npno[p])
        if npno_p == 0:
            continue
        if key in t2_new_dict:
            T2_ref[t2_offsets[p]:t2_offsets[p + 1]] = (
                t2_new_dict[key].ravel())
        elif key in t2_pno_all_old:
            T2_ref[t2_offsets[p]:t2_offsets[p + 1]] = (
                t2_pno_all_old[key].ravel())
    d_T2 = float(np.max(np.abs(T2_class - T2_ref)))

    # ---- Compare energy.  Rebuild t1_cache from t1_pno_new so the
    # tau term uses the freshly-updated T1.  Energy formula matches
    # the class's update_amps_and_energy phase line-by-line.
    # Use cc_ints['K_iajb'] (same K as class consumes via in_.K_iajb)
    # rather than K_pno_cache (which can differ slightly when built
    # from different DF paths — global vs local DF).
    t1_cache_new = build_t1_cache(t1_pno_new, _pi, S_pno_cache, pno_spaces)
    e_ref = 0.0
    for ii in range(nocc):
        if t1_pno_new[ii].size > 0:
            e_ref += float(np.dot(fov_pno[ii], t1_pno_new[ii]))
    for key, T2_p in t2_new_dict.items():
        if T2_p.size == 0:
            continue
        ci = cc_ints.get(key)
        if ci is None or 'K_iajb' not in ci:
            continue
        K_p = ci['K_iajb']
        i, j = key
        t1_i_in_p = t1_cache_new[key][i]
        t1_j_in_p = t1_cache_new[key][j]
        tau_p = T2_p + np.outer(t1_i_in_p, t1_j_in_p)
        contrib = float(np.sum(K_p * (2.0 * tau_p - tau_p.T)))
        e_ref += (1.0 if i == j else 2.0) * contrib

    d_E = float(abs(energy_class - e_ref))

    if verbose:
        print(f'[CCSD MONO] run_one_cycle FULL with R2_external '
              f'(packed n_canon_pairs={n_pairs}):',
              flush=True)
        print(f'    |dR1| = {d_R1:.3e}', flush=True)
        print(f'    |dT1| = {d_T1:.3e}', flush=True)
        print(f'    |dT2| = {d_T2:.3e}', flush=True)
        print(f'    |dE|  = {d_E:.3e}  (E_class={energy_class:.10f}  E_ref={e_ref:.10f})',
              flush=True)

    # ---- G_tilde matrix cross-check (if PySCF reference provided). ----
    if g_tilde_pyscf is not None:
        d_G = float(np.max(np.abs(G_tilde_class - g_tilde_pyscf)))
        print(f'    |dG_tilde| = {d_G:.3e}  '
              f'(class plan extracted from build_G_tilde._batched_plan)',
              flush=True)

    # ---- G_term per-pair cross-check (if PySCF reference provided). ----
    # Run the G_term batched kernel via the class with extracted plan,
    # scatter tiles into flat_G_ij/flat_G_ji, build per-pair G_term,
    # compare to PySCF's _G_term_all dict.
    if g_term_pyscf is not None:
        d_g_term = _validate_g_term_native(
            t2_pno_all_old, key_to_p, keys_reorder, pno_spaces,
            G_tilde_class, g_term_pyscf)
        print(f'    |dG_term|  = {d_g_term:.3e}  (per-pair G_term build)',
              flush=True)

    # ---- BE per-pair cross-check (if PySCF reference provided). ----
    if be_pyscf is not None and b_tilde_per_ij_pyscf is not None:
        d_be_B, d_be_E = _validate_be_native(
            t2_pno_all_old, b_tilde_per_ij_pyscf, pno_spaces,
            be_pyscf['B'], be_pyscf['E'])
        print(f'    |dBE_B|    = {d_be_B:.3e}  |dBE_E| = {d_be_E:.3e}  '
              f'(per-pair BE build)', flush=True)

    # ---- CD per-pair cross-check (if PySCF reference provided). ----
    if (c_term_pyscf is not None and d_term_pyscf is not None
            and jiang_C_pyscf is not None and jiang_D_pyscf is not None):
        d_c_term, d_d_term = _validate_cd_native(
            t2_pno_all_old, jiang_C_pyscf, jiang_D_pyscf,
            pno_spaces, c_term_pyscf, d_term_pyscf)
        print(f'    |dC_term|  = {d_c_term:.3e}  |dD_term| = {d_d_term:.3e}  '
              f'(per-pair CD build)', flush=True)

    # ============================================================
    # NATIVE R2 path: R2_external=null, all plans wired.  Compares
    # class-built R2 to PySCF's r2_all to localize what contributions
    # remain to be implemented (Fab*T2, ooL/ovL, P-symm).
    # ============================================================
    if (b_tilde_per_ij_pyscf is not None and jiang_C_pyscf is not None
            and jiang_D_pyscf is not None):
        natives, native_own = _build_native_r2_plans(
            t2_pno_all_old, key_to_p, keys_reorder, pno_spaces,
            b_tilde_per_ij_pyscf, jiang_C_pyscf, jiang_D_pyscf, n_pairs)

        # Build ord_idx_lookup: (a, b) ordered tuple -> ordered-pair idx.
        ord_idx_lookup = {}
        for o in range(int(aux['ordered_pair_i_idx'].size)):
            a = int(aux['ordered_pair_i_idx'][o])
            b = int(aux['ordered_pair_k_idx'][o])
            ord_idx_lookup[(a, b)] = o
        # Add t3+t4 plans + CD ord_pair_idx for native C_tilde/D_tilde build.
        natives, native_own = _add_t34_plans_to_natives(
            natives, native_own, t1_cache, t2_pno_all_old,
            ord_idx_lookup, key_to_p, nocc)

        # Build a fresh inputs without R2_external; reuse aux's T1/T2.
        # (Reset T1/T2 to old state first.)
        aux['T1_flat'][:] = T1_snapshot
        aux['T2_flat'][:] = T2_snapshot

        plans_native = PyRunCycleInputs()
        for fname in ('g_tilde_plan', 'be_plan', 'c_term_plan',
                      'd_term_plan', 'g_term_plan', 't3_plan', 't4_plan',
                      'g_term_plan_jk'):
            setattr(plans_native, fname, None)
        plans_native.per_kl_plan = ctypes.pointer(plan_struct)
        if g_plan_struct is not None:
            plans_native.g_tilde_plan = ctypes.pointer(g_plan_struct)
        # Wire native R2 plans.
        if natives['g_plan_ik'] is not None:
            plans_native.g_term_plan = ctypes.pointer(natives['g_plan_ik'])
            plans_native.g_term_plan_jk = ctypes.pointer(natives['g_plan_jk'])
            plans_native.g_term_target_pair_idx_ik = (
                natives['g_target_ik'].ctypes.data)
            plans_native.g_term_target_pair_idx_jk = (
                natives['g_target_jk'].ctypes.data)
        else:
            plans_native.g_term_target_pair_idx_ik = None
            plans_native.g_term_target_pair_idx_jk = None
        # BE
        plans_native.be_n_buckets    = natives['be_n_buckets']
        plans_native.be_plan_buckets = (
            ctypes.addressof(natives['be_plan_buckets'])
            if natives['be_plan_buckets'] is not None else 0)
        plans_native.be_n_unique_n_ij = natives['be_n_unique']
        plans_native.be_unique_n_ij = (
            natives['be_unique_n_ij'].ctypes.data
            if natives['be_unique_n_ij'] is not None else None)
        plans_native.be_flat_off_per_n_ij = (
            natives['be_flat_off_per_n_ij'].ctypes.data
            if natives['be_flat_off_per_n_ij'] is not None else None)
        plans_native.be_pair_n_ij_idx = (
            natives['be_pair_n_ij_idx'].ctypes.data
            if natives['be_pair_n_ij_idx'] is not None else None)
        plans_native.be_pair_slot = (
            natives['be_pair_slot'].ctypes.data
            if natives['be_pair_slot'] is not None else None)
        # CD
        if natives['c_plan'] is not None:
            plans_native.c_term_plan = ctypes.pointer(natives['c_plan'])
            plans_native.c_term_target_pair_idx_ij = (
                natives['c_target_ij'].ctypes.data)
            plans_native.c_term_target_pair_idx_ji = (
                natives['c_target_ji'].ctypes.data)
        if natives['d_plan'] is not None:
            plans_native.d_term_plan = ctypes.pointer(natives['d_plan'])
            plans_native.d_term_target_pair_idx_ij = (
                natives['d_target_ij'].ctypes.data)
            plans_native.d_term_target_pair_idx_ji = (
                natives['d_target_ji'].ctypes.data)

        # t3+t4 plans for C_tilde / D_tilde Phase 2 native build.
        for k_struct, k_target, attr in [
                ('c_t3_struct', 'c_t3_target_ord', 'c_t3'),
                ('c_t4_struct', 'c_t4_target_ord', 'c_t4'),
                ('d_t3_struct', 'd_t3_target_ord', 'd_t3'),
                ('d_t4_struct', 'd_t4_target_ord', 'd_t4')]:
            s = natives.get(k_struct)
            t = natives.get(k_target)
            if s is not None:
                setattr(plans_native, f'{attr}_plan', ctypes.pointer(s))
                setattr(plans_native, f'{attr}_target_ord_idx',
                        t.ctypes.data if t is not None else None)
            else:
                setattr(plans_native, f'{attr}_plan', None)
                setattr(plans_native, f'{attr}_target_ord_idx', None)
        # Per-CD-item ord_pair_idx for native ct_flat/dt_flat gather.
        plans_native.c_term_ct_ord_pair_idx = (
            natives['c_ct_ord_pair_idx'].ctypes.data
            if 'c_ct_ord_pair_idx' in natives else None)
        plans_native.d_term_dt_ord_pair_idx = (
            natives['d_dt_ord_pair_idx'].ctypes.data
            if 'd_dt_ord_pair_idx' in natives else None)

        # Disable R2_external to force native path.
        inputs.R2_external = None

        R1_native = np.zeros(R1_size, dtype=np.float64)
        R2_native = np.zeros(R2_total, dtype=np.float64)
        out_native = PyRunCycleOutputs()
        out_native.R1_flat = R1_native.ctypes.data
        out_native.R2_flat = R2_native.ctypes.data
        out_native.energy = 0.0
        out_native.G_tilde_out = 0
        rc = _libcc.DLPNOcompute_lccsd_run_one_cycle(
            ctypes.byref(inputs), ctypes.byref(plans_native),
            ctypes.byref(out_native))
        if rc != 0:
            raise RuntimeError(f'native run_one_cycle rc={rc}')
        # Compare R2 to PySCF r2_all.
        R2_ref = np.zeros(R2_total, dtype=np.float64)
        for p, key in enumerate(keys_reorder):
            if key not in r2_all:
                continue
            sl = slice(int(t2_offsets[p]), int(t2_offsets[p + 1]))
            R2_ref[sl] = r2_all[key].ravel()
        d_R2_native = float(np.max(np.abs(R2_native - R2_ref)))
        d_R1_native = float(np.max(np.abs(R1_native - R1_ref)))
        d_T1_native = float(np.max(np.abs(
            aux['T1_flat'][:int(pno_offsets[nocc])] - T1_ref)))
        d_T2_native = float(np.max(np.abs(aux['T2_flat'] - T2_ref)))
        print(f'[CCSD MONO] NATIVE (no R2_external) vs PySCF '
              f'(packed n_canon_pairs={n_pairs}):',
              flush=True)
        print(f'    |dR1| = {d_R1_native:.3e}', flush=True)
        print(f'    |dR2| = {d_R2_native:.3e}', flush=True)
        print(f'    |dT1| = {d_T1_native:.3e}', flush=True)
        print(f'    |dT2| = {d_T2_native:.3e}', flush=True)
        print(f'    |dE|  = {abs(out_native.energy - e_ref):.3e}  '
              f'(E_native={out_native.energy:.10f})', flush=True)
        # Restore for cleanup.
        inputs.R2_external = R2_external.ctypes.data
        del native_own

    del plan_own, ownership
    return d_R1, d_T1, d_T2, d_E


def _validate_cd_native(t2_pno_all, C_tilde_cache, D_tilde_cache,
                          pno_spaces, C_term_ref, D_term_ref):
    """Run C_term and D_term batched kernels via the class with
    extracted plan + per-iter gathers; assemble per-pair C_term / D_term
    and compare to PySCF references."""
    from pyscf.cc.dlpno_tccsd.residual import (
        compute_CD_terms_batched, _get_or_build_cd_batched_view)
    from pyscf.cc.dlpno_tccsd._cd_gather_cy import (
        gather_t2_with_transpose, gather_u_from_t2)
    cache = getattr(compute_CD_terms_batched, '_plan_cache', None)
    if not cache:
        return float('nan'), float('nan')
    plan = next(iter(cache.values()))
    bv = _get_or_build_cd_batched_view(plan, pno_spaces, t2_pno_all)

    pairs_by_n_pno = plan['pairs_by_n_pno']
    n_pno_offsets = bv['n_pno_offsets']

    # Allocate flat output buffers for C/D, ij/ji.
    flat_C_ij = {n_pno: np.zeros((len(pairs), n_pno, n_pno))
                 for n_pno, pairs in pairs_by_n_pno.items()}
    flat_C_ji = {n_pno: np.zeros((len(pairs), n_pno, n_pno))
                 for n_pno, pairs in pairs_by_n_pno.items()}
    flat_D_ij = {n_pno: np.zeros((len(pairs), n_pno, n_pno))
                 for n_pno, pairs in pairs_by_n_pno.items()}
    flat_D_ji = {n_pno: np.zeros((len(pairs), n_pno, n_pno))
                 for n_pno, pairs in pairs_by_n_pno.items()}

    own_keep = []

    # ---- C side. ----
    c_N = bv['c_N']
    if c_N > 0:
        # Gather ct_flat from C_tilde_cache.
        ct_flat = np.zeros(int(bv['c_ct_off'][-1]))
        c_n_ct = bv['c_n_ct']
        c_ct_keys = bv['c_ct_keys']
        c_ct_off = bv['c_ct_off']
        for n in range(c_N):
            ct_val = (C_tilde_cache.get(c_ct_keys[n])
                      if C_tilde_cache is not None else None)
            if ct_val is not None and ct_val.shape[0] == int(c_n_ct[n]):
                ct_flat[c_ct_off[n]:c_ct_off[n + 1]] = ct_val.ravel()

        # Gather t2_flat.
        t2_flat = np.empty(int(bv['c_t2_off'][-1]))
        gather_t2_with_transpose(
            c_N, bv['c_n_other'],
            bv['c_t2_canon_off'], bv['c_t2_trans_arr'],
            bv['c_t2_off'], t2_pno_all._buffer, t2_flat,
            min(64, c_N),
        )

        max_n_pno = int(bv['c_n_pno'].max(initial=1))
        max_n_ct = int(bv['c_n_ct'].max(initial=1))
        max_n_other = int(bv['c_n_other'].max(initial=1))

        plan_c = PyCTermInputs()
        plan_c.N = int(c_N)
        plan_c.n_pno_arr   = bv['c_n_pno'].ctypes.data
        plan_c.n_ct_arr    = bv['c_n_ct'].ctypes.data
        plan_c.n_other_arr = bv['c_n_other'].ctypes.data
        plan_c.S_big_off   = bv['c_S_big_off'].ctypes.data
        plan_c.ct_off      = bv['c_ct_off'].ctypes.data
        plan_c.S_mid_off   = bv['c_S_mid_off'].ctypes.data
        plan_c.J_bold_off  = bv['c_J_bold_off'].ctypes.data
        plan_c.t2_off      = bv['c_t2_off'].ctypes.data
        plan_c.S_outer_off = bv['c_S_outer_off'].ctypes.data
        plan_c.tile_off    = bv['c_tile_off'].ctypes.data
        plan_c.S_big_flat  = bv['c_S_big_flat'].ctypes.data
        plan_c.S_mid_flat  = bv['c_S_mid_flat'].ctypes.data
        plan_c.J_bold_flat = bv['c_J_bold_flat'].ctypes.data
        plan_c.S_outer_flat= bv['c_S_outer_flat'].ctypes.data
        plan_c.ct_flat     = ct_flat.ctypes.data
        plan_c.t2_flat     = t2_flat.ctypes.data
        plan_c.max_n_pno   = max_n_pno
        plan_c.max_n_ct    = max_n_ct
        plan_c.max_n_other = max_n_other

        c_tiles = np.zeros(int(bv['c_tile_off'][-1]))
        out_c = PyCTermOutputs()
        out_c.tiles_flat = c_tiles.ctypes.data
        rc = _libcc.DLPNOcompute_lccsd_phase_c_term(
            ctypes.byref(PySolverInputs()),
            ctypes.byref(plan_c), ctypes.byref(out_c))
        if rc != 0:
            raise RuntimeError(f'phase_c_term rc={rc}')
        own_keep.extend([ct_flat, t2_flat, c_tiles])

        # Scatter c_tiles -> flat_C_ij / flat_C_ji.  PySCF SUBTRACTS.
        c_target_ij = bv['c_target_off_ij']
        c_target_ji = bv['c_target_off_ji']
        c_n_pno = bv['c_n_pno']
        c_tile_off = bv['c_tile_off']
        flat_C_ij_views = {n_pno: flat_C_ij[n_pno].ravel()
                            for n_pno in pairs_by_n_pno}
        flat_C_ji_views = {n_pno: flat_C_ji[n_pno].ravel()
                            for n_pno in pairs_by_n_pno}
        for n in range(c_N):
            n_pno = int(c_n_pno[n])
            tile_size = n_pno * n_pno
            tile = c_tiles[c_tile_off[n]:c_tile_off[n + 1]]
            if c_target_ij[n] >= 0:
                base = c_target_ij[n] - n_pno_offsets[n_pno]
                flat_C_ij_views[n_pno][base:base + tile_size] -= tile
            else:
                base = c_target_ji[n] - n_pno_offsets[n_pno]
                flat_C_ji_views[n_pno][base:base + tile_size] -= tile

    # ---- D side. ----
    d_N = bv['d_N']
    if d_N > 0:
        # u = 2*t2 - t2.T (anti-sym).
        u_flat = np.empty(int(bv['d_u_off'][-1]))
        gather_u_from_t2(
            d_N, bv['d_n_A'],
            bv['d_t2_canon_off'], bv['d_t2_trans_arr'],
            bv['d_u_off'], t2_pno_all._buffer, u_flat,
            min(64, d_N),
        )

        # dt_flat from D_tilde_cache (per-item Python lookup).
        dt_flat = np.zeros(int(bv['d_dt_off'][-1]))
        d_dt_keys = bv['d_dt_keys']
        d_dt_off = bv['d_dt_off']
        for n in range(d_N):
            dk = d_dt_keys[n]
            dt_val = (D_tilde_cache.get(dk)
                      if D_tilde_cache is not None else None)
            if dt_val is not None:
                dt_flat[d_dt_off[n]:d_dt_off[n + 1]] = dt_val.ravel()

        max_n_pno = int(bv['d_n_pno'].max(initial=1))
        max_n_A = int(bv['d_n_A'].max(initial=1))
        max_n_B = int(bv['d_n_B'].max(initial=1))

        plan_d = PyDTermInputs()
        plan_d.N = int(d_N)
        plan_d.n_pno_arr = bv['d_n_pno'].ctypes.data
        plan_d.n_A_arr   = bv['d_n_A'].ctypes.data
        plan_d.n_B_arr   = bv['d_n_B'].ctypes.data
        plan_d.S_a_off   = bv['d_S_a_off'].ctypes.data
        plan_d.u_off     = bv['d_u_off'].ctypes.data
        plan_d.S_b_off   = bv['d_S_b_off'].ctypes.data
        plan_d.S_c_off   = bv['d_S_c_off'].ctypes.data
        plan_d.dt_off    = bv['d_dt_off'].ctypes.data
        plan_d.KJ_off    = bv['d_KJ_off'].ctypes.data
        plan_d.tile_off  = bv['d_tile_off'].ctypes.data
        plan_d.S_a_flat  = bv['d_S_a_flat'].ctypes.data
        plan_d.S_b_flat  = bv['d_S_b_flat'].ctypes.data
        plan_d.S_c_flat  = bv['d_S_c_flat'].ctypes.data
        plan_d.KJ_flat   = bv['d_KJ_flat'].ctypes.data
        plan_d.u_flat    = u_flat.ctypes.data
        plan_d.dt_flat   = dt_flat.ctypes.data
        plan_d.max_n_pno = max_n_pno
        plan_d.max_n_A   = max_n_A
        plan_d.max_n_B   = max_n_B

        d_tiles = np.zeros(int(bv['d_tile_off'][-1]))
        out_d = PyDTermOutputs()
        out_d.tiles_flat = d_tiles.ctypes.data
        rc = _libcc.DLPNOcompute_lccsd_phase_d_term(
            ctypes.byref(PySolverInputs()),
            ctypes.byref(plan_d), ctypes.byref(out_d))
        if rc != 0:
            raise RuntimeError(f'phase_d_term rc={rc}')
        own_keep.extend([u_flat, dt_flat, d_tiles])

        # Scatter d_tiles -> flat_D_ij / flat_D_ji.  PySCF ADDS 0.5*tile
        # (residual.py:4874 — different convention from C side).
        d_target_ij = bv['d_target_off_ij']
        d_target_ji = bv['d_target_off_ji']
        d_n_pno = bv['d_n_pno']
        d_tile_off = bv['d_tile_off']
        flat_D_ij_views = {n_pno: flat_D_ij[n_pno].ravel()
                            for n_pno in pairs_by_n_pno}
        flat_D_ji_views = {n_pno: flat_D_ji[n_pno].ravel()
                            for n_pno in pairs_by_n_pno}
        for n in range(d_N):
            n_pno = int(d_n_pno[n])
            tile_size = n_pno * n_pno
            tile = d_tiles[d_tile_off[n]:d_tile_off[n + 1]]
            if d_target_ij[n] >= 0:
                base = d_target_ij[n] - n_pno_offsets[n_pno]
                flat_D_ij_views[n_pno][base:base + tile_size] += 0.5 * tile
            else:
                base = d_target_ji[n] - n_pno_offsets[n_pno]
                flat_D_ji_views[n_pno][base:base + tile_size] += 0.5 * tile

    # Assemble per-pair C_term, D_term and compare.
    pair_to_slot = plan['pair_to_slot']
    max_dC = 0.0
    max_dD = 0.0
    for key, slot in pair_to_slot.items():
        n_pno = pno_spaces[key]['C_pno'].shape[1]
        if n_pno == 0:
            continue
        Cij = flat_C_ij[n_pno][slot]
        Cji = flat_C_ji[n_pno][slot]
        Dij = flat_D_ij[n_pno][slot]
        Dji = flat_D_ji[n_pno][slot]
        C_class = 0.5 * Cij + Cij.T + 0.5 * Cji.T + Cji
        D_class = Dij + Dji.T
        if key in C_term_ref:
            d = float(np.max(np.abs(C_class - C_term_ref[key])))
            max_dC = max(max_dC, d)
        if key in D_term_ref:
            d = float(np.max(np.abs(D_class - D_term_ref[key])))
            max_dD = max(max_dD, d)
    return max_dC, max_dD


def _validate_be_native(t2_pno_all, b_tilde_per_ij, pno_spaces,
                          B_ref_dict, E_ref_dict):
    """Run BE kernel via class per bucket; compare to PySCF's B/E dicts."""
    from pyscf.cc.dlpno_tccsd.residual import compute_B_E_batched_v2
    cache = getattr(compute_B_E_batched_v2, '_plan_cache', None)
    if not cache:
        return float('nan'), float('nan')
    plan = next(iter(cache.values()))

    pairs_by_n_ij = plan['pairs_by_n_ij']
    pair_to_slot = plan['pair_to_slot']
    flat_B = {n_ij: np.zeros((len(pairs), n_ij, n_ij))
              for n_ij, pairs in pairs_by_n_ij.items()}
    flat_E = {n_ij: np.zeros((len(pairs), n_ij, n_ij))
              for n_ij, pairs in pairs_by_n_ij.items()}

    ownership = []
    for bucket in plan['buckets']:
        n_ij = bucket['n_ij']
        n_kl = bucket['n_kl']
        N = len(bucket['kl_keys'])

        # Per-iter T_arr build (mirror PySCF non-v2 path).
        T_arr = np.empty((N, n_kl, n_kl))
        beta_kl_arr = np.empty(N)
        beta_lk_arr = np.empty(N)
        for n in range(N):
            T_arr[n] = t2_pno_all[bucket['kl_keys'][n]]
            key_ij, k, l = bucket['beta_coords'][n]
            B_tilde = b_tilde_per_ij[key_ij]
            if isinstance(B_tilde, tuple):
                B_local, p_dense = B_tilde
                beta_kl_arr[n] = B_local[p_dense[k], p_dense[l]]
                beta_lk_arr[n] = (
                    0.0 if k == l
                    else B_local[p_dense[l], p_dense[k]])
            else:
                beta_kl_arr[n] = B_tilde[k, l]
                beta_lk_arr[n] = 0.0 if k == l else B_tilde[l, k]

        plan_struct = PyBEInputs()
        plan_struct.N = int(N)
        plan_struct.n_ij = int(n_ij)
        plan_struct.n_kl = int(n_kl)
        plan_struct.n_slots = int(flat_B[n_ij].shape[0])
        S_c = np.ascontiguousarray(bucket['S'])
        T_c = np.ascontiguousarray(T_arr)
        K_c = np.ascontiguousarray(bucket['K'])
        same_c = np.ascontiguousarray(bucket['same']).astype(np.uint8, copy=False)
        idx_c = np.ascontiguousarray(bucket['item_idx']).astype(np.int64, copy=False)
        plan_struct.S       = S_c.ctypes.data
        plan_struct.T       = T_c.ctypes.data
        plan_struct.K       = K_c.ctypes.data
        plan_struct.beta_kl = beta_kl_arr.ctypes.data
        plan_struct.beta_lk = beta_lk_arr.ctypes.data
        plan_struct.same    = same_c.ctypes.data
        plan_struct.idx     = idx_c.ctypes.data

        out = PyBEOutputs()
        out.out_B = flat_B[n_ij].ctypes.data
        out.out_E = flat_E[n_ij].ctypes.data
        rc = _libcc.DLPNOcompute_lccsd_phase_be(
            ctypes.byref(PySolverInputs()),
            ctypes.byref(plan_struct),
            ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f'phase_be rc={rc}')
        ownership.extend([S_c, T_c, K_c, beta_kl_arr, beta_lk_arr,
                          same_c, idx_c])

    # Build B_all, E_all dicts and compare.
    max_dB, max_dE = 0.0, 0.0
    for key, slot in pair_to_slot.items():
        n_ij = pno_spaces[key]['C_pno'].shape[1]
        if n_ij == 0:
            continue
        B_class = flat_B[n_ij][slot]
        E_class = flat_E[n_ij][slot]
        if key in B_ref_dict:
            d = float(np.max(np.abs(B_class - B_ref_dict[key])))
            max_dB = max(max_dB, d)
        if key in E_ref_dict:
            d = float(np.max(np.abs(E_class - E_ref_dict[key])))
            max_dE = max(max_dE, d)
    return max_dB, max_dE


def _validate_g_term_native(t2_pno_all, key_to_p, keys_reorder, pno_spaces,
                              G_tilde, g_term_pyscf):
    """Run G_term natively via class kernel + plan extraction; compare to
    PySCF's _G_term_all dict.

    Uses PySCF's `pairs_by_n_ij` / `pair_to_slot` from the cached plan
    so the tile-scatter slot indexing matches what `target_slot` was
    built against.
    """
    from pyscf.cc.dlpno_tccsd.residual import compute_G_term_batched
    cache = getattr(compute_G_term_batched, '_plan_cache', None)
    if not cache:
        return float('nan')
    plan = next(iter(cache.values()))
    pairs_by_n_ij = plan['pairs_by_n_ij']
    pair_to_slot_pyscf = plan['pair_to_slot']  # key → slot int
    # Convert to (n_ij, slot) form keyed by key.
    pair_to_slot = {}
    for key, slot in pair_to_slot_pyscf.items():
        n_ij = pno_spaces[key]['C_pno'].shape[1]
        pair_to_slot[key] = (n_ij, slot)

    flat_G_ij = {n_ij: np.zeros((len(pairs), n_ij, n_ij))
                 for n_ij, pairs in pairs_by_n_ij.items()}
    flat_G_ji = {n_ij: np.zeros((len(pairs), n_ij, n_ij))
                 for n_ij, pairs in pairs_by_n_ij.items()}

    G_tilde_c = np.ascontiguousarray(G_tilde)

    for side, flat_out in [('ik', flat_G_ij), ('jk', flat_G_ji)]:
        plan_struct, plan_own, target_slots, _ = _extract_g_term_plan(
            t2_pno_all, key_to_p, side=side)
        if plan_struct is None:
            continue
        plan_struct.G_tilde = G_tilde_c.ctypes.data
        plan_struct.G_stride = int(G_tilde_c.shape[1])

        # Allocate tiles.
        # tile_off is (N+1,) but our struct stores it as ctypes pointer
        # to the SAME numpy array — read it from ownership.
        side_bv_t2_off = plan_own[3]      # t2_off
        side_bv_tile_off = plan_own[4]    # tile_off
        N = plan_struct.N
        tiles_total = int(side_bv_tile_off[N])
        tiles_flat = np.zeros(tiles_total, dtype=np.float64)

        out = PyGTermOutputs()
        out.tiles_flat = tiles_flat.ctypes.data
        rc = _libcc.DLPNOcompute_lccsd_phase_g_term(
            ctypes.byref(PySolverInputs()),  # SolverInputs unused by kernel
            ctypes.byref(plan_struct),
            ctypes.byref(out))
        if rc != 0:
            raise RuntimeError(f'phase_g_term rc={rc}')

        # Scatter tiles into flat_out.  For item n: target_slots[n] = (n_ij, slot);
        # tile is at tiles_flat[tile_off[n]:tile_off[n+1]] reshaped (n_ij, n_ij);
        # PySCF SUBTRACTS the tile (residual.py:877).
        for n in range(N):
            n_ij_n, slot_n = target_slots[n]
            tile = tiles_flat[side_bv_tile_off[n]:side_bv_tile_off[n + 1]]
            flat_out[n_ij_n][slot_n] -= tile.reshape(n_ij_n, n_ij_n)
        del plan_own

    # Build G_term per pair: G_term[key] = flat_G_ij[slot] + flat_G_ji[slot].T.
    max_diff = 0.0
    for key, (n_ij, slot) in pair_to_slot.items():
        G_class = flat_G_ij[n_ij][slot] + flat_G_ji[n_ij][slot].T
        if key not in g_term_pyscf:
            continue
        G_ref = g_term_pyscf[key]
        diff = float(np.max(np.abs(G_class - G_ref)))
        if diff > max_diff:
            max_diff = diff
    return max_diff


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
    `_compute_t1_residual_psi4`.  Caller must have already invoked the
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


def smoke_test():
    """Verify the library loaded and the entry points are callable."""
    n = _libcc.DLPNOcompute_lccsd_solver_inputs_size()
    py_n = ctypes.sizeof(PySolverInputs)
    if n != py_n:
        raise RuntimeError(
            f'sizeof(SolverInputs): C={n} vs Python={py_n} — struct layouts disagree')
    print(f'[CCSD MONO] sizeof(SolverInputs) = {n} bytes (C and Python agree)',
          flush=True)

    e_out = ctypes.c_double(0.0)
    rc = _libcc.DLPNOcompute_lccsd_omp(None, ctypes.byref(e_out))
    if rc != -2:
        raise RuntimeError(f'expected -2 for NULL inputs, got {rc}')
    print(f'[CCSD MONO] NULL-inputs guard returned rc={rc} (expected -2)',
          flush=True)
    return n


def is_enabled():
    """Whether the new monolithic path is requested via env."""
    return bool(int(os.environ.get('DLPNO_CCSD_MONO', '0')))


if __name__ == '__main__':
    smoke_test()
    parity_test_dump_inputs(verbose=False)
    parity_test_phase_t1_ints()
    parity_test_phase_b_tilde()
    parity_test_phase_t1_fock()
    parity_test_phase_d_tilde_ph1()
    parity_test_phase_c_tilde_ph1()
    parity_test_phase_g_tilde_inner()
    parity_test_phase_t1_fock_finalize()
    parity_test_phase_t1_residual_per_kl()
    parity_test_phase_be()
    parity_test_phase_c_term()
    parity_test_phase_d_term()
    parity_test_phase_g_term()
    parity_test_phase_t3()
    parity_test_phase_t4()
    parity_test_phase_k_ladder()
    parity_test_phase_update_amps_and_energy()
    parity_test_phase_t1_fock_fia_bar()
    parity_test_phase_t1_residual_AC_init()
