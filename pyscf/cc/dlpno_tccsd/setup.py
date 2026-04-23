"""Standalone Cython build for the DLPNO-CCSD restructure.

Build in-place::

    cd pyscf/cc/dlpno_tccsd
    python setup.py build_ext --inplace

This compiles every ``*.pyx`` file in this directory into a ``.so``
sitting next to the Python modules (option A from the plan — simplest
for dev iteration, matches the Phase 3 decision).

Phase 4+ kernels should just be added as more ``Extension`` entries
below; the ``extra_compile_args`` / ``extra_link_args`` already cover
OpenMP (``-fopenmp``) and optimisation (``-O3``), and ``scipy``'s
Cython BLAS headers are pulled in by ``scipy.linalg.cython_blas``'s
``cimport`` mechanism.
"""
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

_HERE = Path(__file__).resolve().parent
_PYX_FILES = sorted(p.name for p in _HERE.glob("*.pyx"))

_EXTRA_COMPILE_ARGS = ["-O3", "-fopenmp", "-march=native", "-ffast-math"]
_EXTRA_LINK_ARGS = ["-fopenmp"]

_extensions = [
    Extension(
        # Extension name = module name (no dotted path: we build
        # in-place inside the package, not install to site-packages).
        pyx.removesuffix(".pyx"),
        sources=[pyx],
        include_dirs=[np.get_include()],
        extra_compile_args=_EXTRA_COMPILE_ARGS,
        extra_link_args=_EXTRA_LINK_ARGS,
    )
    for pyx in _PYX_FILES
]

setup(
    name="pyscf_dlpno_tccsd_cython",
    ext_modules=cythonize(
        _extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "initializedcheck": False,
        },
    ),
)
