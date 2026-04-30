#!/bin/bash
# Production build of pyscf/lib/cc with Intel MKL.
#
# Why: at small-matrix DGEMM sizes (npno~25 in DLPNO-CCSD), MKL is
# 2.5-3x faster than OpenBLAS.  Validated on water-10:
#   OpenBLAS:  per-cycle 1.32s, total wall 91s
#   MKL:       per-cycle 0.97s, total wall 84s   (matches Psi4 1.08s/cycle)
#
# Prereq: MKL .so.2 libs available in /environments/miniconda3/envs/psi4/lib
# (or any path with libmkl_intel_lp64.so.2 + libmkl_gnu_thread.so.2 + libmkl_core.so.2).
#
# After build, must launch python with:
#   LD_LIBRARY_PATH=/environments/miniconda3/envs/psi4/lib:$LD_LIBRARY_PATH \
#     python -m pyscf.cc.dlpno_tccsd._test_water10_perf

set -euo pipefail

MKL_LIB_DIR="${MKL_LIB_DIR:-/environments/miniconda3/envs/psi4/lib}"
INSTALL_DIR="${INSTALL_DIR:-/environments/miniconda3/envs/tmc/lib/python3.12/site-packages/pyscf/lib}"

# Verify MKL libs exist.
for lib in libmkl_intel_lp64.so.2 libmkl_gnu_thread.so.2 libmkl_core.so.2; do
    if [[ ! -f "$MKL_LIB_DIR/$lib" ]]; then
        echo "ERROR: $MKL_LIB_DIR/$lib not found"
        exit 1
    fi
done

cd "$(dirname "$0")/../pyscf/lib/build"

cmake \
    -DBLAS_LIBRARIES="$MKL_LIB_DIR/libmkl_intel_lp64.so.2;$MKL_LIB_DIR/libmkl_gnu_thread.so.2;$MKL_LIB_DIR/libmkl_core.so.2" \
    ..

make -j8 cc

# Sync to install dir.
cp ../../libcc.so "$INSTALL_DIR/libcc.so"

echo ""
echo "Build complete.  libcc.so linked against MKL:"
LD_LIBRARY_PATH="$MKL_LIB_DIR:$LD_LIBRARY_PATH" ldd "$INSTALL_DIR/libcc.so" | grep mkl

echo ""
echo "To run with MKL, launch python with:"
echo "  LD_LIBRARY_PATH=$MKL_LIB_DIR:\$LD_LIBRARY_PATH python -m pyscf.cc.dlpno_tccsd._test_water10_perf"
