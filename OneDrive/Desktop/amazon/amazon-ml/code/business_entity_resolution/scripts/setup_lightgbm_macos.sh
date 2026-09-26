#!/usr/bin/env bash
# LightGBM's macOS wheel links @rpath/libomp.dylib, which normally comes from Homebrew (`brew install libomp`).
# Without Homebrew, point it at the libomp bundled with scikit-learn (one OpenMP runtime per process)
# and re-sign the patched library ad hoc. Not needed on Linux.
set -euo pipefail
SP=$(python -c "import site; print(site.getsitepackages()[0])")
LIB="$SP/lightgbm/lib/lib_lightgbm.dylib"
install_name_tool -add_rpath "@loader_path/../../sklearn/.dylibs" "$LIB" 2>/dev/null || true
codesign --force -s - "$LIB"
python -c "import lightgbm; print('lightgbm', lightgbm.__version__, 'OK')"
