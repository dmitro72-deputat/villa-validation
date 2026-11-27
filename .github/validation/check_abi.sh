#!/usr/bin/env bash
# Link a consumer compiled against the unchanged upstream header to the installed reader.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
trap 'rm -r "$work"' EXIT
mkdir -p "$work/include/vc_reader"
git show 659e2fadb836d44063ed831d85517154ec852f11:vc_reader/include/vc_reader/zarr_write.hpp > "$work/include/vc_reader/zarr_write.hpp"
ln -s "$root/vc_reader/src/cpp" "$work/include/vc_reader/native"
cat > "$work/check.cpp" <<'CPP'
#include "vc_reader/zarr_write.hpp"
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    vcr::zarr::writePyramidLevel(std::filesystem::path(argv[1]), 1, {});
}
CPP
g++ -std=c++20 -I"$work/include" -I"$root/vc_reader/include" -c "$work/check.cpp" -o "$work/check.o"
platlib="$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')"
lib="$(find "$platlib" -name libvc_reader_native.so -print -quit)"
test -f "$lib"
symbol="$(nm -u "$work/check.o" | awk '/writePyramidLevel/ {print $2}')"
test -n "$symbol"
nm -D --defined-only "$lib" > "$work/symbols.txt"
grep -F "$symbol" "$work/symbols.txt"
libdir="$(dirname "$lib")"
g++ "$work/check.o" "$lib" -Wl,-rpath,"$libdir" -Wl,-rpath-link,"$libdir" -o "$work/check"
python - "$work/levels" <<'PY'
from pathlib import Path
import sys
from vc_reader.reader import ZarrWriter, create_pyramid_levels

root = Path(sys.argv[1])
source = ZarrWriter.create(root / "0", shape=(4, 4, 4), chunks=(2, 2, 2), dtype="u1")
source[...] = 17
create_pyramid_levels(root, 1)
PY
"$work/check" "$work/levels"
python - "$work/levels" <<'PY'
from pathlib import Path
import sys
from vc_reader.reader import ZarrReader

root = Path(sys.argv[1])
assert (ZarrReader.open(root / "1")[...] == 17).all()
print("An upstream-header C++ consumer linked and ran against the hardened shared reader.")
PY
