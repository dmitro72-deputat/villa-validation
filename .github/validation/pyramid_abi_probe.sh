#!/usr/bin/env bash
# Compare the link symbols expected by the same C++ consumer of both public headers.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
tmp=$(mktemp -d)
trap 'rm -r "$tmp"' EXIT
header=vc_reader/include/vc_reader/zarr_write.hpp
mkdir -p "$tmp/deps/nlohmann"
curl --fail --silent --show-error --location   https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp   -o "$tmp/deps/nlohmann/json.hpp"
cat > "$tmp/consumer.cpp" <<'CPP'
#include "vc_reader/zarr_write.hpp"
void build_level(const std::filesystem::path& root) {
    vcr::zarr::writePyramidLevel(root, 1, {});
}
CPP
for revision in 659e2fadb836d44063ed831d85517154ec852f11 a6b3819; do
    mkdir -p "$tmp/$revision/vc_reader"
    git -C "$root" show "$revision:$header" > "$tmp/$revision/vc_reader/zarr_write.hpp"
    ln -s "$root/vc_reader/src/cpp" "$tmp/$revision/vc_reader/native"
    "${CXX:-g++}" -std=c++20 -I"$tmp/$revision" -I"$root/vc_reader/include" -I"$tmp/deps" \
        -c "$tmp/consumer.cpp" -o "$tmp/$revision.o"
    printf '%s\n' "$revision"
    nm -C -u "$tmp/$revision.o" | grep 'writePyramidLevel'
done
