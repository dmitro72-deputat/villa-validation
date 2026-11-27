"""Validate native sparse/dense pyramids with the actual Windows vc_reader."""
import ast
from dataclasses import dataclass
from hashlib import sha256
import itertools
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen

import numpy as np
import pytest
from vc_reader.reader import (ZarrReader, ZarrWriter, create_group, create_pyramid_levels,
                              ome_multiscales, open_group, write_pyramid, write_pyramid_level)

SOURCE = Path("vesuvius/src/vesuvius/ink_detection/inference/infer_full3d_tifxyz.py")
TESTS = Path("vc_reader/tests/test_zarr_writer.py")
CT = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0139/"
      "volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr/0/")


def load_functions(path, names, environment):
    nodes = list(ast.parse("from __future__ import annotations").body)
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if getattr(node, "name", "") in names:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
            nodes.append(node)
    scope = dict(environment)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope


def main():
    names = ("test_write_pyramid_reductions", "test_write_pyramid_level_ranges_and_multiscales",
             "test_write_pyramid_level_selects_individual_chunks")
    tests = load_functions(TESTS, names, globals())
    for reduction, expected in (("mean", 6), ("nearest", 7), ("mode", 5)):
        with TemporaryDirectory() as path:
            tests[names[0]](Path(path), reduction, expected)
            print(f"PASS {names[0]}[{reduction}]")
    for name in names[1:]:
        with TemporaryDirectory() as path:
            tests[name](Path(path))
            print(f"PASS {name}")

    names = {"ChunkId", "chunk_bounds", "chunks_overlapping_box", "pyramid_shapes",
             "create_output_zarr", "build_downsample_levels"}
    native = load_functions(SOURCE, names, dict(globals(), DEFAULT_LEVELS=6,
                                               ARRAY_DIMENSIONS=["z", "y", "x"]))
    inputs = []
    for key in ("22/18/23", "27/18/23"):
        with urlopen(CT + key, timeout=40) as response:
            raw = response.read()
        print(f"{key} {sha256(raw).hexdigest()}")
        inputs.append(np.frombuffer(raw, dtype=np.uint8).reshape((128,) * 3))
    with TemporaryDirectory() as path:
        before, after = (Path(path) / directory / "prediction.zarr" for directory in ("dense", "sparse"))
        for root, fresh in ((before, False), (after, True)):
            arrays = native["create_output_zarr"](root, shape_zyx=(768, 128, 128), chunks_zyx=(128,) * 3)
            arrays[0][:128, :, :] = inputs[0]
            arrays[0][640:768, :, :] = inputs[1]
            keys = {native["ChunkId"](z, 0, 0) for z in (0, 5)}
            native["build_downsample_levels"](arrays, level0_written_chunks=keys, workers=2, fresh_output=fresh)

        def contents(root):
            return {file.relative_to(root): file.read_bytes()
                    for file in root.rglob("*") if file.is_file()}

        baseline = contents(before)
        assert contents(after) == baseline
        assert int(open_group(after)["1"][128, 0, 0]) == 0
        print(f"PASS exact sparse/dense output on Windows: {len(baseline)} files across six levels")


if __name__ == "__main__":
    main()
