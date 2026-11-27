"""Benchmark Villa's unmodified planner functions from a source file."""
import argparse
import ast
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image


def planner_from_source(source):
    tree = ast.parse(source.read_text())
    names = {
        "ChunkId", "PatchSpec", "NativePlan", "_validate_zyx_shape",
        "chunk_grid_shape", "tifxyz_occupied_chunks", "expand_chunks",
        "_tifxyz_occupied_chunks_indexed",
        "chunk_bounds", "chunks_overlapping_box", "_global_patch_starts_for_interval",
        "build_patch_plan_for_chunks", "compute_chunk_contribution_counts",
        "build_native_plan",
    }
    body = [ast.ImportFrom("__future__", [ast.alias("annotations")], 0)]
    body += [node for node in tree.body if getattr(node, "name", "") in names]
    scope = dict(np=np, dataclass=dataclass, math=math, itertools=itertools)
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(source), "exec"), scope)
    return scope


def input_from_files(path, metadata, coordinate_scale):
    with metadata.open() as file:
        zarr = json.load(file)
    with Image.open(path / "x.tif") as image:
        width, height = image.size
    xyz = np.empty((height, width, 3), dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    for axis, channel in enumerate("xyz"):
        with Image.open(path / f"{channel}.tif") as image:
            coords = np.asarray(image)
        valid &= (coords >= 0) & np.isfinite(coords)
        xyz[..., axis] = coords * coordinate_scale
    return dict(x=xyz[..., 0], y=xyz[..., 1], z=xyz[..., 2], valid=valid,
                array_shape_zyx=tuple(zarr["shape"]), chunk_shape_zyx=tuple(zarr["chunks"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("surface", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("--coordinate-scale", type=float, default=1)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    scope = planner_from_source(args.source)
    kwargs = input_from_files(args.surface, args.metadata, args.coordinate_scale)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    if args.plan:
        plan = scope["build_native_plan"](
            **kwargs, volume_path="public-ct-metadata", patch_size_zyx=(128, 128, 128),
            overlap=0.5, chunk_halo=1, write_region="expanded", max_target_chunks=None
        )
        chunks = plan.occupied_chunks
        signature_input = [sorted(chunks), sorted(plan.expanded_chunks), sorted(plan.target_chunks),
                           plan.patches, sorted(plan.contribution_counts.items()), plan.stride_zyx]
    else:
        plan = None
        chunks = scope["tifxyz_occupied_chunks"](**kwargs)
        signature_input = sorted(chunks)
    elapsed = time.perf_counter() - start
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    digest = hashlib.sha256(repr(signature_input).encode()).hexdigest()
    print(json.dumps(dict(seconds=elapsed, rss_before_kib=rss_before, peak_rss_kib=rss_after,
                          grid=kwargs["valid"].shape, valid=int(np.count_nonzero(kwargs["valid"])),
                          array_shape=kwargs["array_shape_zyx"], chunk_shape=kwargs["chunk_shape_zyx"],
                          occupied=len(chunks), patches=len(plan.patches) if plan else None,
                          signature=digest)))


if __name__ == "__main__":
    main()
