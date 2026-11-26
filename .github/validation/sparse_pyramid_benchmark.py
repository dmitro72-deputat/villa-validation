"""Compare Villa native ink pyramids on a fixed subset of published PHerc. Paris. 4."""
import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import re
import resource
import time
from urllib.request import urlopen

from vc_reader.reader import create_group, ome_multiscales, open_group, open_zarr, write_pyramid_level

BASE = (
    "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/"
    "representations/predictions/ink-3d/20260411134726-ink3d-20260428123845-v3-78k-fullsup.zarr/2/"
)
KEYS = ("8/10/10", "37/10/11", "37/10/12", "73/14/11", "73/15/10")


def load_native(source):
    names = {
        "ChunkId", "chunk_bounds", "chunks_overlapping_box", "pyramid_shapes",
        "create_output_zarr", "build_downsample_levels",
    }
    header = ast.parse("from __future__ import annotations").body
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if getattr(node, "name", "") in names]
    import itertools
    import numpy as np

    scope = {
        "dataclass": dataclass, "Path": Path, "DEFAULT_LEVELS": 6,
        "ARRAY_DIMENSIONS": ["z", "y", "x"], "np": np, "itertools": itertools,
        "create_group": create_group, "ome_multiscales": ome_multiscales,
        "write_pyramid_level": write_pyramid_level,
    }
    # The AST keeps the measured production functions unchanged and avoids importing Torch.
    exec(compile(ast.Module(body=header + nodes, type_ignores=[]), str(source), "exec"), scope)
    return scope


def prepare(native, path):
    published = path.parent / "published"
    for key in (".zarray", *KEYS):
        local = published / key
        if not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(BASE + key, timeout=60) as response:
                local.write_bytes(response.read())
        print(f"input {key} {hashlib.sha256(local.read_bytes()).hexdigest()}", flush=True)
    source = open_zarr(published)
    shape, chunks = tuple(source.shape), tuple(source.chunks)
    array = native["create_output_zarr"](path, shape_zyx=shape, chunks_zyx=chunks)[0]
    for key in KEYS:
        index = tuple(map(int, key.split("/")))
        bounds = tuple(slice(i*c, min((i+1)*c, n)) for i, c, n in zip(index, chunks, shape))
        pixels = source[bounds]
        assert pixels.any(), key
        array[bounds] = pixels
    assert len(array.stored_chunks()) == len(KEYS)


def proc_io():
    return {key: int(value) for key, value in
            (line.split(":") for line in Path("/proc/self/io").read_text().splitlines())}


def run(native, path, workers):
    arrays = [open_group(path)[str(level)] for level in range(6)]
    sources = {native["ChunkId"](*map(int, key.split("/"))) for key in KEYS}
    build = native["build_downsample_levels"]
    options = {}
    if "fresh_output" in inspect.signature(build).parameters:
        options["fresh_output"] = True
    before = proc_io()
    start = time.perf_counter()
    build(arrays, level0_written_chunks=sources, workers=workers, **options)
    seconds = time.perf_counter() - start
    peak_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    after = proc_io()
    print(json.dumps({"seconds": seconds, "peak_rss_kib": peak_kib,
                      "io": {key: after[key] - value for key, value in before.items()}}))


def manifest(path):
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        print(hashlib.sha256(file.read_bytes()).hexdigest(), file.relative_to(path))


def trace_summary(path, trace_prefix):
    counts = Counter()
    bytes_done = Counter()
    failures = Counter()
    root = str(path.resolve())
    for trace in trace_prefix.parent.glob(trace_prefix.name + ".*"):
        for line in trace.read_text(errors="replace").splitlines():
            if root not in line:
                continue
            op = line.split("(", 1)[0]
            counts[op] += 1
            if " = -1 ENOENT" in line:
                failures[op] += 1
            if op in {"read", "pread64", "write", "pwrite64"}:
                result = re.search(r" = (\d+)", line)
                if result:
                    bytes_done[op] += int(result[1])
    print(json.dumps({"dataset_calls": dict(counts), "dataset_enoent": dict(failures),
                      "dataset_payload_bytes": dict(bytes_done)}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run", "manifest", "trace"])
    parser.add_argument("source", type=Path)
    parser.add_argument("path", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--trace-prefix", type=Path)
    args = parser.parse_args()
    if args.mode == "manifest":
        manifest(args.path)
    elif args.mode == "trace":
        trace_summary(args.path, args.trace_prefix)
    else:
        native = load_native(args.source)
        if args.mode == "prepare":
            prepare(native, args.path)
        else:
            run(native, args.path, args.workers)


if __name__ == "__main__":
    main()
