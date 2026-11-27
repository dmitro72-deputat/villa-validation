"""Time production native pyramid geometry on a real 128^3 CT cube."""
import argparse
from collections import Counter
from itertools import product
import json
import math
from pathlib import Path
import resource
import time
from urllib.request import urlopen

import numpy as np
from vc_reader.reader import ZarrWriter

from bench_pyramid import load_native

CT = ("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/"
      "volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2/73/32/40")


def points(layout):
    grid = set(product(range(8), repeat=3))
    if layout == "full":
        return grid
    if layout == "half":
        keep = {key for key in grid if key[0] % 2 == 0}
    elif layout == "quarter-distributed":
        keep = {key for key in grid if key[0] % 2 == 0 and key[1] % 2 == 0}
    elif layout == "quarter-clustered":
        keep = {key for key in grid if key[0] < 2}
    elif layout == "corners":
        return set(product((0, 7), repeat=3))
    else:
        raise ValueError(layout)
    keep.remove(max(keep))
    keep.add((7, 7, 7))
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("source", type=Path)
    parser.add_argument("store", type=Path)
    parser.add_argument("layout")
    args = parser.parse_args()

    native = load_native(args.source)
    keys = points(args.layout)
    if args.mode == "prepare":
        local = args.store.parent / "dense-ct.bin"
        if not local.exists():
            with urlopen(CT, timeout=45) as response:
                local.write_bytes(response.read())
        data = np.frombuffer(local.read_bytes(), dtype=np.uint8).reshape((128,) * 3).copy()
        assert data.min() > 0
        for z, y, x in points("full") - keys:
            data[z*16:z*16+16, y*16:y*16+16, x*16:x*16+16] = 0
        arrays = native["create_output_zarr"](args.store, shape_zyx=data.shape, chunks_zyx=(16,) * 3)
        arrays[0][...] = data
        assert set(arrays[0].stored_chunks()) == keys
        return

    import inspect

    arrays = [ZarrWriter.open(args.store / str(level), auth=None) for level in range(6)]
    build = native["build_downsample_levels"]
    options = {}
    if "fresh_output" in inspect.signature(build).parameters:
        options["fresh_output"] = True
    calls = Counter()
    writer = native["write_pyramid_level"]

    def record(root, level, **kwargs):
        if "chunk_indices" in kwargs:
            calls["explicit_levels"] += 1
            calls["tasks"] += len(kwargs["chunk_indices"])
        else:
            calls["range_levels"] += 1
            lo, hi = kwargs["chunk_range"]
            calls["tasks"] += math.prod(b-a for a, b in zip(lo, hi))
        return writer(root, level, **kwargs)

    native["write_pyramid_level"] = record
    start = time.perf_counter()
    build(arrays, level0_written_chunks={native["ChunkId"](*key) for key in keys}, workers=2, **options)
    print(json.dumps({"seconds": time.perf_counter()-start, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "coverage": len(keys)/512, **calls}))


if __name__ == "__main__":
    main()
