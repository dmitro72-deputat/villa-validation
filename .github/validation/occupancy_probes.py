"""Run differential occupancy probes with only Git, Python and NumPy."""
import ast
from dataclasses import dataclass
import statistics
import subprocess
from time import perf_counter

import numpy as np

SOURCE = "vesuvius/src/vesuvius/ink_detection/inference/infer_full3d_tifxyz.py"
REVISIONS = ("659e2fadb836d44063ed831d85517154ec852f11", "3b31fd9")


def occupancy(revision):
    source = subprocess.check_output(["git", "show", f"{revision}:{SOURCE}"], text=True)
    header = ast.parse("from __future__ import annotations").body
    nodes = [node for node in ast.parse(source).body
             if getattr(node, "name", "") in {"ChunkId", "tifxyz_occupied_chunks"}]
    scope = {"np": np, "dataclass": dataclass}
    exec(compile(ast.Module(body=header + nodes, type_ignores=[]), revision, "exec"), scope)
    return scope["tifxyz_occupied_chunks"]


def outcome(function, x, y, z, valid):
    try:
        chunks = function(x=x, y=y, z=z, valid=valid,
                          array_shape_zyx=(64, 64, 64), chunk_shape_zyx=(8, 8, 8))
    except (ValueError, IndexError, TypeError) as error:
        return type(error).__name__, str(error)
    return "ok", sorted((c.z, c.y, c.x) for c in chunks)


def main():
    before, after = functions = tuple(map(occupancy, REVISIONS))
    rng = np.random.default_rng(1808)
    for index in range(250):
        shape = (515, 513) if index % 25 == 0 else (32, 13)
        dtype = (">f4", "<f8", ">i8", "<i2")[index % 4]
        xyz = rng.integers(-3, 70, size=(*shape, 3)).astype(dtype)
        if index % 5 == 0 and xyz.dtype.kind == "f":
            xyz[1, 1] = np.nan
        xyz = xyz[:, ::-1, :]
        values = (*np.moveaxis(xyz, -1, 0), (xyz >= 0).all(axis=-1))
        assert outcome(before, *values) == outcome(after, *values), index
    print("250 numeric, strided, byte-order and multi-block inputs matched.")

    x = np.broadcast_to(np.arange(4, dtype=np.float32), (2, 4))
    cases = {
        "Broadcast mask": ((x, x, x, np.ones((1, 4), dtype=bool)), ("IndexError", "ok")),
        "Object coordinates": ((x.astype(object),) * 3 + (np.ones((2, 4), dtype=bool),),
                               ("ok", "TypeError")),
    }
    for name, (arguments, expected) in cases.items():
        results = [outcome(function, *arguments) for function in functions]
        assert tuple(result[0] for result in results) == expected
        print(name, results)

    for side, calls in ((2, 4000), (256, 12)):
        x, y = np.indices((side, side), dtype=np.float32)
        values = x % 64, y % 64, (x+y) % 64, np.ones((side, side), dtype=bool)
        trials = [[], []]
        for trial in range(7):
            for index in ((0, 1) if trial % 2 == 0 else (1, 0)):
                start = perf_counter()
                for _ in range(calls):
                    outcome(functions[index], *values)
                trials[index].append((perf_counter() - start) * 1000 / calls)
        print(f"All-valid {side}x{side}, milliseconds min/median/max",
              [(min(times), statistics.median(times), max(times)) for times in trials])


if __name__ == "__main__":
    main()
