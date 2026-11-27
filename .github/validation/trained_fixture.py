"""Prepare one level-0 Paris 4 patch and a verified public EMA checkpoint."""
from hashlib import sha256
from itertools import product
from pathlib import Path
import sys
from urllib.request import urlopen

import numpy as np
from PIL import Image


def fetch(url, output):
    with urlopen(url, timeout=60) as response:
        output.write_bytes(response.read())
    print(f"{sha256(output.read_bytes()).hexdigest()}  {output.name}", flush=True)


def main(root):
    import torch

    mesh = root / "mesh"
    mesh.mkdir(parents=True)
    base = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/"
    mesh_url = base + ("segments/20260604223808-20231210121321_v8/mesh/"
                       "20260604223808-on-20260411134726-2.4um.tifxyz/")
    ct_url = base + "volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"
    coords = []
    for axis in "xyz":
        path = mesh / f"{axis}.tif"
        fetch(mesh_url + path.name, path)
        with Image.open(path) as image:
            coords.append(np.array(image))
    fetch(mesh_url + "meta.json", mesh / "meta.json")
    origin = (308, 100, 160)
    mask = np.ones(coords[0].shape, dtype=bool)
    for axis, index in zip(reversed(coords), origin):
        mask &= (axis >= index*128) & (axis < (index+1)*128)
    nonzero = np.where(mask)
    assert nonzero[0].size, "The selected CT chunk must intersect the published surface"
    lo = tuple(component.min() for component in nonzero)
    hi = tuple(component.max()+1 for component in nonzero)
    region = tuple(slice(a, b) for a, b in zip(lo, hi))
    for axis, values in zip("xyz", coords):
        Image.fromarray(np.where(mask[region], values[region], np.float32(-1))).save(mesh / f"{axis}.tif")
    print(f"Kept {int(mask.sum())} published vertices in output chunk {origin}.")

    ct = root / "ct.zarr"
    (ct / "0").mkdir(parents=True)
    fetch(ct_url + ".zgroup", ct / ".zgroup")
    fetch(ct_url + "0/.zarray", ct / "0" / ".zarray")
    for offset in product(range(2), repeat=3):
        key = "/".join(str(a+b) for a, b in zip(origin, offset))
        path = ct / "0" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        fetch(ct_url + "0/" + key, path)
        assert path.stat().st_size == 128 ** 3
    (mesh / "volume_source.txt").write_text(str(ct.resolve()), encoding="utf-8")

    # Only load tensors and basic objects from the downloaded archive.
    archive = torch.load(root / "release.pth", map_location="cpu", weights_only=True, mmap=True)
    config = archive["config"]
    assert tuple(config["patch_size"]) == (256,) * 3
    torch.save({"config": config, "ema_model": archive["ema_model"]}, root / "ema.pth")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
