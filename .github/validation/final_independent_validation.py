"""Fixtures and measurements for Villa's independent native ink validation."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import resource
import time
from urllib.request import urlopen

import numpy as np
from PIL import Image

S3 = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/"
MESH = S3 + ("PHerc0139/segments/20260317000000-w035_2026031718/mesh/"
             "20260317000000-on-20250728140407-9.362um.tifxyz")
VOLUME = S3 + "PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr"
DENSE = S3 + ("PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/"
              "2/73/32/40")
KEYS = ((22, 18, 23), (27, 18, 23))


def get(url):
    with urlopen(url, timeout=45) as response:
        data = response.read()
    print(f"{sha256(data).hexdigest()}  {url}", flush=True)
    return data


def prepare_ct(root):
    from io import BytesIO
    import torch
    from vesuvius.ink_detection.config import InkConfig
    from vesuvius.ink_detection.models.model import make_model

    mesh = root / "mesh"
    mesh.mkdir(parents=True)
    metadata = get(MESH + "/meta.json")
    coords = []
    for axis in "xyz":
        with Image.open(BytesIO(get(f"{MESH}/{axis}.tif"))) as image:
            coords.append(np.array(image))
    x, y, z = coords
    keep = np.zeros(x.shape, dtype=bool)
    for key in KEYS:
        selected = np.ones(x.shape, dtype=bool)
        for axis, index in zip((z, y, x), key):
            selected &= (axis >= index * 128) & (axis < (index + 1) * 128)
        assert selected.any(), key
        keep |= selected
    for axis, pixels in zip("xyz", coords):
        Image.fromarray(np.where(keep, pixels, np.float32(-1))).save(mesh / f"{axis}.tif")
    (mesh / "meta.json").write_bytes(metadata)
    ct = root / "ct.zarr"
    (ct / "0").mkdir(parents=True)
    (ct / ".zgroup").write_text('{"zarr_format": 2}', encoding="utf-8")
    (ct / "0" / ".zarray").write_bytes(get(VOLUME + "/0/.zarray"))
    for key in KEYS:
        locator = "/".join(map(str, key))
        target = ct / "0" / locator
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = get(VOLUME + "/0/" + locator)
        assert len(raw) == 128 ** 3 and any(raw)
        target.write_bytes(raw)
    (mesh / "volume_source.txt").write_text(str(ct.resolve()), encoding="utf-8")
    config = {
        "out_dir": str(root / "unused"),
        "mode": "full_3d", "model_type": "vesuvius_unet", "in_channels": 1,
        "patch_size": [32, 32, 32], "patch_overlap": 0, "patch_min_labeled_coverage": 0,
        "batch_size": 2, "image_normalization": "none",
        "model_config": {
            "autoconfigure": True, "basic_encoder_block": "BasicBlockD",
            "basic_decoder_block": "ConvBlock", "features_per_stage": [2, 4],
            "n_blocks_per_stage": [1, 1], "n_conv_per_stage_decoder": [1],
            "kernel_sizes": [[3] * 3] * 2, "strides": [[1] * 3, [2] * 3],
            "pool_op_kernel_sizes": [[1] * 3, [2] * 3], "spacing": [1, 1, 1],
        },
        "targets": {"ink": {"out_channels": 1, "activation": "none", "z_projection_mode": "none"}},
        "datasets": [{"segments_path": str(root), "volume_scale": 0, "volume_path": str(ct)}],
    }
    torch.manual_seed(1808)
    model = make_model(InkConfig.from_mapping(config))
    torch.save({"config": config, "model": model.state_dict()}, root / "fixture.pth")
    print(f"Generated a small, seeded, untrained U-Net checkpoint; {int(keep.sum())} real mesh vertices.")


def dense_fixture(path):
    from vesuvius.ink_detection.inference import infer_full3d_tifxyz as native

    voxels = np.frombuffer(get(DENSE), dtype=np.uint8).reshape((128,) * 3)
    assert np.all(voxels)
    arrays = native.create_output_zarr(path, shape_zyx=voxels.shape, chunks_zyx=(16,) * 3)
    arrays[0][...] = voxels
    assert len(arrays[0].stored_chunks()) == 512


def dense_run(path):
    from itertools import product
    from vc_reader.reader import ZarrWriter
    from vesuvius.ink_detection.inference import infer_full3d_tifxyz as native

    arrays = [ZarrWriter.open(path / str(level), auth=None) for level in range(6)]
    keys = {native.ChunkId(*key) for key in product(range(8), repeat=3)}
    import inspect
    options = {"fresh_output": True} if "fresh_output" in inspect.signature(native.build_downsample_levels).parameters else {}
    start = time.perf_counter()
    native.build_downsample_levels(arrays, level0_written_chunks=keys, workers=2, **options)
    print(json.dumps({"seconds": time.perf_counter() - start,
                      "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))


def inspect_output(path):
    from vc_reader.reader import ZarrWriter

    for level in range(6):
        writer = ZarrWriter.open(path / str(level), auth=None)
        stored = writer.stored_chunks()
        assert stored, level
        if level == 0:
            assert set(stored) == set(KEYS)
            lo = tuple(key * chunk for key, chunk in zip(KEYS[0], writer.chunks))
            cut = tuple(slice(start, start + 16) for start in lo)
            values = writer[cut]
            assert np.unique(values).size > 1, "The fixture model output must vary with CT input"
        print(f"level={level} stored={len(stored)}")
    for file in sorted(file for file in path.rglob("*") if file.is_file()):
        print(f"{sha256(file.read_bytes()).hexdigest()}  {file.relative_to(path).as_posix()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare-ct", "dense-prepare", "dense-run", "inspect"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    {"prepare-ct": prepare_ct, "dense-prepare": dense_fixture,
     "dense-run": dense_run, "inspect": inspect_output}[args.mode](args.path)


if __name__ == "__main__":
    main()
