#!/usr/bin/env bash
set -euo pipefail
variant="$1"
checkout="$2"
work="$VILLA_RELEASE_WORK"
out="$work/results/$variant"
mkdir -p "$out"
export PYTHONPATH="$checkout/vesuvius/src"
/usr/bin/time -f "$variant %e s %M KiB RSS" timeout 7m \
  python -m vesuvius.ink_detection.inference.infer_full3d_tifxyz \
    "$work/mesh" "$work/ema.pth" "$out/prediction.zarr" \
    --device cpu --resolution 0 --overlap 0 --chunk-halo 0 --write-region occupied \
    --no-compile --batch-size 1 --num-workers 0 --downsample-workers 1 2>&1 | tee "$out/run.log"
grep -E 'target_chunks=1 inference_patches=1' "$out/run.log"
