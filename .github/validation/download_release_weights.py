"""Fetch the pinned public checkpoint and verify its SHA-256."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import file_digest
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

URL = ("https://huggingface.co/scrollprize/ink_3d_dino_guided/resolve/"
       "73a79525466037432191284dfa237baf830c49ec/ckpt_78k_fullsup.pth?download=true")
SIZE = 1_703_138_075
SHA256 = "5a148c2c1bb730bfa683f2b3e3cdfc2000424003605e42300b527ff90118b303"
SLICE = 64 * 1024 * 1024


def matches(path):
    return path.is_file() and path.stat().st_size == SIZE and digest(path) == SHA256


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def curl(*args):
    return subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--location", "--connect-timeout",
         "12", "--speed-time", "20", "--speed-limit", "262144", *args],
        check=False,
    ).returncode


def fetch_slice(request):
    index, directory = request
    first = index * SLICE
    last = min(SIZE, first + SLICE) - 1
    path = directory / str(index)
    headers = directory / f"{index}.headers"
    for attempt in range(2):
        status = curl("--max-time", "105", "--range", f"{first}-{last}", "--dump-header",
                      str(headers), "--output", str(path), f"{URL}&slice={index}&try={attempt}")
        if status == 0 and path.stat().st_size == last - first + 1:
            received = re.findall(r"^content-range:\s*bytes (\d+-\d+/\d+)",
                                  headers.read_text(errors="replace"), re.I | re.M)
            if received and received[-1] == f"{first}-{last}/{SIZE}":
                print(f"Verified byte range {first}-{last}", flush=True)
                return path
    raise RuntimeError(f"Could not fetch byte range {first}-{last}")


def main(path):
    if curl("--max-time", "105", "--output", str(path), URL) != 0 or not matches(path):
        print("Direct transfer failed verification. Trying bounded HTTP byte ranges.", flush=True)
        path.unlink(missing_ok=True)
        with TemporaryDirectory(dir=path.parent) as temp:
            directory = Path(temp)
            requests = ((index, directory) for index in range((SIZE + SLICE - 1) // SLICE))
            with ThreadPoolExecutor(max_workers=4) as pool, path.open("wb") as output:
                for part in pool.map(fetch_slice, requests):
                    with part.open("rb") as stream:
                        shutil.copyfileobj(stream, output)
                    part.unlink()
    if not matches(path):
        raise RuntimeError("Public checkpoint size or SHA-256 did not match")
    print(f"{SHA256}  ckpt_78k_fullsup.pth ({SIZE} bytes)", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
