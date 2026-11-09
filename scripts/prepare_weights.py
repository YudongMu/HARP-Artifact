#!/usr/bin/env python3
"""Download and verify the upstream DiT or Wan files used by the evaluation runs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import quote


MANIFESTS = Path(__file__).resolve().parents[1] / "manifests"
DIT_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/DiT/models/"
DIT_VAE_URL = (
    "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/"
    "31f26fdeee1355a5c34592e401dd41e45d25a493/"
)


def weight_plan(model):
    if model == "dit":
        for folder, manifest, url in (
            ("dit", "dit-checkpoint.sha256", DIT_CHECKPOINT_URL),
            ("dit-vae", "dit-vae.sha256", DIT_VAE_URL),
        ):
            for line in (MANIFESTS / manifest).read_text().splitlines():
                digest, path = line.split(maxsplit=1)
                filename = Path(path).name
                yield Path(folder) / filename, url + filename, None, "sha256", digest
        return

    plan = json.loads((MANIFESTS / "wan-download-plan.json").read_text())
    base = f"https://huggingface.co/{plan['repo']}/resolve/{plan['revision']}/"
    for row in plan["files"]:
        algorithm = "sha256" if row["sha256"] else "sha1"
        yield (
            Path("wan2.1-t2v-14b") / row["path"],
            base + quote(row["path"]),
            row["bytes"],
            algorithm,
            row["sha256"] or row["git_blob_sha1"],
        )


def verify(path, size, algorithm, expected):
    actual_size = path.stat().st_size
    if size is not None and actual_size != size:
        raise ValueError(f"{path}: expected {size} bytes, found {actual_size}")
    digest = hashlib.new(algorithm)
    if algorithm == "sha1":
        digest.update(f"blob {actual_size}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"{path}: {algorithm} mismatch")


def prepare(model, assets, verify_only):
    weights = assets / "weights"
    for name, url, size, algorithm, expected in weight_plan(model):
        target = weights / name
        if not target.exists() and not verify_only:
            # Store partial transfers beside the asset directory, outside Wan's exact inventory.
            partial = assets / ".downloads" / name
            partial.parent.mkdir(parents=True, exist_ok=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading {name}", flush=True)
            if size is None or not partial.exists() or partial.stat().st_size != size:
                subprocess.run(
                    ["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "30",
                     "--continue-at", "-", "--output", str(partial), url],
                    check=True,
                )
            verify(partial, size, algorithm, expected)
            partial.replace(target)
        else:
            verify(target, size, algorithm, expected)
        print(f"Verified {name}", flush=True)

    if model == "wan":
        expected = {name for name, *_ in weight_plan(model)}
        actual = {path.relative_to(weights) for path in (weights / "wan2.1-t2v-14b").rglob("*")
                  if path.is_file() or path.is_symlink()}
        if actual != expected:
            raise ValueError("Unexpected files in the Wan checkpoint directory: "
                             + ", ".join(map(str, sorted(actual - expected))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("dit", "wan"))
    parser.add_argument("--assets", type=Path, required=True, help="host asset directory")
    parser.add_argument("--verify-only", action="store_true", help="check existing files without downloading")
    args = parser.parse_args()
    try:
        prepare(args.model, args.assets.resolve(), args.verify_only)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
