"""Measure the deployed DiT dense Transformer trajectory on H100."""

import argparse
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
from statistics import median
import sys


def summarize(warmups, measurements):
    if (len(warmups) != 5 or len(measurements) != 20
            or any(not isfinite(ms) or ms <= 0 for ms in [*warmups, *measurements])):
        raise ValueError("Expected five warmups and twenty positive finite timings")
    return {"warmups_ms": warmups, "measurements_ms": measurements,
            "median_ms": median(measurements)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Choose a new output file")

    import torch
    import diffusers
    from quality.dit import denoise, initial_noise, make_scheduler, pilot_items
    from harp.adapters.dit import DiTAdapter
    from harp.runtime import HarpRuntime

    root = Path(__file__).resolve().parents[2]
    upstream = root / "src" / "dit-original"
    sys.path.insert(0, str(upstream))
    from models import DiT_models

    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name():
        raise RuntimeError("Expose one H100 to the benchmark")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    checkpoint = root / "assets" / "weights" / "dit" / "DiT-XL-2-512x512.pt"
    digest = sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    checkpoint_sha = digest.hexdigest()

    model = DiT_models["DiT-XL/2"](input_size=64, num_classes=1000).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.to(torch.bfloat16)
    runtime = HarpRuntime("dense", 32, 32)
    steps = 50
    forward = model.forward
    calls = 0
    first = torch.cuda.Event(enable_timing=True)
    last = torch.cuda.Event(enable_timing=True)

    def mark_forward(*model_args, **kwargs):
        nonlocal calls
        if calls == 0:
            first.record()
        calls += 1
        return forward(*model_args, **kwargs)

    def mark_residual(_block, _inputs, _output):
        if calls == steps:
            last.record()

    model.forward = mark_forward
    hook = model.blocks[-1].register_forward_hook(mark_residual)
    elapsed = []
    inputs = pilot_items(25, seed_base=425000)
    try:
        with torch.inference_mode(), DiTAdapter(model, runtime):
            for item in inputs:
                calls = 0
                noise = initial_noise([item["seed"]], device="cuda")
                torch.cuda.synchronize()
                denoise(model, make_scheduler(), noise, [item["class_id"]],
                        steps=steps, runtime=runtime)
                if calls != steps:
                    raise RuntimeError(f"Expected {steps} denoising forwards, got {calls}")
                last.synchronize()
                elapsed.append(first.elapsed_time(last))
                print(f"trajectory {len(elapsed)}/25: {elapsed[-1]:.3f} ms", flush=True)
    finally:
        hook.remove()
        model.forward = forward
    report = {
        **summarize(elapsed[:5], elapsed[5:]),
        "status": "complete", "workload": "dit-xl2-512", "precision": "BF16",
        "steps": steps, "request_batch": 1, "cfg_lanes": 2, "cfg": 4.0,
        "attention": "PyTorch FlashAttention SDPA", "tf32": False, "cuda_graph": False,
        "boundary": "first DiT forward entry through last block residual of last DiT forward; includes intervening work and launch gaps",
        "scope": "development H100 baseline with deployed scheduler; no profiler, text encoder or VAE",
        "gpu": torch.cuda.get_device_name(), "driver": os.environ.get("HARP_DRIVER"),
        "image_id": os.environ.get("HARP_IMAGE_ID"), "torch": torch.__version__,
        "cuda": torch.version.cuda, "diffusers": diffusers.__version__,
        "checkpoint_sha256": checkpoint_sha, "inputs_in_measurement_order": inputs,
        "source_sha256": {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest()
                          for path in (Path(__file__), upstream / "models.py", root / "quality/dit.py",
                                       root / "harp/adapters/dit.py")},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"median: {report['median_ms']:.3f} ms")


if __name__ == "__main__":
    main()


