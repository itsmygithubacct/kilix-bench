"""Adapt the pinned Neuronte trainer for reproducible two-expert experiments.

Training artifacts belong outside the checkout. This keeps its optimizer,
schedule, field sampling, code features and unlabelled regularizer intact.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


PARTS = ("lower", "upper", "left", "right")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--neuronte-home", required=True, type=Path)
    parser.add_argument("--part", required=True, choices=PARTS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-every", type=int, default=1000)
    args = parser.parse_args()
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    home = args.neuronte_home.resolve()
    sys.path.insert(0, str(home / "train"))
    import checkpoint
    import glyphs
    original_charset, original_field = glyphs.charset, glyphs.field
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    name = "braille-" + args.part
    spatial = args.part in ("left", "right")
    bitmask = 0x47 if args.part == "left" else 0xb8
    glyphs.SETS[name] = ["braille"]

    def charset(requested):
        if requested != name:
            return original_charset(requested)
        rows = original_charset("braille")
        if spatial:
            return [row for row in rows if ((ord(row[0]) - 0x2800) & ~bitmask) == 0]
        high = args.part == "upper"
        return [row for row in rows if bool((ord(row[0]) - 0x2800) & 128) == high]

    glyphs.charset = charset
    if spatial:
        def field(shapes, n, pen, tau):
            # The public API maps each half-cell to x in [0,1]. Its training
            # target therefore samples the original field at x/2 (+1/2 for
            # the right). y remains [0,2]. Doubling file tau cancels the
            # half-width factor in coverage's tau * width * samples scale.
            full = original_field(shapes, n, pen, tau / 2).astype(np.float16).astype(np.float32)
            columns = np.arange(n + 1) / 2 + (n / 2 if args.part == "right" else 0)
            low = columns.astype(np.int64)
            high = np.minimum(low + 1, n)
            fraction = columns - low
            return full[:, low] * (1 - fraction) + full[:, high] * fraction
        glyphs.field = field

    experiment = {
        "schema": "kilix-bench.braille-training/v1", "part": args.part,
        "adapter_sha256": source_hash,
        "input_mapping": f"U+2800 | ((scalar-U+2800) & 0x{bitmask:02x})" if spatial else "original scalar",
        "x_domain": [0, .5] if args.part == "left" else [.5, 1] if args.part == "right" else [0, 1],
        "y_domain": [0, 2], "file_tau": .30 if spatial else .15,
        "original_target_tau": .15, "training_steps": args.steps,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else args.device,
        "tf32": False, "seed": args.seed,
    }
    identity = checkpoint.identity
    def bound_identity(arguments, sources):
        value = identity(arguments, sources)
        value["experiment"] = experiment
        return value
    checkpoint.identity = bound_identity
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(home / "train/train.py"), str(args.out), "--set", name,
                "--width", "24", "--steps", str(args.steps), "--seed", str(args.seed),
                "--device", args.device, "--tau", str(experiment["file_tau"]),
                "--score-sizes", "8", "--log-every", str(args.log_every)]
    if args.checkpoint:
        sys.argv += ["--checkpoint", str(args.checkpoint), "--checkpoint-every", "2000"]
    if args.resume:
        sys.argv += ["--resume", str(args.resume)]
    print(json.dumps({"experiment": experiment}), flush=True)
    runpy.run_path(str(home / "train/train.py"), run_name="__main__")
    doc = json.loads(args.out.read_text())
    doc["training"]["experiment"] = experiment
    args.out.write_text(json.dumps(doc) + "\n")


if __name__ == "__main__":
    main()
