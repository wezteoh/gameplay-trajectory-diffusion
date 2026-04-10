"""Pregenerate observation masks [N, T, A] for a trajectory .npy.

Expects the same (N, T, A, C) layout as training data. Edit the config block below
(paths, seed, ``MASKING``). Uses ``src.data.trajectory_masks`` mixture/even/agent/
hybrid logic. Per-row RNG: ``manual_seed(BASE_SEED + row_index)`` for stable
``val_mask_path`` alignment. CLI overrides input/output/seed only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.trajectory_masks import parse_masking_config  # noqa: E402

# ---------------------------------------------------------------------------
# Edit these defaults (CLI can override input/output/seed only).
# ---------------------------------------------------------------------------

# Relative paths resolve from the repository root unless absolute.
INPUT_NPY = "data/nba_test.npy"
OUTPUT_NPY = "tmp/val_masks_pregen.npy"

# Per-row generator seed: row i uses manual_seed(BASE_SEED + i).
BASE_SEED = 42

# Same shape as ``data.params.masking`` in Hydra (``val_mask_path`` is ignored here).
# MASKING: dict[str, Any] = {
#     "train": {
#         "mixture": [
#             {
#                 "weight": 1.0,
#                 "even": {"mode": "fixed", "visible_prefix_len": 10},
#             },
#         ]
#     },
# }

# Example: mixture (uncomment and adjust).
MASKING = {
    "train": {
        "mixture": [
            {
                "weight": 0.25,
                "even": {"mode": "random", "prefix_min": 1, "prefix_max": 20},
            },
            {
                "weight": 0.5,
                "agent": {
                    "mode": "random",
                    "n_masked_min": 5,
                    "n_masked_max": 11,
                },
            },
            {
                "weight": 0.25,
                "hybrid": {
                    "combine": "union",
                    "a": {
                        "even": {"mode": "random", "prefix_min": 1, "prefix_max": 20},
                    },
                    "b": {
                        "agent": {
                            "mode": "random",
                            "n_masked_min": 5,
                            "n_masked_max": 11,
                        },
                    },
                },
            },
        ]
    },
}


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p.resolve()
    return (_REPO_ROOT / p).resolve()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=str,
        default=INPUT_NPY,
        help="Trajectory .npy (N, T, A, C); C=2. Default: script INPUT_NPY.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=OUTPUT_NPY,
        help="Output .npy (N, T, A) float32 {{0,1}}. Default: script OUTPUT_NPY.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=BASE_SEED,
        help="Base seed; row i uses seed + i. Default: script BASE_SEED.",
    )
    args = p.parse_args()

    inp = _resolve_path(args.input)
    out = _resolve_path(args.output)
    if not inp.is_file():
        raise FileNotFoundError(f"Input not found: {inp}")

    arr = np.load(inp, mmap_mode="r")
    if arr.ndim != 4:
        raise ValueError(f"Expected (N,T,A,C), got shape {arr.shape}")
    n, t, a, c = arr.shape

    train_masking, _ = parse_masking_config(dict(MASKING))
    gen = torch.Generator()
    out_masks = np.empty((n, t, a), dtype=np.float32)

    for i in range(n):
        gen.manual_seed(int(args.seed) + i)
        m = train_masking.sample_mask(t, a, torch.float32, gen)
        out_masks[i] = m.cpu().numpy()

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, out_masks)
    print(f"Wrote {out} shape={out_masks.shape} (from {inp})")


if __name__ == "__main__":
    main()
