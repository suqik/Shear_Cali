#!/usr/bin/env python
"""Train the conditional shape-flow likelihood from NumPy arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shape_flow import TrainingConfig, load_likelihood, train_shape_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train q(e_meas | e_true, cond) with a conditional Zuko NSF."
    )
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument(
        "--data",
        type=Path,
        help="NPZ file containing arrays named e_true, e_meas, and cond.",
    )
    data_group.add_argument(
        "--e-true",
        type=Path,
        help="Path to e_true .npy array. Requires --e-meas and --cond.",
    )
    parser.add_argument("--e-meas", type=Path, help="Path to e_meas .npy array.")
    parser.add_argument("--cond", type=Path, help="Path to cond .npy array.")
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--transforms", type=int, default=5)
    parser.add_argument("--hidden-features", type=int, nargs="+", default=[128, 128])
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Device, e.g. cpu or cuda.")
    return parser.parse_args()


def load_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.data is not None:
        with np.load(args.data) as data:
            missing = {"e_true", "e_meas", "cond"} - set(data.files)
            if missing:
                raise KeyError(f"NPZ file is missing arrays: {sorted(missing)}")
            return data["e_true"], data["e_meas"], data["cond"]

    if args.e_meas is None or args.cond is None:
        raise ValueError("--e-true requires --e-meas and --cond")
    return np.load(args.e_true), np.load(args.e_meas), np.load(args.cond)


def progress(epoch: int, metrics: dict[str, float]) -> None:
    print(
        f"epoch={epoch:04d} "
        f"train_nll={metrics['train_nll']:.6f} "
        f"val_nll={metrics['val_nll']:.6f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    e_true, e_meas, cond = load_arrays(args)
    config = TrainingConfig(
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        transforms=args.transforms,
        hidden_features=tuple(args.hidden_features),
        bins=args.bins,
        device=args.device,
        checkpoint_path=args.output,
    )
    result = train_shape_flow(
        e_true,
        e_meas,
        cond,
        config=config,
        progress_callback=progress,
    )
    reloaded = load_likelihood(args.output, map_location=args.device or "cpu")
    n_example = min(5, len(e_meas))
    example_log_prob = reloaded.log_prob(
        e_meas[:n_example],
        e_true[:n_example],
        cond[:n_example],
    )

    print(f"saved={args.output}")
    print(
        f"best_epoch={result.best_epoch} "
        f"best_val_nll={result.best_val_nll:.6f}"
    )
    print(
        "example_physical_log_prob="
        + np.array2string(example_log_prob.numpy(), precision=6)
    )


if __name__ == "__main__":
    main()
