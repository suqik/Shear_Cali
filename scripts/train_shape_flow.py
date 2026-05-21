#!/usr/bin/env python
"""Train the conditional shape-flow likelihood from NumPy arrays."""

from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train q(e_meas | e_true, cond) with a conditional Zuko NSF."
    )
    parser.add_argument("--config", type=Path, help="INI configuration file.")
    data_group = parser.add_mutually_exclusive_group()
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
    parser.add_argument("--output", type=Path, help="Checkpoint path.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--val-fraction", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--transforms", type=int)
    parser.add_argument("--hidden-features", type=int, nargs="+")
    parser.add_argument("--bins", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default=None, help="Device, e.g. cpu or cuda.")
    return merge_config(parser.parse_args())


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    parser = configparser.ConfigParser()
    config_dir = Path.cwd()
    if args.config is not None:
        read_files = parser.read(args.config)
        if not read_files:
            raise FileNotFoundError(f"Could not read config file: {args.config}")
        config_dir = args.config.resolve().parent

    args.data = _path_option(args.data, parser, "paths", "data", config_dir)
    args.e_true = _path_option(args.e_true, parser, "paths", "e_true", config_dir)
    args.e_meas = _path_option(args.e_meas, parser, "paths", "e_meas", config_dir)
    args.cond = _path_option(args.cond, parser, "paths", "cond", config_dir)
    args.output = _path_option(args.output, parser, "paths", "output", config_dir)

    args.epochs = _int_option(args.epochs, parser, "training", "epochs", 100)
    args.batch_size = _int_option(args.batch_size, parser, "training", "batch_size", 512)
    args.val_fraction = _float_option(
        args.val_fraction,
        parser,
        "training",
        "val_fraction",
        0.2,
    )
    args.learning_rate = _float_option(
        args.learning_rate,
        parser,
        "training",
        "learning_rate",
        1.0e-3,
    )
    args.weight_decay = _float_option(
        args.weight_decay,
        parser,
        "training",
        "weight_decay",
        1.0e-4,
    )
    args.grad_clip_norm = _float_option(
        args.grad_clip_norm,
        parser,
        "training",
        "grad_clip_norm",
        5.0,
    )
    args.seed = _int_option(args.seed, parser, "training", "seed", 0)
    args.device = _str_option(args.device, parser, "training", "device", None)

    args.transforms = _int_option(args.transforms, parser, "model", "transforms", 5)
    args.hidden_features = _int_list_option(
        args.hidden_features,
        parser,
        "model",
        "hidden_features",
        [128, 128],
    )
    args.bins = _int_option(args.bins, parser, "model", "bins", 8)

    if args.data is not None and args.e_true is not None:
        raise ValueError("Use either data or e_true/e_meas/cond inputs, not both")
    if args.data is None and args.e_true is None:
        raise ValueError("Provide data in [paths] or with --data/--e-true")
    if args.output is None:
        raise ValueError("Provide output in [paths] or with --output")
    return args


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


def _path_option(
    cli_value: Path | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    base_dir: Path,
) -> Path | None:
    if cli_value is not None:
        return cli_value
    if not parser.has_option(section, option):
        return None
    value = Path(parser.get(section, option))
    return value if value.is_absolute() else (base_dir / value).resolve()


def _str_option(
    cli_value: str | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    default: str | None,
) -> str | None:
    if cli_value is not None:
        return cli_value
    if parser.has_option(section, option):
        value = parser.get(section, option).strip()
        return value or None
    return default


def _int_option(
    cli_value: int | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    default: int,
) -> int:
    if cli_value is not None:
        return cli_value
    return parser.getint(section, option, fallback=default)


def _float_option(
    cli_value: float | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    default: float,
) -> float:
    if cli_value is not None:
        return cli_value
    return parser.getfloat(section, option, fallback=default)


def _int_list_option(
    cli_value: list[int] | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    default: list[int],
) -> list[int]:
    if cli_value is not None:
        return cli_value
    if not parser.has_option(section, option):
        return default
    raw = parser.get(section, option)
    return [int(item.strip()) for item in raw.replace(",", " ").split()]


def progress(epoch: int, metrics: dict[str, float]) -> None:
    print(
        f"epoch={epoch:04d} "
        f"train_nll={metrics['train_nll']:.6f} "
        f"val_nll={metrics['val_nll']:.6f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    from shape_flow import TrainingConfig, load_likelihood, train_shape_flow

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
