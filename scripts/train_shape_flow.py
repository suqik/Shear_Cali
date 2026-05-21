#!/usr/bin/env python
"""Train the conditional shape-flow model from NumPy arrays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shape_flow.utils import ConfigOption, merge_config, validate_training_config


SHAPE_FIELD_NAMES = ("e1_t", "e2_t", "e1", "e2")
COND_FIELD_NAMES = ("hlf", "mag", "snr")

CONFIG_OPTIONS = (
    ConfigOption("data", "paths", "data", "path"),
    ConfigOption("e_true", "paths", "e_true", "path"),
    ConfigOption("e_meas", "paths", "e_meas", "path"),
    ConfigOption("cond", "paths", "cond", "path"),
    ConfigOption("output", "paths", "output", "path"),
    ConfigOption("resume_checkpoint", "paths", "resume_checkpoint", "path"),
    ConfigOption("stop_after_epoch", "training", "stop_after_epoch", "int", 20),
    ConfigOption(
        "maximum_training_epoch",
        "training",
        "maximum_training_epoch",
        "int",
        None,
    ),
    ConfigOption("epochs", "training", "epochs", "int", None),
    ConfigOption("batch_size", "training", "batch_size", "int", 512),
    ConfigOption("val_fraction", "training", "val_fraction", "float", 0.2),
    ConfigOption("learning_rate", "training", "learning_rate", "float", 1.0e-3),
    ConfigOption("weight_decay", "training", "weight_decay", "float", 1.0e-4),
    ConfigOption("grad_clip_norm", "training", "grad_clip_norm", "float", 5.0),
    ConfigOption("seed", "training", "seed", "int", 0),
    ConfigOption("device", "training", "device", "str", None),
    ConfigOption("transforms", "model", "transforms", "int", 5),
    ConfigOption("hidden_features", "model", "hidden_features", "int_list", [128, 128]),
    ConfigOption("bins", "model", "bins", "int", 8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train q(e_meas | e_true, cond) with a conditional Zuko NSF."
    )
    parser.add_argument("--config", type=Path, help="INI configuration file.")
    data_group = parser.add_mutually_exclusive_group()
    data_group.add_argument(
        "--data",
        type=Path,
        help="Structured .npy file containing shape and condition fields.",
    )
    data_group.add_argument(
        "--e-true",
        type=Path,
        help="Path to e_true .npy array. Requires --e-meas and --cond.",
    )
    parser.add_argument("--e-meas", type=Path, help="Path to e_meas .npy array.")
    parser.add_argument("--cond", type=Path, help="Path to cond .npy array.")
    parser.add_argument("--output", type=Path, help="Checkpoint path.")
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Checkpoint to resume model weights and scalers from.",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Stop after this many epochs without a validation-loss drop.",
    )
    parser.add_argument(
        "--maximum-training-epoch",
        type=int,
        help="Hard cap on the number of training epochs.",
    )
    parser.add_argument("--epochs", type=int, help=argparse.SUPPRESS)
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
    args = merge_config(parser.parse_args(), CONFIG_OPTIONS)
    validate_training_config(args)
    return args


def load_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.data is not None:
        return load_training_data_file(args.data)

    if args.e_meas is None or args.cond is None:
        raise ValueError("--e-true requires --e-meas and --cond")
    return np.load(args.e_true), np.load(args.e_meas), np.load(args.cond)


def load_training_data_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loaded = np.load(path, allow_pickle=False)
    try:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            return _training_arrays_from_npz(loaded)
        return _training_arrays_from_structured(loaded, source=str(path))
    finally:
        close = getattr(loaded, "close", None)
        if close is not None:
            close()


def _training_arrays_from_npz(
    data: np.lib.npyio.NpzFile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = set(data.files)
    if {"e_true", "e_meas", "cond"}.issubset(names):
        return (
            np.asarray(data["e_true"], dtype=np.float32),
            np.asarray(data["e_meas"], dtype=np.float32),
            np.asarray(data["cond"], dtype=np.float32),
        )
    return _training_arrays_from_fields(data.__getitem__, names, source="NPZ file")


def _training_arrays_from_structured(
    data: np.ndarray,
    *,
    source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if data.dtype.names is None:
        raise ValueError(
            f"{source} must be a structured array with fields "
            f"{SHAPE_FIELD_NAMES + COND_FIELD_NAMES}"
        )
    names = set(data.dtype.names)
    if {"e_true", "e_meas", "cond"}.issubset(names):
        return (
            np.asarray(data["e_true"], dtype=np.float32),
            np.asarray(data["e_meas"], dtype=np.float32),
            np.asarray(data["cond"], dtype=np.float32),
        )
    return _training_arrays_from_fields(data.__getitem__, names, source=source)


def _training_arrays_from_fields(
    get_column,
    names: set[str],
    *,
    source: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = SHAPE_FIELD_NAMES + COND_FIELD_NAMES
    missing = [name for name in required if name not in names]
    if missing:
        raise KeyError(f"{source} is missing fields/arrays: {missing}")
    e_true = _column_stack(get_column, ("e1_t", "e2_t"))
    e_meas = _column_stack(get_column, ("e1", "e2"))
    cond = _column_stack(get_column, COND_FIELD_NAMES)
    return e_true, e_meas, cond


def _column_stack(get_column, names: tuple[str, ...]) -> np.ndarray:
    columns = [np.asarray(get_column(name), dtype=np.float32) for name in names]
    return np.column_stack(columns)


def progress(epoch: int, metrics: dict[str, float]) -> None:
    print(
        f"epoch={epoch:04d} "
        f"train_nll={metrics['train_nll']:.6f} "
        f"val_nll={metrics['val_nll']:.6f}",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    from shape_flow import (
        TrainingConfig,
        build_shape_flow,
        load_flow_checkpoint,
        prepare_training_data,
        resolve_device,
        save_flow_checkpoint,
        train_model,
    )

    # 1. Data loading.
    e_true, e_meas, cond = load_arrays(args)

    config = TrainingConfig(
        val_fraction=args.val_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        stop_after_epoch=args.stop_after_epoch,
        maximum_training_epoch=args.maximum_training_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        transforms=args.transforms,
        hidden_features=tuple(args.hidden_features),
        bins=args.bins,
        device=args.device,
        checkpoint_path=args.output,
        resume_checkpoint_path=args.resume_checkpoint,
    )

    # 2. Data preparation and scaler fitting/loading.
    device = resolve_device(config)
    resumed = None
    if args.resume_checkpoint is not None:
        resumed = load_flow_checkpoint(args.resume_checkpoint, device=device)

    prepared = prepare_training_data(
        e_true,
        e_meas,
        cond,
        config=config,
        target_scaler=resumed.target_scaler if resumed is not None else None,
        context_scaler=resumed.context_scaler if resumed is not None else None,
        device=device,
    )

    # 3. Neural density estimator construction/loading.
    if resumed is None:
        model = build_shape_flow(
            prepared.train_arrays.context_features,
            config=config,
            device=device,
        )
    else:
        model = resumed.model
        if model.config.context_features != prepared.train_arrays.context_features:
            raise ValueError(
                "Resume checkpoint context dimension does not match data: "
                f"{model.config.context_features} != "
                f"{prepared.train_arrays.context_features}"
            )

    # 4. Training.
    trained = train_model(
        model,
        prepared.train_loader,
        prepared.val_loader,
        config=config,
        device=device,
        progress_callback=progress,
    )

    # 5. Flow checkpoint saving.
    save_flow_checkpoint(
        trained.model,
        prepared.target_scaler,
        prepared.context_scaler,
        args.output,
        history=trained.history,
        best_epoch=trained.best_epoch,
        best_val_nll=trained.best_val_nll,
        config=config,
    )

    print(f"saved={args.output}")
    print(f"trained_epochs={len(trained.history['train_nll'])}")
    print(
        f"best_epoch={trained.best_epoch} "
        f"best_val_nll={trained.best_val_nll:.6f}"
    )


if __name__ == "__main__":
    main()
