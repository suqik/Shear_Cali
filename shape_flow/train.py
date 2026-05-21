"""Training utilities for the conditional shape normalizing flow."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from .data import ShapeArrays, ShapeDataset, split_arrays
from .model import ConditionalShapeFlow, FlowConfig
from .scaling import Standardizer

ProgressCallback = Callable[[int, dict[str, float]], None]


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainingConfig:
    val_fraction: float = 0.2
    seed: int = 0
    batch_size: int = 512
    stop_after_epoch: int = 20
    maximum_training_epoch: int = 100
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 5.0
    transforms: int = 5
    hidden_features: tuple[int, ...] = (128, 128)
    bins: int = 8
    device: str | torch.device | None = None
    checkpoint_path: str | Path | None = None
    resume_checkpoint_path: str | Path | None = None
    epochs: int | None = None

    def __post_init__(self) -> None:
        if self.epochs is not None:
            if self.epochs < 1:
                raise ValueError("epochs must be positive")
            self.maximum_training_epoch = self.epochs
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.stop_after_epoch < 1:
            raise ValueError("stop_after_epoch must be positive")
        if self.maximum_training_epoch < 1:
            raise ValueError("maximum_training_epoch must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        self.hidden_features = tuple(self.hidden_features)


@dataclass
class TrainingResult:
    model: ConditionalShapeFlow
    target_scaler: Standardizer
    context_scaler: Standardizer
    history: dict[str, list[float]]
    best_epoch: int
    best_val_nll: float
    train_arrays: ShapeArrays = field(repr=False)
    val_arrays: ShapeArrays = field(repr=False)


@dataclass
class PreparedTrainingData:
    """Train/validation arrays, scalers, and loaders for flow training."""

    train_arrays: ShapeArrays = field(repr=False)
    val_arrays: ShapeArrays = field(repr=False)
    target_scaler: Standardizer
    context_scaler: Standardizer
    train_loader: DataLoader[tuple[Tensor, Tensor]] = field(repr=False)
    val_loader: DataLoader[tuple[Tensor, Tensor]] = field(repr=False)


@dataclass
class ModelTrainingResult:
    """Output of fitting a flow model."""

    model: ConditionalShapeFlow
    history: dict[str, list[float]]
    best_epoch: int
    best_val_nll: float


@dataclass
class FlowCheckpoint:
    """Flow model and scalers loaded from a checkpoint."""

    model: ConditionalShapeFlow
    target_scaler: Standardizer
    context_scaler: Standardizer
    extra: dict[str, Any] = field(default_factory=dict)


def resolve_device(config: TrainingConfig) -> torch.device:
    """Return the configured device, falling back to CUDA when available."""

    return torch.device(config.device) if config.device is not None else default_device()


def prepare_training_data(
    e_true: Tensor | Any,
    e_meas: Tensor | Any,
    cond: Tensor | Any,
    *,
    config: TrainingConfig,
    target_scaler: Standardizer | None = None,
    context_scaler: Standardizer | None = None,
) -> PreparedTrainingData:
    """Split arrays, fit scalers on training data, and build data loaders."""

    torch.manual_seed(config.seed)
    train_arrays, val_arrays = split_arrays(
        e_true,
        e_meas,
        cond,
        val_fraction=config.val_fraction,
        seed=config.seed,
        shuffle=True,
    )

    target_scaler = target_scaler or Standardizer().fit(train_arrays.e_meas)
    context_scaler = context_scaler or Standardizer().fit(train_arrays.context())

    train_dataset = ShapeDataset(
        target_scaler.transform(train_arrays.e_meas),
        context_scaler.transform(train_arrays.context()),
    )
    val_dataset = ShapeDataset(
        target_scaler.transform(val_arrays.e_meas),
        context_scaler.transform(val_arrays.context()),
    )

    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    return PreparedTrainingData(
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        target_scaler=target_scaler,
        context_scaler=context_scaler,
        train_loader=train_loader,
        val_loader=val_loader,
    )


def build_shape_flow(
    context_features: int,
    *,
    config: TrainingConfig,
    device: torch.device | str | None = None,
) -> ConditionalShapeFlow:
    """Build the conditional normalizing-flow density estimator."""

    flow_config = FlowConfig(
        context_features=context_features,
        transforms=config.transforms,
        hidden_features=config.hidden_features,
        bins=config.bins,
    )
    device = torch.device(device) if device is not None else resolve_device(config)
    return ConditionalShapeFlow(flow_config).to(device)


def load_flow_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str,
) -> FlowCheckpoint:
    """Load a flow checkpoint without wrapping it as a likelihood."""

    checkpoint = torch.load(path, map_location=device)
    flow_config = FlowConfig.from_dict(checkpoint["flow_config"])
    model = ConditionalShapeFlow(flow_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return FlowCheckpoint(
        model=model,
        target_scaler=Standardizer.from_state_dict(checkpoint["target_scaler"]),
        context_scaler=Standardizer.from_state_dict(checkpoint["context_scaler"]),
        extra=dict(checkpoint.get("extra", {})),
    )


def train_model(
    model: ConditionalShapeFlow,
    train_loader: DataLoader[tuple[Tensor, Tensor]],
    val_loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    config: TrainingConfig,
    device: torch.device | str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> ModelTrainingResult:
    """Fit the flow model and restore the best validation state."""

    device = torch.device(device) if device is not None else resolve_device(config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: dict[str, list[float]] = {"train_nll": [], "val_nll": []}
    best_state: dict[str, Tensor] | None = None
    best_val_nll = float("inf")
    best_epoch = 0

    epochs_without_improvement = 0
    for epoch in range(1, config.maximum_training_epoch + 1):
        train_nll = _train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_nll = _evaluate_nll(model, val_loader, device=device)
        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if progress_callback is not None:
            progress_callback(epoch, {"train_nll": train_nll, "val_nll": val_nll})

        if epochs_without_improvement >= config.stop_after_epoch:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return ModelTrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_nll=best_val_nll,
    )


def save_flow_checkpoint(
    model: ConditionalShapeFlow,
    target_scaler: Standardizer,
    context_scaler: Standardizer,
    path: str | Path,
    *,
    history: dict[str, list[float]],
    best_epoch: int,
    best_val_nll: float,
    config: TrainingConfig,
) -> None:
    """Save flow parameters, scalers, and training metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "flow_config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "target_scaler": target_scaler.state_dict(),
        "context_scaler": context_scaler.state_dict(),
        "extra": {
            "history": history,
            "best_epoch": best_epoch,
            "best_val_nll": best_val_nll,
            "training_config": _training_config_to_dict(config),
        },
    }
    torch.save(checkpoint, path)


def train_shape_flow(
    e_true: Tensor | Any,
    e_meas: Tensor | Any,
    cond: Tensor | Any,
    *,
    config: TrainingConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> TrainingResult:
    """Train q_phi(e_meas | e_true, cond) with separate target/context scalers."""

    config = config or TrainingConfig()
    device = resolve_device(config)
    resumed = None
    if config.resume_checkpoint_path is not None:
        resumed = load_flow_checkpoint(
            config.resume_checkpoint_path,
            device=device,
        )

    prepared = prepare_training_data(
        e_true,
        e_meas,
        cond,
        config=config,
        target_scaler=resumed.target_scaler if resumed is not None else None,
        context_scaler=resumed.context_scaler if resumed is not None else None,
    )
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
    trained = train_model(
        model,
        prepared.train_loader,
        prepared.val_loader,
        config=config,
        device=device,
        progress_callback=progress_callback,
    )
    result = TrainingResult(
        model=trained.model,
        target_scaler=prepared.target_scaler,
        context_scaler=prepared.context_scaler,
        history=trained.history,
        best_epoch=trained.best_epoch,
        best_val_nll=trained.best_val_nll,
        train_arrays=prepared.train_arrays,
        val_arrays=prepared.val_arrays,
    )

    if config.checkpoint_path is not None:
        save_flow_checkpoint(
            trained.model,
            prepared.target_scaler,
            prepared.context_scaler,
            config.checkpoint_path,
            history=trained.history,
            best_epoch=trained.best_epoch,
            best_val_nll=trained.best_val_nll,
            config=config,
        )

    return result


def _train_one_epoch(
    model: ConditionalShapeFlow,
    loader: DataLoader[tuple[Tensor, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    grad_clip_norm: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for x, context in loader:
        x = x.to(device)
        context = context.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = -model.log_prob(x, context).mean()
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        batch_count = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count

    return total_loss / max(total_count, 1)


@torch.no_grad()
def _evaluate_nll(
    model: ConditionalShapeFlow,
    loader: DataLoader[tuple[Tensor, Tensor]],
    *,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_count = 0

    for x, context in loader:
        x = x.to(device)
        context = context.to(device)
        loss = -model.log_prob(x, context).mean()
        batch_count = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_count
        total_count += batch_count

    return total_loss / max(total_count, 1)


def _training_config_to_dict(config: TrainingConfig) -> dict[str, Any]:
    state = config.__dict__.copy()
    state["hidden_features"] = tuple(config.hidden_features)
    state["device"] = str(config.device) if config.device is not None else None
    for key in ("checkpoint_path", "resume_checkpoint_path"):
        value = state[key]
        state[key] = str(value) if value is not None else None
    return state
