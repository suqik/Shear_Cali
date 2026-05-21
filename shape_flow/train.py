"""Training loop for the conditional shape likelihood flow."""

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
from .likelihood import ShapeFlowLikelihood
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
    epochs: int = 100
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 5.0
    transforms: int = 5
    hidden_features: tuple[int, ...] = (128, 128)
    bins: int = 8
    device: str | torch.device | None = None
    checkpoint_path: str | Path | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.epochs < 1:
            raise ValueError("epochs must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive")
        self.hidden_features = tuple(self.hidden_features)


@dataclass
class TrainingResult:
    likelihood: ShapeFlowLikelihood
    history: dict[str, list[float]]
    best_epoch: int
    best_val_nll: float
    train_arrays: ShapeArrays = field(repr=False)
    val_arrays: ShapeArrays = field(repr=False)


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
    torch.manual_seed(config.seed)
    device = torch.device(config.device) if config.device is not None else default_device()

    train_arrays, val_arrays = split_arrays(
        e_true,
        e_meas,
        cond,
        val_fraction=config.val_fraction,
        seed=config.seed,
        shuffle=True,
    )

    target_scaler = Standardizer().fit(train_arrays.e_meas)
    context_scaler = Standardizer().fit(train_arrays.context())

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

    flow_config = FlowConfig(
        context_features=train_arrays.context_features,
        transforms=config.transforms,
        hidden_features=config.hidden_features,
        bins=config.bins,
    )
    model = ConditionalShapeFlow(flow_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: dict[str, list[float]] = {"train_nll": [], "val_nll": []}
    best_state: dict[str, Tensor] | None = None
    best_val_nll = float("inf")
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
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

        if progress_callback is not None:
            progress_callback(epoch, {"train_nll": train_nll, "val_nll": val_nll})

    if best_state is not None:
        model.load_state_dict(best_state)

    likelihood = ShapeFlowLikelihood(model, target_scaler.to(device), context_scaler.to(device))
    result = TrainingResult(
        likelihood=likelihood,
        history=history,
        best_epoch=best_epoch,
        best_val_nll=best_val_nll,
        train_arrays=train_arrays,
        val_arrays=val_arrays,
    )

    if config.checkpoint_path is not None:
        likelihood.save(
            config.checkpoint_path,
            extra={
                "history": history,
                "best_epoch": best_epoch,
                "best_val_nll": best_val_nll,
                "training_config": _training_config_to_dict(config),
            },
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
    state["checkpoint_path"] = (
        str(config.checkpoint_path) if config.checkpoint_path is not None else None
    )
    return state
