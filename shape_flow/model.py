"""Conditional normalizing-flow model definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

try:
    import zuko
except ImportError:  # pragma: no cover - handled when the model is constructed
    zuko = None


@dataclass(frozen=True)
class FlowConfig:
    """Serializable configuration for the conditional NSF model."""

    context_features: int
    features: int = 2
    transforms: int = 5
    hidden_features: tuple[int, ...] = (128, 128)
    bins: int = 8

    def __post_init__(self) -> None:
        if self.features != 2:
            raise ValueError("The first implementation expects two target shape features")
        if self.context_features < 2:
            raise ValueError("context_features must include the two e_true features")
        if self.transforms < 1:
            raise ValueError("transforms must be positive")
        if self.bins < 2:
            raise ValueError("bins must be at least 2")
        if not self.hidden_features:
            raise ValueError("hidden_features must contain at least one layer width")
        if any(width < 1 for width in self.hidden_features):
            raise ValueError("hidden_features must contain positive layer widths")

    def to_dict(self) -> dict[str, Any]:
        state = asdict(self)
        state["hidden_features"] = tuple(self.hidden_features)
        return state

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> "FlowConfig":
        state = dict(state)
        state["hidden_features"] = tuple(state["hidden_features"])
        return cls(**state)


class ConditionalShapeFlow(nn.Module):
    """Conditional neural spline flow for q(e_meas | e_true, cond)."""

    def __init__(self, config: FlowConfig) -> None:
        super().__init__()
        if zuko is None:
            raise ImportError(
                "zuko is required for ConditionalShapeFlow. Install it with `pip install zuko`."
            )
        self.config = config
        self.flow = zuko.flows.NSF(
            features=config.features,
            context=config.context_features,
            transforms=config.transforms,
            hidden_features=config.hidden_features,
            bins=config.bins,
        )

    def forward(self, context: Tensor):
        return self.flow(context)

    def log_prob(self, x: Tensor, context: Tensor) -> Tensor:
        if x.ndim != 2 or x.shape[1] != self.config.features:
            raise ValueError(
                f"x must have shape (N, {self.config.features}), got {tuple(x.shape)}"
            )
        if context.ndim != 2 or context.shape[1] != self.config.context_features:
            raise ValueError(
                "context must have shape "
                f"(N, {self.config.context_features}), got {tuple(context.shape)}"
            )
        if x.shape[0] != context.shape[0]:
            raise ValueError(
                f"x and context must have the same number of rows, got {x.shape[0]} "
                f"and {context.shape[0]}"
            )
        return self.flow(context).log_prob(x)

    @torch.no_grad()
    def sample(self, context: Tensor, sample_shape: torch.Size | tuple[int, ...] = ()) -> Tensor:
        return self.flow(context).sample(sample_shape)
