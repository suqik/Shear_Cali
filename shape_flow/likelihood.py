"""Likelihood evaluation and checkpoint utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .data import as_float_tensor, make_context
from .model import ConditionalShapeFlow, FlowConfig
from .scaling import Standardizer


@dataclass
class ShapeFlowLikelihood:
    """Bundle a trained flow with target and context standardizers."""

    model: ConditionalShapeFlow
    target_scaler: Standardizer
    context_scaler: Standardizer

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to(self, device: torch.device | str) -> "ShapeFlowLikelihood":
        device = torch.device(device)
        self.model.to(device)
        self.target_scaler = self.target_scaler.to(device)
        self.context_scaler = self.context_scaler.to(device)
        return self

    @torch.no_grad()
    def log_prob_standardized(
        self,
        e_meas: Tensor | Any,
        e_true: Tensor | Any,
        cond: Tensor | Any,
        *,
        batch_size: int | None = None,
    ) -> Tensor:
        """Evaluate log q_std(e_meas_std | context_std)."""

        self.model.eval()
        x = self.target_scaler.transform(as_float_tensor(e_meas, "e_meas"))
        context = self.context_scaler.transform(make_context(e_true, cond))
        return self._batched_log_prob(x, context, batch_size=batch_size)

    @torch.no_grad()
    def log_prob(
        self,
        e_meas: Tensor | Any,
        e_true: Tensor | Any,
        cond: Tensor | Any,
        *,
        batch_size: int | None = None,
    ) -> Tensor:
        """Evaluate the physical-unit log likelihood q(e_meas | e_true, cond)."""

        log_prob_std = self.log_prob_standardized(
            e_meas, e_true, cond, batch_size=batch_size
        )
        jacobian = self.target_scaler.log_abs_det_jacobian().to(log_prob_std.device)
        return log_prob_std - jacobian

    def save(self, path: str | Path, *, extra: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "flow_config": self.model.config.to_dict(),
            "model_state_dict": self.model.state_dict(),
            "target_scaler": self.target_scaler.state_dict(),
            "context_scaler": self.context_scaler.state_dict(),
            "extra": extra or {},
        }
        torch.save(checkpoint, path)

    def _batched_log_prob(
        self, x: Tensor, context: Tensor, *, batch_size: int | None
    ) -> Tensor:
        if batch_size is None:
            batch_size = x.shape[0]
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        values: list[Tensor] = []
        for start in range(0, x.shape[0], batch_size):
            stop = start + batch_size
            x_batch = x[start:stop].to(self.device)
            context_batch = context[start:stop].to(self.device)
            values.append(self.model.log_prob(x_batch, context_batch).detach().cpu())
        return torch.cat(values, dim=0)


def load_likelihood(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
) -> ShapeFlowLikelihood:
    """Load a trained flow checkpoint and wrap it as a likelihood."""

    checkpoint = torch.load(path, map_location=map_location or "cpu")
    config = FlowConfig.from_dict(checkpoint["flow_config"])
    model = ConditionalShapeFlow(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    target_scaler = Standardizer.from_state_dict(checkpoint["target_scaler"])
    context_scaler = Standardizer.from_state_dict(checkpoint["context_scaler"])
    likelihood = ShapeFlowLikelihood(model, target_scaler, context_scaler)
    if map_location is not None:
        likelihood.to(map_location)
    return likelihood
