"""Feature-wise standardization utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


def _as_float_tensor(data: Tensor | Any) -> Tensor:
    if isinstance(data, Tensor):
        return data.detach().to(dtype=torch.float32)
    return torch.as_tensor(data, dtype=torch.float32)


@dataclass
class Standardizer:
    """Feature-wise affine standardizer fit on training data only."""

    mean: Tensor | None = None
    std: Tensor | None = None
    eps: float = 1.0e-6

    @property
    def is_fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    @property
    def n_features(self) -> int:
        self._check_fitted()
        assert self.mean is not None
        return int(self.mean.numel())

    @property
    def device(self) -> torch.device:
        self._check_fitted()
        assert self.mean is not None
        return self.mean.device

    def fit(self, data: Tensor | Any) -> "Standardizer":
        values = _as_float_tensor(data)
        if values.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {tuple(values.shape)}")
        if values.shape[0] < 1:
            raise ValueError("Cannot fit a standardizer on an empty array")

        self.mean = values.mean(dim=0)
        self.std = values.std(dim=0, unbiased=False).clamp_min(self.eps)
        return self

    def transform(self, data: Tensor | Any) -> Tensor:
        self._check_fitted()
        values = _as_float_tensor(data)
        self._check_feature_shape(values)
        assert self.mean is not None and self.std is not None
        mean = self.mean.to(device=values.device)
        std = self.std.to(device=values.device)
        return (values - mean) / std

    def inverse_transform(self, data: Tensor | Any) -> Tensor:
        self._check_fitted()
        values = _as_float_tensor(data)
        self._check_feature_shape(values)
        assert self.mean is not None and self.std is not None
        mean = self.mean.to(device=values.device)
        std = self.std.to(device=values.device)
        return values * std + mean

    def log_abs_det_jacobian(self) -> Tensor:
        """Return log |d x / d x_std| for one target vector."""

        self._check_fitted()
        assert self.std is not None
        return torch.log(self.std).sum()

    def to(self, device: torch.device | str) -> "Standardizer":
        self._check_fitted()
        assert self.mean is not None and self.std is not None
        return Standardizer(
            mean=self.mean.to(device),
            std=self.std.to(device),
            eps=self.eps,
        )

    def state_dict(self) -> dict[str, Any]:
        self._check_fitted()
        assert self.mean is not None and self.std is not None
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "eps": float(self.eps),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Standardizer":
        return cls(
            mean=torch.as_tensor(state["mean"], dtype=torch.float32),
            std=torch.as_tensor(state["std"], dtype=torch.float32),
            eps=float(state.get("eps", 1.0e-6)),
        )

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Standardizer must be fit before use")

    def _check_feature_shape(self, values: Tensor) -> None:
        if values.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {tuple(values.shape)}")
        if values.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {values.shape[1]}"
            )
