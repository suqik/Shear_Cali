"""Data validation and preparation helpers for shape-flow training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


def as_float_tensor(data: Tensor | Any, name: str) -> Tensor:
    values = data.detach().clone() if isinstance(data, Tensor) else torch.as_tensor(data)
    values = values.to(dtype=torch.float32)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {tuple(values.shape)}")
    return values


def make_context(e_true: Tensor | Any, cond: Tensor | Any) -> Tensor:
    """Concatenate intrinsic shape and condition variables."""

    e_true_tensor = as_float_tensor(e_true, "e_true")
    cond_tensor = as_float_tensor(cond, "cond")
    if e_true_tensor.shape[0] != cond_tensor.shape[0]:
        raise ValueError(
            "e_true and cond must have the same number of rows, got "
            f"{e_true_tensor.shape[0]} and {cond_tensor.shape[0]}"
        )
    if e_true_tensor.shape[1] != 2:
        raise ValueError(f"e_true must have shape (N, 2), got {tuple(e_true_tensor.shape)}")
    return torch.cat([e_true_tensor, cond_tensor], dim=1)


@dataclass(frozen=True)
class ShapeArrays:
    """Raw arrays with the project shape convention."""

    e_true: Tensor
    e_meas: Tensor
    cond: Tensor

    @classmethod
    def from_arrays(
        cls, e_true: Tensor | Any, e_meas: Tensor | Any, cond: Tensor | Any
    ) -> "ShapeArrays":
        e_true_tensor = as_float_tensor(e_true, "e_true")
        e_meas_tensor = as_float_tensor(e_meas, "e_meas")
        cond_tensor = as_float_tensor(cond, "cond")

        n_rows = e_true_tensor.shape[0]
        if e_meas_tensor.shape[0] != n_rows or cond_tensor.shape[0] != n_rows:
            raise ValueError(
                "e_true, e_meas, and cond must have the same number of rows; got "
                f"{n_rows}, {e_meas_tensor.shape[0]}, and {cond_tensor.shape[0]}"
            )
        if e_true_tensor.shape[1] != 2:
            raise ValueError(f"e_true must have shape (N, 2), got {tuple(e_true_tensor.shape)}")
        if e_meas_tensor.shape[1] != 2:
            raise ValueError(f"e_meas must have shape (N, 2), got {tuple(e_meas_tensor.shape)}")
        return cls(e_true=e_true_tensor, e_meas=e_meas_tensor, cond=cond_tensor)

    @property
    def n_samples(self) -> int:
        return int(self.e_true.shape[0])

    @property
    def cond_features(self) -> int:
        return int(self.cond.shape[1])

    @property
    def context_features(self) -> int:
        return int(self.e_true.shape[1] + self.cond.shape[1])

    def context(self) -> Tensor:
        return make_context(self.e_true, self.cond)

    def subset(self, indices: Tensor) -> "ShapeArrays":
        return ShapeArrays(
            e_true=self.e_true[indices],
            e_meas=self.e_meas[indices],
            cond=self.cond[indices],
        )


class ShapeDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dataset of standardized target and context tensors."""

    def __init__(self, x: Tensor | Any, context: Tensor | Any) -> None:
        self.x = as_float_tensor(x, "x")
        self.context = as_float_tensor(context, "context")
        if self.x.shape[0] != self.context.shape[0]:
            raise ValueError(
                "x and context must have the same number of rows, got "
                f"{self.x.shape[0]} and {self.context.shape[0]}"
            )
        if self.x.shape[1] != 2:
            raise ValueError(f"x must have shape (N, 2), got {tuple(self.x.shape)}")

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.x[index], self.context[index]

    def to(self, device: torch.device | str) -> "ShapeDataset":
        self.x = self.x.to(device)
        self.context = self.context.to(device)
        return self


def split_arrays(
    e_true: Tensor | Any,
    e_meas: Tensor | Any,
    cond: Tensor | Any,
    *,
    val_fraction: float = 0.2,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[ShapeArrays, ShapeArrays]:
    """Split raw arrays before any scaler fitting."""

    arrays = ShapeArrays.from_arrays(e_true, e_meas, cond)
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    if arrays.n_samples < 2:
        raise ValueError("Need at least two samples to create train/validation splits")

    n_val = max(1, int(round(arrays.n_samples * val_fraction)))
    n_val = min(n_val, arrays.n_samples - 1)

    if shuffle:
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(arrays.n_samples, generator=generator)
    else:
        indices = torch.arange(arrays.n_samples)

    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    return arrays.subset(train_indices), arrays.subset(val_indices)
