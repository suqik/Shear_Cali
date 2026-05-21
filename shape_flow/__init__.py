"""Conditional normalizing flows for galaxy shape likelihood calibration."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "ConditionalShapeFlow",
    "ConfigOption",
    "FlowCheckpoint",
    "FlowConfig",
    "MCMCConfig",
    "MCMCResult",
    "ModelTrainingResult",
    "PreparedTrainingData",
    "ShapeArrays",
    "ShapeDataset",
    "ShapeFlowLikelihood",
    "Standardizer",
    "TrainingConfig",
    "TrainingResult",
    "build_shape_flow",
    "initialize_walkers",
    "load_flow_checkpoint",
    "load_likelihood",
    "make_log_posterior",
    "make_context",
    "merge_config",
    "prepare_training_data",
    "read_config",
    "resolve_device",
    "sample_posterior_zeus",
    "save_flow_checkpoint",
    "split_arrays",
    "train_model",
    "train_shape_flow",
    "uniform_ellipticity_prior",
]

_LAZY_EXPORTS = {
    "ConditionalShapeFlow": ".model",
    "ConfigOption": ".utils",
    "FlowCheckpoint": ".train",
    "FlowConfig": ".model",
    "MCMCConfig": ".posterior",
    "MCMCResult": ".posterior",
    "ModelTrainingResult": ".train",
    "PreparedTrainingData": ".train",
    "ShapeArrays": ".data",
    "ShapeDataset": ".data",
    "ShapeFlowLikelihood": ".likelihood",
    "Standardizer": ".scaling",
    "TrainingConfig": ".train",
    "TrainingResult": ".train",
    "build_shape_flow": ".train",
    "initialize_walkers": ".posterior",
    "load_flow_checkpoint": ".train",
    "load_likelihood": ".likelihood",
    "make_log_posterior": ".posterior",
    "make_context": ".data",
    "merge_config": ".utils",
    "prepare_training_data": ".train",
    "read_config": ".utils",
    "resolve_device": ".train",
    "sample_posterior_zeus": ".posterior",
    "save_flow_checkpoint": ".train",
    "split_arrays": ".data",
    "train_model": ".train",
    "train_shape_flow": ".train",
    "uniform_ellipticity_prior": ".posterior",
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'shape_flow' has no attribute {name!r}")
    module = import_module(_LAZY_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
