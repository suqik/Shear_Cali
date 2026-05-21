# Shape Flow

Conditional normalizing-flow likelihood model for galaxy shape-measurement error
calibration.

The implemented first pass learns

```python
q_phi(e_meas | e_true, cond)
```

with a Zuko neural spline flow. The target is `e_meas` with shape `(N, 2)`.
The context is `torch.cat([e_true, cond], dim=1)`, where `e_true` has shape
`(N, 2)` and `cond` has shape `(N, K)`.

## Install

```bash
python -m pip install -e ".[test]"
```

For the Jupyter demo and plotting dependencies:

```bash
python -m pip install -e ".[demo]"
```

## Train From An NPZ

The NPZ must contain arrays named `e_true`, `e_meas`, and `cond`.

```bash
python scripts/train_shape_flow.py \
  --data training_arrays.npz \
  --output checkpoints/shape_flow.pt \
  --epochs 100 \
  --batch-size 512
```

Separate `.npy` inputs are also supported:

```bash
python scripts/train_shape_flow.py \
  --e-true e_true.npy \
  --e-meas e_meas.npy \
  --cond cond.npy \
  --output checkpoints/shape_flow.pt
```

## Python API

```python
from shape_flow import TrainingConfig, load_likelihood, train_shape_flow

config = TrainingConfig(epochs=100, checkpoint_path="checkpoints/shape_flow.pt")
result = train_shape_flow(e_true, e_meas, cond, config=config)

likelihood = load_likelihood("checkpoints/shape_flow.pt", map_location="cpu")
log_q_std = likelihood.log_prob_standardized(e_meas_new, e_true_new, cond_new)
log_q = likelihood.log_prob(e_meas_new, e_true_new, cond_new)
```

`log_prob_standardized` evaluates the flow density in standardized target units.
`log_prob` returns the physical-unit likelihood and applies the target
normalization Jacobian:

```python
log_q = log_q_std - sum(log(target_std))
```

The context scaler is only used to normalize conditioning variables; it does
not add a likelihood Jacobian.

## Demo Notebook

Open [notebooks/shape_flow_demo.ipynb](notebooks/shape_flow_demo.ipynb) for a
small synthetic-data example that trains the model, evaluates likelihoods, and
tests checkpoint reloads.
