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

The same inputs can be supplied through an INI file parsed with
`configparser`:

```bash
python scripts/train_shape_flow.py --config configs/train_shape_flow.ini
```

Training configs support `[paths]`, `[training]`, and `[model]` sections.
Command-line flags override values from the config file. Relative paths inside
the config file are resolved relative to the config file location.

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

## Posterior MCMC Sampling

After training a likelihood, sample the intrinsic-shape posterior for one
observed galaxy with `zeus-mcmc`:

```python
from shape_flow import MCMCConfig, load_likelihood, sample_posterior_zeus

likelihood = load_likelihood("checkpoints/shape_flow.pt", map_location="cpu")
config = MCMCConfig(n_walkers=32, n_steps=1000, burn_in=200, thin=2)

result = sample_posterior_zeus(
    likelihood,
    e_meas_obs,
    cond_obs,
    config=config,
)

samples = result.samples
```

The default prior is uniform inside
`sqrt(e_true_1**2 + e_true_2**2) <= prior_radius`. Pass a vectorized
`prior_log_prob(e_true_batch, cond_batch)` function to use a different prior.

You can also run the sampler from the command line:

```bash
python scripts/sample_shape_posterior.py \
  --checkpoint checkpoints/shape_flow.pt \
  --data training_arrays.npz \
  --index 0 \
  --output posterior_samples.npz
```

Or use an INI file:

```bash
python scripts/sample_shape_posterior.py --config configs/sample_shape_posterior.ini
```

Posterior-sampling configs support `[paths]`, `[observation]`, `[mcmc]`, and
`[runtime]` sections. Command-line flags override config values.

## Demo Notebook

Open [notebooks/shape_flow_demo.ipynb](notebooks/shape_flow_demo.ipynb) for a
small synthetic-data example that trains the model, evaluates likelihoods, and
tests checkpoint reloads and posterior MCMC sampling.
