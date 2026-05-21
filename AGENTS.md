# AGENTS.md

## Project goal

This repository implements a conditional normalizing-flow model for galaxy shape-measurement error calibration.

The main statistical goal is to learn the conditional likelihood

\[
q_\phi(\hat e \mid e_{\rm int}, c),
\]

where:

- \(\hat e = (\hat e_1, \hat e_2)\) is the measured galaxy shape.
- \(e_{\rm int} = (e_1, e_2)\) is the intrinsic galaxy shape.
- \(c\) contains additional galaxy properties, such as half-light radius, signal-to-noise ratio, magnitude or flux, and possibly PSF-related quantities.

After learning the likelihood, the code should support Bayesian inversion:

\[
p(e_{\rm int} \mid \hat e, c)
\propto
q_\phi(\hat e \mid e_{\rm int}, c)\,
p(e_{\rm int}\mid c).
\]

The first implementation should focus on learning the likelihood. Posterior inference can be added later.

## Core modeling choice

Use a conditional normalizing flow implemented with `zuko`.

The flow should model

```python
q_phi(e_meas | e_true, cond)
````

where:

```python
x = e_meas
context = concatenate([e_true, cond])
```

In Zuko notation, this should correspond to:

```python
flow(context).log_prob(x)
```

The target variable is the measured shape. The context variables are the intrinsic shape plus galaxy properties.

## Normalization requirements

All continuous inputs must be standardized using training-set mean and standard deviation.

There must be two separate standardizers:

1. A target scaler for `x = e_meas`.
2. A context scaler for `context = [e_true, cond]`.

The scalers must be fit only on the training set.

The target normalization changes the physical likelihood by a Jacobian factor:

[
\log q(\hat e | e_{\rm int}, c)
===============================

## \log q_{\rm std}(\hat e_{\rm std} | context_{\rm std})

\sum_j \log \sigma_{\hat e,j}.
]

The context normalization does not require a likelihood Jacobian correction because context variables are conditioned on, not modeled as random target variables.

## Residual likelihood option

The code should be designed so that it can later support a residual version:

[
\Delta e = \hat e - e_{\rm int},
]

and learn

[
q_\phi(\Delta e \mid e_{\rm int}, c).
]

However, the first implementation should default to the direct likelihood:

[
q_\phi(\hat e \mid e_{\rm int}, c).
]

Do not implement the residual mode unless explicitly requested.

## Code style

Use Python with PyTorch and Zuko.

Prefer small, composable modules:

* scaler class
* dataset/data-preparation utilities
* flow model class
* training loop
* likelihood-evaluation utilities

Avoid putting all logic into one long script.

The code should be readable and research-friendly rather than over-engineered.

Use type hints where helpful.

Avoid hidden global variables.

Avoid hard-coding the number of conditioning variables.

## Expected package structure

A good initial structure is:

```text
shape_flow/
    __init__.py
    scaling.py
    model.py
    train.py
    likelihood.py
scripts/
    train_shape_flow.py
```

Optional later modules:

```text
shape_flow/
    posterior.py
    diagnostics.py
    plotting.py
tests/
    test_scaling.py
    test_likelihood_shapes.py
```

## Numerical details

Use `float32` tensors by default.

Use GPU if available, but keep CPU compatibility.

Use AdamW optimizer.

Use gradient clipping in the training loop.

The training loop should report both training and validation negative log-likelihood.

The validation split should be done before fitting scalers.

The best validation model should be saved or restorable.

## Shape conventions

Assume:

```python
e_true.shape == (N, 2)
e_meas.shape == (N, 2)
cond.shape == (N, K)
```

Then:

```python
x.shape == (N, 2)
context.shape == (N, 2 + K)
```

The flow should therefore have:

```python
features = 2
context = 2 + K
```

## Important caveats

Do not assume that the condition variables are necessarily true latent properties. In real data they may be noisy observed quantities.

For the first implementation, treat the condition variables as fixed context variables.

Do not implement cosmology-specific logic. This repository is only about the shape-measurement likelihood model.

Do not implement posterior inference in the first pass unless explicitly requested.

## Minimal success criterion

The first working version should be able to:

1. Load arrays `e_true`, `e_meas`, and `cond`.
2. Split into train and validation sets.
3. Fit separate standardizers for target and context.
4. Train a conditional Zuko NSF model for `q(e_meas | e_true, cond)`.
5. Evaluate log likelihood in both standardized units and original physical units.
6. Save the trained model and scalers.
7. Reload them and evaluate likelihood for new inputs.


