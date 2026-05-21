"""Posterior sampling utilities built on a trained shape likelihood."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .likelihood import ShapeFlowLikelihood

PriorLogProb = Callable[[np.ndarray, np.ndarray], np.ndarray | float]
_PARALLEL_LIKELIHOOD: ShapeFlowLikelihood | None = None
_PARALLEL_E_MEAS: np.ndarray | None = None
_PARALLEL_COND: np.ndarray | None = None
_PARALLEL_PRIOR_LOG_PROB: PriorLogProb | None = None
_PARALLEL_PRIOR_RADIUS = 0.999


@dataclass
class MCMCConfig:
    """Configuration for Zeus posterior sampling of one galaxy."""

    n_walkers: int = 32
    n_steps: int = 1000
    burn_in: int = 200
    thin: int = 1
    initial_scale: float = 0.03
    prior_radius: float = 0.999
    random_seed: int | None = 0
    progress: bool = False
    verbose: bool = False
    light_mode: bool = False
    n_processes: int = 1

    def __post_init__(self) -> None:
        if self.n_walkers < 4:
            raise ValueError("n_walkers must be at least 4 for a 2D posterior")
        if self.n_walkers % 2:
            raise ValueError("Zeus requires an even number of walkers")
        if self.n_steps < 1:
            raise ValueError("n_steps must be positive")
        if not 0 <= self.burn_in < self.n_steps:
            raise ValueError("burn_in must satisfy 0 <= burn_in < n_steps")
        if self.thin < 1:
            raise ValueError("thin must be positive")
        if self.initial_scale <= 0.0:
            raise ValueError("initial_scale must be positive")
        if self.prior_radius <= 0.0:
            raise ValueError("prior_radius must be positive")
        if self.n_processes < 1:
            raise ValueError("n_processes must be positive")


@dataclass
class MCMCResult:
    """Samples and diagnostics returned by the Zeus posterior sampler."""

    samples: np.ndarray
    log_prob: np.ndarray
    chain: np.ndarray
    log_prob_chain: np.ndarray
    initial_state: np.ndarray
    ncall: int | None
    efficiency: float | None
    config: MCMCConfig = field(repr=False)


def uniform_ellipticity_prior(
    e_true: np.ndarray,
    cond: np.ndarray,
    *,
    max_radius: float = 0.999,
) -> np.ndarray:
    """Uniform prior inside ``sqrt(e1**2 + e2**2) <= max_radius``."""

    _ = cond
    values = np.asarray(e_true, dtype=np.float64)
    if values.shape[-1] != 2:
        raise ValueError(f"e_true must end with 2 features, got {values.shape}")
    radius2 = np.sum(values * values, axis=-1)
    return np.where(radius2 <= max_radius * max_radius, 0.0, -np.inf)


def make_log_posterior(
    likelihood: ShapeFlowLikelihood,
    e_meas: Any,
    cond: Any,
    *,
    prior_log_prob: PriorLogProb | None = None,
    prior_radius: float = 0.999,
) -> Callable[[np.ndarray], np.ndarray | float]:
    """Build ``log p(e_true | e_meas, cond)`` for Zeus.

    The returned function accepts either one position with shape ``(2,)`` or a
    vectorized walker array with shape ``(n_walkers, 2)``.
    """

    return _make_log_posterior_from_vectors(
        likelihood,
        _as_vector(e_meas, "e_meas", n_features=2),
        _as_vector(cond, "cond"),
        prior_log_prob=prior_log_prob,
        prior_radius=prior_radius,
    )


def _make_log_posterior_from_vectors(
    likelihood: ShapeFlowLikelihood,
    e_meas_vector: np.ndarray,
    cond_vector: np.ndarray,
    *,
    prior_log_prob: PriorLogProb | None,
    prior_radius: float,
) -> Callable[[np.ndarray], np.ndarray | float]:
    def log_posterior(e_true: np.ndarray) -> np.ndarray | float:
        return _evaluate_log_posterior(
            likelihood,
            e_meas_vector,
            cond_vector,
            e_true,
            prior_log_prob=prior_log_prob,
            prior_radius=prior_radius,
        )

    return log_posterior


def _evaluate_log_posterior(
    likelihood: ShapeFlowLikelihood,
    e_meas_vector: np.ndarray,
    cond_vector: np.ndarray,
    e_true: np.ndarray,
    *,
    prior_log_prob: PriorLogProb | None,
    prior_radius: float,
) -> np.ndarray | float:
    e_true_batch, squeeze = _as_e_true_batch(e_true)
    n_batch = e_true_batch.shape[0]
    e_meas_batch = np.repeat(e_meas_vector[None, :], n_batch, axis=0)
    cond_batch = np.repeat(cond_vector[None, :], n_batch, axis=0)

    if prior_log_prob is None:
        prior_values = uniform_ellipticity_prior(
            e_true_batch,
            cond_batch,
            max_radius=prior_radius,
        )
    else:
        prior_values = prior_log_prob(e_true_batch, cond_batch)

    prior_values = _as_log_prob_array(
        prior_values,
        n_batch,
        name="prior_log_prob",
    )
    posterior = np.full(n_batch, -np.inf, dtype=np.float64)
    valid = np.isfinite(prior_values)

    if np.any(valid):
        likelihood_values = likelihood.log_prob(
            e_meas_batch[valid],
            e_true_batch[valid],
            cond_batch[valid],
        )
        posterior[valid] = prior_values[valid] + _to_numpy(likelihood_values)

    return float(posterior[0]) if squeeze else posterior


def sample_posterior_zeus(
    likelihood: ShapeFlowLikelihood,
    e_meas: Any,
    cond: Any,
    *,
    prior_log_prob: PriorLogProb | None = None,
    config: MCMCConfig | None = None,
    initial_state: Any | None = None,
) -> MCMCResult:
    """Sample ``p(e_true | e_meas, cond)`` with ``zeus-mcmc``.

    The posterior is proportional to the learned physical-unit likelihood
    multiplied by the supplied prior. If no prior is supplied, a uniform prior
    inside the ellipticity disk is used.
    """

    try:
        import zeus
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "zeus-mcmc is required for posterior sampling. Install it with "
            "`pip install zeus-mcmc` or `pip install -e .`."
        ) from exc

    config = config or MCMCConfig()
    if config.random_seed is not None:
        np.random.seed(config.random_seed)

    e_meas_vector = _as_vector(e_meas, "e_meas", n_features=2)
    cond_vector = _as_vector(cond, "cond")
    log_posterior = _make_log_posterior_from_vectors(
        likelihood,
        e_meas_vector,
        cond_vector,
        prior_log_prob=prior_log_prob,
        prior_radius=config.prior_radius,
    )
    rng = np.random.default_rng(config.random_seed)

    if initial_state is None:
        initial_state_array = initialize_walkers(
            e_meas,
            config=config,
            rng=rng,
            log_posterior=log_posterior,
        )
    else:
        initial_state_array = _as_initial_state(initial_state, config.n_walkers)
        initial_log_prob = np.asarray(log_posterior(initial_state_array))
        if not np.all(np.isfinite(initial_log_prob)):
            raise ValueError("initial_state contains walkers with non-finite log posterior")

    pool = None
    sampler_log_posterior = log_posterior
    vectorize = True
    if config.n_processes > 1:
        pool = _make_log_posterior_pool(
            config.n_processes,
            likelihood,
            e_meas_vector,
            cond_vector,
            prior_log_prob,
            config.prior_radius,
        )
        sampler_log_posterior = _parallel_log_posterior
        vectorize = False

    try:
        sampler = zeus.EnsembleSampler(
            config.n_walkers,
            2,
            sampler_log_posterior,
            pool=pool,
            vectorize=vectorize,
            verbose=config.verbose,
            light_mode=config.light_mode,
        )
        sampler.run_mcmc(
            initial_state_array,
            config.n_steps,
            progress=config.progress,
        )
    except BaseException:
        if pool is not None:
            pool.terminate()
            pool.join()
        raise
    else:
        if pool is not None:
            pool.close()
            pool.join()

    return MCMCResult(
        samples=sampler.get_chain(
            discard=config.burn_in,
            thin=config.thin,
            flat=True,
        ),
        log_prob=sampler.get_log_prob(
            discard=config.burn_in,
            thin=config.thin,
            flat=True,
        ),
        chain=sampler.get_chain(),
        log_prob_chain=sampler.get_log_prob(),
        initial_state=initial_state_array,
        ncall=_safe_int_attr(sampler, "ncall"),
        efficiency=_safe_float_attr(sampler, "efficiency"),
        config=config,
    )


def initialize_walkers(
    center: Any,
    *,
    config: MCMCConfig | None = None,
    rng: np.random.Generator | None = None,
    log_posterior: Callable[[np.ndarray], np.ndarray | float] | None = None,
) -> np.ndarray:
    """Initialize walkers near a center and inside the default prior disk."""

    config = config or MCMCConfig()
    rng = rng or np.random.default_rng(config.random_seed)
    center_vector = _clip_to_radius(
        _as_vector(center, "center", n_features=2),
        radius=0.8 * config.prior_radius,
    )
    walkers = center_vector + config.initial_scale * rng.normal(
        size=(config.n_walkers, 2)
    )

    if log_posterior is None:
        return walkers.astype(np.float64)

    for _ in range(100):
        log_prob = np.asarray(log_posterior(walkers), dtype=np.float64)
        invalid = ~np.isfinite(log_prob)
        if not np.any(invalid):
            return walkers.astype(np.float64)
        walkers[invalid] = _draw_uniform_disk(
            rng,
            int(np.sum(invalid)),
            config.prior_radius,
        )

    raise ValueError(
        "Could not initialize all walkers at finite posterior probability. "
        "Provide an explicit initial_state or loosen the prior."
    )


def _make_log_posterior_pool(
    n_processes: int,
    likelihood: ShapeFlowLikelihood,
    e_meas_vector: np.ndarray,
    cond_vector: np.ndarray,
    prior_log_prob: PriorLogProb | None,
    prior_radius: float,
) -> Any:
    return mp.get_context().Pool(
        processes=n_processes,
        initializer=_initialize_parallel_log_posterior,
        initargs=(
            likelihood,
            e_meas_vector,
            cond_vector,
            prior_log_prob,
            prior_radius,
        ),
    )


def _initialize_parallel_log_posterior(
    likelihood: ShapeFlowLikelihood,
    e_meas_vector: np.ndarray,
    cond_vector: np.ndarray,
    prior_log_prob: PriorLogProb | None,
    prior_radius: float,
) -> None:
    global _PARALLEL_LIKELIHOOD
    global _PARALLEL_E_MEAS
    global _PARALLEL_COND
    global _PARALLEL_PRIOR_LOG_PROB
    global _PARALLEL_PRIOR_RADIUS

    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:  # pragma: no cover
        pass

    likelihood.model.eval()
    _PARALLEL_LIKELIHOOD = likelihood
    _PARALLEL_E_MEAS = e_meas_vector
    _PARALLEL_COND = cond_vector
    _PARALLEL_PRIOR_LOG_PROB = prior_log_prob
    _PARALLEL_PRIOR_RADIUS = prior_radius


def _parallel_log_posterior(e_true: np.ndarray) -> float:
    if (
        _PARALLEL_LIKELIHOOD is None
        or _PARALLEL_E_MEAS is None
        or _PARALLEL_COND is None
    ):
        raise RuntimeError("parallel log-posterior worker is not initialized")
    return float(
        _evaluate_log_posterior(
            _PARALLEL_LIKELIHOOD,
            _PARALLEL_E_MEAS,
            _PARALLEL_COND,
            e_true,
            prior_log_prob=_PARALLEL_PRIOR_LOG_PROB,
            prior_radius=_PARALLEL_PRIOR_RADIUS,
        )
    )


def _to_numpy(data: Any) -> np.ndarray:
    if hasattr(data, "detach"):
        return data.detach().cpu().numpy()
    return np.asarray(data)


def _as_vector(data: Any, name: str, n_features: int | None = None) -> np.ndarray:
    values = np.asarray(_to_numpy(data), dtype=np.float32)
    if values.ndim == 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector, got shape {values.shape}")
    if n_features is not None and values.shape[0] != n_features:
        raise ValueError(f"{name} must have {n_features} features, got {values.shape[0]}")
    return values


def _as_e_true_batch(data: Any) -> tuple[np.ndarray, bool]:
    values = np.asarray(_to_numpy(data), dtype=np.float32)
    if values.ndim == 1:
        if values.shape[0] != 2:
            raise ValueError(f"e_true must have 2 features, got {values.shape[0]}")
        return values[None, :], True
    if values.ndim == 2 and values.shape[1] == 2:
        return values, False
    raise ValueError(f"e_true must have shape (2,) or (N, 2), got {values.shape}")


def _as_log_prob_array(data: Any, n_expected: int, *, name: str) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim == 0:
        return np.full(n_expected, float(values), dtype=np.float64)
    if values.shape != (n_expected,):
        raise ValueError(f"{name} must return shape ({n_expected},), got {values.shape}")
    return values


def _as_initial_state(data: Any, n_walkers: int) -> np.ndarray:
    values = np.asarray(_to_numpy(data), dtype=np.float64)
    if values.shape != (n_walkers, 2):
        raise ValueError(
            f"initial_state must have shape ({n_walkers}, 2), got {values.shape}"
        )
    return values


def _clip_to_radius(values: np.ndarray, *, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    if norm <= radius:
        return values.astype(np.float64)
    return (values * (radius / norm)).astype(np.float64)


def _draw_uniform_disk(
    rng: np.random.Generator,
    n_samples: int,
    radius: float,
) -> np.ndarray:
    angle = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    radial = radius * np.sqrt(rng.uniform(0.0, 1.0, size=n_samples))
    return np.column_stack([radial * np.cos(angle), radial * np.sin(angle)])


def _safe_int_attr(obj: Any, name: str) -> int | None:
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    return None if value is None else int(value)


def _safe_float_attr(obj: Any, name: str) -> float | None:
    try:
        value = getattr(obj, name)
    except Exception:
        return None
    return None if value is None else float(value)
