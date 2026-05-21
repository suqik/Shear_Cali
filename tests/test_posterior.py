import pytest
import tempfile
from pathlib import Path

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("zuko")
pytest.importorskip("zeus")

from shape_flow import MCMCConfig, load_likelihood, sample_posterior_zeus, train_shape_flow
from shape_flow.posterior import uniform_ellipticity_prior
from shape_flow.train import TrainingConfig


def test_uniform_ellipticity_prior():
    e_true = np.array([[0.1, 0.2], [1.0, 1.0]], dtype=np.float32)
    cond = np.zeros((2, 1), dtype=np.float32)

    log_prior = uniform_ellipticity_prior(e_true, cond, max_radius=0.5)

    assert np.isfinite(log_prior[0])
    assert not np.isfinite(log_prior[1])


def test_mcmc_config_rejects_invalid_process_count():
    with pytest.raises(ValueError, match="n_processes"):
        MCMCConfig(n_processes=0)


def test_zeus_posterior_sampling_shapes():
    generator = torch.Generator().manual_seed(11)
    n_samples = 64
    e_true = torch.randn(n_samples, 2, generator=generator) * 0.2
    cond = torch.randn(n_samples, 2, generator=generator)
    e_meas = e_true + torch.randn(n_samples, 2, generator=generator) * 0.05

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "shape_flow.pt"
        training_config = TrainingConfig(
            stop_after_epoch=2,
            maximum_training_epoch=1,
            batch_size=16,
            val_fraction=0.25,
            hidden_features=(16,),
            transforms=1,
            seed=11,
            device="cpu",
            checkpoint_path=checkpoint,
        )
        train_shape_flow(e_true, e_meas, cond, config=training_config)
        likelihood = load_likelihood(checkpoint, map_location="cpu")
    mcmc_config = MCMCConfig(
        n_walkers=8,
        n_steps=8,
        burn_in=2,
        thin=1,
        random_seed=11,
        progress=False,
        n_processes=2,
    )

    result = sample_posterior_zeus(
        likelihood,
        e_meas[0],
        cond[0],
        config=mcmc_config,
    )

    assert result.samples.shape == ((8 - 2) * 8, 2)
    assert result.log_prob.shape == ((8 - 2) * 8,)
    assert result.chain.shape == (8, 8, 2)
    assert np.isfinite(result.samples).all()
