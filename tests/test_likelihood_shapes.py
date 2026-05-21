import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("zuko")

from shape_flow import TrainingConfig, load_likelihood, train_shape_flow


def test_train_save_load_and_log_prob_shapes():
    generator = torch.Generator().manual_seed(7)
    n_samples = 48
    e_true = torch.randn(n_samples, 2, generator=generator) * 0.2
    cond = torch.randn(n_samples, 3, generator=generator)
    noise = torch.randn(n_samples, 2, generator=generator) * 0.05
    e_meas = e_true + noise

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint = Path(tmpdir) / "shape_flow.pt"
        config = TrainingConfig(
            stop_after_epoch=2,
            maximum_training_epoch=1,
            batch_size=16,
            val_fraction=0.25,
            hidden_features=(16,),
            transforms=1,
            seed=7,
            device="cpu",
            checkpoint_path=checkpoint,
        )
        result = train_shape_flow(e_true, e_meas, cond, config=config)
        likelihood = load_likelihood(checkpoint, map_location="cpu")

    log_prob_std = likelihood.log_prob_standardized(e_meas[:6], e_true[:6], cond[:6])
    log_prob = likelihood.log_prob(e_meas[:6], e_true[:6], cond[:6])

    assert result.model.config.features == 2
    assert log_prob_std.shape == (6,)
    assert log_prob.shape == (6,)
    assert torch.isfinite(log_prob).all()
