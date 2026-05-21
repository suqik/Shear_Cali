import argparse
import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_script(path: str):
    spec = importlib.util.spec_from_file_location("script_under_test", ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_train_config_file_merges_defaults_and_paths():
    module = load_script("scripts/train_shape_flow.py")
    args = argparse.Namespace(
        config=ROOT / "configs" / "train_shape_flow.ini",
        data=None,
        e_true=None,
        e_meas=None,
        cond=None,
        output=None,
        resume_checkpoint=ROOT / "checkpoints" / "previous.pt",
        epochs=None,
        stop_after_epoch=None,
        maximum_training_epoch=None,
        batch_size=None,
        val_fraction=None,
        learning_rate=None,
        weight_decay=None,
        grad_clip_norm=None,
        transforms=None,
        hidden_features=None,
        bins=None,
        seed=None,
        device=None,
    )

    merged = module.merge_config(args, module.CONFIG_OPTIONS)
    module.validate_training_config(merged)

    assert merged.data == ROOT / "training_arrays.npy"
    assert merged.output == ROOT / "checkpoints" / "shape_flow.pt"
    assert merged.resume_checkpoint == ROOT / "checkpoints" / "previous.pt"
    assert merged.hidden_features == [128, 128]
    assert merged.batch_size == 512
    assert merged.stop_after_epoch == 20
    assert merged.maximum_training_epoch == 100


def test_sampler_config_file_allows_cli_override():
    module = load_script("scripts/sample_shape_posterior.py")
    args = argparse.Namespace(
        config=ROOT / "configs" / "sample_shape_posterior.ini",
        checkpoint=None,
        output=None,
        data=None,
        index=3,
        e_meas=None,
        cond=None,
        n_walkers=40,
        n_steps=None,
        burn_in=None,
        thin=None,
        n_processes=None,
        initial_scale=None,
        prior_radius=None,
        light_mode=None,
        seed=None,
        device=None,
        progress=None,
        verbose=None,
    )

    merged = module.merge_config(args, module.CONFIG_OPTIONS)
    module.validate_sampling_config(merged)

    assert merged.checkpoint == ROOT / "checkpoints" / "shape_flow.pt"
    assert merged.output == ROOT / "posterior_samples.npz"
    assert merged.data == ROOT / "training_arrays.npy"
    assert merged.index == 3
    assert merged.n_walkers == 40
    assert merged.n_processes == 1
    assert merged.progress is False


def test_show_mcmc_figure_uses_sampler_output_config():
    module = load_script("scripts/show_mcmc_figure.py")
    args = argparse.Namespace(
        config=ROOT / "configs" / "sample_shape_posterior.ini",
        input=None,
        savefig=None,
        chain_savefig=None,
        data=None,
        index=None,
        show=None,
        dpi=150,
        max_points=5000,
        truth=None,
        title="MCMC posterior contour",
    )

    merged = module.merge_config(args, module.CONFIG_OPTIONS)

    assert merged.input == ROOT / "posterior_samples.npz"
    assert merged.data == ROOT / "training_arrays.npy"
    assert merged.index == 0


def test_show_mcmc_figure_loads_posterior_npz(tmp_path):
    module = load_script("scripts/show_mcmc_figure.py")
    path = tmp_path / "posterior_samples.npz"
    samples = np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32)
    np.savez(path, samples=samples, log_prob=np.array([1.0, 2.0]), ncall=12)

    loaded = module.load_posterior_file(path)

    np.testing.assert_allclose(loaded.samples, samples)
    assert loaded.log_prob.shape == (2,)
    assert loaded.ncall == 12


def test_show_mcmc_figure_loads_truth_from_structured_npy(tmp_path):
    module = load_script("scripts/show_mcmc_figure.py")
    path = tmp_path / "training_arrays.npy"
    data = structured_shape_data()
    np.save(path, data)

    truth = module.load_truth_from_data_file(path, index=1)

    np.testing.assert_allclose(truth, [0.3, -0.4])


def test_train_loader_accepts_structured_npy(tmp_path):
    module = load_script("scripts/train_shape_flow.py")
    path = tmp_path / "training_arrays.npy"
    data = structured_shape_data()
    np.save(path, data)

    args = argparse.Namespace(data=path, e_true=None, e_meas=None, cond=None)
    e_true, e_meas, cond = module.load_arrays(args)

    np.testing.assert_allclose(e_true, [[0.1, -0.2], [0.3, -0.4]])
    np.testing.assert_allclose(e_meas, [[0.11, -0.21], [0.31, -0.41]])
    np.testing.assert_allclose(cond, [[1.2, 23.0, 40.0], [1.4, 24.0, 60.0]])


def test_sampler_loader_accepts_structured_npy(tmp_path):
    module = load_script("scripts/sample_shape_posterior.py")
    path = tmp_path / "training_arrays.npy"
    data = structured_shape_data()
    np.save(path, data)

    args = argparse.Namespace(data=path, index=1, e_meas=None, cond=None)
    e_meas, cond = module.load_observation(args)

    np.testing.assert_allclose(e_meas, [0.31, -0.41])
    np.testing.assert_allclose(cond, [1.4, 24.0, 60.0])


def structured_shape_data():
    data = np.zeros(
        2,
        dtype=[
            ("e1_t", "f4"),
            ("e2_t", "f4"),
            ("e1", "f4"),
            ("e2", "f4"),
            ("hlf", "f4"),
            ("mag", "f4"),
            ("snr", "f4"),
        ],
    )
    data["e1_t"] = [0.1, 0.3]
    data["e2_t"] = [-0.2, -0.4]
    data["e1"] = [0.11, 0.31]
    data["e2"] = [-0.21, -0.41]
    data["hlf"] = [1.2, 1.4]
    data["mag"] = [23.0, 24.0]
    data["snr"] = [40.0, 60.0]
    return data
