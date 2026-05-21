import argparse
import importlib.util
from pathlib import Path


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
        epochs=None,
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

    merged = module.merge_config(args)

    assert merged.data == ROOT / "training_arrays.npz"
    assert merged.output == ROOT / "checkpoints" / "shape_flow.pt"
    assert merged.hidden_features == [128, 128]
    assert merged.batch_size == 512


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
        initial_scale=None,
        prior_radius=None,
        light_mode=None,
        seed=None,
        device=None,
        progress=None,
        verbose=None,
    )

    merged = module.merge_config(args)

    assert merged.checkpoint == ROOT / "checkpoints" / "shape_flow.pt"
    assert merged.output == ROOT / "posterior_samples.npz"
    assert merged.index == 3
    assert merged.n_walkers == 40
    assert merged.progress is False
