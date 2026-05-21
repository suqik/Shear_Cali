#!/usr/bin/env python
"""Sample intrinsic-shape posteriors with Zeus from a trained flow checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shape_flow.utils import ConfigOption, merge_config, validate_sampling_config


CONFIG_OPTIONS = (
    ConfigOption("checkpoint", "paths", "checkpoint", "path"),
    ConfigOption("output", "paths", "output", "path"),
    ConfigOption("data", "paths", "data", "path"),
    ConfigOption("index", "observation", "index", "int", 0),
    ConfigOption("e_meas", "observation", "e_meas", "float_list", None),
    ConfigOption("cond", "observation", "cond", "float_list", None),
    ConfigOption("n_walkers", "mcmc", "n_walkers", "int", 32),
    ConfigOption("n_steps", "mcmc", "n_steps", "int", 1000),
    ConfigOption("burn_in", "mcmc", "burn_in", "int", 200),
    ConfigOption("thin", "mcmc", "thin", "int", 1),
    ConfigOption("initial_scale", "mcmc", "initial_scale", "float", 0.03),
    ConfigOption("prior_radius", "mcmc", "prior_radius", "float", 0.999),
    ConfigOption("seed", "mcmc", "seed", "int", 0),
    ConfigOption("progress", "mcmc", "progress", "bool", False),
    ConfigOption("verbose", "mcmc", "verbose", "bool", False),
    ConfigOption("light_mode", "mcmc", "light_mode", "bool", False),
    ConfigOption("device", "runtime", "device", "str", "cpu"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample p(e_true | e_meas, cond) using zeus-mcmc."
    )
    parser.add_argument("--config", type=Path, help="INI configuration file.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data", type=Path, help="NPZ containing e_meas and cond arrays.")
    parser.add_argument("--index", type=int, help="Row to sample from --data.")
    parser.add_argument("--e-meas", type=float, nargs=2, help="Observed e1 e2.")
    parser.add_argument("--cond", type=float, nargs="*", help="Condition values.")
    parser.add_argument("--n-walkers", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--burn-in", type=int)
    parser.add_argument("--thin", type=int)
    parser.add_argument("--initial-scale", type=float)
    parser.add_argument("--prior-radius", type=float)
    parser.add_argument(
        "--light-mode",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = merge_config(parser.parse_args(), CONFIG_OPTIONS)
    validate_sampling_config(args)
    return args


def load_observation(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.data is not None:
        with np.load(args.data) as data:
            missing = {"e_meas", "cond"} - set(data.files)
            if missing:
                raise KeyError(f"NPZ file is missing arrays: {sorted(missing)}")
            return data["e_meas"][args.index], data["cond"][args.index]

    if args.e_meas is None or args.cond is None:
        raise ValueError("Provide either --data or both --e-meas and --cond")
    return np.asarray(args.e_meas, dtype=np.float32), np.asarray(args.cond, dtype=np.float32)


def main() -> None:
    args = parse_args()

    from shape_flow import MCMCConfig, load_likelihood, sample_posterior_zeus

    e_meas, cond = load_observation(args)
    likelihood = load_likelihood(args.checkpoint, map_location=args.device)
    config = MCMCConfig(
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        burn_in=args.burn_in,
        thin=args.thin,
        initial_scale=args.initial_scale,
        prior_radius=args.prior_radius,
        random_seed=args.seed,
        progress=args.progress,
        verbose=args.verbose,
        light_mode=args.light_mode,
    )
    result = sample_posterior_zeus(
        likelihood,
        e_meas,
        cond,
        config=config,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        samples=result.samples,
        log_prob=result.log_prob,
        chain=result.chain,
        log_prob_chain=result.log_prob_chain,
        initial_state=result.initial_state,
        e_meas=e_meas,
        cond=cond,
        ncall=result.ncall if result.ncall is not None else -1,
        efficiency=result.efficiency if result.efficiency is not None else np.nan,
    )
    print(f"saved={args.output}")
    print(f"samples={result.samples.shape}")
    print(f"ncall={result.ncall}")
    print(f"efficiency={result.efficiency}")


if __name__ == "__main__":
    main()
