#!/usr/bin/env python
"""Sample intrinsic-shape posteriors with emcee from a trained likelihood."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shape_flow import MCMCConfig, load_likelihood, sample_posterior_emcee


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample p(e_true | e_meas, cond) using emcee."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="NPZ containing e_meas and cond arrays.")
    parser.add_argument("--index", type=int, default=0, help="Row to sample from --data.")
    parser.add_argument("--e-meas", type=float, nargs=2, help="Observed e1 e2.")
    parser.add_argument("--cond", type=float, nargs="*", help="Condition values.")
    parser.add_argument("--n-walkers", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=200)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--initial-scale", type=float, default=0.03)
    parser.add_argument("--prior-radius", type=float, default=0.999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


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
    )
    result = sample_posterior_emcee(
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
        acceptance_fraction=result.acceptance_fraction,
        initial_state=result.initial_state,
        e_meas=e_meas,
        cond=cond,
    )
    print(f"saved={args.output}")
    print(f"samples={result.samples.shape}")
    print(f"mean_acceptance={result.acceptance_fraction.mean():.3f}")


if __name__ == "__main__":
    main()
