#!/usr/bin/env python
"""Sample intrinsic-shape posteriors with Zeus from a trained likelihood."""

from __future__ import annotations

import argparse
import configparser
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
    return merge_config(parser.parse_args())


def merge_config(args: argparse.Namespace) -> argparse.Namespace:
    parser = configparser.ConfigParser()
    config_dir = Path.cwd()
    if args.config is not None:
        read_files = parser.read(args.config)
        if not read_files:
            raise FileNotFoundError(f"Could not read config file: {args.config}")
        config_dir = args.config.resolve().parent

    args.checkpoint = _path_option(
        args.checkpoint,
        parser,
        "paths",
        "checkpoint",
        config_dir,
    )
    args.output = _path_option(args.output, parser, "paths", "output", config_dir)
    args.data = _path_option(args.data, parser, "paths", "data", config_dir)

    args.index = _int_option(args.index, parser, "observation", "index", 0)
    args.e_meas = _float_list_option(
        args.e_meas,
        parser,
        "observation",
        "e_meas",
        None,
    )
    args.cond = _float_list_option(
        args.cond,
        parser,
        "observation",
        "cond",
        None,
    )

    args.n_walkers = _int_option(args.n_walkers, parser, "mcmc", "n_walkers", 32)
    args.n_steps = _int_option(args.n_steps, parser, "mcmc", "n_steps", 1000)
    args.burn_in = _int_option(args.burn_in, parser, "mcmc", "burn_in", 200)
    args.thin = _int_option(args.thin, parser, "mcmc", "thin", 1)
    args.initial_scale = _float_option(
        args.initial_scale,
        parser,
        "mcmc",
        "initial_scale",
        0.03,
    )
    args.prior_radius = _float_option(
        args.prior_radius,
        parser,
        "mcmc",
        "prior_radius",
        0.999,
    )
    args.seed = _int_option(args.seed, parser, "mcmc", "seed", 0)
    args.progress = _bool_option(args.progress, parser, "mcmc", "progress", False)
    args.verbose = _bool_option(args.verbose, parser, "mcmc", "verbose", False)
    args.light_mode = _bool_option(
        args.light_mode,
        parser,
        "mcmc",
        "light_mode",
        False,
    )
    args.device = _str_option(args.device, parser, "runtime", "device", "cpu")

    if args.checkpoint is None:
        raise ValueError("Provide checkpoint in [paths] or with --checkpoint")
    if args.output is None:
        raise ValueError("Provide output in [paths] or with --output")
    if args.data is not None and args.e_meas is not None:
        raise ValueError("Use either data/index or explicit e_meas/cond, not both")
    if args.data is None and args.e_meas is None:
        raise ValueError("Provide data in [paths] or e_meas/cond in [observation]")
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


def _path_option(
    cli_value: Path | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    base_dir: Path,
) -> Path | None:
    if cli_value is not None:
        return cli_value
    if not parser.has_option(section, option):
        return None
    value = Path(parser.get(section, option))
    return value if value.is_absolute() else (base_dir / value).resolve()


def _str_option(
    cli_value: str | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    default: str | None,
) -> str | None:
    if cli_value is not None:
        return cli_value
    if parser.has_option(section, option):
        value = parser.get(section, option).strip()
        return value or None
    return default


def _int_option(
    cli_value: int | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    default: int,
) -> int:
    if cli_value is not None:
        return cli_value
    return parser.getint(section, option, fallback=default)


def _float_option(
    cli_value: float | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    default: float,
) -> float:
    if cli_value is not None:
        return cli_value
    return parser.getfloat(section, option, fallback=default)


def _bool_option(
    cli_value: bool | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    default: bool,
) -> bool:
    if cli_value is not None:
        return cli_value
    return parser.getboolean(section, option, fallback=default)


def _float_list_option(
    cli_value: list[float] | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    default: list[float] | None,
) -> list[float] | None:
    if cli_value is not None:
        return cli_value
    if not parser.has_option(section, option):
        return default
    raw = parser.get(section, option)
    return [float(item.strip()) for item in raw.replace(",", " ").split()]


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
