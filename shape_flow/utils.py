"""Small utilities shared by command-line entry points."""

from __future__ import annotations

import configparser
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class ConfigOption:
    """Map one argparse attribute to one INI option."""

    dest: str
    section: str
    option: str
    kind: str
    default: Any = None


def read_config(config_path: Path | None) -> tuple[configparser.ConfigParser, Path]:
    """Read an optional INI config and return it with its path base directory."""

    parser = configparser.ConfigParser()
    if config_path is None:
        return parser, Path.cwd()

    read_files = parser.read(config_path)
    if not read_files:
        raise FileNotFoundError(f"Could not read config file: {config_path}")
    return parser, config_path.resolve().parent


def merge_config(
    args: Namespace,
    options: Sequence[ConfigOption],
    *,
    config_attr: str = "config",
) -> Namespace:
    """Merge CLI args with INI values according to ``ConfigOption`` specs."""

    parser, config_dir = read_config(getattr(args, config_attr, None))
    for spec in options:
        cli_value = getattr(args, spec.dest, None)
        value = _option_value(cli_value, parser, spec, config_dir)
        setattr(args, spec.dest, value)
    return args


def validate_training_config(args: Namespace) -> None:
    """Validate merged training CLI/config inputs."""

    if args.data is not None and args.e_true is not None:
        raise ValueError("Use either data or e_true/e_meas/cond inputs, not both")
    if args.data is None and args.e_true is None:
        raise ValueError("Provide data in [paths] or with --data/--e-true")
    if args.output is None:
        raise ValueError("Provide output in [paths] or with --output")
    if getattr(args, "maximum_training_epoch", None) is None:
        legacy_epochs = getattr(args, "epochs", None)
        args.maximum_training_epoch = 100 if legacy_epochs is None else legacy_epochs
    if getattr(args, "stop_after_epoch", None) is None:
        args.stop_after_epoch = 20
    if args.maximum_training_epoch < 1:
        raise ValueError("maximum_training_epoch must be positive")
    if args.stop_after_epoch < 1:
        raise ValueError("stop_after_epoch must be positive")


def validate_sampling_config(args: Namespace) -> None:
    """Validate merged posterior-sampling CLI/config inputs."""

    if args.checkpoint is None:
        raise ValueError("Provide checkpoint in [paths] or with --checkpoint")
    if args.output is None:
        raise ValueError("Provide output in [paths] or with --output")
    if args.data is not None and args.e_meas is not None:
        raise ValueError("Use either data/index or explicit e_meas/cond, not both")
    if args.data is None and args.e_meas is None:
        raise ValueError("Provide data in [paths] or e_meas/cond in [observation]")


def _option_value(
    cli_value: Any,
    parser: configparser.ConfigParser,
    spec: ConfigOption,
    config_dir: Path,
) -> Any:
    if spec.kind == "path":
        return _path_option(cli_value, parser, spec.section, spec.option, config_dir)
    if spec.kind == "str":
        return _str_option(cli_value, parser, spec.section, spec.option, spec.default)
    if spec.kind == "int":
        return _int_option(cli_value, parser, spec.section, spec.option, spec.default)
    if spec.kind == "float":
        return _float_option(cli_value, parser, spec.section, spec.option, spec.default)
    if spec.kind == "bool":
        return _bool_option(cli_value, parser, spec.section, spec.option, spec.default)
    if spec.kind == "int_list":
        return _int_list_option(cli_value, parser, spec.section, spec.option, spec.default)
    if spec.kind == "float_list":
        return _float_list_option(
            cli_value,
            parser,
            spec.section,
            spec.option,
            spec.default,
        )
    raise ValueError(f"Unknown config option kind: {spec.kind}")


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
    default: int | None,
) -> int | None:
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


def _int_list_option(
    cli_value: list[int] | None,
    parser: configparser.ConfigParser,
    section: str,
    option: str,
    default: list[int],
) -> list[int]:
    if cli_value is not None:
        return cli_value
    if not parser.has_option(section, option):
        return default
    raw = parser.get(section, option)
    return [int(item.strip()) for item in raw.replace(",", " ").split()]


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


path_option = _path_option
str_option = _str_option
int_option = _int_option
float_option = _float_option
bool_option = _bool_option
int_list_option = _int_list_option
float_list_option = _float_list_option


__all__ = [
    "ConfigOption",
    "_bool_option",
    "_float_list_option",
    "_float_option",
    "_int_list_option",
    "_int_option",
    "_path_option",
    "_str_option",
    "bool_option",
    "float_list_option",
    "float_option",
    "int_list_option",
    "int_option",
    "merge_config",
    "path_option",
    "read_config",
    "str_option",
    "validate_sampling_config",
    "validate_training_config",
]
