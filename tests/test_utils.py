import configparser
from pathlib import Path

from shape_flow.utils import (
    ConfigOption,
    _bool_option,
    _float_list_option,
    _float_option,
    _int_list_option,
    _int_option,
    _path_option,
    _str_option,
    merge_config,
)


def test_config_option_helpers_parse_values(tmp_path):
    parser = configparser.ConfigParser()
    parser["section"] = {
        "flag": "true",
        "float_value": "1.25",
        "float_values": "1.0, 2.5 3.5",
        "int_value": "7",
        "int_values": "16, 32",
        "path": "relative/file.npy",
        "string": "cpu",
    }

    assert _bool_option(None, parser, "section", "flag", False) is True
    assert _float_option(None, parser, "section", "float_value", 0.0) == 1.25
    assert _float_list_option(None, parser, "section", "float_values", None) == [
        1.0,
        2.5,
        3.5,
    ]
    assert _int_option(None, parser, "section", "int_value", 0) == 7
    assert _int_list_option(None, parser, "section", "int_values", []) == [16, 32]
    assert _path_option(None, parser, "section", "path", tmp_path) == (
        tmp_path / "relative" / "file.npy"
    ).resolve()
    assert _str_option(None, parser, "section", "string", None) == "cpu"


def test_config_option_helpers_prefer_cli_values(tmp_path):
    parser = configparser.ConfigParser()
    parser["section"] = {"int_value": "7", "path": "from_config.npy"}

    cli_path = Path("from_cli.npy")

    assert _int_option(3, parser, "section", "int_value", 0) == 3
    assert _path_option(cli_path, parser, "section", "path", tmp_path) == cli_path


def test_merge_config_uses_option_specs(tmp_path):
    config = tmp_path / "settings.ini"
    config.write_text(
        "[paths]\n"
        "input = arrays.npz\n"
        "\n"
        "[runtime]\n"
        "epochs = 5\n"
    )
    args = type(
        "Args",
        (),
        {"config": config, "input": None, "epochs": 9},
    )()

    merged = merge_config(
        args,
        [
            ConfigOption("input", "paths", "input", "path"),
            ConfigOption("epochs", "runtime", "epochs", "int", 1),
        ],
    )

    assert merged.input == tmp_path / "arrays.npz"
    assert merged.epochs == 9
