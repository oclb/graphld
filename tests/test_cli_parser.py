"""Focused tests for GraphLD CLI parser and dispatch seams."""

import argparse

from graphld import cli
from graphld._cli_dispatch import dispatch_command
from graphld._cli_parser import build_parser


def _recording_handler(calls, name):
    def handler(*args):
        calls.append((name, args))
        return name

    return handler


def test_cli_module_preserves_private_parser_imports():
    """Existing private imports from graphld.cli should keep resolving."""
    assert callable(cli.main)
    assert callable(cli._main)
    assert callable(cli._blup)
    assert callable(cli._add_reml_parser)
    assert callable(cli._add_common_arguments)
    assert callable(cli._construct_cmd_string)
    assert callable(cli.build_parser)
    assert cli.dispatch_command is dispatch_command

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    cli._add_blup_parser(subparsers)
    args = parser.parse_args(["blup", "trait.sumstats", "weights.tsv", "-H", "0.4"])
    assert args.func is cli._blup


def test_build_parser_parses_all_graphld_subcommands():
    handlers = {
        "blup": lambda: None,
        "clump": lambda: None,
        "surrogates": lambda: None,
        "simulate": lambda: None,
        "reml": lambda: None,
    }
    parser = build_parser(handlers)

    blup_args = parser.parse_args(
        ["blup", "trait.sumstats", "weights.tsv", "-H", "0.4", "--run-in-serial", "-q"]
    )
    assert blup_args.cmd == "blup"
    assert blup_args.sumstats == "trait.sumstats"
    assert blup_args.out == "weights.tsv"
    assert blup_args.heritability == 0.4
    assert blup_args.run_in_serial is True
    assert blup_args.quiet is True
    assert blup_args.func is handlers["blup"]

    clump_args = parser.parse_args(
        ["clump", "trait.sumstats", "clumped.tsv", "--min-chisq", "5", "--max-rsq", "0.2"]
    )
    assert clump_args.cmd == "clump"
    assert clump_args.min_chisq == 5
    assert clump_args.max_rsq == 0.2
    assert clump_args.func is handlers["clump"]

    surrogate_args = parser.parse_args(
        ["surrogates", "trait.snplist", "surrogates.h5", "--chromosome", "22"]
    )
    assert surrogate_args.cmd == "surrogates"
    assert surrogate_args.chromosome == 22
    assert surrogate_args.population == "EUR"
    assert surrogate_args.func is handlers["surrogates"]

    simulate_args = parser.parse_args(
        ["simulate", "sim.sumstats", "--num-samples", "1000", "--component-weight", "0.7,0.3"]
    )
    assert simulate_args.cmd == "simulate"
    assert simulate_args.sumstats_out == "sim.sumstats"
    assert simulate_args.num_samples == 1000
    assert simulate_args.component_weight == [0.7, 0.3]
    assert simulate_args.func is handlers["simulate"]

    reml_args = parser.parse_args(
        [
            "reml",
            "trait.sumstats",
            "out/reml",
            "--annot-dir",
            "annotations",
            "--annotation-columns",
            "base,coding",
            "--initial-params",
            "0.1,0.2",
        ]
    )
    assert reml_args.cmd == "reml"
    assert reml_args.out == "out/reml"
    assert reml_args.annot_dir == "annotations"
    assert reml_args.annotation_columns == ["base", "coding"]
    assert reml_args.initial_params == [0.1, 0.2]
    assert reml_args.func is handlers["reml"]


def test_dispatch_command_uses_existing_command_call_shapes():
    calls = []
    handlers = {
        name: _recording_handler(calls, name)
        for name in ["blup", "clump", "surrogates", "simulate", "reml"]
    }
    parser = build_parser(handlers)

    assert dispatch_command(
        parser.parse_args(["blup", "trait.sumstats", "weights.tsv", "-H", "0.4"])
    ) == "blup"
    assert calls.pop() == (
        "blup",
        (
            "trait.sumstats",
            "weights.tsv",
            "data/ldgms/metadata.csv",
            None,
            0.4,
            None,
            False,
            None,
            "EUR",
            False,
            False,
        ),
    )

    assert dispatch_command(
        parser.parse_args(["clump", "trait.sumstats", "clumped.tsv", "--min-chisq", "5"])
    ) == "clump"
    assert calls.pop() == (
        "clump",
        (
            "trait.sumstats",
            "clumped.tsv",
            "data/ldgms/metadata.csv",
            None,
            5,
            0.1,
            None,
            False,
            None,
            "EUR",
            False,
            False,
        ),
    )

    assert dispatch_command(
        parser.parse_args(["surrogates", "trait.snplist", "surrogates.h5", "--chromosome", "22"])
    ) == "surrogates"
    assert calls.pop() == (
        "surrogates",
        (
            "trait.snplist",
            "surrogates.h5",
            "data/ldgms/metadata.csv",
            None,
            False,
            "EUR",
            False,
            False,
            22,
        ),
    )

    assert dispatch_command(
        parser.parse_args(["simulate", "sim.sumstats", "--num-samples", "1000"])
    ) == "simulate"
    assert calls.pop() == (
        "simulate",
        (
            "sim.sumstats",
            "data/ldgms/metadata.csv",
            0.2,
            1000,
            [1.0],
            [1.0],
            -0.5,
            False,
            None,
            None,
            None,
            False,
            None,
            "EUR",
            False,
            False,
            None,
        ),
    )

    reml_args = parser.parse_args(["reml", "trait.sumstats", "out/reml", "--annot-dir", "annotations"])
    assert dispatch_command(reml_args) == "reml"
    assert calls.pop() == ("reml", (reml_args,))
