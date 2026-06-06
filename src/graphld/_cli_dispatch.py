"""Dispatch helpers for the GraphLD CLI."""

import os
import sys
from collections.abc import Callable, Sequence
from importlib import metadata

from ._cli_parser import build_parser


def _construct_cmd_string(args, parser):
    """Reconstruct the command line string."""
    cmd_str = "graphld"
    for arg, value in vars(args).items():
        if arg not in ["func", "cmd"]:
            if value is True:
                cmd_str += f" --{arg}"
            elif value is not False and value is not None:
                cmd_str += f" --{arg} {value}"
    return cmd_str


def _write_masthead(parsed_args, parser) -> None:
    cmd_str = _construct_cmd_string(parsed_args, parser)
    version = f"v{metadata.version('graphld')}"
    masthead = f"""
    ***********************************************************************
    ************************** GraphLD {version} *****************************
    ***********************************************************************

    """

    if not parsed_args.quiet:
        sys.stdout.write(masthead)
        sys.stdout.write(cmd_str + os.linesep)


def dispatch_command(parsed_args):
    """Run the command selected by parsed CLI arguments."""
    if parsed_args.cmd == "blup":
        return parsed_args.func(
            parsed_args.sumstats,
            parsed_args.out,
            parsed_args.metadata,
            parsed_args.num_samples,
            parsed_args.heritability,
            parsed_args.num_processes,
            parsed_args.run_in_serial,
            parsed_args.chromosome,
            parsed_args.population,
            parsed_args.verbose,
            parsed_args.quiet,
        )
    elif parsed_args.cmd == "clump":
        return parsed_args.func(
            parsed_args.sumstats,
            parsed_args.out,
            parsed_args.metadata,
            parsed_args.num_samples,
            parsed_args.min_chisq,
            parsed_args.max_rsq,
            parsed_args.num_processes,
            parsed_args.run_in_serial,
            parsed_args.chromosome,
            parsed_args.population,
            parsed_args.verbose,
            parsed_args.quiet,
        )
    elif parsed_args.cmd == "surrogates":
        return parsed_args.func(
            parsed_args.sumstats,
            parsed_args.out,
            parsed_args.metadata,
            parsed_args.num_processes,
            parsed_args.run_in_serial,
            parsed_args.population,
            parsed_args.verbose,
            parsed_args.quiet,
            parsed_args.chromosome,
        )
    elif parsed_args.cmd == "simulate":
        return parsed_args.func(
            parsed_args.sumstats_out,
            parsed_args.metadata,
            parsed_args.heritability,
            parsed_args.num_samples,
            parsed_args.component_variance,
            parsed_args.component_weight,
            parsed_args.alpha_param,
            parsed_args.annotation_dependent_polygenicity,
            parsed_args.random_seed,
            parsed_args.annotation_columns,
            parsed_args.num_processes,
            parsed_args.run_in_serial,
            parsed_args.chromosome,
            parsed_args.population,
            parsed_args.verbose,
            parsed_args.quiet,
            parsed_args.annot_dir,
        )
    elif parsed_args.cmd == "reml":
        return parsed_args.func(parsed_args)

    raise ValueError(f"Unknown graphld command: {parsed_args.cmd}")


def run_cli(args: Sequence[str], command_handlers: dict[str, Callable]):
    """Parse arguments, print the CLI preamble, and dispatch to a command."""
    argp = build_parser(command_handlers)
    parsed_args = argp.parse_args(args)
    _write_masthead(parsed_args, argp)
    return dispatch_command(parsed_args)
