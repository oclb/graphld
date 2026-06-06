"""Argument parser construction for the GraphLD CLI."""

import argparse
from collections.abc import Callable, Mapping


def _add_common_arguments(parser):
    """Add arguments that are common to all subcommands."""
    parser.add_argument("-n", "--num-samples", type=int,
                      help="Sample size")
    parser.add_argument("--metadata", type=str, default="data/ldgms/metadata.csv",
                      help="Path to LDGM metadata file")
    parser.add_argument("--num-processes", type=int,
                      help="Number of processes (default: None)")
    parser.add_argument("--run-in-serial", action="store_true",
                      help="Run in serial mode")
    parser.add_argument("-c", "--chromosome", type=int, default=None,
                      help="Chromosome to filter analysis")
    parser.add_argument("-p", "--population", type=str, default="EUR",
                      help="Population to filter analysis")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-q", "--quiet", action="store_true", default=False)


def _add_io_arguments(parser, out_required=True):
    """Add common input/output arguments."""
    parser.add_argument(
        'sumstats',
        help='Path to summary statistics file (.vcf or .sumstats)',
    )
    if out_required:
        parser.add_argument(
            'out',
            help='Output file path',
        )
    else:
        parser.add_argument(
            'out',
            nargs='?',
            default=None,
            help='Output file path (optional)',
        )
    parser.add_argument(
        '--maximum-missingness',
        type=float,
        default=0.1,
        help='Maximum fraction of missing samples allowed (default: 0.1)',
    )


def _set_handler(parser, handler: Callable | None) -> None:
    if handler is not None:
        parser.set_defaults(func=handler)


def _add_blup_parser(subparsers, handler: Callable | None = None):
    """Add parser for blup command."""
    parser = subparsers.add_parser(
        'blup',
        help='Run BLUP',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add common I/O arguments
    _add_io_arguments(parser)

    # Add common arguments
    _add_common_arguments(parser)

    # Add BLUP-specific arguments
    parser.add_argument(
        "-H", '--heritability',
        type=float,
        required=True,
        help='Heritability for the analyzed variant scope (between 0 and 1)',
    )

    _set_handler(parser, handler)


def _add_surrogates_parser(subparsers, handler: Callable | None = None):
    """Add parser for surrogates command."""
    parser = subparsers.add_parser(
        'surrogates',
        help='Find surrogate markers',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add common I/O arguments
    _add_io_arguments(parser)

    # Add common arguments
    _add_common_arguments(parser)

    _set_handler(parser, handler)


def _add_clump_parser(subparsers, handler: Callable | None = None):
    """Add parser for clump command."""
    parser = subparsers.add_parser(
        'clump',
        help='Run LD clumping',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add common I/O arguments
    _add_io_arguments(parser)

    # Add common arguments
    _add_common_arguments(parser)

    # Add clump-specific arguments
    parser.add_argument(
        '--min-chisq',
        type=float,
        default=30,
        help='Minimum chi-squared value for variant inclusion',
    )
    parser.add_argument(
        '--max-rsq',
        type=float,
        default=0.1,
        help='Maximum R-squared threshold for LD pruning',
    )

    _set_handler(parser, handler)


def _add_simulate_parser(subparsers, handler: Callable | None = None):
    """Add parser for simulate command."""
    parser = subparsers.add_parser(
        "simulate",
        help="Perform genetic simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Single positional argument for output sumstats
    parser.add_argument(
        'sumstats_out',
        help='Path to output summary statistics file'
    )

    # Add common arguments
    _add_common_arguments(parser)

    # Simulation-specific arguments
    parser.add_argument(
        "-H", "--heritability",
        type=float,
        default=0.2,
        help="Heritability (default: 0.2)"
    )
    parser.add_argument(
        "--component-variance",
        type=lambda s: [float(x) for x in s.split(',')],
        default=[1.0],
        help="Component variance (default: [1.0])"
    )
    parser.add_argument(
        "--component-weight",
        type=lambda s: [float(x) for x in s.split(',')],
        default=[1.0],
        help="Component weight (default: [1.0])"
    )
    parser.add_argument(
        "--alpha-param",
        type=float,
        default=-0.5,
        help="Alpha parameter (default: -0.5)"
    )
    parser.add_argument(
        "--annotation-dependent-polygenicity",
        action="store_true",
        help="Annotation dependent polygenicity"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed (default: None)"
    )
    parser.add_argument(
        "--annotation-columns",
        type=lambda s: s.split(','),
        default=None,
        help="Annotation columns"
    )
    parser.add_argument(
        "-a", "--annot-dir",
        type=str,
        default=None,
        help="Directory containing annotation files ending in .annot"
    )

    _set_handler(parser, handler)


def _add_reml_parser(subparsers, handler: Callable | None = None):
    """Add parser for reml command."""
    parser = subparsers.add_parser(
        'reml',
        help='Run GraphREML',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add common I/O arguments (out is optional for reml)
    _add_io_arguments(parser, out_required=False)

    # Annotation arguments (one of these is required)
    parser.add_argument(
        "-a", '--annot-dir',
        help='Path to annotation directory. Must contain per-chromosome .annot files, can also contain .bed files. '
             'Either --annot-dir or --gene-annot-dir must be provided.',
        default=None,
    )

    # Add common arguments
    _add_common_arguments(parser)

    # Optional arguments specific to reml
    parser.add_argument(
        '--no-save',
        help='Do not save results (if output filename is not provided) or only save logs (if output filename is provided)',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--name',
        help='Name for this analysis, used in --alt-output files and in score test .hdf5 files',
        default=None,
    )
    parser.add_argument(
        '--intercept',
        help='LD score regression intercept',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '--num-iterations',
        help='Maximum number of iterations',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--convergence-tol',
        help='Convergence tolerance',
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        '--convergence-window',
        help='Number of iterations to consider for convergence',
        type=int,
        default=3,
    )
    parser.add_argument(
        '--num-jackknife-blocks',
        help='Number of jackknife blocks',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--match-by-position',
        help='Match variants by position instead of RSID',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--reset-trust-region',
        help='Reset trust region size to initial value at every iteration',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--xtrace-num-samples',
        help='Number of samples for gradient estimation',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--max-chisq-threshold',
        help='Maximum allowed chi^2 value in a block. Blocks with chi^2 > threshold are excluded.',
        type=float,
        default=None,
    )
    parser.add_argument(
        '--alt-output',
        help='Write results in wide format with separate files for heritability, enrichment, and parameters (default is tall format with all metrics in one file)',
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--score-test-filename',
        help='Name of the hdf5 file that will contain precomputed statistics for the graphREML enrichment score test',
        type=str,
        default=None,
    )
    parser.add_argument(
        "--annotation-columns",
        type=lambda s: s.split(','),
        default=None,
        help="Annotation columns"
    )
    parser.add_argument(
        "--binary-annotations-only",
        action="store_true",
        help="Only include annotations that are binary (0/1 valued)"
    )

    # Optional path to surrogate markers HDF5 (produced by `graphld surrogates`)
    parser.add_argument(
        "--surrogates",
        type=str,
        default=None,
        help="Path to surrogate markers HDF5 file to use during GraphREML"
    )
    parser.add_argument(
        "--initial-params",
        type=lambda s: [float(x) for x in s.split(',')],
        default=None,
        help="Initial parameter values (comma-separated). If fewer than num_params, remaining are set to 0."
    )

    # Gene-level annotation options
    parser.add_argument(
        "-g", "--gene-annot-dir",
        type=str,
        default=None,
        help="Directory containing gene-level annotation files (.gmt format). "
             "Gene sets will be converted to variant-level annotations using nearest-gene weighting."
    )
    parser.add_argument(
        "--gene-table",
        type=str,
        default="data/genes.tsv",
        help="Path to gene table TSV file with columns: gene_id, gene_name, start, end, CHR. "
             "Required when using --gene-annot-dir."
    )
    parser.add_argument(
        "--nearest-weights",
        type=str,
        default="0.4,0.2,0.1,0.1,0.1,0.05,0.05",
        help="Comma-separated weights for k-nearest genes when converting gene annotations to variant annotations. "
             "Default assigns decreasing weights to the 7 nearest genes."
    )

    _set_handler(parser, handler)


def build_parser(command_handlers: Mapping[str, Callable] | None = None):
    """Build the top-level GraphLD argument parser."""
    command_handlers = command_handlers or {}
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add common arguments to main parser
    _add_common_arguments(argp)

    # Subparsers
    subp = argp.add_subparsers(dest="cmd", required=True, help="Subcommands for graphld")

    # BLUP command
    _add_blup_parser(subp, command_handlers.get("blup"))

    # LD clumping command
    _add_clump_parser(subp, command_handlers.get("clump"))

    # Surrogates command
    _add_surrogates_parser(subp, command_handlers.get("surrogates"))

    # Genetic simulation command
    _add_simulate_parser(subp, command_handlers.get("simulate"))

    # GraphREML command
    _add_reml_parser(subp, command_handlers.get("reml"))

    return argp
