#!/usr/bin/env python3
import argparse
import os
import sys
from importlib import metadata
from typing import Optional, List, Dict, Union
from pathlib import Path
import time
import numpy as np

import graphld as gld
from graphld.vcf_io import read_gwas_vcf
from graphld.ldsc_io import read_ldsc_sumstats
import polars as pl
from .io import load_annotations
from .heritability import ModelOptions, MethodOptions, run_graphREML

title = """
██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗██╗     ██████╗ 
██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██║     ██╔══██╗
██║  ███╗██████╔╝███████║██████╔╝███████║██║     ██║  ██║
██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║██║     ██║  ██║
╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║███████╗██████╔╝
 ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝╚═════╝ 
"""

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

def _blup(
    sumstats: str,
    out: str,
    metadata: str,
    num_samples: Optional[int],
    heritability: float,
    num_processes: Optional[int],
    run_in_serial: bool,
    chromosome: Optional[int],
    population: Optional[str],
    verbose: bool,
    quiet: bool
) -> None:
    """Run BLUP (Best Linear Unbiased Prediction) command.
    
    Args:
        sumstats: Path to summary statistics file (.vcf or .sumstats)
        out: Output file path
        metadata: Path to LDGM metadata file
        num_samples: Optional sample size override
        heritability: Heritability parameter (between 0 and 1)
        num_processes: Number of processes for parallel computation
        run_in_serial: Whether to run in serial mode
        chromosome: Optional chromosome to filter analysis
        population: Optional population to filter analysis
        verbose: Whether to print verbose output
        quiet: Whether to suppress all output except errors
    
    Raises:
        ValueError: If input file format is invalid or required columns are missing
        FileNotFoundError: If input files don't exist
    """
    # Validate input files exist
    if not os.path.exists(sumstats):
        raise FileNotFoundError(f"Summary statistics file not found: {sumstats}")
    if not os.path.exists(metadata):
        raise FileNotFoundError(f"Metadata file not found: {metadata}")
    
    # Validate heritability
    if not 0 <= heritability <= 1:
        raise ValueError(f"Heritability must be between 0 and 1, got {heritability}")
    
    # Determine input format and read data
    if sumstats.endswith('.vcf'):
        sumstats = read_gwas_vcf(sumstats)
        match_by_position = True
        sample_size_col = 'NS'
    elif sumstats.endswith('.sumstats'):
        sumstats = read_ldsc_sumstats(sumstats)
        match_by_position = False
        sample_size_col = 'N'
    else:
        raise ValueError("Input file must end in .vcf or .sumstats")
    
    # Get sample size
    sample_size = num_samples
    if sample_size is None:
        if sample_size_col not in sumstats.columns:
            raise ValueError(f"Sample size column '{sample_size_col}' not found in input file")
        sample_size = sumstats[sample_size_col].median()
    
    # Run BLUP
    blup = gld.BLUP.compute_blup(
        ldgm_metadata_path=metadata,
        sumstats=sumstats,
        sigmasq=heritability,
        sample_size=sample_size,
        num_processes=num_processes,
        run_in_serial=run_in_serial,
        match_by_position=match_by_position,
        chromosomes=chromosome,
        populations=population,
        verbose=verbose
    )
    
    # Write output
    blup.write_csv(out, separator='\t')

def _clump(
    sumstats: str,
    out: str,
    metadata: str,
    num_samples: Optional[int],
    min_chisq: float,
    max_rsq: float,
    num_processes: Optional[int],
    run_in_serial: bool,
    chromosome: Optional[int],
    population: Optional[str],
    verbose: bool,
    quiet: bool
) -> None:
    """Run LD clumping command to identify independent variants.
    
    Args:
        sumstats: Path to summary statistics file (.vcf or .sumstats)
        out: Output file path
        metadata: Path to LDGM metadata file
        num_samples: Optional sample size override
        min_chisq: Minimum chi-squared value for variant inclusion
        max_rsq: Maximum R-squared threshold for LD pruning
        num_processes: Number of processes for parallel computation
        run_in_serial: Whether to run in serial mode
        chromosome: Optional chromosome to filter analysis
        population: Optional population to filter analysis
        verbose: Whether to print verbose output
        quiet: Whether to suppress all output except errors
    
    Raises:
        ValueError: If input file format is invalid
        FileNotFoundError: If input files don't exist
    """
    # Validate input files exist
    if not os.path.exists(sumstats):
        raise FileNotFoundError(f"Summary statistics file not found: {sumstats}")
    if not os.path.exists(metadata):
        raise FileNotFoundError(f"Metadata file not found: {metadata}")
    
    # Validate thresholds
    if min_chisq < 0:
        raise ValueError(f"min_chisq must be non-negative, got {min_chisq}")
    if not 0 <= max_rsq <= 1:
        raise ValueError(f"max_rsq must be between 0 and 1, got {max_rsq}")
    
    # Read summary statistics
    if sumstats.endswith('.vcf'):
        sumstats_df = read_gwas_vcf(sumstats)
    elif sumstats.endswith('.sumstats'):
        sumstats_df = read_ldsc_sumstats(sumstats)
    else:
        raise ValueError("Input file must end in .vcf or .sumstats")
    
    # Run LD clumping
    clumped = gld.LDClumper.clump(
        ldgm_metadata_path=metadata,
        sumstats=sumstats_df,
        rsq_threshold=max_rsq,
        chisq_threshold=min_chisq,
        num_processes=num_processes,
        run_in_serial=run_in_serial,
        chromosomes=chromosome,
        populations=population,
        verbose=verbose,
    ).filter(pl.col('is_index')).drop('is_index')
    
    # Write output
    clumped.write_csv(out, separator='\t')

def _simulate(
    sumstats_out: str,
    metadata: str,
    heritability: float,
    component_variance: Optional[List[float]] = None,
    component_weight: Optional[List[float]] = None,
    alpha_param: float = -1,
    annotation_dependent_polygenicity: bool = False,
    random_seed: Optional[int] = None,
    annotation_columns: Optional[List[str]] = None,
    num_processes: Optional[int] = None,
    run_in_serial: bool = False,
    chromosome: Optional[int] = None,
    population: Optional[str] = None,
    verbose: bool = False,
    quiet: bool = False,
    sample_size: int = 1000,
    annotations: Optional[str] = None
) -> None:
    """Run genetic simulation with configurable parameters.
    
    Args:
        sumstats_out: Path to output summary statistics file
        metadata: Path to LDGM metadata file
        heritability: Total heritability of simulated trait
        component_variance: List of variance components
        component_weight: List of weights for components
        alpha_param: Alpha parameter for polygenicity
        annotation_dependent_polygenicity: Whether to use annotation-dependent polygenicity
        random_seed: Optional seed for reproducibility
        annotation_columns: Optional list of annotation columns
        num_processes: Number of processes for parallel computation
        run_in_serial: Whether to run in serial mode
        chromosome: Optional chromosome to filter analysis
        population: Optional population to filter analysis
        verbose: Whether to print verbose output
        quiet: Whether to suppress all output except errors
        sample_size: Number of samples to simulate (default: 1000)
        annotations: Path to annotation file for simulation
    
    Raises:
        ValueError: If parameters are invalid
        FileNotFoundError: If required files don't exist
    """
    # Validate input files and parameters
    if not os.path.exists(metadata):
        raise FileNotFoundError(f"Metadata file not found: {metadata}")
    if heritability < 0 or heritability > 1:
        raise ValueError(f"Heritability must be between 0 and 1, got {heritability}")
    if sample_size < 1:
        raise ValueError(f"Sample size must be positive, got {sample_size}")
    
    # Load annotations if needed
    annotations_df = None
    if annotations:
        annotations_df = load_annotations(
            annotations,
            chromosome,
            add_alleles=True,
        )
    
    # Create Simulate object
    sim = gld.Simulate(
        sample_size=sample_size, 
        heritability=heritability,
        component_variance=component_variance or [1.0],
        component_weight=component_weight or [1.0],
        alpha_param=alpha_param,
        annotation_dependent_polygenicity=annotation_dependent_polygenicity,
        random_seed=random_seed
    )
    
    # Run simulation
    sim_result = sim.simulate(
        ldgm_metadata_path=metadata,
        populations=population,
        chromosomes=chromosome,
        annotations=annotations_df,
        run_in_serial=run_in_serial,
        num_processes=num_processes,
        verbose=verbose,
    )
    
    # Write output
    sim_result.write_csv(sumstats_out, separator='\t')

def _detect_sumstats_type(sumstats_path: str):
    """Detect summary statistics file type based on file extension."""
    from graphld.ldsc_io import read_ldsc_sumstats
    from graphld.vcf_io import read_gwas_vcf
    
    # Lowercase extension for case-insensitive matching
    ext = os.path.splitext(sumstats_path)[1].lower()
    
    if ext in ['.vcf', '.vcf.gz']:
        return read_gwas_vcf(sumstats_path)
    else:
        return read_ldsc_sumstats(sumstats_path)

def _reml(args):
    """Run GraphREML command."""
    start_time = time.time()
    
    if not args.quiet:
        if args.verbose:
            print('Loading summary statistics from', args.sumstats)
    sumstats = _detect_sumstats_type(args.sumstats)
    
    if args.verbose:
        print('Loading annotations from', args.annot)
    annotations = load_annotations(args.annot, chromosome=args.chromosome)
    num_snps_annot = len(annotations)

    # Get annotation columns
    excluded_cols = {'SNP', 'CM', 'BP', 'CHR', 'POS', 'A1', 'A2'}
    annotation_columns = [col for col in annotations.columns if col not in excluded_cols]
    
    # Create model and method options
    model_options = ModelOptions(
        sample_size=args.num_samples or 1,
        intercept=args.intercept,
        annotation_columns=annotation_columns,
        link_fn_denominator=num_snps_annot,
    )
    
    method_options = MethodOptions(
        match_by_position=not args.match_by_rsid,  # Note: inverted since we changed default
        num_iterations=args.num_iterations,
        convergence_tol=args.convergence_tol,
        run_serial=args.run_in_serial,
        num_processes=args.num_processes,
        verbose=args.verbose,
        num_jackknife_blocks=args.num_jackknife_blocks,
    )
    
    # Run GraphREML
    if not args.quiet:
        print('Running GraphREML...')

    
    results = run_graphREML(
        model_options=model_options,
        method_options=method_options,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=args.metadata,
        populations=[args.population] if args.population else None,
        chromosomes=[args.chromosome] if args.chromosome else None,
    )
    
    runtime = time.time() - start_time
    if not args.quiet:
        if args.verbose:
            print(f"Time to run GraphREML: {runtime:.3f}s")
            print(f"Estimated heritability: {results['heritability']}")
            print(f"SE of estimated heritability: {results['heritability_se']}")
            print(f"Estimated enrichment: {results['enrichment']}")
            print(f"SE of estimated enrichment: {results['enrichment_se']}")
            print(f"Likelihood changes: {np.diff(np.array(results['likelihood_history']))}")
        else:
            print(f"Finished GraphREML analysis in {runtime:.3f}s")
    
    # Prepare output files
    heritability_file = args.out + '.heritability.csv'
    enrichment_file = args.out + '.enrichment.csv'
    parameters_file = args.out + '.parameters.csv'
    
    # Function to write results
    def write_results(filename, values, std_errors, header_prefix=''):
        file_exists = Path(filename).exists()
        mode = 'a' if file_exists else 'w'
        with open(filename, mode) as f:
            if not file_exists:
                # Write header
                headers = ['Name', 'File']
                for col in annotation_columns:
                    headers.extend([f'{header_prefix}{col}', f'SE_{header_prefix}{col}'])
                f.write(','.join(headers) + '\n')
            
            # Write data
            row = [args.name or 'NA', args.sumstats]
            for val, se in zip(values, std_errors):
                row.extend([str(val), str(se)])
            f.write(','.join(row) + '\n')
    
    write_results(heritability_file, 
                 results['heritability'], 
                 results.get('heritability_se', [0]*len(results['heritability'])))
    
    write_results(enrichment_file, 
                 results['enrichment'], 
                 results.get('enrichment_se', [0]*len(results['enrichment'])))
    
    write_results(parameters_file, 
                    results['params'], 
                    results.get('param_se', [0]*len(results['params'])))

def _add_common_arguments(parser):
    """Add arguments that are common to all subcommands."""
    parser.add_argument("-n", "--num_samples", type=int,
                      help="Sample size (default: inferred from sumstats)")
    parser.add_argument("--metadata", type=str, default="data/ldgms/metadata.csv",
                      help="Path to LDGM metadata file")
    parser.add_argument("--num_processes", type=int,
                      help="Number of processes (default: None)")
    parser.add_argument("--run_in_serial", action="store_true",
                      help="Run in serial mode")
    parser.add_argument("-c", "--chromosome", type=int, default=None,
                      help="Chromosome to filter analysis")
    parser.add_argument("-p", "--population", type=str, default="EUR",
                      help="Population to filter analysis")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-q", "--quiet", action="store_true", default=False)

def _add_io_arguments(parser):
    """Add common input/output arguments."""
    parser.add_argument(
        'sumstats',
        help='Path to summary statistics file (.vcf or .sumstats)',
    )
    parser.add_argument(
        'out',
        help='Output file path',
    )

def _add_blup_parser(subparsers):
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
        help='Heritability parameter (between 0 and 1)',
    )
    
    parser.set_defaults(func=_blup)

def _add_clump_parser(subparsers):
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
        '--min_chisq',
        type=float,
        default=0,
        help='Minimum chi-squared value for variant inclusion',
    )
    parser.add_argument(
        '--max_rsq',
        type=float,
        default=0.1,
        help='Maximum R-squared threshold for LD pruning',
    )
    
    parser.set_defaults(func=_clump)

def _add_simulate_parser(subparsers):
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
        "--component_variance", 
        type=lambda s: [float(x) for x in s.split(',')],
        default=[1.0],
        help="Component variance (default: [1.0])"
    )
    parser.add_argument(
        "--component_weight", 
        type=lambda s: [float(x) for x in s.split(',')],
        default=[1.0],
        help="Component weight (default: [1.0])"
    )
    parser.add_argument(
        "--alpha_param", 
        type=float, 
        default=-0.5,
        help="Alpha parameter (default: -0.5)"
    )
    parser.add_argument(
        "--annotation_dependent_polygenicity", 
        action="store_true",
        help="Annotation dependent polygenicity"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=None,
        help="Random seed (default: None)"
    )
    parser.add_argument(
        "--annotation_columns", 
        type=lambda s: s.split(','),
        default=None,
        help="Annotation columns"
    )
    parser.add_argument(
        "-a", "--annot", 
        type=str, 
        default=None,
        help="Directory containing annotation files ending in .annot"
    )
    
    parser.set_defaults(func=_simulate)

def _add_reml_parser(subparsers):
    """Add parser for reml command."""
    parser = subparsers.add_parser(
        'reml',
        help='Run GraphREML',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add common I/O arguments
    _add_io_arguments(parser)

    # Required arguments
    parser.add_argument(
        "-a", '--annot',
        help='Path to annotation directory',
        required=True,
    )

    # Add common arguments
    _add_common_arguments(parser)

    # Optional arguments specific to reml
    parser.add_argument(
        '--name',
        help='Name for this analysis (will be included in output files)',
        default=None,
    )
    parser.add_argument(
        '--intercept',
        help='LD score regression intercept',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '--num_iterations',
        help='Maximum number of iterations',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--convergence_tol',
        help='Convergence tolerance',
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        '--num_jackknife_blocks',
        help='Number of jackknife blocks',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--match-by-rsid',
        help='Match variants by RSID instead of position',
        action='store_true',
        default=False,
    )

    parser.set_defaults(func=_reml)

def _main(args):
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Add common arguments to main parser
    _add_common_arguments(argp)

    # Subparsers
    subp = argp.add_subparsers(dest="cmd", required=True, help="Subcommands for graphld")

    # BLUP command
    _add_blup_parser(subp)

    # LD clumping command
    _add_clump_parser(subp)

    # Genetic simulation command
    _add_simulate_parser(subp)

    # GraphREML command
    _add_reml_parser(subp)

    # Parse arguments
    parsed_args, additional_args = argp.parse_known_args(args)

    # Merge known and additional arguments
    if parsed_args.cmd == "blup":
        # Add additional arguments to the parsed arguments
        for i in range(0, len(additional_args), 2):
            if i+1 < len(additional_args):
                if additional_args[i] in ['-n', '--num_samples']:
                    parsed_args.num_samples = int(additional_args[i+1])
                elif additional_args[i] in ['--metadata']:
                    parsed_args.metadata = additional_args[i+1]
                elif additional_args[i] in ['--num_processes']:
                    parsed_args.num_processes = int(additional_args[i+1])
                elif additional_args[i] in ['--run_in_serial']:
                    parsed_args.run_in_serial = True
                elif additional_args[i] in ['-c', '--chromosome']:
                    parsed_args.chromosome = int(additional_args[i+1])
                elif additional_args[i] in ['-p', '--population']:
                    parsed_args.population = additional_args[i+1]
                elif additional_args[i] in ['-H', '--heritability']:
                    parsed_args.heritability = float(additional_args[i+1])
                elif additional_args[i] in ['-v', '--verbose']:
                    parsed_args.verbose = True
                elif additional_args[i] in ['-q', '--quiet']:
                    parsed_args.quiet = True
    elif parsed_args.cmd == "clump":
        # Add additional arguments to the parsed arguments
        for i in range(0, len(additional_args), 2):
            if i+1 < len(additional_args):
                if additional_args[i] in ['-n', '--num_samples']:
                    parsed_args.num_samples = int(additional_args[i+1])
                elif additional_args[i] in ['--metadata']:
                    parsed_args.metadata = additional_args[i+1]
                elif additional_args[i] in ['--num_processes']:
                    parsed_args.num_processes = int(additional_args[i+1])
                elif additional_args[i] in ['--run_in_serial']:
                    parsed_args.run_in_serial = True
                elif additional_args[i] in ['-c', '--chromosome']:
                    parsed_args.chromosome = int(additional_args[i+1])
                elif additional_args[i] in ['-p', '--population']:
                    parsed_args.population = additional_args[i+1]
                elif additional_args[i] in ['-m', '--min_chisq']:
                    parsed_args.min_chisq = float(additional_args[i+1])
                elif additional_args[i] in ['-r', '--max_rsq']:
                    parsed_args.max_rsq = float(additional_args[i+1])
                elif additional_args[i] in ['-v', '--verbose']:
                    parsed_args.verbose = True
                elif additional_args[i] in ['-q', '--quiet']:
                    parsed_args.quiet = True
    elif parsed_args.cmd == "simulate":
        # Add additional arguments to the parsed arguments
        for i in range(0, len(additional_args), 2):
            if i+1 < len(additional_args):
                if additional_args[i] in ['--num_processes']:
                    parsed_args.num_processes = int(additional_args[i+1])
                elif additional_args[i] in ['--run_in_serial']:
                    parsed_args.run_in_serial = True
                elif additional_args[i] in ['-c', '--chromosome']:
                    parsed_args.chromosome = int(additional_args[i+1])
                elif additional_args[i] in ['-p', '--population']:
                    parsed_args.population = additional_args[i+1]
                elif additional_args[i] in ['-H', '--heritability']:
                    parsed_args.heritability = float(additional_args[i+1])
                elif additional_args[i] in ['--component_variance']:
                    parsed_args.component_variance = [float(x) for x in additional_args[i+1].split(',')]
                elif additional_args[i] in ['--component_weight']:
                    parsed_args.component_weight = [float(x) for x in additional_args[i+1].split(',')]
                elif additional_args[i] in ['--alpha_param']:
                    parsed_args.alpha_param = float(additional_args[i+1])
                elif additional_args[i] in ['--annotation_dependent_polygenicity']:
                    parsed_args.annotation_dependent_polygenicity = True
                elif additional_args[i] in ['--random_seed']:
                    parsed_args.random_seed = int(additional_args[i+1])
                elif additional_args[i] in ['--annotation_columns']:
                    parsed_args.annotation_columns = additional_args[i+1].split(',')
                elif additional_args[i] in ['-v', '--verbose']:
                    parsed_args.verbose = True
                elif additional_args[i] in ['-q', '--quiet']:
                    parsed_args.quiet = True
                elif additional_args[i] in ['-n', '--num_samples']:
                    parsed_args.num_samples = int(additional_args[i+1])
                elif additional_args[i] in ['-a', '--annot']:
                    parsed_args.annotations = additional_args[i+1]
    elif parsed_args.cmd == "reml":
        # Add additional arguments to the parsed arguments
        for i in range(0, len(additional_args), 2):
            if i+1 < len(additional_args):
                if additional_args[i] in ['--populations']:
                    parsed_args.populations = additional_args[i+1].split(',')
                elif additional_args[i] in ['--chromosomes']:
                    parsed_args.chromosomes = [int(x) for x in additional_args[i+1].split(',')]
                elif additional_args[i] in ['--sample-size']:
                    parsed_args.sample_size = float(additional_args[i+1])
                elif additional_args[i] in ['--intercept']:
                    parsed_args.intercept = float(additional_args[i+1])
                elif additional_args[i] in ['--match-by-position']:
                    parsed_args.match_by_position = True
                elif additional_args[i] in ['--num-iterations']:
                    parsed_args.num_iterations = int(additional_args[i+1])
                elif additional_args[i] in ['--convergence-tol']:
                    parsed_args.convergence_tol = float(additional_args[i+1])
                elif additional_args[i] in ['--run-serial']:
                    parsed_args.run_serial = True
                elif additional_args[i] in ['--num-processes']:
                    parsed_args.num_processes = int(additional_args[i+1])
                elif additional_args[i] in ['--verbose']:
                    parsed_args.verbose = True
                elif additional_args[i] in ['--num-jackknife-blocks']:
                    parsed_args.num_jackknife_blocks = int(additional_args[i+1])

    # Pull passed arguments/options as a string for printing
    cmd_str = _construct_cmd_string(parsed_args, argp)

    # Setup version string
    version = f"v{metadata.version('graphld')}"
    buff_size = (87 + 22 + 4 - len(version)) // 2
    version = (" " * buff_size) + version + (" " * buff_size)
    title_and_ver = f"{title}{os.linesep}{version}"
    masthead = title_and_ver + os.linesep

    if not parsed_args.quiet:
        sys.stdout.write(masthead)
        sys.stdout.write(cmd_str + os.linesep)

    # Execute the command
    if parsed_args.cmd == "blup":
        return _blup(
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
            parsed_args.quiet
        )
    elif parsed_args.cmd == "clump":
        return _clump(
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
            parsed_args.quiet
        )
    elif parsed_args.cmd == "simulate":
        return _simulate(
            parsed_args.sumstats_out, 
            parsed_args.metadata, 
            parsed_args.heritability, 
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
            parsed_args.num_samples,
            parsed_args.annotations
        )
    elif parsed_args.cmd == "reml":
        return _reml(parsed_args)

def main():
    """Entry point for the graphld command line interface."""
    _main(sys.argv[1:])

if __name__ == "__main__":
    main()
