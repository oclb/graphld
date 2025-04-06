#!/usr/bin/env python3
import argparse
import os
import sys
import time
from importlib import metadata
from typing import Optional, List, Dict, Union
from pathlib import Path
import numpy as np

import graphld as gld
from graphld.vcf_io import read_gwas_vcf
from graphld.ldsc_io import read_ldsc_sumstats
import polars as pl
from .io import load_annotations
from .heritability import ModelOptions, MethodOptions, run_graphREML

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
    quiet: bool,
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
    if not quiet:
        sys.stdout.write("Running BLUP...\n")
        sys.stdout.flush()
    start_time = time.time()

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
        verbose=verbose,
    )
    
    # Write output
    blup.write_csv(out, separator='\t')

    end_time = time.time()
    if not quiet:
        sys.stdout.write(f"Completed in {end_time - start_time:.2f} s\n")
        sys.stdout.flush()

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
    quiet: bool,
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
    if not quiet:
        sys.stdout.write("Starting clumping...\n")
        sys.stdout.flush()
    start_time = time.time()

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

    end_time = time.time()
    if not quiet:
        sys.stdout.write(f"Completed in {end_time - start_time:.2f} s\n")
        sys.stdout.flush()

def _simulate(
    sumstats_out: str,
    metadata: str,
    heritability: float,
    sample_size: int,
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
    annotations: Optional[str] = None,
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
    if not quiet:
        sys.stdout.write("Running simulate...\n")
        sys.stdout.flush()
    start_time = time.time()

    # Validate input files and parameters
    if not os.path.exists(metadata):
        raise FileNotFoundError(f"Metadata file not found: {metadata}")
    if heritability < 0 or heritability > 1:
        raise ValueError(f"Heritability must be between 0 and 1, got {heritability}")
    if sample_size is None:
        raise ValueError("Sample size must be specified")
    if sample_size < 1:
        raise ValueError(f"Sample size must be >1, got {sample_size}")
    
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
        random_seed=random_seed,
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

    end_time = time.time()
    if not quiet:
        sys.stdout.write(f"Completed in {end_time - start_time:.2f} s\n")
        sys.stdout.flush()

def _detect_sumstats_type(sumstats_path: str, maximum_missingness: float = 1):
    """Detect summary statistics file type based on file extension."""
    from graphld.ldsc_io import read_ldsc_sumstats
    from graphld.vcf_io import read_gwas_vcf
    
    # Lowercase extension for case-insensitive matching
    ext = os.path.splitext(sumstats_path)[1].lower()
    
    if ext in ['.vcf', '.vcf.gz']:
        return read_gwas_vcf(sumstats_path, maximum_missingness=maximum_missingness)
    else:
        return read_ldsc_sumstats(sumstats_path, maximum_missingness=maximum_missingness)

def write_results(filename: str, values: list, std_errors: list, p_values: list, annot_names: list, name: str):
    """Write REML results to a CSV file.
    
    Args:
        filename: Output file path
        values: List of values
        std_errors: List of standard errors
        p_values: List of p-values
        annot_names: List of annotation names, same size as values, std_errors, p_values
        name: Name for the first column
    """
    # Check if the file already has a header and its length
    try:
        with open(filename, 'r') as f:
            header = f.readline().strip()
            has_header = header != ''
            expected_header_length = 1 + 3 * len(values)
            msg = f"Header length {len(header.split(','))} does not match expected length {expected_header_length}."
            if has_header and len(header.split(',')) != expected_header_length:
                raise ValueError(msg)
    except FileNotFoundError:
        has_header = False

    # Open the file in append mode
    with open(filename, 'a') as f:
        if not has_header:
            headers = ['name']
            for annot in annot_names:
                headers.extend([f'{annot}', f'{annot}_SE', f'{annot}_log10pval'])
            f.write(','.join(headers) + '\n')
        
        # Write all data on one row
        row = [name]
        for val, se, pval in zip(values, std_errors, p_values):
            row.extend([str(val), str(se), str(pval)])
        f.write(','.join(row) + '\n')

def write_tall_results(filename: str, model_options: ModelOptions, results: dict):
    """Write REML results in tall format to a CSV file.
    
    Args:
        filename: Output file path
        model_options: Model options containing annotation names
        results: Dictionary containing results from GraphREML
    """
    # Check if file exists
    if os.path.exists(filename):
        raise FileExistsError(f"Output file {filename} already exists")
        
    # Create DataFrame with all metrics
    rows = []
    for i, name in enumerate(model_options.annotation_columns):
        row = {
            'name': name,
            'enrichment': results['enrichment'][i],
            'enrichment_SE': results['enrichment_se'][i],
            'enrichment_log10pval': results['enrichment_log10pval'][i],
            'heritability': results['heritability'][i],
            'heritability_SE': results['heritability_se'][i],
            'heritability_log10pval': results['heritability_log10pval'][i],
            'parameter': results['parameters'][i],
            'parameter_SE': results['parameters_se'][i],
            'parameter_log10pval': results['parameters_log10pval'][i]
        }
        rows.append(row)
    
    # Write to CSV
    with open(filename, 'w') as f:
        headers = ['name', 
                  'enrichment', 'enrichment_SE', 'enrichment_log10pval',
                  'heritability', 'heritability_SE', 'heritability_log10pval',
                  'parameter', 'parameter_SE', 'parameter_log10pval']
        f.write(','.join(headers) + '\n')
        
        for row in rows:
            values = [str(row[h]) for h in headers]
            f.write(','.join(values) + '\n')

def write_convergence_results(filename: str, results: dict):
    """Write REML convergence results to a CSV file.
    
    Args:
        filename: Output file path
        results: Dictionary containing results from GraphREML
    """
    log = results['log']
    
    # Write to CSV
    with open(filename, 'w') as f:
        # Write header
        headers = ['converged', 'num_iterations', 'final_likelihood']
        f.write(','.join(headers) + '\n')
        
        # Write convergence info
        values = [str(log['converged']), str(log['num_iterations']), str(log['final_likelihood'])]
        f.write(','.join(values) + '\n')
        
        # Write iteration-by-iteration info
        f.write('\n')  # Empty line for readability
        f.write('iteration,likelihood_change,trust_region_lambda\n')
        for i, (change, trust_lambda) in enumerate(zip(log['likelihood_changes'], log['trust_region_lambdas'])):
            f.write(f"{i+1},{change},{trust_lambda}\n")

def _reml(args):
    """Run GraphREML command."""
    # Check for existing output files
    tall_output = args.out + '.tall.csv'
    if not args.alt_output:
        if os.path.exists(tall_output):
            raise FileExistsError(f"Output file {tall_output} already exists")
    
    start_time = time.time()
    
    if not args.quiet:
        if args.verbose:
            print('Loading summary statistics from', args.sumstats)
    sumstats = _detect_sumstats_type(args.sumstats, args.maximum_missingness)
    
    if args.verbose:
        print('Loading annotations from', args.annot_dir)
    annotations = load_annotations(args.annot_dir, chromosome=args.chromosome, add_positions=False)
    num_snps_annot = len(annotations)

    # Get annotation columns
    if args.annotation_columns:
        annotation_columns = args.annotation_columns
        for col in annotation_columns:
            if col not in annotations.columns:
                raise ValueError(f"Column {col} not found in annotations")
    else:
        excluded_cols = {'SNP', 'CM', 'BP', 'CHR', 'POS', 'A1', 'A2'}
        annotation_columns = [col for col in annotations.columns if col not in excluded_cols]
    
    # Create model and method options
    model_options = ModelOptions(
        sample_size=args.num_samples,
        intercept=args.intercept,
        annotation_columns=annotation_columns,
        link_fn_denominator=num_snps_annot,
    )
    
    method_options = MethodOptions(
        match_by_position=args.match_by_position,
        num_iterations=args.num_iterations,
        convergence_tol=args.convergence_tol,
        run_serial=args.run_in_serial,
        num_processes=args.num_processes,
        verbose=args.verbose,
        num_jackknife_blocks=args.num_jackknife_blocks,
        reset_trust_region=args.reset_trust_region,
        gradient_num_samples=args.xtrace_num_samples,
        max_chisq_threshold=args.max_chisq_threshold,
        score_test_hdf5_file_name=args.score_test_filename,
        score_test_hdf5_trait_name=args.name or args.sumstats,
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
        print(f"Time to run GraphREML: {runtime:.3f}s")
    if args.verbose:
        print(f"Likelihood changes: {np.diff(np.array(results['likelihood_history']))}")
    
    # Prepare output files
    convergence_file = args.out + '.convergence.csv'
    write_convergence_results(convergence_file, results)

    if args.alt_output:
        heritability_file = args.out + '.heritability.csv'
        write_results(heritability_file, 
                        results['heritability'], 
                        results['heritability_se'],
                        results['heritability_log10pval'],
                        model_options.annotation_columns,
                        args.name or args.sumstats)
        
        enrichment_file = args.out + '.enrichment.csv'
        write_results(enrichment_file, 
                        results['enrichment'], 
                        results['enrichment_se'],
                        results['enrichment_log10pval'],
                        model_options.annotation_columns,
                        args.name or args.sumstats)
        
        parameters_file = args.out + '.parameters.csv'    
        write_results(parameters_file, 
                        results['parameters'], 
                        results['parameters_se'],
                        results['parameters_log10pval'],
                        model_options.annotation_columns,
                        args.name or args.sumstats)
    else:
        write_tall_results(tall_output, model_options, results)
    
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
    parser.add_argument(
        '--maximum-missingness',
        type=float,
        default=0.1,
        help='Maximum fraction of missing samples allowed (default: 0.1)',
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
        "-a", '--annot-dir',
        help='Path to annotation directory. Must contain per-chromosome .annot files, can also contain .bed files',
        required=True,
    )

    # Add common arguments
    _add_common_arguments(parser)

    # Optional arguments specific to reml
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
        default=False,
    )
    parser.add_argument(
        "--annotation-columns", 
        type=lambda s: s.split(','),
        default=None,
        help="Annotation columns"
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
    parsed_args = argp.parse_args(args)

    # Pull passed arguments/options as a string for printing
    cmd_str = _construct_cmd_string(parsed_args, argp)

    # Setup version string
    version = f"v{metadata.version('graphld')}"
    masthead = f"""
    ***********************************************************************
    ************************** GraphLD {version} *****************************
    ***********************************************************************

    """

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
            parsed_args.quiet,
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
            parsed_args.quiet,
        )
    elif parsed_args.cmd == "simulate":
        return _simulate(
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
        return _reml(parsed_args)

def main():
    """Entry point for the graphld command line interface."""
    _main(sys.argv[1:])

if __name__ == "__main__":
    main()
