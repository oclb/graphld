#!/usr/bin/env python3
import os
import sys
import time
from typing import List, Optional

import numpy as np
import polars as pl

import graphld as gld
from graphld.ldsc_io import read_ldsc_sumstats, read_ldsc_snplist
from graphld.parquet_io import get_parquet_traits, read_parquet_sumstats
from graphld.vcf_io import read_gwas_vcf

from .heritability import MethodOptions, ModelOptions, run_graphREML
from .io import load_annotations
from .surrogates import get_surrogate_markers
from ._cli_dispatch import _construct_cmd_string, dispatch_command, run_cli
from ._cli_parser import (
    _add_blup_parser as _parser_add_blup_parser,
    _add_clump_parser as _parser_add_clump_parser,
    _add_common_arguments,
    _add_io_arguments,
    _add_reml_parser as _parser_add_reml_parser,
    _add_simulate_parser as _parser_add_simulate_parser,
    _add_surrogates_parser as _parser_add_surrogates_parser,
    build_parser as _parser_build_parser,
)

__all__ = [
    "_add_blup_parser",
    "_add_clump_parser",
    "_add_common_arguments",
    "_add_io_arguments",
    "_add_reml_parser",
    "_add_simulate_parser",
    "_add_surrogates_parser",
    "_blup",
    "_clump",
    "_construct_cmd_string",
    "_detect_sumstats_type",
    "_main",
    "_reml",
    "_run_reml_single_trait",
    "_simulate",
    "_surrogates",
    "build_parser",
    "dispatch_command",
    "main",
    "run_cli",
    "write_convergence_results",
    "write_results",
    "write_tall_results",
]


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
        heritability: Heritability for the analyzed variant scope (between 0 and 1)
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
    elif sumstats.endswith('.parquet'):
        sumstats = read_parquet_sumstats(sumstats)
        match_by_position = False
        sample_size_col = 'N'
    elif sumstats.endswith('.sumstats'):
        sumstats = read_ldsc_sumstats(sumstats)
        match_by_position = False
        sample_size_col = 'N'
    else:
        raise ValueError("Input file must end in .vcf, .parquet, or .sumstats")

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
        heritability=heritability,
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
    elif sumstats.endswith('.parquet'):
        sumstats_df = read_parquet_sumstats(sumstats)
    elif sumstats.endswith('.sumstats'):
        sumstats_df = read_ldsc_sumstats(sumstats)
    else:
        raise ValueError("Input file must end in .vcf, .parquet, or .sumstats")

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

def _surrogates(
    sumstats: str,
    out: str,
    metadata: str,
    num_processes: Optional[int],
    run_in_serial: bool,
    population: Optional[str],
    verbose: bool,
    quiet: bool,
    chromosome: Optional[int],
) -> None:
    """Run surrogate marker identification command.
    
    Args:
        sumstats: Path to summary statistics file (.vcf or .sumstats)
        out: Output file path
        metadata: Path to LDGM metadata file
        num_processes: Number of processes for parallel computation
        run_in_serial: Whether to run in serial mode
        population: Population to use for surrogate analysis
        verbose: Whether to print verbose output
        quiet: Whether to suppress all output except errors
    
    Raises:
        ValueError: If input file format is invalid or population is not provided
        FileNotFoundError: If input files don't exist
    """
    if not quiet:
        sys.stdout.write("Starting surrogate marker identification...\n")
        sys.stdout.flush()
    start_time = time.time()

    if population is None:
        raise ValueError("--population is required for the surrogates command.")

    # Validate input files exist
    if not os.path.exists(sumstats):
        raise FileNotFoundError(f"Summary statistics file not found: {sumstats}")
    if not os.path.exists(metadata):
        raise FileNotFoundError(f"Metadata file not found: {metadata}")

    # Read summary statistics
    if sumstats.endswith('.vcf'):
        sumstats_df = read_gwas_vcf(sumstats)
    elif sumstats.endswith('.parquet'):
        sumstats_df = read_parquet_sumstats(sumstats)
    elif sumstats.endswith('.snplist'):
        sumstats_df = read_ldsc_snplist(sumstats)
    elif sumstats.endswith('.sumstats'):
        sumstats_df = read_ldsc_sumstats(sumstats)
    else:
        raise ValueError("Input file must end in .vcf, .parquet, .sumstats, or .snplist")

    # Run surrogate marker identification -> writes an HDF5 file and returns its path
    out_path = get_surrogate_markers(
        metadata_path=metadata,
        nonmissing_variant_ids=sumstats_df,
        population=population,
        run_serial=run_in_serial,
        num_processes=num_processes,
        output_path=out,
        chromosomes=chromosome,
    )

    if not quiet:
        sys.stdout.write(f"Wrote surrogates to {out_path}\n")
        sys.stdout.flush()

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

def _detect_sumstats_type(sumstats_path: str, maximum_missingness: float = 1, trait: Optional[str] = None):
    """Detect summary statistics file type based on file extension.
    
    Args:
        sumstats_path: Path to summary statistics file
        maximum_missingness: Maximum fraction of missing samples allowed
        trait: For parquet files with multiple traits, which trait to extract
        
    Returns:
        DataFrame with summary statistics
    """
    from graphld.ldsc_io import read_ldsc_sumstats
    from graphld.parquet_io import read_parquet_sumstats
    from graphld.vcf_io import read_gwas_vcf

    # Lowercase extension for case-insensitive matching
    ext = os.path.splitext(sumstats_path)[1].lower()

    if ext in ['.vcf', '.vcf.gz']:
        return read_gwas_vcf(sumstats_path, maximum_missingness=maximum_missingness)
    elif ext == '.parquet':
        return read_parquet_sumstats(sumstats_path, trait=trait, maximum_missingness=maximum_missingness)
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
        for val, se, pval in zip(values, std_errors, p_values, strict=False):
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
        for i, (change, trust_lambda) in enumerate(zip(log['likelihood_changes'], log['trust_region_lambdas'], strict=False)):
            f.write(f"{i+1},{change},{trust_lambda}\n")

def _run_reml_single_trait(
    args,
    sumstats: pl.DataFrame,
    annotations: pl.DataFrame,
    annotation_columns: list,
    trait_name: str,
) -> dict:
    """Run GraphREML for a single trait.
    
    Args:
        args: Parsed command line arguments
        sumstats: Summary statistics DataFrame
        annotations: Annotations DataFrame
        annotation_columns: List of annotation column names
        trait_name: Name of the trait being analyzed
        
    Returns:
        Dictionary containing REML results
    """
    num_snps_annot = len(annotations)

    # Handle initial parameters
    initial_params = None
    if args.initial_params is not None:
        num_params = len(annotation_columns)
        initial_params = np.zeros((num_params, 1))
        for i, val in enumerate(args.initial_params[:num_params]):
            initial_params[i, 0] = val

    model_options = ModelOptions(
        sample_size=args.num_samples,
        intercept=args.intercept,
        annotation_columns=annotation_columns,
        link_fn_denominator=num_snps_annot,
        binary_annotations_only=args.binary_annotations_only,
        params=initial_params
    )

    method_options = MethodOptions(
        match_by_position=args.match_by_position,
        num_iterations=args.num_iterations,
        convergence_tol=args.convergence_tol,
        convergence_window=args.convergence_window,
        run_serial=args.run_in_serial,
        num_processes=args.num_processes,
        verbose=args.verbose,
        num_jackknife_blocks=args.num_jackknife_blocks,
        reset_trust_region=args.reset_trust_region,
        gradient_num_samples=args.xtrace_num_samples,
        max_chisq_threshold=args.max_chisq_threshold,
        score_test_hdf5_file_name=args.score_test_filename,
        score_test_hdf5_trait_name=trait_name,
        surrogate_markers_path=args.surrogates,
    )

    results = run_graphREML(
        model_options=model_options,
        method_options=method_options,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=args.metadata,
        populations=[args.population] if args.population else None,
        chromosomes=[args.chromosome] if args.chromosome else None,
    )

    return results, model_options


def _reml(args):
    """Run GraphREML command."""
    # Check for output filename requirement
    if not args.out and not args.no_save:
        raise ValueError("Output filename is required unless --no-save is specified.")

    # Check for existing output files
    if args.out:
        # Create output directory if it doesn't exist
        out_dir = os.path.dirname(args.out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)

        if not args.no_save:
            tall_output = args.out + '.tall.csv'
            if not args.alt_output:
                if os.path.exists(tall_output):
                    raise FileExistsError(f"Output file {tall_output} already exists")

    start_time = time.time()

    # Determine traits to process for parquet files
    ext = os.path.splitext(args.sumstats)[1].lower()
    is_parquet = ext == '.parquet'

    if is_parquet:
        available_traits = get_parquet_traits(args.sumstats)
        if not available_traits:
            raise ValueError(f"No traits found in parquet file {args.sumstats}")

        # Parse --name argument for trait selection
        if args.name:
            requested_traits = [t.strip() for t in args.name.split(',')]
            # Validate all requested traits exist
            missing = set(requested_traits) - set(available_traits)
            if missing:
                raise ValueError(f"Traits not found in parquet file: {missing}. "
                               f"Available traits: {available_traits}")
            traits_to_process = requested_traits
        else:
            # Process all traits
            traits_to_process = available_traits

        if not args.quiet:
            print(f"Found {len(available_traits)} traits in parquet file, processing {len(traits_to_process)}: {traits_to_process}")
    else:
        # For non-parquet files, use a single trait with the provided name
        traits_to_process = [args.name or args.sumstats]

    # Load annotations once (shared across all traits)
    # Check that exactly one of annot_dir or gene_annot_dir is provided
    has_variant_annot = args.annot_dir is not None
    has_gene_annot = args.gene_annot_dir is not None

    if has_variant_annot and has_gene_annot:
        raise ValueError("Cannot specify both --annot-dir and --gene-annot-dir. Choose one annotation source.")
    if not has_variant_annot and not has_gene_annot:
        raise ValueError("Must specify either --annot-dir (variant annotations) or --gene-annot-dir (gene annotations).")

    if has_gene_annot:
        # Load gene annotations from GMT files and convert to variant-level
        from .genesets import load_gene_annotations

        if args.verbose:
            print(f'Loading gene annotations from {args.gene_annot_dir}')

        # Parse nearest weights
        nearest_weights = np.array([float(w) for w in args.nearest_weights.split(',')], dtype=np.float64)

        # We need to load a reference variant table first to get positions
        # Load from the first sumstats file to get variant positions
        first_sumstats = _detect_sumstats_type(args.sumstats, args.maximum_missingness,
                                                trait=traits_to_process[0] if is_parquet else None)

        # Create variant table with required columns
        variant_table = first_sumstats.select([
            pl.col('CHR').cast(pl.Int64) if 'CHR' in first_sumstats.columns else pl.lit(0).alias('CHR'),
            pl.col('POS').cast(pl.Int64) if 'POS' in first_sumstats.columns else pl.col('BP').cast(pl.Int64).alias('POS'),
            pl.col('SNP') if 'SNP' in first_sumstats.columns else pl.col('RSID').alias('SNP'),
        ])

        # Filter by chromosome if specified
        if args.chromosome:
            variant_table = variant_table.filter(pl.col('CHR') == args.chromosome)

        annotations = load_gene_annotations(
            gene_annot_dir=args.gene_annot_dir,
            variant_table=variant_table,
            gene_table_path=args.gene_table,
            nearest_weights=nearest_weights,
            annot_names=args.annotation_columns,
        )

        if args.verbose:
            gene_annot_cols = [c for c in annotations.columns if c not in {'CHR', 'BP', 'SNP', 'CM'}]
            print(f'Loaded {len(gene_annot_cols)} gene annotations: {gene_annot_cols[:5]}...' if len(gene_annot_cols) > 5 else f'Loaded {len(gene_annot_cols)} gene annotations: {gene_annot_cols}')
    else:
        # Load variant-level annotations
        if args.verbose:
            print('Loading annotations from', args.annot_dir)
        annotations = load_annotations(args.annot_dir, chromosome=args.chromosome, add_positions=False)

    # Get annotation columns
    if args.annotation_columns:
        annotation_columns = args.annotation_columns
        for col in annotation_columns:
            if col not in annotations.columns:
                raise ValueError(f"Column {col} not found in annotations")
    else:
        excluded_cols = {'SNP', 'CM', 'BP', 'CHR', 'POS', 'A1', 'A2'}
        annotation_columns = [col for col in annotations.columns if col not in excluded_cols]

    # Process each trait
    for trait_name in traits_to_process:
        trait_start_time = time.time()

        if not args.quiet:
            if args.verbose:
                print(f'Loading summary statistics for trait: {trait_name}')

        # Load summary statistics for this trait
        if is_parquet:
            sumstats = _detect_sumstats_type(args.sumstats, args.maximum_missingness, trait=trait_name)
        else:
            sumstats = _detect_sumstats_type(args.sumstats, args.maximum_missingness)

        # Run GraphREML
        if not args.quiet:
            print(f'Running GraphREML for {trait_name}...')

        results, model_options = _run_reml_single_trait(
            args, sumstats, annotations, annotation_columns, trait_name
        )

        trait_runtime = time.time() - trait_start_time
        if not args.quiet:
            print(f"Time to run GraphREML for {trait_name}: {trait_runtime:.3f}s")
        if args.verbose:
            print(f"Likelihood changes: {np.diff(np.array(results['likelihood_history']))}")

        # Write output files only if out is specified
        if args.out:
            # Prepare output files
            convergence_file = args.out + '.convergence.csv'
            write_convergence_results(convergence_file, results)

            if not args.no_save:
                if args.alt_output:
                    heritability_file = args.out + '.heritability.csv'
                    write_results(heritability_file,
                                    results['heritability'],
                                    results['heritability_se'],
                                    results['heritability_log10pval'],
                                    model_options.annotation_columns,
                                    trait_name)

                    enrichment_file = args.out + '.enrichment.csv'
                    write_results(enrichment_file,
                                    results['enrichment'],
                                    results['enrichment_se'],
                                    results['enrichment_log10pval'],
                                    model_options.annotation_columns,
                                    trait_name)

                    parameters_file = args.out + '.parameters.csv'
                    write_results(parameters_file,
                                    results['parameters'],
                                    results['parameters_se'],
                                    results['parameters_log10pval'],
                                    model_options.annotation_columns,
                                    trait_name)
                else:
                    tall_output = args.out + '.tall.csv'
                    write_tall_results(tall_output, model_options, results)

    total_runtime = time.time() - start_time
    if not args.quiet and len(traits_to_process) > 1:
        print(f"Total time for all {len(traits_to_process)} traits: {total_runtime:.3f}s")


def _command_handlers():
    return {
        "blup": _blup,
        "clump": _clump,
        "surrogates": _surrogates,
        "simulate": _simulate,
        "reml": _reml,
    }


def _add_blup_parser(subparsers):
    """Add parser for blup command."""
    return _parser_add_blup_parser(subparsers, _blup)


def _add_clump_parser(subparsers):
    """Add parser for clump command."""
    return _parser_add_clump_parser(subparsers, _clump)


def _add_surrogates_parser(subparsers):
    """Add parser for surrogates command."""
    return _parser_add_surrogates_parser(subparsers, _surrogates)


def _add_simulate_parser(subparsers):
    """Add parser for simulate command."""
    return _parser_add_simulate_parser(subparsers, _simulate)


def _add_reml_parser(subparsers):
    """Add parser for reml command."""
    return _parser_add_reml_parser(subparsers, _reml)


def build_parser():
    """Build the GraphLD parser with the CLI command handlers attached."""
    return _parser_build_parser(_command_handlers())


def _main(args):
    return run_cli(args, _command_handlers())

def main():
    """Entry point for the graphld command line interface."""
    _main(sys.argv[1:])

if __name__ == "__main__":
    main()
