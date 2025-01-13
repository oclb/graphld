#!/usr/bin/env python3
import argparse
import os
import sys
from importlib import metadata
from typing import Optional, List, Dict, Union
from pathlib import Path

import graphld as gld
from graphld.vcf_io import read_gwas_vcf
import polars as pl

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
    verbose: bool
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
        sumstats = pl.read_csv(sumstats, separator='\t')
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
    verbose: bool
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
        sumstats_df = pl.read_csv(sumstats, separator='\t')
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
        populations=population
    ).filter(pl.col('is_index'))
    
    # Write output
    clumped.write_csv(out, separator='\t')

def _simulate(
    sumstats: str,
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
    sample_size: int = 1000
) -> None:
    """Run genetic simulation with configurable parameters.
    
    Args:
        sumstats: Path to output summary statistics file
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
        sample_size: Number of samples to simulate (default: 1000)
    
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
    annotations = None
    if annotation_dependent_polygenicity or annotation_columns:
        annotations = load_annotations(
            "data/annot/",  # Default annotation path
            chromosome
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
        annotations=annotations,
        run_in_serial=run_in_serial
    )
    
    # Write output
    sim_result.write_csv(sumstats, separator='\t')

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
    parser.add_argument("-p", "--population", type=str, default=None,
                      help="Population to filter analysis")
    parser.add_argument("-v", "--verbose", action="store_true", default=False)
    parser.add_argument("-q", "--quiet", action="store_true", default=False)

def _main(args):
    argp = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Add common arguments to main parser
    _add_common_arguments(argp)

    # Subparsers
    subp = argp.add_subparsers(dest="cmd", required=True, help="Subcommands for graphld")

    # BLUP command
    blup_p = subp.add_parser("blup", help="Compute BLUP effect sizes")
    _add_common_arguments(blup_p)  # Add common arguments to BLUP subparser
    blup_p.add_argument("sumstats", type=str, 
                        help="Path to summary statistics file (.vcf or .sumstats)")
    blup_p.add_argument("out", type=str, 
                        help="Output file path")
    blup_p.add_argument("-H", "--heritability", type=float, default=0.2,
                        help="Heritability (default: 0.2)")
    blup_p.set_defaults(func=_blup)

    # LD clumping command
    clump_p = subp.add_parser("clump", help="Perform LD clumping")
    _add_common_arguments(clump_p)  # Add common arguments to clump subparser
    clump_p.add_argument("sumstats", type=str, 
                         help="Path to summary statistics file (.vcf or .sumstats)")
    clump_p.add_argument("out", type=str, 
                         help="Output file path")
    clump_p.add_argument("--min_chisq", type=float, default=30,
                         help="Minimum chi-squared value (default: 0.001)")
    clump_p.add_argument("--max_rsq", type=float, default=0.1,
                         help="Maximum R-squared value (default: 0.1)")
    clump_p.set_defaults(func=_clump)

    # Genetic simulation command
    sim_p = subp.add_parser("simulate", help="Perform genetic simulation")
    _add_common_arguments(sim_p)  # Add common arguments to simulate subparser
    sim_p.add_argument("sumstats", type=str, 
                       help="Path to output summary statistics file (.sumstats)")
    sim_p.add_argument("--sample_size", type=int, default=1000,
                       help="Number of samples to simulate (default: 1000)")
    sim_p.add_argument("-H", "--heritability", type=float, default=0.2,
                       help="Heritability (default: 0.2)")
    sim_p.add_argument("--component_variance", type=lambda s: [float(x) for x in s.split(',')],
                       default=[1.0],
                       help="Component variance (default: [1.0])")
    sim_p.add_argument("--component_weight", type=lambda s: [float(x) for x in s.split(',')],
                       default=[1.0],
                       help="Component weight (default: [1.0])")
    sim_p.add_argument("--alpha_param", type=float, default=-0.5,
                       help="Alpha parameter (default: -0.5)")
    sim_p.add_argument("--annotation_dependent_polygenicity", action="store_true",
                       help="Annotation dependent polygenicity")
    sim_p.add_argument("--random_seed", type=int, default=42,
                       help="Random seed (default: 42)")
    sim_p.add_argument("--annotation_columns", type=lambda s: s.split(','),
                       default=None,
                       help="Annotation columns")
    sim_p.set_defaults(func=_simulate)

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
            parsed_args.verbose
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
            parsed_args.verbose
        )
    elif parsed_args.cmd == "simulate":
        return _simulate(
            parsed_args.sumstats, 
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
            parsed_args.sample_size
        )

def main():
    """Entry point for the graphld command line interface."""
    _main(sys.argv[1:])

if __name__ == "__main__":
    main()
