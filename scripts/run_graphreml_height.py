#!/usr/bin/env python3
"""Run GraphREML on height summary statistics with real annotation data."""

import time
import numpy as np
from pathlib import Path
import polars as pl
from graphld.heritability import ModelOptions, MethodOptions, run_graphREML
from graphld.io import load_annotations
from graphld.vcf_io import read_gwas_vcf
from graphld.ldsc_io import read_ldsc_sumstats

METADATA_PATH = "data/ldgms/metadata.csv"
ANNOT_PATH = "data/annot/"
SUMSTATS_PATH = "data/sumstats/body_HEIGHTz.sumstats"
VCF_PATH = "data/sumstats/height.hg38.vcf"
USE_VCF = False
CHROMOSOME = None
POPULATION = "EUR"
MATCH_POSITION = True
RUN_SERIAL = False
NUM_REPS = 100
NUM_PROCESSES = 12
NUM_SAMPLES = 20
SAMPLE_SIZE = 400000
NUM_ANNOT = 10
VERBOSE = True
ADD_MISSINGNESS = False
USE_SURROGATE_MARKERS = True
TRUST_REGION_RHO_LB = 0.001
TRUST_REGION_RHO_UB = 0.9
RESET_TRUST_REGION = False
CONVERGENCE_TOL = 1e-2

def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax implementation."""
    y = x + np.log1p(np.exp(-x))
    return y

def log_results(h2, enrichment, likelihood, runtime, output_file="results.csv"):
    """Log results to a CSV file.
    
    Args:
        h2: Heritability estimates
        enrichment: Enrichment estimates
        likelihood: List of likelihood values
        runtime: Runtime in seconds
        output_file: Path to output CSV file
    """
    # Create dictionary of all parameters and results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'metadata_path': METADATA_PATH,
        'sumstats_path': VCF_PATH if USE_VCF else SUMSTATS_PATH,
        'use_vcf': USE_VCF,
        'chromosome': CHROMOSOME,
        'population': POPULATION,
        'match_position': MATCH_POSITION,
        'run_serial': RUN_SERIAL,
        'num_reps': NUM_REPS,
        'num_processes': NUM_PROCESSES,
        'num_samples': NUM_SAMPLES,
        'sample_size': SAMPLE_SIZE,
        'num_annot': NUM_ANNOT,
        'use_surrogate_markers': USE_SURROGATE_MARKERS,
        'trust_region_rho_lb': TRUST_REGION_RHO_LB,
        'trust_region_rho_ub': TRUST_REGION_RHO_UB,
        'runtime': runtime,
        'final_likelihood': likelihood[-1] if likelihood else None,
        'likelihood_change': likelihood[-1] - likelihood[-2] if len(likelihood) >= 2 else None,
        'h2_baseline': h2[0],  # First element of heritability vector
        'enrichment_coding': enrichment[1],  # Second element of enrichment vector (coding)
    }
    
    # Convert to DataFrame
    results_df = pl.DataFrame([results])
    
    # Check if file exists
    file_exists = Path(output_file).exists()
    
    # Write to CSV
    with open(output_file, mode="a" if file_exists else "w") as f:
        results_df.write_csv(f, include_header=not file_exists)

def main():
    # sumstats = read_gwas_vcf(VCF_PATH)
    # print(sumstats.head())
    # sumstats = read_ldsc_sumstats(SUMSTATS_PATH)
    # print(sumstats.head())

    # assert False
    t = time.time()
    if USE_VCF:
        sumstats = read_gwas_vcf(VCF_PATH)
    else:
        sumstats = read_ldsc_sumstats(SUMSTATS_PATH)
    print(len(sumstats))
    print(f"Time to read sumstats: {time.time() - t:.3f}s")

    if ADD_MISSINGNESS:
        sumstats = sumstats.with_row_index('index_name')
        sumstats = sumstats.filter(pl.col('index_name') % 2 == 0)

    t = time.time()
    annotations = load_annotations(ANNOT_PATH, CHROMOSOME)
    print(f"Time to load annotations: {time.time() - t:.3f}s")

    # Create model with one parameter per annotation
    excluded_cols = {'SNP', 'CM', 'BP', 'CHR', 'POS', 'A1', 'A2'}
    annotation_columns = [col for col in annotations.columns if col not in excluded_cols]
    annotation_columns = annotation_columns[:NUM_ANNOT]
    num_annotations = len(annotation_columns)

    model_options = ModelOptions(
        # params=np.zeros((num_annotations,1)),  # Initialize parameters evenly
        annotation_columns=annotation_columns,
        sample_size=SAMPLE_SIZE,
        link_fn_denominator = len(sumstats),
    )

    method_options = MethodOptions(
        gradient_num_samples=NUM_SAMPLES,
        match_by_position=MATCH_POSITION,
        num_processes=NUM_PROCESSES,
        run_serial = RUN_SERIAL,
        verbose = VERBOSE,
        num_iterations = NUM_REPS,
        use_surrogate_markers = USE_SURROGATE_MARKERS,
        trust_region_rho_lb = TRUST_REGION_RHO_LB,
        trust_region_rho_ub = TRUST_REGION_RHO_UB,
        convergence_tol = CONVERGENCE_TOL,
        reset_trust_region = RESET_TRUST_REGION,
    )

    print(f"\nRunning GraphREML with {len(annotation_columns)} annotations")
    print(f"Number of variants: {len(sumstats):,}")

    # Run GraphREML
    start_time = time.time()
    results = run_graphREML(
        model_options=model_options,
        method_options=method_options,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=METADATA_PATH,
        populations=POPULATION,
        chromosomes=CHROMOSOME,
    )
    runtime = time.time() - start_time
    print(f"Time to run GraphREML: {runtime:.3f}s")
    print(f"Estimated heritability: {results['heritability']}")
    print(f"SE of estimated heritability: {results['heritability_se']}")

    print(f"Estimated enrichment: {results['enrichment']}")
    print(f"SE of estimated enrichment: {results['enrichment_se']}")
    print(f"Likelihood changes: {np.diff(np.array(results['likelihood_history']))}")

    # Log results
    log_results(results['heritability'], results['enrichment'], results['likelihood_history'], runtime)

if __name__ == '__main__':
    main()
