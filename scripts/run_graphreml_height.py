#!/usr/bin/env python3
"""Run GraphREML on height summary statistics with real annotation data."""

import time
import numpy as np
from graphld.heritability import ModelOptions, MethodOptions, run_graphREML
from graphld.io import load_annotations
from graphld.vcf_io import read_gwas_vcf

METADATA_PATH = "data/ldgms/metadata.csv"
ANNOT_PATH = "data/annot/"
SUMSTATS_PATH = "data/sumstats/height.hg38.vcf"
CHROMOSOME = None
POPULATION = "EUR"
MATCH_POSITION = True
RUN_SERIAL = False
NUM_REPS = 5
NUM_PROCESSES = 12
NUM_SAMPLES = 100
SAMPLE_SIZE = 400000
NUM_ANNOT = None
VERBOSE = True

def main():
    t = time.time()
    sumstats = read_gwas_vcf(SUMSTATS_PATH)
    print(len(sumstats))
    print(f"Time to read sumstats: {time.time() - t:.3f}s")
    
    t = time.time()
    annotations = load_annotations(ANNOT_PATH, CHROMOSOME)
    print(f"Time to load annotations: {time.time() - t:.3f}s")

    # Create model with one parameter per annotation
    excluded_cols = {'SNP', 'CM', 'BP', 'CHR', 'POS', 'A1', 'A2'}
    annotation_columns = [col for col in annotations.columns if col not in excluded_cols]
    annotation_columns = annotation_columns[:NUM_ANNOT]
    num_annotations = len(annotation_columns)

    model_options = ModelOptions(
        params=np.zeros((num_annotations,1)),  # Initialize parameters evenly
        annotation_columns=annotation_columns,
        sample_size=SAMPLE_SIZE
    )

    method_options = MethodOptions(
        gradient_num_samples=NUM_SAMPLES,
        match_by_position=MATCH_POSITION,
        num_processes=NUM_PROCESSES,
        run_serial = RUN_SERIAL,
        verbose = VERBOSE,
        num_iterations = NUM_REPS,
    )

    print(f"\nRunning GraphREML with {len(annotation_columns)} annotations")
    print(f"Number of variants: {len(sumstats):,}")


    # Run GraphREML
    t = time.time()
    h2, likelihood = run_graphREML(
        model_options=model_options,
        method_options=method_options,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=METADATA_PATH,
        populations=POPULATION,
        chromosomes=CHROMOSOME,
    )
    print(f"Time to run GraphREML: {time.time() - t:.3f}s")

    print(f"Estimated heritability: {h2 / SAMPLE_SIZE}")
    print(f"Likelihood changes: {np.diff(np.array(likelihood))}")

if __name__ == '__main__':
    main()
