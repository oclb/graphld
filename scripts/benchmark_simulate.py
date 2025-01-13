#!/usr/bin/env python3
METADATA_PATH = "data/ldgms/metadata.csv"
ANNOT_PATH = "data/annot/"
CHROMOSOME = 22
POPULATION = "EUR"
RUN_IN_SERIAL = False

import time
from graphld import Simulate, read_ldgm_metadata, LDClumper
from graphld.io import load_annotations
from typing import *
import polars as pl
import numpy as np
import os

def main():
    """Test simulation with variant annotations."""

    # Set random seed for reproducibility
    np.random.seed(42)

    t = time.time()
    annotations = load_annotations(ANNOT_PATH, CHROMOSOME, add_alleles=True)
    print(f"Time to load annotations: {time.time() - t:.3f}s")
    t = time.time()

    sim = Simulate(
        sample_size=100_000,
        heritability=0.2,
        component_variance=[1.0],
        component_weight=[0.0001],
        alpha_param=-1,
        random_seed=42
    )

    sim_result = sim.simulate(
        ldgm_metadata_path=METADATA_PATH,
        populations=POPULATION,
        chromosomes=CHROMOSOME,
        annotations=annotations,
        run_in_serial=RUN_IN_SERIAL
    )
    assert sim_result['beta'].is_nan().sum() == 0
    sim_time = time.time() - t

    print(f"Simulation took {sim_time:.2f} seconds")
    print(f"Number of variants: {len(sim_result)}")
    print(f"Number of non-zero betas: {(sim_result['beta'] != 0).sum()}")
    print(f"Maximum chisq: {sim_result['Z'].max() ** 2}")

    # Run LD clumping
    print("\nRunning LD clumping...")
    t = time.time()
    clumped = LDClumper.clump(
        METADATA_PATH,
        sim_result,
        rsq_threshold=0.05,  # Relatively stringent LD threshold
        chisq_threshold=30.0,
        populations=POPULATION,
        chromosomes=CHROMOSOME,
        run_in_serial=RUN_IN_SERIAL,
    )
    clump_time = time.time() - t

    # Get index variants
    index_variants = clumped.filter(pl.col('is_index'))

    print(f"LD clumping took {clump_time:.2f} seconds")
    print(f"Number of index variants: {len(index_variants)}")


if __name__ == "__main__":
    main()
