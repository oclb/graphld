#!/usr/bin/env python3

"""Benchmark multiprocessing framework on whole genome LDGM solve."""
SKIP_SERIAL = True
NUM_SOLVES = 200
NUM_FACTORS = 0
NUM_PROCESSES = 12
CHROMOSOME = None
POPULATION = "EUR"


import time
from multiprocessing import Array
from pathlib import Path

import numpy as np
import polars as pl

from graphld.io import load_ldgm, read_ldgm_metadata
from graphld.multiprocessing import ParallelProcessor, SharedData, WorkerManager


class SolveProcessor(ParallelProcessor):
    """Test processor that solves LDGMs with vectors from DataFrames."""

    @staticmethod
    def create_shared_memory(metadata, block_data, seed=None, **kwargs) -> SharedData:
        """Create shared memory arrays."""
        # Calculate total size needed
        total_size = sum(block['numIndices'] for block in metadata.iter_rows(named=True))

        # Create shared arrays for solutions only
        shared_data = SharedData({
            'solution': Array('d', total_size, lock=False)
        })

        return shared_data

    @classmethod
    def prepare_block_data(cls, metadata, seed=None, **kwargs) -> list:
        """Prepare random vectors for each block."""
        t = time.time()
        rng = np.random.RandomState(seed)
        block_data = []
        for block in metadata.iter_rows(named=True):
            n = block['numIndices']
            vector = rng.randn(n)
            # Create a Polars DataFrame with the vector and extra columns
            df = pl.DataFrame({
                'vector': vector,
                'index': np.arange(n),
                'position': np.arange(1000, 1000 + n),  # Mock positions
                'ref': ['A'] * n,  # Mock reference alleles
                'alt': ['G'] * n   # Mock alternate alleles
            })
            block_data.append(df)
        print(f"Time to prepare block data: {time.time() - t:.3f}s")
        return block_data

    @staticmethod
    def supervise(manager: WorkerManager, shared_data: SharedData, block_data: list, **kwargs):
        """Wait for all workers to finish and reshape results."""
        # Wait for workers to finish
        t = time.time()
        manager.start_workers()
        manager.await_workers()
        return shared_data['solution']

    @staticmethod
    def process_block(ldgm, flag, shared_data, block_offset, block_data=None, worker_params=None):
        """Process single block by solving with random vectors."""
        # Get input vector from block_data DataFrame
        vector = block_data['vector'].to_numpy()

        # Touch other columns
        n_variants = len(block_data)
        max_pos = block_data['position'].max()
        n_ref_a = block_data.filter(pl.col('ref') == 'A').height
        n_alt_g = block_data.filter(pl.col('alt') == 'G').height

        for i in range(NUM_FACTORS):
            ldgm.del_factor()
            ldgm.factor()

        for i in range(NUM_SOLVES):
            solution = ldgm.solve(vector)

        # Store solution in shared memory
        shared_data[('solution', slice(block_offset, block_offset + solution.size))] = solution

        return solution.size


def solve_serial(metadata_file, population=None, chromosomes=None, seed=None):
    """Solve LDGMs in series for comparison."""
    # Read metadata
    metadata = read_ldgm_metadata(
        metadata_file,
        populations=population,
        chromosomes=chromosomes
    )

    # Get list of files
    ldgm_path = Path(metadata_file).parent

    # Calculate total size needed
    total_size = sum(block['numIndices'] for block in metadata.iter_rows(named=True))

    # Create input array with random values
    rng = np.random.RandomState(seed)
    input_array = rng.randn(total_size)

    # Process each block
    t = time.time()
    solution_array = np.zeros(total_size)
    block_offset = 0

    for block in metadata.iter_rows(named=True):
        # Load and factor LDGM
        ldgm = load_ldgm(str(ldgm_path / block['name']))
        ldgm.factor()

        # Get input slice and solve - match parallel version exactly
        vector = input_array[block_offset:block_offset + ldgm.shape[0]]
        for i in range(NUM_FACTORS):
            ldgm.del_factor()
            ldgm.factor()

        for i in range(NUM_SOLVES):
            solution = ldgm.solve(vector)

        # Store solution
        solution_array[block_offset:block_offset + solution.size] = solution
        block_offset += solution.size

    return time.time() - t, solution_array, input_array


def main():
    """Run benchmark."""
    metadata_file = "data/ldgms/metadata.csv"
    seed = 42  # Fixed seed for reproducibility
    print(f"Running {NUM_SOLVES} solves with {NUM_FACTORS} extra factorizations on chromosome {CHROMOSOME or 'all'}")

    serial_results = None
    if not SKIP_SERIAL:
        # Run serial version first to get input array
        print("\nRunning serial solution...")
        serial_time, serial_results, input_array = solve_serial(
            metadata_file,
            population=POPULATION,
            chromosomes=CHROMOSOME,
            seed=seed
        )
        print(f"Serial time: {serial_time:.2f}s")

    # Run parallel version with same input array
    t = time.time()
    parallel_results = SolveProcessor.run(
        metadata_file,
        num_processes=NUM_PROCESSES,
        populations=POPULATION,
        chromosomes=CHROMOSOME,
        seed=seed  # Pass pre-created shared data
    )
    parallel_time = time.time() - t
    print(f"Parallel time with {NUM_PROCESSES} processes: {parallel_time:.2f}s")

    # Validate results if serial was run
    if not SKIP_SERIAL:
        if np.allclose(parallel_results, serial_results, rtol=1e-10, atol=1e-10):
            print("Results match exactly!")
        else:
            max_diff = np.max(np.abs(parallel_results - serial_results))
            print(f"Results differ! Maximum absolute difference: {max_diff}")
            print(f"First few parallel results: {parallel_results[:5]}")
            print(f"First few serial results: {serial_results[:5]}")
            print("Input array that was used:")
            print(f"input_array[:5] = {input_array[:5]}")


if __name__ == "__main__":
    main()
