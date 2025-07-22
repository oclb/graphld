#!/usr/bin/env python3

"""Test multiprocessing framework."""

import time
from multiprocessing import Array
from pathlib import Path

import numpy as np

from graphld.io import load_ldgm, read_ldgm_metadata
from graphld.multiprocessing_template import ParallelProcessor, SharedData, WorkerManager

# Constants
NUM_SOLVES = 1
NUM_FACTORS = 1
NUM_PROCESSES = 2


class SolveProcessor(ParallelProcessor):
    """Processor for solving LDGM blocks in parallel."""

    @staticmethod
    def create_shared_memory(metadata, block_data, seed=None, **kwargs) -> SharedData:
        """Create shared memory arrays."""
        # Calculate total size needed for arrays
        total_size = sum(block['numIndices'] for block in metadata.iter_rows(named=True))

        # Create shared arrays and initialize to zeros
        shared_data = SharedData({
            'input': Array('d', total_size, lock=False),
            'solution': Array('d', total_size, lock=False)
        })

        # Initialize input array with random values
        rng = np.random.RandomState(seed)
        shared_data['input'] = rng.randn(total_size)

        return shared_data

    @staticmethod
    def supervise(manager: WorkerManager, shared_data: SharedData, block_data: list, **kwargs):
        """Wait for all workers to finish and reshape results."""
        # Wait for workers to finish
        manager.start_workers()
        manager.await_workers()
        return shared_data['solution']

    @staticmethod
    def process_block(ldgm, flag, shared_data, block_offset, block_data=None, worker_params=None):
        """Process single block by solving with random vectors."""
        # Get input vector slice for this block
        vector = shared_data[('input', slice(block_offset, block_offset + ldgm.shape[0]))]

        for _ in range(NUM_FACTORS):
            ldgm.del_factor()
            ldgm.factor()

        for _ in range(NUM_SOLVES):
            solution = ldgm.solve(vector)

        # Store solution in shared memory
        shared_data[('solution', slice(block_offset, block_offset + solution.size))] = solution

        return solution.size


def solve_serial(metadata_file, population=None, chromosomes=None, seed=None):
    """Run serial solution for comparison."""
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

        for _ in range(NUM_FACTORS):
            ldgm.del_factor()
            ldgm.factor()

        # Get input slice and solve - match parallel version exactly
        vector = input_array[block_offset:block_offset + ldgm.shape[0]]
        for _ in range(NUM_SOLVES):
            solution = ldgm.solve(vector)

        # Store solution
        solution_array[block_offset:block_offset + solution.size] = solution
        block_offset += solution.size

    return time.time() - t, solution_array, input_array


def test_multiprocessing():
    """Test that parallel and serial solutions match."""
    metadata_file = "data/test/metadata.csv"
    seed = 42  # Fixed seed for reproducibility

    # Run serial version first to get input array
    serial_time, serial_results, input_array = solve_serial(
        metadata_file,
        seed=seed
    )

    # Run parallel version with same input array
    parallel_results = SolveProcessor.run(
        ldgm_metadata_path=metadata_file,
        num_processes=NUM_PROCESSES,
        seed=seed  # Pass pre-created shared data
    )

    # Validate results
    assert np.allclose(parallel_results, serial_results, rtol=1e-10, atol=1e-10), \
        "Parallel and serial results do not match"

    # Run serial version of SolveProcessor
    serial_results = SolveProcessor.run_serial(
        ldgm_metadata_path=metadata_file,
        num_processes=NUM_PROCESSES,
        seed=seed  # Pass pre-created shared data
    )

    # Validate results
    assert np.allclose(parallel_results, serial_results, rtol=1e-10, atol=1e-10), \
        "Parallel and serial results do not match"

