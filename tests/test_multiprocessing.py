#!/usr/bin/env python3

"""Test multiprocessing framework."""

import os
import multiprocessing as mp
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from graphld.io import load_ldgm, read_ldgm_metadata
from graphld.multiprocessing_template import (
    ParallelProcessor,
    SharedData,
    WorkerManager,
    _MP_CONTEXT,
)

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
            'input': total_size,
            'solution': total_size,
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


class FirstAlleleFrequencyProcessor(ParallelProcessor):
    """Processor that records the first loaded allele frequency in each block."""

    @staticmethod
    def create_shared_memory(metadata, block_data, **kwargs) -> SharedData:
        """Create output for one allele frequency per metadata block."""
        return SharedData({'af': len(metadata)})

    @staticmethod
    def prepare_block_data(metadata, **kwargs):
        """Attach block indices for writing one output value per block."""
        return [{'block_index': i} for i in range(len(metadata))]

    @staticmethod
    def supervise(manager: WorkerManager, shared_data: SharedData, block_data: list, **kwargs):
        """Wait for workers and return recorded allele frequencies."""
        manager.start_workers()
        manager.await_workers()
        return shared_data['af']

    @staticmethod
    def process_block(ldgm, flag, shared_data, block_offset, block_data=None, worker_params=None):
        """Record the first loaded `af` value for this LDGM."""
        block_index = block_data['block_index']
        shared_data['af', slice(block_index, block_index + 1)] = [ldgm.variant_info['af'][0]]


def abrupt_exit_worker(flag):
    """Exit without updating the shared flag, simulating a native crash."""
    while flag.value == 0:
        time.sleep(0.01)
    os._exit(17)


def flagged_exception_worker(flag):
    """Report failure through the flag and then exit with a Python exception."""
    while flag.value == 0:
        time.sleep(0.01)
    try:
        raise ValueError("worker failed")
    except BaseException:
        flag.value = -1
        raise


def unresponsive_worker(flag):
    """Ignore the shutdown flag so WorkerManager must terminate this process."""
    while True:
        time.sleep(0.01)


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
        ldgm = load_ldgm(str(ldgm_path / block['name']), population=block['population'])

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


def test_parallel_context_uses_spawn():
    """GraphLD must not inherit the platform's threaded-library state."""
    assert _MP_CONTEXT.get_start_method() == "spawn"


def test_parallel_processor_uses_spawn_under_fork_default(tmp_path):
    """A fork-default parent must still launch fresh GraphLD interpreters."""
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork start method is unavailable")

    script = tmp_path / "forced_fork_blup.py"
    script.write_text(
        "import multiprocessing as mp\n"
        "import runpy\n"
        "if __name__ == '__main__':\n"
        "    mp.set_start_method('fork')\n"
        "    namespace = runpy.run_path('tests/test_blup.py')\n"
        "    namespace['test_blup']()\n"
    )
    repo_root = Path(__file__).parent.parent
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(repo_root / "src"), env.get("PYTHONPATH")])
    )
    process = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(
            "forced-fork subprocess timed out\n"
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        )

    assert process.returncode == 0, stderr


def test_worker_manager_reports_abrupt_exit():
    """A dead child must raise instead of leaving the parent spinning forever."""
    manager = WorkerManager(1)
    manager.add_process(abrupt_exit_worker, (manager.flags[0],))
    manager.start_workers()
    try:
        with pytest.raises(RuntimeError, match=r"worker 0 .* code 17"):
            manager.await_workers()
    finally:
        manager.shutdown()


def test_worker_manager_reports_flagged_exception_exit_code():
    """Caught worker failures should retain their eventual process exit code."""
    manager = WorkerManager(1)
    manager.add_process(flagged_exception_worker, (manager.flags[0],))
    manager.start_workers()
    try:
        with pytest.raises(RuntimeError, match=r"worker 0 .* exit code 1"):
            manager.await_workers()
    finally:
        manager.shutdown()


def test_worker_manager_shutdown_terminates_unresponsive_worker():
    """Shutdown has a bounded fallback for children that ignore their flag."""
    manager = WorkerManager(1)
    manager.add_process(unresponsive_worker, (manager.flags[0],))
    manager.start_workers()

    start = time.monotonic()
    manager.shutdown()

    assert time.monotonic() - start < 3.0
    assert not manager.processes[0].is_alive()


def test_parallel_processor_cleans_up_after_supervisor_error(monkeypatch):
    """Supervisor failures must not leave spawned workers behind."""
    shutdown_managers = []
    original_shutdown = WorkerManager.shutdown

    def tracking_shutdown(manager):
        original_shutdown(manager)
        shutdown_managers.append(manager)

    def raising_supervisor(*args, **kwargs):
        raise RuntimeError("supervisor failed")

    monkeypatch.setattr(WorkerManager, "shutdown", tracking_shutdown)
    monkeypatch.setattr(SolveProcessor, "supervise", staticmethod(raising_supervisor))

    with pytest.raises(RuntimeError, match="supervisor failed"):
        SolveProcessor.run(
            ldgm_metadata_path="data/test/metadata.csv",
            num_processes=NUM_PROCESSES,
            seed=42,
        )

    assert len(shutdown_managers) == 1
    assert all(
        not process.is_alive()
        for process in shutdown_managers[0].processes
    )


def test_worker_loads_population_specific_allele_frequencies():
    """Workers should load the population column from each metadata row."""
    metadata_file = "data/test/metadata.csv"
    metadata = read_ldgm_metadata(metadata_file, populations="EAS")
    ldgm_path = Path(metadata_file).parent
    expected = np.array([
        load_ldgm(
            str(ldgm_path / block['name']),
            population=block['population'],
        ).variant_info['af'][0]
        for block in metadata.iter_rows(named=True)
    ])
    eur_default = np.array([
        load_ldgm(str(ldgm_path / block['name'])).variant_info['af'][0]
        for block in metadata.iter_rows(named=True)
    ])
    assert not np.array_equal(expected, eur_default)

    parallel_af = FirstAlleleFrequencyProcessor.run(
        ldgm_metadata_path=metadata_file,
        populations="EAS",
        num_processes=2,
    )
    serial_af = FirstAlleleFrequencyProcessor.run_serial(
        ldgm_metadata_path=metadata_file,
        populations="EAS",
    )

    np.testing.assert_array_equal(parallel_af, expected)
    np.testing.assert_array_equal(serial_af, expected)
