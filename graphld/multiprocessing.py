"""Base class for parallel processing applications."""


import time
from abc import ABC, abstractmethod
from multiprocessing import Array, Process, Value, cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl

from graphld.precision import PrecisionOperator

from .io import load_ldgm, read_ldgm_metadata


class SharedData:
    """Wrapper for shared memory data structures.

    Attributes:
        _data_dict: Dictionary mapping keys to shared memory objects
    """

    def __init__(self, sizes: Dict[str, Union[int, None]]):
        """Initialize shared memory objects.

        Args:
            sizes: Dictionary mapping keys to array sizes.
                  If size is None, creates a shared float value.
        """
        self._data_dict = {}
        for key, size in sizes.items():
            if size is None:
                self._data_dict[key] = Value('d', 0.0)
            else:
                self._data_dict[key] = Array('d', size)

    def __getitem__(self, key: Union[str, Tuple[str, slice]]) -> Union[np.ndarray, float]:
        """Get numpy array view or float value.

        Args:
            key: Key in data dictionary or tuple of (key, slice)

        Returns:
            numpy array view for Array, float for Value
        """
        if isinstance(key, tuple):
            array_key, slice_obj = key
            return self[array_key][slice_obj]

        data = self._data_dict[key]
        if hasattr(data, 'get_obj'):  # Array has get_obj, Value doesn't
            return np.frombuffer(data.get_obj(), dtype=np.float64)
        return data.value

    def __setitem__(self, key: Union[str, Tuple[str, slice]], value: Union[np.ndarray, float]):
        """Set array contents or float value.

        Args:
            key: Key in data dictionary or tuple of (key, slice)
            value: Array or float to set
        """
        if isinstance(key, tuple):
            array_key, slice_obj = key
            data = self._data_dict[array_key]
            if hasattr(data, 'get_obj'):
                data[slice_obj] = value
            else:
                raise ValueError("Slice assignment only supported for Array types")
            return

        data = self._data_dict[key]
        if hasattr(data, 'get_obj'):  # Array has get_obj, Value doesn't
            np.copyto(np.frombuffer(data.get_obj(), dtype=np.float64), value)
        else:
            data.value = float(value)



class WorkerManager:
    """Manager for coordinating parallel worker processes.

    Attributes:
        flags: List of shared flags for worker control
        processes: List of worker processes
    """

    def __init__(self, num_processes: int):
        """Initialize worker manager.

        Args:
            num_processes: Number of worker processes to manage
        """
        self.flags = [Value('i', 0) for _ in range(num_processes)]
        self.processes: List[Process] = []

    def start_workers(self, flag: Optional[int] = None) -> None:
        """Signal workers to start processing.

        Args:
            flag: Optional flag value to set (default: 1)
        """
        if flag is None:
            flag = 1
        for f in self.flags:
            f.value = flag

    def await_workers(self) -> None:
        """Wait for all workers to finish current task."""
        while any(flag.value >= 1 for flag in self.flags):
            time.sleep(0.01)

    def add_process(self, target: Callable, args: Tuple) -> None:
        """Add a worker process.

        Args:
            target: Function to run in process
            args: Arguments to pass to function
        """
        process = Process(target=target, args=args)
        process.start()
        self.processes.append(process)

    def shutdown(self) -> None:
        """Shutdown all worker processes."""
        # Signal shutdown
        for flag in self.flags:
            flag.value = -1

        # Wait for processes to finish
        for process in self.processes:
            process.join()


class ParallelProcessor(ABC):
    """Abstract base class for parallel processing applications.

    This class provides a framework for parallel processing of LDGM files.
    Subclasses must implement the following methods:
        - initialize: Set up shared memory arrays and data structures
        - supervise: Monitor and control worker processes
        - process_block: Process a single LDGM block
    """


    @staticmethod
    @abstractmethod
    def create_shared_memory(metadata: pl.DataFrame, **kwargs) -> 'SharedData':
        """Initialize shared memory and data structures.

        Args:
            metadata: polars dataframe continaing LDGM metadata for each LD block
            **kwargs: Additional arguments

        Returns:
            SharedData object containing shared memory arrays
        """
        pass

    @staticmethod
    @abstractmethod
    def supervise(manager: WorkerManager, shared_data: SharedData) -> Any:
        """Monitor workers and process results.

        Args:
            manager: Worker manager for controlling processes
            shared_data: Shared memory data

        Returns:
            Results of the parallel computation
        """
        pass

    @staticmethod
    @abstractmethod
    def process_block(ldgm: PrecisionOperator,
                    flag: Value,
                    shared_data: SharedData,
                    block_offset: int) -> int:
        """Process single block.

        Args:
            ldgm: LDGM object
            flag: Worker flag
            shared_data: Shared memory data
            block_offset: Index that can be used to index into shared data

        Returns:
            Number of indices in the current block, which is used to update block_offset
        """
        return ldgm.shape[0]

    @classmethod
    def worker(cls,
               files: list,
               flag: Value,
               shared_data: SharedData,
               offset: int,
               ) -> None:
        """Worker process that loads LDGMs and processes blocks.

        Args:
            files: List of LDGM files to process
            flag: Shared flag for worker control
            shared_data: Shared memory data
            offset: In shared data, where to start processing
        """
        try:
            # Load LDGMs once
            ldgms = []
            for file in files:
                ldgm = load_ldgm(str(file))
                ldgm.factor()
                ldgms.append(ldgm)

            while True:
                # Wait for signal to start new iteration
                while flag.value == 0:
                    time.sleep(0.01)

                if flag.value == -1:  # shutdown signal
                    break

                # Process all blocks and collect solutions
                block_offset = offset
                for ldgm in ldgms:
                    block_offset += cls.process_block(ldgm, flag, shared_data, block_offset)

                # Signal completion
                flag.value = 0

        except Exception as e:
            print(f"Error in worker: {e}")
            flag.value = -1


    @classmethod
    def _split_blocks(cls,
                metadata: pl.DataFrame,
                num_processes: int
                ) -> tuple[list[tuple[int, int]], list[int]]:
        """Divide LD blocks among workers for parallel processing. Attempts to
        split work evenly assuming that work is proportional to number of entries in each block

        Args:
            metadata: Data for each block
            num_processes: Number of processes

        Returns:
            Tuple containing two lists:
                - Of ranges of block indices for each process
                - Of index offsets for each process
        """
        # Use numIndices consistently for both splitting and offsets
        size_array = metadata.get_column('numEntries').to_numpy()
        size_cumsum = np.insert(np.cumsum(size_array), 0, 0)
        chunk_size = size_cumsum[-1] / num_processes

        # Find indices where cumsum crosses multiples of chunk_size
        block_indices = []
        for i in range(1, num_processes):
            target_sum = i * chunk_size
            idx = np.searchsorted(size_cumsum, target_sum)
            block_indices.append(idx)
        block_indices.append(len(size_cumsum) - 1)  # Add last index

        # Insert start index
        block_indices = np.array([0] + block_indices)

        # Calculate offsets
        indices_array = metadata.get_column('numIndices').to_numpy()
        indices_cumsum = np.insert(np.cumsum(indices_array), 0, 0)
        offset_array = indices_cumsum[block_indices]

        block_ranges = [(start, end) for start, end in zip(block_indices[:-1], block_indices[1:], strict=False)]
        return block_ranges, list(offset_array)

    @classmethod
    def run(cls,
            ldgm_metadata_path: str,
            populations: Optional[Union[str, List[str]]] = None,
            chromosomes: Optional[Union[int, List[int]]] = None,
            num_processes: Optional[int] = None,
            **kwargs) -> Any:
        """Run parallel computation.

        Args:
            input_path: Path to metadata file
            populations: Populations to process; None -> all
            chromosomes: Chromosomes to process; None -> all
            num_processes: Number of processes to use
            **kwargs: Additional arguments

        Returns:
            Results of the parallel computation
        """
        start_time = time.time()

        # Read metadata first
        metadata = read_ldgm_metadata(
            ldgm_metadata_path,
            populations=populations,
            chromosomes=chromosomes
        )

        # Get list of files from metadata
        ldgm_directory = Path(ldgm_metadata_path).parent
        edgelist_files = [ldgm_directory / block['name'] for block in metadata.iter_rows(named=True)]
        if not edgelist_files:
            raise FileNotFoundError("No edgelist files found in metadata")

        if num_processes is None:
            num_processes = min(len(edgelist_files), cpu_count())

        # Create shared memory and data structures
        shared_data = cls.create_shared_memory(metadata, **kwargs)
        print(f"Time after creating shared data: {time.time() - start_time}")

        # Calculate block offsets for each worker
        process_block_ranges, process_offsets = cls._split_blocks(metadata, num_processes)

        # Split files according to block ranges
        process_files = [edgelist_files[start:end] for start, end in process_block_ranges]

        # Create worker manager
        manager = WorkerManager(num_processes)

        # Start workers
        metadata.get_column('numIndices').to_numpy()
        for i in range(num_processes):
            manager.add_process(
                target=cls.worker,
                args=(process_files[i], manager.flags[i], shared_data, process_offsets[i])
            )
            start, end = process_block_ranges[i]

        # Run supervisor process
        results = cls.supervise(manager, shared_data)

        # Cleanup
        manager.shutdown()

        return results
