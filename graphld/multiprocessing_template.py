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

    def __getitem__(
        self, key: Union[str, Tuple[str, slice]]
    ) -> Union[np.ndarray, float]:
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

    def __setitem__(
        self,
        key: Union[str, Tuple[str, slice]],
        value: Union[np.ndarray, float]
    ):
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


class SerialManager:
    """Manager for debugging by running workers in serial."""

    def __init__(self, num_processes: int):
        """Initialize worker manager.

        Args:
            num_processes: Number of worker processes to manage
        """
        self.flags = [Value('i', 0) for _ in range(num_processes)]
        self.functions: List[Callable] = []
        self.arguments: List[tuple] = []
        self.states: List[Any] = []

    def start_workers(self, flag: Optional[int] = None) -> None:
        """Signal workers to start processing.

        Args:
            flag: Optional flag value to set (default: 1)
        """
        for f in self.flags:
            f.value = flag or 1

        offset = 0
        for i in range(len(self.flags)):
            func, args, state = self.functions[i], self.arguments[i], self.states[i]
            self.states[i] = func(state, offset, *args)
            offset += self.states[i].shape[0]

        assert all([f.value == 0 for f in self.flags])

    def await_workers(self) -> None:
        pass

    def add_process(self, target: Callable, args: Tuple) -> None:
        """Add a worker process.

        Args:
            target: Function to run in process
            args: Arguments to pass to function
        """
        self.functions.append(target)
        self.arguments.append(args)
        self.states.append(None)

    def shutdown(self) -> None:
        """Shutdown all worker processes."""
        pass


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


    @classmethod
    @abstractmethod
    def create_shared_memory(
        cls, metadata: pl.DataFrame, block_data: list, **kwargs
    ) -> 'SharedData':
        """Initialize shared memory and data structures.

        Args:
            metadata: polars dataframe continaing LDGM metadata for each LD block
            block_data: List of block-specific data
            **kwargs: Additional arguments passed from run()

        Returns:
            SharedData object containing shared memory arrays
        """
        pass

    @classmethod
    @abstractmethod
    def supervise(cls, manager: Union[WorkerManager, SerialManager],
                shared_data: SharedData,
                block_data: list, **kwargs) -> Any:
        """Monitor workers and process results.

        Args:
            manager: Worker manager for controlling processes
            shared_data: Shared memory data
            **kwargs: Additional arguments passed from run()

        Returns:
            Results of the parallel computation
        """
        pass

    @classmethod
    @abstractmethod
    def process_block(cls, ldgm: PrecisionOperator,
                    flag: Value,
                    shared_data: SharedData,
                    block_offset: int,
                    block_data: Any = None,
                    worker_params: Any = None) -> None:
        """Process single block.

        Args:
            ldgm: LDGM object
            flag: Worker flag
            shared_data: Dictionary-like shared data object
            block_offset: Offset for this block
            block_data: Optional block-specific data from prepare_block_data
            worker_params: Optional parameters passed to each worker process

        Returns:
            None
        """
        pass

    @classmethod
    def prepare_block_data(cls, metadata: pl.DataFrame, **kwargs) -> list:
        """Prepare data specific to each block for processing.

        This method should return a list of length equal to the number of blocks,
        where each element contains any block-specific data needed by process_block.
        The base implementation returns None for each block.

        Args:
            metadata: Metadata DataFrame containing block information
            **kwargs: Additional arguments passed from run()

        Returns:
            List of block-specific data, length equal to number of blocks
        """
        return [None] * len(metadata)

    @classmethod
    def worker(cls,
               files: list,
               block_data: list,
               flag: Value,
               shared_data: SharedData,
               offset: int,
               worker_params: Any = None
               ) -> None:
        """Worker process that loads LDGMs and processes blocks.

        Args:
            files: List of LDGM files to process
            block_data: List of block-specific data
            flag: Shared flag for worker control
            shared_data: Shared memory data
            offset: In shared data, where to start processing
            worker_params: Optional parameters passed to process_block
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
                starting_flag = flag.value
                for ldgm, data in zip(ldgms, block_data, strict=False):
                    cls.process_block(ldgm, flag, shared_data, block_offset, data, worker_params)
                    block_offset += ldgm.shape[0]
                assert flag.value == starting_flag, "process_block should not change flag"
                # Signal completion
                flag.value = 0

        except Exception as e:
            print(f"Error in worker: {e}")
            flag.value = -1

    @classmethod
    def serial_worker(cls,
               ldgm: Optional[PrecisionOperator],
               offset: int,
               file: str,
               data: list,
               flag: Value,
               shared_data: SharedData,
               worker_params: Any
               ) -> PrecisionOperator:
        """Worker process that loads LDGMs and processes blocks in serial.

        Args:
            ldgm: the LDGM that was previously loaded, or None
            offset: In shared data, where to start processing
            files: List of LDGM files to process
            data: List of block-specific data
            flag: Shared flag for worker control
            shared_data: Shared memory data
            worker_params: Optional parameters passed to process_block
        """
        if flag.value <= 0:
            raise ValueError("Serial worker should never be started with flag <= 0")

        # Load LDGMs once
        if ldgm is None:
            ldgm = load_ldgm(str(file))
            ldgm.factor()

        # Process all blocks and collect solutions
        cls.process_block(ldgm, flag, shared_data, offset, data, worker_params)

        # Signal completion
        flag.value = 0

        return ldgm


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

        block_ranges = [
            (start, end)
            for start, end in zip(block_indices[:-1], block_indices[1:], strict=False)
        ]
        return block_ranges, list(offset_array)

    @classmethod
    def run(cls,
            ldgm_metadata_path: str,
            populations: Optional[Union[str, List[str]]] = None,
            chromosomes: Optional[Union[int, List[int]]] = None,
            num_processes: Optional[int] = None,
            worker_params: Any = None,
            **kwargs) -> Any:
        """Run parallel computation.

        Args:
            ldgm_metadata_path: Path to metadata file
            populations: Populations to process; None -> all
            chromosomes: Chromosomes to process; None -> all
            num_processes: Number of processes to use
            worker_params: Optional parameters passed to each worker process
            **kwargs: Additional arguments

        Returns:
            Results of the parallel computation
        """


        # Read metadata first
        metadata = read_ldgm_metadata(
            ldgm_metadata_path,
            populations=populations,
            chromosomes=chromosomes
        )

        # Get list of files from metadata
        ldgm_directory = Path(ldgm_metadata_path).parent
        edgelist_files = [
            ldgm_directory / block['name']
            for block in metadata.iter_rows(named=True)
        ]
        if not edgelist_files:
            raise FileNotFoundError("No edgelist files found in metadata")

        if num_processes is None:
            num_processes = min(len(edgelist_files), cpu_count())

        # Split files among processes
        process_block_ranges, process_offsets = cls._split_blocks(metadata, num_processes)
        process_files = [edgelist_files[start:end] for start, end in process_block_ranges]

        # Data to be sent to each block individually
        block_data = cls.prepare_block_data(metadata, **kwargs)
        process_block_data = [block_data[start:end] for start, end in process_block_ranges]

        # Data shared among all blocks
        shared_data = cls.create_shared_memory(metadata, block_data, **kwargs)

        # Create worker manager
        manager = WorkerManager(num_processes)

        # Start workers
        for i in range(num_processes):
            manager.add_process(
                target=cls.worker,
                args=(
                    process_files[i],
                    process_block_data[i],
                    manager.flags[i],
                    shared_data,
                    process_offsets[i],
                    worker_params,
                )
            )

        # Run supervisor process
        results = cls.supervise(manager, shared_data, block_data, **kwargs)

        # Cleanup
        manager.shutdown()

        return results

    @classmethod
    def run_serial(cls,
            ldgm_metadata_path: str,
            populations: Optional[Union[str, List[str]]] = None,
            chromosomes: Optional[Union[int, List[int]]] = None,
            num_processes: Optional[int] = None,
            worker_params: Any = None,
            **kwargs) -> Any:
        """Run parallel computation.

        Args:
            ldgm_metadata_path: Path to metadata file
            populations: Populations to process; None -> all
            chromosomes: Chromosomes to process; None -> all
            num_processes: Number of processes to use
            worker_params: Optional parameters passed to each worker process
            **kwargs: Additional arguments

        Returns:
            Results of the parallel computation
        """

        # Read metadata first
        metadata = read_ldgm_metadata(
            ldgm_metadata_path,
            populations=populations,
            chromosomes=chromosomes
        )

        # Get list of files from metadata
        ldgm_directory = Path(ldgm_metadata_path).parent
        edgelist_files = [
            ldgm_directory / block['name']
            for block in metadata.iter_rows(named=True)
        ]
        if not edgelist_files:
            raise FileNotFoundError("No edgelist files found in metadata")

        # Data to be sent to each block individually
        block_data = cls.prepare_block_data(metadata, **kwargs)
        num_blocks = len(block_data)

        # Data shared among all blocks
        shared_data = cls.create_shared_memory(metadata, block_data, **kwargs)

        manager = SerialManager(num_blocks)

        # Start workers
        for i in range(num_blocks):
            manager.add_process(
                target=cls.serial_worker,
                args=(
                    edgelist_files[i],
                    block_data[i],
                    manager.flags[i],
                    shared_data,
                    worker_params,
                )
            )

        # Run supervisor process
        results = cls.supervise(manager, shared_data, block_data, **kwargs)

        # Cleanup
        manager.shutdown()

        return results
