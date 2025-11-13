"""Generate surrogate-marker mapping files for LD graphical models.

Each output is an HDF5 file *per population*.  Within the file there is one
HDF5 *group* per LD block.  Every group stores three 1-D datasets with a
common length equal to the number of variants in the LDGM block:

    variant_id       — UTF-8 string     (snplist ``site_ids`` column)
    variant_index    — int32            (row index inside the LDGM)
    surrogate_index  — int32            (row index of best surrogate)

For variants that are already non-missing in the training sum-stats the
surrogate index equals the variant index.

The heavy (and slow) search for a surrogate marker therefore needs to be
performed only once per population and block; subsequent GraphREML runs can
simply look up the stored mapping.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import polars as pl
import h5py
from filelock import FileLock

from .heritability import _surrogate_marker
from .io import partition_variants
from .multiprocessing_template import ParallelProcessor, SharedData, WorkerManager


class Surrogates(ParallelProcessor):

    @classmethod
    def prepare_block_data(cls, metadata: pl.DataFrame, **kwargs) -> list[dict]:
        """Partition dataframe with nonmissing variants among blocks and attach block names."""
        nonmissing_variant_ids: pl.DataFrame = kwargs.get('nonmissing_variant_ids')
        dfs = partition_variants(metadata, nonmissing_variant_ids)
        names = metadata.get_column('name').to_numpy().tolist()
        return [
            {'df': df, 'block_name': name}
            for df, name in zip(dfs, names, strict=False)
        ]

    @classmethod
    def create_shared_memory(cls, metadata, block_data, **kwargs) -> SharedData:
        """No shared memory needed; return empty container for interface compliance."""
        return SharedData({})

    @classmethod
    def supervise(
        cls,
        manager: WorkerManager,
        shared_data: SharedData,
        block_data: list[dict],
        **kwargs
    ) -> np.ndarray:
        """Wait for all workers to finish and return output path."""
        manager.start_workers()
        manager.await_workers()
        return kwargs.get('output_path')

    @classmethod
    def process_block(
        cls,
        ldgm,
        flag,
        shared_data: SharedData,
        block_offset: int,
        block_data: dict,
        worker_params: dict,
    ):
        """Compute surrogate markers and save to file."""
        df = (block_data['df']
            .with_columns(pl.lit(True).alias('is_nonmissing'))
            .group_by('SNP')
            .first()
            .join(
                ldgm.variant_info,
                left_on='SNP',
                right_on='site_ids',
                how='right',
                maintain_order='right'
            )
            .with_columns(pl.col('is_nonmissing').fill_null(False))
        )
        # Work at index-level: length equals number of unique LDGM indices in this block
        num_indices = ldgm.shape[0]
        mapping = np.arange(num_indices, dtype=np.int32)

        # Candidates are unique indices among non-missing rows
        candidates = (
            df.filter(pl.col('is_nonmissing'))
              .select('index')
              .unique()
              .with_row_index(name='surrogate_nr')
        )

        missing_indices = np.setdiff1d(np.arange(num_indices), candidates['index'].to_numpy())
        for mi in missing_indices:
            surrogate = _surrogate_marker(ldgm, mi, candidates)
            mapping[mi] = int(surrogate['index'])

        # Persist to HDF5: one dataset per block named by metadata 'name'
        out_path = worker_params['output_path']

        # Ensure directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize writes across processes
        lock = FileLock(str(out_path) + ".lock")
        with lock:
            with h5py.File(out_path, 'a') as h5:
                dset_name = str(block_data['block_name'])
                if dset_name in h5:
                    raise ValueError(f"Dataset {dset_name} already exists in {out_path}")
                h5.create_dataset(dset_name, data=mapping, compression='lzf', chunks=True)
        

def get_surrogate_markers(
    metadata_path: Union[str, os.PathLike],
    nonmissing_variant_ids: pl.DataFrame,
    *,
    population: str,
    run_serial: bool = False,
    num_processes: Optional[int] = None,
    output_path: Optional[Union[str, os.PathLike]] = None,
) -> Path:
    """Create an HDF5 file with one dataset per LD block containing index-level surrogates.

    Returns the path to the HDF5 file.
    """
    run_fn = Surrogates.run_serial if run_serial else Surrogates.run

    # Default output path: alongside metadata file
    if output_path is None:
        output_path = Path(metadata_path).parent / f"surrogates.{population}.h5"

    result_path = run_fn(
        ldgm_metadata_path=metadata_path,
        populations=population,
        nonmissing_variant_ids=nonmissing_variant_ids,
        num_processes=num_processes,
        worker_params={'output_path': Path(output_path)},
        output_path=Path(output_path),
    )

    return result_path
