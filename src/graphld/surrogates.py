"""Generate surrogate-marker mapping files for LD graphical models.

Each output is an HDF5 file per population, with one int32 1-D dataset per LD
block. The dataset length is the full loaded LDGM size, and both dataset
positions and values are full loaded LDGM row indices. For variants already
present in the training sumstats, the surrogate index equals the variant index.

The heavy surrogate-marker search therefore needs to be performed once per
population and block; later GraphREML runs can look up the stored mapping.
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
from .io import merge_snplists, partition_variants
from .multiprocessing_template import ParallelProcessor, SharedData, WorkerManager


SURROGATE_FORMAT_VERSION = "1"
SURROGATE_COORDINATE_SYSTEM = "ldgm_full_index"


def _read_attr_string(h5: h5py.File, name: str) -> Optional[str]:
    """Read an HDF5 root attribute as a normal Python string."""
    value = h5.attrs.get(name)
    if isinstance(value, bytes):
        return value.decode()
    return value


def _ensure_surrogate_hdf5_contract(h5: h5py.File, path: Path) -> None:
    """Stamp or validate the surrogate-map HDF5 coordinate contract."""
    version = _read_attr_string(h5, "graphld_surrogate_format")
    coordinate_system = _read_attr_string(h5, "coordinate_system")
    if version is None and coordinate_system is None and len(h5.keys()) == 0:
        h5.attrs["graphld_surrogate_format"] = SURROGATE_FORMAT_VERSION
        h5.attrs["coordinate_system"] = SURROGATE_COORDINATE_SYSTEM
        return

    if (
        version != SURROGATE_FORMAT_VERSION
        or coordinate_system != SURROGATE_COORDINATE_SYSTEM
    ):
        raise ValueError(
            f"{path} is not a GraphLD surrogate marker file with format "
            f"{SURROGATE_FORMAT_VERSION} and coordinate system "
            f"{SURROGATE_COORDINATE_SYSTEM}. Regenerate it with "
            "`graphld surrogates`."
        )


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
        # Use merge_snplists to match the merge logic in heritability.py
        # This ensures allele matching is applied consistently
        sumstats_df = block_data['df']
        match_by_position = worker_params.get('match_by_position', False)

        # Do not modify ldgm in place: the output map is keyed by full LDGM row
        # coordinates, while merge_snplists renumbers variant_info["index"].
        merged_ldgm, _ = merge_snplists(
            ldgm, sumstats_df,
            match_by_position=match_by_position,
            pos_col='POS',
            ref_allele_col='REF',
            alt_allele_col='ALT',
            modify_in_place=False
        )

        # After merge_snplists, merged_ldgm.variant_info contains only variants that
        # matched sumstats (with allele checking). Get their unique indices.
        # Work at index-level: length equals number of unique LDGM indices in this block
        num_indices = ldgm.shape[0]
        mapping = np.arange(num_indices, dtype=np.int32)

        # Candidates are unique full-LDGM row indices among non-missing variants.
        candidate_indices = np.array(
            merged_ldgm._which_indices
            if merged_ldgm._which_indices is not None
            else np.arange(merged_ldgm.shape[0]),
            dtype=np.int64,
        )
        candidates = pl.DataFrame({'index': candidate_indices}).with_row_index(
            name='surrogate_nr'
        )

        if len(candidates) > 0:
            missing_indices = np.setdiff1d(np.arange(num_indices), candidates['index'].to_numpy())
            for mi in missing_indices:
                surrogate = _surrogate_marker(ldgm, mi, candidates)
                mapping[mi] = int(surrogate['index'])
        else:
            print(f"No non-missing variants found in block {block_data['block_name']}. Skipping.")

        # Persist to HDF5: one dataset per block named by metadata 'name'
        out_path = worker_params['output_path']

        # Ensure directory exists
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize writes across processes
        lock = FileLock(str(out_path) + ".lock")
        with lock:
            with h5py.File(out_path, 'a') as h5:
                _ensure_surrogate_hdf5_contract(h5, out_path)
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
    chromosomes: Optional[int] = None,
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
        chromosomes=chromosomes,
        nonmissing_variant_ids=nonmissing_variant_ids,
        num_processes=num_processes,
        worker_params={'output_path': Path(output_path)},
        output_path=Path(output_path),
    )

    return result_path
