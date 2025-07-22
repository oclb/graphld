from multiprocessing import Value
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

from .io import partition_variants
from .multiprocessing_template import ParallelProcessor, SharedData, WorkerManager
from .precision import PrecisionOperator


class BLUP(ParallelProcessor):
    """Computes the best linear unbiased predictor using LDGMs and GWAS summary statistics."""


    @classmethod
    def prepare_block_data(cls, metadata: pl.DataFrame, **kwargs) -> list[tuple]:
        """Split summary statistics into blocks whose positions match the LDGMs.
        
        Args:
            metadata: DataFrame containing LDGM metadata
            **kwargs: Additional arguments from run(), including:
                annotations: Optional DataFrame containing variant annotations
                
        Returns:
            List of block-specific annotation DataFrames, or None if no annotations
        """
        sumstats = kwargs.get('sumstats')

        # Partition annotations into blocks
        sumstats_blocks: list[pl.DataFrame] = partition_variants(metadata, sumstats)

        cumulative_num_variants = np.cumsum(np.array([len(df) for df in sumstats_blocks]))
        cumulative_num_variants = [0] + list(cumulative_num_variants[:-1])

        return list(zip(sumstats_blocks, cumulative_num_variants, strict=False))

    @staticmethod
    def create_shared_memory(metadata: pl.DataFrame, block_data: list[tuple], **kwargs) -> SharedData:
        """Create output array with length number of variants in the summary statistics that 
        migtht match to one of the blocks.
        
        Args:
            metadata: Metadata DataFrame containing block information
            block_data: List of block-specific sumstats DataFrames
            **kwargs: Not used
        """
        total_variants = sum([len(df) for df, _ in block_data])
        return SharedData({
            'beta': total_variants,    # BLUP effect sizes
        })


    @classmethod
    def process_block(cls, ldgm: PrecisionOperator, flag: Value,
                     shared_data: SharedData, block_offset: int,
                     block_data: tuple,
                     worker_params: tuple) -> None:
        """Run BLUP on a single block."""
        sigmasq, sample_size, match_by_position = worker_params
        assert isinstance(sigmasq, float), "sigmasq parameter must be a float"
        assert isinstance(block_data, tuple), "block_data must be a tuple"
        sumstats, variant_offset = block_data
        num_variants = len(sumstats)

        # Merge annotations with LDGM variant info and get indices of merged variants
        from .io import merge_snplists
        ldgm, sumstat_indices = merge_snplists(
            ldgm, sumstats,
            match_by_position=match_by_position,
            pos_col='POS',
            ref_allele_col='REF',
            alt_allele_col='ALT',
            add_allelic_cols=['Z'],
        )

        # Keep only first occurrence of each index
        first_index_mask = ldgm.variant_info.select(pl.col('index').is_first_distinct()).to_numpy().flatten()
        ldgm.variant_info = ldgm.variant_info.filter(first_index_mask)
        sumstat_indices = sumstat_indices[first_index_mask]

        # Get Z-scores from the merged variant info
        z = ldgm.variant_info.select('Z').to_numpy()

        # Compute the BLUP for this block
        beta = ldgm @ z
        ldgm.update_matrix(np.full(ldgm.shape[0], sample_size*sigmasq))
        beta = np.sqrt(sample_size) * sigmasq * ldgm.solve(beta)
        ldgm.del_factor()

        # Store results for variants that were successfully merged
        beta_reshaped = np.zeros((num_variants,1))
        # Get indices of variants that were actually merged
        beta_reshaped[sumstat_indices, 0] = beta.flatten()

        # Update the shared memory array
        block_slice = slice(variant_offset, variant_offset + num_variants)
        shared_data['beta', block_slice] = beta_reshaped

    @classmethod
    def supervise(cls, manager: WorkerManager, shared_data: Dict[str, Any], block_data: list, **kwargs) -> pl.DataFrame:
        """Supervise worker processes and collect results.
        
        Args:
            manager: Worker manager
            shared_data: Dictionary of shared memory arrays
            **kwargs: Additional arguments
            
        Returns:
            DataFrame containing simulated summary statistics
        """

        manager.start_workers()
        manager.await_workers()
        beta = shared_data['beta']
        sumstats = pl.concat([df for df, _ in block_data])
        return sumstats.with_columns(pl.Series('weight', beta))


    @classmethod
    def compute_blup(cls,
            ldgm_metadata_path: str,
            sumstats: pl.DataFrame,
            sigmasq: float,
            sample_size: float,
            populations: Optional[Union[str, List[str]]] = None,
            chromosomes: Optional[Union[int, List[int]]] = None,
            num_processes: Optional[int] = None,
            run_in_serial: bool = False,
            match_by_position: bool = False,
            verbose: bool = False,
            ) -> pl.DataFrame:
        """Simulate GWAS summary statistics for multiple LD blocks.
        
        Args:
            ldgm_metadata_path: Path to metadata CSV file
            sumstats: Sumstats dataframe containing Z scores
            populations: Optional population name
            chromosomes: Optional chromosome or list of chromosomes
            verbose: Print additional information if True
            
        Returns:
            Array of BLUP effect sizes, same length as sumstats
        """
        run_fn = cls.run_serial if run_in_serial else cls.run
        result = run_fn(
            ldgm_metadata_path=ldgm_metadata_path,
            populations=populations,
            chromosomes=chromosomes,
            num_processes=num_processes,
            worker_params=(sigmasq, sample_size, match_by_position),
            sumstats=sumstats
        )

        if verbose:
            print(f"Number of variants in summary statistics: {len(result)}")
            nonzero_count = (result['weight'] != 0).sum()
            print(f"Number of variants with nonzero weights: {nonzero_count}")

        return result

def run_blup(*args, **kwargs):
    """
    Compute Best Linear Unbiased Prediction (BLUP) weights.

    Args:
        ldgm_metadata_path (str): Path to LDGM metadata file
        sumstats (pl.DataFrame): Summary statistics DataFrame
        heritability (float): Heritability parameter (between 0 and 1)
        num_samples (Optional[int], optional): Number of samples. Defaults to None.
        num_processes (Optional[int], optional): Number of processes for parallel computation. Defaults to None.
        run_in_serial (bool, optional): Whether to run in serial mode. Defaults to False.
        chromosome (Optional[int], optional): Chromosome to filter. Defaults to None.
        population (Optional[str], optional): Population to filter. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        quiet (bool, optional): Whether to suppress all output except errors. Defaults to False.

    Returns:
        pl.DataFrame: DataFrame with BLUP weights and associated statistics
    """
    return BLUP.compute_blup(*args, **kwargs)
