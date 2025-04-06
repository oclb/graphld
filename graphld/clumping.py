"""LD clumping implementation using ParallelProcessor framework."""

from typing import Any, Union, List, Optional
import polars as pl
import numpy as np
from multiprocessing import Value

from .multiprocessing_template import ParallelProcessor, WorkerManager, SerialManager, SharedData
from .precision import PrecisionOperator
from .io import partition_variants, merge_snplists


class LDClumper(ParallelProcessor):
    """Fast LD clumping to find unlinked lead SNPs from GWAS summary statistics."""

    @classmethod
    def prepare_block_data(cls, metadata: pl.DataFrame, **kwargs) -> list[tuple]:
        """Split summary statistics into blocks whose positions match the LDGMs.
        
        Args:
            metadata: DataFrame containing LDGM metadata
            **kwargs: Additional arguments from run(), including:
                sumstats: DataFrame containing summary statistics
                
        Returns:
            List of block-specific sumstats DataFrames and their offsets
        """
        sumstats = kwargs.get('sumstats')
            
        # Partition annotations into blocks
        sumstats_blocks: list[pl.DataFrame] = partition_variants(metadata, sumstats)

        cumulative_num_variants = np.cumsum(np.array([len(df) for df in sumstats_blocks]))
        cumulative_num_variants = [0] + list(cumulative_num_variants[:-1])

        return list(zip(sumstats_blocks, cumulative_num_variants))

    @staticmethod
    def create_shared_memory(metadata: pl.DataFrame, block_data: list[tuple], **kwargs) -> SharedData:
        """Create output array with length number of variants in the summary statistics.
        
        Args:
            metadata: Metadata DataFrame containing block information
            block_data: List of block-specific sumstats DataFrames
            **kwargs: Not used
            
        Returns:
            SharedData containing arrays for clumping results
        """
        total_variants = sum([len(df) for df, _ in block_data])
        return SharedData({
            'is_index': total_variants,  # Will be converted to boolean later
        })

    @classmethod
    def process_block(cls, ldgm: PrecisionOperator,
                    flag: Value,
                    shared_data: SharedData,
                    block_offset: int,
                    block_data: tuple,
                    worker_params: tuple) -> None:
        """Process single block for LD clumping.

        Args:
            ldgm: LDGM object
            flag: Worker flag
            shared_data: Dictionary-like shared data object
            block_offset: Offset for this block
            block_data: Tuple of (sumstats DataFrame, variant_offset)
            worker_params: Tuple of (rsq_threshold, chisq_threshold, z_col, match_by_position, variant_id_col)
        """
        rsq_threshold, chisq_threshold, z_col, match_by_position, variant_id_col = worker_params
        assert isinstance(block_data, tuple), "block_data must be a tuple"
        sumstats, variant_offset = block_data
        num_variants = len(sumstats)

        # Merge variants with LDGM variant info and get indices of merged variants
        ldgm, sumstat_indices = merge_snplists(
            ldgm, sumstats,
            match_by_position=match_by_position,
            pos_col='POS',
            variant_id_col=variant_id_col,
            ref_allele_col='REF',
            alt_allele_col='ALT',
            add_allelic_cols=[z_col],  # Z score column
        )

        # Keep only first occurrence of each index
        first_index_mask = pl.Series(ldgm.variant_info.select(pl.col('is_representative')).to_numpy().flatten().astype(bool))
        ldgm.variant_info = ldgm.variant_info.filter(first_index_mask)
        sumstat_indices = sumstat_indices[first_index_mask]

        # Get Z scores and compute chi-square statistics
        z_scores = ldgm.variant_info.select(z_col).to_numpy().flatten()  # Z score column
        chisq = z_scores ** 2

        # Check original sumstats chi-square values
        original_z = sumstats.select(z_col).to_numpy().flatten()  # Z score column
        original_chisq = original_z ** 2
        assert np.allclose(original_chisq[sumstat_indices.flatten()], chisq), "Chi-square values changed after merging"

        # Sort variants by chi-square statistic
        sort_idx = np.argsort(chisq)[::-1]  # Descending order
        
        # Initialize arrays for tracking pruned and index variants
        n = len(z_scores)
        was_pruned = np.zeros(n, dtype=bool)
        is_index = np.zeros(n, dtype=bool)

        # Iterate through variants in order of decreasing chi-square
        for i in sort_idx:
            # Stop if we reach variants below threshold
            if chisq[i] < chisq_threshold:
                break
                
            # Skip if this variant was already pruned
            if was_pruned[i]:
                continue
                
            # This is an index variant
            is_index[i] = True
            
            # Compute LD with all other variants
            indicator = np.zeros(n)
            indicator[i] = 1
            ld = ldgm.solve(indicator)
            
            # Mark variants in high LD for pruning
            to_prune = (ld ** 2 >= rsq_threshold)
            assert to_prune[i]  # Lead SNP should be in LD with itself
            was_pruned[to_prune] = True

        # Initialize results array with all False
        results = np.zeros(num_variants, dtype=float)  # Use float for shared memory
        
        # Map results back to original variants using sumstat_indices
        # Only set results for variants that were successfully merged
        results[sumstat_indices.flatten()] = is_index.astype(float)  # Convert to float for shared memory
        
        # Store results
        block_slice = slice(variant_offset, variant_offset + num_variants)
        shared_data['is_index', block_slice] = results

    @classmethod
    def supervise(cls, manager: Union[WorkerManager, SerialManager], 
                shared_data: SharedData, 
                block_data: list, **kwargs) -> pl.DataFrame:
        """Monitor workers and process results.

        Args:
            manager: Worker manager for controlling processes
            shared_data: Shared memory data
            block_data: List of block-specific data
            **kwargs: Additional arguments passed from run()

        Returns:
            DataFrame with clumping results
        """
        manager.start_workers()
        manager.await_workers()
        is_index = shared_data['is_index']
        
        # Concatenate the original sumstats DataFrames in order
        sumstats = pl.concat([df for df, _ in block_data])
        
        # Add is_index column with results, converting back to boolean
        return sumstats.with_columns(pl.Series('is_index', is_index.astype(bool)))

    @classmethod
    def clump(cls,
            sumstats: pl.DataFrame,
            ldgm_metadata_path: str = 'data/ldgms/ldgm_metadata.csv',
            rsq_threshold: float = 0.1,
            chisq_threshold: float = 30.0,
            populations: Optional[Union[str, List[str]]] = None,
            chromosomes: Optional[Union[int, List[int]]] = None,
            run_in_serial: bool = False,
            num_processes: Optional[int] = None,
            z_col: str = 'Z',
            match_by_position: bool = True,
            variant_id_col: str = 'SNP',
            verbose: bool = False,
            ) -> pl.DataFrame:
        """Perform LD clumping on summary statistics.
        
        Args:
            sumstats: Summary statistics DataFrame containing Z scores
            ldgm_metadata_path: Path to metadata CSV file (default 'data/ldgms/ldgm_metadata.csv')
            rsq_threshold: r² threshold for clumping (default 0.1)
            chisq_threshold: χ² threshold for significance (default 30.0)
            populations: Optional population name(s)
            chromosomes: Optional chromosome(s)
            num_processes: Optional number of processes
            run_in_serial: Whether to run in serial mode
            z_col: Name of column containing Z scores
            match_by_position: Whether to match SNPs by position instead of ID
            variant_id_col: Name of column containing variant IDs if not matching by position
            
        Returns:
            DataFrame with additional column 'is_index' indicating index variants
        """

        run_fn = cls.run_serial if run_in_serial else cls.run
        result = run_fn(ldgm_metadata_path=ldgm_metadata_path,
                populations=populations,
                chromosomes=chromosomes,
                num_processes=num_processes,
                worker_params=(rsq_threshold, chisq_threshold, z_col, match_by_position, variant_id_col),
                sumstats=sumstats)

        if verbose:
            print(f"Number of variants in summary statistics: {len(result)}")
            nonzero_count = (result['is_index']).sum()
            print(f"Number of index variants: {nonzero_count}")
        
        return result


def run_clump(*args, **kwargs):
    """
    Perform LD-based clumping on summary statistics.

    Args:
        ldgm_metadata_path (str): Path to LDGM metadata file
        sumstats (pl.DataFrame): Summary statistics DataFrame
        min_chisq (float, optional): Minimum chi-squared threshold. Defaults to 5.0.
        max_rsq (float, optional): Maximum R-squared threshold. Defaults to 0.1.
        num_processes (Optional[int], optional): Number of processes for parallel computation. Defaults to None.
        run_in_serial (bool, optional): Whether to run in serial mode. Defaults to False.
        chromosome (Optional[int], optional): Chromosome to filter. Defaults to None.
        population (Optional[str], optional): Population to filter. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        quiet (bool, optional): Whether to suppress all output except errors. Defaults to False.

    Returns:
        pl.DataFrame: DataFrame with clumped summary statistics and index variant information
    """
    return LDClumper.clump(*args, **kwargs)
