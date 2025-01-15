"""
Functions and classes for estimating heritability using LDGM precision matrices.
"""
from multiprocessing import Value, Array
from .multiprocessing import ParallelProcessor, WorkerManager, SharedData
from dataclasses import dataclass, field
from typing import *

import numpy as np
import polars as pl

from .precision import PrecisionOperator
from .likelihood import *
from .io import *

@dataclass
class ModelOptions:
    """Stores model parameters for graphREML.

    Attributes:
        annotation_columns: names of columns to be
            used as annotations
        params: Starting parameter values
        sample_size: GWAS sample size, only needed
            for heritability scaling
        intercept: LDSC intercept or 1
        link_fn_denominator: Number of SNPs to be 
            used as denominator for link function
    """
    annotation_columns: Optional[List[str]] = None
    params: Optional[np.ndarray] = None
    sample_size: float = 1.0
    intercept: float = 1.0
    link_fn_denominator: float = 1.0

    def __post_init__(self):
        if self.annotation_columns is None:
            self.annotation_columns = ['base']
        if self.params is None:
            self.params = np.zeros((len(self.annotation_columns),1))
        
        assert self.params.ndim == 2

@dataclass
class MethodOptions:
    """Stores method parameters for graphREML.

    Attributes:
        gradient_num_samples: Number of samples for gradient estimation
        match_by_position: Use position/allele instead of RSID
            for merging
        num_iterations: Optimization steps
        convergence_tol: Convergence tolerance (TODO)
        run_serial: Run in serial rather than parallel
        num_processes: If None, autodetect
        verbose: Flag for verbose output
    """
    gradient_num_samples: int = 100
    match_by_position: bool = False
    num_iterations: int = 10
    convergence_tol: float = 1e-1
    run_serial: bool = False
    num_processes: Optional[int] = None
    verbose: bool = False


def _newton_step(gradient: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    """Compute Newton step: -H^{-1}g."""
    # print(f"Performing step with gradient: {gradient}, hessian diagonal: {np.diag(hessian)}")
    # print(hessian)
    
    ll = np.trace(hessian) / len(hessian) * 2000
    return -np.linalg.solve(hessian + ll * np.eye(len(hessian)), gradient)

def softmax_robust(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax implementation."""
    y = x + np.log1p(np.exp(-x))
    mask = x < 0
    y[mask] = np.log1p(np.exp(x[mask]))
    return y
    
def _get_softmax_link_function(n_snps: int) -> tuple[Callable, Callable]:
    """Create softmax link function and its gradient.

    Args:
        n_snps: Total number of SNPs across all blocks

    Returns:
        Tuple containing:
        - Link function mapping (annot, theta) to per-SNP heritabilities
        - Gradient of the link function
    """
    np.seterr(over='ignore')

    def _link_fn(annot: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Softmax link function."""
        return softmax_robust(annot @ theta) / n_snps

    def _link_fn_grad(annot: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Gradient of softmax link function."""
        x = annot @ theta
        result = annot / n_snps / (1 + np.exp(-x))
        mask = x.flatten() < 0
        if mask.any():
            result[mask,:] = (annot[mask,:] * np.exp(x[mask]) / 
                            (1 + np.exp(x[mask])) / n_snps)
        return result

    return _link_fn, _link_fn_grad


class GraphREML(ParallelProcessor):

    @classmethod
    def _sum_blocks(cls, array: np.ndarray, shape_each_block: tuple) -> np.ndarray:
        array = array.reshape((*shape_each_block, -1))
        result = array.sum(axis=-1)
        assert result.shape == shape_each_block
        return result

    @classmethod
    def prepare_block_data(cls, metadata: pl.DataFrame, **kwargs) -> list[tuple]:
        """Prepare block-specific data for processing.
        
        Args:
            metadata: DataFrame containing LDGM metadata
            **kwargs: Additional arguments from run(), including:
                sumstats: DataFrame containing Z scores and variant info, optionally annotations
                annotation_columns: list of column names for the annotations in sumstats
                
        Returns:
            List of dictionaries containing block-specific data with keys:
                sumstats: DataFrame for this block
                variant_offset: Cumulative number of variants before this block
                block_index: Index of this block
                Pz: Pre-computed precision-premultiplied Z scores for this block
        """
        sumstats = kwargs.get('sumstats')
        # Partition annotations into blocks
        sumstats_blocks: list[pl.DataFrame] = partition_variants(metadata, sumstats)

        cumulative_num_variants = np.cumsum(np.array([len(df) for df in sumstats_blocks]))
        cumulative_num_variants = [0] + list(cumulative_num_variants[:-1])
        block_indices = list(range(len(sumstats_blocks)))
        block_Pz = [None for _ in block_indices]
        
        return [
            {
                'sumstats': sumstats,
                'variant_offset': offset,
                'block_index': index,
                'Pz': Pz
            }
            for sumstats, offset, index, Pz 
            in zip(sumstats_blocks, cumulative_num_variants, block_indices, block_Pz)
        ]
    
    @staticmethod
    def create_shared_memory(metadata: pl.DataFrame, block_data: list[tuple], **kwargs) -> SharedData:
        """Create output array.

        Args:
            metadata: Metadata DataFrame containing block information
            block_data: List of block-specific sumstats DataFrames
            **kwargs: Not used
        """
        num_params = kwargs.get("num_params")
        num_blocks = len(metadata)
        num_variants = sum([len(d['sumstats']) for d in block_data])

        result = SharedData({
            'params': num_params,
            'variant_h2': num_variants,
            'likelihood': num_blocks,
            'gradient': num_blocks * num_params,
            'hessian': num_blocks * num_params ** 2,
            'is_first_iter': None
        })

        result['is_first_iter'] = 1

        return result

    @staticmethod
    def _initialize_block_zscores(ldgm: PrecisionOperator, 
                                sumstats: pl.DataFrame,
                                annotation_columns: List[str],
                                match_by_position: bool
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize Z-scores for a block by merging variants and computing Pz.
        
        Args:
            ldgm: LDGM object
            sumstats: Summary statistics DataFrame
            annotation_columns: List of annotation column names

        Returns:
            Tuple containing:
            - z: Raw Z-scores after filtering
            - Pz: Z-scores premultiplied by precision matrix
        """
        # Merge annotations with LDGM variant info
        from .io import merge_snplists
        ldgm, sumstat_indices = merge_snplists(
            ldgm, sumstats,
            match_by_position=match_by_position,
            pos_col='POS',
            ref_allele_col='REF',
            alt_allele_col='ALT',
            add_allelic_cols=['Z'],
            add_cols=annotation_columns,
            modify_in_place=True
        )

        # Keep only first occurrence of each index for Z-scores
        first_index_mask = ldgm.variant_info.select(pl.col('index').is_first_distinct()).to_numpy().flatten()
        z = ldgm.variant_info.select('Z').filter(first_index_mask).to_numpy()

        # Compute Pz
        Pz = ldgm @ z

        return ldgm, Pz


    @staticmethod
    def _compute_block_likelihood(ldgm: PrecisionOperator,
                           Pz: np.ndarray,
                           annotations: np.ndarray,
                           params: np.ndarray,
                           num_snps: int,
                           old_variant_h2: float,
                           n_samples: int,
                           likelihood_only: bool,
                           ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Compute likelihood, gradient, and hessian for a single block.

        Args:
            ldgm: LDGM object
            Pz: Z-scores premultiplied by precision matrix
            annotations: Annotation matrix
            params: Model parameters
            num_snps: Total number of SNPs
            old_variant_h2: Previous values of variant_h2, so that ldgm is updated using the difference

        Returns:
            Tuple containing:
            - likelihood: Log likelihood value
            - gradient: Gradient vector
            - hessian: Hessian matrix
            - per_variant_h2: With current parameters, heritability per variant
        """
        # Get link function and its gradient
        link_fn, link_fn_grad = _get_softmax_link_function(num_snps)

        # Compute precision matrix diagonal, aggregating variants with same index
        per_variant_h2 = link_fn(annotations, params)
        delta_D = np.zeros(ldgm.shape[0])
        np.add.at(delta_D, ldgm.variant_indices, per_variant_h2.flatten() - old_variant_h2.flatten())
        ldgm.update_matrix(delta_D)

        likelihood = gaussian_likelihood(Pz, ldgm)

        if likelihood_only:
            ldgm.del_factor()
            return likelihood, None, None, per_variant_h2

        # Compute gradient of precision matrix wrt parameters
        del_h2_del_a = link_fn_grad(annotations, params)
        
        del_M_del_a = np.zeros((ldgm.shape[0], params.shape[0]))
        np.add.at(del_M_del_a, ldgm.variant_indices, del_h2_del_a)


        gradient = gaussian_likelihood_gradient(
            Pz, ldgm, del_M_del_a=del_M_del_a,
            n_samples=n_samples,
        )

        # Compute hessian
        hessian = gaussian_likelihood_hessian(
            Pz, ldgm, del_M_del_a=del_M_del_a
        )
        assert ~np.any(np.isnan(hessian))

        ldgm.del_factor()

        return likelihood, gradient, hessian, per_variant_h2

    @classmethod
    def process_block(cls, ldgm: PrecisionOperator,
                     flag: Value,
                     shared_data: SharedData,
                     block_offset: int,
                     block_data: Any = None,
                     worker_params: Tuple[ModelOptions, MethodOptions] = None) -> None:
        """Computes likelihood, gradient, and hessian for a single block. If flag is 2, it only computes the likelihood.
        """
        model_options, method_options = worker_params
        num_annot = len(model_options.annotation_columns)
        block_data_dict = block_data
        sumstats = block_data_dict['sumstats']
        variant_offset = block_data_dict['variant_offset']
        block_index = block_data_dict['block_index']
        
        if shared_data['is_first_iter'] == 1:
            # First iteration - initialize Z-scores
            ldgm, Pz = cls._initialize_block_zscores(ldgm, 
                                                    sumstats, 
                                                    model_options.annotation_columns,
                                                    method_options.match_by_position)
            block_data_dict['Pz'] = Pz
        else:
            Pz = block_data_dict['Pz']

        # Get annotations from sumstats
        annot = ldgm.variant_info.select(model_options.annotation_columns).to_numpy()

        block_variants = slice(variant_offset, variant_offset + len(ldgm.variant_info))
        variant_h2 = shared_data['variant_h2', block_variants]

        likelihood_only = (flag.value==2)
        likelihood, gradient, hessian, variant_h2 = cls._compute_block_likelihood(
            ldgm=ldgm,
            Pz=Pz,
            annotations=annot,
            params=shared_data['params'].reshape(-1,1),
            num_snps=model_options.link_fn_denominator,
            old_variant_h2=variant_h2,
            n_samples=method_options.gradient_num_samples,
            likelihood_only=likelihood_only
        )

        shared_data['likelihood', block_index] = likelihood
        shared_data['variant_h2', block_variants] = variant_h2
        if likelihood_only:
            return

        gradient_slice = slice(block_index * num_annot,
                             (block_index + 1) * num_annot)
        shared_data['gradient', gradient_slice] = gradient.flatten()

        hessian_slice = slice(block_index * num_annot**2,
                            (block_index + 1) * num_annot**2)
        shared_data['hessian', hessian_slice] = hessian.flatten()


    @classmethod
    def supervise(cls, manager: WorkerManager, shared_data: SharedData, block_data: list, **kwargs):
        """Runs graphREML.
        Args:
            manager: used to start parallel workers
            shared_data: used to communicate with workers
            block_data: annotation + gwas data passed to workers
            **kwargs: Additional arguments
        """
        flags = {
            'ERROR': -2,
            'SHUTDOWN': -1,
            'FINISHED': 0,
            'COMPUTE_ALL': 1,
            'COMPUTE_LIKELIHOOD_ONLY': 2,
        }

        num_iterations = kwargs.get('num_iterations')
        num_params = kwargs.get('num_params')
        verbose = kwargs.get('verbose')
        sample_size = kwargs.get('sample_size')
        
        log_likelihood_history = []
        for rep in range(num_iterations):
            if verbose:
                print(f"starting iteration {rep} with params {shared_data['params']}")
            
            # Calculate likelihood, gradient, and hessian for each block
            manager.start_workers(flags['COMPUTE_ALL'])
            manager.await_workers()
            shared_data['is_first_iter'] = 0

            likelihood = cls._sum_blocks(shared_data['likelihood'], (1,))
            gradient = cls._sum_blocks(shared_data['gradient'], (num_params,))
            hessian = cls._sum_blocks(shared_data['hessian'], (num_params,num_params))

            # TODO replace with trust region
            step = _newton_step(gradient, hessian)
            shared_data['params'] = shared_data['params'] + step

            # as placeholder - similar will be needed for trust region
            manager.start_workers(flags['COMPUTE_LIKELIHOOD_ONLY'])
            manager.await_workers()
            assert likelihood != cls._sum_blocks(shared_data['likelihood'], (1,))
            
            log_likelihood_history.append(likelihood[0])
            heritability = np.sum(shared_data['variant_h2'] / sample_size)
            if verbose:
                print(f"heritability: {heritability}")
                print(f"Variants with nonzero h2: {np.sum(shared_data['variant_h2'] != 0)}")
                print(f"Variants: {len(shared_data['variant_h2'])}")

        return heritability, log_likelihood_history

def run_graphREML(model_options: ModelOptions,
                  method_options: MethodOptions,
                  summary_stats: pl.DataFrame,
                  annotation_data: pl.DataFrame,
                  ldgm_metadata_path: str,
                  populations: Optional[Union[str, List[str]]] = None,
                  chromosomes: Optional[Union[int, List[int]]] = None,
                  ):
    """Wrapper function for GraphREML.

    Args:
        model: Model instance containing parameters and settings
        summary_stats: DataFrame containing GWAS summary statistics
        annotation_data: DataFrame containing variant annotations
        ldgm_metadata_path: Path to LDGM metadata file
        populations: Optional list of populations to include
        chromosomes: Optional list of chromosomes to include
        num_processes: Optional number of processes for parallel computation
        max_iterations: Maximum number of iterations
        convergence_tol: Convergence tolerance
        match_by_position: If True, match variants by position instead of variant ID
        run_in_serial: If True, run in serial instead of parallel
        num_iterations: Number of repetitions

    Returns:
        Dictionary containing:
        - estimated parameters
        - heritability estimates
        - standard errors
        - convergence diagnostics
    """
    # Merge summary stats with annotations
    join_cols = ['CHR', 'POS'] if method_options.match_by_position else ['SNP']
    merged_data = summary_stats.join(
        annotation_data,
        on=join_cols,
        how='inner'
    )
    print(f"Number of variants merged with annotations: {len(merged_data)}")


    run_fn = GraphREML.run_serial if method_options.run_serial else GraphREML.run
    return run_fn(
        ldgm_metadata_path,
        populations=populations,
        chromosomes=chromosomes,
        sumstats=merged_data,
        num_processes=method_options.num_processes,
        worker_params=(model_options,method_options),
        num_params = len(model_options.annotation_columns),
        model=model_options,
        method=method_options,
        num_iterations=method_options.num_iterations,
        verbose=method_options.verbose,
        convergence_tol=method_options.convergence_tol,
        sample_size=model_options.sample_size,
    )