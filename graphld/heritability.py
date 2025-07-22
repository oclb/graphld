"""
GraphREML
"""
from dataclasses import dataclass
from multiprocessing import Value
from typing import *

import h5py
import numpy as np
import polars as pl
import scipy.stats as sps
from filelock import FileLock

from .io import *
from .likelihood import *
from .multiprocessing_template import ParallelProcessor, SharedData, WorkerManager
from .precision import PrecisionOperator

CHUNK_SIZE = 1000

FLAGS = {
    'ERROR': -2,
    'SHUTDOWN': -1,
    'FINISHED': 0,
    'COMPUTE_ALL': 1,
    'COMPUTE_LIKELIHOOD_ONLY': 2,
    'INITIALIZE': 3,
    'WRITE_VARIANT_INFO': 4,
    'COMPUTE_VARIANT_SCORE': 5,
    'COMPUTE_VARIANT_HESSIAN': 5,
}

VARIANT_INFO_COMPRESSION_TYPE = 'lzf'

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
        link_fn_denominator: Scalar denominator for link function. 
            Recommended to be roughly the number of SNPs. Defaults to 6e6,
            roughly the number of common SNPs in Europeans.
    """
    annotation_columns: Optional[List[str]] = None
    params: Optional[np.ndarray] = None
    sample_size: Optional[float] = None
    intercept: float = 1.0
    link_fn_denominator: float = 6e6
    binary_annotations_only: bool = False

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
        convergence_tol: Convergence tolerance 
        run_serial: Run in serial rather than parallel
        num_processes: If None, autodetect
        verbose: Flag for verbose output
        use_surrogate_markers: Whether to use surrogate markers for missing variants
        trust_region_size: Initial trust region size parameter
        trust_region_rho_lb: Lower bound for trust region ratio
        trust_region_rho_ub: Upper bound for trust region ratio
        trust_region_scalar: Scaling factor for trust region updates
        max_trust_iterations: Maximum number of trust region iterations
        reset_trust_region: Whether to reset trust region size at each iteration
        num_jackknife_blocks: Number of blocks to use for jackknife estimation
        max_chisq_threshold: Maximum allowed chi^2 value in a block. Blocks with chi^2 > threshold are excluded.
        score_test_hdf5_file_name: Optional file name to create or append to an hdf5 file with pre-computed
            derivatives for the score test.
        score_test_hdf5_trait_name: Name of the trait's subdirectory within the score test HDF5 file.
    """
    gradient_num_samples: int = 100
    gradient_seed: Optional[int] = 123
    match_by_position: bool = False
    num_iterations: int = 50
    convergence_tol: float = 0.01
    run_serial: bool = False
    num_processes: Optional[int] = None
    verbose: bool = False
    use_surrogate_markers: bool = True
    trust_region_size: float = 1e-1
    trust_region_rho_lb: float = 1e-4
    trust_region_rho_ub: float = .99
    trust_region_scalar: float = 5
    max_trust_iterations: int = 100
    minimum_likelihood_increase: float = 1e-6
    convergence_window: int = 3
    reset_trust_region: bool = False
    num_jackknife_blocks: int = 100
    max_chisq_threshold: Optional[float] = None
    score_test_hdf5_file_name: Optional[str] = None
    score_test_hdf5_trait_name: Optional[str] = None

    def __post_init__(self):
        if self.score_test_hdf5_file_name is not None:
            if self.score_test_hdf5_trait_name is None:
                raise ValueError("score_test_hdf5_trait_name must be specified if score_test_hdf5_file_name is specified.")
            if not self.use_surrogate_markers:
                raise ValueError("use_surrogate_markers must be True if score_test_hdf5_file_name is specified.")
            if self.match_by_position:
                raise ValueError("match_by_position must be False if score_test_hdf5_file_name is specified.")


def _filter_binary_annotations(annotation_data: pl.DataFrame,
                               annotation_columns: List[str],
                               verbose: bool = False
                               ) -> Tuple[pl.DataFrame, List[str]]:
    """Filters annotation data to keep only binary columns (Boolean or Int64 {0,1}).

    Args:
        annotation_data: DataFrame containing annotations.
        annotation_columns: List of column names to consider for filtering.
        verbose: If True, print discarded columns.

    Returns:
        A tuple containing:
        - The filtered annotation DataFrame (may be the original if no columns dropped).
        - The list of annotation column names that were kept.
    """
    cols_to_drop = []
    cols_to_keep = []
    for c in annotation_columns:
        col_dtype = annotation_data[c].dtype
        is_binary = col_dtype == pl.Boolean
        if col_dtype == pl.Int64:
            is_binary = annotation_data.select(pl.col(c).is_in([0, 1]).all()).item()

        if is_binary:
            cols_to_keep.append(c)
        else:
            cols_to_drop.append(c)

    if verbose:
        print(f"Binary filtering: Discarding {len(cols_to_drop)} non-binary/non-Boolean annotations: {cols_to_drop}")

    filtered_annotation_data = annotation_data.drop(cols_to_drop)

    return filtered_annotation_data, cols_to_keep


def _surrogate_marker(ldgm: PrecisionOperator, missing_index: int, candidates: pl.DataFrame) -> int:
    """Find a surrogate marker for a missing variant.

    Args:
        ldgm: LDGM object
        missing_variant: Variant ID to be imputed
        candidates: Array of candidate variant IDs

    Returns:
        ID of the surrogate marker
    """
    indicator = np.zeros(ldgm.shape[0])
    indicator[missing_index] = 1
    index_correlations = ldgm.solve(indicator)
    candidate_correlations = index_correlations[candidates.select('index').to_numpy()]
    surrogate = np.argmax(candidate_correlations ** 2)

    return candidates.filter(pl.col('surrogate_nr') == surrogate).to_dicts()[0]

def softmax_robust(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax implementation."""
    y = x + np.log1p(np.exp(-x))
    mask = x < 0
    y[mask] = np.log1p(np.exp(x[mask]))
    return y

def _get_softmax_link_function(denominator: int) -> tuple[Callable, Callable, Callable]:
    """Create softmax link function and its gradient.

    Args:
        denominator: roughly num_snps / num_samples (if using Z scores) or num_snps (if using effect size estimates)

    Returns:
        Tuple containing:
        - Link function mapping (annot, theta) to per-SNP heritabilities
        - Gradient of the link function
    """
    np.seterr(over='ignore')

    def _link_arg(annot: np.ndarray, theta: np.ndarray) -> np.ndarray:
        if annot.ndim == 0:
            return annot * theta

        # Avoid platform-specific divide by zero warning for matmul with zeros matrix
        if not np.any(annot):
            return np.zeros((annot.shape[0], theta.shape[1]))

        return annot @ theta

    def _link_fn(annot: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Softmax link function."""
        return softmax_robust(_link_arg(annot, theta)) / denominator

    def _link_fn_grad(annot: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Gradient of softmax link function."""
        x = _link_arg(annot, theta)
        result = annot / denominator / (1 + np.exp(-x))
        equivalent_result = annot / denominator * np.exp(x) / (1 + np.exp(x))
        result[x.ravel() < 0, ...] = equivalent_result[x.ravel() < 0, ...]
        return result

    def _link_fn_hess(annot: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # abs() because exp(x)/(1+exp(x))^2 == 1/(1+exp(x))/(exp(-x)+1) == exp(-x)/(exp(-x)+1)^2
        x = -np.abs(_link_arg(annot, theta))
        result = np.exp((x)) * np.square(annot) / np.square(1 + np.exp(x)) / denominator
        return result


    return _link_fn, _link_fn_grad, _link_fn_hess

def _project_out(y: np.ndarray, x: np.ndarray):
    """Projects out x from y in place."""
    beta = np.linalg.solve(x.T @ x, x.T @ y.reshape(-1,1))
    y -= (x @ beta).reshape(y.shape)
    assert np.allclose(x.T @ y, np.zeros(x.shape[1]), rtol=1e-3)
    print(f"Sum of y: {np.sum(y)}")
    print(f"Shape of y: {y.shape}")

class GraphREML(ParallelProcessor):

    @classmethod
    def _sum_blocks(cls, array: np.ndarray, shape_each_block: tuple) -> np.ndarray:
        array = array.reshape((-1,*shape_each_block))
        return array.sum(axis=0)

    @classmethod
    def prepare_block_data(cls, metadata: pl.DataFrame, **kwargs) -> list[tuple]:
        """Prepare block-specific data for processing.

        Args:
            metadata: DataFrame containing LDGM metadata
            **kwargs: Additional arguments from run(), including:
                sumstats: DataFrame containing Z scores and variant info, optionally annotations
                annotation_columns: list of column names for the annotations in sumstats
                method: MethodOptions instance containing method parameters

        Returns:
            List of dictionaries containing block-specific data with keys:
                sumstats: DataFrame for this block, or None if max Z² exceeds threshold
                variant_offset: Cumulative number of variants before this block
                block_index: Index of this block
                Pz: Pre-computed precision-premultiplied Z scores for this block
        """
        sumstats: pl.DataFrame = kwargs.get('sumstats')
        method: MethodOptions = kwargs.get('method')
        sumstats_blocks: list[pl.DataFrame] = partition_variants(metadata, sumstats)
        num_block_variants = sum(len(block) for block in sumstats_blocks)
        assert num_block_variants <= sumstats.shape[0], f"Too many variants in blocks: {num_block_variants} > {sumstats.shape[0]}"
        print(f"{sum(len(block) for block in sumstats_blocks)} variants in blocks, {sumstats.shape[0]} initially")

        # Filter blocks based on max Z² threshold
        if method.max_chisq_threshold is not None:
            max_z2s = [float(np.nanmax(block.select('Z').to_numpy() ** 2)) for block in sumstats_blocks]
            keep_block = [max_z2 <= method.max_chisq_threshold for max_z2 in max_z2s]
            sumstats_blocks = [block if keep_block
                else pl.DataFrame([]) for block, max_z2 in zip(sumstats_blocks, max_z2s, strict=False)]
            if method.verbose and not all(keep_block):
                print(f"{len(sumstats_blocks)-sum(keep_block)} out of {len(sumstats_blocks)} blocks discarded due to\n"
                      f"max chisq threshold of {method.max_chisq_threshold}")

        cumulative_num_variants = np.cumsum(np.array([len(df) for df in sumstats_blocks]))
        cumulative_num_variants = [0] + list(cumulative_num_variants[:-1])
        block_indices = list(range(len(sumstats_blocks)))
        block_Pz = [None for _ in block_indices]

        return [
            {
                'sumstats': block,
                'variant_offset': offset,
                'block_index': index,
                'Pz': Pz
            }
            for block, offset, index, Pz
            in zip(sumstats_blocks, cumulative_num_variants, block_indices, block_Pz, strict=False)
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
        num_variants = sum([len(d['sumstats']) for d in block_data if d['sumstats'] is not None])

        result = SharedData({
            'params': num_params,
            'variant_data': num_variants,
            'likelihood': num_blocks,
            'gradient': num_blocks * num_params,
            'hessian': num_blocks * num_params ** 2,
        })

        return result

    @staticmethod
    def _initialize_block_zscores(ldgm: PrecisionOperator,
                                annot_df: pl.DataFrame,
                                annotation_columns: List[str],
                                match_by_position: bool,
                                verbose: bool = False
                                ):
        """Initialize Z-scores for a block by merging variants and computing Pz.
        
        Args:
            ldgm: LDGM object
            annot_df: Dataframe containing merged annotations and summary statistics
            annotation_columns: List of annotation column names

        Returns:
            Tuple containing:
            - z: Raw Z-scores after filtering
            - Pz: Z-scores premultiplied by precision matrix
        """
        # Merge annotations with LDGM variant info
        from .io import merge_snplists
        ldgm, annot_indices = merge_snplists(
            ldgm, annot_df,
            match_by_position=match_by_position,
            pos_col='POS',
            ref_allele_col='REF',
            alt_allele_col='ALT',
            add_allelic_cols=['Z'],
            add_cols=annotation_columns,
            modify_in_place=True
        )
        if verbose:
            print(f"Number of variants in sumstats before merging: {len(annot_df)}")
            print(f"Number of variants after merging: {len(ldgm.variant_info)}")
            if len(ldgm.variant_info) == 0:
                print("No variants left after merging annotations and sumstats")

        variant_info = ldgm.variant_info.with_row_index(name="vi_row_nr")
        variant_info_nonmissing = variant_info.filter(pl.col('Z').is_not_null()).with_row_index(name="surrogate_nr")
        variant_info_missing = variant_info.filter(pl.col('Z').is_null())
        indices_arr = np.array(variant_info.get_column('index').to_numpy())
        z_arr = np.array(variant_info.get_column('Z').to_numpy())
        for row in variant_info_missing.to_dicts():
            surrogate_row = _surrogate_marker(ldgm, row['index'], variant_info_nonmissing)
            indices_arr[row['vi_row_nr']] = surrogate_row['index']
            z_arr[row['vi_row_nr']] = surrogate_row['Z']

        ldgm.variant_info = variant_info.with_columns([
            pl.Series('index', indices_arr),
            pl.Series('Z', z_arr),
            pl.Series('annot_indices', annot_indices)
        ]).drop('vi_row_nr')

        if verbose:
            print(f"Number of missing rows assigned surrogates: {len(variant_info_missing)}")

        # Keep only first occurrence of each index for Z-scores
        first_index_mask = variant_info.select(pl.col('index').is_first_distinct()).to_numpy().flatten()
        z = ldgm.variant_info.select('Z').filter(first_index_mask).to_numpy()

        # Compute Pz
        Pz = ldgm @ z

        return ldgm, Pz


    @staticmethod
    def _compute_block_likelihood(ldgm: PrecisionOperator,
                           Pz: np.ndarray,
                           annotations: np.ndarray,
                           params: np.ndarray,
                           link_fn_denominator: float,
                           old_variant_h2: float,
                           num_samples: int,
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
        link_fn, link_fn_grad, _ = _get_softmax_link_function(link_fn_denominator)

        # Compute change in diag(M), aggregating variants with same index
        per_variant_h2 = link_fn(annotations, params)
        delta_D = np.zeros(ldgm.shape[0])
        np.add.at(delta_D, ldgm.variant_indices, per_variant_h2.flatten() - old_variant_h2.flatten())

        # Update diag(M)
        ldgm.update_matrix(delta_D)

        # New log likelihood
        likelihood = gaussian_likelihood(Pz, ldgm)

        if likelihood_only:
            ldgm.del_factor() # To reduce memory usage
            return likelihood, None, None, per_variant_h2

        # Gradient of per-variant h2 wrt parameters
        del_h2_del_a = link_fn_grad(annotations, params)

        # Gradient of diag(M)
        del_M_del_a = np.zeros((ldgm.shape[0], params.shape[0]))
        np.add.at(del_M_del_a, ldgm.variant_indices, del_h2_del_a)

        # Gradient of log likelihood
        gradient = gaussian_likelihood_gradient(
            Pz, ldgm, del_M_del_a=del_M_del_a,
            n_samples=num_samples,
        )

        hessian = gaussian_likelihood_hessian(
            Pz, ldgm, del_M_del_a=del_M_del_a
        )

        ldgm.del_factor() # To reduce memory usage

        return likelihood, gradient, hessian, per_variant_h2

    @staticmethod
    def _link_fn_derivatives(
        annot: np.ndarray,
        params: np.ndarray,
        link_fn_denominator: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute 1st and 2nd derivatives of the link function for each variant.

        Args:
            annot: Annotation matrix
            params: Model parameters
            link_fn_denominator: Link function denominator

        Returns:
            Tuple containing:
            - del_h2i_del_xi: Derivative of h^2_i with respect to x_i
            - del2_h2i_del_xi2: Second derivative of h^2_i with respect to x_i
        """
        _, link_fn_grad, link_fn_hess = _get_softmax_link_function(link_fn_denominator)
        x = annot @ params
        del_h2i_del_xi = link_fn_grad(np.array(1), x)
        del2_h2i_del_xi2 = link_fn_hess(np.array(1), x)
        return del_h2i_del_xi, del2_h2i_del_xi2

    @staticmethod
    def _compute_variant_grad(ldgm: PrecisionOperator,
                                  Pz: np.ndarray,
                                  del_M_del_x: np.ndarray,
                                  ) -> np.ndarray:
        """Compute gradient of log likelihood with respect to the argument of the link function for each variant.

        Args:
            ldgm: LDGM object
            Pz: Z-scores premultiplied by precision matrix

        Returns:
            Array of gradients dlogL / dx_i, where h^2_i = link(x_i)
        """
        many_samples = 200
        grad = gaussian_likelihood_gradient(Pz, ldgm, n_samples=many_samples, del_M_del_a=None)
        grad = grad[ldgm.variant_indices].ravel() * del_M_del_x.ravel()
        ldgm.del_factor()  # Free memory used by Cholesky factorization
        return grad

    @staticmethod
    def _compute_variant_hessian_diag(ldgm: PrecisionOperator,
                                      Pz: np.ndarray,
                                      del_L_del_xi: np.ndarray,
                                      del_h2i_del_xi: np.ndarray,
                                      del2_h2i_del_xi2: np.ndarray,
                                      ) -> np.ndarray:
        """Compute diagonal of the Hessian matrix with respect to the argument of the link 
        function for each variant.

        Args:
            ldgm: LDGM object
            Pz: Z-scores premultiplied by precision matrix

        Returns:
            Array containing derivatives d^2logL / dx_i^2, h_i^2 = link(x_i)
        """
        many_samples = 200

        # First term of d2L/dx2: d2L/dm2 (dm/dx)2
        first_term = gaussian_likelihood_hessian(Pz, ldgm, del_M_del_a=None,
                        diagonal_method="xdiag", n_samples=many_samples)
        first_term *= del_h2i_del_xi ** 2

        # Second term of d2L/dx2: dL/dM d2m/dx2 where dL/dM = dL/dx / dm/dx
        hess_diag = first_term + del_L_del_xi / del_h2i_del_xi * del2_h2i_del_xi2

        ldgm.del_factor()  # Free memory used by Cholesky factorization
        return hess_diag[ldgm.variant_indices]

    @staticmethod
    def _append_to_hdf5(dataset, data):
        """Helper function to append data to an HDF5 dataset.
        
        Args:
            dataset: The HDF5 dataset to append to
            data: The data to append
        """
        current_size = dataset.shape[0]
        dataset.resize(current_size + len(data), axis=0)
        if dataset.ndim == 1:
            dataset[current_size:] = data.ravel()
        else:
            dataset[current_size:] = data

    @classmethod
    def _write_variant_data(cls, hdf5_filename: str, variant_data: pl.DataFrame, annotations: np.ndarray) -> bool:
        """Checks if the file already contains a 'variants' group. 
        If not, creates it and writes variant information to it.
        
        Args:
            hdf5_filename: The name of the HDF5 file to write to.
            variant_data: Variant data to write to the 'variants' group.
            annotations: Annotations to write to the 'variants' group.

        Returns:
            bool: True if the 'variants' group was created, False otherwise.
        """
        lock = FileLock(hdf5_filename + ".lock")
        with lock:
            with h5py.File(hdf5_filename, 'a') as f:
                if 'variants' in f:
                    return False

                f.create_group('traits')
                variants_group = f.create_group('variants')
                variants_group.create_dataset('annotations',
                                            data=annotations,
                                            compression=VARIANT_INFO_COMPRESSION_TYPE,
                                            chunks=(CHUNK_SIZE, annotations.shape[1]),
                                            )
                variants_group.create_dataset('POS',
                                            data=variant_data.select('POS').to_numpy(),
                                            compression=VARIANT_INFO_COMPRESSION_TYPE,
                                            )
                variants_group.create_dataset('RSID',
                                            data=variant_data.select('SNP').to_numpy(),
                                            compression=VARIANT_INFO_COMPRESSION_TYPE,
                                            dtype=h5py.special_dtype(vlen=str),
                                            )

        return True

    @classmethod
    def _write_trait_stats(
        cls,
        method_options: MethodOptions,
        parameters: np.ndarray,
        jackknife_parameters: np.ndarray,
        score: np.ndarray,
        hessian: np.ndarray,
        jackknife_blocks: np.ndarray,
    ) -> None:
        """Create the 'trait_stats' group and write trait statistics to it.
        
        Args:
            hdf5_filename: The name of the HDF5 file to write to.
            trait_name: The name of the trait for which statistics are being created.
            parameters: The parameters of the model fit.
            jackknife_parameters: The jackknife estimates of the model parameters.
        """
        hdf5_filename = method_options.score_test_hdf5_file_name
        trait_name = method_options.score_test_hdf5_trait_name
        lock = FileLock(hdf5_filename + ".lock")
        with lock:
            with h5py.File(hdf5_filename, 'a') as f:
                traits_group = f.require_group('traits')
                if trait_name in traits_group:
                    raise ValueError(f"The group 'traits/{trait_name}' already exists.")
                group = traits_group.create_group(trait_name)

                group.create_dataset('gradient',
                            data=score,
                            compression=VARIANT_INFO_COMPRESSION_TYPE,
                            chunks=(CHUNK_SIZE,),
                            )
                print(f"Gradient shape: {score.shape}")

                group.create_dataset('hessian',
                            data=hessian,
                            compression=VARIANT_INFO_COMPRESSION_TYPE,
                            chunks=(CHUNK_SIZE,),
                            )

                group.create_dataset('jackknife_blocks',
                            data=jackknife_blocks,
                            compression=VARIANT_INFO_COMPRESSION_TYPE,
                            chunks=(CHUNK_SIZE,),
                            )

                group.create_dataset('parameters',
                            data=parameters,
                            compression=VARIANT_INFO_COMPRESSION_TYPE,
                            )

                group.create_dataset('jackknife_parameters',
                            data=jackknife_parameters,
                            compression=VARIANT_INFO_COMPRESSION_TYPE,
                            )

    @classmethod
    def process_block(cls, ldgm: PrecisionOperator,
                     flag: Value,
                     shared_data: SharedData,
                     block_offset: int,
                     block_data: Any = None,
                     worker_params: Tuple[ModelOptions, MethodOptions] = None):
        """Computes likelihood, gradient, and hessian for a single block.
        """
        model_options: ModelOptions
        method_options: MethodOptions
        model_options, method_options = worker_params

        sumstats: pl.DataFrame = block_data['sumstats']
        if len(sumstats) == 0:  # Skip blocks that were filtered out
            return

        Pz: np.ndarray
        if flag.value == FLAGS['INITIALIZE']:
            ldgm, Pz = cls._initialize_block_zscores(ldgm,
                                                    sumstats,
                                                    model_options.annotation_columns,
                                                    method_options.match_by_position,
                                                    method_options.verbose)

            # Work in effect-size as opposed to Z score units
            Pz /= np.sqrt(model_options.sample_size)
            ldgm.times_scalar(model_options.intercept / model_options.sample_size)
            block_data['Pz'] = Pz
        # ldgm is modified in place and re-used in subsequent iterations

        else:
            Pz = block_data['Pz']

        annot_indices: np.ndarray = ldgm.variant_info.select('annot_indices').to_numpy().flatten()
        max_index: int = np.max(annot_indices) + 1 if len(annot_indices) > 0 else 0
        variant_offset: int = block_data['variant_offset']
        block_variants: slice = slice(variant_offset, variant_offset + max_index)
        annot: np.ndarray = ldgm.variant_info.select(model_options.annotation_columns).to_numpy()
        params: np.ndarray = shared_data['params'].reshape(-1,1)

        # Handle variant-specific gradient and Hessian computation
        if flag.value == FLAGS['COMPUTE_VARIANT_SCORE']:
            del_h2i_del_xi, _ = cls._link_fn_derivatives(annot, params, model_options.link_fn_denominator)
            result = cls._compute_variant_grad(ldgm, Pz, del_h2i_del_xi)
            result_padded = np.zeros(max_index)
            result_padded[annot_indices] = result
            shared_data['variant_data', block_variants] = result_padded
            return

        if flag.value == FLAGS['COMPUTE_VARIANT_HESSIAN']:
            del_h2i_del_xi, del2_h2i_del_xi2 = cls._link_fn_derivatives(annot, params, model_options.link_fn_denominator)
            del_L_del_xi = shared_data['variant_data', block_variants][annot_indices].ravel()
            result = cls._compute_variant_hessian_diag(ldgm, Pz, del_L_del_xi, del_h2i_del_xi, del2_h2i_del_xi2)
            result_padded = np.zeros(max_index)
            result_padded[annot_indices] = result.ravel()
            shared_data['variant_data', block_variants] = result_padded
            return

        block_index: int = block_data['block_index']
        num_annot = len(model_options.annotation_columns)

        old_variant_h2: np.ndarray = shared_data['variant_data', block_variants][annot_indices]

        likelihood_only = (flag.value == FLAGS['COMPUTE_LIKELIHOOD_ONLY'])
        likelihood, gradient, hessian, variant_h2 = cls._compute_block_likelihood(
            ldgm=ldgm,
            Pz=Pz,
            annotations=annot,
            params=params,
            link_fn_denominator=model_options.link_fn_denominator,
            old_variant_h2=old_variant_h2,
            num_samples=method_options.gradient_num_samples,
            likelihood_only=likelihood_only
                )

        shared_data['likelihood', block_index] = likelihood
        variant_h2_padded = np.zeros(max_index)
        variant_h2_padded[annot_indices] = variant_h2.ravel()
        shared_data['variant_data', block_variants] = variant_h2_padded
        if likelihood_only:
            return

        gradient_slice = slice(block_index * num_annot,
                            (block_index + 1) * num_annot)
        shared_data['gradient', gradient_slice] = gradient.flatten()

        hessian_slice = slice(block_index * num_annot**2,
                            (block_index + 1) * num_annot**2)
        shared_data['hessian', hessian_slice] = hessian.flatten()

    @staticmethod
    def _group_blocks(blocks: np.ndarray, num_groups: int) -> np.ndarray:
        """Group blocks into a smaller number of groups.
        
        Args:
            blocks: Array of shape (..., num_blocks, *block_shape)
            num_groups: Number of groups to create
            
        Returns:
            Array of shape (..., num_groups, *block_shape) containing summed blocks
        """
        num_blocks = blocks.shape[0]
        assert num_groups <= num_blocks
        block_size = num_blocks // num_groups
        remainder = num_blocks % num_groups

        # Initialize output array
        block_shape = blocks.shape[1:]
        grouped = np.zeros((num_groups,) + block_shape)

        # Sum blocks within each group
        start_idx = 0
        for i in range(num_groups):
            # Add one extra block to some groups to handle remainder
            extra = 1 if i < remainder else 0
            end_idx = start_idx + block_size + extra

            # Sum along the block dimension
            grouped[i, ...] = blocks[start_idx:end_idx, ...].sum(axis=0)

            start_idx = end_idx

        return grouped

    @classmethod
    def _get_variant_jackknife_assignments(cls, block_indptrs: List[int], num_groups: int) -> np.ndarray:
        """Get group assignment for each variant."""
        num_blocks = len(block_indptrs) - 1
        group_sizes = cls._group_blocks(np.ones(num_blocks, dtype=int), num_groups).astype(int)

        jackknife_block_assignments = np.zeros(num_blocks)
        idx = 0
        for group, size in enumerate(group_sizes):
            jackknife_block_assignments[idx:idx+size] = group
            idx += size

        jackknife_variant_assignments = np.zeros(block_indptrs[-1], dtype=int)
        for start, end, group in zip(block_indptrs[:-1], block_indptrs[1:], jackknife_block_assignments, strict=False):
            jackknife_variant_assignments[start:end] = group

        assert np.sum(np.diff(jackknife_variant_assignments)!=0) == num_groups-1

        return jackknife_variant_assignments

    @classmethod
    def _get_groups(cls, num_blocks: int, num_groups: int) -> np.ndarray:
        """Get group assignment for each block."""
        sizes = cls._group_blocks(np.ones(num_blocks, dtype=int), num_groups).astype(int)
        result = np.zeros(num_blocks)
        idx: int = 0
        for group, size in enumerate(sizes):
            result[idx:idx+size] = group
            idx += size
        return result

    @staticmethod
    def _compute_pseudojackknife(gradient_blocks: np.ndarray, hessian_blocks: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Compute pseudo-jackknife estimates for parameters.
        
        Args:
            gradient_blocks: Array of shape (num_blocks, num_params) containing block-wise gradients
            hessian_blocks: Array of shape (num_blocks, num_params, num_params) containing block-wise Hessians
            params: Array of shape (num_params,) containing parameter values
            
        Returns:
            Array of shape (num_blocks, num_params) containing jackknife parameter estimates
        """
        num_blocks = gradient_blocks.shape[0]
        num_params = params.shape[0]

        # Sum blocks to get full gradient and Hessian
        gradient = gradient_blocks.sum(axis=0)  # Shape: (num_params,) - sum across blocks
        hessian = hessian_blocks.sum(axis=0)    # Shape: (num_params, num_params) - sum across blocks

        # Initialize output array
        jackknife = np.zeros((num_blocks, num_params))

        # Compute jackknife estimate for each block
        for block in range(num_blocks):
            # Remove current block from total gradient and Hessian
            block_gradient = gradient_blocks[block]  # Shape: (num_params,)
            block_hessian = hessian_blocks[block]  # Shape: (num_params, num_params)

            # Compute leave-one-out gradient and Hessian
            loo_gradient = gradient - block_gradient
            loo_hessian = hessian - block_hessian + 1e-12 * np.eye(num_params)

            # Compute jackknife estimate for this block
            jackknife[block] = params + np.linalg.solve(loo_hessian, loo_gradient)

        return jackknife

    @staticmethod
    def _compute_jackknife_heritability(block_data: list, jackknife_params: np.ndarray, model: ModelOptions) -> Tuple[np.ndarray, np.ndarray]:
        """Compute jackknife heritability estimates.
        
        Args:
            block_data: List of dictionaries containing block-specific data
            jackknife_params: Array of shape (num_jk_blocks, num_params) containing jackknife parameter estimates
            model: Model options containing annotation column names
            
        Returns:
            Tuple containing:
            - Array of shape (num_jk_blocks, num_annotations) containing jackknife heritability estimates
            - Array of shape (num_jk_blocks, num_annotations) containing jackknife annotation sums
        """
        num_blocks = len(block_data)
        num_jk = jackknife_params.shape[0]
        num_params = jackknife_params.shape[1]

        # Get link function
        link_fn, _, _ = _get_softmax_link_function(model.link_fn_denominator)

        # Initialize output arrays
        jackknife_h2 = np.zeros((num_jk, num_params))
        jackknife_annot_sums = np.zeros((num_jk, num_params))

        # Pre-compute annotations for each block
        block_annotations = []
        for block in range(num_blocks):
            if len(block_data[block]['sumstats']) == 0:
                continue
            annotations = block_data[block]['sumstats'].select(model.annotation_columns).to_numpy()
            block_annotations.append(annotations)

        # Compute heritability and annotation sums for each jackknife estimate
        for jk in range(num_jk):
            # Compute per-SNP heritability using jackknife parameters
            for annotations in block_annotations:
                per_snp_h2 = link_fn(annotations, jackknife_params[jk, :])
                # Add to total heritability (reshape per_snp_h2 to column vector)
                jackknife_h2[jk, :] += (per_snp_h2[:, np.newaxis] * annotations).sum(axis=0)
                # Add to annotation sums
                jackknife_annot_sums[jk, :] += annotations.sum(axis=0)

        return jackknife_h2, jackknife_annot_sums

    @staticmethod
    def _annotation_heritability(
        variant_h2: np.ndarray, annot: pl.DataFrame, ref_col: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes the heritability of each annotation based on the given h2 values.
        """
        annot_mat = annot.to_numpy().T
        annot_h2 = annot_mat @ variant_h2
        annot_size = np.sum(annot_mat, axis=1)
        annot_enrichment = annot_size[ref_col] * annot_h2 / (annot_h2[ref_col] * annot_size)

        return annot_h2, annot_enrichment

    @staticmethod
    def _wald_log10pvalue(jackknife_estimates: np.ndarray) -> float:
        """Compute Wald test log10(p-value) from jackknife estimates.

        Args:
            jackknife_estimates: Array of jackknife estimates

        Returns:
            log10(p-value): Two-sided log10(p-value) from Wald test. Returns 0.0 if all estimates are identical.
        """
        n_blocks = jackknife_estimates.shape[0]
        point_estimate = np.mean(jackknife_estimates, axis=0)

        # Check if all estimates are identical (within numerical precision)
        if np.allclose(jackknife_estimates, point_estimate, rtol=1e-24, atol=0):
            return 0.0

        standard_error = np.sqrt((n_blocks - 1) * np.var(jackknife_estimates, axis=0, ddof=1))
        # Calculate two-sided log10(p-value)
        log10pval = (np.log(2) + sps.norm.logcdf(-np.abs(point_estimate / standard_error))) / np.log(10)
        return log10pval

    @classmethod
    def supervise(cls, manager: WorkerManager, shared_data: SharedData, block_data: list, **kwargs):
        """Runs graphREML.
        Args:
            manager: used to start parallel workers
            shared_data: used to communicate with workers
            block_data: annotation + gwas data passed to workers
            **kwargs: Additional arguments
        """

        num_iterations = kwargs.get('num_iterations')
        num_params = kwargs.get('num_params')
        verbose = kwargs.get('verbose')
        method: MethodOptions = kwargs.get('method')
        model: ModelOptions = kwargs.get('model')
        trust_region_lambda = method.trust_region_size
        log_likelihood_history = []
        trust_region_history = []

        def _trust_region_step(gradient: np.ndarray, hessian: np.ndarray, trust_region_lambda: float) -> np.ndarray:
            """Compute trust region step by solving (H + λD)x = -g.
            """
            hess_mod = hessian + trust_region_lambda * np.diag(np.diag(hessian) - np.finfo(float).eps)
            step = np.linalg.solve(hess_mod, -gradient)
            # predicted_increase = step.T @ gradient + 0.5 * step.T @ (hess_mod @ step)
            predicted_increase = step.T @ gradient + 0.5 * step.T @ (hessian @ step)
            assert predicted_increase > -1e-6, f"Predicted increase must be greater than -epsilon but is {predicted_increase}."

            return step, predicted_increase


        # if model.params is not None:
        #     shared_data['params'] = model.params
        shared_data['params', 0] = -10 # Leave this for now

        last_step_bad = True
        for rep in range(num_iterations):
            if verbose:
                print(f"\n\tStarting iteration {rep}...")

            # Calculate likelihood, gradient, and hessian for each block
            flag = FLAGS['INITIALIZE'] if rep == 0 else FLAGS['COMPUTE_ALL']
            manager.start_workers(flag)
            manager.await_workers()

            likelihood = cls._sum_blocks(shared_data['likelihood'], (1,))[0]
            gradient = cls._sum_blocks(shared_data['gradient'], (num_params,))
            hessian = cls._sum_blocks(shared_data['hessian'], (num_params,num_params))

            old_params = shared_data['params'].copy()
            old_likelihood = likelihood

            # Reset trust region size if specified
            if method.reset_trust_region or last_step_bad:
                trust_region_lambda = method.trust_region_size

            # Trust region optimization loop
            previous_lambda = trust_region_lambda
            for trust_iter in range(method.max_trust_iterations):
                # Compute proposed step
                step, predicted_increase = _trust_region_step(gradient, hessian, trust_region_lambda)
                shared_data['params'] = old_params + step

                # Evaluate proposed step
                manager.start_workers(FLAGS['COMPUTE_LIKELIHOOD_ONLY'])
                manager.await_workers()
                new_likelihood = cls._sum_blocks(shared_data['likelihood'], (1,))[0]

                # Compute actual vs predicted increase
                actual_increase = new_likelihood - old_likelihood
                if verbose:
                    print(f"\tIncrease in log-likelihood: {actual_increase}, predicted increase: {predicted_increase}")

                # Check if step is acceptable and update trust region size if needed
                rho = actual_increase / predicted_increase
                if rho < method.trust_region_rho_lb:
                    if predicted_increase < method.minimum_likelihood_increase and rho >= 0:
                        if verbose:
                            print(f"\tTerminated trust region size search with predicted likelihood increase {predicted_increase}.")
                        break
                    # Reset trust region size to initial value if its below that
                    trust_region_lambda = max(method.trust_region_size,
                                            trust_region_lambda * method.trust_region_scalar)
                    shared_data['params'] = old_params  # Revert step
                elif rho > method.trust_region_rho_ub:
                    trust_region_lambda /= method.trust_region_scalar
                    break  # Accept step and continue to next iteration
                else:
                    break  # Accept step with current trust region size

                if trust_iter == method.max_trust_iterations - 1:
                    if verbose:
                        print("Warning: Maximum trust region iterations reached")

            last_step_bad = trust_region_lambda > previous_lambda * method.trust_region_scalar

            log_likelihood_history.append(new_likelihood)
            trust_region_history.append(trust_region_lambda)

            if verbose:
                print(f"log likelihood: {new_likelihood}")
                print(f"Trust region lambda: {trust_region_lambda}")
                if len(log_likelihood_history) >= 2:
                    print(f"Change in likelihood: {log_likelihood_history[-1] - log_likelihood_history[-2]}")
                total_h2 = np.sum(shared_data['variant_data'])
                print(f"Total h2 at current iteration: {total_h2}")


            # Check convergence
            converged = False
            if len(log_likelihood_history) >= 1+method.convergence_window:
                if abs(log_likelihood_history[-1] - log_likelihood_history[-method.convergence_window]) < (method.convergence_window * method.convergence_tol):
                    converged = True
                    break

        if verbose:
            print(f"-----Finished optimization after {trust_iter} out of {method.max_trust_iterations} steps-----")

        # Point estimates
        variant_h2 = shared_data['variant_data'].copy()
        annotations = pl.concat([
            dict['sumstats'].select(model.annotation_columns + ['SNP', 'POS'])
            for dict in block_data
            if len(dict['sumstats']) > 0
            ]
        )
        ref_col = 0 # Maybe TODO
        annotation_heritability, annotation_enrichment = cls._annotation_heritability(
            variant_h2, annotations.select(model.annotation_columns), ref_col)

        # Get block-wise gradients and Hessians for jackknife
        num_blocks = len(block_data)
        gradient_blocks = shared_data['gradient'].reshape((num_blocks, num_params))
        hessian_blocks = shared_data['hessian'].reshape((num_blocks, num_params, num_params))

        # Group blocks for jackknife
        num_jackknife_blocks = min(method.num_jackknife_blocks, num_blocks)
        jk_gradient_blocks = cls._group_blocks(gradient_blocks, num_jackknife_blocks)
        jk_hessian_blocks = cls._group_blocks(hessian_blocks, num_jackknife_blocks)

        # Compute jackknife estimates using the grouped blocks
        params = shared_data['params'].copy()
        jackknife_params = cls._compute_pseudojackknife(jk_gradient_blocks, jk_hessian_blocks, params)

        # Compute jackknife heritability estimates and standard errors
        jackknife_h2, jackknife_annot_sums = cls._compute_jackknife_heritability(block_data, jackknife_params, model)


        if method.score_test_hdf5_file_name is not None:
            manager.start_workers(FLAGS['COMPUTE_VARIANT_SCORE'])
            manager.await_workers()
            variant_score = shared_data['variant_data'].copy()
            manager.start_workers(FLAGS['COMPUTE_VARIANT_HESSIAN'])
            manager.await_workers()
            variant_hessian = shared_data['variant_data'].copy()

            # Jackknife block to which each variant belongs
            block_indptrs = [dict['variant_offset'] for dict in block_data] + [len(variant_score)]
            jackknife_variant_assignments = cls._get_variant_jackknife_assignments(
                block_indptrs, num_jackknife_blocks)

            # Project the annotations out of the score
            annotations_matrix = annotations.select(model.annotation_columns).to_numpy()
            _project_out(variant_score, annotations_matrix)

            cls._write_variant_data(method.score_test_hdf5_file_name,
                                    annotations.select('SNP', 'POS'),
                                    annotations_matrix,
                                    )

            cls._write_trait_stats(method,
                                    params,
                                    jackknife_params,
                                    variant_score,
                                    variant_hessian,
                                    jackknife_variant_assignments)

        # Compute standard errors using jackknife formula: SE = sqrt((n-1) * var(estimates))
        n_blocks = jackknife_params.shape[0]
        params_se = np.sqrt((n_blocks - 1) * np.var(jackknife_params, axis=0, ddof=1))
        h2_se = np.sqrt((n_blocks - 1) * np.var(jackknife_h2, axis=0, ddof=1))

        # Compute normalized heritability for each jackknife estimate
        jackknife_h2_normalized = jackknife_h2 / jackknife_annot_sums

        # Compute quotient for point estimates and SE
        jackknife_enrichment_quotient = jackknife_h2_normalized / jackknife_h2_normalized[:, [0]]
        enrichment_se = np.sqrt((n_blocks - 1) * np.var(jackknife_enrichment_quotient, axis=0, ddof=1))

        # Compute difference for p-values
        jackknife_enrichment_diff = jackknife_h2_normalized - jackknife_h2_normalized[:, [0]]

        # Two-tailed log10(p-values) using jackknife estimates
        annotation_heritability_p = np.array([
            cls._wald_log10pvalue(jackknife_h2[:, i]) for i in range(jackknife_h2.shape[1])
        ])
        # Use differences as opposed to quotients for log10(p-values)
        annotation_enrichment_p = np.array([
            cls._wald_log10pvalue(jackknife_enrichment_diff[:, i]) for i in range(jackknife_enrichment_diff.shape[1])
        ])
        params_p = np.array([
            cls._wald_log10pvalue(jackknife_params[:, i]) for i in range(jackknife_params.shape[1])
        ])

        likelihood_changes = [a - b for a, b in zip(log_likelihood_history[1:], log_likelihood_history[:-1], strict=False)]

        if verbose:
            num_annotations = len(annotation_heritability)
            print(f"Heritability: {annotation_heritability[:min(5, num_annotations)]}")
            print(f"Enrichment: {annotation_enrichment[:min(5, num_annotations)]}")
            print(f"Enrichment -log10(p-values): {-annotation_enrichment_p[:min(5, num_annotations)]}")

        return {
            'parameters': params,
            'parameters_se': params_se,
            'parameters_log10pval': params_p,
            'heritability': annotation_heritability,
            'heritability_se': h2_se,
            'heritability_log10pval': annotation_heritability_p,
            'enrichment': annotation_enrichment,
            'enrichment_se': enrichment_se,
            'enrichment_log10pval': annotation_enrichment_p,
            'likelihood_history': log_likelihood_history,
            'jackknife_h2': jackknife_h2,
            'jackknife_params': jackknife_params,
            'jackknife_enrichment': jackknife_enrichment_quotient,
            'variant_h2': variant_h2,
            'log': {
                'converged': converged,
                'num_iterations': rep + 1,
                'likelihood_changes': likelihood_changes,
                'final_likelihood': log_likelihood_history[-1],
                'trust_region_lambdas': trust_region_history,
            }
        }

def run_graphREML(model_options: ModelOptions,
                  method_options: MethodOptions,
                  summary_stats: pl.DataFrame,
                  annotation_data: pl.DataFrame,
                  ldgm_metadata_path: str,
                  populations: Union[str, List[str]] = None,
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
    if populations is None:
        raise ValueError('Populations must be provided')

    if model_options.binary_annotations_only:
        annotation_data, model_options.annotation_columns = _filter_binary_annotations(
            annotation_data,
            model_options.annotation_columns,
            method_options.verbose
        )

    if method_options.gradient_seed is not None:
        np.random.seed(method_options.gradient_seed)

    # Merge summary stats with annotations
    merge_how = 'right' if method_options.use_surrogate_markers else 'inner'
    if not method_options.match_by_position:
        join_cols = ['SNP']
        merged_data = summary_stats.join(
            annotation_data,
            on=join_cols,
            how=merge_how
        )

        # Handle column renaming - ensure we have consistent CHR and POS columns
        if 'CHR_right' in merged_data.columns and 'POS_right' in merged_data.columns:
            # Drop existing CHR and POS columns if they exist to avoid duplicates
            merged_data = merged_data.drop('CHR')\
                .drop('POS')\
                .rename({
                'CHR_right': 'CHR',
                'POS_right': 'POS'
            })

    else:
        join_cols = ['CHR', 'POS']
        merged_data = summary_stats.join(
            annotation_data,
            on=join_cols,
            how=merge_how
        )

    # Deduplicate chr/pos pairs with multiple entries in the summary statistics
    merged_data = merged_data.unique(subset=join_cols, keep='first')

    if merged_data.is_empty():
        raise ValueError(
            "No overlapping variants found between summary statistics and annotations."
        )

    # Print shape of each DataFrame
    if method_options.verbose:
        print(f"Summary stats shape: {summary_stats.shape}")
        print(f"Annotation data shape: {annotation_data.shape}")
        print(f"Merged data shape: {merged_data.shape}")
        mean_chisq = (merged_data['Z']**2).mean()
        print(f"Mean chisq: {mean_chisq}")
        max_chisq = (merged_data['Z']**2).max()
        print(f"Max chisq: {max_chisq}")

    # If sample size not provided, try to get it from sumstats
    if model_options.sample_size is None:
        if 'N' in merged_data.columns:
            mean_n = merged_data['N'].mean()
            if mean_n is not None:
                model_options.sample_size = float(mean_n)
                if method_options.verbose:
                    print(f"Using sample size N={model_options.sample_size} from sumstats")
            else:
                model_options.sample_size = 1

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
