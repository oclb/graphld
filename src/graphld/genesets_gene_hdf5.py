import numpy as np
from scipy.sparse import csr_matrix
import polars as pl
import h5py
import click
from graphld.score_test import load_variant_data, load_trait_data

COMPRESSION_TYPE = 'lzf'
CHUNK_SIZE = 1000
POSITION_SCALE = 1e9  # Scale factor for chromosome positions

def get_nearest_genes(var_pos: np.ndarray, 
                        gene_pos: np.ndarray, 
                        num_nearest: int, 
                        ) -> np.ndarray:

    # Ensure 1D inputs (HDF5 may store as Nx1)
    var_pos = np.asarray(var_pos).ravel()
    gene_pos = np.asarray(gene_pos).ravel()
    if not np.array_equal(var_pos, np.sort(var_pos)):
        raise ValueError("Variant positions must be sorted")
    if not np.array_equal(gene_pos, np.sort(gene_pos)):
        raise ValueError("Gene positions must be sorted")
    nvar, ngene = var_pos.size, gene_pos.size
    
    # how many genes come before each variant?
    order = np.argsort(np.concatenate((var_pos, gene_pos)))
    perm = np.empty(nvar + ngene)
    perm[order] = np.arange(nvar + ngene) # number of genes or variants before
    num_genes_before = perm[:nvar] - np.arange(nvar)

    dist_k_flanking = np.empty((nvar, 2*num_nearest), dtype=np.int32)
    genes_k_flanking = np.empty((nvar, 2*num_nearest), dtype=np.int32)
    for k in range(-num_nearest, num_nearest):
        # fancy indexing + wraparound
        genes_k_flanking[:,k] = (num_genes_before + k) % ngene
        dist_k_flanking[:,k] = abs(var_pos - gene_pos[genes_k_flanking[:,k]])
    
    dist_k_flanking = dist_k_flanking.ravel()
    genes_k_flanking = genes_k_flanking.ravel()

    result = np.empty((nvar, num_nearest), dtype=np.int32)
    left_contestant = np.arange(0, 2*nvar*num_nearest, 2*num_nearest, dtype=np.int32)
    right_contestant = left_contestant + num_nearest*2-1
    for k in range(num_nearest):
        right_wins = dist_k_flanking[right_contestant] < dist_k_flanking[left_contestant]
        left_wins = ~right_wins
        result[right_wins,k] = genes_k_flanking[right_contestant[right_wins]]
        result[left_wins,k] = genes_k_flanking[left_contestant[left_wins]]
        right_contestant -= right_wins
        left_contestant += left_wins
    
    return result

def get_gene_variant_matrix(
    var_pos: np.ndarray,
    gene_pos: np.ndarray,
    nearest_weights: np.ndarray,
    dtype: np.dtype = np.int8,
) -> csr_matrix:
    """Build a sparse variants x genes weighted matrix.

    M[i, j] = nearest_weights[k] if gene j is the k-th closest gene to variant i, else 0.

    Parameters
    ----------
    var_pos: np.ndarray
        Variant positions (1D array of length nvar).
    gene_pos: np.ndarray
        Gene positions (1D array of length ngene).
    nearest_weights: np.ndarray
        Weights for the k-nearest genes (1D array of length num_nearest).
    dtype: np.dtype
        Data type of the sparse matrix values (defaults to int8).

    Returns
    -------
    scipy.sparse.csr_matrix
        CSR matrix of shape (nvar, ngene) where entry (i,j) equals nearest_weights[k]
        if gene j is the k-th closest to variant i, otherwise 0.
    """
    nvar, ngene = len(var_pos), len(gene_pos)
    num_nearest = len(nearest_weights)
    assert num_nearest > 0 and num_nearest <= ngene

    nearest = get_nearest_genes(var_pos, gene_pos, num_nearest)
    # Row indices: each variant repeated num_nearest times
    rows = np.repeat(np.arange(nvar, dtype=np.int32), num_nearest)
    cols = nearest.ravel()
    data = np.tile(nearest_weights, nvar)
    return csr_matrix((data, (rows, cols)), shape=(nvar, ngene))

def compute_gene_jackknife_blocks(M: csr_matrix, variant_blocks: pl.Series) -> np.ndarray:
    """Assign a jackknife block to each gene by averaging variant block indices.

    Uses: blocks_gene = round( (variant_blocks @ M) / (1 @ M) ).

    For genes with no assigned variants (denominator == 0), returns -1.
    """
    denom = np.asarray(M.sum(axis=0)).ravel().astype(float)
    num = np.asarray(variant_blocks.to_numpy().reshape(1, -1) @ M).ravel()
    mean_blocks = np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)
    gene_blocks = np.rint(mean_blocks).astype(np.int32)
    gene_blocks[denom == 0] = -1
    return gene_blocks
    
def compute_variant_positions(variant_data: pl.DataFrame) -> np.ndarray:
    """Compute global positions for variants (CHR * POSITION_SCALE + POS)."""
    return (variant_data['POS'] + variant_data['CHR'] * POSITION_SCALE).to_numpy().astype(np.int64)

def compute_gene_positions(gene_table: pl.DataFrame) -> np.ndarray:
    """Compute global positions for genes (CHR * POSITION_SCALE + midpoint)."""
    return (gene_table['midpoint'] + gene_table['CHR'].cast(pl.Int64) * POSITION_SCALE).to_numpy().astype(np.int64)

def read_genes_tsv(gene_table: str) -> pl.DataFrame:
    schema = {
        'gene_id': pl.Utf8,
        'gene_id_version': pl.Utf8,
        'gene_name': pl.Utf8,
        'start': pl.Int64,
        'end': pl.Int64,
        'CHR': pl.Utf8,
    }
    return pl.scan_csv(gene_table, schema=schema, separator='\t', has_header=True) \
        .filter(pl.col('CHR').is_in([str(i) for i in range(1,23)])) \
        .filter(pl.col('gene_id').is_not_null()) \
        .with_columns(pl.col('gene_name').fill_null('NA')) \
        .with_columns(((pl.col('start') + pl.col('end')) / 2).alias('midpoint')) \
        .sort(pl.col('CHR').cast(pl.Int64), 'midpoint') \
        .collect()

def get_test_gene_set(gene_table: pl.DataFrame) -> set[str]:
    """Get an arbitrary test gene set (first 10% of genes alphabetically by name)."""
    all_genes = sorted(gene_table['gene_name'].to_list())
    n_test = max(1, len(all_genes) // 10)
    return set(all_genes[:n_test])


def compute_gene_variant_matrix_from_data(variant_data: pl.DataFrame,
                                         gene_table: pl.DataFrame,
                                         nearest_weights: np.ndarray) -> csr_matrix:
    """Compute gene-variant matrix from variant and gene data.
    
    This is the canonical function for computing the gene-variant matrix.
    Both convert_variant_to_gene_hdf5 and create_variant_annot_internal should use this.
    """
    variant_positions = compute_variant_positions(variant_data)
    gene_positions = compute_gene_positions(gene_table)
    return get_gene_variant_matrix(variant_positions, gene_positions, nearest_weights)


def compute_gene_level_data(variant_data: pl.DataFrame,
                           gene_table: pl.DataFrame,
                           nearest_weights: np.ndarray) -> tuple[pl.DataFrame, csr_matrix, np.ndarray, np.ndarray]:
    """Compute gene-level data from variant-level data.
    
    Parameters
    ----------
    variant_data : pl.DataFrame
        Variant-level data with columns: CHR, POS, annotations, jackknife_blocks
    gene_table : pl.DataFrame
        Gene table with columns: gene_id, gene_name, CHR, start, end, midpoint
    nearest_weights : np.ndarray
        Weights for k-nearest genes
        
    Returns
    -------
    gene_table : pl.DataFrame
        Filtered gene table (only chromosomes present in variant data)
    gene_variant_matrix : csr_matrix
        Sparse matrix mapping variants to genes
    gene_blocks : np.ndarray
        Jackknife block assignments for genes
    gene_annotations : np.ndarray
        Gene-level annotations (aggregated from variants)
    """
    # Get chromosomes present in variant data
    variant_chrs = variant_data['CHR'].unique().sort().to_list()
    
    # Filter genes to only those on chromosomes present in variant data
    gene_table = gene_table.filter(pl.col('CHR').cast(pl.Int64).is_in(variant_chrs))
    
    # Compute gene-variant matrix using canonical function
    gene_variant_matrix = compute_gene_variant_matrix_from_data(variant_data, gene_table, nearest_weights)
    gene_blocks = compute_gene_jackknife_blocks(gene_variant_matrix, variant_data['jackknife_blocks'])
    
    # Assert that gene blocks are contiguous
    unique_blocks = len(np.unique(gene_blocks[gene_blocks >= 0]))
    transitions = np.sum(np.diff(gene_blocks) != 0)
    assert transitions == unique_blocks - 1, \
        f"Gene jackknife blocks must be contiguous: found {transitions} transitions for {unique_blocks} unique blocks"

    return gene_variant_matrix, gene_blocks


def convert_variant_to_gene_hdf5(variant_hdf5: str, 
                                gene_table_path: str, 
                                gene_hdf5: str, 
                                nearest_weights: np.ndarray) -> None:
    """Convert variant-level HDF5 to gene-level HDF5.
    
    Parameters
    ----------
    variant_hdf5 : str
        Path to variant-level HDF5 file
    gene_table_path : str
        Path to gene table TSV file
    gene_hdf5 : str
        Path to output gene-level HDF5 file
    nearest_weights : np.ndarray
        Weights for k-nearest genes
    """
    # Load data
    variant_data = load_variant_data(variant_hdf5)
    gene_table = read_genes_tsv(gene_table_path)
    
    # Get chromosomes present in variant data
    variant_chrs = variant_data['CHR'].unique().sort().to_list()
    print(f"Chromosomes in variant data: {variant_chrs}")
    
    # Filter gene table to chromosomes present in variant data
    gene_table = gene_table.filter(pl.col('CHR').cast(pl.Int64).is_in(variant_chrs))
    print(f"Filtered to {len(gene_table)} genes on chromosomes {variant_chrs}")
    
    # Compute gene-level data
    gene_variant_matrix, gene_blocks = compute_gene_level_data(
        variant_data, gene_table, nearest_weights
    )

    # Load a trait to get hessian for computing gene-level annotations
    # We need to compute J = X^T @ diag(H) @ W where X is variant annotations, 
    # H is variant hessian, W is gene-variant matrix
    with h5py.File(variant_hdf5, 'r') as vf:
        trait_names = list(vf['traits'].keys())
    # Use first trait's hessian to compute gene annotations
    first_trait_data = load_trait_data(variant_hdf5, trait_name=trait_names[0])
    gene_annotations = (variant_data['annotations'].to_numpy().T @ 
                       (first_trait_data['hessian'].reshape(-1,1) * gene_variant_matrix.toarray())).T
    
    print(gene_table.head())
    
    # TEST: Compute score for arbitrary test gene set
    test_gene_set = get_test_gene_set(gene_table)
    print(f"\n=== TEST GENE SET (convert_variant_to_gene_hdf5) ===")
    print(f"Test gene set size: {len(test_gene_set)}")
    gene_indicator = np.array([1.0 if name in test_gene_set else 0.0 
                               for name in gene_table['gene_name'].to_list()], dtype=np.float64)
    variant_annot = gene_variant_matrix @ gene_indicator.reshape(-1, 1)
    trait_data = load_trait_data(variant_hdf5, trait_name='data_sumstats_sumstats_body_BMIz.sumstats')
    variant_score = np.dot(trait_data['gradient'].ravel(), variant_annot.ravel())
    print(f"Variant-level score: {variant_score:.10f}")
    gene_grad = gene_variant_matrix.T @ trait_data['gradient'].reshape(-1, 1)
    gene_score = np.dot(gene_grad.ravel(), gene_indicator)
    print(f"Gene-level score: {gene_score:.10f}")
    print(f"Difference: {abs(variant_score - gene_score):.2e}")
    print("=" * 50)
    
    # Compute gene positions for saving to HDF5
    gene_positions = compute_gene_positions(gene_table)

    with h5py.File(gene_hdf5, 'w') as f:
        variants_group = f.create_group('variants')
        chunk_size = min(CHUNK_SIZE, gene_annotations.shape[0])
        variants_group.create_dataset('annotations',
                                    data=gene_annotations,
                                    compression=COMPRESSION_TYPE,
                                    chunks=(chunk_size, gene_annotations.shape[1]),
                                    dtype=np.float64,
                                    )
        variants_group.create_dataset('CHR',
                                    data=gene_table.select(pl.col('CHR').cast(pl.Int32)).to_numpy().ravel(),
                                    compression=COMPRESSION_TYPE,
                                    dtype=np.int32,
                                    )
        variants_group.create_dataset('POS',
                                    data=gene_positions,
                                    compression=COMPRESSION_TYPE,
                                    dtype=np.int32,
                                    )
        variants_group.create_dataset('ENSGID',
                                    data=gene_table.select('gene_id').to_series().to_list(),
                                    compression=COMPRESSION_TYPE,
                                    )
        variants_group.create_dataset('gene_name',
                                    data=gene_table.select('gene_name').to_series().to_list(),
                                    compression=COMPRESSION_TYPE,
                                    )
        # Save jackknife blocks once for variants
        variants_group.create_dataset('jackknife_blocks',
                                      data=gene_blocks,
                                      compression=COMPRESSION_TYPE,
                                      dtype=np.int32,
                                      )

        # Determine trait names from the input HDF5
        with h5py.File(variant_hdf5, 'r') as vf:
            trait_names = list(vf['traits'].keys())

        traits_group = f.create_group('traits')
        for trait in trait_names:
            trait_data = load_trait_data(variant_hdf5, trait_name=trait)
            gene_grad = gene_variant_matrix.T @ trait_data['gradient'].reshape(-1, 1)
            # Gene-level hessian is all-ones vector
            gene_hess = np.ones((len(gene_table), 1), dtype=np.float64)
            trait_group = traits_group.create_group(trait)
            trait_group.create_dataset('gradient',
                                        data=gene_grad,
                                        compression=COMPRESSION_TYPE,
                                        dtype=np.float64,
                                        )
            trait_group.create_dataset('hessian',
                                        data=gene_hess,
                                        compression=COMPRESSION_TYPE,
                                        dtype=np.float64,
                                        )
            trait_group.create_dataset('jackknife_parameters',
                                        data=trait_data['jackknife_parameters'],
                                        compression=COMPRESSION_TYPE,
                                        dtype=np.float64,
                                        )
            trait_group.create_dataset('parameters',
                                        data=trait_data['parameters'],
                                        compression=COMPRESSION_TYPE,
                                        dtype=np.float64,
                                        )
                                        
                                    
        
def create_variant_annot_internal(variant_hdf5: str, env_dir: str, gene_list: str, add_random_float: float | None, 
                    chromosome: str, nearest_weights: tuple[float, ...], output_prefix: str):
    """Internal function to create variant and gene annotations for a gene set."""
    import os
    
    # Find gene table in env_dir
    gene_table_path = os.path.join(env_dir, "genes.tsv")
    if not os.path.exists(gene_table_path):
        raise FileNotFoundError(f"genes.tsv not found in {env_dir}")
    
    # Load variant data
    variant_data = load_variant_data(variant_hdf5)
    
    # Get chromosomes present in variant data
    variant_chrs = variant_data['CHR'].unique().sort().to_list()
    print(f"Chromosomes in variant data: {variant_chrs}")
    
    # Load and filter gene table
    gene_table = read_genes_tsv(gene_table_path)
    gene_table = gene_table.filter(pl.col('CHR').cast(pl.Int64).is_in(variant_chrs))
    
    # Apply chromosome filter if specified
    if chromosome is not None:
        gene_table = gene_table.filter(pl.col('CHR') == str(chromosome))
        print(f"Filtered to chromosome {chromosome}")
    
    print(f"Total genes available: {len(gene_table)}")
    
    # Get gene names to test
    if add_random_float is not None:
        # Randomly sample genes
        n_sample = int(len(gene_table) * add_random_float)
        sampled_indices = np.random.choice(len(gene_table), size=n_sample, replace=False)
        sampled_gene_names = [gene_table['gene_name'][int(i)] for i in sampled_indices]
        gene_names_in_set = set(sampled_gene_names)
        print(f"Randomly sampled {n_sample} genes ({add_random_float*100:.1f}%)")
        print(f"Unique gene names in sample: {len(gene_names_in_set)}")
    else:
        # Load gene names from file
        if gene_list is None:
            raise ValueError("Must provide gene_list file or --add-random-float")
        with open(gene_list, 'r') as f:
            gene_names_in_set = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(gene_names_in_set)} genes from {gene_list}")
    
    # Get gene-variant matrix using canonical function
    weights = np.array(nearest_weights, dtype=np.float64)
    gene_variant_matrix = compute_gene_variant_matrix_from_data(variant_data, gene_table, weights)
    
    print(f"\nGene-variant matrix shape: {gene_variant_matrix.shape}")
    print(f"Number of non-zero entries: {gene_variant_matrix.nnz}")
    print(f"Sparsity: {gene_variant_matrix.nnz / (gene_variant_matrix.shape[0] * gene_variant_matrix.shape[1]):.4f}")
    
    # TEST: Compute score for arbitrary test gene set
    test_gene_set = get_test_gene_set(gene_table)
    print(f"\n=== TEST GENE SET (create_variant_annot_internal) ===")
    print(f"Test gene set size: {len(test_gene_set)}")
    test_gene_indicator = np.array([1.0 if name in test_gene_set else 0.0 
                                    for name in gene_table['gene_name'].to_list()], dtype=np.float64)
    test_variant_annot = gene_variant_matrix @ test_gene_indicator.reshape(-1, 1)
    trait_data = load_trait_data(variant_hdf5, trait_name='data_sumstats_sumstats_body_BMIz.sumstats')
    test_variant_score = np.dot(trait_data['gradient'].ravel(), test_variant_annot.ravel())
    print(f"Variant-level score: {test_variant_score:.10f}")
    test_gene_grad = gene_variant_matrix.T @ trait_data['gradient'].reshape(-1, 1)
    test_gene_score = np.dot(test_gene_grad.ravel(), test_gene_indicator)
    print(f"Gene-level score: {test_gene_score:.10f}")
    print(f"Difference: {abs(test_variant_score - test_gene_score):.2e}")
    print("=" * 50)
    
    # Create indicator vector for genes in the set
    all_gene_names = gene_table['gene_name'].to_list()
    gene_indicator = np.array([1.0 if name in gene_names_in_set else 0.0 
                                for name in all_gene_names], dtype=np.float64)
    print(f"Genes in set: {gene_indicator.sum():.0f} out of {len(gene_indicator)}")
    
    # Check for duplicate gene names
    unique_all_genes = len(set(all_gene_names))
    if unique_all_genes != len(all_gene_names):
        print(f"WARNING: Gene table has {len(all_gene_names) - unique_all_genes} duplicate gene names")
    
    # Compute variant annotation by dotting gene_variant_matrix with indicator
    variant_annot = gene_variant_matrix @ gene_indicator.reshape(-1, 1)
    print(f"Variant annotation mean: {variant_annot.mean():.6f}")
    
    # Save variant annotations in LDSC format
    variant_annot_df = pl.DataFrame({
        'CHR': variant_data['CHR'],
        'BP': variant_data['POS'],
        'SNP': variant_data['RSID'],
        'CM': pl.Series([0.0] * len(variant_data)),  # Placeholder
        'gene_set': variant_annot.ravel()
    })
    
    variant_output = f"{output_prefix}.variant.annot"
    variant_annot_df.write_csv(variant_output, separator='\t')
    print(f"Saved variant annotations to {variant_output}")
    
    # Create gene annotations (indicator for genes in set)
    gene_annot_df = pl.DataFrame({
        'gene_id': gene_table['gene_id'],
        'gene_name': gene_table['gene_name'],
        'CHR': gene_table['CHR'],
        'start': gene_table['start'],
        'end': gene_table['end'],
        'in_gene_set': gene_indicator
    })
    
    gene_output = f"{output_prefix}.gene.annot"
    gene_annot_df.write_csv(gene_output, separator='\t')
    print(f"Saved gene annotations to {gene_output}")
    
    return variant_output, gene_output


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("variant_hdf5", type=click.Path(exists=True, dir_okay=False))
@click.argument("env_dir", type=click.Path(exists=True, file_okay=False))
@click.argument("output_prefix", type=str)
@click.argument("gene_list", type=click.Path(exists=True, dir_okay=False), required=False)
@click.option(
    "--add-random-float",
    type=float,
    default=None,
    help="If specified, randomly sample this fraction of genes from gene_table instead of using gene_list file.",
)
@click.option(
    "-c", "--chr",
    "chromosome",
    type=str,
    default=None,
    help="Chromosome name to filter (e.g., '22').",
)
@click.option(
    "--nearest-weight",
    "nearest_weights",
    type=float,
    multiple=True,
    default=[.5,.2,.1,.1,.1],
    help="Repeat to specify weights for the k-nearest genes (e.g., --nearest-weight 1 --nearest-weight 0.5).",
)
def create_variant_annot(variant_hdf5: str, env_dir: str, output_prefix: str, gene_list: str, 
                         add_random_float: float, chromosome: str, nearest_weights: tuple[float, ...]):
    """Create variant and gene annotations for a gene set.
    
    VARIANT_HDF5: Path to variant HDF5 file
    ENV_DIR: Directory containing genes.tsv
    OUTPUT_PREFIX: Prefix for output files (will create .variant.annot and .gene.annot)
    GENE_LIST: Optional file with gene names (one per line)
    """
    weights = tuple(nearest_weights)
    create_variant_annot_internal(variant_hdf5, env_dir, gene_list, add_random_float, 
                                   chromosome, weights, output_prefix)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("variant_hdf5", type=click.Path(exists=True, dir_okay=False))
@click.argument("gene_table", type=click.Path(exists=True, dir_okay=False))
@click.argument("gene_hdf5", type=click.Path(dir_okay=False))
@click.option(
    "--nearest-weight",
    "nearest_weights",
    type=float,
    multiple=True,
    default=[.5,.2,.1,.1,.1],
    help="Repeat to specify weights for the k-nearest genes (e.g., --nearest-weight 1 --nearest-weight 0.5).",
)
def main(variant_hdf5: str, gene_table: str, gene_hdf5: str, nearest_weights: tuple[float, ...]):
    """Convert variant-level HDF5 to gene-level HDF5 using nearest-gene weights."""
    weights = np.array(nearest_weights, dtype=np.float64)
    convert_variant_to_gene_hdf5(variant_hdf5, gene_table, gene_hdf5, weights)


if __name__ == "__main__":
    # Run as a module for package-relative imports: uv run -m graphld.genesets --help
    main()