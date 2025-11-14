import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

# Handle imports when running either as a script or as a package
try:
    from .score_test_io import load_variant_data
except ImportError:
    from score_test_io import load_variant_data

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

def compute_variant_positions(variant_data: pl.DataFrame) -> np.ndarray:
    """Compute global positions for variants (CHR * POSITION_SCALE + POS)."""
    return (variant_data['POS'] + variant_data['CHR'] * POSITION_SCALE).to_numpy().astype(np.int64)

def compute_gene_positions(gene_table: pl.DataFrame) -> np.ndarray:
    """Compute global positions for genes (CHR * POSITION_SCALE + midpoint)."""
    return (gene_table['midpoint'] + gene_table['CHR'].cast(pl.Int64) * POSITION_SCALE).to_numpy().astype(np.int64)

def compute_gene_variant_matrix_from_data(variant_data: pl.DataFrame, gene_table: pl.DataFrame, 
                                         nearest_weights: np.ndarray):
    """Compute gene-variant matrix from data.
    
    Args:
        variant_data: DataFrame with CHR, POS columns
        gene_table: DataFrame with CHR, start, end, midpoint columns
        nearest_weights: Weights for k-nearest genes
        
    Returns:
        Sparse matrix mapping genes to variants (ngenes x nvariants)
    """
    variant_positions = compute_variant_positions(variant_data)
    gene_positions = compute_gene_positions(gene_table)
    return get_gene_variant_matrix(variant_positions, gene_positions, nearest_weights)


def load_gene_sets_from_gmt(gene_annot_dir: str) -> dict[str, list[str]]:
    """Load gene sets from GMT files in a directory.
    
    GMT format: set_name<tab>description<tab>gene1<tab>gene2<tab>...
    
    Returns:
        Dictionary mapping set names to lists of genes
    """
    import glob
    from pathlib import Path
    
    gmt_files = glob.glob(str(Path(gene_annot_dir) / '*.gmt'))
    if not gmt_files:
        raise FileNotFoundError(f"No .gmt files found in {gene_annot_dir}")
    
    gene_sets = {}
    for gmt_file in gmt_files:
        with open(gmt_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    set_name = parts[0]
                    genes = parts[2:]  # Skip description
                    gene_sets[set_name] = genes
    
    return gene_sets


def load_gene_table(gene_table_path: str, chromosomes: list[int] | None = None) -> pl.DataFrame:
    """Load gene table and optionally filter to specific chromosomes.
    
    Args:
        gene_table_path: Path to gene table TSV
        chromosomes: Optional list of chromosome numbers to filter to
        
    Returns:
        Gene table DataFrame
    """
    schema = {
        'gene_id': pl.Utf8,
        'gene_id_version': pl.Utf8,
        'gene_name': pl.Utf8,
        'start': pl.Int64,
        'end': pl.Int64,
        'CHR': pl.Utf8,
    }
    gene_table = pl.scan_csv(gene_table_path, schema=schema, separator='\t', has_header=True) \
        .filter(pl.col('CHR').is_in([str(i) for i in range(1,23)])) \
        .filter(pl.col('gene_id').is_not_null()) \
        .with_columns(pl.col('gene_name').fill_null('NA')) \
        .with_columns(((pl.col('start') + pl.col('end')) / 2).alias('midpoint')) \
        .sort(pl.col('CHR').cast(pl.Int64), 'midpoint') \
        .collect()
    
    if chromosomes:
        gene_table = gene_table.filter(pl.col('CHR').cast(pl.Int64).is_in(chromosomes))
    
    return gene_table


def _is_gene_id(gene: str) -> bool:
    """Check if a gene identifier is an Ensembl ID (vs gene symbol)."""
    return 'ENSG' in gene


def convert_gene_sets_to_variant_annotations(gene_sets: dict[str, list[str]],
                                             variant_data: pl.DataFrame,
                                             gene_table: pl.DataFrame,
                                             nearest_weights: np.ndarray) -> pl.DataFrame:
    """Convert gene sets to variant-level annotations.
    
    Args:
        gene_sets: Dictionary mapping set names to lists of genes (symbols or IDs)
        variant_data: Variant data DataFrame
        gene_table: Gene table DataFrame
        nearest_weights: Weights for k-nearest genes
        
    Returns:
        DataFrame with variant-level annotations in LDSC format
    """
    # Determine if using gene IDs or symbols
    first_set = next(iter(gene_sets.values()))
    use_gene_id = _is_gene_id(first_set[0]) if first_set else False
    gene_key = 'gene_id' if use_gene_id else 'gene_name'
    
    # Get gene-variant matrix
    gene_variant_matrix = compute_gene_variant_matrix_from_data(variant_data, gene_table, nearest_weights)
    
    # Convert each gene set to variant-level annotation
    variant_annots = {}
    gene_identifiers = gene_table[gene_key].to_list()
    
    for set_name, genes in gene_sets.items():
        gene_set = set(genes)
        gene_values = np.array([1.0 if gene in gene_set else 0.0 
                                for gene in gene_identifiers], dtype=np.float64)
        variant_values = (gene_variant_matrix @ gene_values.reshape(-1, 1)).ravel()
        variant_annots[set_name] = variant_values
    
    # Create output DataFrame in LDSC format
    return pl.DataFrame({
        'CHR': variant_data['CHR'],
        'BP': variant_data['POS'],
        'SNP': variant_data['RSID'],
        'CM': pl.Series([0.0] * len(variant_data)),
        **variant_annots
    })


