import numpy as np
import polars as pl
from scipy.sparse import csr_matrix

POSITION_SCALE = 1e9  # Scale factor for chromosome positions

def _nearest_indices_for_position(
    position: np.integer,
    gene_pos: np.ndarray,
    gene_indices: np.ndarray,
    num_nearest: int,
) -> np.ndarray:
    """Find nearest original gene indices from a sorted gene-position slice."""
    ngene = gene_pos.size
    if num_nearest > ngene:
        raise ValueError("num_nearest cannot exceed number of genes")

    insertion = np.searchsorted(gene_pos, position, side="left")
    left = max(0, insertion - num_nearest)
    right = min(ngene, insertion + num_nearest)

    if right - left < num_nearest:
        if left == 0:
            right = min(ngene, num_nearest)
        else:
            left = max(0, ngene - num_nearest)

    candidates = np.arange(left, right)
    distances = np.abs(gene_pos[candidates] - position)
    order = np.lexsort((gene_indices[candidates], distances))
    return gene_indices[candidates[order[:num_nearest]]]

def _get_nearest_genes(var_pos: np.ndarray,
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
    if num_nearest <= 0 or num_nearest > ngene:
        raise ValueError("num_nearest must be between 1 and the number of genes")

    result = np.empty((nvar, num_nearest), dtype=np.int32)
    gene_indices = np.arange(ngene, dtype=np.int32)
    var_pos = var_pos.astype(np.int64, copy=False)
    gene_pos = gene_pos.astype(np.int64, copy=False)
    for i, position in enumerate(var_pos):
        result[i] = _nearest_indices_for_position(
            position, gene_pos, gene_indices, num_nearest
        )

    return result

def _get_gene_variant_matrix(
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

    nearest = _get_nearest_genes(var_pos, gene_pos, num_nearest)
    # Row indices: each variant repeated num_nearest times
    rows = np.repeat(np.arange(nvar, dtype=np.int32), num_nearest)
    cols = nearest.ravel()
    data = np.tile(nearest_weights, nvar)
    return csr_matrix((data, (rows, cols)), shape=(nvar, ngene))

def _compute_positions(table: pl.DataFrame) -> np.ndarray:
    """Compute global positions (CHR * POSITION_SCALE + POS).
    """
    if table['POS'].max() >= POSITION_SCALE:
        raise ValueError(f"POS values must be less than POSITION_SCALE: {POSITION_SCALE}")
    return (table['POS'] + table['CHR'].cast(pl.Int64) * POSITION_SCALE).to_numpy().astype(np.int64)

def _get_chromosome_aware_nearest_genes(
    variant_table: pl.DataFrame,
    gene_table: pl.DataFrame,
    num_nearest: int,
) -> np.ndarray:
    """Find nearest gene indices, prioritizing same-chromosome genes."""
    nvar, ngene = len(variant_table), len(gene_table)
    if num_nearest <= 0 or num_nearest > ngene:
        raise ValueError("num_nearest must be between 1 and the number of genes")

    variant_chr = variant_table["CHR"].cast(pl.Int64).to_numpy()
    gene_chr = gene_table["CHR"].cast(pl.Int64).to_numpy()
    variant_positions = _compute_positions(variant_table)
    gene_positions = _compute_positions(gene_table)

    all_sorted_indices = np.argsort(gene_positions, kind="stable").astype(np.int32)
    genes_by_chr = {
        chromosome: np.flatnonzero(gene_chr == chromosome).astype(np.int32)
        for chromosome in np.unique(gene_chr)
    }
    for chromosome, indices in genes_by_chr.items():
        order = np.argsort(gene_positions[indices], kind="stable")
        genes_by_chr[chromosome] = indices[order]

    fallback_by_chr: dict[int, np.ndarray] = {}
    for chromosome in np.unique(variant_chr):
        other = all_sorted_indices[gene_chr[all_sorted_indices] != chromosome]
        fallback_by_chr[int(chromosome)] = other.astype(np.int32, copy=False)

    nearest = np.empty((nvar, num_nearest), dtype=np.int32)
    for i, (chromosome, position) in enumerate(zip(variant_chr, variant_positions)):
        same_chr_indices = genes_by_chr.get(chromosome, np.array([], dtype=np.int32))
        n_same = min(num_nearest, same_chr_indices.size)
        selected: list[int] = []

        if n_same:
            selected.extend(
                _nearest_indices_for_position(
                    position,
                    gene_positions[same_chr_indices],
                    same_chr_indices,
                    n_same,
                ).tolist()
            )

        if len(selected) < num_nearest:
            fallback_indices = fallback_by_chr[int(chromosome)]
            selected.extend(
                _nearest_indices_for_position(
                    position,
                    gene_positions[fallback_indices],
                    fallback_indices,
                    num_nearest - len(selected),
                ).tolist()
            )

        nearest[i] = np.asarray(selected, dtype=np.int32)

    return nearest

def gene_variant_matrix(variant_table: pl.DataFrame, gene_table: pl.DataFrame,
                        nearest_weights: np.ndarray) -> csr_matrix:
    """Compute gene-variant matrix from data.
    
    Args:
        variant_table: DataFrame with CHR, POS columns
        gene_table: DataFrame with CHR, POS columns (POS is midpoint for genes)
        nearest_weights: Weights for k-nearest genes
        
    Returns:
        Sparse matrix mapping genes to variants (ngenes x nvariants)
    """
    nvar, ngene = len(variant_table), len(gene_table)
    num_nearest = len(nearest_weights)
    if num_nearest <= 0 or num_nearest > ngene:
        raise ValueError("nearest_weights length must be between 1 and the number of genes")

    nearest = _get_chromosome_aware_nearest_genes(variant_table, gene_table, num_nearest)
    rows = np.repeat(np.arange(nvar, dtype=np.int32), num_nearest)
    cols = nearest.ravel()
    data = np.tile(nearest_weights, nvar)
    return csr_matrix((data, (rows, cols)), shape=(nvar, ngene))



def _is_gene_id(gene: str) -> bool:
    """Check if a gene identifier is an Ensembl ID (vs gene symbol)."""
    return 'ENSG' in gene

def convert_gene_set_to_gene_annotations(gene_sets: dict[str, list[str]],
                                         gene_table: pl.DataFrame,
                                         ) -> pl.DataFrame:
    """Convert gene sets to gene-level annotations.
    
    Returns a DataFrame with one row per gene and one column per gene set,
    plus a gene_id or gene_name column for merging.
    """
    # Determine if using gene IDs or gene symbols from first gene in first set
    first_gene = next(iter(next(iter(gene_sets.values()))))
    use_gene_id = _is_gene_id(first_gene)
    gene_key = 'gene_id' if use_gene_id else 'gene_name'

    if gene_key in gene_sets:
        raise ValueError(f"The gene key '{gene_key}' is also the name of a gene set.")

    columns: dict[str, np.ndarray] = {}
    columns[gene_key] = gene_table[gene_key].unique().to_numpy()
    gene_indices = {gene: i for i, gene in enumerate(columns[gene_key])}

    num_genes = len(gene_indices)
    for name, genes in gene_sets.items():
        columns[name] = np.zeros(num_genes, dtype=np.float64)
        for gene in genes:
            if gene in gene_indices:
                columns[name][gene_indices[gene]] = 1.0

    return pl.DataFrame(columns)


def convert_gene_to_variant_annotations(gene_annot: object,
                                        variant_table: pl.DataFrame,
                                        gene_table: pl.DataFrame,
                                        nearest_weights: np.ndarray) -> object:
    """Convert gene annotations to variant-level annotations.
    
    Can accept either:
    - GeneAnnot object with gene_sets and annot_names attributes
    - dict[str, list[str]] mapping gene set names to gene lists
    
    Args:
        gene_annot: GeneAnnot object or dict mapping set names to lists of genes (symbols or IDs)
        variant_table: Variant table DataFrame with CHR, POS, RSID columns
        gene_table: Gene table DataFrame
        nearest_weights: Weights for k-nearest genes
        
    Returns:
        VariantAnnot object with variant-level annotations, if gene_annot is GeneAnnot.
        DataFrame with variant-level annotations in LDSC format, if gene_annot is dict.
    """
    # Import at runtime to avoid circular import
    try:
        from .score_test import VariantAnnot
    except ImportError:
        from score_test import VariantAnnot

    # Handle both GeneAnnot objects and plain dicts
    if hasattr(gene_annot, 'gene_sets'):
        gene_sets = gene_annot.gene_sets
        annot_names = gene_annot.annot_names
        return_variant_annot = True
    else:
        gene_sets = gene_annot
        annot_names = None
        return_variant_annot = False

    # Determine if using gene IDs or symbols
    first_set = next(iter(gene_sets.values()))
    use_gene_id = _is_gene_id(first_set[0]) if first_set else False
    gene_key = 'gene_id' if use_gene_id else 'gene_name'

    # Get gene-variant matrix
    gv_matrix = gene_variant_matrix(variant_table, gene_table, nearest_weights)

    # Convert each gene set to variant-level annotation
    variant_annots = {}
    gene_identifiers = gene_table[gene_key].to_list()

    for set_name, genes in gene_sets.items():
        gene_set = set(genes)
        gene_values = np.array([1.0 if gene in gene_set else 0.0
                                for gene in gene_identifiers], dtype=np.float64)
        variant_values = (gv_matrix @ gene_values.reshape(-1, 1)).ravel()
        variant_annots[set_name] = variant_values

    # Create output DataFrame in LDSC format
    df_annot = pl.DataFrame({
        'CHR': variant_table['CHR'],
        'BP': variant_table['POS'],
        'RSID': variant_table['RSID'],
        'CM': pl.Series([0.0] * len(variant_table)),
        **variant_annots
    })

    if return_variant_annot:
        return VariantAnnot(df_annot, annot_names)
    else:
        return df_annot
