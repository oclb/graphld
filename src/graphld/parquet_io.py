"""Functions for reading parquet summary statistics files.

This module supports the parquet format produced by linear_dag's GWAS pipeline,
which stores summary statistics with columns like {trait}_BETA and {trait}_SE.
Multiple traits can be stored in a single file.
"""

from pathlib import Path
from typing import Optional, Union

import polars as pl


_SNP_ALIASES = ('site_ids', 'SNP', 'rsid')
_CHR_ALIASES = ('chrom', 'CHR')
_POS_ALIASES = ('position', 'POS', 'BP')
_REF_ALIASES = ('ref', 'REF', 'A2')
_ALT_ALIASES = ('alt', 'ALT', 'A1')
_N_ALIASES = ('N', 'n')


def get_parquet_traits(file: Union[str, Path]) -> list[str]:
    """Get list of trait names from a parquet summary statistics file.

    A trait name is recognized if EITHER its `_BETA` or `_SE` column is present;
    this is a discovery helper and does not guarantee the trait is complete.
    `read_parquet_sumstats` is responsible for raising when a requested trait
    lacks one of its halves.

    Args:
        file: Path to parquet file

    Returns:
        List of trait names found in the file
    """
    schema = pl.read_parquet_schema(file)
    traits = set()
    for col in schema:
        if col.endswith('_BETA'):
            traits.add(col[:-5])  # Remove '_BETA' suffix
        elif col.endswith('_SE'):
            traits.add(col[:-3])  # Remove '_SE' suffix
    return sorted(traits)


def read_parquet_sumstats(
    file: Union[str, Path],
    trait: Optional[str] = None,
    maximum_missingness: float = 1.0,
) -> pl.DataFrame:
    """Read parquet summary statistics file and return a single-trait DataFrame.

    The parquet format stores per-trait columns as `{trait}_BETA` and `{trait}_SE`.
    This function extracts a single trait, computes `Z = BETA / SE`, and maps
    recognized variant-info columns to the standard names used elsewhere in
    GraphLD.

    Args:
        file: Path to parquet file.
        trait: Name of trait to extract. If None, uses the first trait found
            (after sorting trait names alphabetically).
        maximum_missingness: Maximum fraction of missing samples allowed, applied
            as a per-row filter against the maximum observed `N`. Only effective
            when an `N`/`n` column is present in the input file; otherwise
            silently a no-op.

    Returns:
        Polars DataFrame containing:
            * `Z` (always): computed as BETA / SE, restricted to finite values.
            * `SNP` and/or position columns (`CHR`, `POS`): at least one of `SNP`
            or `POS` is guaranteed to be present. Callers must select a
            `merge_snplists` mode matching what is present: `SNP` enables
            SNP-ID merge, `POS` enables position-based merge. A file containing
            only `POS` is read successfully here but will fail downstream if
            the caller fixes `match_by_position=False`. `CHR` is present only
            if a chromosome column was mapped.
            * `REF`, `ALT`: present only if matching allele columns were mapped.
            * `N`: present if an `N`/`n` column was mapped; added as a null
            Float64 column if not.

        Recognized input column aliases (case-sensitive): SNP via
        ``('site_ids', 'SNP', 'rsid')``, chrom via ``('chrom', 'CHR')``,
        position via ``('position', 'POS', 'BP')``, ref via
        ``('ref', 'REF', 'A2')``, alt via ``('alt', 'ALT', 'A1')``, N via
        ``('N', 'n')``.

    Raises:
        ValueError: If the file contains no `_BETA`/`_SE` columns; if the
            requested trait is not in the discovered trait list; if the
            requested trait is missing its `_BETA` or `_SE` half; or if the
            file has neither a SNP-identifier alias nor a position alias
            (in which case no downstream merge mode could succeed).
    """
    schema = pl.read_parquet_schema(file)

    available_traits = get_parquet_traits(file)
    if not available_traits:
        raise ValueError(
            f"No traits found in parquet file {file}. "
            "Expected columns ending in _BETA and _SE."
        )

    if trait is None:
        trait = available_traits[0]
    elif trait not in available_traits:
        raise ValueError(
            f"Trait '{trait}' not found in parquet file. "
            f"Available traits: {available_traits}"
        )

    beta_col = f"{trait}_BETA"
    se_col = f"{trait}_SE"
    missing_halves = [c for c in (beta_col, se_col) if c not in schema]
    if missing_halves:
        raise ValueError(
            f"Trait '{trait}' is incomplete in parquet file {file}: "
            f"missing column(s) {missing_halves}. "
            f"Expected both {beta_col} and {se_col}."
        )

    variant_cols: list[str] = []
    col_renames: dict[str, str] = {}

    def _map_first(aliases: tuple[str, ...], canonical: str) -> bool:
        for alias in aliases:
            if alias in schema:
                variant_cols.append(alias)
                if alias != canonical:
                    col_renames[alias] = canonical
                return True
        return False

    has_snp = _map_first(_SNP_ALIASES, 'SNP')
    _map_first(_CHR_ALIASES, 'CHR')
    has_pos = _map_first(_POS_ALIASES, 'POS')
    _map_first(_REF_ALIASES, 'REF')
    _map_first(_ALT_ALIASES, 'ALT')
    has_n = _map_first(_N_ALIASES, 'N')

    if not (has_snp or has_pos):
        raise ValueError(
            f"Parquet file {file} has no usable variant identifier. "
            f"Expected one of {list(_SNP_ALIASES)} (for SNP-ID merge) or "
            f"{list(_POS_ALIASES)} (for position-based merge). "
            f"Found columns: {sorted(schema)}"
        )

    columns_to_read = variant_cols + [beta_col, se_col]
    df = pl.read_parquet(file, columns=columns_to_read)

    if col_renames:
        df = df.rename(col_renames)

    df = df.with_columns(
        (pl.col(beta_col) / pl.col(se_col)).cast(pl.Float64).alias('Z')
    )
    df = df.drop([beta_col, se_col])

    df = df.filter(pl.col('Z').is_finite())

    if has_n and maximum_missingness < 1.0:
        max_n = df['N'].max()
        min_n = (1 - maximum_missingness) * max_n
        df = df.filter(pl.col('N') >= min_n)

    if 'N' not in df.columns:
        df = df.with_columns(pl.lit(None).cast(pl.Float64).alias('N'))

    return df


def read_parquet_sumstats_multi(
    file: Union[str, Path],
    traits: Optional[list[str]] = None,
    maximum_missingness: float = 1.0,
) -> dict[str, pl.DataFrame]:
    """Read multiple traits from a parquet summary statistics file.

    Args:
        file: Path to parquet file.
        traits: List of trait names to extract. If None, extracts every trait
            discovered by `get_parquet_traits`. Note that `get_parquet_traits`
            uses union semantics across `_BETA` and `_SE`; if an incomplete
            trait is included (explicitly or via the default), the per-trait
            `read_parquet_sumstats` call raises a `ValueError` naming the
            missing half.
        maximum_missingness: Forwarded to `read_parquet_sumstats`.

    Returns:
        Dictionary mapping trait names to DataFrames produced by
        `read_parquet_sumstats`.

    Raises:
        ValueError: If any explicitly requested trait is not in the discovered
            trait list, or if any selected trait is incomplete or the file
            lacks a usable variant identifier (propagated from
            `read_parquet_sumstats`).
    """
    available_traits = get_parquet_traits(file)

    if traits is None:
        traits = available_traits
    else:
        missing = set(traits) - set(available_traits)
        if missing:
            raise ValueError(
                f"Traits not found in parquet file: {missing}. "
                f"Available traits: {available_traits}"
            )

    result = {}
    for trait in traits:
        result[trait] = read_parquet_sumstats(
            file, trait=trait, maximum_missingness=maximum_missingness
        )

    return result
