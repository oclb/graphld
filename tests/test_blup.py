"""Test BLUP functionality."""

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import polars as pl

from graphld import BLUP, read_ldgm_metadata


def test_blup():
    """Test BLUP on the packaged LDGM fixture."""
    def create_sumstats(
        ldgm_metadata_path: str,
        populations: Optional[Union[str, List[str]]]
    ) -> pl.DataFrame:
        metadata: pl.DataFrame = read_ldgm_metadata(
            ldgm_metadata_path,
            populations=populations
        )
        sumstats = {
            'CHR': [],
            'POS': [],
            'REF': [],
            'ALT': [],
            'Z': []
        }
        for row in metadata.iter_rows(named=True):
            snplist_dir = os.path.dirname(ldgm_metadata_path)
            snplist_path = os.path.join(snplist_dir, row['snplistName'])
            snplist = pl.read_csv(snplist_path, separator=',', has_header=True)
            chromosome = int(row['chrom'])
            sumstats['CHR'].extend([chromosome] * len(snplist))
            sumstats['POS'].extend(snplist['position'].to_list())
            sumstats['REF'].extend(snplist['anc_alleles'].to_list())
            sumstats['ALT'].extend(snplist['deriv_alleles'].to_list())
            sumstats['Z'].extend(list(np.random.randn(len(snplist))))

        return pl.DataFrame(sumstats)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Create annotations from metadata
    metadata_path = "data/test/metadata.csv"
    sumstats = create_sumstats(metadata_path, populations="EUR")

    assert isinstance(sumstats, pl.DataFrame)
    print(sumstats.head())

    heritability = 0.01
    sample_size = 10_000
    blup = BLUP.compute_blup(
        metadata_path,
        sumstats,
        heritability,
        sample_size,
        populations="EUR",
        run_in_serial=False,
        match_by_position=True
    )

    assert(len(blup) == np.sum(sumstats.select(pl.col('POS').is_first_distinct()).to_numpy()))


def test_blup_heritability_scales_to_matched_effect_trace(tmp_path: Path):
    """Total heritability is distributed over matched LDGM effect indices."""
    metadata_path = tmp_path / "metadata.csv"
    edgelist_path = tmp_path / "tiny.EUR.edgelist"
    snplist_path = tmp_path / "tiny.snplist"

    edgelist_path.write_text("0,0,1.0\n1,1,1.0\n")
    pl.DataFrame({
        "index": [0, 1],
        "anc_alleles": ["A", "C"],
        "deriv_alleles": ["G", "T"],
        "EUR": [0.2, 0.3],
        "site_ids": ["rs1", "rs2"],
        "position": [10, 20],
        "swap": ["+", "+"],
    }).write_csv(snplist_path)
    pl.DataFrame({
        "chrom": [1],
        "chromStart": [1],
        "chromEnd": [100],
        "name": [edgelist_path.name],
        "snplistName": [snplist_path.name],
        "population": ["EUR"],
        "numVariants": [2],
        "numIndices": [2],
        "numEntries": [2],
        "info": [""],
    }).write_csv(metadata_path)

    sumstats = pl.DataFrame({
        "SNP": ["rs1", "rs2", "unmatched"],
        "CHR": [1, 1, 1],
        "POS": [10, 20, 30],
        "REF": ["A", "C", "G"],
        "ALT": ["G", "T", "A"],
        "Z": [2.0, -1.0, 5.0],
    })

    heritability = 0.2
    sample_size = 100.0
    result = BLUP.compute_blup(
        str(metadata_path),
        sumstats,
        heritability=heritability,
        sample_size=sample_size,
        populations="EUR",
        run_in_serial=True,
    )

    per_effect_variance = heritability / 2
    coefficient = np.sqrt(sample_size) * per_effect_variance / (
        1 + sample_size * per_effect_variance
    )
    expected_weights = np.array([2.0, -1.0, 0.0]) * coefficient
    np.testing.assert_allclose(result["weight"].to_numpy(), expected_weights)
