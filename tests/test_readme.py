from typing import List

import numpy as np
import polars as pl

import graphld as gld
from graphld.precision import PrecisionOperator


def test_io_merging_snippet():
    """
    Tests the code snippet from the README.md file.
    """
    # Load metadata
    ldgm_metadata: pl.DataFrame = gld.read_ldgm_metadata(
        "data/test/metadata.csv", populations=["EUR"]
    )

    # Load summary statistics
    sumstats: pl.DataFrame = gld.read_ldsc_sumstats("data/test/example.sumstats")
    partitioned_sumstats: List[pl.DataFrame] = gld.partition_variants(
        ldgm_metadata, sumstats
    )

    # Load LDGMs and merge with summary statistics
    merged_ldgms = []
    for row, df in zip(ldgm_metadata.iter_rows(named=True), partitioned_sumstats):
        ldgm: PrecisionOperator = gld.load_ldgm(
            filepath="data/test/" + row["name"],
            snplist_path="data/test/" + row["snplistName"],
        )
        ldgm, _ = gld.merge_snplists(ldgm, df)
        merged_ldgms.append(ldgm)

    # Perform a solve operation to ensure it works
    for ldgm in merged_ldgms:
        # Retain only the first SNP that matches each row/col of the LDGM
        z_scores = ldgm.variant_info.group_by('index', maintain_order=True) \
            .agg(pl.col('Z').first()).select('Z').to_numpy()
        solution = ldgm.solve(z_scores)
        assert solution is not None
        assert solution.shape[0] == ldgm.shape[0]


def test_heritability_snippet():
    """
    Tests the heritability estimation snippet from the README.md file.
    """
    sumstats: pl.DataFrame = gld.read_ldsc_sumstats("data/test/example.sumstats")
    annotations: pl.DataFrame = gld.load_annotations(
        "data/test/annot", chromosome=1
    )

    default_model_options = gld.ModelOptions()
    default_method_options = gld.MethodOptions()

    reml_results: dict = gld.run_graphREML(
        model_options=default_model_options,
        method_options=default_method_options,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path="data/test/metadata.csv",
        populations="EUR",
    )
    assert isinstance(reml_results, dict)
    assert "heritability" in reml_results


def test_ldgm_loading_snippet() -> None:
    """Test the LDGM loading snippet from README.md."""
    ldgm: gld.PrecisionOperator = gld.load_ldgm(
        filepath="data/test/1kg_chr1_16103_2888443.EAS.edgelist",
        snplist_path="data/test/1kg_chr1_16103_2888443.snplist",
    )

    vector = np.random.randn(ldgm.shape[0])
    precision_times_vector = ldgm @ vector
    correlation_times_vector = ldgm.solve(precision_times_vector)
    assert np.allclose(correlation_times_vector, vector)


def test_clumping_snippet() -> None:
    """Test the LD clumping snippet from README.md."""
    sumstats_dataframe_with_z_scores = gld.read_ldsc_sumstats(
        "data/test/example.sumstats"
    )
    sumstats_clumped: pl.DataFrame = gld.run_clump(
        sumstats=sumstats_dataframe_with_z_scores,
        z_col='Z',
        ldgm_metadata_path="data/test/metadata.csv",
        populations='EUR',
        rsq_threshold=0.1,
        chisq_threshold=30.0,
    ).filter(pl.col('is_index'))
    assert isinstance(sumstats_clumped, pl.DataFrame)
