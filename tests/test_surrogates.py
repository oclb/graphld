import numpy as np
import polars as pl
import h5py
import pytest
import tempfile

from graphld.surrogates import get_surrogate_markers
from graphld.heritability import GraphREML, MethodOptions, ModelOptions, run_graphREML
from graphld.io import (
    load_ldgm,
    merge_snplists,
    partition_variants,
    read_ldgm_metadata,
)


def _right_join_sumstats_annotations(
    sumstats: pl.DataFrame, annotations: pl.DataFrame
) -> pl.DataFrame:
    merged = sumstats.join(annotations, on=["SNP"], how="right")
    if "CHR_right" in merged.columns and "POS_right" in merged.columns:
        merged = (
            merged.drop("CHR")
            .drop("POS")
            .rename({"CHR_right": "CHR", "POS_right": "POS"})
        )
    return merged.unique(subset=["SNP"], keep="first")


def _active_subset_inputs(metadata_path, create_sumstats, create_annotations):
    sumstats = create_sumstats(ldgm_metadata_path=str(metadata_path), populations="EUR")
    annotations = create_annotations(metadata_path, populations="EUR")
    metadata = read_ldgm_metadata(str(metadata_path), populations="EUR")
    block = metadata.row(0, named=True)

    nonmissing = sumstats.filter(pl.col('POS') % 7 != 0)
    active_annotations = annotations.filter(pl.col('POS') % 13 != 0)
    annot_df = _right_join_sumstats_annotations(nonmissing, active_annotations)
    annot_block = partition_variants(metadata, annot_df)[0]

    assert annot_block.get_column("Z").null_count() > 0
    return metadata, block, nonmissing, annot_block


@pytest.mark.parametrize("run_serial", [True, False], ids=["serial", "parallel"])
def test_get_surrogate_markers(metadata_path, create_sumstats, run_serial):
    """Surrogate maps are keyed by full LDGM row coordinates."""
    sumstats = create_sumstats(ldgm_metadata_path=metadata_path, populations="EUR")
    metadata = read_ldgm_metadata(str(metadata_path), populations="EUR")

    # Case 1: ~10% missing (keep POS % 10 != 0)
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        nonmissing_10 = sumstats.filter(pl.col('POS') % 10 != 0)
        h5_path_10 = get_surrogate_markers(
            metadata_path,
            nonmissing_10,
            population="EUR",
            run_serial=run_serial,
            num_processes=None if run_serial else 2,
            output_path=tmp.name,
        )

        nonmissing_blocks = partition_variants(metadata, nonmissing_10)
        with h5py.File(h5_path_10, 'r') as h5:
            assert h5.attrs["graphld_surrogate_format"] == "1"
            assert h5.attrs["coordinate_system"] == "ldgm_full_index"

            for row, nonmissing_block in zip(
                metadata.iter_rows(named=True), nonmissing_blocks, strict=False
            ):
                ldgm = load_ldgm(
                    str(metadata_path.parent / row["name"]),
                    population=row["population"],
                )
                merged_ldgm, _ = merge_snplists(
                    ldgm,
                    nonmissing_block,
                    pos_col='POS',
                    ref_allele_col='REF',
                    alt_allele_col='ALT',
                    modify_in_place=False,
                )
                candidate_indices = merged_ldgm._which_indices
                assert candidate_indices is not None

                mapping = h5[row["name"]][:]
                expected_identity = np.arange(ldgm.shape[0], dtype=mapping.dtype)
                missing_indices = np.setdiff1d(expected_identity, candidate_indices)

                assert mapping.shape == (ldgm.shape[0],)
                np.testing.assert_array_equal(
                    mapping[candidate_indices], candidate_indices
                )
                assert np.all(np.isin(mapping[missing_indices], candidate_indices))
                assert np.all(mapping[missing_indices] != missing_indices)


def test_get_surrogate_markers_writes_identity_for_zero_candidate_blocks(
    metadata_path, monkeypatch
):
    """Blocks with no candidate variants still get contract-valid datasets."""
    metadata = read_ldgm_metadata(str(metadata_path), populations="EUR")
    empty_sumstats = pl.DataFrame(
        schema={
            "SNP": pl.Utf8,
            "CHR": pl.Int64,
            "POS": pl.Int64,
            "REF": pl.Utf8,
            "ALT": pl.Utf8,
        }
    )

    def fail_surrogate_search(*args, **kwargs):
        raise AssertionError("surrogate search should not run without candidates")

    monkeypatch.setattr("graphld.surrogates._surrogate_marker", fail_surrogate_search)

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        h5_path = get_surrogate_markers(
            metadata_path,
            empty_sumstats,
            population="EUR",
            run_serial=True,
            output_path=tmp.name,
        )

        with h5py.File(h5_path, "r") as h5:
            assert h5.attrs["graphld_surrogate_format"] == "1"
            assert h5.attrs["coordinate_system"] == "ldgm_full_index"
            assert set(h5.keys()) == set(metadata.get_column("name").to_list())

            for row in metadata.iter_rows(named=True):
                ldgm = load_ldgm(
                    str(metadata_path.parent / row["name"]),
                    population=row["population"],
                )
                mapping = h5[row["name"]][:]
                np.testing.assert_array_equal(
                    mapping,
                    np.arange(ldgm.shape[0], dtype=mapping.dtype),
                )


def test_unversioned_surrogate_map_is_rejected(metadata_path):
    metadata = read_ldgm_metadata(str(metadata_path), populations="EUR")
    block_name = metadata.row(0, named=True)["name"]

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        with h5py.File(tmp.name, "w") as h5:
            h5.create_dataset(block_name, data=np.arange(5, dtype=np.int32))

        with pytest.raises(ValueError, match="Regenerate it with `graphld surrogates`"):
            GraphREML._load_block_surrogate_map(tmp.name, block_name)


def test_get_surrogate_markers_rejects_unversioned_existing_output(
    metadata_path, create_sumstats
):
    sumstats = create_sumstats(ldgm_metadata_path=metadata_path, populations="EUR")
    metadata = read_ldgm_metadata(str(metadata_path), populations="EUR")
    block_name = metadata.row(0, named=True)["name"]
    nonmissing = sumstats.filter(pl.col('POS') % 10 != 0)

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        with h5py.File(tmp.name, "w") as h5:
            h5.create_dataset(block_name, data=np.arange(5, dtype=np.int32))

        with pytest.raises(ValueError, match="Regenerate it with `graphld surrogates`"):
            get_surrogate_markers(
                metadata_path,
                nonmissing,
                population="EUR",
                run_serial=True,
                output_path=tmp.name,
            )


def test_wrong_length_surrogate_map_is_rejected(
    metadata_path, create_sumstats, create_annotations
):
    _, block, _, annot_block = _active_subset_inputs(
        metadata_path, create_sumstats, create_annotations
    )
    ldgm = load_ldgm(
        str(metadata_path.parent / block["name"]),
        population=block["population"],
    )

    with pytest.raises(ValueError, match="Surrogate map length 5"):
        GraphREML._initialize_block_zscores(
            ldgm,
            annot_block,
            ["base"],
            match_by_position=False,
            surrogate_map=np.arange(5, dtype=np.int32),
        )


def test_precomputed_surrogates_translate_for_active_subset(
    metadata_path, create_sumstats, create_annotations
):
    _, block, nonmissing, annot_block = _active_subset_inputs(
        metadata_path, create_sumstats, create_annotations
    )

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        h5_path = get_surrogate_markers(
            metadata_path,
            nonmissing,
            population="EUR",
            run_serial=True,
            output_path=tmp.name,
        )
        surrogate_map = GraphREML._load_block_surrogate_map(
            str(h5_path), block["name"]
        )

        ldgm_probe = load_ldgm(
            str(metadata_path.parent / block["name"]),
            population=block["population"],
        )
        ldgm_probe, _ = merge_snplists(
            ldgm_probe,
            annot_block,
            add_allelic_cols=["Z"],
            add_cols=["base"],
            match_by_position=False,
            pos_col="POS",
            ref_allele_col="REF",
            alt_allele_col="ALT",
            modify_in_place=True,
        )
        active_to_full = ldgm_probe._which_indices
        assert active_to_full is not None
        assert not np.array_equal(active_to_full, np.arange(ldgm_probe.shape[0]))
        full_to_active = {
            int(full_idx): int(active_idx)
            for active_idx, full_idx in enumerate(active_to_full)
        }
        probe_info = ldgm_probe.variant_info.with_row_index(name="vi_row_nr")
        nonmissing_rows = (
            probe_info.filter(pl.col("Z").is_not_null())
            .group_by("index")
            .first()
        )
        nonmissing_by_index = {
            row["index"]: row for row in nonmissing_rows.to_dicts()
        }
        expected_row = None
        for row in probe_info.filter(pl.col("Z").is_null()).to_dicts():
            if row["index"] in nonmissing_by_index:
                continue
            full_idx = int(active_to_full[row["index"]])
            surrogate_active_idx = full_to_active.get(int(surrogate_map[full_idx]))
            if surrogate_active_idx in nonmissing_by_index:
                expected_row = (row, nonmissing_by_index[surrogate_active_idx])
                break
        assert expected_row is not None

        ldgm_precomputed = load_ldgm(
            str(metadata_path.parent / block["name"]),
            population=block["population"],
        )
        ldgm_precomputed, pz_precomputed = GraphREML._initialize_block_zscores(
            ldgm_precomputed,
            annot_block,
            ["base"],
            match_by_position=False,
            surrogate_map=surrogate_map,
        )

        assert ldgm_precomputed._which_indices is not None
        assert not np.array_equal(
            ldgm_precomputed._which_indices,
            np.arange(ldgm_precomputed.shape[0]),
        )
        missing_row, surrogate_row = expected_row
        initialized_row = ldgm_precomputed.variant_info.row(
            missing_row["vi_row_nr"], named=True
        )
        assert initialized_row["index"] == surrogate_row["index"]
        assert initialized_row["Z"] == surrogate_row["Z"]
        assert np.all(np.isfinite(pz_precomputed))


def test_precomputed_surrogates_fall_back_when_target_not_active(
    metadata_path, create_sumstats, create_annotations, monkeypatch
):
    _, block, _, annot_block = _active_subset_inputs(
        metadata_path, create_sumstats, create_annotations
    )
    ldgm_probe = load_ldgm(
        str(metadata_path.parent / block["name"]),
        population=block["population"],
    )
    ldgm_probe, _ = merge_snplists(
        ldgm_probe,
        annot_block,
        add_allelic_cols=["Z"],
        add_cols=["base"],
        match_by_position=False,
        pos_col="POS",
        ref_allele_col="REF",
        alt_allele_col="ALT",
        modify_in_place=True,
    )
    active_to_full = ldgm_probe._which_indices
    assert active_to_full is not None
    probe_info = ldgm_probe.variant_info.with_row_index(name="vi_row_nr")
    nonmissing_indices = set(
        probe_info.filter(pl.col("Z").is_not_null()).get_column("index").to_list()
    )
    missing_row = next(
        row
        for row in probe_info.filter(pl.col("Z").is_null()).to_dicts()
        if row["index"] not in nonmissing_indices
    )
    inactive_full_indices = np.setdiff1d(
        np.arange(ldgm_probe._matrix.shape[0]), active_to_full
    )
    assert len(inactive_full_indices) > 0

    surrogate_map = np.arange(ldgm_probe._matrix.shape[0], dtype=np.int32)
    surrogate_map[int(active_to_full[missing_row["index"]])] = int(
        inactive_full_indices[0]
    )

    fallback_rows = {}

    def fake_surrogate_marker(ldgm, missing_index, candidates):
        surrogate_row = candidates.to_dicts()[0]
        fallback_rows[missing_index] = surrogate_row
        return surrogate_row

    monkeypatch.setattr(
        "graphld.heritability._surrogate_marker", fake_surrogate_marker
    )

    ldgm_precomputed = load_ldgm(
        str(metadata_path.parent / block["name"]),
        population=block["population"],
    )
    ldgm_precomputed, _ = GraphREML._initialize_block_zscores(
        ldgm_precomputed,
        annot_block,
        ["base"],
        match_by_position=False,
        surrogate_map=surrogate_map,
    )

    assert missing_row["index"] in fallback_rows
    precomputed_row = ldgm_precomputed.variant_info.row(
        missing_row["vi_row_nr"], named=True
    )
    assert precomputed_row["index"] == fallback_rows[missing_row["index"]]["index"]
    assert precomputed_row["Z"] == fallback_rows[missing_row["index"]]["Z"]


def test_surrogates_integration_with_graphreml(metadata_path, create_sumstats, create_annotations):
    """End-to-end: create HDF5 surrogates and ensure GraphREML can consume them via workers.

    This ensures that `MethodOptions.surrogate_markers_path` is honored and mappings are
    loaded inside workers (not in prepare_block_data).
    """
    # Prepare input data
    sumstats = create_sumstats(ldgm_metadata_path=str(metadata_path), populations="EUR")
    annotations = create_annotations(metadata_path, populations="EUR")

    # Make some variants missing to force surrogate lookup (pattern A)
    nonmissing = sumstats.filter(pl.col('POS') % 7 != 0)

    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        # Generate surrogate mapping file
        h5_path = get_surrogate_markers(
            metadata_path,
            nonmissing,
            population="EUR",
            run_serial=True,
            output_path=tmp.name,
        )

        # Minimal model/method options
        model = ModelOptions(
            params=np.zeros((1, 1)),
            sample_size=1000,
        )
        method = MethodOptions(
            match_by_position=False,
            run_serial=True,
            num_iterations=1,
            use_surrogate_markers=True,
            surrogate_markers_path=str(h5_path),
        )

        # Build a different missingness pattern (pattern B): drop every 8th and 12th
        sumstats_with_missing = sumstats.filter((pl.col('POS') % 8 != 0) & (pl.col('POS') % 12 != 0))

        # Run GraphREML, expecting it to load per-block surrogate mappings in workers
        result = run_graphREML(
            model_options=model,
            method_options=method,
            summary_stats=sumstats_with_missing,
            annotation_data=annotations,
            ldgm_metadata_path=metadata_path,
            populations="EUR",
        )

        assert result is not None
        # Check heritability is finite
        h2 = result.get('heritability')
        assert h2 is not None
        if isinstance(h2, np.ndarray):
            assert np.all(np.isfinite(h2))
        else:
            assert np.isfinite(h2)
