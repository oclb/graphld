"""Tests for heritability estimation."""

import os
import tempfile
import warnings

import h5py
import numpy as np
import polars as pl
import pytest

from graphld.heritability import (
    FLAGS,
    GraphREML,
    MethodOptions,
    ModelOptions,
    _block_max_chisq,
    _get_softmax_link_function,
    partition_variants,
    run_graphREML,
)
from graphld.io import read_ldgm_metadata
from graphld.multiprocessing_template import SharedData


def test_run_graphREML(metadata_path, create_annotations, create_sumstats):
    """Test heritability estimation with simulated data."""
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')

    # Run GraphREML
    model = ModelOptions(params=np.zeros((1,1)),
                sample_size=1000
    )
    method = MethodOptions(
        match_by_position=True
    )
    result = run_graphREML(
        model_options=model,
        method_options=method,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=metadata_path,
        populations='EUR',
    )

    assert result is not None


def test_softmax_link_function():
    """Test softmax link function against MATLAB output."""
    link_fn, link_fn_grad, link_fn_hess = _get_softmax_link_function(denominator=10)

    # Test case from MATLAB: linkFnGrad((1:3),(2:4)')
    annot = np.array([[1, 2, 3]])
    theta = np.array([[0.2], [0.3], [0.4]])

    # Test case from MATLAB: linkFn((1:3),(2:4)')
    val = link_fn(annot, theta)
    assert np.allclose(val, 0.2127, rtol=1e-3), f"Expected 0.2127, got {val}"

    grad = link_fn_grad(annot, theta)
    assert np.allclose(grad, [0.0881,0.1762,0.2642], rtol=1e-3)

    hess = link_fn_hess(annot, theta)
    assert np.allclose(hess, [0.0105, 0.0420, 0.0945], rtol=1e-3)

    annot = np.array([[1,2,3],[1,1,1]])
    val = link_fn_grad(np.array(1), annot @ theta)
    print(val)
    expected = [[0.0881], [0.0711]]
    assert np.allclose(val, expected, rtol=1e-3), f"Expected {expected}, got {val}"


def test_softmax_link_grad_large_positive_has_no_invalid_warning():
    """Large positive rows should not evaluate the discarded negative branch."""
    _, link_fn_grad, _ = _get_softmax_link_function(denominator=1)
    annot = np.ones((2, 1))
    theta = np.array([[1000.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        grad = link_fn_grad(annot, theta)

    np.testing.assert_allclose(grad, annot)


def test_max_z_squared_threshold_discards_high_chisq_blocks(
    metadata_path, create_sumstats
):
    """Regression test that max_chisq_threshold is applied per block."""
    metadata = read_ldgm_metadata(metadata_path, populations='EUR')
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    original_blocks = partition_variants(metadata, sumstats)
    first_pos = original_blocks[0].select('POS').head(1).item()
    sumstats = sumstats.with_columns(
        pl.when(pl.col('POS') == first_pos)
        .then(10.0)
        .otherwise(pl.col('Z'))
        .alias('Z')
    )

    block_data = GraphREML.prepare_block_data(
        metadata,
        sumstats=sumstats,
        method=MethodOptions(max_chisq_threshold=50.0),
    )

    assert len(block_data[0]['sumstats']) == 0
    assert len(block_data[1]['sumstats']) == len(original_blocks[1])
    assert block_data[0]['variant_offset'] == 0
    assert block_data[1]['variant_offset'] == 0


def test_max_z_squared_threshold_errors_when_all_blocks_discarded(
    metadata_path, create_sumstats
):
    """All-discarded runs fail clearly instead of later concatenation errors."""
    metadata = read_ldgm_metadata(metadata_path, populations='EUR')
    sumstats = create_sumstats(str(metadata_path), 'EUR')

    with pytest.raises(ValueError, match='No LD blocks remain'):
        GraphREML.prepare_block_data(
            metadata,
            sumstats=sumstats,
            method=MethodOptions(max_chisq_threshold=0.0),
        )


def test_max_z_squared_threshold_ignores_blocks_with_no_finite_z(
    metadata_path, create_sumstats
):
    """Blocks with no finite Z do not emit nanmax warnings or fail filtering."""
    metadata = read_ldgm_metadata(metadata_path, populations='EUR')
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    first_block = partition_variants(metadata, sumstats)[0]
    positions = first_block.get_column('POS').to_list()
    sumstats = sumstats.with_columns(
        pl.when(pl.col('POS').is_in(positions))
        .then(float('nan'))
        .otherwise(pl.col('Z'))
        .alias('Z')
    )

    block_data = GraphREML.prepare_block_data(
        metadata,
        sumstats=sumstats,
        method=MethodOptions(max_chisq_threshold=50.0),
    )

    assert _block_max_chisq(block_data[0]['sumstats']) == -np.inf
    assert len(block_data[0]['sumstats']) == len(first_block)


def test_jackknife_allows_empty_filtered_blocks():
    """GraphREML handles duplicate offsets from filtered empty blocks."""
    assignments = GraphREML._get_variant_jackknife_assignments(
        [0, 0, 3], num_groups=2
    )

    np.testing.assert_array_equal(assignments, np.array([1, 1, 1]))


def test_binary_annotation_filter_subsets_existing_params(
    monkeypatch, metadata_path, create_annotations, create_sumstats
):
    """Binary filtering keeps parameter rows aligned to retained annotations."""
    captured = {}

    def fake_run(cls, *args, **kwargs):
        captured.update(kwargs)
        model = kwargs['worker_params'][0]
        return {'parameters': model.params.flatten()}

    monkeypatch.setattr(GraphREML, 'run_serial', classmethod(fake_run))

    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR').with_columns(
        pl.Series('continuous', np.linspace(0.0, 1.0, len(sumstats))),
        pl.Series('binary_flag', np.arange(len(sumstats)) % 2),
    )
    model = ModelOptions(
        annotation_columns=['base', 'continuous', 'binary_flag'],
        params=np.array([[0.1], [0.2], [0.3]]),
        sample_size=1000,
        binary_annotations_only=True,
    )
    method = MethodOptions(match_by_position=True, run_serial=True)

    result = run_graphREML(
        model_options=model,
        method_options=method,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=metadata_path,
        populations='EUR',
    )

    assert model.annotation_columns == ['base', 'binary_flag']
    np.testing.assert_allclose(model.params, np.array([[0.1], [0.3]]))
    assert captured['num_params'] == 2
    np.testing.assert_allclose(result['parameters'], np.array([0.1, 0.3]))


@pytest.mark.parametrize('missing_mode', ['absent', 'all_null'])
def test_missing_sample_size_warns_and_defaults_to_one(
    monkeypatch, metadata_path, create_annotations, create_sumstats, missing_mode
):
    """GraphREML warns before falling back to sample_size=1.0."""
    captured = {}

    def fake_run(cls, *args, **kwargs):
        captured.update(kwargs)
        return {'sample_size': kwargs['sample_size']}

    monkeypatch.setattr(GraphREML, 'run_serial', classmethod(fake_run))

    sumstats = create_sumstats(str(metadata_path), 'EUR')
    if missing_mode == 'absent':
        sumstats = sumstats.drop('N')
    else:
        sumstats = sumstats.with_columns(pl.lit(None).cast(pl.Float64).alias('N'))

    model = ModelOptions(params=np.zeros((1, 1)), sample_size=None)
    method = MethodOptions(match_by_position=True, run_serial=True)

    with pytest.warns(RuntimeWarning, match='sample_size=1.0'):
        result = run_graphREML(
            model_options=model,
            method_options=method,
            summary_stats=sumstats,
            annotation_data=create_annotations(metadata_path, 'EUR'),
            ldgm_metadata_path=metadata_path,
            populations='EUR',
        )

    assert model.sample_size == 1.0
    assert captured['sample_size'] == 1.0
    assert result['sample_size'] == 1.0


def test_sample_size_inference_ignores_nan_values(
    monkeypatch, metadata_path, create_annotations, create_sumstats
):
    """Sample-size inference uses valid N rows instead of falling back on NaN."""
    captured = {}

    def fake_run(cls, *args, **kwargs):
        captured.update(kwargs)
        return {'sample_size': kwargs['sample_size']}

    monkeypatch.setattr(GraphREML, 'run_serial', classmethod(fake_run))

    sumstats = create_sumstats(str(metadata_path), 'EUR').with_row_index('row_nr')
    sumstats = sumstats.with_columns(
        pl.when(pl.col('row_nr') == 0)
        .then(float('nan'))
        .otherwise(1000.0)
        .alias('N')
    ).drop('row_nr')
    model = ModelOptions(params=np.zeros((1, 1)), sample_size=None)
    method = MethodOptions(match_by_position=True, run_serial=True)

    result = run_graphREML(
        model_options=model,
        method_options=method,
        summary_stats=sumstats,
        annotation_data=create_annotations(metadata_path, 'EUR'),
        ldgm_metadata_path=metadata_path,
        populations='EUR',
    )

    assert model.sample_size == 1000.0
    assert captured['sample_size'] == 1000.0
    assert result['sample_size'] == 1000.0


def test_max_z_squared_threshold(metadata_path, create_annotations, create_sumstats):
    """Test that blocks with high Z² values are discarded."""
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')

    # Print initial Z² values
    print("\nInitial Z² values:")
    metadata = read_ldgm_metadata(metadata_path, populations='EUR')
    for i, block in enumerate(partition_variants(metadata, sumstats)):
        max_z2 = float(np.max(block.select('Z').to_numpy() ** 2))
        print(f"Block {i}: max Z² = {max_z2}")

    # Spike in a large Z-score in the first block
    first_pos = sumstats.select('POS').head(1).item()
    sumstats = sumstats.with_columns(pl.when(pl.col('POS') == first_pos)
                                   .then(10.0)  # This will give a Z² of 100
                                   .otherwise(pl.col('Z'))
                                   .alias('Z'))

    # Print Z² values after spiking
    print("\nZ² values after spiking:")
    metadata = read_ldgm_metadata(metadata_path, populations='EUR')
    for i, block in enumerate(partition_variants(metadata, sumstats)):
        max_z2 = float(np.max(block.select('Z').to_numpy() ** 2))
        print(f"Block {i}: max Z² = {max_z2}")

    # Set threshold to filter out the block with the large Z-score
    threshold = 50.0  # This will filter out blocks with Z² > 50

    # Run GraphREML with Z² threshold
    model = ModelOptions(params=np.zeros((1,1)),
                sample_size=1000
    )
    method = MethodOptions(
        match_by_position=True,
        max_chisq_threshold=threshold,
        verbose=True,  # Enable verbose output
        num_iterations=1,  # Only need one iteration for this test
    )
    result = run_graphREML(
        model_options=model,
        method_options=method,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=metadata_path,
        populations='EUR',
    )

    assert result is not None


def _create_variant_specific_statistics_hdf5(
    metadata_path, create_annotations, create_sumstats, *, run_serial=True
) -> str:
    """Run GraphREML once and return the generated score-test HDF5 path."""
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')

    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
        variant_stats_path = tmp.name

    model = ModelOptions(
        params=np.zeros((1,1)),
        sample_size=1000,
    )
    method = MethodOptions(
        match_by_position=False,
        num_iterations=1,  # Only need one iteration for this test
        verbose=True,
        run_serial=run_serial,
        num_processes=None if run_serial else 2,
        score_test_hdf5_file_name=variant_stats_path,
        score_test_hdf5_trait_name='test',
    )
    result = run_graphREML(
        model_options=model,
        method_options=method,
        summary_stats=sumstats,
        annotation_data=annotations,
        ldgm_metadata_path=metadata_path,
        populations='EUR'
    )

    assert result is not None
    return variant_stats_path


@pytest.mark.parametrize("run_serial", [True, False], ids=["serial", "parallel"])
def test_variant_specific_statistics(
    metadata_path, create_annotations, create_sumstats, run_serial
):
    """Score-test HDF5 output contains usable row and trait statistics."""
    variant_stats_path = _create_variant_specific_statistics_hdf5(
        metadata_path,
        create_annotations,
        create_sumstats,
        run_serial=run_serial,
    )

    try:
        assert os.path.exists(variant_stats_path), (
            f"HDF5 file {variant_stats_path} does not exist"
        )

        with h5py.File(variant_stats_path, 'r') as f:
            assert f.attrs['data_type'] == 'variant'
            assert 'traits' in f, "HDF5 file missing 'traits' group"
            assert 'row_data' in f, "HDF5 file missing 'row_data' group"
            assert 'test' in f['traits']

            row_data = f['row_data']
            for name in ['CHR', 'POS', 'RSID', 'jackknife_blocks']:
                assert name in row_data

            n_rows = row_data['CHR'].shape[0]
            assert n_rows > 0
            assert row_data['POS'].shape[0] == n_rows
            assert row_data['RSID'].shape[0] == n_rows
            assert row_data['jackknife_blocks'].shape == (n_rows,)
            assert row_data['jackknife_blocks'][:].min() >= 0

            gradient = f['traits/test/gradient'][:]
            assert gradient.shape == (n_rows,)
            assert np.all(np.isfinite(gradient))

        method = MethodOptions(
            score_test_hdf5_file_name=variant_stats_path,
            score_test_hdf5_trait_name='test',
        )
        with pytest.raises(ValueError, match="traits/test"):
            GraphREML._write_trait_stats(method, np.zeros(n_rows))

    finally:
        if os.path.exists(variant_stats_path):
            os.unlink(variant_stats_path)


def test_score_file_append_preflight_accepts_matching_variants(
    tmp_path, metadata_path, create_annotations, create_sumstats
):
    """A compatible score file passes before graphREML starts workers."""
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')
    merged = sumstats.join(annotations, on='SNP', how='right').unique(
        subset='SNP', keep='first'
    )
    metadata = read_ldgm_metadata(metadata_path, populations='EUR')
    base_method = MethodOptions(num_jackknife_blocks=2)
    block_data = GraphREML.prepare_block_data(
        metadata, sumstats=merged, method=base_method
    )
    variants = pl.concat(
        [block['sumstats'].select(['SNP', 'CHR', 'POS']) for block in block_data]
    )
    assignments = GraphREML._get_variant_jackknife_assignments(
        [block['variant_offset'] for block in block_data] + [len(variants)], 2
    )
    score_path = tmp_path / 'scores.h5'
    GraphREML._write_variant_data(str(score_path), variants, assignments)

    append_method = MethodOptions(
        num_jackknife_blocks=2,
        score_test_hdf5_file_name=str(score_path),
        score_test_hdf5_trait_name='new_trait',
    )
    GraphREML.prepare_block_data(metadata, sumstats=merged, method=append_method)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("row_count", "existing row_data has .* variants but this run has"),
        ("rsid", "row_data/RSID does not match"),
        ("jackknife", "row_data/jackknife_blocks does not match"),
    ],
)
def test_score_file_append_preflight_rejects_mismatches(
    tmp_path, metadata_path, create_annotations, create_sumstats, mutation, match
):
    """Incompatible score files fail before graphREML starts workers."""
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')
    merged = sumstats.join(annotations, on='SNP', how='right').unique(
        subset='SNP', keep='first'
    )
    metadata = read_ldgm_metadata(metadata_path, populations='EUR')
    block_data = GraphREML.prepare_block_data(
        metadata, sumstats=merged, method=MethodOptions(num_jackknife_blocks=2)
    )
    variants = pl.concat(
        [block['sumstats'].select(['SNP', 'CHR', 'POS']) for block in block_data]
    )
    assignments = GraphREML._get_variant_jackknife_assignments(
        [block['variant_offset'] for block in block_data] + [len(variants)], 2
    )

    if mutation == "row_count":
        variants = variants.head(len(variants) - 1)
        assignments = assignments[:-1]
    elif mutation == "rsid":
        variants = variants.with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit("different"))
            .otherwise(pl.col("SNP"))
            .alias("SNP")
        )
    elif mutation == "jackknife":
        assignments[0] = assignments[0] + 1

    score_path = tmp_path / 'scores.h5'
    GraphREML._write_variant_data(str(score_path), variants, assignments)
    append_method = MethodOptions(
        num_jackknife_blocks=2,
        score_test_hdf5_file_name=str(score_path),
        score_test_hdf5_trait_name='new_trait',
    )

    with pytest.raises(ValueError, match=match):
        GraphREML.prepare_block_data(metadata, sumstats=merged, method=append_method)


@pytest.mark.parametrize("gradient_shape", [None, (3, 2)])
def test_score_file_append_preflight_rejects_malformed_existing_traits(
    tmp_path, gradient_shape
):
    """Existing traits must have one gradient value per row."""
    score_path = tmp_path / "scores.h5"
    variants = pl.DataFrame(
        {"SNP": ["rs1", "rs2", "rs3"], "CHR": [1, 1, 1], "POS": [1, 2, 3]}
    )
    GraphREML._write_variant_data(str(score_path), variants, np.zeros(3, dtype=int))
    with h5py.File(score_path, "a") as f:
        trait = f["traits"].create_group("broken")
        if gradient_shape is not None:
            trait.create_dataset("gradient", data=np.zeros(gradient_shape))

    metadata = pl.DataFrame(
        {
            "chrom": [1],
            "chromStart": [0],
            "chromEnd": [4],
            "name": ["unused.edgelist"],
            "snplistName": ["unused.snplist"],
            "population": ["EUR"],
            "numVariants": [3],
            "numIndices": [3],
            "numEntries": [3],
            "info": [""],
        }
    )
    method = MethodOptions(
        num_jackknife_blocks=1,
        score_test_hdf5_file_name=str(score_path),
        score_test_hdf5_trait_name="new_trait",
    )
    with pytest.raises(ValueError, match="existing traits do not match row_data"):
        GraphREML.prepare_block_data(metadata, sumstats=variants, method=method)


def test_score_file_write_revalidates_after_empty_file_preflight(tmp_path):
    """A competing writer cannot establish incompatible row data after preflight."""
    score_path = tmp_path / "scores.h5"
    first_variants = pl.DataFrame(
        {"SNP": ["rs1", "rs2"], "CHR": [1, 1], "POS": [1, 2]}
    )
    second_variants = first_variants.with_columns(
        pl.Series("SNP", ["other1", "other2"])
    )
    assignments = np.zeros(2, dtype=int)

    GraphREML._write_variant_data(
        str(score_path), first_variants, assignments, trait_name="first"
    )
    with pytest.raises(ValueError, match="row_data/RSID does not match"):
        GraphREML._write_variant_data(
            str(score_path), second_variants, assignments, trait_name="second"
        )


def test_score_test(metadata_path, create_annotations, create_sumstats):
    """Test the score test functionality using the refactored run_score_test function."""
    variant_stats_path = _create_variant_specific_statistics_hdf5(
        metadata_path,
        create_annotations,
        create_sumstats
    )

    try:
        # Import the necessary functions from score_test
        from score_test.score_test_io import load_trait_data, load_row_data
        from score_test.score_test import run_score_test, VariantAnnot

        # Load variant and trait data
        variant_table = load_row_data(variant_stats_path)
        trait_data = load_trait_data(variant_stats_path, trait_name='test', variant_table=variant_table)

        # Create a simple test annotation dataframe
        annotations = create_annotations(metadata_path, 'EUR')
        test_annot_name = "test_score_annot"

        # Rename SNP to RSID for consistency with new format
        annotations = annotations.rename({'SNP': 'RSID'})
        
        # Add a random annotation column
        np.random.seed(42)
        annotations = annotations.with_columns(
            pl.Series(name=test_annot_name, values=np.random.normal(0, 1, annotations.height))
        )

        # Create VariantAnnot object
        annot = VariantAnnot(annotations, [test_annot_name])

        # Run the score test with the loaded dataframes
        point_estimates, jackknife_estimates = run_score_test(
            trait_data=trait_data,
            annot=annot,
        )

        # Verify the results
        assert point_estimates is not None
        assert jackknife_estimates is not None

    finally:
        # Clean up the variant statistics file
        if os.path.exists(variant_stats_path):
            os.unlink(variant_stats_path)


def test_trust_region_rejects_bad_step_before_accepting(monkeypatch):
    """Rejected trust-region steps should restore params before retrying."""
    class FakeTrustRegionManager:
        def __init__(self, shared_data):
            self.shared_data = shared_data
            self.likelihood_only_calls = 0
            self.flags = []

        def start_workers(self, flag=None):
            self.flags.append(flag)
            if flag == FLAGS["INITIALIZE"]:
                self.shared_data["likelihood"] = np.array([0.0, 0.0])
                self.shared_data["gradient"] = np.array([1.0, 0.0])
                self.shared_data["hessian"] = np.array([-1.0, 0.0])
                self.shared_data["variant_data"] = np.array([0.2, 0.2])
            elif flag == FLAGS["COMPUTE_LIKELIHOOD_ONLY"]:
                self.likelihood_only_calls += 1
                if self.likelihood_only_calls == 1:
                    self.shared_data["likelihood"] = np.array([-1.0, 0.0])
                else:
                    self.shared_data["likelihood"] = np.array([0.1, 0.0])

        def await_workers(self):
            pass

    monkeypatch.setattr(
        GraphREML,
        "_annotation_heritability",
        staticmethod(lambda variant_h2, annot, ref_col: (
            np.array([0.4]),
            np.array([1.0]),
        )),
    )
    monkeypatch.setattr(
        GraphREML,
        "_compute_pseudojackknife",
        staticmethod(lambda gradient_blocks, hessian_blocks, params: np.array([
            params,
            params + 0.01,
        ])),
    )
    monkeypatch.setattr(
        GraphREML,
        "_compute_jackknife_heritability",
        staticmethod(lambda block_data, jackknife_params, model: (
            np.array([[0.4], [0.41]]),
            np.array([[1.0], [1.0]]),
        )),
    )

    shared_data = SharedData({
        "likelihood": 2,
        "gradient": 2,
        "hessian": 2,
        "variant_data": 2,
        "params": 1,
    })
    manager = FakeTrustRegionManager(shared_data)
    block_data = [
        {
            "sumstats": pl.DataFrame({
                "SNP": ["rs1"],
                "CHR": [1],
                "POS": [10],
                "base": [1.0],
            }),
            "variant_offset": 0,
        },
        {
            "sumstats": pl.DataFrame({
                "SNP": ["rs2"],
                "CHR": [1],
                "POS": [20],
                "base": [1.0],
            }),
            "variant_offset": 1,
        },
    ]
    model = ModelOptions(
        annotation_columns=["base"],
        params=np.zeros((1, 1)),
        sample_size=1000,
    )
    method = MethodOptions(
        num_iterations=1,
        trust_region_size=1.0,
        trust_region_scalar=5.0,
        max_trust_iterations=3,
        num_jackknife_blocks=2,
    )

    result = GraphREML.supervise(
        manager,
        shared_data,
        block_data,
        num_iterations=1,
        num_params=1,
        verbose=False,
        method=method,
        model=model,
    )

    assert manager.flags == [
        FLAGS["INITIALIZE"],
        FLAGS["COMPUTE_LIKELIHOOD_ONLY"],
        FLAGS["COMPUTE_LIKELIHOOD_ONLY"],
    ]
    assert manager.likelihood_only_calls == 2
    np.testing.assert_allclose(result["parameters"], np.array([1.0 / 6.0]))
    np.testing.assert_allclose(shared_data["params"], np.array([1.0 / 6.0]))
    assert result["log"]["trust_region_lambdas"] == [5.0]
