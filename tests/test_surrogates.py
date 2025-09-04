import numpy as np
import polars as pl
import h5py
import tempfile

from graphld.surrogates import get_surrogate_markers
from graphld.heritability import MethodOptions, ModelOptions, run_graphREML


def _identity_stats(h5_path: str) -> tuple[int, int]:
    total = 0
    identical = 0
    with h5py.File(h5_path, 'r') as h5:
        for name in h5.keys():
            arr = h5[name][:]
            total += arr.size
            identical += np.count_nonzero(arr == np.arange(arr.size, dtype=arr.dtype))
    return identical, total


def test_get_surrogate_markers(metadata_path, create_sumstats):
    """Smoke-test: write HDF5 per-block surrogates and validate missingness fraction."""
    sumstats = create_sumstats(ldgm_metadata_path=metadata_path, populations="EUR")

    # Case 1: ~10% missing (keep POS % 10 != 0)
    with tempfile.NamedTemporaryFile(suffix=".h5") as tmp:
        nonmissing_10 = sumstats.filter(pl.col('POS') % 10 != 0)
        h5_path_10 = get_surrogate_markers(
            metadata_path,
            nonmissing_10,
            population="EUR",
            run_serial=True,
            output_path=tmp.name,
        )

        ident10, total10 = _identity_stats(h5_path_10)
        frac_nonident10 = 1.0 - ident10 / total10
        exp_missing10 = 1.0 - len(nonmissing_10) / len(sumstats)

        assert abs(frac_nonident10 - exp_missing10) < 0.05


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
