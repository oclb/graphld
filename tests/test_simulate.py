"""Test simulation functionality."""

from multiprocessing import Value

import numpy as np
import polars as pl
import pytest
from scipy.sparse import csc_matrix

from graphld import PrecisionOperator, Simulate
from graphld.multiprocessing_template import SharedData
from graphld.simulate import _SimulationSpecification


# Module-level function for testing workaround 3
def module_level_link_fn(x: np.ndarray) -> np.ndarray:
    """Example module-level link function that can be pickled."""
    return x[:, 0]


def test_seeded_effect_size_rng_is_block_offset_specific_in_process_block():
    """Seeded effect-size draws should not repeat across identical blocks."""
    variant_info = pl.DataFrame({
        'index': np.arange(6),
        'af': np.full(6, 0.2),
    })
    ldgm = PrecisionOperator(csc_matrix(np.eye(6)), variant_info)
    spec = _SimulationSpecification(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[1.0],
        alpha_param=-1,
        random_seed=42,
    )
    shared_data = SharedData({
        'beta': 2 * len(variant_info),
        'alpha': 2 * len(variant_info),
        'noise': 2 * len(variant_info),
    })

    Simulate.process_block(
        ldgm,
        Value('i', 1),
        shared_data,
        block_offset=0,
        worker_params=spec,
    )
    Simulate.process_block(
        ldgm,
        Value('i', 1),
        shared_data,
        block_offset=len(variant_info),
        worker_params=spec,
    )

    beta_block_1 = shared_data['beta'][:len(variant_info)]
    beta_block_2 = shared_data['beta'][len(variant_info):]
    assert not np.array_equal(beta_block_1, beta_block_2)


def test_process_block_phases_simulated_outputs_to_annotation_alleles():
    """Swapped A1/A2 annotations should flip simulated output signs."""
    variant_info = pl.DataFrame({
        'index': [0, 1],
        'site_ids': ['rs1', 'rs2'],
        'position': [10, 20],
        'anc_alleles': ['A', 'C'],
        'deriv_alleles': ['G', 'T'],
        'af': [0.2, 0.3],
    })
    spec = _SimulationSpecification(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[1.0],
        alpha_param=-1,
        random_seed=7,
    )
    aligned_annotations = pl.DataFrame({
        'SNP': ['rs1', 'rs2'],
        'CHR': [1, 1],
        'POS': [10, 20],
        'A2': ['A', 'C'],
        'A1': ['G', 'T'],
        'base': [1, 1],
    })
    swapped_annotations = aligned_annotations.with_columns(
        pl.when(pl.col('SNP') == 'rs2')
        .then(pl.col('A1'))
        .otherwise(pl.col('A2'))
        .alias('A2'),
        pl.when(pl.col('SNP') == 'rs2')
        .then(pl.col('A2'))
        .otherwise(pl.col('A1'))
        .alias('A1'),
    )

    def run_block(annotations: pl.DataFrame) -> SharedData:
        ldgm = PrecisionOperator(csc_matrix(np.eye(2)), variant_info)
        shared_data = SharedData({
            'beta': len(annotations),
            'alpha': len(annotations),
            'noise': len(annotations),
        })
        Simulate.process_block(
            ldgm,
            Value('i', 1),
            shared_data,
            block_offset=0,
            block_data=(annotations, 0),
            worker_params=spec,
        )
        return shared_data

    aligned = run_block(aligned_annotations)
    swapped = run_block(swapped_annotations)

    for output_name in ['beta', 'alpha', 'noise']:
        np.testing.assert_allclose(swapped[output_name][0], aligned[output_name][0])
        np.testing.assert_allclose(swapped[output_name][1], -aligned[output_name][1])


def test_simulate_with_annotations(metadata_path, create_annotations):
    """Test simulation with variant annotations."""
    # Create simulator with specific settings
    sim = Simulate(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[0.3],
        alpha_param=-1,
        random_seed=42
    )

    # Create annotations from metadata
    annotations = create_annotations(metadata_path, populations="EUR")

    sim_result = sim.simulate(
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        annotations=annotations
    )

    num_annotated_variants = np.sum(
        annotations.select(pl.col('POS').is_first_distinct()).to_numpy()
    )
    assert len(sim_result) == num_annotated_variants
    assert np.sum(sim_result.select('beta').to_numpy() != 0) > 0


def test_component_mixture(metadata_path, create_annotations):
    """Test simulation with multiple variance components."""
    sim = Simulate(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0, 0.1],
        component_weight=[0.01, 0.1],
        alpha_param=-1,
        random_seed=44
    )

    # Create annotations from metadata
    annotations = create_annotations(metadata_path, populations="EUR")

    sim_result = sim.simulate(
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        annotations=annotations
    )

    # Check that we have the expected proportion of non-zero effects
    beta = sim_result['beta'].to_numpy()
    non_zero = beta != 0
    assert 0.01 < np.mean(non_zero) < 0.2

    # Check sparse architecture
    sim.component_variance = [0.0001, 0.00001]

    # Create annotations from metadata
    annotations = create_annotations(metadata_path, populations="EUR")

    sim_result = sim.simulate(
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        annotations=annotations
    )

    # Check that we have the expected proportion of non-zero effects
    beta = sim_result['beta'].to_numpy()
    non_zero = beta != 0
    assert 0.01 < np.mean(non_zero) < 0.2


@pytest.mark.skip(reason="Annotation-dependent polygenicity not implemented yet")
def test_annotation_dependent_polygenicity(metadata_path, create_annotations):
    """Test annotation-dependent polygenicity."""
    # Create simulator with annotation-dependent polygenicity
    sim = Simulate(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[0.3],
        alpha_param=-1,
        annotation_dependent_polygenicity=True,
        random_seed=42
    )

    # Create annotations with a binary feature
    annotations = create_annotations(metadata_path, populations="EUR")
    annotations = annotations.with_columns([
        (pl.col('BP') % 2 == 0).alias('binary_feature')  # Binary feature
    ])

    sim_result = sim.simulate(
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        annotations=annotations
    )

    # Check that binary feature affects probability of being causal
    beta = sim_result['beta'].to_numpy()
    pos = annotations['BP'].to_numpy()
    is_even = pos % 2 == 0

    causal_even = (beta[is_even] != 0).mean()
    causal_odd = (beta[~is_even] != 0).mean()
    assert abs(causal_even - causal_odd) > 0.05  # Should see difference in causal probability


def test_heritability_scaling(metadata_path, create_annotations):
    """Test that total heritability scales correctly."""
    h2_values = [0.1, 0.5, 0.9]

    for h2 in h2_values:
        sim = Simulate(
            sample_size=100_000,
            heritability=h2,
            component_variance=[1.0],
            component_weight=[0.3],
            alpha_param=-1,
            random_seed=42
        )

        result = sim.simulate(
            ldgm_metadata_path=metadata_path,
            populations="EUR",
        )

        # Compute realized heritability
        beta = result['beta'].to_numpy()
        beta_marginal = result['beta_marginal'].to_numpy()
        realized_h2 = np.dot(beta, beta_marginal)

        np.testing.assert_allclose(realized_h2, h2, rtol=1e-2)


def test_reproducibility(metadata_path, create_annotations):
    """Test that simulations are reproducible with same random seed."""
    sim1 = Simulate(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[0.3],
        alpha_param=-1,
        random_seed=42
    )

    sim2 = Simulate(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[0.3],
        alpha_param=-1,
        random_seed=42
    )

    # Create annotations from metadata
    annotations = create_annotations(metadata_path, populations="EUR")

    result1 = sim1.simulate(
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        annotations=annotations
    )

    result2 = sim2.simulate(
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        annotations=annotations
    )

    # Check that all arrays are identical
    np.testing.assert_array_almost_equal(
        result1['beta'].to_numpy(),
        result2['beta'].to_numpy()
    )
    np.testing.assert_array_almost_equal(
        result1['beta_marginal'].to_numpy(),
        result2['beta_marginal'].to_numpy()
    )
    np.testing.assert_array_almost_equal(
        result1['Z'].to_numpy(),
        result2['Z'].to_numpy()
    )


def test_serial_and_parallel_simulation_match_fixed_seed(
    metadata_path, create_annotations
):
    """Serial and multiprocessing runs should produce identical seeded output."""
    annotations = create_annotations(metadata_path, populations="EUR")
    kwargs = dict(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[0.3],
        alpha_param=-1,
        random_seed=42,
    )

    serial = Simulate(**kwargs).simulate(
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        annotations=annotations,
        run_in_serial=True,
    )
    parallel = Simulate(**kwargs).simulate(
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        annotations=annotations,
        run_in_serial=False,
        num_processes=2,
    )

    assert serial.select(["SNP", "CHR", "POS"]).to_dict(as_series=False) == (
        parallel.select(["SNP", "CHR", "POS"]).to_dict(as_series=False)
    )
    for column in ["beta", "beta_marginal", "Z"]:
        np.testing.assert_allclose(
            serial[column].to_numpy(),
            parallel[column].to_numpy(),
        )
    serial_noise = (
        serial["Z"].to_numpy()
        - np.sqrt(serial["N"].to_numpy()) * serial["beta_marginal"].to_numpy()
    )
    parallel_noise = (
        parallel["Z"].to_numpy()
        - np.sqrt(parallel["N"].to_numpy()) * parallel["beta_marginal"].to_numpy()
    )
    np.testing.assert_allclose(serial_noise, parallel_noise)


def test_custom_link_function_pickling_error(metadata_path, create_annotations):
    """Test that lambda functions cause informative error when using multiprocessing."""
    # Lambda function that can't be pickled
    custom_link = lambda x: x[:, 0] + 9 * x[:, 1] if x.shape[1] > 1 else x[:, 0]
    
    sim = Simulate(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[0.3],
        alpha_param=-1,
        random_seed=42,
        link_fn=custom_link,
        annotation_columns=['af']
    )
    
    annotations = create_annotations(metadata_path, populations="EUR")
    
    # This should raise an informative ValueError when trying to use multiprocessing
    with pytest.raises(ValueError) as exc_info:
        result = sim.simulate(
            ldgm_metadata_path=metadata_path,
            populations="EUR",
            annotations=annotations,
            run_in_serial=False  # Force multiprocessing
        )
    
    # Check that it's an informative error message
    error_msg = str(exc_info.value)
    assert "picklable" in error_msg.lower()
    assert "module level" in error_msg.lower() or "run_in_serial" in error_msg.lower()


def test_custom_link_function_workaround_serial(metadata_path, create_annotations):
    """Test workaround 1: Use run_in_serial=True with lambda functions."""
    # Lambda works fine when run_in_serial=True
    # Simple link function that just uses allele frequency
    custom_link = lambda x: x[:, 0]
    
    sim = Simulate(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[0.3],
        alpha_param=-1,
        random_seed=42,
        link_fn=custom_link,
        annotation_columns=['af']
    )
    
    annotations = create_annotations(metadata_path, populations="EUR")
    
    # This should work with run_in_serial=True
    result = sim.simulate(
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        annotations=annotations,
        run_in_serial=True  # Avoids multiprocessing
    )
    
    assert len(result) > 0
    assert 'beta' in result.columns


def test_custom_link_function_workaround_module_level(metadata_path, create_annotations):
    """Test workaround 2: Use a module-level function instead of lambda."""
    # Module-level functions can be pickled
    sim = Simulate(
        sample_size=100_000,
        heritability=0.5,
        component_variance=[1.0],
        component_weight=[0.3],
        alpha_param=-1,
        random_seed=42,
        link_fn=module_level_link_fn,  # Use the module-level function defined at top
        annotation_columns=['af']
    )
    
    annotations = create_annotations(metadata_path, populations="EUR")
    
    # This should work with multiprocessing
    result = sim.simulate(
        ldgm_metadata_path=metadata_path,
        populations="EUR",
        annotations=annotations,
        run_in_serial=False
    )
    
    assert len(result) > 0
    assert 'beta' in result.columns
