"""Test for VariantAnnot.merge with 2D annotations array."""

import numpy as np
import polars as pl
import pytest
import h5py
from pathlib import Path
from scipy.sparse import csr_matrix

from score_test.convert_scores import convert_hdf5, convert_variant_to_gene_scores
from score_test.score_test import VariantAnnot, TraitData, get_block_boundaries, run_score_test
from score_test.score_test_io import (
    create_random_variant_annotations,
    get_trait_groups,
    is_gene_level_hdf5,
    load_trait_data,
    load_variant_data,
)


def test_get_block_boundaries_returns_half_open_slice_boundaries():
    blocks = np.array([7, 7, 9, 9, 9, 12])

    boundaries = get_block_boundaries(blocks)

    np.testing.assert_array_equal(boundaries, np.array([0, 2, 5, 6]))


def test_run_score_test_uses_exact_jackknife_block_slices():
    trait_data = TraitData(
        pl.DataFrame({
            "RSID": ["rs0", "rs1", "rs2", "rs3", "rs4"],
            "CHR": [1, 1, 1, 1, 1],
            "POS": [0, 1, 2, 3, 4],
            "jackknife_blocks": [0, 0, 1, 1, 2],
            "gradient": [1.0, 2.0, 3.0, 4.0, 5.0],
        }),
        keys=["RSID"],
    )
    annot = VariantAnnot(
        pl.DataFrame({
            "RSID": ["rs0", "rs1", "rs2", "rs3", "rs4"],
            "annot_a": [1.0, 0.0, 1.0, 0.0, 1.0],
            "annot_b": [0.0, 1.0, 1.0, 1.0, 0.0],
        }),
        ["annot_a", "annot_b"],
    )

    point_estimates, jackknife_estimates = run_score_test(trait_data, annot)

    np.testing.assert_array_equal(point_estimates, np.array([[9.0, 9.0]]))
    np.testing.assert_array_equal(
        jackknife_estimates,
        np.array([
            [8.0, 7.0],
            [6.0, 2.0],
            [4.0, 9.0],
        ]),
    )


def test_convert_variant_to_gene_scores_uses_correct_variant_block_boundaries():
    trait_data = TraitData(
        pl.DataFrame({
            "CHR": [1, 1, 1, 1, 1],
            "POS": [0, 1, 2, 3, 4],
            "jackknife_blocks": [0, 0, 1, 1, 2],
            "gradient": [1.0, 2.0, 3.0, 4.0, 5.0],
        })
    )
    gene_table = pl.DataFrame({
        "CHR": [1, 1, 1, 1],
        "midpoint": [0, 1, 2, 3],
        "gene_id": ["ENSG0", "ENSG1", "ENSG2", "ENSG3"],
        "gene_name": ["G0", "G1", "G2", "G3"],
    })
    variant_gene_matrix = csr_matrix(
        (
            np.ones(5),
            (
                np.array([0, 1, 2, 3, 4]),
                np.array([0, 0, 1, 2, 3]),
            ),
        ),
        shape=(5, 4),
    )

    converted = convert_variant_to_gene_scores(trait_data, variant_gene_matrix, gene_table)

    assert converted.df["jackknife_blocks"].to_list() == [0, 1, 1, 2]
    np.testing.assert_array_equal(
        converted.df["gradient"].to_numpy(),
        np.array([3.0, 3.0, 4.0, 5.0]),
    )


def test_convert_hdf5_writes_loadable_gene_level_hdf5(tmp_path):
    variant_hdf5 = tmp_path / "variant_scores.h5"
    gene_hdf5 = tmp_path / "gene_scores.h5"
    genes_tsv = tmp_path / "genes.tsv"
    genes_tsv.write_text(
        "\n".join([
            "gene_id\tgene_id_version\tgene_name\tstart\tend\tCHR",
            "ENSG0\tENSG0.1\tG0\t90\t110\t1",
            "ENSG1\tENSG1.1\tG1\t190\t210\t1",
            "ENSG2\tENSG2.1\tG2\t290\t310\t1",
        ])
        + "\n"
    )

    with h5py.File(variant_hdf5, "w") as f:
        f.attrs["metadata"] = ""
        f.attrs["data_type"] = "variant"
        f.attrs["keys"] = ["RSID", "POS"]

        row_data = f.create_group("row_data")
        row_data.create_dataset("CHR", data=np.array([1, 1, 1]))
        row_data.create_dataset("POS", data=np.array([100, 200, 300]))
        row_data.create_dataset("RSID", data=np.array(["rs0", "rs1", "rs2"], dtype="S"))
        row_data.create_dataset("jackknife_blocks", data=np.array([0, 1, 2]))

        trait = f.create_group("traits").create_group("height")
        params = trait.create_group("parameters")
        params.create_dataset("parameters", data=np.array([0.0]))
        params.create_dataset("jackknife_parameters", data=np.zeros((3, 1)))
        trait.create_dataset("gradient", data=np.array([1.0, 2.0, 3.0]))
        trait.create_dataset("hessian", data=np.ones(3))

        f.create_group("groups").create_dataset(
            "all_traits",
            data=np.array(["height"], dtype="S"),
        )

    convert_hdf5(str(variant_hdf5), str(gene_hdf5), str(genes_tsv), [1.0])

    assert is_gene_level_hdf5(str(gene_hdf5))
    assert get_trait_groups(str(gene_hdf5)) == {"all_traits": ["height"]}

    gene_table = load_variant_data(str(gene_hdf5))
    assert gene_table["gene_id"].to_list() == ["ENSG0", "ENSG1", "ENSG2"]
    assert gene_table["gene_name"].to_list() == ["G0", "G1", "G2"]
    assert gene_table["jackknife_blocks"].to_list() == [0, 1, 2]

    trait_data = load_trait_data(str(gene_hdf5), "height", gene_table)
    np.testing.assert_array_equal(
        trait_data.df["gradient"].to_numpy(),
        np.array([1.0, 2.0, 3.0]),
    )


def test_variant_annot_merge_with_2d_annotations(tmp_path):
    """Test that VariantAnnot.merge handles 2D annotations array correctly."""
    # Create test HDF5 file with 2D annotations
    hdf5_path = tmp_path / "test_merge.h5"
    
    n_variants = 100
    n_params = 5
    n_blocks = 10
    
    with h5py.File(hdf5_path, 'w') as f:
        # Use new format
        f.attrs['metadata'] = ''
        f.attrs['data_type'] = 'variant'
        f.attrs['keys'] = ['RSID', 'POS']
        
        # Create row_data group
        row_data_group = f.create_group('row_data')
        row_data_group.create_dataset('CHR', data=np.ones(n_variants, dtype=int))
        row_data_group.create_dataset('POS', data=np.arange(n_variants))
        # Split annotations into separate columns
        annot_data = np.random.randn(n_variants, n_params)
        for i in range(n_params):
            row_data_group.create_dataset(f'annot_{i}', data=annot_data[:, i])
        row_data_group.create_dataset('jackknife_blocks', data=np.repeat(np.arange(n_blocks), n_variants // n_blocks))
        row_data_group.create_dataset('RSID', data=np.array([f'rs{i}' for i in range(n_variants)], dtype='S10'))
        
        # Create traits group
        traits_group = f.create_group('traits')
        trait_group = traits_group.create_group('test_trait')
        params_group = trait_group.create_group('parameters')
        params_group.create_dataset('parameters', data=np.random.randn(n_params))
        params_group.create_dataset('jackknife_parameters', data=np.random.randn(n_blocks, n_params))
        trait_group.create_dataset('gradient', data=np.random.randn(n_variants))
        trait_group.create_dataset('hessian', data=np.random.randn(n_variants))
        
        f.create_group('groups')
    
    # Load variant and trait data
    variant_table = load_variant_data(str(hdf5_path))
    trait_data = load_trait_data(str(hdf5_path), 'test_trait', variant_table)
    
    # Create random variant annotations
    np.random.seed(42)
    probs = [0.1, 0.2]
    variant_annot = create_random_variant_annotations(trait_data.df, probs)
    
    # This should not raise an error
    grad, hessian, model_annot, test_annot, block_boundaries = variant_annot.merge(trait_data)
    
    # Verify shapes
    assert grad.shape == (n_variants,)
    assert hessian.shape == (n_variants,)
    assert model_annot.shape == (n_variants, n_params)
    assert test_annot.shape == (n_variants, len(probs))
    assert len(block_boundaries) == n_blocks + 1


def test_run_score_test_with_2d_annotations(tmp_path):
    """Test full run_score_test with 2D annotations array."""
    # Create test HDF5 file
    hdf5_path = tmp_path / "test_score.h5"
    
    n_variants = 100
    n_params = 5
    n_blocks = 10
    
    with h5py.File(hdf5_path, 'w') as f:
        # Use new format
        f.attrs['metadata'] = ''
        f.attrs['data_type'] = 'variant'
        f.attrs['keys'] = ['RSID', 'POS']
        
        # Create row_data group
        row_data_group = f.create_group('row_data')
        row_data_group.create_dataset('CHR', data=np.ones(n_variants, dtype=int))
        row_data_group.create_dataset('POS', data=np.arange(n_variants))
        annot_data = np.random.randn(n_variants, n_params)
        for i in range(n_params):
            row_data_group.create_dataset(f'annot_{i}', data=annot_data[:, i])
        row_data_group.create_dataset('jackknife_blocks', data=np.repeat(np.arange(n_blocks), n_variants // n_blocks))
        row_data_group.create_dataset('RSID', data=np.array([f'rs{i}' for i in range(n_variants)], dtype='S10'))
        
        # Create traits group
        traits_group = f.create_group('traits')
        trait_group = traits_group.create_group('test_trait')
        params_group = trait_group.create_group('parameters')
        params_group.create_dataset('parameters', data=np.random.randn(n_params))
        params_group.create_dataset('jackknife_parameters', data=np.random.randn(n_blocks, n_params))
        trait_group.create_dataset('gradient', data=np.random.randn(n_variants))
        trait_group.create_dataset('hessian', data=np.random.randn(n_variants))
        
        f.create_group('groups')
    
    # Load variant and trait data
    variant_table = load_variant_data(str(hdf5_path))
    trait_data = load_trait_data(str(hdf5_path), 'test_trait', variant_table)
    
    # Create random variant annotations
    np.random.seed(42)
    probs = [0.1, 0.2]
    variant_annot = create_random_variant_annotations(trait_data.df, probs)
    
    # Run score test - this should not raise an error
    point_estimates, jackknife_estimates = run_score_test(
        trait_data=trait_data,
        annot=variant_annot,
    )
    
    # Verify output shapes
    assert point_estimates.shape == (1, len(probs))
    assert jackknife_estimates.shape == (n_blocks, len(probs))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
