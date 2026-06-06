"""Test CLI functionality."""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from graphld import LDClumper
from graphld import cli
from graphld.cli import _blup, _clump, _reml, _simulate
from graphld.ldsc_io import read_ldsc_sumstats


@pytest.fixture
def test_data_dir():
    """Get test data directory."""
    return Path("data/test")


@pytest.fixture
def metadata_path(test_data_dir):
    """Get test metadata path."""
    return test_data_dir / "metadata.csv"


@pytest.fixture
def sumstats_path(test_data_dir):
    """Get test sumstats path."""
    return test_data_dir / "example.sumstats"


@pytest.fixture
def annotation_dir(test_data_dir):
    """Get test annotation directory."""
    return test_data_dir / "annot"


def test_blup(metadata_path, sumstats_path):
    """Test BLUP command."""
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        _blup(
            sumstats=str(sumstats_path),
            out=tmp.name,
            metadata=str(metadata_path),
            num_samples=1000,
            heritability=0.5,
            num_processes=None,
            run_in_serial=True,
            chromosome=None,
            population=None,
            verbose=False,
            quiet=True
        )

        # Check output exists and has expected columns
        result = pl.read_csv(tmp.name, separator='\t')
        assert 'Z' in result.columns  # BLUP outputs Z-scores
        assert len(result) > 0


def test_clump(metadata_path, sumstats_path):
    """Test clumping command."""
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        _clump(
            sumstats=str(sumstats_path),
            out=tmp.name,
            metadata=str(metadata_path),
            num_samples=None,
            min_chisq=1.0,  # Lower threshold to get some results
            max_rsq=0.9,    # Higher threshold to get some results
            num_processes=None,
            run_in_serial=True,
            chromosome=None,
            population=None,
            verbose=False,
            quiet=True
        )

        # Check output exists and has expected columns
        result = pl.read_csv(tmp.name, separator='\t')
        expected = LDClumper.clump(
            read_ldsc_sumstats(str(sumstats_path)),
            ldgm_metadata_path=str(metadata_path),
            rsq_threshold=0.9,
            chisq_threshold=1.0,
            run_in_serial=True,
        ).filter(pl.col('is_index')).drop('is_index')

        # CLI output remains narrowed to retained index variants.
        assert 'SNP' in result.columns
        assert 'Z' in result.columns
        assert 'is_index' not in result.columns
        assert result.select('SNP').to_series().to_list() == (
            expected.select('SNP').to_series().to_list()
        )


def test_simulate(metadata_path):
    """Test simulation command."""
    with tempfile.NamedTemporaryFile(suffix=".sumstats") as tmp:
        _simulate(
            sumstats_out=tmp.name,
            metadata=str(metadata_path),
            heritability=0.5,
            component_variance=[1.0],
            component_weight=[1.0],
            alpha_param=-0.5,
            annotation_dependent_polygenicity=False,
            random_seed=42,
            annotation_columns=None,
            num_processes=None,
            run_in_serial=True,
            chromosome=None,
            population=None,
            verbose=False,
            sample_size=1000
        )

        # Check output exists and has expected columns
        result = pl.read_csv(tmp.name, separator='\t')
        assert 'Z' in result.columns
        assert 'POS' in result.columns
        assert len(result) > 0


def test_invalid_sumstats_format():
    """Test error handling for invalid sumstats format."""
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        with pytest.raises(ValueError, match=r"Input file must end in \.vcf, \.vcf\.gz, \.parquet, or \.sumstats"):
            _blup(
                sumstats=tmp.name,
                out="out.csv",
                metadata="data/test/metadata.csv",
                num_samples=1000,
                heritability=0.5,
                num_processes=None,
                run_in_serial=True,
                chromosome=None,
                population=None,
                verbose=False,
                quiet=True
            )


def test_vcf_gz_dispatch_uses_gwas_vcf_reader(monkeypatch):
    """`.vcf.gz` should not fall through to LDSC summary-stat parsing."""
    calls = []

    def fake_read_gwas_vcf(path, maximum_missingness=1):
        calls.append((path, maximum_missingness))
        return pl.DataFrame({"Z": [1.0]})

    monkeypatch.setattr("graphld.vcf_io.read_gwas_vcf", fake_read_gwas_vcf)

    result = cli._detect_sumstats_type("trait.vcf.gz", maximum_missingness=0.25)

    assert result["Z"].to_list() == [1.0]
    assert calls == [("trait.vcf.gz", 0.25)]


def test_command_sumstats_reader_accepts_vcf_gz(monkeypatch):
    """Commands using strict extension validation should also accept `.vcf.gz`."""
    calls = []

    def fake_read_gwas_vcf(path):
        calls.append(path)
        return pl.DataFrame({"Z": [1.0]})

    monkeypatch.setattr(cli, "read_gwas_vcf", fake_read_gwas_vcf)

    result, file_format = cli._read_sumstats_for_command("trait.vcf.gz")

    assert file_format == "vcf"
    assert result["Z"].to_list() == [1.0]
    assert calls == ["trait.vcf.gz"]


def test_reml_sumstats_detector_rejects_unknown_extension():
    """REML should not parse arbitrary suffixes as LDSC summary statistics."""
    with pytest.raises(ValueError, match=r"Input file must end in \.vcf, \.vcf\.gz, \.parquet, or \.sumstats"):
        cli._detect_sumstats_type("trait.txt")


def test_simulate_cli_forwards_annotation_columns(monkeypatch, tmp_path):
    """The CLI `--annotation-columns` value should configure Simulate."""
    init_kwargs = []

    class FakeSimulate:
        def __init__(self, **kwargs):
            init_kwargs.append(kwargs)

        def simulate(self, **kwargs):
            return pl.DataFrame({
                "CHR": [1],
                "SNP": ["rs1"],
                "POS": [100],
                "A1": ["A"],
                "A2": ["G"],
                "Z": [0.0],
            })

    monkeypatch.setattr(cli.gld, "Simulate", FakeSimulate)

    out = tmp_path / "sim.sumstats"
    _simulate(
        sumstats_out=str(out),
        metadata="data/test/metadata.csv",
        heritability=0.5,
        sample_size=1000,
        annotation_columns=["base", "coding"],
        quiet=True,
    )

    assert init_kwargs[0]["annotation_columns"] == ["base", "coding"]


def test_reml_multi_trait_parquet_writes_per_trait_tall_outputs(monkeypatch, tmp_path):
    """Multi-trait parquet REML should not reuse one tall/convergence path."""
    parquet = tmp_path / "traits.parquet"
    pl.DataFrame({
        "SNP": ["rs1", "rs2"],
        "height_BETA": [0.1, 0.2],
        "height_SE": [0.1, 0.2],
        "bmi_BETA": [0.2, 0.3],
        "bmi_SE": [0.1, 0.1],
    }).write_parquet(parquet)

    monkeypatch.setattr(
        cli,
        "load_annotations",
        lambda *args, **kwargs: pl.DataFrame({"SNP": ["rs1"], "base": [1]}),
    )
    monkeypatch.setattr(
        cli,
        "_detect_sumstats_type",
        lambda *args, **kwargs: pl.DataFrame({"SNP": ["rs1"], "Z": [1.0], "N": [1000]}),
    )

    seen_traits = []

    def fake_run_reml_single_trait(args, sumstats, annotations, annotation_columns, trait_name):
        seen_traits.append(trait_name)
        results = {
            "enrichment": [1.0],
            "enrichment_se": [0.1],
            "enrichment_log10pval": [2.0],
            "heritability": [0.2],
            "heritability_se": [0.01],
            "heritability_log10pval": [3.0],
            "parameters": [0.4],
            "parameters_se": [0.02],
            "parameters_log10pval": [4.0],
            "log": {
                "converged": True,
                "num_iterations": 1,
                "final_likelihood": -1.0,
                "likelihood_changes": [0.0],
                "trust_region_lambdas": [1.0],
            },
            "likelihood_history": [0.0, 1.0],
        }
        return results, SimpleNamespace(annotation_columns=["base"])

    monkeypatch.setattr(cli, "_run_reml_single_trait", fake_run_reml_single_trait)

    args = SimpleNamespace(
        sumstats=str(parquet),
        annot_dir="annot",
        out=str(tmp_path / "reml"),
        metadata="data/test/metadata.csv",
        num_samples=1000,
        name=None,
        intercept=1.0,
        num_iterations=1,
        convergence_tol=0.001,
        convergence_window=1,
        run_in_serial=True,
        num_processes=None,
        verbose=False,
        quiet=True,
        num_jackknife_blocks=10,
        match_by_position=False,
        reset_trust_region=False,
        chromosome=None,
        xtrace_num_samples=10,
        max_chisq_threshold=None,
        alt_output=False,
        maximum_missingness=1.0,
        population="EUR",
        annotation_columns=None,
        score_test_filename=None,
        binary_annotations_only=False,
        surrogates=None,
        no_save=False,
        initial_params=None,
        gene_annot_dir=None,
        gene_table="data/genes.tsv",
        nearest_weights="0.4,0.2,0.1",
    )

    _reml(args)

    assert seen_traits == ["bmi", "height"]
    assert (tmp_path / "reml.bmi.tall.csv").exists()
    assert (tmp_path / "reml.bmi.convergence.csv").exists()
    assert (tmp_path / "reml.height.tall.csv").exists()
    assert (tmp_path / "reml.height.convergence.csv").exists()
    assert not (tmp_path / "reml.tall.csv").exists()


def test_reml_basic(metadata_path, create_annotations, create_sumstats):
    """Test basic REML functionality."""
    # Create test data using fixtures
    sumstats = create_sumstats(str(metadata_path), 'EUR')
    annotations = create_annotations(metadata_path, 'EUR')
    annotations = annotations.rename({'POS': 'BP'})

    # Ensure CHR is Int64 in both DataFrames
    sumstats = sumstats.with_columns(pl.col('CHR').cast(pl.Int64))
    annotations = annotations.with_columns(pl.col('CHR').cast(pl.Int64))

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "test"

        # Write sumstats to temporary file
        sumstats_file = Path(tmpdir) / "sumstats.sumstats"
        sumstats.write_csv(sumstats_file, separator='\t')

        # Write annotations to temporary file
        annot_dir = Path(tmpdir) / "annot"
        annot_dir.mkdir()
        annotations.write_csv(annot_dir / "baselineLD.22.annot", separator='\t')

        args = {
                "sumstats": str(sumstats_file),
                "annot_dir": str(annot_dir),
                "out": str(out_prefix),
                "metadata": str(metadata_path),
                "num_samples": 1000,
                "name": "test",
                "intercept": 1.0,
                "num_iterations": 2,  # Small number for testing
                "convergence_tol": 0.001,
                "convergence_window": 2,
                "run_in_serial": True,
                "num_processes": None,
                "verbose": False,
                "quiet": True,
                "num_jackknife_blocks": 100,
                "match_by_position": True,
                "reset_trust_region": False,
                "chromosome": None,
                "xtrace_num_samples": 100,
                "max_chisq_threshold": None,
                "alt_output": False,  # Use default tall format
                "maximum_missingness": 1.0,
                "variant_stats_output": None,
                "population": 'EUR',
                "annotation_columns": None,
                "score_test_filename": None,
                "binary_annotations_only": False,
                "surrogates": None,
                "no_save": False,
                "initial_params": None,
                "gene_annot_dir": None,
                "gene_table": "data/genes.tsv",
                "nearest_weights": "0.4,0.2,0.1,0.1,0.1,0.05,0.05",
            }

        _reml(
            type("Args", (), args)()
        )

        # Check that output files exist
        assert (out_prefix.with_suffix(".tall.csv")).exists()
        assert (out_prefix.with_suffix(".convergence.csv")).exists()

        # Also test alt_output format
        args['alt_output'] = True
        _reml(
            type("Args", (), args)()
        )

        # Check that alt output files exist
        assert (out_prefix.with_suffix(".heritability.csv")).exists()
        assert (out_prefix.with_suffix(".enrichment.csv")).exists()
        assert (out_prefix.with_suffix(".parameters.csv")).exists()
        assert (out_prefix.with_suffix(".convergence.csv")).exists()
