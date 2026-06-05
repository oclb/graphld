"""Tests for VCF reading functionality."""

import tempfile

import pytest

from graphld.vcf_io import read_gwas_vcf


def test_read_gwas_vcf():
    """Test reading GWAS-VCF files."""
    # Read test VCF file
    vcf_df = read_gwas_vcf("data/test/example.gwas.vcf")

    # Test required columns are present
    required_columns = ['CHR', 'POS', 'ID', 'REF', 'ALT', 'ES', 'SE', 'LP', 'Z', 'AF']
    missing_columns = [col for col in required_columns if col not in vcf_df.columns]
    assert not missing_columns, f"Missing columns: {missing_columns}"

    # Test column renaming
    assert '#CHROM' not in vcf_df.columns
    assert 'CHR' in vcf_df.columns

    # Test Z-score calculation
    z_scores = (vcf_df['ES'] / vcf_df['SE']).to_numpy()
    assert (vcf_df['Z'] - z_scores).abs().max() < 1e-10  # Check Z-scores are calculated correctly

    # Test verbose output
    with pytest.raises(ValueError, match="Missing required columns"):
        # Create a temporary file with missing required FORMAT fields
        with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf') as temp_vcf:
            temp_vcf.write("##fileformat=VCFv4.2\n")
            temp_vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
            temp_vcf.write("1\t13116\trs62635286\tT\tG\t.\t.\t.\tSE:LP:AF\t0.00370718:1.42022:0.189015\n")
            temp_vcf.flush()
            read_gwas_vcf(temp_vcf.name, verbose=True)


def test_read_gwas_vcf_arbitrary_sample_name_and_row_format_order():
    """VCF sample names and FORMAT key order are row-specific."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf') as temp_vcf:
        temp_vcf.write("##fileformat=VCFv4.2\n")
        temp_vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\theight\n")
        temp_vcf.write("1\t100\trs1\tA\tG\t.\t.\t.\tES:SE:LP\t0.2:0.1:3\n")
        temp_vcf.write("1\t101\trs2\tA\tG\t.\t.\t.\tSE:LP:ES\t0.2:4:0.6\n")
        temp_vcf.flush()

        vcf_df = read_gwas_vcf(temp_vcf.name)

    assert vcf_df['ID'].to_list() == ['rs1', 'rs2']
    assert vcf_df['ES'].to_list() == [0.2, 0.6]
    assert vcf_df['SE'].to_list() == [0.1, 0.2]
    assert vcf_df['LP'].to_list() == [3.0, 4.0]
    assert vcf_df['Z'].to_list() == pytest.approx([2.0, 3.0])


def test_read_gwas_vcf_rejects_multiple_sample_columns():
    """Multi-sample VCF requires an explicit reader contract."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vcf') as temp_vcf:
        temp_vcf.write("##fileformat=VCFv4.2\n")
        temp_vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\theight\tbmi\n")
        temp_vcf.write("1\t100\trs1\tA\tG\t.\t.\t.\tES:SE:LP\t0.2:0.1:3\t0.3:0.1:5\n")
        temp_vcf.flush()

        with pytest.raises(ValueError, match="exactly one sample column"):
            read_gwas_vcf(temp_vcf.name)
