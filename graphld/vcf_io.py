from typing import Optional

import polars as pl


def split_sample_column(df: pl.DataFrame) -> pl.DataFrame:
    # Get the FORMAT from the first row
    format_columns = df.head(1).select('FORMAT').row(0)[0].split(':')

    # Validate the FORMAT columns
    validate_vcf_format_columns(format_columns, verbose=False)

    # Split the SAMPLE column and unnest into new columns
    result = df.with_columns(
        pl.col("SAMPLE")
        .str.split_exact(":", len(format_columns)-1)
        .struct.rename_fields(format_columns)
        .alias("fields")
    ).unnest("fields")


    # Cast columns to Float64 where possible
    for col in format_columns:
        result = result.with_columns(
            pl.col(col).cast(pl.Float64, strict=False)
        )

    # Add 'Z' column (ES/SE)
    result = result.with_columns(
        (pl.col('ES') / pl.col('SE')).alias('Z')
    )
    result = result.filter(~pl.col('Z').is_nan())

    return result

def process_chromosome_column(df: pl.DataFrame) -> pl.DataFrame:
    # Check if the column is already numeric
    if df.schema['#CHROM'] == pl.Int64:
        result = df.rename({'#CHROM': 'CHR'})
    else:
        # If not numeric, try to remove 'chr' prefix and convert
        result = df.with_columns(
            pl.col('#CHROM')
            .str.replace('chr', '')  # Remove 'chr' prefix
            .cast(pl.Int64, strict=False)
        ).filter(
            pl.col('#CHROM').is_not_null()
        ).rename({'#CHROM': 'CHR'})

    return result

def validate_vcf_format_columns(columns: list, verbose: bool = False) -> dict:
    """
    Validate and describe VCF FORMAT columns according to GWAS-VCF specification.

    Args:
        columns (list): List of column names from VCF FORMAT
        verbose (bool, optional): If True, print detailed information about columns.
            Defaults to False.

    Returns:
        dict: Dictionary of validated columns with their descriptions and requirements
    """
    # Define the VCF FORMAT column specifications
    vcf_format_spec = {
        'NS': {
            'description': 'Variant-specific number of samples/individuals with called '
                         'genotypes used to test association with specified trait',
            'required': False
        },
        'EZ': {
            'description': 'Z-score provided if it was used to derive the ES and SE fields',
            'required': False
        },
        'SI': {
            'description': 'Accuracy score of association statistics imputation',
            'required': False
        },
        'NC': {
            'description': 'Variant-specific number of cases used to estimate genetic '
                         'effect (binary traits only)',
            'required': False
        },
        'ES': {
            'description': 'Effect size estimate relative to the alternative allele',
            'required': True
        },
        'SE': {
            'description': 'Standard error of effect size estimate',
            'required': True
        },
        'LP': {
            'description': '-log10 p-value for effect estimate',
            'required': True
        },
        'AF': {
            'description': 'Alternative allele frequency in trait subset',
            'required': False
        },
        'AC': {
            'description': 'Alternative allele count in the trait subset',
            'required': False
        }
    }

    # Check for missing required columns
    missing_required = [
        col for col, spec in vcf_format_spec.items()
        if spec['required'] and col not in columns
    ]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Check for extra columns not in specification
    extra_columns = [col for col in columns if col not in vcf_format_spec]
    if extra_columns:
        raise ValueError(
            f"Extra columns not in VCF FORMAT specification: {extra_columns}"
        )

    # If verbose, print detailed information
    if verbose:
        print("GWAS-VCF keys and descriptions:")
        for col in columns:
            spec = vcf_format_spec[col]
            req_status = 'Required' if spec['required'] else 'Optional'
            print(f"- {col}: {spec['description']} ({req_status})")

    # Return the specification for the found columns
    return {col: vcf_format_spec[col] for col in columns}


def read_gwas_vcf(
    file_path: str,
    num_rows: Optional[int] = None,
    maximum_missingness: float = 0.1,
    verbose: bool = False
) -> pl.DataFrame:
    """
    Reads a GWAS-VCF file using Polars and returns a DataFrame.

    Args:
        file_path (str): Path to the VCF file.
        num_rows (Optional[int], optional): Number of rows to read. Defaults to None.
        maximum_missingness (float, optional): Maximum fraction of missing samples
            allowed. Defaults to 0.1.
        verbose (bool, optional): Print detailed information about FORMAT columns.
            Defaults to False.

    Returns:
        pl.DataFrame: DataFrame containing the VCF data.
    """
    # Read the VCF file, skipping the header lines
    df = pl.read_csv(
        file_path,
        separator='\t',
        comment_prefix='##',
        has_header=True,
        n_rows=num_rows
    )

    df = split_sample_column(df)
    df = process_chromosome_column(df)

    # Filter based on missingness using NS (number of samples)
    if 'NS' in df.columns:
        min_samples = (1 - maximum_missingness) * pl.col('NS').max()
        df = df.filter(pl.col('NS') >= min_samples)

    return df
