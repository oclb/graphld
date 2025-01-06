"""Run BLUP on height summary statistics."""
RUN_SERIAL = False
CHROMOSOME = None

from time import time
import polars as pl
from graphld import BLUP

def main():
    # Load summary statistics
    sumstats = pl.read_csv(
        "data/sumstats/body_HEIGHTz.sumstats",
        separator="\t"
    )

    # Convert Beta/SE to Z-scores if not already present
    if 'Z' not in sumstats.columns:
        sumstats = sumstats.with_columns(
            (pl.col('Beta') / pl.col('se')).alias('Z')
        )
    # Map column names
    sumstats = sumstats.with_columns([
        pl.col('A1').alias('ALT'),  # Effect allele
        pl.col('A2').alias('REF'),  # Reference allele
    ])

    # Ensure required columns are present
    required_cols = ['SNP', 'CHR', 'POS', 'REF', 'ALT', 'Z', 'N']
    missing_cols = [col for col in required_cols if col not in sumstats.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Get median sample size if it varies
    num_snps = len(sumstats)
    sample_size = float(sumstats.select('N').mean().item())

    # Compute per-SNP variance based on total heritability
    total_h2 = 0.5  # expected heritability for height
    sigmasq = total_h2 / num_snps

    # Run BLUP
    print(f"Running BLUP with {num_snps:,} variants")
    print(f"Sample size: {sample_size:,.0f}")
    print(f"Per-SNP variance: {sigmasq:.2e}")

    t = time()
    blup = BLUP.compute_blup(
        ldgm_metadata_path="data/ldgms/metadata.csv",
        sumstats=sumstats,
        sigmasq=sigmasq,
        sample_size=sample_size,
        populations="EUR",
        chromosomes=CHROMOSOME,
        run_in_serial=RUN_SERIAL,
        match_by_position=False,
    )

    print(f"BLUP took {time() - t:.2f} seconds")

    print("\nBLUP results:")
    print(blup.head())

if __name__ == '__main__':
    main()
