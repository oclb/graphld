"""Run BLUP on height summary statistics."""
RUN_SERIAL = False
CHROMOSOME = 22
MATCH_POSITION = True
POPULATION = "EUR"
SUMSTATS_PATH = "data/sumstats/height.hg38.vcf"
SAMPLE_SIZE = 100_000
from time import time
import polars as pl
from graphld import BLUP
from graphld.vcf_io import read_gwas_vcf

def main():
    sumstats = read_gwas_vcf(SUMSTATS_PATH)

    # Get median sample size if it varies
    num_snps = len(sumstats)
    if 'N' in sumstats.columns:
        sample_size = float(sumstats.select('N').mean().item())
    else:
        sample_size = SAMPLE_SIZE
    
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
        populations=POPULATION,
        chromosomes=CHROMOSOME,
        run_in_serial=RUN_SERIAL,
        match_by_position=MATCH_POSITION,
    )

    print(f"BLUP took {time() - t:.2f} seconds")

    print("\nBLUP results:")
    print(blup.head())

if __name__ == '__main__':
    main()
