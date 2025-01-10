#!/usr/bin/env bash

# Print the single header
echo "chrom,site_ids,position"

# Loop over each *.snplist file
for file in ldgms/*.snplist; do
    # Extract the chromosome number/name from the filename
    # e.g., 1kg_chr10_100331627_104378781.snplist -> 10
    chrom=$(basename "$file" | sed -E 's/.*chr([^_]+)_.*/\1/')

    # Use awk to:
    # - skip the file's first line (header)
    # - skip lines where site_ids == "NA"
    # - print CHR,site_ids,position
    awk -v CHR="$chrom" '
        BEGIN { FS = "," }    # split on commas
        NR == 1 { next }      # skip header line of each file
        $9 != "NA" {
            print CHR "," $9 "," $10
        }
    ' "$file"
done
