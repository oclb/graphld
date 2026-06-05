"""BED file parsing and region annotation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl


def _get_range_mask(values: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    result = np.zeros_like(values, dtype=bool)
    if len(end) == 0:
        return result

    order = np.lexsort((end, start))
    start = start[order]
    end = end[order]

    merged_start = []
    merged_end = []
    for interval_start, interval_end in zip(start, end, strict=False):
        if not merged_start or interval_start > merged_end[-1]:
            merged_start.append(interval_start)
            merged_end.append(interval_end)
        else:
            merged_end[-1] = max(merged_end[-1], interval_end)

    start = np.asarray(merged_start)
    end = np.asarray(merged_end)
    range_idx = np.searchsorted(start, values, side="right") - 1
    valid = range_idx >= 0
    result[valid] = values[valid] < end[range_idx[valid]]

    return result


def list_bed_files(annot_path: str | Path) -> list[Path]:
    """Return BED files in an annotation directory in stable order."""
    return sorted(Path(annot_path).glob("*.bed"))


def add_bed_annotations(
    annotations: pl.DataFrame,
    bed_files: Iterable[str | Path],
    *,
    position_col: str = "POS",
) -> pl.DataFrame:
    """Add one Boolean annotation column per BED file to variant annotations."""
    bed_annotations = {}
    for bed_file in bed_files:
        bed_file = Path(bed_file)
        bed_df = read_bed(str(bed_file)).with_columns(
            pl.col("chrom").str.replace("chr", "").cast(pl.Int64).alias("chrom")
        )

        new_annot = np.zeros(len(annotations), dtype=bool)
        unique_chromosomes = annotations.get_column("CHR").unique()
        for chrom in unique_chromosomes:
            chrom_indices = (annotations["CHR"] == chrom).to_numpy()
            if not chrom_indices.any():
                continue
            bed_regions = (
                bed_df.filter(bed_df["chrom"] == chrom)
                .select("chromStart", "chromEnd")
                .to_numpy()
            )
            positions = annotations.filter(annotations["CHR"] == chrom)[position_col].to_numpy()
            new_annot[chrom_indices] = _get_range_mask(
                values=positions,
                start=bed_regions[:, 0],
                end=bed_regions[:, 1],
            )

        bed_annotations[bed_file.stem] = new_annot

    if not bed_annotations:
        return annotations
    return annotations.with_columns(**bed_annotations)


def read_bed(
    bed_file: str,
    min_fields: int = 3,
    max_fields: int = 12,
    zero_based: bool = True,
) -> pl.DataFrame:
    """Read a UCSC BED format file.

    The BED format has 3 required fields and 9 optional fields:
    Required:
        1. chrom - Chromosome name
        2. chromStart - Start position (0-based)
        3. chromEnd - End position (not included in feature)
    Optional:
        4. name - Name of BED line
        5. score - Score from 0-1000
        6. strand - Strand: "+" or "-" or "."
        7. thickStart - Starting position at which feature is drawn thickly
        8. thickEnd - Ending position at which feature is drawn thickly
        9. itemRgb - RGB value (e.g., "255,0,0")
        10. blockCount - Number of blocks (e.g., exons)
        11. blockSizes - Comma-separated list of block sizes
        12. blockStarts - Comma-separated list of block starts relative to chromStart

    Args:
        bed_file: Path to BED format file
        min_fields: Minimum number of fields required (default: 3)
        max_fields: Maximum number of fields to read (default: 12)
        zero_based: If True (default), keeps positions 0-based. If False, adds 1 to start positions.

    Returns:
        Polars DataFrame containing the BED data with appropriate column names and types.

    Raises:
        ValueError: If min_fields < 3 or max_fields > 12 or if file has
            inconsistent number of fields
    """
    if min_fields < 3:
        raise ValueError("BED format requires at least 3 fields")
    if max_fields > 12:
        raise ValueError("BED format has at most 12 fields")
    if min_fields > max_fields:
        raise ValueError("min_fields cannot be greater than max_fields")

    bed_columns = [
        ("chrom", pl.Utf8),
        ("chromStart", pl.Int64),
        ("chromEnd", pl.Int64),
        ("name", pl.Utf8),
        ("score", pl.Int64),
        ("strand", pl.Utf8),
        ("thickStart", pl.Int64),
        ("thickEnd", pl.Int64),
        ("itemRgb", pl.Utf8),
        ("blockCount", pl.Int64),
        ("blockSizes", pl.Utf8),
        ("blockStarts", pl.Utf8),
    ]

    data = []
    with open(bed_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(("browser", "track", "#")):
                continue
            fields = [field for field in line.split() if field]
            if not (min_fields <= len(fields) <= max_fields):
                raise ValueError(
                    f"BED line has {len(fields)} fields, expected between "
                    f"{min_fields} and {max_fields}: {line}"
                )
            fields.extend([None] * (max_fields - len(fields)))
            data.append(fields[:max_fields])

    schema = {name: dtype for name, dtype in bed_columns[:max_fields]}
    df = pl.from_records(data, schema=schema, orient="row")

    if not zero_based:
        df = df.with_columns([pl.col("chromStart") + 1])
        if "thickStart" in df.columns:
            df = df.with_columns([pl.col("thickStart") + 1])

    return df
