"""Create metadata file for LDGM files."""

from pathlib import Path
from graphld.io import create_ldgm_metadata

# Get path to data directory
data_dir = Path(__file__).parent.parent / "data/ldgms"
output_file = data_dir / "metadata.csv"

# Create metadata
df = create_ldgm_metadata(data_dir, output_file)
print(f"\nCreated metadata for {len(df)} blocks:")
print(df)
