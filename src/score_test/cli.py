"""CLI for score test functionality."""

import logging
import sys
from pathlib import Path

import click
import h5py
import numpy as np

try:
    from .convert_scores import convert_hdf5
    from .score_test import main as score_test_main
    from .score_test_io import (
        get_trait_groups,
        get_trait_names,
        load_variant_data,
        save_trait_groups,
    )
except ImportError:
    from convert_scores import convert_hdf5
    from score_test_io import (
        get_trait_groups,
        get_trait_names,
        load_variant_data,
        save_trait_groups,
    )

    from score_test import main as score_test_main


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli():
    """Score test for annotation enrichment analysis.

    Commands:
        (no command)  Run score test (default, legacy behavior)
        show          Display traits and meta-analyses in HDF5 file
        add-meta      Add a meta-analysis group
        rm            Remove a meta-analysis or trait
        mv            Rename a trait or meta-analysis
        convert       Convert variant-level to gene-level statistics
    """
    pass


@cli.command(name="test", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("variant_stats_hdf5", type=click.Path(exists=True))
@click.argument("output_fp", required=False, default=None)
@click.option(
    "-a",
    "--variant-annot-dir",
    "variant_annot_dir",
    type=click.Path(exists=True),
    help="Directory containing variant-level annotation files (.annot).",
)
@click.option(
    "-g",
    "--gene-annot-dir",
    "gene_annot_dir",
    type=click.Path(exists=True),
    help="Directory containing gene-level annotations to convert to variant-level.",
)
@click.option(
    "--random-genes",
    "random_genes",
    help="Comma-separated probabilities (0-1) for random gene-level annotations (e.g., '0.1,0.01').",
)
@click.option(
    "--random-variants",
    "random_variants",
    help="Comma-separated probabilities (0-1) for random variant-level annotations (e.g., '0.1,0.01').",
)
@click.option(
    "--gene-table",
    default="data/genes.tsv",
    type=click.Path(exists=True),
    help="Path to gene table TSV file (required for gene-level options).",
)
@click.option(
    "--nearest-weights",
    default="0.4,0.2,0.1,0.1,0.1,0.05,0.05",
    help="Comma-separated weights for k-nearest genes (for gene-level options).",
)
@click.option(
    "--annotations",
    help="Optional comma-separated list of specific annotation names to test.",
)
@click.option(
    "-n",
    "--name",
    "trait_name",
    help="Specific trait name to process from HDF5 file. If omitted, all traits are processed.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enable verbose output (log messages and results to console).",
)
@click.option(
    "--seed", type=int, default=None, help="Seed for generating random annotations."
)
@click.option(
    "--approximate",
    is_flag=True,
    help="Use approximate score test that does not correct for uncertainty in model parameters.",
)
def test(
    variant_stats_hdf5,
    output_fp,
    variant_annot_dir,
    gene_annot_dir,
    random_genes,
    random_variants,
    gene_table,
    nearest_weights,
    annotations,
    trait_name,
    verbose,
    seed,
    approximate,
):
    """Run score test for annotation enrichment (default command)."""
    # Call the original main function with all arguments
    ctx = click.Context(score_test_main)
    ctx.params = {
        "variant_stats_hdf5": variant_stats_hdf5,
        "output_fp": output_fp,
        "variant_annot_dir": variant_annot_dir,
        "gene_annot_dir": gene_annot_dir,
        "random_genes": random_genes,
        "random_variants": random_variants,
        "gene_table": gene_table,
        "nearest_weights": nearest_weights,
        "annotations": annotations,
        "trait_name": trait_name,
        "verbose": verbose,
        "seed": seed,
        "approximate": approximate,
    }
    ctx.invoke(score_test_main, **ctx.params)


@cli.command(name="convert", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("variant_stats_hdf5", type=click.Path(exists=True))
@click.argument("gene_stats_hdf5", type=click.Path())
@click.option(
    "--gene-table",
    default="data/genes.tsv",
    type=click.Path(exists=True),
    help="Path to gene table TSV file",
)
@click.option(
    "--nearest-weights",
    default="0.4,0.2,0.1,0.1,0.1,0.05,0.05",
    help="Comma-separated weights for k-nearest genes",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output")
def convert(variant_stats_hdf5, gene_stats_hdf5, gene_table, nearest_weights, verbose):
    """Convert variant-level statistics to gene-level statistics."""
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    weights = [float(w) for w in nearest_weights.split(",")]
    convert_hdf5(variant_stats_hdf5, gene_stats_hdf5, gene_table, weights)
    logging.info("Conversion complete!")


@cli.command(name="show", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("hdf5_file", type=click.Path(exists=True))
@click.option("-v", "--verbose", is_flag=True, help="Show detailed information")
def show(hdf5_file, verbose):
    """Display traits, meta-analyses, and datasets in HDF5 file."""
    with h5py.File(hdf5_file, "r") as f:
        # Show traits
        trait_names = get_trait_names(hdf5_file)
        click.echo(f"\n{'=' * 60}")
        click.echo(f"HDF5 File: {hdf5_file}")
        click.echo(f"{'=' * 60}")

        click.echo(f"\nTraits ({len(trait_names)}):")
        for trait in sorted(trait_names):
            click.echo(f"  - {trait}")

        # Show meta-analyses
        groups = get_trait_groups(hdf5_file)
        if groups:
            click.echo(f"\nMeta-analyses ({len(groups)}):")
            for group_name, traits in sorted(groups.items()):
                click.echo(f"  - {group_name}: {', '.join(traits)}")
        else:
            click.echo("\nMeta-analyses: None defined")

        # Show variant datasets (check both row_data and variants for backward compatibility)
        data_group = (
            "row_data" if "row_data" in f else "variants" if "variants" in f else None
        )
        if data_group:
            click.echo(f"\nVariant datasets:")
            for dataset_name in sorted(f[data_group].keys()):
                dataset = f[data_group][dataset_name]
                if hasattr(dataset, "shape"):
                    click.echo(
                        f"  - {dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}"
                    )
                else:
                    click.echo(f"  - {dataset_name}: (group)")

        # Show detailed trait information if verbose
        if verbose and "traits" in f:
            click.echo(f"\nDetailed trait information:")
            for trait_name in sorted(trait_names):
                trait_group = f["traits"][trait_name]
                click.echo(f"\n  {trait_name}:")
                for dataset_name in trait_group.keys():
                    dataset = trait_group[dataset_name]
                    if hasattr(dataset, "shape"):
                        click.echo(
                            f"    - {dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}"
                        )


@cli.command(name="add-meta", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("hdf5_file", type=click.Path(exists=True))
@click.argument("meta_name")
@click.argument("traits", nargs=-1, required=True)
def add_meta(hdf5_file, meta_name, traits):
    """Add a meta-analysis group to HDF5 file.

    Example: estest add-meta scores.h5 body height bmi
    Example: estest add-meta scores.h5 all '*'
    """
    # Load existing traits and groups
    existing_traits = get_trait_names(hdf5_file)
    existing_groups = get_trait_groups(hdf5_file)

    # Validate meta_name is not a trait name
    if meta_name in existing_traits:
        click.echo(
            f"Error: '{meta_name}' is already a trait name. Choose a different name.",
            err=True,
        )
        sys.exit(1)

    # Match glob patterns and collect all matching traits
    import fnmatch

    matched_traits = set()
    for pattern in traits:
        matches = [t for t in existing_traits if fnmatch.fnmatch(t, pattern)]
        if not matches:
            click.echo(
                f"Error: Trait(s) matching pattern '{pattern}' do not exist", err=True
            )
            click.echo(
                f"Available traits: {', '.join(sorted(existing_traits))}", err=True
            )
            sys.exit(1)
        matched_traits.update(matches)

    # Convert to sorted list for consistency
    matched_traits = sorted(matched_traits)

    # Require at least 2 traits
    if len(matched_traits) < 2:
        click.echo("Error: Meta-analysis requires at least 2 traits.", err=True)
        sys.exit(1)

    # Add the new group
    existing_groups[meta_name] = sorted(matched_traits)
    save_trait_groups(hdf5_file, existing_groups)

    click.echo(
        f"Added meta-analysis '{meta_name}' with {len(matched_traits)} traits: {', '.join(matched_traits)}"
    )


@cli.command(name="rm", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("hdf5_file", type=click.Path(exists=True))
@click.argument("patterns", nargs=-1, required=True)
@click.option("-f", "--force", is_flag=True, help="Force removal without confirmation")
def rm(hdf5_file, patterns, force):
    """Remove meta-analyses or traits from HDF5 file.

    Supports wildcards (e.g., '*', 'BMI*', '*_EAS').
    Auto-detects whether names are traits or meta-analyses.

    Examples:
        estest rm data.h5 body                # Remove specific meta-analysis or trait
        estest rm data.h5 '*_EAS' -f          # Remove all EAS traits
        estest rm data.h5 bmi height cancer   # Remove multiple items
    """
    import fnmatch

    groups = get_trait_groups(hdf5_file)
    existing_traits = get_trait_names(hdf5_file)

    # Match patterns against traits and meta-analyses
    matched_metas = set()
    matched_traits = set()

    for pattern in patterns:
        # Try exact match first
        if pattern in groups:
            matched_metas.add(pattern)
        elif pattern in existing_traits:
            matched_traits.add(pattern)
        else:
            # Try wildcard matching
            meta_matches = [m for m in groups.keys() if fnmatch.fnmatch(m, pattern)]
            trait_matches = [t for t in existing_traits if fnmatch.fnmatch(t, pattern)]

            if not meta_matches and not trait_matches:
                click.echo(
                    f"Error: No traits or meta-analyses match pattern '{pattern}'",
                    err=True,
                )
                sys.exit(1)

            matched_metas.update(meta_matches)
            matched_traits.update(trait_matches)

    # Confirm removal if not forced
    if not force:
        if matched_metas:
            click.echo(f"\nMeta-analyses to remove ({len(matched_metas)}):")
            for meta in sorted(matched_metas):
                click.echo(f"  - {meta}: {', '.join(groups[meta])}")
        if matched_traits:
            click.echo(f"\nTraits to remove ({len(matched_traits)}):")
            for trait in sorted(matched_traits):
                click.echo(f"  - {trait}")

        if not click.confirm(
            f"\nRemove {len(matched_metas)} meta-analyses and {len(matched_traits)} traits?"
        ):
            click.echo("Cancelled.")
            return

    # Remove meta-analyses
    if matched_metas:
        for meta in matched_metas:
            del groups[meta]
        save_trait_groups(hdf5_file, groups)
        click.echo(
            f"Removed {len(matched_metas)} meta-analysis(es): {', '.join(sorted(matched_metas))}"
        )

    # Remove traits
    if matched_traits:
        # Remove traits from HDF5
        with h5py.File(hdf5_file, "a") as f:
            if "traits" in f:
                for trait in matched_traits:
                    if trait in f["traits"]:
                        del f["traits"][trait]

        # Remove traits from any meta-analyses that reference them
        groups = get_trait_groups(hdf5_file)
        groups_to_remove = set()
        updated = False

        for group_name, group_traits in list(groups.items()):
            # Remove matched traits from this group
            original_len = len(group_traits)
            group_traits = [t for t in group_traits if t not in matched_traits]

            if len(group_traits) != original_len:
                updated = True
                if len(group_traits) < 2:
                    # Mark for removal if insufficient traits remain
                    groups_to_remove.add(group_name)
                else:
                    groups[group_name] = group_traits

        # Remove groups with insufficient traits
        for group_name in groups_to_remove:
            del groups[group_name]
            if not force:  # Only show message if not using -f
                click.echo(
                    f"  Also removed meta-analysis '{group_name}' (insufficient traits)"
                )

        if updated:
            save_trait_groups(hdf5_file, groups)

        click.echo(
            f"Removed {len(matched_traits)} trait(s): {', '.join(sorted(matched_traits))}"
        )


@cli.command(name="mv", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("hdf5_file", type=click.Path(exists=True))
@click.argument("old_name")
@click.argument("new_name")
def mv(hdf5_file, old_name, new_name):
    """Rename a trait or meta-analysis in HDF5 file.

    Auto-detects whether old_name is a trait or meta-analysis.
    """
    groups = get_trait_groups(hdf5_file)
    existing_traits = get_trait_names(hdf5_file)

    is_meta = old_name in groups
    is_trait = old_name in existing_traits

    if not is_meta and not is_trait:
        click.echo(
            f"Error: '{old_name}' not found as trait or meta-analysis.", err=True
        )
        sys.exit(1)

    if is_meta:
        # Rename meta-analysis
        groups = get_trait_groups(hdf5_file)
        if old_name not in groups:
            click.echo(f"Error: Meta-analysis '{old_name}' not found.", err=True)
            click.echo(f"Available meta-analyses: {', '.join(sorted(groups.keys()))}")
            sys.exit(1)

        # Check if new name already exists as a trait or meta-analysis
        existing_traits = get_trait_names(hdf5_file)
        if new_name in existing_traits:
            click.echo(
                f"Error: '{new_name}' is already a trait name. Choose a different name.",
                err=True,
            )
            sys.exit(1)

        if new_name in groups:
            click.echo(f"Error: Meta-analysis '{new_name}' already exists.", err=True)
            sys.exit(1)

        # Rename: copy and delete old
        groups[new_name] = groups[old_name]
        del groups[old_name]
        save_trait_groups(hdf5_file, groups)
        click.echo(f"Renamed meta-analysis '{old_name}' to '{new_name}'")

    elif is_trait:
        # Rename trait
        existing_traits = get_trait_names(hdf5_file)
        if old_name not in existing_traits:
            click.echo(f"Error: Trait '{old_name}' not found.", err=True)
            click.echo(f"Available traits: {', '.join(sorted(existing_traits))}")
            sys.exit(1)

        if new_name in existing_traits:
            click.echo(f"Error: Trait '{new_name}' already exists.", err=True)
            sys.exit(1)

        # Check if new name conflicts with meta-analysis
        groups = get_trait_groups(hdf5_file)
        if new_name in groups:
            click.echo(
                f"Error: '{new_name}' is already a meta-analysis name. Choose a different name.",
                err=True,
            )
            sys.exit(1)

        # Rename trait in HDF5
        with h5py.File(hdf5_file, "a") as f:
            if "traits" in f and old_name in f["traits"]:
                # Copy the group to new name
                f.copy(f["traits"][old_name], f["traits"], name=new_name)
                # Delete the old group
                del f["traits"][old_name]

        # Update meta-analyses that reference this trait
        updated = False
        for group_name, traits in groups.items():
            if old_name in traits:
                idx = traits.index(old_name)
                traits[idx] = new_name
                groups[group_name] = traits
                updated = True
                click.echo(f"  Updated meta-analysis '{group_name}'")

        if updated:
            save_trait_groups(hdf5_file, groups)

        click.echo(f"Renamed trait '{old_name}' to '{new_name}'")


def main():
    """Main entry point that handles both subcommands and legacy direct invocation."""
    # Check if we're being called with subcommand syntax or help
    if len(sys.argv) > 1 and sys.argv[1] in [
        "test",
        "convert",
        "show",
        "add-meta",
        "rm",
        "mv",
        "--help",
        "-h",
    ]:
        # Use the new CLI with subcommands
        cli()
    else:
        # Legacy behavior: direct invocation runs the test command
        # Remove the program name and call score_test_main directly
        score_test_main()


if __name__ == "__main__":
    main()
