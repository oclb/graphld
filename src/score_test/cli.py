"""CLI for score test functionality."""

import sys
import logging
import click
import h5py
import numpy as np
from pathlib import Path

try:
    from .score_test import main as score_test_main
    from .score_test_io import (
        get_trait_names,
        get_trait_groups,
        save_trait_groups,
        load_variant_data,
    )
    from .convert_scores import convert_hdf5
except ImportError:
    from score_test import main as score_test_main
    from score_test_io import (
        get_trait_names,
        get_trait_groups,
        save_trait_groups,
        load_variant_data,
    )
    from convert_scores import convert_hdf5


@click.group(context_settings={'help_option_names': ['-h', '--help']})
def cli():
    """Score test for annotation enrichment analysis.
    
    Commands:
        (no command)  Run score test (default, legacy behavior)
        show          Display traits and meta-analyses in HDF5 file
        add-meta      Add a meta-analysis group
        rm            Remove a meta-analysis or trait
        convert       Convert variant-level to gene-level statistics
    """
    pass


@cli.command(name='test', context_settings={'help_option_names': ['-h', '--help']})
@click.argument('variant_stats_hdf5', type=click.Path(exists=True))
@click.argument('output_fp', required=False, default=None)
@click.option('-a', '--variant-annot-dir', 'variant_annot_dir', type=click.Path(exists=True),
              help="Directory containing variant-level annotation files (.annot).")
@click.option('-g', '--gene-annot-dir', 'gene_annot_dir', type=click.Path(exists=True),
              help="Directory containing gene-level annotations to convert to variant-level.")
@click.option('--random-genes', 'random_genes',
              help="Comma-separated probabilities (0-1) for random gene-level annotations (e.g., '0.1,0.01').")
@click.option('--random-variants', 'random_variants',
              help="Comma-separated probabilities (0-1) for random variant-level annotations (e.g., '0.1,0.01').")
@click.option('--gene-table', default='data/genes.tsv', type=click.Path(exists=True),
              help="Path to gene table TSV file (required for gene-level options).")
@click.option('--nearest-weights', default='0.4,0.2,0.1,0.1,0.1,0.05,0.05',
              help="Comma-separated weights for k-nearest genes (for gene-level options).")
@click.option('--annotations',
              help="Optional comma-separated list of specific annotation names to test.")
@click.option('-n', '--name', 'trait_name',
              help="Specific trait name to process from HDF5 file. If omitted, all traits are processed.")
@click.option('-v', '--verbose', is_flag=True,
              help='Enable verbose output (log messages and results to console).')
@click.option('--seed', type=int, default=None,
              help='Seed for generating random annotations.')
@click.option('--approximate', is_flag=True,
              help='Use approximate score test that does not correct for uncertainty in model parameters.')
def test(variant_stats_hdf5, output_fp, variant_annot_dir, gene_annot_dir, random_genes,
         random_variants, gene_table, nearest_weights, annotations, trait_name, verbose, seed, approximate):
    """Run score test for annotation enrichment (default command)."""
    # Call the original main function with all arguments
    ctx = click.Context(score_test_main)
    ctx.params = {
        'variant_stats_hdf5': variant_stats_hdf5,
        'output_fp': output_fp,
        'variant_annot_dir': variant_annot_dir,
        'gene_annot_dir': gene_annot_dir,
        'random_genes': random_genes,
        'random_variants': random_variants,
        'gene_table': gene_table,
        'nearest_weights': nearest_weights,
        'annotations': annotations,
        'trait_name': trait_name,
        'verbose': verbose,
        'seed': seed,
        'approximate': approximate,
    }
    ctx.invoke(score_test_main, **ctx.params)


@cli.command(name='convert', context_settings={'help_option_names': ['-h', '--help']})
@click.argument('variant_stats_hdf5', type=click.Path(exists=True))
@click.argument('gene_stats_hdf5', type=click.Path())
@click.option('--gene-table', default='data/genes.tsv', type=click.Path(exists=True),
              help="Path to gene table TSV file")
@click.option('--nearest-weights', default='0.4,0.2,0.1,0.1,0.1,0.05,0.05',
              help="Comma-separated weights for k-nearest genes")
@click.option('-v', '--verbose', is_flag=True,
              help='Enable verbose output')
def convert(variant_stats_hdf5, gene_stats_hdf5, gene_table, nearest_weights, verbose):
    """Convert variant-level statistics to gene-level statistics."""
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    weights = [float(w) for w in nearest_weights.split(',')]
    convert_hdf5(variant_stats_hdf5, gene_stats_hdf5, gene_table, weights)
    logging.info("Conversion complete!")


@cli.command(name='show', context_settings={'help_option_names': ['-h', '--help']})
@click.argument('hdf5_file', type=click.Path(exists=True))
@click.option('-v', '--verbose', is_flag=True, help='Show detailed information')
def show(hdf5_file, verbose):
    """Display traits, meta-analyses, and datasets in HDF5 file."""
    with h5py.File(hdf5_file, 'r') as f:
        # Show traits
        trait_names = get_trait_names(hdf5_file)
        click.echo(f"\n{'='*60}")
        click.echo(f"HDF5 File: {hdf5_file}")
        click.echo(f"{'='*60}")
        
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
        
        # Show variant datasets
        if 'variants' in f:
            click.echo(f"\nVariant datasets:")
            for dataset_name in sorted(f['variants'].keys()):
                dataset = f['variants'][dataset_name]
                if hasattr(dataset, 'shape'):
                    click.echo(f"  - {dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}")
                else:
                    click.echo(f"  - {dataset_name}: (group)")
        
        # Show detailed trait information if verbose
        if verbose and 'traits' in f:
            click.echo(f"\nDetailed trait information:")
            for trait_name in sorted(trait_names):
                trait_group = f['traits'][trait_name]
                click.echo(f"\n  {trait_name}:")
                for dataset_name in trait_group.keys():
                    dataset = trait_group[dataset_name]
                    if hasattr(dataset, 'shape'):
                        click.echo(f"    - {dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}")


@cli.command(name='add-meta', context_settings={'help_option_names': ['-h', '--help']})
@click.argument('hdf5_file', type=click.Path(exists=True))
@click.argument('meta_name')
@click.argument('traits', nargs=-1, required=True)
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
        click.echo(f"Error: '{meta_name}' is already a trait name. Choose a different name.", err=True)
        sys.exit(1)
    
    # Match glob patterns and collect all matching traits
    import fnmatch
    matched_traits = set()
    for pattern in traits:
        matches = [t for t in existing_traits if fnmatch.fnmatch(t, pattern)]
        if not matches:
            click.echo(f"Error: No traits match pattern '{pattern}'", err=True)
            click.echo(f"Available traits: {', '.join(sorted(existing_traits))}")
            sys.exit(1)
        matched_traits.update(matches)
    
    # Require at least 2 traits
    if len(matched_traits) < 2:
        click.echo("Error: Meta-analysis requires at least 2 traits.", err=True)
        sys.exit(1)
    
    # Add the new group
    existing_groups[meta_name] = sorted(matched_traits)
    save_trait_groups(hdf5_file, existing_groups)
    
    click.echo(f"Added meta-analysis '{meta_name}' with {len(matched_traits)} traits: {', '.join(matched_traits)}")


@cli.command(name='rm', context_settings={'help_option_names': ['-h', '--help']})
@click.argument('hdf5_file', type=click.Path(exists=True))
@click.argument('name')
@click.option('--meta', is_flag=True, help='Remove a meta-analysis group (default)')
@click.option('--trait', is_flag=True, help='Remove a trait')
@click.option('-f', '--force', is_flag=True, help='Force removal without confirmation')
def rm(hdf5_file, name, meta, trait, force):
    """Remove a meta-analysis or trait from HDF5 file.
    
    Examples:
        estest rm data.h5 body --meta     # Remove meta-analysis 'body'
        estest rm data.h5 bmi --trait     # Remove trait 'bmi'
    """
    # Default to removing meta-analysis if neither specified
    if not meta and not trait:
        meta = True
    
    if meta and trait:
        click.echo("Error: Cannot specify both --meta and --trait", err=True)
        sys.exit(1)
    
    if meta:
        # Remove meta-analysis
        groups = get_trait_groups(hdf5_file)
        if name not in groups:
            click.echo(f"Error: Meta-analysis '{name}' not found.", err=True)
            click.echo(f"Available meta-analyses: {', '.join(sorted(groups.keys()))}")
            sys.exit(1)
        
        if not force:
            traits_str = ', '.join(groups[name])
            if not click.confirm(f"Remove meta-analysis '{name}' ({traits_str})?"):
                click.echo("Cancelled.")
                return
        
        del groups[name]
        save_trait_groups(hdf5_file, groups)
        click.echo(f"Removed meta-analysis '{name}'")
    
    elif trait:
        # Remove trait
        existing_traits = get_trait_names(hdf5_file)
        if name not in existing_traits:
            click.echo(f"Error: Trait '{name}' not found.", err=True)
            click.echo(f"Available traits: {', '.join(sorted(existing_traits))}")
            sys.exit(1)
        
        if not force:
            if not click.confirm(f"Remove trait '{name}' and all its data?"):
                click.echo("Cancelled.")
                return
        
        # Remove trait from HDF5
        with h5py.File(hdf5_file, 'a') as f:
            if 'traits' in f and name in f['traits']:
                del f['traits'][name]
        
        # Also remove from any meta-analyses that reference it
        groups = get_trait_groups(hdf5_file)
        updated = False
        for group_name, traits in list(groups.items()):
            if name in traits:
                traits.remove(name)
                if len(traits) < 2:
                    # Remove meta-analysis if it has fewer than 2 traits
                    del groups[group_name]
                    click.echo(f"  Also removed meta-analysis '{group_name}' (insufficient traits)")
                else:
                    groups[group_name] = traits
                    click.echo(f"  Removed '{name}' from meta-analysis '{group_name}'")
                updated = True
        
        if updated:
            save_trait_groups(hdf5_file, groups)
        
        click.echo(f"Removed trait '{name}'")


def main():
    """Main entry point that handles both subcommands and legacy direct invocation."""
    # Check if we're being called with subcommand syntax or help
    if len(sys.argv) > 1 and sys.argv[1] in ['test', 'convert', 'show', 'add-meta', 'rm', '--help', '-h']:
        # Use the new CLI with subcommands
        cli()
    else:
        # Legacy behavior: direct invocation runs the test command
        # Remove the program name and call score_test_main directly
        score_test_main()


if __name__ == '__main__':
    main()
