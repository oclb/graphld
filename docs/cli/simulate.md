# Simulation (`simulate`)

Use `graphld simulate` to generate GWAS summary statistics from the GraphLD simulation model.

## Basic Usage

```bash
uv run graphld simulate \
    output.sumstats \
    --heritability 0.5 \
    --component-variance 1,10 \
    --component-weight 0.01,0.001
```

Unlike the other `graphld` subcommands, `simulate` takes an output summary-statistics path as its positional argument rather than an input sumstats file.

## Core Options

| Option | Default | Description |
|--------|---------|-------------|
| `-H, --heritability` | `0.2` | Total trait heritability |
| `--component-variance` | `1.0` | Comma-separated mixture variances |
| `--component-weight` | `1.0` | Comma-separated mixture weights |
| `--alpha-param` | `-0.5` | Frequency-dependence parameter |
| `--random-seed` | `None` | Random seed |

## Annotation-Dependent Simulation

Use annotation-dependent effect-size scaling with:

| Option | Description |
|--------|-------------|
| `-a, --annot-dir` | Annotation directory |
| `--annotation-columns` | Comma-separated annotation columns |
| `--annotation-dependent-polygenicity` | Reserved for future support; currently raises `NotImplementedError` |

The shared options from the [CLI overview](../cli.md) also apply.
