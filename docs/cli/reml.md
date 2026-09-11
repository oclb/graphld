# graphREML (`reml`)

Use `graphld reml` for heritability partitioning and annotation enrichment estimation.

## Basic Usage

```bash
uv run graphld reml \
    /path/to/sumstats.sumstats \
    output_prefix \
    --annot-dir /path/to/annotations/
```

You must provide one annotation source:

- `--annot-dir` for variant or region annotations
- `--gene-annot-dir` for GMT gene sets

## Input Types

- Summary statistics: LDSC-style `.sumstats`, GWAS-VCF `.vcf`/`.vcf.gz`, or kodama-style `.parquet`; see [Summary Statistics](../file_formats.md#summary-statistics).
- Variant annotations: per-chromosome `.annot` files, optionally alongside `.bed` files; see [Annotations](../file_formats.md#annotations).
- Gene annotations: `.gmt` files converted to variant-level annotations with nearest-gene weighting; see [GMT Format](../file_formats.md#gmt-format-gmt).

## Optimizer and optional Firth approximation

Safeguarded BFGS is the default optimizer. Use `--optimizer ai` to use average-information
proposals in the same line search, trace-correction, and precise-stopping framework.
Both choices evaluate actual objective values before accepting a step. A predictor
change cap limits extreme proposals. The primary proposal is retained when its
predicted gain is numerically resolvable and reaches at least 10% of the best
capped reference proposal's gain. BFGS uses AI and scaled-gradient references;
AI uses a scaled-gradient reference. Otherwise the most productive reference
supplies the direction for the same line search. Precise inverse-diagonal audits refresh the
fixed-probe correction and are required before ordinary convergence is reported.
Boundary-profile acceptance is not part of either optimizer.

The current development benchmarks do not support releasing BFGS as the default:
it remained nonstationary on Weight and was slower than AI on BMI. See the
[methods and evaluation write-up](../methods/graphreml_optimization/README.md)
for convergence, runtime, and enrichment results.

```bash
graphld reml summary.sumstats output --annot-dir annotations --optimizer bfgs
graphld reml summary.sumstats output_firth --annot-dir annotations --firth
```

`--firth` enables the average-information-based approximation: one half the log
determinant of the globally summed average-information matrix is added to the
likelihood. It is off by default. The equivalent `--information-penalty` option
also accepts a nonnegative weight, for example `--information-penalty 0.5`.
This is an information penalty; formal finite-sample bias reduction has not been
established for this approximation. Its ability to retain information in depleted
variance directions requires empirical validation.

The default `--penalty-trial-strategy exact` evaluates the penalty at every trial.
The experimental `linear_screen` strategy first checks the likelihood plus a
linear penalty prediction; `likelihood_screen` checks likelihood alone. Accepted
steps always pass the exact penalized objective test. After two consecutive
screen rejections, or an exhausted screened search, the original proposals are
retried exactly. Final signed stopping checks bypass screening. Logs count
screen rejections, exact fallbacks, detected false rejections, and counterfactual
linear-screen rejections among exactly evaluated proposals. These diagnostics
cover evaluated proposals, not every proposal on an alternative trajectory.

The penalty uses the global information inverse in every block gradient and
requires positive-definite global information at initialization. Use finite,
nonsaturated starting coefficients. Its determinant is computed without a ridge
or eigenvalue truncation; numerical regularization applies only to proposals.

### Migration from likelihood-window stopping

The defaults are now 100 iterations and `--convergence-tol 0.001`. The tolerance
bounds the precise information-scaled score criterion and actual improvements
in four signed direction checks, in objective units. Previously it controlled
recent likelihood changes. Small accepted changes trigger an audit and do not
by themselves establish convergence. Explicit iteration budgets and tolerances
remain respected. The compatibility arguments `--convergence-window` and
`--reset-trust-region`, and the Python damping settings, no longer control the
new line search. `--optimizer ai` selects the new AI framework.

A converged status certifies the reported score and tested-direction criteria;
it does not certify global optimality. Iteration limits, stalled searches, and
singular information have distinct statuses. Inspect these statuses when
comparing runtime or enrichment estimates.

The default `--link-function softplus` includes its first and second
derivatives. `--link-function exponential` is also available. In Python, set
`ModelOptions(link_function="softplus")` and
`MethodOptions(optimizer="bfgs", information_penalty=1.0, penalty_trial_strategy="exact")`.
A custom link factory may be passed as `ModelOptions(link_function=factory)`;
it takes the denominator and returns `(value, gradient, second_derivative)`.
Each function accepts `(annotations, parameters)`. For scalar annotation `1`,
the derivative functions return derivatives with respect to the linear predictor;
for a matrix, they return the corresponding parameter derivatives (the
second-derivative function returns the diagonal second derivatives). Use a module-level factory
so it can be passed to spawned workers.

Optimization uses the penalized score with the selected BFGS or AI proposal
metric. Average information supplies the stopping scale; the penalty Hessian is
omitted. Reported uncertainties retain the
existing unpenalized-information pseudo-jackknife approximation; their calibration
for penalized estimates has not been evaluated. The result log records the
penalty weight, trial strategy, final likelihood and penalized objective,
evaluation counts/times, optimizer status, selected proposal and cap multiplier,
metric resets, and uncertainty method/status.
Finite well-conditioned endpoints retain that approximation; unresolved endpoints
are marked provisional, and singular or invalid delete calculations are unavailable.
Pseudo-jackknife updates use diagonally normalized Cholesky solves of unregularized
information, so changing coefficient units preserves the updates.
Optimization time includes final precise refreshes; total time also includes input
preparation and post-fit calculations. Convergence CSV files from both integrated
optimizers include optimizer status, likelihood, penalty, and objective histories,
and evaluation counts and times. Detailed proposal diagnostics are in the result log.
The gradient pass retains per-block information intermediates and sparse factors
until the global information inverse is available, increasing peak worker memory.

## Output Files

Default output:

- `output_prefix.tall.csv`: heritability, enrichment, and coefficient estimates
- `output_prefix.convergence.csv`: optimization diagnostics

With `--alt-output`:

- `output_prefix.heritability.csv`
- `output_prefix.enrichment.csv`
- `output_prefix.parameters.csv`

Use `--name` to label runs when appending to alternate output files or score-test HDF5 outputs.

To create score-test derivatives for a new trait, pass `--score-test-filename`.
See [Creating Derivatives For A New Trait](../score_test.md#creating-derivatives-for-a-new-trait)
for the full workflow, including the UKBB population setting used by the
downloaded European score file.

## Common Options

| Option | Default | Description |
|--------|---------|-------------|
| `--intercept` | `1.0` | LD score regression intercept |
| `--name` | `None` | Run label for outputs and score-test artifacts |
| `--metadata` | `data/ldgms/metadata.csv` | LDGM metadata CSV |
| `--score-test-filename` | `None` | HDF5 file for score-test precomputation |
| `--surrogates` | `None` | Precomputed surrogate-marker HDF5 |
| `--no-save` | `False` | Skip result-file writing or write logs only |

## Optimization Options

| Option | Default | Description |
|--------|---------|-------------|
| `--num-iterations` | `100` | Maximum optimization iterations |
| `--convergence-tol` | `0.001` | Precise score and signed objective tolerance |
| `--convergence-window` | `3` | Compatibility argument |
| `--num-jackknife-blocks` | `100` | Jackknife blocks for standard errors |
| `--xtrace-num-samples` | `100` | Samples for stochastic gradient estimation |
| `--reset-trust-region` | `False` | Compatibility argument |
| `--initial-params` | `None` | Comma-separated initial coefficient values |

## Variant Matching And Filtering

| Option | Default | Description |
|--------|---------|-------------|
| `--match-by-position` | `False` | Match variants by genomic position instead of RSID |
| `--maximum-missingness` | `0.1` | Maximum missing-variant fraction allowed |
| `--max-chisq-threshold` | `None` | Drop LD blocks above a chi-squared threshold |
| `--annotation-columns` | all | Restrict to specific annotation columns |
| `--binary-annotations-only` | `False` | Keep only 0/1-valued annotations |

## Gene Set Annotations

Use `--gene-annot-dir` to supply GMT files:

```bash
uv run graphld reml \
    /path/to/sumstats.sumstats \
    output_prefix \
    --gene-annot-dir /path/to/gmt/files/ \
    --gene-table data/genes.tsv
```

Related options:

| Option | Default | Description |
|--------|---------|-------------|
| `--gene-table` | `data/genes.tsv` | Gene coordinate table |
| `--nearest-weights` | `0.4,0.2,0.1,0.1,0.1,0.05,0.05` | Weights for nearest-gene mapping |

GMT rows are:

```text
gene_set_name<TAB>description<TAB>gene1<TAB>gene2<TAB>...
```

## Surrogate Markers

When GWAS variants are missing from the LDGM reference, graphREML can use surrogate markers in high LD. To avoid recomputing them for repeated analyses:

```bash
uv run graphld surrogates /path/to/sumstats.sumstats out.h5 --population EUR
uv run graphld reml /path/to/sumstats.sumstats output_prefix --annot-dir /path/to/annot --surrogates out.h5
```

## Parquet Files

For a multi-trait parquet input, select the trait to analyze when writing the
default saved output files:

```bash
uv run graphld reml sumstats.parquet output --name height
```

Default tall-output runs write one file pair per trait, such as
`output.height.tall.csv` and `output.height.convergence.csv`.
