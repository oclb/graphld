# graphREML optimization and information penalty

The [methodological write-up](graphreml_methods.pdf) describes the numerical repairs,
optimizer choices, optional average-information log-determinant penalty, and real-data
comparisons. Its [LaTeX source](graphreml_methods.tex) and aggregate plotting inputs
are included here.

The tested BFGS variant failed to converge on Weight and was slower than AI on BMI.
The development default remains BFGS pending review; these results do not support
releasing that default. The information penalty is off by default, and exact trial
evaluation remains the supported penalty strategy. Penalized uncertainty calibration
is deferred.

To compile the document, run `pdflatex graphreml_methods.tex` twice from this directory.
The supplied tables and PDF figures are sufficient for compilation. To regenerate
the figures with Python, NumPy and Matplotlib:

```sh
python plot_matched_optimizer_comparison.py --source safeguarded_comparison/matched_start_enrichment_estimates.tsv --output safeguarded_comparison --variant safeguarded
python plot_penalty_comparison.py
```

The source tables retain convergence status and timing definitions. The historical
Weight comparison uses the earlier runner's current-factor direct solve; the
misconfigured guard-off attempt is excluded. The prior BMI result used its original
likelihood-window stopping rule. Primary annotations match Figure 3 of
[Li et al. (2026)](https://doi.org/10.1038/s41588-026-02649-0). The three secondary
binary annotations were selected from Weight and then displayed for both Weight
and Hemoglobin. Full annotation summaries retain their source model interpretation:
signed continuous annotation weights are not SNP-set heritability fractions.
