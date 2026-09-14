# graphREML optimization and information penalty

The [methodological write-up](graphreml_methods.pdf) describes the numerical repairs,
AI optimization, optional average-information log-determinant penalty, and real-data
comparisons. Its [LaTeX source](graphreml_methods.tex) and aggregate plotting inputs
are included here.

AI is the supported optimizer. It combines backtracking line search with score
correction and precise endpoint audits. The information penalty is off by default,
and exact trial evaluation is the default penalty strategy. Penalized uncertainty
calibration is deferred. A short appendix reports the exploratory BFGS comparison.

All 33 fits in the previous presentation cohort have finished.
[Aggregate results](cohort_final/cohort_status_and_estimates.tsv) retain all
traits: thirteen passed numerical validation, nineteen reached the iteration
limit, and one passed the local stopping checks but had singular information.
All six binary traits reached the iteration limit. Acute sinusitis also had
singular information, making two outcomes with unavailable alpha and uncertainty.
The table flags provisional estimates and boundary-affected uncertainty, including
the degenerate zero alpha standard error for cellulitis. Numerical validation
does not establish a global likelihood maximum or uncertainty calibration.

The cohort table records optimizer, native API, input-preparation and pre-extraction
runner times separately, in seconds. The last timer stops before downstream
extraction and is not the complete process lifetime. These are local execution measurements with early
concurrent workloads and later serial fits, rather than matched-accuracy runtime
comparisons. Each primary fit used twelve workers; the information penalty was off.
The alpha-validation scope is recorded per trait: ten eligible traits have
independent full-and-100-delete optimum checks, while BMI, Weight and Hemoglobin
have benchmark parity and saved arithmetic checks.
The phenotype mappings are preserved from the earlier cohort, including weak
binary mappings with 112 acute-sinusitis cases and 448 obesity cases.
Experimental curvature corrections did not improve efficiency on the fixed
synthetic controls.

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
