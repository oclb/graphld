# graphld.blup

Best Linear Unbiased Prediction (BLUP) for effect size estimation.

Under the infinitesimal model with per-s.d. effect sizes $\beta \sim N(0, D)$, the BLUP effect sizes are:

$$E(\beta) = \sqrt{n} D (nD + R^{-1})^{-1} R^{-1}z$$

where $R^{-1}$ is approximated with the LDGM precision matrix.

::: graphld.blup
    options:
      show_root_heading: true
      members_order: source
