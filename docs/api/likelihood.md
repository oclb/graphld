# graphld.likelihood

Gaussian likelihood functions for GWAS summary statistics under an infinitesimal model.

## Model

The likelihood of GWAS summary statistics under an infinitesimal model is:

$$\beta \sim N(0, D)$$

$$z|\beta \sim N(n^{1/2}R\beta, R)$$

where:

- $\beta$ is the effect-size vector in s.d-per-s.d. units
- $D$ is a diagonal matrix of per-variant heritabilities
- $z$ is the GWAS summary statistic vector
- $R$ is the LD correlation matrix
- $n$ is the sample size

## Precision-Premultiplied Statistics

The likelihood functions operate on precision-premultiplied GWAS summary statistics:

$$pz = n^{-1/2} R^{-1}z \sim N(0, M), \quad M = D + n^{-1}R^{-1}$$

::: graphld.likelihood
    options:
      show_root_heading: true
      members_order: source
