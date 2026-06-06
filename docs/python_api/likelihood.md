# Likelihood Functions

GraphLD provides Gaussian likelihood helpers for precision-premultiplied GWAS summary statistics under an infinitesimal model.

The model is:

> β ~ N(0, D)
>
> z|β ~ N(n^(1/2) R β, R)

where β is the effect-size vector in s.d.-per-s.d. units, D is a diagonal matrix of per-variant heritabilities, z is the GWAS summary statistic vector, R is the LD correlation matrix, and n is the sample size.

The likelihood functions operate on precision-premultiplied summary statistics:

> pz = n^(-1/2) R^(-1) z ~ N(0, M), where M = D + n^(-1) R^(-1)

Available functions:

- `gaussian_likelihood(pz, M)`: compute the log-likelihood
- `gaussian_likelihood_gradient(pz, M, del_M_del_a=None, n_samples=10, seed=None, trace_estimator="xdiag")`: compute the gradient with respect to either the diagonal of `M` or parameters `a`
- `gaussian_likelihood_hessian(pz, M, del_M_del_a=None, trace_estimator="xdiag")`: compute an approximate Hessian with respect to `a`, or diagonal-only Hessian output when `del_M_del_a` is omitted

The Hessian approximation is minus the average of the Fisher information matrix and the observed information matrix, and is most useful near the optimum.

See also:

- [graphld.likelihood API Reference](../api/likelihood.md)
- [Heritability Estimation](heritability.md)
- [graphld package overview](../api/graphld.md)
