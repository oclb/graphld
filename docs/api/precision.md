# graphld.precision

Precision matrix operations using LDGM graphical models.

The `PrecisionOperator` class is the core abstraction for working with LD matrices. It subclasses SciPy's `LinearOperator` and represents an LDGM precision matrix or its Schur complement.

## Key Concepts

- To compute `correlation_matrix[indices, indices] @ vector`, use `ldgm[indices].solve(vector)`
- To compute `inv(correlation_matrix[indices, indices]) @ vector`, use `ldgm[indices] @ vector`

You cannot do this indexing manually - the submatrix of an inverse is not the inverse of a submatrix. See Section 5 of the [supplementary material](https://pubmed.ncbi.nlm.nih.gov/37640881/).

::: graphld.precision
    options:
      show_root_heading: true
      members_order: source
