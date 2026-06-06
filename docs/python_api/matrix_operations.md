# Matrix Operations

LD matrix operations are centered on `PrecisionOperator`, which subclasses SciPy's [`LinearOperator`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html). It represents an LDGM precision matrix or its [Schur complement](https://en.wikipedia.org/wiki/Schur_complement).

If you need `correlation_matrix[indices, indices] @ vector`, use `ldgm[indices].solve(vector)`. If you need `inv(correlation_matrix[indices, indices]) @ vector`, use `ldgm[indices] @ vector`. Do not index the matrix stored inside of the LDGM object manually: the submatrix of an inverse is not the inverse of a submatrix (instead, it is the inverse of the Schur complement).

```python
import numpy as np
import graphld as gld

ldgm = gld.load_ldgm(
    filepath="data/test/1kg_chr1_16103_2888443.EAS.edgelist",
    snplist_path="data/test/1kg_chr1_16103_2888443.snplist",
)

vector = np.random.randn(ldgm.shape[0])
precision_times_vector = ldgm @ vector
correlation_times_vector = ldgm.solve(precision_times_vector)
assert np.allclose(correlation_times_vector, vector)
```

See also:

- [I/O and Merging](io_and_merging.md)
- [graphld.precision API Reference](../api/precision.md)
- [graphld package overview](../api/graphld.md)
