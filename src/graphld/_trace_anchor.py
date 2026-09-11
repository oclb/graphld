"""Fixed-probe inverse-diagonal error correction for one REML worker block."""

import numpy as np


class InverseDiagonalAnchor:
    """Refresh exact error at audits; contract with the current model Jacobian.

    Anchors may span covariance updates. A cached raw estimate is reusable only
    at the identical matrix version, probe configuration, operator and selection.
    """

    def __init__(self):
        self.anchor = None
        self.raw_cache = None
        self.refreshes = self.uses = self.cache_hits = 0

    @staticmethod
    def _same_selection(a, b):
        return (a is None and b is None) or (
            a is not None and b is not None and np.array_equal(a, b)
        )

    def evaluate(self, operator, *, precise=False, n_samples=100, seed=123):
        if seed is None:
            raise ValueError(
                "Trace correction requires a fixed probe seed within a fit"
            )
        selection = getattr(operator, "_which_indices", None)
        selection = None if selection is None else selection.copy()
        metadata = (id(operator), tuple(operator.shape), "xdiag", n_samples, seed)
        version_fn = getattr(operator, "_current_matrix_version", None)
        version = version_fn() if version_fn is not None else None
        reusable = (
            version is not None
            and self.raw_cache is not None
            and self.raw_cache[0] == (metadata, version)
            and self._same_selection(selection, self.raw_cache[1])
        )
        if precise and reusable:
            raw = self.raw_cache[2]
            self.cache_hits += 1
        else:
            raw = np.asarray(
                operator.inverse_diagonal(
                    method="xdiag", n_samples=n_samples, seed=seed
                )
            )
        if precise:
            exact = np.asarray(operator.inverse_diagonal(method="exact"))
            self.anchor = (metadata, selection, raw - exact)
            self.refreshes += 1
            return exact
        self.raw_cache = ((metadata, version), selection, raw.copy())
        if (
            self.anchor is not None
            and self.anchor[0] == metadata
            and self._same_selection(selection, self.anchor[1])
        ):
            self.uses += 1
            return raw - self.anchor[2]
        return raw
