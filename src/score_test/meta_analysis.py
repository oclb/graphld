import numpy as np

class MetaAnalysis:
    point_estimates: np.ndarray | None
    jackknife_estimates: np.ndarray | None

    def __init__(self):
        self.point_estimates = None
        self.jackknife_estimates = None

    def update(self, point_estimates, jackknife_estimates):
        point_estimates = np.asarray(point_estimates)
        jackknife_estimates = np.asarray(jackknife_estimates)
        precision = 1 / np.var(jackknife_estimates, axis=0)

        weighted_point_estimates = precision * point_estimates
        weighted_jackknife_estimates = precision * jackknife_estimates

        if self.point_estimates is None:
            self.point_estimates = weighted_point_estimates.copy()
            self.jackknife_estimates = weighted_jackknife_estimates.copy()
            return

        self.point_estimates += weighted_point_estimates
        self.jackknife_estimates += weighted_jackknife_estimates

    @property
    def std(self):
        if self.jackknife_estimates is None:
            raise ValueError("MetaAnalysis has no estimates; call update() first")

        n = self.jackknife_estimates.shape[0] - 1
        return np.sqrt(np.var(self.jackknife_estimates, axis=0) * n)

    @property
    def z_scores(self):
        if self.point_estimates is None:
            raise ValueError("MetaAnalysis has no estimates; call update() first")

        return self.point_estimates / self.std
