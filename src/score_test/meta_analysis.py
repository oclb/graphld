import numpy as np

class MetaAnalysis:
    point_estimates: np.ndarray
    jackknife_estimates: np.ndarray

    def __init__(self):
        self.point_estimates = 0
        self.jackknife_estimates = 0

    def update(self, point_estimates, jackknife_estimates):
        precision = 1 / np.var(jackknife_estimates, axis=0)
        self.point_estimates += precision * point_estimates
        self.jackknife_estimates += precision * jackknife_estimates

    @property
    def std(self):
        n = self.jackknife_estimates.shape[0] - 1
        return np.sqrt(np.var(self.jackknife_estimates, axis=0) * n)

    @property
    def z_scores(self):
        return self.point_estimates / self.std
