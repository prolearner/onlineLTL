from scipy import stats
import numpy as np

class mod_logistic_dist(stats.rv_continuous):
    def _cdf(self, x, *args):
        return 1/(1 - np.exp())