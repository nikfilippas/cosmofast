"""Bump kernel RBF."""
import warnings
import numpy as np
from scipy.interpolate import Rbf

class BumpRBF(Rbf):
    """Bump RBF kernel."""

    def _h_bump(Rbf, r):
        # optimization
        r = np.atleast_1d(r)
        if np.all(np.abs(r) >= Rbf.epsilon):
            return np.zeros_like(r)
        # catch RunTime warnings for infinite values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            R = np.exp(-1 / (1 - (r/Rbf.epsilon)**2))
        R[np.abs(r) >= Rbf.epsilon] = 0
        return R
