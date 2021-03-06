"""Bump kernel RBF."""
import numpy as np
from scipy.interpolate import Rbf

class BumpRBF(Rbf):
    """Bump RBF kernel.

    .. math::
        R = \\exp{\\left( -\\frac{1}{(\\epsilon r)^2} \\right)},
        \\ ||r||<1/\epsilon

    The bump function is compactly supported. It has gaussian-like
    behavior, but concentrates the entire probability mass function
    in the range :math:`||r||<1/\epsilon` The derivatives at the
    boundaries are zero, and it is :math:`C^{\\infty}` differentiable."""

    def _h_bump(Rbf, r):
        conditions = [np.abs(r) < 1/Rbf.epsilon, np.abs(r) >= 1/Rbf.epsilon]
        funcs = [lambda x: np.exp(-1 / (1 - (Rbf.epsilon*x)**2)), 0]
        R = np.piecewise(r, conditions, funcs)
        return R
