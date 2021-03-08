"""More RBFs defined here."""
import numpy as np
from scipy.interpolate import Rbf


class RBF_ext(Rbf):
    """
    Extension to standard `scipy.interpolate.Rbf` function library.
    All functions follow the same naming conventions (prepended by `_h_`)
    as the superclass.
    """

    def _h_bump(Rbf, r):
        """Bump RBF kernel.

        .. math::
            R = \\exp{\\left( -\\frac{1}{(r/\\epsilon)^2} \\right)},
            \\ ||r||<1/\epsilon

        The bump function is compactly supported. It has gaussian-like
        behavior, but concentrates the entire probability mass function
        in the range :math:`||r||<1/\epsilon` The derivatives at the
        boundaries are zero, and it is :math:`C^{\\infty}` differentiable.
        """
        conditions = [r < 1/Rbf.epsilon, r >= 1/Rbf.epsilon]
        funcs = [lambda x: np.exp(-1 / (1 - (Rbf.epsilon*x)**2)), 0]
        R = np.piecewise(r, conditions, funcs)
        return R

    def _h_polyharmonic(Rbf, r):
        """A set of polyharmonic RBFs.

        .. math::
          R = r^k,\\ k={1,3,5,...} \n
          R = r^{k-1} \\ln(r),\\ k={2,4,6,...}

        .. note::
            Even though this set of radial basis functions does not have
            a shape parameter, ``Rbf.epsilon`` is used as the polyharmonic
            power.
        """
        if Rbf.epsilon < 1:
            raise ValueError("Polyharmonic power should be >= 1.")
        if np.round(Rbf.epsilon) != Rbf.epsilon:
            raise ValueError("Only integer powers for polyharmonic RBF.")

        if Rbf.epsilon % 2 == 1:
            return r**Rbf.epsilon
        else:
            conditions = [r < 1, r >= 1]
            funcs = [lambda x: x**(Rbf.epsilon-1)*np.log(x**x),
                     lambda x: x**Rbf.epsilon*np.log(x)]
            R = np.piecewise(r, conditions, funcs)
            return R


# repeat funcs for direct call (debugging only)
def _h_bump(r, epsilon):
    conditions = [np.abs(r) < 1/epsilon, np.abs(r) >= 1/epsilon]
    funcs = [lambda x: np.exp(-1 / (1 - (epsilon*x)**2)), 0]
    R = np.piecewise(r, conditions, funcs)
    return R

def _h_polyharmonic(r, epsilon):
    """This function's independent parameter epsilon is treated
    as the polyharmonic power.
    """
    if epsilon < 2:
        raise ValueError("Polyharmonic power should be >= 2.")
    if np.round(epsilon) != epsilon:
        raise ValueError("Only integer powers for polyharmonic RBF.")

    if epsilon % 2 == 1:
        return r**epsilon
    else:
        conditions = [np.abs(r) < 1, np.abs(r) >= 1]
        funcs = [lambda x: x**(epsilon-1)*np.log(x**x),
                 lambda x: x**epsilon*np.log(x)]
        R = np.piecewise(r, conditions, funcs)
        return R
