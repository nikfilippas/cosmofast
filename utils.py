"""Generic useful functions that do not belong to any class."""
import numpy as np
import pyccl as ccl


def Planck18():
    """Dictionary of Planck 2018 cosmology.

    This is only used to get relative sampling densities in the
    cosmological parameter space, so even if a default fixed-params
    dictionary is not set, sampling ultimately becomes slightly
    less efficient.
    """
    cosmo = {"Omega_c" : 0.2589,
             "Omega_b" : 0.0486,
             "h"       : 0.6774,
             "sigma8"  : 0.8159,
             "n_s"     : 0.9667}
    return cosmo


def linear_matter_power(cosmo, k_arr, a_arr):
    """Re-definition of `pyccl.linear_matter_power`
    to accept `array-like` scale factor.
    """
    a_arr = np.atleast_1d(a_arr).astype(float)
    k_arr = np.atleast_1d(k_arr).astype(float)
    Pk = np.array([ccl.linear_matter_power(cosmo, k_arr, a) for a in a_arr])
    return Pk.squeeze()


def arc(x, y):
    """Calculate the arc length of `y`."""
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    segments = np.sqrt(np.gradient(x, axis=-1)**2
                       + np.gradient(y, axis=-1)**2)
    return np.sum(segments, axis=-1).squeeze()
