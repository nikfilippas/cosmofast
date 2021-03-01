"""
Deduce how the sample points should be spread along each dimension.
"""
import os
import numpy as np
import pyccl as ccl
from tqdm import tqdm

class weights(object):
    """
    Deduce how the available sampling points should be
    spread along each cosmological dimension.

    Initially, each dimension is assigned equal weight
    and equal number of points are linearly spaced along
    the cosmological dimensions.

    Cosmological parameters are assigned weights according
    to how much they vary `P(k,a)`.

    Weights are the arc lengths of the maximum partial derivatives
    of `P(k,a)` with respect to each cosmological parameter.

    This is more accurate than simply finding the magnitude
    of the maximum derivative, as curves with more features are
    upweighted in a process similar to feature extraction.

    Parameters
    ----------
    priors : ``dict`` (key: [vmin, vmax])
        Upper and lower parameter boundaries.
    cosmo_default : ``dict`` (key: val)
        Fiducial fixed cosmological parameters.
    k_arr : ``numpy.array``
        Wavenumbers to sample at.
        If `None`, use arguments `lkmin`, `lkmax`, `kpts`
        to construct the wavenumber array.
    a_arr : ``numpy.array``
        Scale factors to sample at.
        If `None`, use arguments `amin`, `amax`, `apts`
        to construct the scale factor array.
    wpts : ``int``
        Initial number of linearly spaced sampling points
        along each cosmological dimension.
    prefix : ``str``
        Prefix used to save the output.
    """

    def __init__(self, priors, cosmo_default,
                 k_arr=None, a_arr=None, *,
                 wpts=16, prefix=""):
        self.priors = priors
        self.cd = cosmo_default
        self.pre = prefix
        if (k_arr is None) or (a_arr is None):
            raise ValueError("`k_arr` and `a_arr` should be array-like")
        self.k_arr = k_arr
        self.a_arr = a_arr
        self.pars = list(self.priors.keys())
        self.wpts = wpts

    def sample_pts(self, key):
        """Create an array of sampling points."""
        return np.linspace(self.priors[key][0],
                           self.priors[key][1],
                           self.wpts)

    @classmethod
    def linear_matter_power(weights, cosmo, k_arr, a_arr):
        """Re-definition of `pyccl.linear_matter_power`
        to accept `array-like` scale factor.
        """
        a_arr = np.atleast_1d(a_arr).astype(float)
        k_arr = np.atleast_1d(k_arr).astype(float)
        Pk = np.array([ccl.linear_matter_power(cosmo, k_arr, a) for a in a_arr])
        return Pk.squeeze()

    def gradient(self, arr, key):
        kw = self.cd.copy()  # clean copy to avoid surprises
        lPk_full = np.zeros((self.wpts, len(self.a_arr), len(self.k_arr)))
        for i, val in enumerate(tqdm(arr, desc=key)):
            kw[key] = val
            cosmo = ccl.Cosmology(**kw)
            lPk_full[i] = np.log10(weights.linear_matter_power(cosmo,
                                                               self.k_arr,
                                                               self.a_arr))
        grads = np.max(np.gradient(lPk_full, axis=0), axis=(1,2))
        dx = np.gradient(arr)
        grads /= dx
        return grads

    def get_gradients(self, save=True, output=True, overwrite=False):
        f_grads = "_".join(filter(None, ["res/gradients", self.pre])) + ".npz"
        gradients = dict.fromkeys(self.pars)
        # load from memory if possible
        if not overwrite and os.path.isfile(f_grads):
            f = np.load(f_grads)
            for par in f.files:
                gradients[par] = f[par]
        else:
            print("Calculating partial derivatives and weights...")
            for par in self.pars:
                pts = self.sample_pts(par)
                grad = self.gradient(pts, par)
                gradients[par] = np.c_[pts, grad]

        if save:
            os.makedirs("res", exist_ok=True)
            np.savez(f_grads, **gradients)
        if output:
            return gradients

    @classmethod
    def C(weights, x, y):
        """Calculate the arc length of `y`."""
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)
        segments = np.sqrt(np.gradient(x, axis=-1)**2
                           + np.gradient(y, axis=-1)**2)
        return np.sum(segments, axis=-1).squeeze()

    def get_weights(self, ref=100, save=True, output=True, overwrite=False):
        """Calculate re-distribution weights of the sample points."""
        f_weights = "_".join(filter(None, ["res/weights", self.pre])) + ".npz"
        gradients = self.get_gradients(save=save,
                                        output=output,
                                        overwrite=overwrite)
        weights = dict.fromkeys(gradients)
        for par, grad in gradients.items():
            weights[par] = self.C(*grad.T)
        norm = (ref/np.product(list(weights.values())))**(1/len(self.pars))
        for par, w in weights.items():
            weights[par] = np.ceil(norm*w).astype(int)

        if save:
            os.makedirs("res", exist_ok=True)
            np.savez(f_weights, **weights)
        if output:
            return weights
