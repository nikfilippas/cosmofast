"""
Deduce how the sample points should be spread along each dimension.
"""
import os
import numpy as np
import pyccl as ccl
from tqdm import tqdm
from utils import arc
from utils import linear_matter_power as linpow


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
    """

    def __init__(self, priors, cosmo_default,
                 k_arr=None, a_arr=None, *,
                 wpts=16):
        self.priors = priors
        self.cosmo_default = cosmo_default
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

    def get_fname(self, which, dic="res"):
        """Produce saving code string."""
        fixed = list(set(self.cosmo_default.keys() - set(self.pars)))
        code = "/".join([dic, which]) + "_"

        for par in sorted(self.pars):
            code += "_".join([par,
                              str(self.priors[par][0]),
                              str(self.priors[par][1])])
            code += "_"

        if len(fixed) > 0:
            for par in sorted(fixed):
                code += "".join([par, str(self.cosmo_default[par])])
                code += "_"

        code += "wpts%s.npz" % self.wpts
        return code

    def gradient(self, arr, key):
        kw = self.cosmo_default.copy()  # clean copy to avoid surprises
        lPk_full = np.zeros((self.wpts, len(self.a_arr), len(self.k_arr)))
        for i, val in enumerate(tqdm(arr, desc=key)):
            kw[key] = val
            cosmo = ccl.Cosmology(**kw)
            lPk_full[i] = np.log10(linpow(cosmo, self.k_arr, self.a_arr))
        grads = np.max(np.gradient(lPk_full, axis=0), axis=(1,2))
        dx = np.gradient(arr)
        grads /= dx
        return grads

    def get_gradients(self, output=True, overwrite=False):
        f_grads = self.get_fname("gradients")
        gradients = dict.fromkeys(self.pars)
        # load from memory if possible
        if not overwrite and os.path.isfile(f_grads):
            f = np.load(f_grads, allow_pickle=True)
            for par in f.files:
                gradients[par] = f[par]
        else:
            print("Calculating partial derivatives and weights...")
            for par in self.pars:
                pts = self.sample_pts(par)
                grad = self.gradient(pts, par)
                gradients[par] = np.c_[pts, grad]

        os.makedirs(f_grads.split("/")[0], exist_ok=True)
        np.savez(f_grads, **gradients)
        if output:
            return gradients

    def get_weights(self, ref=100,
                    int_samples_func="ceil",
                    output=True, overwrite=False):
        """Calculate re-distribution weights of the sample points."""
        ifunc = getattr(np, int_samples_func)
        f_weights = self.get_fname("weights")
        gradients = self.get_gradients(output=output,
                                       overwrite=overwrite)
        weights = dict.fromkeys(gradients)
        for par, grad in gradients.items():
            weights[par] = arc(*grad.T)
        norm = (ref/np.product(list(weights.values())))**(1/len(self.pars))
        # save the arc lengths before calculating weights
        # relative to the requested number of samples (ref)
        os.makedirs(f_weights.split("/")[0], exist_ok=True)
        np.savez(f_weights, **weights)
        # weight according to ref
        for par, w in weights.items():
            weights[par] = ifunc(norm*w).astype(int)

        if output:
            return weights
