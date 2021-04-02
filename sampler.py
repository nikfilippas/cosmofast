""" P(k,a) sampler. """
import os
import warnings
import textwrap
from tqdm import tqdm
import numpy as np
import pyccl as ccl
from weights import weights as wts
from utils import linear_matter_power as linpow


class Sampler(object):
    """
    """

    def __init__(self, priors, cosmo_default=None,
                 k_arr=None, a_arr=None,
                 samples=50, int_samples_func="ceil", *,
                 weigh_dims=False, wpts=None,
                 overwrite=False):
        # cosmo params
        self.priors = priors
        self.pars = list(self.priors.keys())
        self.cosmo_default = cosmo_default
        if self.cosmo_default is None:
            warnings.warn("Fixed cosmological parameters from Planck 2018.")
            from utils import Planck18
            self.cosmo_default = Planck18()

        # k, a
        self.k_arr = np.sort(k_arr)
        self.kpts = len(self.k_arr)
        self.a_arr = np.sort(a_arr)
        self.apts = len(self.a_arr)

        # sampler
        self.samples = samples
        self.overwrite = overwrite

        # parameter weights
        self.int_samples_func = int_samples_func
        self.weigh_dims = weigh_dims
        self.wpts = wpts
        if self.weigh_dims and (self.wpts is None):
            warnings.warn("wpts not set; defaulting to 16 per dimension")
            self.wpts = 16
        self.get_weights()
        if (self.weights == 1).any():
            warnings.warn(textwrap.fill(textwrap.dedent("""
            Very small number of samples. Parameter space will not be
            adequately sampled. Increasing the number of samples from 1
            to 2 in parameter(s) %s.
            """ % [par for i, par in enumerate(self.pars)
                    if self.weights[i] == 1])))
            self.weights[self.weights == 1] += 1

        # build cosmological parameter space
        self.get_nodes()

    def get_fname(self, which="", dic="res"):
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

        code += "a_arr%s_%s_%s_"%(self.a_arr.min(),self.a_arr.max(),self.apts)
        code += "k_arr%s_%s_%s_"%(self.k_arr.min(),self.k_arr.max(),self.kpts)
        code += "samples%s.npy" % self.samples
        return code

    def get_weights(self):
        """Calculate how the available samples are distributed
        in each dimension.
        """
        if self.weigh_dims:
            W = wts(self.priors, self.cosmo_default,
                    k_arr=self.k_arr, a_arr=self.a_arr,
                    wpts=self.wpts)
            weights_dict = W.get_weights(ref=self.samples,
                                          int_samples_func=self.int_samples_func,
                                          output=True,
                                          overwrite=self.overwrite)
            self.weights = np.array([weights_dict[par] for par in self.pars])
        else:
            ifunc = getattr(np, self.int_samples_func)
            w = ifunc(self.samples**(1/len(self.pars))).astype(int)
            self.weights = np.repeat(w, len(self.pars))

    def get_nodes(self):
        """Calculate the coordinates of the nodal points."""
        self.points = [np.linspace(*self.priors[key], ww)
                       for key, ww in zip(self.pars, self.weights)]
        mg = np.meshgrid(*self.points, indexing="ij")
        self.pos = np.vstack(list(map(np.ravel, mg))).T

    def sample(self):
        """
        Sample the linear matter power spectrum.

        Compute the fractional error between the CAMB `P(k,a)`
        and the Eisenstein & Hu `P(k,a)` at each cosmological node.
        Using `numpy.memmap` to avoid MemoryError for large sample numbers.
        """
        # check if file exists
        f_err = self.get_fname("err")
        if not self.overwrite and os.path.isfile(f_err):
            warnings.warn("Found samples %s with `overwrite==False`." % f_err)
            return

        # create mmap array
        shp = (np.product(self.weights), self.apts, self.kpts)
        os.makedirs(f_err.split("/")[0], exist_ok=True)
        err = np.memmap(f_err, dtype=float, mode="w+", shape=shp)
        del err

        # sample
        for i, p in enumerate(tqdm(self.pos, desc="Sampling grid")):
            kw = self.cosmo_default.copy()
            kw.update(dict(zip(self.pars, p)))

            # Eisenstein & Hu approximation
            cosmo_eh = ccl.Cosmology(**kw, transfer_function="eisenstein_hu")
            Pl_eh = linpow(cosmo_eh, self.k_arr, self.a_arr)

            # CAMB (high accuracy)
            cosmo_0 = ccl.Cosmology(**kw, transfer_function="boltzmann_camb")
            Pl_0 = linpow(cosmo_0, self.k_arr, self.a_arr)

            # open, write, flush
            err = np.memmap(f_err, dtype=float, mode="r+", shape=shp)
            err[i, :] = Pl_eh / Pl_0
            del err
