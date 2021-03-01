"""
Create a cosmological linear matter power spectrum interpolator.
"""
import textwrap
import warnings
from tqdm import tqdm
import numpy as np
import pyccl as ccl
from scipy.interpolate import Rbf
from weights import weights as wts


class interpolator(object):
    """
    Interpolator of the cosmological linear matter power spectrum.

    #TODO: a few more words

    Parameters
    ----------
    priors : ``dict`` (key: [vmin, vmax])
        Upper and lower parameter boundaries.
    cosmo_default : ``dict`` (key: val)
        Fiducial fixed cosmological parameters.
        If `None`, defaults to Planck 2018.
    k_arr : ``numpy.array``
        Wavenumbers to sample at.
        If `None`, use arguments `lkmin`, `lkmax`, `kpts`
        to construct the wavenumber array.
    a_arr : ``numpy.array``
        Scale factors to sample at.
        If `None`, use arguments `amin`, `amax`, `apts`
        to construct the scale factor array.
    samples : ``int``
        Target number of interpolation nodes.
        The actual number of samples will vary (always greater,
        but close to the target number) if the cosmological
        dimensions are weihted, or if the `n-th` root of `samples`,
        where `n` is the number of cosmological parameters
        is not an integer.
    interpf : ``str``
        The radial basis function. See `scipy.interpolate.Rbf`.
    check_cosmo : ``bool``
        Check that every ``pyccl.Cosmology`` object passed in the
        interpolator is compatible with the interpolation.
        Save time by setting it to `False`, but do so at your own risk!
    weigh_dims : ``bool``
        Distribute available samples along the cosmological
        dimensions using weights, according to how much `P(k,a)`
        changes in the interval of the passed prior.
        Defaults to `True`. Will independently sample each
        cosmological dimension.
    wpts : ``int``
        Initial number of sampling points along each
        cosmological dimension.
    prefix : ``str``
        Prefix used to save the output.
    overwrite : ``bool``
        Overwrite any saved output. Defaults to `True`.
    save: ``bool``
        Save the weights in compressed `.npz` format.
    """

    def __init__(self, priors, cosmo_default=None,
                 k_arr=None, a_arr=None,
                 samples=50, interpf="multiquadric",
                 check_cosmo = True,
                 weigh_dims=True, wpts=None,
                 prefix="", overwrite=True, save=True):
        # cosmo params
        self.priors = priors
        self.pars = self.priors.keys()
        self.cosmo_default = cosmo_default
        if self.cosmo_default is None:
            print("Fixed cosmological parameters from Planck 2018.")
            self.cd = interpolator.Planck18()
        # k, a
        self.k_arr = np.sort(k_arr)
        self.kpts = len(self.k_arr)
        self.a_arr = np.sort(a_arr)
        self.apts = len(self.a_arr)
        # interp
        self.samples = samples
        self.interpf = interpf
        self.check_cosmo = check_cosmo
        self.weigh_dims = weigh_dims
        # weights
        self.wpts = wpts
        if self.weigh_dims and (self.wpts is None):
            warnings.warn("wpts not set; defaulting to 16 per dimension")
            self.wpts = 16
        # I/O
        self.pre = prefix
        self.overwrite = overwrite
        self.save = save

        # calculate parameter weights
        self.get_weights()
        # build cosmological parameter space
        self.get_nodes()
        # sample parameter space at weighted axes
        Pk = self.Pka()
        # interpolate
        self.interpolate(Pk)

    @classmethod
    def Planck18(interpolator):
        """Return dictionary of Planck 2018 cosmology."""
        cosmo = {"Omega_c" : 0.2589,
                 "Omega_b" : 0.0486,
                 "h"       : 0.6774,
                 "sigma8"  : 0.8159,
                 "n_s"     : 0.9667}
        return cosmo

    def get_weights(self):
        """Calculate how the available samples are distributed
        in each dimension.
        """
        if self.weigh_dims:
            W = wts(self.priors, self.cd,
                    k_arr=self.k_arr, a_arr=self.a_arr,
                    wpts=self.wpts, prefix=self.pre)
            weights_dict = W.get_weights(ref=self.samples,
                                         output=True,
                                         save=self.save,
                                         overwrite=self.overwrite)
            self.weights = np.array([weights_dict[par] for par in self.pars])
        else:
            w = np.ceil(self.samples**(1/len(self.pars)))
            self.weights = np.repeat(w, len(self.pars))

    def get_nodes(self):
        """Calculate the coordinates of the nodal points."""
        self.points = [np.linspace(*self.priors[key], ww)
                       for key, ww in zip(self.pars, self.weights)]
        mg = np.meshgrid(*self.points, indexing="ij")
        self.pos = np.vstack(list(map(np.ravel, mg))).T

    def Pka(self):
        """Compute `P(k,a)` at each cosmological node."""
        Pk = np.zeros((np.product(self.weights), self.apts, self.kpts))
        for i, p in enumerate(tqdm(self.pos, desc="Sampling P(k) grid")):
            kw = self.cd.copy()
            kw.update(dict(zip(self.pars, p)))
            cosmo = ccl.Cosmology(**kw)
            Pk[i] = wts.linear_matter_power(cosmo, self.k_arr, self.a_arr)
        Pk = Pk.reshape(*np.r_[self.weights, self.apts, self.kpts])
        return Pk

    def interpolate(self, Pk):
        """
        Interpolate `P(k,a)`.

        To overcome memory constraints in the calculation of
        the metric distance, we take advantage of `a_arr` and `k_arr`
        being fixed and construct `apts*kpts` independent interpolators.

        To achieve smoother interpolation over the extent of the
        power spectrum, we interpolate `log_10(Pk)`.
        """
        print("Interpolating...")
        lPk = np.log10(Pk).reshape(*np.r_[self.weights, self.apts*self.kpts])
        self.F = [Rbf(*np.c_[self.pos, lPk[..., i].flatten()].T,
                      function=self.interpf)
                  for i in range(self.apts*self.kpts)]
        self.F = np.array(self.F).reshape((self.apts, self.kpts))

    def linear_matter_power(self, cosmo, k_arr, a_arr):
        """Interpolated linear matter power spectrum with the same
        function call as `pyccl.linear_matter_power`
        """
        # check if `cosmo` is compatible with interpolation
        if self.check_cosmo:
            fixed = list(set(self.cd.keys() - set(self.pars)))
            for par in fixed:
                if cosmo[par] != self.cd[par]:
                    raise ValueError(textwrap.fill(textwrap.dedent("""
            Cosmological parameter %s not compatible with interpolation.
            """ % par)))
        if not all(np.in1d(a_arr, self.a_arr)):
            raise ValueError("Value(s) in a_arr not matching interpolation.")
        if not all(np.in1d(k_arr, self.k_arr)):
            raise ValueError("Value(s) in k_arr not matching interpolation.")
        a_arr = np.atleast_1d(a_arr).astype(float)
        k_arr = np.atleast_1d(k_arr).astype(float)

        pars = [cosmo[par] for par in self.pars]

        ia = np.searchsorted(self.a_arr, a_arr)
        ik = np.searchsorted(self.k_arr, k_arr)

        lPk = np.array([f(*pars) for f in self.F[ia][:, ik].flatten()])
        Pk = 10**lPk.reshape((len(a_arr), len(k_arr)))
        return Pk.squeeze()
