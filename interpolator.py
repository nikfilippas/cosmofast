"""
Create a cosmological linear matter power spectrum interpolator.

#TODO: sklearn optimal epsilon
#TODO: run profiler
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

    Nodes for the cosmological hypercube are computed using weights.
    Weights aim to increase efficiency by allocating more samples
    to the parameters that change more and/or have more feautures.
    This is calculated by `weights.weights` in the following way:
        - Compute the partial derivative of `P(k,a)` with respect
          to each cosmological parameter, at every `(k,a)`.
        - Define the 'effective' derivative as the maximum of that.
        - Calculate the normalized arc length to upsample hilly
          hypersurfaces.
        - Define this as the weight. Normalize for the target
          number of samples.

    The power spectrum is computed at the nodes and interpolated
    using radial basis functions (RBFs). These are functions whose
    value depends only on the distance from a reference point (node).

    Euclidean metric is used to calculate distance, because the
    cosmological parameters are independent (enough) from one another.

    The used RBF can be changed to any of the functions outlined in
    `scipy.interpolate.Rbf`; however, `gaussian` is a good choice
    as it is similar to a KDE in multiple dimensions.
    Use `linear` for calculation speed-up.

    For `gaussian`, `multiquadric`, and `inverse` (quadric) RBFs,
    the hyperparameter `epsilon` controls the width of the kernel.

    Class method `interpolator.linear_matter_power` ultimately has
    the same function call as `pyccl.linear_matter_power`. That is
    to introduce compatibility; however, serious speed-up can be
    achieved if multiple cosmological proposals are made directly into
    the `interpolator.F` attribute rather than calling it iteratively.
    This is realized in `interpolator.callF`.


    Parameters
    ----------
    priors : ``dict`` (key: [vmin, vmax])
        Upper and lower parameter boundaries.
    cosmo_default : ``dict`` (key: val)
        Fiducial fixed cosmological parameters.
        If `None`, defaults to Planck 2018.
    k_arr : ``numpy.array``
        Wavenumbers to sample at.
    a_arr : ``numpy.array``
        Scale factors to sample at.
    samples : ``int``
        Target number of interpolation nodes.
        The actual number of samples will vary (always greater,
        but close to the target number) if the cosmological
        dimensions are weihted, or if the `n-th` root of `samples`,
        where `n` is the number of cosmological parameters,
        is not an integer.
    int_samples_func : ``str`` {'round', 'ceil', 'floor'}
        Method to approximate integer samples along each axis.
        'round' will make the effective number of samples nearest
        to the target `samples`. Defaults to `ceil`.
    check_cosmo : ``bool``
        Check that every ``pyccl.Cosmology`` object passed in the
        interpolator is compatible with the interpolation.
        Save time by setting it to `False`, but do so at your own risk!
    interpf : ``str``
        The radial basis function. See `scipy.interpolate.Rbf`.
    epsilon : ``float``
        Adjustment knob for gaussian and multiquadric functions.
        A good start for cosmological interpolation is 50x the
        average distance between the nodes.
        See `scipy.interpolate.Rbf`.
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

    Attributes
    ----------
    All arguments become class attributes.
    Additional attributes are listed below.
    pars : ``list`` of ``str``
        Names of the free cosmological parameters.
    kpts : ``float``
        Number of wavenumber points.
    apts : ``float``
        Number of scale factor points.
    weights : ``list``
        Number of sampling points per cosmological parameter.
    points : ``list`` of ``numpy.array``
        Sampling points of each parameter.
    pos : ``numpy.array`` (samples_effective, len(pars))
        Coordinates of the nodal points.
    F : ``numpy.array`` of ``scipy.interpolate.rbf.Rbf`` (apts, kpts)
        Array containing the interpolators for each (k, a) combination.
    """

    def __init__(self, priors, cosmo_default=None,
                 k_arr=None, a_arr=None, samples=50, *,
                 int_samples_func="ceil", check_cosmo=True,
                 interpf="gaussian", epsilon=None,
                 weigh_dims=True, wpts=None,
                 prefix="", overwrite=True, save=True):
        # cosmo params
        self.priors = priors
        self.pars = self.priors.keys()
        self.cosmo_default = cosmo_default
        if self.cosmo_default is None:
            print("Fixed cosmological parameters from Planck 2018.")
            self.cosmo_default = interpolator.Planck18()
        # k, a
        self.k_arr = np.sort(k_arr)
        self.kpts = len(self.k_arr)
        self.a_arr = np.sort(a_arr)
        self.apts = len(self.a_arr)
        # interp
        self.samples = samples
        self.int_samples_func = int_samples_func
        self.check_cosmo = check_cosmo
        self.interpf = interpf
        self.epsilon = epsilon
        if self.interpf not in ["multiquadric", "inverse", "gaussian"]:
            warnings.warn("epsilon not defined for function %s" % self.interpf)
            self.epsilon = None
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
            W = wts(self.priors, self.cosmo_default,
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
        for i, p in enumerate(tqdm(self.pos, desc="Sampling P(k,a) grid")):
            kw = self.cosmo_default.copy()
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
        lPk = np.log10(Pk).reshape(*np.r_[self.weights, self.apts*self.kpts])
        self.F = [Rbf(*np.c_[self.pos, lPk[..., i].flatten()].T,
                      function=self.interpf,
                      epsilon=self.epsilon)
                  for i in tqdm(range(self.apts*self.kpts), desc="Interpolating")]
        self.F = np.array(self.F).reshape((self.apts, self.kpts))

    def callF(self, *pars):
        """
        Call the interpolators on a list of parameters.

        Arguments
        ---------
        *pars : ``list``
            List of query parameters.
            The final 2 rows should be `a_arr`, `k_arr` in that ordering.
            Caution: a, k order is swapped relative to `pyccl.linear_matter_power`!

        Return
        -------
        Pk : ``numpy.array``
            Cosmological linear power spectrum evaluated at `*pars`.
            Extra dimensions are squeezed out.
        """
        pars, (a_arr, k_arr) = pars[:-2], pars[-2:]
        if not all(np.in1d(a_arr, self.a_arr)):
            raise ValueError("Value(s) in a_arr not matching interpolation.")
        if not all(np.in1d(k_arr, self.k_arr)):
            raise ValueError("Value(s) in k_arr not matching interpolation.")
        # k, a
        a_arr = np.atleast_1d(a_arr).astype(float)
        k_arr = np.atleast_1d(k_arr).astype(float)
        ia = np.searchsorted(self.a_arr, a_arr)
        ik = np.searchsorted(self.k_arr, k_arr)
        # cosmo
        mg = np.meshgrid(*pars, indexing="ij")
        pos = np.vstack(list(map(np.ravel, mg)))

        lPk = np.array([f(*pos) for f in self.F[ia][:, ik].flatten()])
        Pk = 10**lPk.reshape((len(a_arr), len(k_arr), pos.shape[1]))
        return Pk.squeeze()

    def linear_matter_power(self, cosmo, k_arr, a_arr):
        """Interpolated linear matter power spectrum with the same
        function call as `pyccl.linear_matter_power`
        """
        # check if `cosmo` is compatible with interpolation
        if self.check_cosmo:
            fixed = list(set(self.cosmo_default.keys() - set(self.pars)))
            for par in fixed:
                if cosmo[par] != self.cosmo_default[par]:
                    raise ValueError(textwrap.fill(textwrap.dedent("""
            Cosmological parameter %s not compatible with interpolation.
            """ % par)))
        if not all(np.in1d(a_arr, self.a_arr)):
            raise ValueError("Value(s) in a_arr not matching interpolation.")
        if not all(np.in1d(k_arr, self.k_arr)):
            raise ValueError("Value(s) in k_arr not matching interpolation.")

        pars = [cosmo[par] for par in self.pars]
        return self.callF(*pars, a_arr, k_arr)
