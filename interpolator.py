"""
Create a cosmological linear matter power spectrum interpolator.
"""
import warnings
import textwrap
import psutil
from itertools import product
from tqdm import tqdm
import numpy as np
import pyccl as ccl
from scipy.interpolate import Rbf
from scipy.linalg.misc import LinAlgWarning
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
    interpf : ``str``
        The radial basis function. See `scipy.interpolate.Rbf`.
    epsilon : ``float``
        Adjustment knob for gaussian and multiquadric functions.
        A good start for cosmological interpolation is 50x the
        average distance between the nodes.  #FIXME: new optimal epsilon
        See `scipy.interpolate.Rbf`.
    a_blocksize : ``float``
        Bin `a_arr` in blocks of that size.
        Increases P(k,a) evaluation speed in expense of memory and
        interpolator construction time. Defaults to no binning.
        If memory is an issue, prefer to bin in k-blocks (use `k_blocksize`)
        instead, as function calls are usually made at a single value of a.
    k_blocksize : ``float``
        Bin `k_arr` in blocks of that size, as in `a_blocksize`.
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
    _logger : ``numpy.array``
        Array of same shape as `F`, where `scipy.linalg` warnings are logged.
        These warnings are raised during (pseudo-) inversion of matrices
        in the RBF calculation, when the reciprocal condition number is `<<1`.
        0 if no warning has been raised, 1 otherwise.
    """

    def __init__(self, priors, cosmo_default=None,
                 k_arr=None, a_arr=None, samples=50, *,
                 int_samples_func="ceil",
                 interpf="gaussian", epsilon=None,
                 a_blocksize=None, k_blocksize=None,
                 weigh_dims=True, wpts=None,
                 prefix="", overwrite=True, save=True):
        # cosmo params
        self.priors = priors
        self.pars = list(self.priors.keys())
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
        self.interpf = interpf
        self.epsilon = epsilon
        if self.interpf not in ["multiquadric", "inverse", "gaussian"] \
           and self.epsilon is not None:
            warnings.warn("epsilon not defined for function %s" % self.interpf)
            self.epsilon = None
        self.a_blocksize = 1 if a_blocksize is None else a_blocksize
        if self.apts % self.a_blocksize != 0:
            raise ValueError("blocksize should divide a_arr exactly")
        self.k_blocksize = 1 if k_blocksize is None else k_blocksize
        if self.kpts % self.k_blocksize != 0:
            raise ValueError("blocksize should divide k_arr exactly")
        # weights
        self.weigh_dims = weigh_dims
        self.wpts = wpts
        if self.weigh_dims and (self.wpts is None):
            warnings.warn("wpts not set; defaulting to 16 per dimension")
            self.wpts = 16
        # I/O
        self.pre = prefix
        self.overwrite = overwrite
        self.save = save

        # confirm enough available memory
        mn = 4*(self.samples*self.a_blocksize*self.k_blocksize)**2 / 1024**3
        ma = psutil.virtual_memory()[1] / 1024**3

        if mn > ma:
            warnings.warn(textwrap.fill(textwrap.dedent("""
            Need ~%.1f GB of memory for RBF distance calculation
            but only %.1f GB are available. Reduce total blocksize
            at least %d times.
            """ % (mn, ma, int(mn/ma)))))
        elif mn/ma < 0.85:
            print(textwrap.fill(textwrap.dedent("""
            PSA: With current memory usage, total blocksize can be
            increased %.1f times. This will result to %.1f faster evaluation
            time when calling the interpolator at a single scale factor.
            The `cosmo.Cosmology` object could be instantiated ~%d times
            faster than the time it takes CCL to computes distances.
            """ % (ma/mn, np.sqrt(ma/mn), 3*ma/mn))))

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
                                         int_samples_func=self.int_samples_func,
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

        Overcome memory constraints in the calculation of
        the metric distance, by taking advantage of `a_arr` and `k_arr`
        being fixed and construct `apts*kpts` independent interpolators.

        Increase evaluation speed of P(k,a) interpolators by block binning.

        Achieve smoother interpolation over the extent of the power spectrum
        by interpolating :math:`log_{10}(Pk)` and :math:`log_{10}(k)`.
        """
        lPk = np.log10(Pk)
        points = [pnts.tolist() for pnts in self.points]

        # determine a, k number of blocks in each dimension
        Na = self.apts//self.a_blocksize
        Nk = self.kpts//self.k_blocksize

        # find block boundaries
        ablocks = np.asarray(np.split(self.a_arr, Na))
        kblocks = np.asarray(np.split(np.log10(self.k_arr), Nk))

        # convenience functions
        rbf = lambda x: Rbf(*x.T, function=self.interpf, epsilon=self.epsilon)
        reset_logger = lambda: np.zeros((Na, Nk), dtype=int)
        def func(points, lPka):
            """Interpolate and log warnings."""
            pos = list(product(*points))  # build grid
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                interp = rbf(np.c_[pos, lPka.flatten()])
                assert len(w) in [0, 1]
                if len(w) > 0:
                    assert issubclass(LinAlgWarning, w[0].category)
                    self._logger[ia, ik] = 1
            self.F[ia, ik] = interp
            pbar.update(1)

        # interpolation on block grid
        self._logger = reset_logger()
        self.F = np.empty((Na, Nk), dtype="object")
        with tqdm(total=Na*Nk, desc="Interpolating") as pbar:
            for ia, ab in enumerate(ablocks):
                idx1a, idx2a = self.a_blocksize*np.array([ia, ia+1])
                if self.a_blocksize > 1:
                    points.extend([ab.tolist()])

                for ik, kb in enumerate(kblocks):
                    idx1k, idx2k = self.k_blocksize*np.array([ik, ik+1])
                    if self.k_blocksize > 1:
                        points.extend([kb.tolist()])

                    # can't interpolate dims of size 1 - squeeze them out
                    lPka = lPk[..., idx1a:idx2a, idx1k:idx2k].squeeze()
                    func(points, lPka)

                    if self.k_blocksize > 1:
                        points.pop()
                if self.a_blocksize > 1:
                    points.pop()
