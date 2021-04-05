"""
Create a cosmological linear matter power spectrum interpolator.
"""
import os
import warnings
import textwrap
import psutil
from itertools import product
from tqdm import tqdm
import pyccl as ccl
import numpy as np
from scipy.interpolate import Rbf
from scipy.linalg.misc import LinAlgWarning
from weights import Weights
from utils import linear_matter_power as linpow


class Sampler(object):
    """
    Sample and interpolate the cosmological linear matter power spectrum.

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

    Interpolated parameters are rescaled so that a single value
    of epsilon works for all dimensions (kernel is n-spherical).


    Parameters
    ----------
    priors : ``dict`` (key: [vmin, vmax])
        Upper and lower parameter boundaries.
    k_arr : ``numpy.array``
        Wavenumbers to sample at.
    a_arr : ``numpy.array``
        Scale factors to sample at.
    cosmo_default : ``dict`` (key: val)
        Fiducial fixed cosmological parameters.
        If `None`, defaults to Planck 2018.
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
    weigh_dims : ``bool``
        Distribute available samples along the cosmological
        dimensions using weights, according to how much `P(k,a)`
        changes in the interval of the passed prior.
        Defaults to `True`. Will independently sample each
        cosmological dimension.
    wpts : ``int``
        Initial number of sampling points along each
        cosmological dimension.
    overwrite : ``bool``
        Overwrite any saved output. Every choice of fixed cosmological
        parameters (`cosmo_default`), sampled cosmological parameters
        with linear spacing (`wpts`), and computed fractional error of the
        power spectrum (`Pk`) gets its own unique code, so it can be reused.
        Defaults to `False`.


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
    """

    def __init__(self, priors, k_arr, a_arr,
                 cosmo_default=None,
                 samples=100, int_samples_func="ceil", *,
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
        self.get_weights()

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

            if self.wpts is None:
                warnings.warn("wpts not set; defaulting to 16 per dimension")
                self.wpts = 16

            W = Weights(self.priors, self.cosmo_default,
                        k_arr=self.k_arr, a_arr=self.a_arr,
                        wpts=self.wpts)
            weights_dict = W.get_weights(ref=self.samples,
                                         int_samples_func=self.int_samples_func,
                                         output=True,
                                         overwrite=self.overwrite)
            self.weights = np.array([weights_dict[par] for par in self.pars])

            if (self.weights == 1).any():
                warnings.warn(textwrap.fill(textwrap.dedent("""
                Very small number of samples. Parameter space will not be
                adequately sampled. Increasing the number of samples from 1
                to 2 in parameter(s) %s.
                """ % [par for i, par in enumerate(self.pars)
                        if self.weights[i] == 1])))
                self.weights[self.weights == 1] += 1

        else:
            ifunc = getattr(np, self.int_samples_func)
            self.wpts = ifunc(self.samples**(1/len(self.pars))).astype(int)
            self.weights = np.repeat(self.wpts, len(self.pars))

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

    def _initialize(self,
                    a_blocksize, k_blocksize,
                    interpf, scale):
        self.a_blocksize = a_blocksize
        self.k_blocksize = k_blocksize
        self.interpf = interpf
        self.scale = scale
        self.pStep = 0.01  # global uniform stepsize
        self.epsilon = self.scale*self.pStep

        if self.apts % self.a_blocksize != 0:
            raise ValueError("blocksize should divide a_arr exactly")
        if self.kpts % self.k_blocksize != 0:
            raise ValueError("blocksize should divide k_arr exactly")

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
            warnings.warn(textwrap.fill(textwrap.dedent("""
            PSA: With current memory usage, total blocksize can be
            increased %.1f times. This will result to %.1f faster evaluation
            time when calling the interpolator at a single scale factor.
            The `cosmo.Cosmology` object could be instantiated ~%d times
            faster than the time it takes CCL to computes distances.
            """ % (ma/mn, np.sqrt(ma/mn), 3*ma/mn))))

    def interpolate(self,
                    a_blocksize=1, k_blocksize=1,
                    interpf="gaussian", scale=3):
        """
        Interpolate the sampled points.


        Arguments
        ----------
        a_blocksize : ``float``
            Bin `a_arr` in blocks of that size.
            Increases P(k,a) evaluation speed in expense of memory and
            interpolator construction time. Defaults to no binning.
            If memory is an issue, bin in k-blocks (use `k_blocksize`) instead,
            as function calls are usually made at a single value of a.
        k_blocksize : ``float``
            Bin `k_arr` in blocks of that size, as in `a_blocksize`.
        interpf : ``str``
            The radial basis function. See `scipy.interpolate.Rbf`.
        scale : ``float``
            Length-scale parameter passed into the RBF kernel.


        Attributes
        ----------
        F : ``numpy.array`` of ``scipy.interpolate.rbf.Rbf`` (apts, kpts)
            Array containing the interpolators for each (k, a) combination.
        pStep : ``float``
            Uniform stepsize for all interpolated parameters.
        epsilon : ``float``
            Adjustment knob for gaussian and multiquadric functions.
            Calculated from `scale` through `pStep`.

        rescale : ``numpy.array``
            Rescale by factor. All interpolated dimensions are rescaled
            before interpolation to get an n-spherical kerne with the same
            shape parameter (epsilon) in every dimension. In this way,
            epsilon works as a tuning parameter for the entire interpolation.
            Elements are ordered according to `pars`.
            If any of the (a, k) blocksizes are not 1 (i.e. they are
            also interpolated), the respective parameter is also rescaled.
        _logger : ``numpy.array``
            Flag singular RBF interpolation matrices.
            Very large condition numbers effectively means that matrices
            are singular as far as default floating point format is concerned
            (float64 - equivalent to C double). In that case LU decomposition
            fails and the pseudoinverse is computed instead.
        """
        self._initialize(a_blocksize, k_blocksize, interpf, scale)

        def _rescale(points, a_arr, lk_arr):
            """Collect the entire rescaling routine in here."""
            # rescale parameters to uniform stepsize
            pars = self.pars.copy()
            if self.a_blocksize > 1:
                points.extend([self.a_arr.tolist()])
                pars.append("a")
            if self.k_blocksize > 1:
                points.extend([lk_arr.tolist()])
                pars.append("k")
            steps = np.array([p[1]-p[0] for p in points])
            self.rescale = self.pStep/steps
            points = [(r*np.asarray(p)).tolist()
                      for r, p in zip(self.rescale, points)]
            # verify everything works
            for par, pnt in zip(pars, points):
                assert np.allclose(np.diff(pnt), self.pStep), \
                "Rescaling parameter %s failed. Is it linearly spaced?" % par

            # redefinitions of parameters (order of ifs is important)
            if self.k_blocksize > 1:
                lk_arr = np.asarray(points.pop())
            if self.a_blocksize > 1:
                a_arr = np.asarray(points.pop())

            return points, a_arr, lk_arr

        # define memory map parameters
        f_err = self.get_fname("err")
        shp = tuple(np.r_[self.weights, self.apts, self.kpts].tolist())
        err = np.memmap(f_err, dtype=float, shape=shp, mode="c")

        # get parameter space
        points = [pnts.tolist() for pnts in self.points]
        a_arr = self.a_arr
        lk_arr = np.log10(self.k_arr)
        points, a_arr, lk_arr =  _rescale(points, a_arr, lk_arr)

        # determine a, k number of blocks in each dimension
        Na = self.apts // self.a_blocksize
        Nk = self.kpts // self.k_blocksize

        # find block boundaries
        ablocks = np.asarray(np.split(a_arr, Na))
        kblocks = np.asarray(np.split(lk_arr, Nk))

        # convenience functions
        rbf = lambda x: Rbf(*x.T, function=self.interpf, epsilon=self.epsilon)
        reset_logger = lambda: np.zeros((Na, Nk), dtype=int)
        def func(points, A):
            """Interpolate and log warnings."""
            pos = list(product(*points))  # build grid
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                interp = rbf(np.c_[pos, A.flatten()])
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

                    A = err[..., idx1a:idx2a, idx1k:idx2k]
                    A = A.squeeze()  # squeeze out dims of size 1
                    func(points, A)

                    if self.k_blocksize > 1:
                        points.pop()
                if self.a_blocksize > 1:
                    points.pop()

        del err  # close memory mapping

    def save(self, path=None):
        """Save the class instance to an '.npy' file with pickle."""
        if path is None:
            # blocks = "_ablk_%d_kblk_%d" % (self.a_blocksize, self.k_blocksize)
            blocks = ""
            path = self.get_fname("interp"+blocks)
            self.interp_path = path
        np.save(path, self, allow_pickle=True)
