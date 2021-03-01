"""
Create a cosmological linear matter power spectrum interpolator.
"""
import numpy as np
import pyccl as ccl
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import textwrap
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
    amin, amax : ``float``
        Scale factor boundaries.
    apts : ``int``
        Number of scale factor sampling points.
    lkmin, lkmax: ``float``
        Base-10 log of wavenumber sampling points.
    kpts : ``int``
        Number of wavenumber sampling points.
    samples : ``int``
        Number of interpolation points used by the interpolator.
        Note that since the weights will be used, this number is
        approximate, but the final number will be very close.
    interp_pts : ``int``
        Initial number of sampling points along each
        cosmological dimension.
    prefix : ``str``
        Prefix used to save the output.
    overwrite : ``bool``
        Overwrite any saved output. Defaults to `True`.
    save: ``bool``
        Save the weights in compressed `.npz` format.
    """

    def __init__(self, priors, cosmo_default=None, *,
                 k_arr = None, a_arr=None,
                 amin=0.01, amax=1, apts=2048,
                 lkmin=-4, lkmax=2, kpts=4096,
                 samples=50, interp_pts=16,
                 prefix="", overwrite=True, save=True):
        self.priors = priors
        if cosmo_default is None:
            self.cd = interpolator.Planck18()
        else:
            self.cd = cosmo_default
        self.k_arr = k_arr
        self.a_arr = a_arr
        self.samples = samples
        self.pre = prefix

        # build z-space and k-space
        if self.k_arr is not None:
            self.k_arr = np.atleast_1d(self.k_arr)
            self.lkmin, self.lkmax = np.log10(self.k_arr.take([0, -1]))
            self.kpts = len(self.k_arr)
        else:
            self.lkmin, self.lkmax, self.kpts = lkmin, lkmax, kpts
            # self.k_arr = np.logspace(self.lkmin, self.lkmax, self.kpts)
            self.k_arr = self.get_k_arr()

        if a_arr is not None:
            self.a_arr = np.atleast_1d(self.a_arr)
            self.amin, self.amax = self.a_arr.take([0, -1])
            self.apts = len(self.a_arr)
        else:
            self.amin, self.amax, self.apts = amin, amax, apts
            self.a_arr = np.linspace(self.amin, self.amax, self.apts)

        # calculate parameter weights
        self.pars = self.priors.keys()
        W = wts(self.priors, self.cd,
                k_arr=self.k_arr, a_arr=self.a_arr,
                interp_pts=interp_pts, prefix=self.pre)
        weights_dict = W.get_weights(ref=self.samples,
                                     output=True,
                                     save=save,
                                     overwrite=overwrite)
        self.weights = np.array([weights_dict[par] for par in self.pars])

        # build cosmological parameter space
        points = [np.linspace(*priors[key], ww)
                       for key, ww in zip(self.pars, self.weights)]
        mg = np.meshgrid(*points, indexing="ij")

        # extract parameter coordinates
        self.pos = np.vstack(list(map(np.ravel, mg))).T
        # sample parameter space at weighted axes
        Pk = np.zeros((np.product(self.weights),
                       len(self.a_arr),
                       len(self.k_arr)))
        for i, p in enumerate(tqdm(self.pos, desc="Sampling P(k) grid")):
            kw = self.cd.copy()
            kw.update(dict(zip(self.pars, p)))
            cosmo = ccl.Cosmology(**kw)
            Pk[i] = wts.linear_matter_power(cosmo, self.k_arr, self.a_arr)
        Pk = Pk.reshape(*np.append(self.weights,
                                   [len(self.a_arr),
                                    len(self.k_arr)]))

        # `k_arr` and `P(k)` behave better in logspace
        points.extend([self.a_arr, np.log10(self.k_arr)])
        self.F = RegularGridInterpolator(points, np.log10(Pk))

    @classmethod
    def Planck18(interpolator):
        """Return dictionary of Planck 2018 cosmology."""
        cosmo = {"Omega_c" : 0.2589,
                 "Omega_b" : 0.0486,
                 "h"       : 0.6774,
                 "sigma8"  : 0.8159,
                 "n_s"     : 0.9667}
        return cosmo

    @classmethod
    def repeat(interpolator, f, n, arg):
        """Repeat function n times on argument."""
        from functools import reduce
        def rfunc(arg):
            return reduce(lambda x, _: f(x), range(n), arg)
        return rfunc(arg)

    @classmethod
    def get_BAO_idx(interpolator, Pk):
        """Determine the scales of the acoustic peaks."""
        d2lPk = interpolator.repeat(np.gradient, 2, np.log10(Pk))
        d2lPk = d2lPk[4: -4]  # remove boundary effects
        idx = np.where(np.fabs(d2lPk) > np.max(d2lPk)/5)[0].take([0, -1])
        idx += 4  # back to original indexing
        return idx

    def get_k_arr(self):
        """
        Construct k-array of varying sampling densities.

        Upsample regions of the power spectrum with many features
        and downsample uninteresting regions. Split P(k) in 3
        regions: (start, BAO, end).

        Choice of scale factor does not matter because acoustic
        scales are the same throughout structure growth.

        #TODO: see how BAO peaks change with given priors
        and sacrifice some samples in those regions as well.
        """
        kw = self.cd.copy()
        cosmo = ccl.Cosmology(**kw)
        lk = np.linspace(self.lkmin, self.lkmax, self.kpts)

        # get BAO indices
        Pk = ccl.linear_matter_power(cosmo, np.power(10, lk), 1)
        BAO = interpolator.get_BAO_idx(Pk)
        idx = np.r_[0, BAO, self.kpts-1]

        # find weights using arc lengths of `dP(k)/dk`
        dlPk = np.gradient(np.log10(Pk))
        norm = np.abs((lk[-1]-lk[0])/(dlPk[-1]-dlPk[0]))  # renormalization
        dlPk *= norm
        C_tot = wts.C(lk, dlPk)
        N = [self.kpts*wts.C(lk[i1:i2+1], dlPk[i1:i2+1])/C_tot
                                 for i1, i2 in zip(idx, idx[1:])]
        N = np.round(N).astype(int)
        # fix truncation error --> distribute to the least sampled region
        if np.sum(N) < self.kpts:
            N[np.argmin(N)] += self.kpts-np.sum(N)

        idxs = np.split(np.arange(self.kpts), BAO)
        k = np.hstack([np.logspace(*lk[i.take([0, -1])], num=int(n))
                                           for i, n in zip(idxs, N)])
        return k

    def linear_matter_power(self, cosmo, k_arr, a_arr):
        """Interpolated linear matter power spectrum with the same
        function call as `pyccl.linear_matter_power`.
        """
        # check if `cosmo` is compatible with interpolation
        fixed = list(set(self.cd.keys() - set(self.pars)))
        for par in fixed:
            if cosmo[par] != self.cd[par]:
                raise ValueError(textwrap.fill(textwrap.dedent("""
        Cosmological parameter %s not compatible with interpolation.
        """ % par)))

        a_arr = np.atleast_1d(a_arr).astype(float)
        k_arr = np.atleast_1d(k_arr).astype(float)
        pars = [cosmo[par] for par in self.pars]
        pars.extend([a_arr, np.log10(k_arr)])
        pnt = np.meshgrid(*pars)
        pnt = np.vstack(list(map(np.ravel, pnt))).T
        Pk = self.F(pnt).reshape((len(a_arr), len(k_arr)))
        return 10**Pk.squeeze()
