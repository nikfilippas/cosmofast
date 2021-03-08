"""
Call the interpolator.
"""
import textwrap
from itertools import product
import numpy as np
from scipy.interpolate import RectBivariateSpline


class Cosmology(object):
    """
    Call the interpolator of the cosmological linear matter power spectrum.

    Given a fixed cosmology, construct a cosmology object
    which can be efficiently called multiple times.

    During initialization, the entire (k,a)-space is interpolated using
    a fixed-grid bivariate spline (`scipy.interpolate.RecBivariateSpline`),
    which is a fast `FITPACK` routine.


    Parameters
    ----------
    interp : ``interpolator.interpolator`` object
        And object of the `interpolator` class. Contains the interpolators
        along cosmo-, k-, and a- spaces, as well as information on their
        construction and limitations.
    kw : ``dict``
        Key-value pairs of the queried cosmological parameters.
        Single values shall be used only; if you want to query several
        cosmologies, initialize another instance of this class.
    a_Chb : ``bool``
        Scale a_arr using fake Chebyshev nodes before interpolation.
        No resampling needed. Supresses Runge's phenomenon near the edges.
        Default is `False`.
    k_Chb : ``bool``
        Scale k_arr using fake Chebyshev nodes before interpolation.
        No resampling needed. Supresses Runge's phenomenon near the edges.
        Default is `False`.


    Attributes
    ----------
    All arguments become class attributes.
    Additional attributes are listed below.

    vals : `array_like`
        Interpolated (free) cosmological parameters.
    Fka : ``scipy.interpolate.fitpack2.RectBivariateSpline``
        (k,a)-space interpolator for input cosmological parameters.
    """

    def __init__(self, interp, kw, a_Chb=False, k_Chb=False):
        self.interp = interp
        self.kw = kw
        # interp
        self.a_Chb = a_Chb
        self.k_Chb = k_Chb

        self.vals = [self.kw[par] for par in self.interp.pars]
        self._check_compatibility(self.vals)

        self.interpolate()

    def _check_compatibility(self, vals):
        """Check that all queried points are compatible with the interpolator.
        """
        vals = np.asarray(vals)
        # check fixed parameters
        fixed = list(set(self.interp.cosmo_default.keys() - \
                     set(self.interp.pars)))
        if len(fixed) > 0:
            for par in fixed:
                if self.kw[par] != self.interp.cosmo_default[par]:
                    raise ValueError(textwrap.fill(textwrap.dedent("""
            Parameter %s=%s different to assumed fixed value %s.
            """ % (par, self.kw[par], self.interp.cosmo_default[par]))))
        # check free parameters
        ask = {key: np.atleast_1d(val)
               for key, val in zip(self.interp.pars, vals.T)}
        for par, val in ask.items():
            if not ((val >= self.interp.priors[par][0]).all()
                    and (val <= self.interp.priors[par][1]).all()):
                raise ValueError(textwrap.fill(textwrap.dedent("""
        One or more values in parameter %s outside the interpolation range %s.
        """ % (par, self.interp.priors[par]))))

    def callF(self, *vals):
        """
        Call the cosmological interpolators on a list of parameters.

        Arguments
        ---------
        *vals : ``list``
            List of query parameters.
            The final 2 rows should be `a_arr`, `k_arr` in that ordering.
            Caution: a,k order swapped relative to `pyccl.linear_matter_power`!

        Returns
        -------
        Pk : ``numpy.array``
            Cosmological linear power spectrum evaluated at `*vals`.
            Extra dimensions are squeezed out.
        """
        vals, (a_arr, k_arr) = vals[:-2], vals[-2:]
        vals = np.asarray(vals).squeeze()
        self._check_compatibility(vals)

        if self.interp.a_blocksize == 1 and \
            not all(np.in1d(a_arr, self.interp.a_arr)):
            raise ValueError("Values between nodes in a_arr not interpolated.")
        if self.interp.k_blocksize == 1 and \
            not all(np.in1d(k_arr, self.interp.k_arr)):
            raise ValueError("Values between nodes in k_arr not interpolated.")

        # a,k : which blocks to use
        a_arr = np.atleast_1d(a_arr).astype(float)
        k_arr = np.atleast_1d(k_arr).astype(float)
        a_idx = np.searchsorted(self.interp.a_arr, a_arr) // self.interp.a_blocksize
        k_idx = np.searchsorted(self.interp.k_arr, k_arr) // self.interp.k_blocksize

        # rescale parameters back (order of ifs is important)
        rescale = self.interp.rescale.tolist()
        if self.interp.k_blocksize > 1:
            k_arr = np.power(10, np.log10(self.interp.k_arr) * rescale.pop())
        if self.interp.a_blocksize > 1:
            a_arr = self.interp.a_arr * rescale.pop()
        vals = np.asarray(rescale) * np.atleast_1d(self.vals)

        # cosmo : disassemble cosmological parameters
        points = [val.tolist() for val in np.atleast_2d(vals).T]

        # query values from block grid
        lPk = np.zeros((len(a_arr), len(k_arr)))
        for ia, fa in enumerate(self.interp.F):
            if ia not in a_idx:
                continue
            idx1a, idx2a = self.interp.a_blocksize*np.array([ia, ia+1])
            a_use = a_arr[np.where(a_idx == ia)[0]]
            if self.interp.a_blocksize > 1:
                points.extend([a_use.tolist()])

            for ik, fka in enumerate(fa):
                if ik not in k_idx:
                    continue
                idx1k, idx2k = self.interp.k_blocksize*np.array([ik, ik+1])
                k_use = k_arr[np.where(k_idx == ik)[0]]
                if self.interp.k_blocksize > 1:
                    points.extend([np.log10(k_use).tolist()])

                ask = np.asarray(list(product(*points)))
                block = fka(*ask.T).reshape((self.interp.a_blocksize,
                                             self.interp.k_blocksize))
                lPk[idx1a:idx2a, idx1k:idx2k] = block

                if len(k_use) > 1:
                    points.pop()
            if len(a_use) > 1:
                points.pop()

        return 10**lPk.squeeze()

    @classmethod
    def Chebyshev(Cosmology, arr):
        """Map the input array to Chebyshev-Lovatto nodes to supress
        Runge's phenomenon when interpolating.

        Inspired by Davide Poggiali's 'FakeNodes':
        https://github.com/pog87/FakeNodes
        """
        a, b = arr.min(), arr.max()
        CL = (b-a)/2 * np.cos(np.pi*(arr-a)/(b-a)) + (a+b)/2
        return CL[::-1]  # reverse the mapping
        # return arr

    def interpolate(self):
        """Interpolate (k,a) space for faster evaluation time."""
        vals = np.atleast_1d(self.vals)
        k_arr = self.interp.k_arr
        a_arr = self.interp.a_arr
        lPk = np.log10(self.callF(*vals, a_arr, k_arr))

        # rescale using approximate Chebyshev nodes?
        if self.a_Chb:
            self._aF = Cosmology.Chebyshev
        else:
            self._aF = lambda x: x

        if self.k_Chb:
            self._kF = Cosmology.Chebyshev
        else:
            self._kF = lambda x: x

        self.Fka = RectBivariateSpline(self._aF(a_arr),
                                       self._kF(np.log10(k_arr)),
                                       lPk)

    def linear_matter_power(self, k_arr, a_arr):
        """Interpolated linear matter power spectrum with
        (almost) the same function call as `pyccl.linear_matter_power`.
        Call the (k,a)-space interpolator on any a and k.
        """
        lPk = self.Fka(self._aF(a_arr), self._kF(np.log10(k_arr)))
        return 10**lPk.squeeze()
