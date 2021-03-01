"""
Refined k_arr for automatic creation.
"""
import numpy as np
import pyccl as ccl
from weights import weights as wts

class interpolator(object):
    def __init__(self,
                 amin=0.01, amax=1, apts=2048,
                 lkmin=-4, lkmax=2, kpts=4096):
        return None

    def interpolate(self):
        lPk = None  # calculate `P(k,a)`
        if self.method == "linear":
            from scipy.interpolate import RegularGridInterpolator
            points = self.points.copy()
            points.extend([self.a_arr, np.log10(self.k_arr)])
            self.F = RegularGridInterpolator(points, lPk)

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

        Split P(k) in 3 regions: (start, BAO, end).
        Upsample regions of the power spectrum with many features
        and downsample uninteresting regions.

        Choice of scale factor does not matter because acoustic
        scales remain the same throughout structure growth.

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
