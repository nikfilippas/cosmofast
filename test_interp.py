"""
Test the interpolator against CCL's default cosmology emulator (CAMB).

#TODO: sparse distance matrices to overcome RAM issue (KD-tree not working)
#TODO: sklearn optimal epsilon for all scales ??? (On^2)
#TODO: check Fortran spline routines
#TODO: interpolate slightly outside range to eliminate Runge's phenomenon
#TODO: overcome memoery constraint by finding neighbours within distance of bump RBF
#TODO: run profiler

#TODO: rescale cosmo parameters to fixed step :: DONE
#TODO: fake nodes (Chebyshev) :: DONE

############ BF priors ##############
# epsilon = 0.16
# priors = {"h" : [0.55, 0.91],
#           "sigma8": [0.50, 1.25],
#           "Omega_b": [0.03, 0.07]}

# epsilon = 0.08
# priors = {"h" : [0.65, 0.75],
#           "sigma8": [0.77, 0.87]}

################################ BENCHMARKS ##################################
==============================================================================
## TIME ##

1) len(a_arr) = 16, len(k_arr) = 512 (2x2 blocking)
---------------------------------------------------
>>> %timeit wts.linear_matter_power(ccl.Cosmology(**kw), k_arr, a_arr)
1.04 s ± 8.59 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit wts.linear_matter_power(cosmo, k_arr, a_arr)
1.38 ms ± 8.1 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit Cosmology(interp, interpolator.Planck18())
199 ms ± 1.11 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit csm.linear_matter_power(k_arr, a_arr)
398 µs ± 2.49 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
==============================================================================

## ACCURACY ##
"""
import numpy as np
from numpy.random import uniform
import pyccl as ccl
from sampler import Sampler
from RBF import RBF_ext
from cosmo import Cosmology
from utils import Planck18
from utils import linear_matter_power as linpow
import matplotlib as mpl
import matplotlib.pyplot as plt

priors = {"Omega_c": [0.10, 0.50],
          "h"      : [0.50, 0.90],
          "Omega_b": [0.02, 0.08]}


testnum = 5
test_cosmo = "antinodes"  # {'nodes', 'antinodes', 'random'} - cosmo space
test_ak = "nodes"  # {'nodes', 'antinodes'} - (k,a)-space

## SAMPLER ##
samples = 100# 7**len(priors.keys())
k_arr = np.logspace(-4, 2, 512)
# k_new = np.linspace(0.01, 0.35, 128)
# k_arr = np.sort(np.r_[k_arr, k_new])
a_arr = np.linspace(0.01, 1, 16)

s = Sampler(priors, k_arr, a_arr,
            cosmo_default=None,
            samples=samples,
            weigh_dims=False,
            overwrite=False)
s.sample()

s.interpolate(a_blocksize=1,
              k_blocksize=1,
              interpf="gaussian",
              scale=3)

# # save
# s.save()
# # load
# interp = np.load(s.get_fname("interp"), allow_pickle=True).item()
interp = s


## PREP TESTING ##
if "nodes" in test_cosmo:
    Xnode = interp.points
    if test_cosmo == "antinodes":
        Xnode = [n[:-1] + np.diff(n)/2 for n in Xnode]
    mgrid = np.meshgrid(*Xnode)
    points = np.vstack(list(map(np.ravel, mgrid))).T
elif test_cosmo == "random":
    # When the number of cosmological dimensions is large,
    # sample randomly from points within the cosmological
    # parameter space. This gives an estimate of the error.
    num = 100
    points = np.vstack([[uniform(*priors[par])
                         for par in priors.keys()]
                        for i in range(num)])

if test_ak == "nodes":
    a_arr = interp.a_arr
    k_arr = interp.k_arr
elif test_ak == "antinodes":
    # linear mean
    a_arr = 0.5*(interp.a_arr[1:] + interp.a_arr[:-1])
    # geometric mean (because we interpolated lk_arr)
    k_arr = (interp.k_arr[1:] * interp.k_arr[:-1])**(0.5)


## TESTING ##
kw = Planck18()
res = []
for pnt in points:
    kw.update(dict(zip(interp.pars, pnt)))  # update cosmology
    # CCL
    cosmo = ccl.Cosmology(**kw)
    CCL = linpow(cosmo, k_arr, a_arr)
    # COSMOFAST
    csm = Cosmology(interp, kw)
    EMU = csm.linear_matter_power(k_arr, a_arr)
    # errors
    d = 1-EMU/CCL
    res.append(d)
    print(np.fabs(d).max())

d = np.mean(np.fabs(res), axis=0)
errs = np.array(np.max(np.fabs(res), axis=(1,2)))

# stats
e1, e2, e3 = errs.min(), errs.mean(), errs.max()
print("errors: (min,avg,max)=(%.2e,%.2e,%.2e)" % (e1, e2, e3))

# Plot X.1
norm = mpl.colors.SymLogNorm(1e-2, base=10)
extent = (np.log10(k_arr[0]), np.log10(k_arr[-1]), a_arr[-1], a_arr[0])
plt.figure()
plt.imshow(d, aspect="auto", extent=extent, interpolation="nearest")
plt.colorbar()
# plt.savefig("benchmarks/accu-%d.1.png" % testnum)

# Plot X.2
plt.figure()
plt.plot(errs)
# plt.savefig("benchmarks/accu-%d.2.png" % testnum)


# plt.figure()
# tst = np.abs(1-NICK/CCL)
# # tst[tst <= 5e-4] = np.nan
# plt.imshow(tst, aspect="auto",
#            extent=(np.log10(interp.k_arr[0]),
#                    np.log10(interp.k_arr[-1]),
#                    a_arr[-1],
#                    a_arr[0]))
# plt.colorbar()
