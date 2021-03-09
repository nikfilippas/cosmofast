"""
Test the interpolator against CCL's default cosmology emulator (CAMB).

We test the approximation at the grid nodes and expect
high accuracy, but also test it at the midpoints between
the nodes (centres of ND cubes).

#FIXME: blocksize decreases accuracy at the BAO scales??

#TODO: sparse distance matrices to overcome RAM issue (KD-tree not working)
#TODO: sklearn optimal epsilon for all scales ??? (On^2)
#TODO: check Fortran spline routines
#TODO: interpolate slightly outside range to eliminate Runge's phenomenon
#TODO: save/load interpolator
#TODO: overcome memoery constraint by finding neighbours within distance of bump RBF
#TODO: run profiler

#TODO: blocks of interpolators on a, k - constructing & calling :: DONE
#TODO: possibility to interpolate a, k within blocks :: DONE
#TODO: if cosmo is the same, interplate a,k space and do not re-compute :: DONE
#TODO: rescale cosmo parameters to fixed step :: DONE
#TODO: my own RBF (bump gaussian) :: DONE (+ polyharmonic splines)
#TODO: fake nodes (Chebyshev) :: DONE
#TODO: numpy.memmap for full-scale Pk ~65 GB size :: DONE

#TODO: option to quadruple number of interpolators to interpolate all a, k :: NO NEED
#TODO: recale a_arr and k_arr during interpolation with RectBivariate :: NO NEED

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

1) 2 cosmo pars; narrow; cosmo nodes; ka nodes
errors: (min,avg,max)=(5.01e-8,1.30e-7,2.67e-7)
------------------------------------------------------------------------------
2) 2 cosmo pars; narrow; cosmo antinodes; ka nodes
errors: (min,avg,max)=(9.35e-05,2.43e-04,4.36e-04)
------------------------------------------------------------------------------
3) 2 cosmo pars; narrow; cosmo nodes; ka antinodes
errors: (min,avg,max)=(6.26e-03,6.27e-03,6.28e-03)
------------------------------------------------------------------------------
4) 2 cosmo pars; narrow; cosmo antinodes; ka antinodes
errors: (min,avg,max)=(6.24e-03,6.35e-03,6.58e-03)
------------------------------------------------------------------------------
5) 2 cosmo pars; wide; cosmo antinodes; ka nodes
"""
import numpy as np
from numpy.random import uniform
import pyccl as ccl
from interpolator import interpolator
from RBF import RBF_ext
from cosmo import Cosmology
from utils import Planck18
from utils import linear_matter_power as linpow
import matplotlib as mpl
import matplotlib.pyplot as plt

# priors = {"h" : [0.55, 0.91],
#           "sigma8": [0.50, 1.25],
#           "Omega_b": [0.03, 0.07],
#           "Omega_c": [0.1, 0.9],
#           "n_s": [0.87, 1.07]}

priors = {"h" : [0.55, 0.595],
          "sigma8": [0.50, 0.59375],
          "Omega_b": [0.03, 0.0345],
          "Omega_c": [0.1, 0.2],
          "n_s": [0.87, 0.895]}

# epsilon = 0.16
# priors = {"h" : [0.55, 0.91],
#           "sigma8": [0.50, 1.25],
#           "Omega_b": [0.03, 0.07]}

# epsilon = 0.08
priors = {"h" : [0.65, 0.75],
          "sigma8": [0.77, 0.87]}

testnum = 5
test_cosmo = "antinodes"  # {'nodes', 'antinodes', 'random'} - cosmo space
test_ak = "nodes"  # {'nodes', 'antinodes'} - (k,a)-space

## INTERPOLATOR ##
samples = 7**len(priors.keys())
k_arr = np.logspace(-4, 2, 512)
a_arr = np.linspace(0.1, 1, 16)

interp = interpolator(priors,
                      cosmo_default=None,
                      k_arr=k_arr,
                      a_arr=a_arr,
                      samples=100,
                      a_blocksize=1,
                      k_blocksize=1,
                      interpf="gaussian",
                      epsilon=0.08,
                      pStep=0.01,
                      overwrite=False,
                      Pk=None,
                      just_sample=False)
# save
interp.save()
# load
interp = np.load(interp.get_fname("interp"), allow_pickle=True).item()

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
    vals = [np.sort(uniform(*priors[par], size=4)) for par in priors.keys()]
    mgrid = np.meshgrid(*vals)
    points = np.vstack(list(map(np.ravel, mgrid))).T

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
    NICK = csm.linear_matter_power(k_arr, a_arr)
    # errors
    d = 1-NICK/CCL
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
plt.imshow(np.log10(d), aspect="auto", extent=extent)
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
