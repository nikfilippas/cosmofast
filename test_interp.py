"""
Test the interpolator against CCL's default cosmology emulator (CAMB).

We test the approximation at the grid nodes and expect
high accuracy, but also test it at the midpoints between
the nodes (centres of ND cubes).
"""
import numpy as np
from numpy.random import uniform
import pyccl as ccl
from weights import weights as wts
from interpolator import interpolator
import matplotlib as mpl
import matplotlib.pyplot as plt

# priors = {"h" : [0.55, 0.91],
#           "sigma8": [0.50, 1.25],
#           "Omega_b": [0.03, 0.07],
#           "Omega_c": [0.1, 0.9],
#           "n_s": [0.87, 1.07]}

priors = {"h" : [0.55, 0.91],
          "sigma8": [0.50, 1.25],
          "Omega_b": [0.03, 0.07]}

# priors = {"h" : [0.65, 0.75],
#           "sigma8": [0.77, 0.87]}

samples = 7**len(priors.keys())
k_arr = np.logspace(-4, 2, 512)
a_arr = np.linspace(0.1, 1, 16)

q = interpolator(priors,
                 samples=500,
                 k_arr=k_arr,
                 a_arr=a_arr,
                 interpf="gaussian",
                 epsilon=0.1,
                 prefix="rbf-gauss-3-wide",
                 overwrite=False)

node = [np.linspace(*priors[par], num=w) for par, w in zip(priors, q.weights)]
anti = [n[:-1] + np.diff(n)/2 for n in node]
mgrid = np.meshgrid(*anti)
points = np.vstack(list(map(np.ravel, mgrid))).T

# vals = [np.sort(uniform(*priors[par], size=4))
#                         for par in priors.keys()]
# mgrid = np.meshgrid(*vals)
# points = np.vstack(list(map(np.ravel, mgrid))).T

# linear mean
# a_mid = 0.5*(q.a_arr[1:] + q.a_arr[:-1])
# a_arr = np.sort(np.append(q.a_arr, a_mid))
# # geometric mean (because we interpolated lk_arr)
# k_mid = (q.k_arr[1:] * q.k_arr[:-1])**(0.5)
# k_arr = np.sort(np.append(q.k_arr, k_mid))

kw = interpolator.Planck18()
res = []
for pnt in points:
    # initiate new cosmology
    kw.update(dict(zip(q.pars, pnt)))
    cosmo = ccl.Cosmology(**kw)

    approx = q.linear_matter_power(cosmo, k_arr, a_arr)
    CAMB = wts.linear_matter_power(cosmo, k_arr, a_arr)

    d = 100*(1-approx/CAMB)
    res.append(d)
    print(np.fabs(d).max())



errs = np.array(np.max(np.fabs(res), axis=(1,2)))
d = np.mean(np.fabs(res), axis=0)


norm = mpl.colors.SymLogNorm(1e-2, base=10)
plt.figure()
plt.imshow(d, aspect="auto",
           extent=(np.log10(k_arr[0]),
                   np.log10(k_arr[-1]),
                   a_arr[-1],
                   a_arr[0]))
plt.colorbar()
