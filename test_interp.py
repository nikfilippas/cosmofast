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


priors = {"h" : [0.65, 0.75],
          "sigma8": [0.77, 0.87],
          "Omega_b": [0.01, 0.10]}
samples = 7**len(priors.keys())
k_arr = np.logspace(-4, 2, 512)
a_arr = np.linspace(0.1, 1, 16)

q = interpolator(priors, samples=250,
                 k_arr=k_arr, a_arr=a_arr,
                 prefix="3-free", overwrite=True,
                 apts=512, kpts=512,
                 interp_pts=16)

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
errs = []
for pnt in points:
    # initiate new cosmology
    kw.update(dict(zip(q.pars, pnt)))
    cosmo = ccl.Cosmology(**kw)

    approx = q.linear_matter_power(cosmo, k_arr, a_arr)
    CAMB = wts.linear_matter_power(cosmo, k_arr, a_arr)

    err = np.fabs(100*(1-approx/CAMB)).max()
    errs.append(err)
    print(err)
errs = np.array(errs)

d = 100*(1-approx/CAMB)


norm = mpl.colors.SymLogNorm(1e-2, base=10)
plt.imshow(d, aspect="auto",
           extent=(np.log10(k_arr[0]),
                   np.log10(k_arr[-1]),
                   a_arr[-1],
                   a_arr[0]))
plt.colorbar()
