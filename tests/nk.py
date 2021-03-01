"""
Testing the optimal number of wavenumber (k) points and redshift (z) points.
"""
import pyccl as ccl
import numpy as np
from scipy.interpolate import interp2d
from tqdm import tqdm

def linear_matter_power(cosmo, k_arr, a_arr):
    """Re-definition of `pyccl.linear_matter_power`
    to accept `array-like` scale factor.
    """
    a_arr = np.atleast_1d(a_arr)
    Pk = np.array([ccl.linear_matter_power(cosmo, k_arr, a) for a in a_arr])
    return Pk.squeeze()

cosmo = ccl.Cosmology(Omega_c=0.25,
                      Omega_b=0.05,
                      sigma8=0.81,
                      n_s=0.96,
                      h=0.67)

# scale factor
lzmin, lzmax = (-3, 2)
a0 = 1/(1+np.logspace(-3, 2, 400))
# wavenumber
lkmin, lkmax = (-4, 2)
k0 = np.logspace(lkmin, lkmax, 10000)
# true P(k,a)
Pk0 = linear_matter_power(cosmo, k0, a0)

nk_arr = np.arange(5000, 0, -50)  # number of k points
na_arr = np.arange(400, 0, -20)   # number of a points
err_arr = np.zeros((len(nk_arr), len(na_arr)))  # error space
for i_k, nk in enumerate(tqdm(nk_arr)):
    for i_a, na in enumerate(na_arr):
        k_arr = np.logspace(lkmin, lkmax, nk)
        a_arr = 1/(1+np.logspace(lzmin, lzmax, na))

        Pk = linear_matter_power(cosmo, k_arr, a_arr)
        Pkf = interp2d(k_arr, a_arr, Pk, kind="linear")

        err_arr[i_k, i_a] = np.max(np.abs(1 - np.flipud(Pkf(k0, a0))/Pk0))


import matplotlib.pyplot as plt
from matplotlib import cm
fig, (ax, cax) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [30, 1]})
ax.set_xlabel("number of k points", fontsize=14)
ax.set_ylabel("number of z points", fontsize=14)

levels = [-4, -3, -2]
im = ax.imshow(np.log10(err_arr.T),
               extent=(nk_arr[0], nk_arr[-1], na_arr[-1], na_arr[0]),
               aspect="auto",
               cmap=cm.inferno)
C = ax.contour(nk_arr, na_arr, np.log10(err_arr.T),
               levels=levels,
               colors="palegreen",
               linestyles="solid",
               linewidths=2)
ax.clabel(C, levels=levels, fmt="%1.f")
cb = fig.colorbar(im, cax=cax, ticks=levels, orientation="vertical")
cb.ax.set_ylabel(r"$\log_{10} \frac{P(k)}{P_0(k)}$", fontsize=12)

fig.tight_layout()
fig.savefig("img/npts.pdf", bbox_inches="tight")
