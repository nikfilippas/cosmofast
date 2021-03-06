"""Sample P(k,a) with wide cosmological priors."""
import numpy as np
from interpolator import interpolator

priors = {"h" : [0.55, 0.91],
          "sigma8": [0.50, 1.25],
          "Omega_b": [0.03, 0.07],
          "Omega_c": [0.1, 0.9],
          "n_s": [0.87, 1.07]}

## INTERPOLATOR ##
wpts = 32
samples = 16**len(priors.keys())
k_arr = np.logspace(-4, 2, 512)
a_arr = np.linspace(0.1, 1, 32)

interp = interpolator(priors,
                      cosmo_default=None,
                      k_arr=k_arr,
                      a_arr=a_arr,
                      samples=samples,
                      wpts=wpts,
                      overwrite=False,
                      just_sample=True)
