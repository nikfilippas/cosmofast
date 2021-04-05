"""
Tautological plot of the cosmological parameters.
"""
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

pre = "_P18"  # prefix

# Cosmology
q = np.load("../res/pts%s.npz" % pre)
Oc_arr, Ob_arr, h_arr, s8_arr, ns_arr = [q[f] for f in q.files]
kw = {"Omega_c" : Oc_arr.mean(),
      "Omega_b" : Ob_arr.mean(),
      "h"       : h_arr.mean(),
      "sigma8"  : s8_arr.mean(),
      "n_s"     : ns_arr.mean()}

# Interpolate
q = np.load("../res/grads%s.npz" % pre)
gOc, gOb, gh, gs8, gns = [q[f] for f in q.files]

gOcf = interp1d(Oc_arr, gOc, kind="linear", bounds_error=False, fill_value="extrapolate")
gObf = interp1d(Ob_arr, gOb, kind="linear", bounds_error=False, fill_value="extrapolate")
ghf = interp1d(h_arr, gh, kind="linear", bounds_error=False, fill_value="extrapolate")
gs8f = interp1d(s8_arr, gs8, kind="linear", bounds_error=False, fill_value="extrapolate")
gnsf = interp1d(ns_arr, gns, kind="linear", bounds_error=False, fill_value="extrapolate")

# Plot
fig, ax = plt.subplots()
if gOc[-1]/gOc[0] > 10: ax.set_yscale("log")  # Omega_c has most extreme values
ax.set_xlabel("parameter value", fontsize=14)
ax.set_ylabel("maximum local gradient", fontsize=14)

ax.plot(Oc_arr, gOc, "r", label=r"$\Omega_c$")
ax.plot(Ob_arr, gOb, "orange", label=r"$\Omega_b$")
ax.plot(h_arr, gh, "y", label=r"$h$")
ax.plot(s8_arr, gs8, "g", label=r"$\sigma_8$")
ax.plot(ns_arr, gns, "b", label=r"$n_s$")

ax.plot(kw["Omega_c"], gOcf(kw["Omega_c"]), "ko")
ax.plot(kw["Omega_b"], gObf(kw["Omega_b"]), "ko")
ax.plot(kw["h"], ghf(kw["h"]), "ko")
ax.plot(kw["sigma8"], gs8f(kw["sigma8"]), "ko")
ax.plot(kw["n_s"], gnsf(kw["n_s"]), "ko")
mingrad = np.mean([gOcf(kw["Omega_c"]),
                   gObf(kw["Omega_b"]),
                   ghf(kw["h"]),
                   gs8f(kw["sigma8"]),
                   gnsf(kw["n_s"])])
ax.axhline(y=mingrad, c="k", ls="--")

if pre == "":
    ax.set_title(r"$\max{\left(\frac{\partial P(k,a)}{\partial \lambda}\right)}$ using vanilla $\Lambda$CDM", fontsize=14)
    # add all partial derivatives (not in quadrature)
    # and divide by muber of parameters
    vals = np.linspace(0.01, 2.00, 500)
    Grad = gOcf(vals) + gObf(vals) + ghf(vals) + gs8f(vals) + gnsf(vals)
    ax.plot(vals, Grad/5, "k", lw=2, label=r"$\sum {\frac{\partial P_i(k,a)}{\partial \lambda_i}}$")

ax.legend(loc="upper right", ncol=2, fontsize=12)
fig.savefig("../img/grads%s.pdf" % pre, bbox_inches="tight")
