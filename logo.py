"""Logo of cosmofast."""
from utils import linear_matter_power as linpow
from utils import Planck18
import pyccl as ccl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

k_arr = np.logspace(-4, 2, 512)

num = 16
# a_arr = np.linspace(0.1, 1, num)
Oc_arr = np.linspace(0.1, 0.9, num)
h_arr = np.linspace(0.55, 0.91, num)
s8_arr = np.linspace(0.5, 1.25, num)
ns_arr = np.linspace(0.87, 1.07, num)

pars = {"Omega_c": Oc_arr,
        "h": h_arr,
        "sigma8": s8_arr,
        "n_s": ns_arr}

Pk_dic = dict.fromkeys(pars.keys())
for par in Pk_dic:
    print("Sampling %s..." % par)
    kw = Planck18()
    Pka = np.zeros((num, len(k_arr)))
    for i, val in enumerate(pars[par]):
        print("  sample %d" % i)
        kw[par] = val
        cosmo = ccl.Cosmology(**kw)
        Pka[i] = linpow(cosmo, k_arr, 1)
    Pk_dic[par] = Pka


lk = np.log10(k_arr)
for par, val in Pk_dic.items():
    Pk_dic[par] = np.log10(val)



#################
#################
cmaps = [cm.Reds, cm.Wistia, cm.Blues, cm.Greens]
colors = [cmp(np.linspace(0.2, 1, num)) for cmp in cmaps]
for cl, (par, Pka) in enumerate(Pk_dic.items()):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_axis_off()
    for i, Pk in enumerate(Pka):
        ax.plot(lk, Pk, c=colors[cl][i])
    fig.tight_layout(pad=0)
    fig.savefig("img/im%d.svg" % cl, bbox_inches="tight", transparent=True)
plt.close("all")
