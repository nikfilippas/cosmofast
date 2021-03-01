from interpolator import interpolator

# Estimate how many points you can run...
npts = None
# ...or fill in the folloiwng variables.
comps_sec_core = 0.3
sec_day = 86400
cores = 48
hours = 2
days = 0

if npts is not None:
    samples = npts
else:
    samples = int(comps_sec_core*sec_day*cores*(days+hours/24))

# global

## PRIORS ##
# key : [p0, vmin, vmax]
# priors = {"Omega_c" : [0.2589, 0.23, 0.30],
#           "Omega_b" : [0.0486, 0.04, 0.06],
#           "h"       : [0.6774, 0.65, 0.75],
#           "sigma8"  : [0.8159, 0.77, 0.87],
#           "n_s"     : [0.9667, 0.94, 0.98]}

samples = 100
priors = {"h" : [0.65, 0.75], "sigma8": [0.77, 0.87]}

q = interpolator(priors, samples=samples, spacing="linear",
                 prefix="_splinear", overwrite=False)
