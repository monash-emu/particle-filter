import numpy as np
#from scipy.stats import truncnorm

from jax import numpy as jnp
from jax import scipy as jsp
from jax.scipy.stats import truncnorm


# Calculation of importance weights
def truncnorm_importance(p_vals, mean, target_sd):
    zero_trunc_vals = -p_vals / target_sd
    target = jnp.repeat(mean, len(p_vals))
    return truncnorm.pdf(target, zero_trunc_vals, jnp.repeat(jnp.inf, len(p_vals)), loc=p_vals, scale=target_sd)
