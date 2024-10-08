import numpy as np
from scipy.stats import truncnorm


# Calculation of importance weights
def truncnorm_importance(p_vals, mean, target_sd):
    zero_trunc_vals = -p_vals / target_sd
    target = np.repeat(mean, len(p_vals))
    return truncnorm.pdf(target, zero_trunc_vals, np.inf, loc=p_vals, scale=target_sd)
