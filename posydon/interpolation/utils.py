""" Module that contains interpolation utility functions """

__authors__ = [
    "Philipp Moura Srivastava <philipp.msrivastava@northwestern.edu>"
]

import numpy as np


def set_valid(ic, X, classes, interp_in_q):
    """Set binary tracks as (in)valid depending on termination flags."""

    valid = np.zeros(len(ic), dtype=int)

    invalid = np.where((ic == "not_converged") | (ic == "ignored_no_BH") | (ic == "ignored_no_RLO"))[0]
    print(f"Disccarded {invalid.shape[0]} binaries")
    valid[invalid] = -1
    

    which = np.isnan(np.sum(X, axis=1))
    valid[which] = -1
    print(f"Discarded {np.sum(which)} binaries with nans in input values.")
    
    for i, flag in enumerate(classes):
        valid[ic == flag] = i + 1

    if interp_in_q:   # if HMS-HMS grid, take out q = 1
        for i, iv in enumerate(X):
            if(iv[1] == 1):
                valid[i] = -1

    return valid