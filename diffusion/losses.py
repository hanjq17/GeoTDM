"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np

import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def gaussian_log_likelihood(x, *, means, log_scales, batch, num_nodes, subspace_dim_reduce=0):
    """
    Compute the log-likelihood of a Gaussian distribution
    """
    assert x.shape == means.shape == log_scales.shape
    n_dim = num_nodes * x.size(1) * x.size(2) - subspace_dim_reduce  # [B]
    n_dim = n_dim[batch]  # [BN]
    constants = n_dim * (log_scales[:, 0, 0] + 0.5 * np.log(2 * np.pi))  # [BN]
    constants = constants / num_nodes[batch]  # [BN], divide by the number of nodes to avoid repetitive compute
    # constants = (n_dim * 0.5 * np.log(2 * np.pi)) / num_nodes[batch]  # [BN]
    # term = 0.5 * ((x - means) ** 2) / th.exp(2 * log_scales) + log_scales  # [BN, 3, T]
    term = 0.5 * ((x - means) ** 2) / th.exp(2 * log_scales)  # [BN, 3, T]
    return constants, term

