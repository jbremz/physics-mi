import os
import random

import numpy as np
import torch


def set_all_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def unique(x, dim=0):
    """
    Returns the unique elements of a tensor along a specified dimension.
    (required because torch.unique does not preserve order of appearance and doesn't provide indices (https://github.com/pytorch/pytorch/issues/36748))

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension along which to find unique elements. Default is 0.

    Returns:
        tuple: A tuple containing:
            - unique (torch.Tensor): The unique elements of the input tensor.
            - inverse (torch.Tensor): The indices that reconstruct the input tensor from the unique tensor.
            - counts (torch.Tensor): The number of occurrences of each unique element.
            - index (torch.Tensor): The indices that sort the unique tensor in ascending order.

    """
    unique, inverse, counts = torch.unique(
        x, dim=dim, sorted=True, return_inverse=True, return_counts=True
    )
    inv_sorted = inverse.argsort(stable=True)
    tot_counts = torch.cat((counts.new_zeros(1), counts.cumsum(dim=0)))[:-1]
    index = inv_sorted[tot_counts]
    return unique, inverse, counts, index
