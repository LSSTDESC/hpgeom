import numpy as np


__all__ = ['match_arrays']


def match_arrays(arr1, arr2):
    """Match arrays and return indices.

    Parameters
    ----------
    arr1 : `np.ndarray`
    arr2 : `np.ndarray`

    Returns
    -------
    sub1 : `np.ndarray`
        Indices of arr1 which match arr2
    sub2 : `np.ndarray`
        Indices of arr2 which match arr1
    """
    sub1 = np.searchsorted(arr1, arr2)
    sub1[sub1 == arr1.size] = arr1.size - 1
    sub2, = (arr1[sub1] == arr2).nonzero()
    sub1 = sub1[sub2]

    return sub1, sub2
