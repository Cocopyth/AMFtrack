import numpy as np
from amftrack.util.aliases import coord
from typing import List


def is_in(array: coord, array_list: List[coord]):
    """
    Check if array is in the list of array. This function is intended for list of coordinates.
    `array` and element of `array_list` can be either list or numpy arrays.
    """
    for element in array_list:
        if np.all(array == element):
            return True
    return False
