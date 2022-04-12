from typing import List
import numpy as np


def find_transformation(old_coord_list: List, new_coord_list: List):
    """
    Compute the rotation and translation to transform the old plane into the new one.
    old_coord_list and new_coord_list must contain at least 2 points.
    Ex:
    find_transformation([[16420,26260],[17120, 28480]], [[15760, 26500],[16420, 28780]])
    :return: Two arrays: a rotation and a translation
    """

    H = np.dot(
        np.transpose(np.array(old_coord_list) - np.mean(old_coord_list, axis=0)),
        np.array(new_coord_list) - np.mean(new_coord_list, axis=0),
    )

    U, S, V = np.linalg.svd(H)
    R = np.dot(V, np.transpose(U))
    t = np.mean(new_coord_list, axis=0) - np.dot(R, np.mean(old_coord_list, axis=0))

    return R, t


def get_transformation(R, t):
    return lambda x: np.dot(R, x) + t


if __name__ == "__main__":
    R, t = find_transformation(
        [[16420, 26260], [17120, 28480]], [[15760, 26500], [16420, 28780]]
    )
    f = get_transformation(R, t)
    print(f([16420, 26260]))
