import numpy as np
from typing import Tuple, List
from amftrack.util.param import DIM_X, DIM_Y, CAMERA_RES


def convert_to_micrometer(pixel_length, magnification=2):
    """
    Converts pixels into micrometers, based on the magnification of the microscope.
    """
    return pixel_length * CAMERA_RES / magnification


def is_in_image(x_im: float, y_im: float, x: float, y: float) -> bool:
    """
    Determines if (x,y) is in the image of coordinates (x_im, y_im)
    """
    return x >= x_im and x < x_im + DIM_X and y >= y_im and y < y_im + DIM_Y


def find_image_index(im_coord_list, x: float, y: float):
    """
    Find the first image that contains the coordinates (x, y)
    """
    for (i, (x_im, y_im)) in enumerate(im_coord_list):
        if is_in_image(x_im, y_im, x, y):
            return i


def find_image_indexes(im_coord_list, x: float, y: float):
    """
    Find the images that contain the coordinates
    contained in the list of (x, y) coordinates.
    """
    l = []
    for (i, (x_im, y_im)) in enumerate(im_coord_list):
        if is_in_image(x_im, y_im, x, y):
            l.append(i)
    return l


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
    """Returns the function to apply to the points to apply the transformation"""
    return lambda x: np.dot(R, x) + t


def reverse_transformation(R, t) -> Tuple[np.array, np.array]:
    """
    Return Rotation and translation to have the reverse transformation
    :param R: is a matrix (2,2)
    :param t: is a matrix (2,)
    """
    return np.linalg.inv(R), -t


if __name__ == "__main__":

    R, t = find_transformation(
        [[16420, 26260], [17120, 28480]], [[15760, 26500], [16420, 28780]]
    )
    print(R)
    print(t)
    f = get_transformation(R, t)
    print(f([16420, 26260]))

    assert convert_to_micrometer(10) == 17.25
