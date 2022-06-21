import numpy as np
from scipy import ndimage
from typing import Tuple, List
import matplotlib.pyplot as plt
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
    Computes the rotation and translation to transform the old plane into the new one.
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


def extract_inscribed_rotated_image(image, angle=10):
    """
    Extract the image that is inscribed in the `image` with a degree angle of `angle` clockwise.
    WARNING: angle mustn't be equal to 45 degrees
    WARNING: depending on image.shape[0]/image.shape[1] an interval around 45 degrees is inaccessible
    """
    bounds = np.degrees(
        np.arctan(
            np.array([image.shape[0] / image.shape[1], image.shape[1] / image.shape[0]])
        )
    )
    bounds = np.array([np.min(bounds), np.max(bounds)])
    if (np.abs(angle) < 46 and np.abs(angle) > 44) or (
        np.abs(angle) < bounds[1] and np.abs(angle) > bounds[0]
    ):
        raise Error("Angle too close to 45 degrees")

    angle_rad = np.radians(angle)
    rotated_image = ndimage.rotate(image, -angle)

    original_X, original_Y = image.shape[0], image.shape[1]
    rotated_X, rotated_Y = rotated_image.shape[0], rotated_image.shape[1]
    final_X = int(
        (np.cos(angle_rad) * original_X - np.abs(np.sin(angle_rad)) * original_Y)
        / np.cos(2 * angle_rad)
    )
    final_Y = int(
        (np.cos(angle_rad) * original_Y - np.abs(np.sin(angle_rad)) * original_X)
        / np.cos(2 * angle_rad)
    )

    dx = int((rotated_X - final_X) / 2)
    dy = int((rotated_Y - final_Y) / 2)
    final = rotated_image[
        dx : -dx - 1, dy : -dy - 1, :
    ]  # case without last layer + -1??
    return final


if __name__ == "__main__":

    R, t = find_transformation(
        [[16420, 26260], [17120, 28480]], [[15760, 26500], [16420, 28780]]
    )
    print(R)
    print(t)
    f = get_transformation(R, t)
    print(f([16420, 26260]))

    assert convert_to_micrometer(10) == 17.25
