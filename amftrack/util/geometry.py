from amftrack.util.aliases import coord
from typing import Tuple
import numpy as np
from typing import List
import bisect


def distance_point_pixel_line(point: coord, line: List[coord], step=1) -> float:
    """
    Compute the minimum distance between the `point` and the `line`.
    The `step` parameter determine how frequently we compute the distance along the line.

    :param line: list of coordinates of points making the line
    There can be several use cases:
    - step == 1: we compute the distance for every point on the line.
    It is computationally expensive but the result will have no error.
    - step == n: we compute the distance every n point on the line.
    The higher the n, the less expensive the function is, but the minimal distance won't be exact.

    NB: With a high n, the error will be mostly important for lines that are closed to the point.
    A point on a line, that should have a distance of 0 could have a distance of n for example.
    """

    pixel_indexes = generate_index_along_sequence(len(line), resolution=step, offset=0)

    # NB: we don't need norm here, the square root is expensive
    d_square = np.min(
        [np.sum(np.square(np.array(point) - np.array(line[i]))) for i in pixel_indexes]
    )

    # NB: we take the square root at the end
    return np.sqrt(d_square)


def get_closest_lines(
    point: coord, lines: List[List[coord]], step=1, n_nearest=3
) -> Tuple[List[int], List[float]]:
    """
    Returns the `n_nearest` closest lines to `point` (in `lines`).
    Along with their respective distances to the `point`.
    :param n_nearest: is the number of closest edge we want returned
    :param step: determines the step at which we check for points along the lines
    If step=1 the result is exact, else the function is faster but not always exact
    (distances are then surestimated)
    :return: (indexes of closest lines, corresponding distances to point)
    """
    l = []
    for i in range(len(lines)):
        d = distance_point_pixel_line(point, lines[i], step=step)
        bisect.insort(l, (d, i))
        if len(l) > n_nearest:
            del l[-1]
    return [index for index, _ in l], [distance for _, distance in l]


def aux_get_closest_line_opt(
    point: coord, lines: List[List[coord]], step=1, n_nearest=1
) -> Tuple[int, float]:
    """ """
    distances = [
        distance_point_pixel_line(point, lines[i], step=step) for i in range(len(lines))
    ]
    l = [(distances[i], i) for i in range(len(lines))]
    d_min = np.min(distances)
    error_margin = 1.4142 * step

    l = [(d, i) for (d, i) in l if d <= d_min + error_margin]
    # NB: as we want only the single closest, no need to order here

    return [index for index, _ in l], [distance for _, distance in l]


def get_closest_line_opt(
    point: coord, lines: List[List[coord]], starting_step=1000
) -> Tuple[int, float]:

    step = starting_step
    while len(lines) > 1 and step > 1:
        # NB: distances could be used to tune the `step` at each iteration
        kept_indexes, distances = aux_get_closest_line_opt(point, lines, step)
        lines = [lines[i] for i in kept_indexes]
        step = step // 2

    line = lines[0]  # NB: Could return a list of all lines at equal distance also
    d = distance_point_pixel_line(point, line, step=1)
    return line, d


# def get_closest_lines_with_error(
#     point: coord, lines: List[List[coord]], step=1
# ) -> Tuple[List[int], List[float]]:
#     """
#     """
#     error = 1.4142
#     l = []
#     for i in range(len(lines)):
#         d = distance_point_pixel_line(point, lines[i], step=step)
#         if d < limit:
#             bisect.insort(l, (d, i))
#             potential_new_limit = d + error * step
#             limit = np.min(
#                 limit, potential_new_limit
#             )  # changes only if d is the new min
#     return [index for index, _ in l], [distance for _, distance in l]


def generate_index_along_sequence(n: int, resolution=3, offset=5) -> List[int]:
    """
    From the length `n` of the list, generate indexes at interval `resolution`
    with `offset` at the start and at the end.
    :param n: length of the sequence
    :param resolution: step between two chosen indexes
    :param offset: offset at the begining and at the end
    """
    x_min = offset
    x_max = n - 1 - offset
    # Small case
    if x_min > x_max:
        return [n // 2]
    # Normal case
    k_max = (x_max - x_min) // resolution
    l = [x_min + k * resolution for k in range(k_max + 1)]
    return l


def get_section_segment(
    orientation: coord, pivot: coord, target_length: float
) -> Tuple[coord, coord]:
    """
    Return the coordinate of a segment giving the section.
    :param orientation: a non nul vector giving the perpendicular to the future segment
    :param pivot: center point for the future segment
    :param target_length: length of the future segment
    :warning: the returned coordinates are float and not int, rounding up will change slightly the length
    """
    perpendicular = (
        [1, -orientation[0] / orientation[1]] if orientation[1] != 0 else [0, 1]
    )
    perpendicular_norm = np.array(perpendicular) / np.sqrt(
        perpendicular[0] ** 2 + perpendicular[1] ** 2
    )
    point1 = np.array(pivot) + target_length * perpendicular_norm / 2
    point2 = np.array(pivot) - target_length * perpendicular_norm / 2

    return (point1, point2)


def expand_segment(point1: coord, point2: coord, factor: float) -> Tuple[coord, coord]:
    """
    From the segment [point1, point2], return a new segment dilated by `factor`
    and centered on the same point.
    :return: the new coordinates for the segment after the dilatation by `factor`
    """
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    vx = x2 - x1
    vy = y2 - y1
    x_mid = (x1 + x2) / 2
    y_mid = (y1 + y2) / 2
    x1_, y1_ = [x_mid - factor * vx / 2, y_mid - factor * vy / 2]
    x2_, y2_ = [x_mid + factor * vx / 2, y_mid + factor * vy / 2]
    return [x1_, y1_], [x2_, y2_]


def compute_factor(point1: coord, point2: coord, target_length: float) -> float:
    """
    Determine the dilatation factor to apply to the segment [point1, point2]
    in order to reach the `target_length`.
    :param point1: coordinates. ex: [2, 5]
    """
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    vx = x2 - x1
    vy = y2 - y1
    length = np.sqrt(vx**2 + vy**2)

    return target_length / length


if __name__ == "__main__":
    point1 = [0, 0]
    point2 = [2, 4]
    assert expand_segment(point1, point2, 0.5) == ([0.5, 1.0], [1.5, 3.0])
    assert expand_segment(point1, point2, 2) == ([-1.0, -2.0], [3.0, 6.0])
    assert expand_segment(point2, point1, 2) == ([3.0, 6.0], [-1.0, -2.0])

    assert compute_factor([0, 0], [0, 3], 6) == 2
    assert compute_factor([1, 0], [10, 0], 4.5) == 0.5

    print(get_section_segment([-2, 2], [1, 1], 2.82))
