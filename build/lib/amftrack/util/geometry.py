from xmlrpc.client import Boolean
from amftrack.util.aliases import coord
from typing import Tuple, List
import numpy as np
import bisect
import math


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
    # NB: Could maybe speed up by generating a np array of all the relevant points before using `sum` and `square`
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
    return [index for _, index in l], [distance for distance, _ in l]


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

    return [index for _, index in l], [distance for distance, _ in l]


def get_closest_line_opt(
    point: coord, lines: List[List[coord]], step=1000, factor=2
) -> Tuple[int, float]:
    """
    Returns the closest line to `point` (in `lines`).
    Along with its respective distance to the `point`.

    This version is optimized. We search with different resolution and stop when
    the resolution is good enough to conclude.

    :param step: this is the initial resolution. It will be decreased at each iteration
    :param factor: determines how we change the resolution between two iteration (2, 3, 4 work well)
    :return: (index of closest lines, corresponding distance to point)
    NB: if there are several lines at same distance, we return only one

    In the worst case, this function has (factor/factor-1) the cost of the computing the
    distance to every point in every line.
    The worst case is: all the lines are at same distance.

    The best cases are:
    - the first closest line is much closer than the second (then we stop before reaching a small resolution)
    - a lot of lines are far from the point and will be removed early in the iterations
    """
    # The length of lines evolves over time, d dict keeps track of the original indexes
    mapping = {i: i for i in range(len(lines))}
    while len(lines) > 1 and step >= 1:
        # NB: distances could be used to tune the `step` at each iteration
        kept_indexes, distances = aux_get_closest_line_opt(point, lines, step)
        # Update index directory
        mapping_ = {}
        for i in range(len(kept_indexes)):
            mapping_[i] = mapping[kept_indexes[i]]
        mapping = mapping_
        lines = [lines[i] for i in kept_indexes]
        step = step // factor
    ind = mapping[0]  # NB: Could return a list of all lines at equal distance also
    line = lines[0]
    d = distance_point_pixel_line(point, line, step=1)
    return ind, d


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


def is_overlapping(segment1: coord, segment2: coord, strict=False) -> bool:
    """
    Determines if the two segments are overlapping.
    If they are just touching, it returns True in strict=False mode and False otherwise.
    """
    if strict:
        return segment1[0] < segment2[1] and segment2[0] < segment1[1]
    else:
        return segment1[0] <= segment2[1] and segment2[0] <= segment1[1]


def intersect_rectangle(
    rect1_coord1: coord, rect1_coord2, rect2_coord1, rect2_coord2, strict=False
) -> Boolean:
    """
    Determine if two rectangles are overlapping.
    :param rect1_coord1: [x, y] coordinates of the first point of the first rectangle
    :param rect2_coord2: [x, y] coordinates of the second point of the first rectangle
    :param strict: if strict = false if there are just touching they are considered overlapping
    """
    # intersection along the first dimention
    if is_overlapping(
        [rect1_coord1[0], rect1_coord2[0]],
        [rect2_coord1[0], rect2_coord2[0]],
        strict,
    ):
        # intersection along the second dimention
        if is_overlapping(
            [rect1_coord1[1], rect1_coord2[1]],
            [rect2_coord1[1], rect2_coord2[1]],
            strict,
        ):
            return True
    return False


def get_overlap(
    rect1_coord1: coord, rect1_coord2, rect2_coord1, rect2_coord2, strict=False
) -> Tuple[coord]:
    """
    Returns the overlap of two rectangles.
    The boundaries are included in case of int coordinates.
    """

    if not intersect_rectangle(
        rect1_coord1, rect1_coord2, rect2_coord1, rect2_coord2, strict
    ):
        raise Exception("Can't get overlapp of two rectangles that are not overlapping")

    # along axis 1
    l1 = [rect1_coord1[0], rect1_coord2[0], rect2_coord1[0], rect2_coord2[0]]
    l1.sort()

    # along axis 2
    l2 = [rect1_coord1[1], rect1_coord2[1], rect2_coord1[1], rect2_coord2[1]]
    l2.sort()

    return [[l1[1], l2[1]], [l1[2], l2[2]]]


def format_region(region):
    """
    From [[a, b], [c, d]] defining a region in the plane.
    Return a definition of the region where [a', b'] is the upper left-corner
    and [c', d'] is the lower-right corner (ie c' > a' and d' > b')
    """
    return [
        [min(region[0][0], region[1][0]), min(region[0][1], region[1][1])],
        [max(region[0][0], region[1][0]), max(region[0][1], region[1][1])],
    ]


def get_bounding_box(coodinate_list: List[coord], margin=0) -> List[coord]:
    """
    From a list of coordinates, this function hands out the two points defining
    a rectangle that contains all of the points.

    :param margin: expand the bounding box by `margin` in all direction
    """
    x_min = int(np.min([coord[0] for coord in coodinate_list]))
    x_max = int(np.max([coord[0] for coord in coodinate_list]))
    y_min = int(np.min([coord[1] for coord in coodinate_list]))
    y_max = int(np.max([coord[1] for coord in coodinate_list]))
    strict_bounding_box = [
        [math.floor(x_min), math.floor(y_min)],
        [math.ceil(x_max), math.ceil(y_max)],
    ]
    return expand_bounding_box(strict_bounding_box, margin)


def centered_bounding_box(coordinate: coord, size=200) -> List[coord]:
    """
    From a given coordinate, return the two points defining a square bounding box
    around this point, of size `size`x`size`.
    """
    return get_bounding_box([coordinate], margin=size // 2)


def expand_bounding_box(bounding_box: List[coord], margin: int) -> List[coord]:
    """
    Expand the bounding box in all the direction by `margin`
    """
    bounding_box = format_region(bounding_box)
    [[x_min, y_min], [x_max, y_max]] = bounding_box
    return [[x_min - margin, y_min - margin], [x_max + margin, y_max + margin]]


def is_in_bounding_box(coordinate: coord, bounding_box: List[coord]) -> bool:
    """
    Check if the coordinate `coordinate` is in the bounding box.
    """
    bounding_box = format_region(bounding_box)
    if bounding_box[0][0] <= coordinate[0] < bounding_box[1][0]:
        if bounding_box[0][1] <= coordinate[1] < bounding_box[1][1]:
            return True
    return False


if __name__ == "__main__":
    point1 = [0, 0]
    point2 = [2, 4]
    assert expand_segment(point1, point2, 0.5) == ([0.5, 1.0], [1.5, 3.0])
    assert expand_segment(point1, point2, 2) == ([-1.0, -2.0], [3.0, 6.0])
    assert expand_segment(point2, point1, 2) == ([3.0, 6.0], [-1.0, -2.0])

    assert compute_factor([0, 0], [0, 3], 6) == 2
    assert compute_factor([1, 0], [10, 0], 4.5) == 0.5

    print(get_section_segment([-2, 2], [1, 1], 2.82))
