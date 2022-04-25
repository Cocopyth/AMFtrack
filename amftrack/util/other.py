from amftrack.util.aliases import coord
from typing import Tuple
import numpy as np
from typing import List


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
