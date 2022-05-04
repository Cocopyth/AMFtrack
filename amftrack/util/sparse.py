from typing import List
from amftrack.util.aliases import coord_int


def dilate_coord_list(coordinates: List[coord_int], iteration=1) -> List[coord_int]:
    """
    This functions add to the list of coordinates all points that are neigboors to
    points in the original coord list.
    It is equivalent to cv.dilate with a kernel of 3.
    :param iteration: iteration of 2 is like a kernel of 5, 3 is a kernel of 7, ..
    """

    if iteration == 0:
        return coordinates
    else:
        new_coordinates = []
        for c in coordinates:
            positions = [
                c,
                [c[0], c[1] + 1],
                [c[0], c[1] - 1],
                [c[0] - 1, c[1] - 1],
                [c[0] + 1, c[1] - 1],
                [c[0] + 1, c[1] + 1],
                [c[0] - 1, c[1] + 1],
                [c[0] - 1, c[1]],
                [c[0] + 1, c[1]],
            ]
            for position in positions:
                if position not in new_coordinates:
                    new_coordinates.append(position)

    return dilate_coord_list(new_coordinates, iteration=iteration - 1)
