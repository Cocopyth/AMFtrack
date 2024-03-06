from amftrack.util.aliases import coord
from typing import List


def coord_list_to_str(l: List[coord], delimiter="X") -> str:
    """
    This function encode a list of coordinates to a string.
    """
    result = ""
    for coordinate in l:
        result += str(coordinate[0])
        result += delimiter
        result += str(coordinate[1])
        result += delimiter
    return result


def str_to_coord_list(string: str, delimiter="X") -> List[coord]:
    """
    This function decode a string representing a list and encoded
    with coord_list_to_str function.
    """
    coord_list = []
    numbers = string.split(delimiter)
    for i in range(len(numbers)):
        if numbers[i]:
            if i % 2 == 0:
                new = [float(numbers[i])]
            elif i % 2 == 1:
                new.append(float(numbers[i]))
                coord_list.append(new)
    return coord_list
