"""
Utils to handle files and directories.
"""

import os
import random


def chose_file(directory_path: str) -> str:
    """
    Return the path of an element in the `directory_path`.
    The element is chosen randomly
    """
    listdir = [file_name for file_name in os.listdir(directory_path)]
    if listdir:
        return os.path.join(
            directory_path, listdir[random.randint(0, len(listdir) - 1)]
        )
    else:
        return None


if __name__ == "__main__":
    print(chose_file("."))
