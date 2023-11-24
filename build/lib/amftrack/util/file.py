"""
Utils to handle files and directories.
"""

import random
import os


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


def hash_file(filename):
    """ "This function returns the SHA-1 hash
    of the file passed into it"""

    # make a hash object
    h = hashlib.sha256()

    # open file for reading in binary mode
    with open(filename, "rb") as file:

        # loop till the end of the file
        chunk = 0
        while chunk != b"":
            # read only 1024 bytes at a time
            chunk = file.read(4 * 1024 * 1024)
            h.update(chunk)

    # return the hex representation of digest
    return h.hexdigest()


if __name__ == "__main__":
    print(chose_file("."))
