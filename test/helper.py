import os
from amftrack.util.sys import data_path


def has_test_plate():
    if len(os.listdir(os.path.join(data_path, "test"))) > 0:
        return True
    else:
        return False
