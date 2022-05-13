import os
import re
import random
import shutil
from amftrack.util.sys import storage_path
from amftrack.util.file import chose_file

source_folder = "/mnt/sun/shimizu-tmp/TEMP/Transport"
target_folder = os.path.join(storage_path, "width2/labels")

model = re.compile(r"\d\d\d_\d\d\d")  # d is for decimal


def is_valid_file(name):
    match = model.search(name)
    if match:
        return True
    else:
        return False


a = input("Are you sure? (y-n)")
if a == "y":
    listdir = os.listdir(source_folder)
    for file_name in listdir:
        if is_valid_file(file_name):
            print(f"Valid file: {file_name}")
            sub_dir = os.path.join(source_folder, file_name, "Img")
            chosen_image_path = chose_file(sub_dir)
            chosen_image_name = os.path.basename(chosen_image_path)
            shutil.copy(chosen_image_path, target_folder)
            os.rename(
                os.path.join(target_folder, chosen_image_name),
                os.path.join(target_folder, file_name),
            )
