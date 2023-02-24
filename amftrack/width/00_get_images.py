# Get images from Loreto transport data. Only select some images from each flow_processing

import os
import re
import random
import shutil

source_folder = "/mnt/sun/shimizu-tmp/TEMP/Transport"
target_folder = "/data/felix/width1"

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
            sub_dir = source_folder + f"/{file_name}" + "/Img"
            # Create directory
            target_sub_folder = os.path.join(target_folder, file_name)
            os.mkdir(target_sub_folder)
            os.chdir(sub_dir)
            for i in range(3):
                image = random.choice(list(os.listdir(sub_dir)))
                path = os.path.join(sub_dir, image)
                print(f"Download image: {image}")
                shutil.copy(image, target_sub_folder)
