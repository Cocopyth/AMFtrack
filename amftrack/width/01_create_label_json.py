# Creating a json file with the names of the plates
import os
import re
import random
import shutil
import pandas as pd
import json
from amftrack.util.sys import storage_path

folder = os.path.join(storage_path, "width1/groundtruth")
target = os.path.join(storage_path, "measures/data_width.json")

plates = {}

listdir = os.listdir(folder)
for file_name in listdir:
    print(f"Add entry: {file_name}")
    plates[file_name] = []

with open(target, "w") as f:
    json.dump(plates, f)
