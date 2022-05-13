# Fetch the labels given with label me to construct a new dictionnary with all the labels together

import os
import random
import json
from amftrack.util.sys import storage_path

source = os.path.join(storage_path, "width1/labels")
destination = os.path.join(storage_path, "width1/labels/labels.json")


def is_valid(name):
    return ".json" in name


d = {}
for file in os.listdir(source):
    if is_valid(file):
        new_data = {}
        path = os.path.join(source, file)
        with open(path) as f:
            json_from_file = json.load(f)
        for shape in json_from_file["shapes"]:
            if shape["label"] == "width":
                new_data["width"] = shape["points"]
            if shape["label"] == "axis":
                new_data["axis"] = shape["points"]
        # name = os.path.splitext(file)[0] + ".tiff"
        name = json_from_file["imagePath"]
        d[name] = new_data

with open(destination, "w") as f:
    json.dump(d, f)
