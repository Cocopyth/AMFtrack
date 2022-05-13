# Get a dictionnary with x, y for each image from loreto's file

import os
import re
import random
import shutil
import pandas as pd
import json
import numpy as np
from amftrack.util.sys import storage_path

source = os.path.join(storage_path, "width1/measures/data_width_907_2.json")
destination = os.path.join(storage_path, "width1/measures/data_width_907_3.json")

# Get the json
with open(source) as f:
    d = json.load(f)

for key in d.keys():
    d[key]["value"] = int(np.mean(d[key]["values"]))
    d[key]["variance"] = int(np.var(d[key]["values"]))
    d[key]["std_deviation"] = int(np.std(d[key]["values"]))

# Save the new dictionnary
with open(destination, "w") as f:
    json.dump(d, f)
