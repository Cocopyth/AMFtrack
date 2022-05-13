# Get a dictionnary with x, y for each image from loreto's file

import os
import re
import random
import shutil
import pandas as pd
import json
from amftrack.util.sys import storage_path

excel_file_path = os.path.join(storage_path, "width1/220324_Plate907.xls")
raw_data = os.path.join(storage_path, "width1/measures/data_width_907.json")
new_data = os.path.join(storage_path, "width1/measures/data_width_907_2.json")

# Get x and y
df = pd.read_excel(excel_file_path)
df = df[["Name", "x (um)", "y (um)"]]
df.rename(columns={"x (um)": "x", "y (um)": "y"}, inplace=True)

# Get the rest of the data
with open(raw_data) as f:
    d = json.load(f)

# Get the x and y
d_coord_x, d_coord_y = {}, {}
for i in range(len(df)):
    name = df["Name"].iloc[i]
    x, y = int(df["x"].iloc[i]), int(df["y"].iloc[i])
    print(f"File: {str(name)} Coordinates: {x}, {y}")
    d_coord_x[name] = x
    d_coord_y[name] = y

# Add the x and y
for key in d.keys():
    d[key]["x"] = d_coord_x[d[key]["plate"]]
    d[key]["y"] = d_coord_y[d[key]["plate"]]

# Save the new dictionnary
with open(new_data, "w") as f:
    json.dump(d, f)
