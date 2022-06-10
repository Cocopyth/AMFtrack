# Define general parameters here

from dataclasses import dataclass


DIM_X = 3000
DIM_Y = 4096
CAMERA_RES = 3.45  # in micrometers

# ML params: TODO (FK): move elsewhere
@dataclass
class WidthParam:
    SLICE_LENGTH = 120  # TODO(FK): increase it for make_dataset
    WIDTH_MODEL = "Unknown"
