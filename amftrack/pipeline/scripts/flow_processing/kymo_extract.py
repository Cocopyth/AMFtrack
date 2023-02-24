import sys

from amftrack.pipeline.functions.image_processing.extract_width_fun import (
    get_width_info,
    get_width_info_new,
)


from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
from amftrack.util.sys import temp_path
import pickle
import networkx as nx
import pandas as pd
from time import time_ns

directory = str(sys.argv[1])
name_job = str(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
print(run_info)