import sys
from amftrack.util.sys import temp_path
import pandas as pd
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment
from amftrack.pipeline.functions.image_processing.node_id_2 import (
    create_labeled_graph
)
directory = str(sys.argv[1])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json")

unique_ids = list(set(run_info["unique_id"].values))
unique_ids.sort()
select = run_info.loc[run_info["unique_id"] == unique_ids[i]]
exp = Experiment(directory)
exp.load(select, suffix='_width')
create_labeled_graph(exp)
exp.save_graphs(suffix = '_labeled')