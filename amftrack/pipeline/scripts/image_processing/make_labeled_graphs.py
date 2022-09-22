import sys
from amftrack.util.sys import temp_path
import pandas as pd
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
from amftrack.pipeline.functions.image_processing.node_id_2 import create_labeled_graph
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_nx_to_tab,
)
from amftrack.util.sys import get_dirname
import os
import scipy.io as sio

directory = str(sys.argv[1])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])


run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})

unique_ids = list(set(run_info["unique_id"].values))
unique_ids.sort()
select = run_info.loc[run_info["unique_id"] == unique_ids[i]]
exp = Experiment(directory)
exp.load(select, suffix="_width")
create_labeled_graph(exp)
exp.save_graphs(suffix="_labeled")

for i, date in enumerate(exp.dates):
    tab = from_nx_to_tab(exp.nx_graph[i], exp.positions[i])
    directory_name = get_dirname(date, exp.folders)
    path_snap = os.path.join(exp.directory, directory_name)
    path_save = path_snap + "/Analysis/graph_full_labeled.mat"
    sio.savemat(path_save, {name: col.values for name, col in tab.items()})
