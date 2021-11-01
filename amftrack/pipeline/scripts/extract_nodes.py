from path import path_code_dir
import sys  
sys.path.insert(0, path_code_dir)
from amftrack.util import get_dates_datetime, get_dirname
from amftrack.pipeline.functions.node_id import (
    second_identification,
)
from amftrack.pipeline.functions.extract_graph import (
    from_nx_to_tab,
)
import scipy.io as sio
import pickle
import pandas as pd
from amftrack.pipeline.paths.directory import directory_scratch

directory = str(sys.argv[1])
limit = int(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f'{directory_scratch}temp/{op_id}.json')

plates = list(set(run_info['Plate'].values))
plates.sort()
print(plates[i])
select_folders = run_info.loc[run_info['Plate'] == plates[i]]
folder_list = list(select_folders['folder'])
folder_list.sort()
nx_graph_pos = []
for i,directory_name in enumerate(folder_list[:limit]):
    path_snap = directory + directory_name
    path_save = path_snap + "/Analysis/nx_graph_pruned_width.p"
    nx_graph_pos.append(pickle.load(open(path_save, "rb")))
nx_graph_pruned = [c[0] for c in nx_graph_pos]
poss_aligned = [c[1] for c in nx_graph_pos]
downstream_graphs = []
downstream_pos = []
begin = len(folder_list[:limit])-1
downstream_graphs = [nx_graph_pruned[begin]]
downstream_poss = [poss_aligned[begin]]
for i in range(begin - 1, -1, -1):
    print("i=", i)
    new_graphs, new_poss = second_identification(
        nx_graph_pruned[i],
        downstream_graphs[0],
        poss_aligned[i],
        downstream_poss[0],
        50,
        downstream_graphs[1:],
        downstream_poss[1:],
        tolerance=30,
    )
    downstream_graphs = new_graphs
    downstream_poss = new_poss

nx_graph_pruned = downstream_graphs
poss_aligned = downstream_poss
for i, g in enumerate(nx_graph_pruned):
    directory_name = folder_list[i]
    path_snap = directory + directory_name
    path_save = path_snap + "/Analysis/nx_graph_pruned_labeled.p"
    pos = poss_aligned[i]
    pickle.dump((g, pos), open(path_save, "wb"))

for i,directory_name in enumerate(folder_list[:limit]):
    tab = from_nx_to_tab(nx_graph_pruned[i], poss_aligned[i])
    path_snap = directory + directory_name
    path_save = path_snap + "/Analysis/graph_full_labeled.mat"
    sio.savemat(path_save, {name: col.values for name, col in tab.items()})
