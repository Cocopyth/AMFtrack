from path import path_code_dir
import sys  
sys.path.insert(0, path_code_dir)
from sample.util import get_dates_datetime, get_dirname
from sample.pipeline.functions.node_id import (
    second_identification,
)
from sample.pipeline.functions.extract_graph import (
    from_nx_to_tab,
)
import scipy.io as sio
import pickle

plate = int(sys.argv[1])
begin = int(sys.argv[2])
end = int(sys.argv[3])
directory = str(sys.argv[4])

dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
dates_datetime_chosen = dates_datetime[begin : end + 1]
dates = dates_datetime_chosen

nx_graph_pos = []
for date in dates:
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    path_save = path_snap + "/Analysis/nx_graph_pruned_width.p"
    nx_graph_pos.append(pickle.load(open(path_save, "rb")))
nx_graph_pruned = [c[0] for c in nx_graph_pos]
poss_aligned = [c[1] for c in nx_graph_pos]
downstream_graphs = []
downstream_pos = []
begin = len(dates) - 1
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
    date = dates[i]
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    path_save = path_snap + "/Analysis/nx_graph_pruned_labeled.p"
    pos = poss_aligned[i]
    pickle.dump((g, pos), open(path_save, "wb"))

for i, date in enumerate(dates):
    tab = from_nx_to_tab(nx_graph_pruned[i], poss_aligned[i])
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    path_save = path_snap + "/Analysis/graph_full_labeled.mat"
    sio.savemat(path_save, {name: col.values for name, col in tab.items()})
