from path import path_code_dir
import sys  
sys.path.insert(0, path_code_dir)
from amftrack.pipeline.functions.extract_width_fun import *
from amftrack.pipeline.functions.experiment_class_surf import Experiment
from amftrack.util import get_dates_datetime, get_dirname
import pickle
import networkx as nx
import pandas as pd
from amftrack.pipeline.paths.directory import directory_scratch
from path import path_code_dir

directory = str(sys.argv[1])
skip = str(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
run_info = pd.read_json(f'{directory_scratch}temp/{op_id}.json')
folder_list = list(run_info['folder'])
folder_list.sort()
print(folder_list)
directory_name = folder_list[i]
path_snap = directory + directory_name
begin = i
end = i
dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
dates_datetime_chosen = dates_datetime[begin : end + 1]
dates = dates_datetime_chosen
exp = Experiment(plate, directory)
exp.load(dates,False)
G,pos = exp.nx_graph[0],exp.positions[0]
edge_test = get_width_info(exp,0,skip=skip)
nx.set_edge_attributes(G, edge_test, 'width')
date = exp.dates[0]
directory_name = get_dirname(date, exp.plate)
path_snap = directory + directory_name
print(f'saving {path_snap}')
pickle.dump((G,pos), open(f'{path_snap}/Analysis/nx_graph_pruned_width.p', "wb"))