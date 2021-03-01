from extract_width_fun import *
from util import get_path, get_dates_datetime, get_dirname
import pickle
import networkx as nx
import sys

plate = int(sys.argv[1])
directory = str(sys.argv[2])
i = int(sys.argv[-1])
dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
date_datetime = dates_datetime[i]
date = date_datetime
directory_name = get_dirname(date, plate)
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
edge_test = get_width_info(exp,0)
nx.set_edge_attributes(G, edge_test, 'width')
date = exp.dates[0]
directory_name = get_dirname(date, exp.plate)
path_snap = exp.directory + directory_name
print(f'saving {path_snap}')
pickle.dump((G,pos), open(f'{path_snap}/Analysis/nx_graph_pruned_width.p', "wb"))