from path import path_code_dir
import sys  
sys.path.insert(0, path_code_dir)
from amftrack.pipeline.functions.image_processing.extract_width_fun import *
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment, save_graphs, load_graphs
from amftrack.util import get_dates_datetime, get_dirname
import pickle
import networkx as nx
import pandas as pd
from amftrack.pipeline.paths.directory import directory_scratch
from path import path_code_dir
import os
import json
from datetime import datetime
from amftrack.util import *

directory = str(sys.argv[1])
overwrite =  eval(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
run_info = pd.read_json(f'{directory_scratch}temp/{op_id}.json')
list_f,list_args = pickle.load(open(f'{directory_scratch}temp/{op_id}.pick', "rb"))
folder_list = list(run_info['folder_analysis'])
directory_name = folder_list[i]
select = run_info.loc[run_info['folder_analysis'] == directory_name]
row = [row for index, row in select.iterrows()][0]
path_exp = f'{directory}{row["path_exp"]}'
exp = pickle.load(open(path_exp, "rb"))
load_graphs(exp)

folder = row['folder_analysis']
path = f'{directory}{row["folder_analysis"]}'
path_global_plate_info = row["path_global_plate_info"]
exp.center = np.load(f'{path}/center.npy')
exp.orthog = np.load(f'{path}/orthog.npy')
exp.reach_out = np.load(f'{path}/reach_out.npy')
if not os.path.isfile(f'{directory}{path_global_plate_info}') or overwrite:
    global_plate_info = {}
else:
    global_plate_info = json.load(open(f'{directory}{path_global_plate_info}', 'r'))
global_plate_info['Plate'] = row['Plate']
global_plate_info['version'] = row['version']
global_plate_info['PrincePos'] = row['PrincePos']
global_plate_info['root'] = row['root']
global_plate_info['strain'] = row['strain']
global_plate_info['medium'] = row['medium']
global_plate_info['split'] = row['split']
global_plate_info['Temp'] = row['Temp']
global_plate_info['CrossDate'] = row['CrossDate']
global_plate_info['Pbait'] = row['Pbait']
global_plate_info['date_begin'] = row['date_begin']
global_plate_info['date_end'] = row['date_end']
global_plate_info['number_timepoints'] = row['number_timepoints']
global_plate_info['path_exp'] = row['path_exp']

for index,f in enumerate(list_f):
    column,result = f(exp,list_args[index])
    global_plate_info[column] = result
    
with open(f'{directory}{path_global_plate_info}', 'w') as jsonf:
    json.dump(global_plate_info, jsonf,  indent=4)