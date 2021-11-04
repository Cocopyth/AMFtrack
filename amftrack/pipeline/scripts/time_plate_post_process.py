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
import os
import json
from datetime import datetime

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
path_analysis_info = row['path_analysis_info']

if not os.path.isfile(f'{directory}{path_analysis_info}') or overwrite:
    plate_level_data =  {}
    path_exp = f'{directory}{row["path_exp"]}'
    exp = pickle.load(open(path_exp, "rb"))
    folder = row['folder_analysis']
    path = f'{directory}{row["folder_analysis"]}'
    exp.center = np.load(f'{path}/center.npy')
    exp.orthog = np.load(f'{path}/orthog.npy')
    exp.reach_out = np.load(f'{path}/reach_out.npy')
    for t in range(exp.ts):
        data_t = {}
        date = exp.dates[t]
        date_str = datetime.strftime(date, "%d.%m.%Y, %H:%M:")
        for index,f in enumerate(list_f):
            column,result = f(exp,t,list_args[index])
            data_t[column] = result
        data_t['date'] = date_str
        data_t['Plate'] = row["Plate"]
        data_t['path_exp'] = row["path_exp"]
        data_t['path_analysis_info'] = row["path_analysis_info"]
        data_t['path_dynamic_infos'] = f'{folder}/dynamic_infos/dyn_inf_{date_str}.json'
        plate_level_data[t] = data_t
    with open(f'{directory}{path_analysis_info}', 'w') as jsonf:
        json.dump(plate_level_data, jsonf,  indent=4)