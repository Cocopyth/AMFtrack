from path import path_code_dir
import sys  
sys.path.insert(0, path_code_dir)
from amftrack.pipeline.functions.image_processing.extract_width_fun import *
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment
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
print(run_info.columns)

folder_list = list(run_info['t'])
t = folder_list[i]
select = run_info.loc[run_info['t'] == t]
row = [row for index, row in select.iterrows()][0]
path_dynamic_info = row['path_dynamic_infos']
path_exp = f'{directory}{row["path_exp"]}'
exp = pickle.load(open(path_exp, "rb"))
path_analysis_info = row['path_analysis_info']
whole_plate_info = pd.read_json(f'{directory}{path_analysis_info}',
convert_dates=True).transpose()
whole_plate_info.index.name = 't'
whole_plate_info.reset_index(inplace=True)    
if not os.path.isfile(f'{directory}{path_dynamic_info}') or overwrite:
        dynamic_data = {}
        t = row['t']
        tp1 = t+1
        if tp1<exp.ts:
            for hypha in exp.hyphaes:
                if t in hypha.ts and tp1 in hypha.ts:
                    data_hypha = {}
                    data_hypha['end'] = hypha.end.label
                    for index,f in enumerate(list_f):
                        column,result = f(hypha,t,tp1,list_args[index])
                        data_hypha[column] = result
                    dynamic_data[hypha.end.label] = data_hypha
path_folder = '/'.join(path_analysis_info.split('/')[:-1])
print(path_folder)
path = f'{directory}{path_folder}/dynamic_infos'
print(path)
try:
    os.mkdir(path)
except OSError as error:
    print(error)  
with open(f'{directory}{path_dynamic_info}', 'w') as jsonf:
    json.dump(dynamic_data, jsonf,  indent=4)