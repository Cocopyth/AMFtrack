from path import path_code_dir
import sys  
sys.path.insert(0, path_code_dir)
from amftrack.util import get_dirname
import pandas as pd
import ast
from scipy import sparse
from datetime import datetime
import cv2
import imageio
import numpy as np
import os
from time import time
from amftrack.util import get_dates_datetime, get_dirname
from amftrack.pipeline.paths.directory import directory_scratch
from subprocess import call
from amftrack.transfer.functions.transfer import upload, zip_file
import json

directory = str(sys.argv[1])
dir_drop = str(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
print(f'{os.getenv("TEMP")}{op_id}.json')
run_info = pd.read_json(f'{os.getenv("TEMP")}{op_id}.json')

folder_list = list(run_info['folder'])
folder_list.sort()
directory_name = folder_list[i]
run_info['unique_id'] = run_info['Plate'].astype(str) + "_" + run_info['CrossDate'].astype(str)
line = run_info.loc[run_info['folder']==directory_name]
id_unique = line['unique_id'].iloc[0]

path_snap=directory+directory_name
API = str(np.load('/home/cbisot/pycode/API_drop.npy'))

path_zip =f'{os.getenv("TEMP")}{directory_name}.zip'
path_info =f'{os.getenv("TEMP")}{directory_name}_info.json'
line.to_json(path_info)
zip_file(path_snap,path_zip)
upload(API,path_zip,f'/{dir_drop}/{id_unique}/{directory_name}.zip',chunk_size=256 * 1024 * 1024)
upload(API,path_info,f'/{dir_drop}/{id_unique}/{directory_name}_info.json',chunk_size=256 * 1024 * 1024)
os.remove(path_info) 
os.remove(path_zip) 