from path import path_code_dir
import sys  
sys.path.insert(0, path_code_dir)
from amftrack.util import get_dates_datetime, get_dirname
import pandas as pd
import ast
import os
from subprocess import call


i = int(sys.argv[-1])
plate = int(sys.argv[1])

from amftrack.pipeline..paths.directory import directory_scratch
listdir=os.listdir(directory_scratch)
list_dir_interest=[name for name in listdir if name.split('_')[-1]==f'Plate{0 if plate<10 else ""}{plate}']
dates_datetime = get_dates_datetime(directory_scratch, plate)
dates_datetime.sort()
dates_datetime_chosen=dates_datetime
dates = dates_datetime_chosen
date = dates[i]
directory_name = get_dirname(date, plate)
path_snap= directory_scratch + directory_name
path_tile=path_snap+'/Img/TileConfiguration.txt.registered'
try:
    tileconfig = pd.read_table(path_tile,sep=';',skiprows=4,header=None,converters={2 : ast.literal_eval},skipinitialspace=True)
except:
    print('error_name')
    path_tile=path_snap+'/Img/TileConfiguration.registered.txt'
    tileconfig = pd.read_table(path_tile,sep=';',skiprows=4,header=None,converters={2 : ast.literal_eval},skipinitialspace=True)
print(f'cleaning {i} {path_snap}')
for name in tileconfig[0]:
    imname = '/Img/'+name.split('/')[-1]
    call(f'rm {directory_scratch + directory_name + imname}', shell=True)
