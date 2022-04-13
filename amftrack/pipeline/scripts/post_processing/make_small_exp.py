from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
from amftrack.pipeline.functions.image_processing.extract_width_fun import *
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    save_graphs,
    load_graphs,
)
from amftrack.util.sys import get_dates_datetime, get_dirname, temp_path
import pickle
import networkx as nx
import pandas as pd
from amftrack.pipeline.paths.directory import directory_scratch
from path import path_code_dir
import os
import json
from datetime import datetime
from pymatreader import read_mat
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
from amftrack.plotutil import plot_t_tp1
from amftrack.notebooks.analysis.util import directory_scratch
import imageio

directory = str(sys.argv[1])
overwrite = eval(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
run_info = pd.read_json(f"{temp_path}/{op_id}.json")
list_f, list_args = pickle.load(open(f"{temp_path}/{op_id}.pick", "rb"))
folder_list = list(run_info["folder_analysis"])
directory_name = folder_list[i]
select = run_info.loc[run_info["folder_analysis"] == directory_name]
row = [row for index, row in select.iterrows()][0]
plate_num = row["Plate"]
path_exp = f'{directory}{row["path_exp"]}'
exp = pickle.load(open(path_exp, "rb"))
exp.dates.sort()
save_graphs(exp)
exp.nx_graph = None
dirName = exp.save_location
exp.pickle_save(f"{dirName}/")
