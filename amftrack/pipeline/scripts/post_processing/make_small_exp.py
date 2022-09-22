from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
from amftrack.pipeline.functions.image_processing.extract_width_fun import *
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    save_graphs,
)
from amftrack.util.sys import temp_path
import pickle
import pandas as pd

directory = str(sys.argv[1])
overwrite = eval(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
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
