from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
import os
from amftrack.pipeline.functions.image_processing.hyphae_id_surf import (
    get_mother,
    save_hyphaes,
    width_based_cleaning,
    resolve_anastomosis_crossing_by_root,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    save_graphs,
)

import pandas as pd
from amftrack.pipeline.paths.directory import directory_scratch
import json
from time import time_ns
from amftrack.util.dbx import temp_path

directory = str(sys.argv[1])
limit = int(sys.argv[2])
version = str(sys.argv[3])
labeled = eval(sys.argv[4])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json")

plates = list(set(run_info["Plate"].values))
plates.sort()
plate = plates[i]
print(plate)
select_folders = run_info.loc[run_info["Plate"] == plate]

corrupted_rotation = select_folders.loc[
    ((select_folders["/Analysis/transform.mat"] == False))
    & (select_folders["/Analysis/transform_corrupt.mat"])
]["folder"]

folder_list = list(select_folders["folder"])
folder_list.sort()
indexes = [folder_list.index(corrupt_folder) for corrupt_folder in corrupted_rotation]
indexes = [index for index in indexes if index < limit]
indexes.sort()
indexes += [limit]
start = 0
for index in indexes:
    stop = index
    select_folder_names = folder_list[start:stop]
    plate = int(folder_list[0].split("_")[-1][5:])
    # confusion between plate number and position in Prince
    exp = Experiment(plate, directory)
    select_folders = run_info.loc[run_info["folder"].isin(select_folder_names)]
    exp.load(select_folders, labeled)
    exp.dates.sort()
    # when no width is included
    # width_based_cleaning(exp)
    if labeled:
        resolve_anastomosis_crossing_by_root(exp)
    # get_mother(exp.hyphaes)
    # solve_two_ends = resolve_ambiguity_two_ends(exp_clean.hyphaes)
    # solved = solve_degree4(exp_clean)
    # clean_obvious_fake_tips(exp_clean)
    dates = exp.dates
    op_id = time_ns()
    exp.dates.sort()
    save_graphs(exp)
    exp.nx_graph = None
    dirName = f"{directory}Analysis_{op_id}_{start}_{stop}_Version{version}"
    try:
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")
    # hyphs, gr_inf = save_hyphaes(
    #     exp, f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/"
    # )
    # exp.save(f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/")
    exp.save_location = dirName
    exp.pickle_save(f"{dirName}/")
    with open(f"{dirName}/folder_info.json", "w") as jsonf:
        json.dump(folder_list[start:stop], jsonf, indent=4)
    start = stop
