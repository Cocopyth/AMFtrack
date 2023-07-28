import pandas as pd
import numpy as np
import os

colors = {
    "'A5''001P100N'": "orange",
    "'A5''001P100N100C'": "orange",
    "'A5''001P100N200C'": "red",
    "'A5''001P100N200C-T'": "red",
    "'A5sp3''001P100N100C'": "orange",
    "'C2''001P100N100C'": "blue",
    "'C2''001P100N200C'": "purple",
    "'Agg''001P100N100C'": "black",
    "'Agg''001P100N200C'": "black",
}
orders = {
    "'A5''001P100N'": 1,
    "'A5''001P100N100C'": 1,
    "'A5''001P100N200C'": 2,
    "'A5''001P100N200C-T'": 2,
    "'A5sp3''001P100N100C'": 2,
    "'C2''001P100N100C'": 3,
    "'C2''001P100N200C'": 4,
    "'Agg''001P100N100C'": 5,
    "'Agg''001P100N200C'": 5,
}


def get_treatment(plate, folders):
    strain = folders.loc[folders["Plate"] == plate].iloc[0]["strain"]
    medium = folders.loc[folders["Plate"] == plate].iloc[0]["medium"]
    treatment = strain + medium
    return treatment


def get_color(plate, folders):
    treatment = get_treatment(plate, folders)
    return colors[treatment]


def get_order(plate, folders):
    treatment = get_treatment(plate, folders)
    return orders[treatment]


def select_movement(plate_id, time_hypha_info, min_num_occ=1):
    time_hypha_plate = time_hypha_info.loc[time_hypha_info["unique_id"] == plate_id]
    select = time_hypha_plate
    max_speeds = select.groupby("end").max()["speed"]
    correct_tracks = max_speeds.loc[max_speeds <= 450]
    select = select.loc[select["end"].isin(correct_tracks.index)]
    select = select.loc[select["distance_final_pos"] >= 400]
    select = select.loc[select["speed"].between(80, 400)]
    select = select.loc[select["in_ROI"] == "True"]
    num_occ = select.groupby("end").count()["speed"]
    correct_tracks = num_occ.loc[num_occ >= min_num_occ]
    select = select.loc[select["end"].isin(correct_tracks.index)]
    select_movements = select
    return select_movements


def get_average_time_data(plate_id, time_hypha_info, min_num_occ=1):
    select_movements = select_movement(plate_id, time_hypha_info, min_num_occ)
    group = select_movements.groupby(["time_since_begin_h"])["speed"]
    data = group.median()
    dy = group.std() / np.sqrt(group.count())

    return (data, dy, select_movements)


def get_hyphae_hull(plate_id, analysis_folders):
    selection = analysis_folders.loc[analysis_folders["unique_id"] == plate_id]
    analysis_dirs = selection["total_path"]
    folders = pd.DataFrame()
    for analysis_dir in analysis_dirs:
        hyphae_hull = {}
        path_time_hypha = os.path.join(analysis_dir, "time_hull_info")
        if os.path.exists(path_time_hypha):
            json_paths = os.listdir(path_time_hypha)
            for path in json_paths:
                index = int(path.split("_")[-1].split(".")[0])
                hyphae = np.load(os.path.join(path_time_hypha, path))
                hyphae_hull[index] = hyphae

    return hyphae_hull
