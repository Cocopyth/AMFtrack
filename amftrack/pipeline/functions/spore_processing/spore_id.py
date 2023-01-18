import numpy as np
from amftrack.pipeline.functions.image_processing.node_id import recursive_mapping
import scipy.io as sio
import os
import pandas as pd


def find_corresp(spore_position_t, spore_position_tp1, thresh=30):
    corresp = {sporet: [] for sporet in spore_position_t.keys()}
    for sporet in spore_position_t.keys():
        for sporetp1 in spore_position_tp1.keys():
            if (
                np.linalg.norm(spore_position_t[sporet] - spore_position_tp1[sporetp1])
                <= thresh
            ):
                corresp[sporet].append(sporetp1)
    final_corresp = {}
    for sporet in spore_position_t.keys():
        if len(corresp[sporet]) > 0:
            mini = np.inf
            for sporetp1 in corresp[sporet]:
                dist = np.linalg.norm(
                    spore_position_t[sporet] - spore_position_tp1[sporetp1]
                )
                if dist < mini:
                    spore_select = sporetp1
                    mini = dist
            final_corresp[spore_select] = sporet
    return final_corresp


def create_labeled_spores(corresp_list, spore_positions_dic, spore_radius_dic, folders):
    final_positions = {}
    final_radiuses = {}
    for t in range(len(corresp_list) + 1):

        def mapping(spore, t=t):
            return recursive_mapping(corresp_list, t, spore)

        final_positions[t] = {
            mapping(spore): spore_positions_dic[t][spore]
            for spore in spore_positions_dic[t].keys()
        }
        final_radiuses[t] = {
            mapping(spore): spore_radius_dic[t][spore]
            for spore in spore_radius_dic[t].keys()
        }

    spore_data_total = pd.DataFrame()
    for t in range(len(corresp_list) + 1):
        spore_dataf = pd.concat(
            (
                pd.DataFrame(final_positions[t], index=["x", "y"]).transpose(),
                pd.DataFrame(final_radiuses[t], index=["radius"]).transpose(),
            ),
            axis=1,
        )
        spore_dataf["t"] = t
        spore_dataf["folder"] = folders[t]["folder"]
        spore_dataf["datetime"] = folders[t]["datetime"]

        spore_data_total = pd.concat((spore_data_total, spore_dataf))
    return spore_data_total


def make_spore_data(exp):
    corresp_list = []
    spore_positions_dic = {}
    spore_radiuses_dic = {}
    folders = {}
    for t in range(exp.ts - 1):
        exp.load_tile_information(t)
        select = exp.folders
        table = select.sort_values(by="datetime", ascending=True)
        path_spore_data = os.path.join(exp.directory, table["folder"].iloc[t])
        spore_data = sio.loadmat(
            os.path.join(path_spore_data, "Analysis", "spores.mat")
        )["spores"]
        positionst = {
            i: exp.timestep_to_general(np.flip(spore_data[i, :2]), t)
            for i in range(len(spore_data))
        }
        spore_positions_dic[t] = positionst
        radiusest = {i: spore_data[i, 2] for i in range(len(spore_data))}
        spore_radiuses_dic[t] = radiusest

        tp1 = t + 1

        exp.load_tile_information(tp1)
        select = exp.folders
        table = select.sort_values(by="datetime", ascending=True)
        path_spore_data = os.path.join(exp.directory, table["folder"].iloc[tp1])
        spore_data = sio.loadmat(
            os.path.join(path_spore_data, "Analysis", "spores.mat")
        )["spores"]
        positionstp1 = {
            i: exp.timestep_to_general(np.flip(spore_data[i, :2]), tp1)
            for i in range(len(spore_data))
        }
        corresp_list.append(find_corresp(positionst, positionstp1))
        folders[t] = {
            "folder": table["folder"].iloc[t],
            "datetime": table["datetime"].iloc[t],
        }

    t = exp.ts - 1
    exp.load_tile_information(t)
    select = exp.folders
    table = select.sort_values(by="datetime", ascending=True)
    path_spore_data = os.path.join(exp.directory, table["folder"].iloc[t])
    spore_data = sio.loadmat(os.path.join(path_spore_data, "Analysis", "spores.mat"))[
        "spores"
    ]
    positionst = {
        i: exp.timestep_to_general(np.flip(spore_data[i, :2]), t)
        for i in range(len(spore_data))
    }
    spore_positions_dic[t] = positionst
    radiusest = {i: spore_data[i, 2] for i in range(len(spore_data))}
    spore_radiuses_dic[t] = radiusest
    folders[t] = {
        "folder": table["folder"].iloc[t],
        "datetime": table["datetime"].iloc[t],
    }
    spore_data_total = create_labeled_spores(
        corresp_list, spore_positions_dic, spore_radiuses_dic, folders
    )
    return spore_data_total
