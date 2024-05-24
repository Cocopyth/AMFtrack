
from amftrack.util.sys import (

    update_plate_info,

    get_current_folders,
)

from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)

from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Node,
)
from amftrack.pipeline.functions.post_processing.extract_study_zone import (
    load_study_zone,
)
from amftrack.pipeline.functions.post_processing.exp_plot import *
import pickle
import scipy.io as sio
import networkx as nx
import numpy as np
from sthype.graph_functions import spatial_temporal_graph_from_spatial_graphs
from sthype import SpatialGraph
import gc
import os
import pickle
from tqdm import tqdm

plates = [
    # "441_20230807",
    "449_20230807",
    "310_20230830"
]
directory_targ = "/projects/0/einf914/transport/"
update_plate_info(directory_targ, local=True)
all_folders = get_current_folders(directory_targ, local=True)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
folders = folders.loc[folders["/Analysis/nx_graph_pruned_width.p"] == True]



path_root = f"/projects/0/einf914/graph_stacks"
maxes = {
    "441_20230807" : 55,
    "449_20230807" : 80,
    "310_20230830" : 65
}
# for plate_id in plates:
#     # update_plate_info(directory_targ, local=True,strong_constraint=False)
#     # all_folders = get_current_folders(directory_targ, local=True)
#     folders = all_folders.loc[all_folders["unique_id"] == plate_id]
#     folders = folders.loc[folders["/Analysis/nx_graph_pruned_width.p"] == True]
#     folders = folders.sort_values(by="folder")
#
#     folders = folders.sort_values(by="datetime")
#
#     # i = indexes[plate_id_video]
#     # i = np.where(folders['folder'] == indexes[plate_id_video])[0][0]
#     # selection = folders[folders['folder'].isin(indexes.values())]
#     for i in range(1, maxes[plate_id]+1):
#         exp = Experiment(directory_targ)
#
#         selection = folders.iloc[i: i + 1]
#         folder_list = list(selection["folder"])
#         directory_name = folder_list[0]
#         path_snap = directory_targ + directory_name
#         transform = sio.loadmat(path_snap + "/Analysis/transform_new.mat")
#         exp.load(selection, suffix="_realigned")
#         for t in range(exp.ts):
#             exp.load_tile_information(t)
#             exp.save_location = ""
#
#             load_study_zone(exp)
#         os.makedirs(os.path.join(path_root, plate_id, ), exist_ok=True)
#         graph = exp.nx_graph[0]
#         graph_to_save = graph.copy()
#         for u, v, data in graph_to_save.edges(data=True):
#             data["pixels"] = data.pop("pixel_list")
#         node_not_in_ROI = []
#         for node in graph_to_save:
#             if not is_in_ROI_node(Node(node, exp), 0):
#                 node_not_in_ROI.append(node)
#         graph_to_save.remove_nodes_from(node_not_in_ROI)
#         nx.set_node_attributes(graph_to_save, exp.positions[0], 'position')
#         path_tot = os.path.join(path_root, plate_id, f"graph{i:03d}.pickle")
#         pickle.dump((graph_to_save, transform, selection.iloc[0]), open(path_tot, 'wb'))
#         # break

for plate_id in plates:
    saved_infos = []

    str_directory = os.path.join(path_root,plate_id,)
    directory = os.fsencode(str_directory)
    files = os.listdir(directory)
    files.sort()
    for file in tqdm(files[:maxes[plate_id]]):
        filename = os.fsdecode(file)
        file_path = os.path.join(str_directory, filename)
        if filename.endswith('pickle'):
            saved_infos.append(pickle.load(open(file_path, 'rb')))

    graphs = [graph for graph,_,_ in saved_infos]
    folder_infos = [folder_info for _,_, folder_info in saved_infos]
    transforms = [(transform['R'],transform['t']) for _, transform,_ in saved_infos]
    spatial_graphs = [SpatialGraph(graph) for graph in graphs]
    print('cleaned garbage')
    spatial_temporal_graph = spatial_temporal_graph_from_spatial_graphs(spatial_graphs,
                                                                        np.arange(len(spatial_graphs)),
                                                                        verbose=1,
                                                                        threshold = 30,
                                                                        segments_length= 30,
    )
    for u, v in spatial_temporal_graph.edges():
        if "pixels" in spatial_temporal_graph[u][v]["initial_edge_attributes"].keys():
            del spatial_temporal_graph[u][v]["initial_edge_attributes"]["pixels"]
        for i in range(len(transforms)):
            if "pixels" in spatial_temporal_graph[u][v][str(i)].keys():
                del spatial_temporal_graph[u][v][str(i)]["pixels"]
    gc.collect()
    print("finished making spatial_temporal_graph")
    spatial_temporal_graph.transforms = transforms
    spatial_temporal_graph.folder_infos = folder_infos
    path_tot = os.path.join(path_root,f"graph{plate_id}.pickle")
    pickle.dump(spatial_temporal_graph, open(path_tot, 'wb'))
    target = f"/DATA/CocoTransport/graphs/graph{plate_id}.pickle"
    source = path_tot
    upload(source, target)
    # del graphs
    # del saved_infos
    # gc.collect()