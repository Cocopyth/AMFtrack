import networkx as nx
import numpy as np
from sthype import SpatialGraph, HyperGraph
from sthype.hypergraph.hypergraph_from_spatial_graphs import spatial_temporal_graph_from_spatial_graphs
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
plates = [
    "441_20230807", "449_20230807", "310_20230830"
]
path_root = f"/scratch-shared/amftrack/graph_stacks"
for plate_id in plates:
    saved_infos = []

    str_directory = os.path.join(path_root,plate_id,)
    directory = os.fsencode(str_directory)
    files = os.listdir(directory)
    files.sort()
    for file in tqdm(files):
        filename = os.fsdecode(file)
        file_path = os.path.join(str_directory, filename)
        if filename.endswith('pickle'):
            saved_infos.append(pickle.load(open(file_path, 'rb')))

    graphs = [graph for graph,_,_ in saved_infos]
    folder_infos = [folder_info for _,_, folder_info in saved_infos]
    transforms = [(transform['R'],transform['t']) for _, transform,_ in saved_infos]
    spatial_graphs = [SpatialGraph(graph) for graph in graphs]
    spatial_temporal_graph = spatial_temporal_graph_from_spatial_graphs(spatial_graphs, np.arange(len(spatial_graphs)), verbose=1)
    spatial_temporal_graph.transforms = transforms
    spatial_temporal_graph.folder_infos = folder_infos
    path_root = f"/scratch-shared/amftrack/graph_stacks"
    path_tot = os.path.join(path_root,f"graph{plate_id}.pickle")

    pickle.dump(spatial_temporal_graph, open(path_tot, 'wb'))
