from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment, Node, Edge
from amftrack.pipeline.functions.transport_processing.high_mag_videos.loading import load_video_dataset
from amftrack.pipeline.functions.transport_processing.high_mag_videos.register_videos import register_dataset, \
    add_attribute, check_hasedges
from amftrack.util.dbx import upload
from amftrack.util.sys import update_plate_info, get_current_folders
from amftrack.pipeline.functions.transport_processing.high_mag_videos.add_BC import get_abcisse, add_betweenness
import pandas as pd
import numpy as np
import os

plate_id = "310_20230830"
indexes = {
"20230901_Plate310" : 20,
"20230902_Plate310" : 33,
"20230903_Plate310" : 42,
"20230904_Plate310" : 52,
"20230905_Plate310" : 64,
"20230906_Plate310" : 73,
}
plate_id = "441_20230807"
indexes = {
"20230810_Plate441" : "20230810_1005_Plate14",
"20230811_Plate441" : "20230811_1605_Plate14",
"20230812_Plate441" : "20230812_1006_Plate14",
"20230813_Plate441" : "20230813_1618_Plate14",
}
videos_folder = "/projects/0/einf914/videos/"

analysis_folder = "/projects/0/einf914/analysis_videos/CocoTransport/"
analysis_folder_root = "/projects/0/einf914/analysis_videos/"


# directory_targ = os.path.join(directory_scratch, "stitch_temp2") + "/"
directory_targ = '/projects/0/einf914/transport/'
update_plate_info(directory_targ, local=True)
all_folders = get_current_folders(directory_targ, local=True)
folders = all_folders.loc[all_folders["unique_id"] == plate_id]
folders = folders.loc[folders["/Analysis/nx_graph_pruned_labeled.p"] == True]
folders = folders.sort_values(by="datetime")
hyphae = pd.read_excel("hyphae.xlsx")
def add_hyphal_attributes(edge_data_csv,edge,mapping,t,exp):
    for index,row in hyphae.iterrows():
        begin,end = row['begin'],row['end']
        fun = lambda edge : get_abcisse(edge,begin,end,t,exp)
        add_attribute(edge_data_csv, edge, fun, f"abcisse_{begin}_{end}", mapping)


dropbox_address = "/DATA/CocoTransport/"
upl_targ = dropbox_address
exp = Experiment(directory_targ)
selection = folders[folders['folder'].isin(indexes.values())]
exp.load(selection, suffix="_labeled")
for t in range(exp.ts):
    exp.load_tile_information(t)
    add_betweenness(exp,t)

for t, plate_id_video in enumerate(list(indexes.keys())):
    data_obj = load_video_dataset(plate_id_video, videos_folder, analysis_folder, analysis_folder_root)

    # break
    data_obj.video_objs = sorted(data_obj.video_objs, key=lambda video: video.dataset['video_int'])
    for vid_obj in data_obj.video_objs:
        drop_targ = os.path.relpath(f"/{vid_obj.dataset['tot_path_drop']}", upl_targ)
        db_address = f"{upl_targ}KymoSpeeDExtract/{drop_targ}"
        if check_hasedges(vid_obj) and vid_obj.dataset['magnification'] != 4:
            edge_data_csv = pd.read_csv(vid_obj.edge_adr)
            mapping = {}
            for edge in vid_obj.edge_objs:
                if "network_begin" in edge.mean_data.keys():
                    edge_begin = edge.mean_data["network_begin"]
                    edge_end = edge.mean_data["network_end"]  # break
                    network_edge = Edge(Node(edge_begin, exp), Node(edge_end, exp), exp)
                    mapping[edge.edge_name] = network_edge
            for edge in vid_obj.edge_objs:
                if "network_begin" in edge.mean_data.keys() and len(mapping[edge.edge_name].ts()) > 0:
                    add_attribute(edge_data_csv, edge, lambda edge: edge.width(t), "width_automate", mapping)
                    # add_attribute(edge_data_csv, edge, lambda edge: edge.betweeness(t), "betweenness_automate", mapping)
                    # add_attribute(edge_data_csv, edge, lambda edge: get_derivative(edge,t,fun), "betweenness_derivative", mapping)
                    add_hyphal_attributes(edge_data_csv, edge, mapping, t, exp)
            edge_data_csv.to_csv(vid_obj.edge_adr, index=False)
            drop_targ = os.path.relpath(f"/{vid_obj.dataset['tot_path_drop']}", upl_targ)
            db_address = f"{upl_targ}KymoSpeeDExtract/{drop_targ}"
            source = vid_obj.edge_adr
            target = db_address + "/edges_data.csv"
            # print(source,target)
            upload(source, target)
            # break
    # break
