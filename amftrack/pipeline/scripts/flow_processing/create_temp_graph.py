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

plates = ["792_20230324"]
# plates = [
# "50_20240703",
# "59_20240615",
# "60_20240625",
# "69_20240625",
# "72_20240722"
# ]
# directory_targ = '/scratch-shared/amftrack/redo_width_temp/'
# directory_targ = "/projects/0/einf914/make_temp_graph/"
# directory_targ = "/projects/0/einf914/NewAggC3/"
directory_targ = "/scratch-shared/amftrack/make_temp_graph/"
mins = {'772_20230317': '20230317_1710_Plate13',
'777_20230328': '20230328_2010_Plate09',
'784_20230324': '20230324_2009_Plate03',
'771_20230411': '20230411_1419_Plate04',
'782_20230406': '20230406_2216_Plate17',
'935_20230620': '20230620_1844_Plate18',
'74_20230601': '20230601_1621_Plate18',
'418_20230626': '20230626_2020_Plate09',
'928_20230707': '20230707_1553_Plate14',
'429_20230822': '20230823_0947_Plate13',
'805_20230311': '20230313_2237_Plate16',
'796_20230419': '20230419_1213_Plate10',
'792_20230324': '20230330_0137_Plate11',
'894_20230516': '20230516_1438_Plate08',
'797_20230421': '20230422_1230_Plate08',
'947_20230706': '20230706_1933_Plate06',
'868_20230504': '20230504_1624_Plate03',
'954_20230717': '20230717_1525_Plate12',
'828_20230330': '20230404_1752_Plate08',
'822_20230327': '20230403_1537_Plate03',
'926_20230510': '20230510_1429_Plate13',
'905_20230525': '20230527_2017_Plate16',
'910_202305016': '20230516_1429_Plate05',
'969_20230801': '20230801_1541_Plate05',
'914_20230522': '20230524_2021_Plate14',
"513_20240531": "20240531_1853_Plate16",
"393_20231129":"20231130_0045_Plate06",
"366_20230921":"20230921_1421_Plate07",
'3_20220426':'20220504_2136_Plate16',
'1045_20220504':'20220505_0500_Plate02',
'559_20230109':'20230109_1638_Plate18',
'219_20230307':'20230307_1723_Plate11',
'252_20230316':'20230316_2048_Plate12'


        }

maxes = {'772_20230317': '20230322_1030_Plate13',
'777_20230328': '20230407_0752_Plate09',
'784_20230324': '20230327_0809_Plate03',
'771_20230411': '20230420_1201_Plate04',
'782_20230406': '20230412_0837_Plate17',
'935_20230620': '20230703_0841_Plate18',
'74_20230601': '20230606_1035_Plate18',
'418_20230626': '20230706_0942_Plate09',
'928_20230707': '20230712_0600_Plate14',
'429_20230822': '20230905_1202_Plate13',
'805_20230311': '20230317_1516_Plate16',
'796_20230419': '20230501_0829_Plate10',
'792_20230324': '20230403_0757_Plate11',
'894_20230516': '20230526_0801_Plate08',
'797_20230421': '20230426_1217_Plate08',
'947_20230706': '20230714_0739_Plate06',
'868_20230504': '20230509_1012_Plate03',
'954_20230717': '20230801_0947_Plate12',
'828_20230330': '20230411_0752_Plate08',
'822_20230327': '20230410_1737_Plate03',
'926_20230510': '20230522_0616_Plate13',
'905_20230525': '20230530_0817_Plate16',
'910_202305016': '20230526_1351_Plate05',
'969_20230801': '20230805_0544_Plate05',
'914_20230522': '20230601_1205_Plate14',
'907_20230525': '20230606_1041_Plate20',
"513_20240531":"20240609_0138_Plate16",
'393_20231129':'20231208_1225_Plate06',
'366_20230921':'20230929_1421_Plate07',
'3_20220426':'20220510_0808_Plate16',
'1045_20220504':'20220510_2207_Plate02',
'559_20230109':'20230118_0033_Plate18',
'219_20230307':'20230312_1240_Plate11',
'252_20230316':'20230324_0627_Plate12'
         }


update_plate_info(directory_targ, local=True)
all_folders = get_current_folders(directory_targ, local=True)
path_root = f"/projects/0/einf914/graph_stacks"

for plate_id in plates:
    # update_plate_info(directory_targ, local=True,strong_constraint=False)
    # all_folders = get_current_folders(directory_targ, local=True)
    folders = all_folders.loc[all_folders["unique_id"] == plate_id]
    # folders = folders.sort_values(by="folder")

    folders = folders.sort_values(by="datetime")
    init = mins[plate_id]
    end = maxes[plate_id]

    init_datetime = folders[folders['folder'] == init]['datetime'].values[0]
    end_datetime = folders[folders['folder'] == end]['datetime'].values[0]

    # Filter the DataFrame to include rows between "init" and "end" datetimes
    folders = folders[(folders['datetime'] >= init_datetime) & (folders['datetime'] <= end_datetime)]
    folders = folders.loc[folders["/Analysis/nx_graph_pruned_width.p"] == True]
    # i = indexes[plate_id_video]
    # i = np.where(folders['folder'] == indexes[plate_id_video])[0][0]
    # selection = folders[folders['folder'].isin(indexes.values())]
    for i in range(len(folders)):
        exp = Experiment(directory_targ)

        selection = folders.iloc[i : i + 1]
        folder_list = list(selection["folder"])
        directory_name = folder_list[0]
        path_snap = directory_targ + directory_name
        transform = sio.loadmat(path_snap + "/Analysis/transform_final.mat")
        exp.load(selection, suffix="_realigned")
        for t in range(exp.ts):
            exp.load_tile_information(t)
            exp.save_location = ""

            load_study_zone(exp)
        os.makedirs(
            os.path.join(
                path_root,
                plate_id,
            ),
            exist_ok=True,
        )
        graph = exp.nx_graph[0]
        graph_to_save = graph.copy()
        for u, v, data in graph_to_save.edges(data=True):
            data["pixels"] = data.pop("pixel_list")
        node_not_in_ROI = []
        for node in graph_to_save:
            if not is_in_ROI_node(Node(node, exp), 0):
                node_not_in_ROI.append(node)
        graph_to_save.remove_nodes_from(node_not_in_ROI)
        nx.set_node_attributes(graph_to_save, exp.positions[0], "position")
        path_tot = os.path.join(path_root, plate_id, f"graph{i:03d}.pickle")
        with open(path_tot, "wb") as f:
            pickle.dump((graph_to_save, transform, selection.iloc[0]), f)        # break
    gc.collect()

for plate_id in plates:
    saved_infos = []

    str_directory = os.path.join(
        path_root,
        plate_id,
    )
    directory = os.fsencode(str_directory)
    files = os.listdir(directory)
    files.sort()
    for file in tqdm(files):
        filename = os.fsdecode(file)
        file_path = os.path.join(str_directory, filename)
        if filename.endswith("pickle"):
            saved_infos.append(pickle.load(open(file_path, "rb")))

    graphs = [graph for graph, _, _ in saved_infos]
    folder_infos = [folder_info for _, _, folder_info in saved_infos]
    transforms = [(transform["R"], transform["t"]) for _, transform, _ in saved_infos]
    spatial_graphs = [SpatialGraph(graph) for graph in graphs]
    print("cleaned garbage")
    spatial_temporal_graph = spatial_temporal_graph_from_spatial_graphs(
        spatial_graphs,
        np.arange(len(spatial_graphs)),
        verbose=1,
        threshold=30,
        segments_length=30,
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
    path_tot = os.path.join(path_root, f"graph{plate_id}.pickle")
    pickle.dump(spatial_temporal_graph, open(path_tot, "wb"))
    target = f"/DATA/PRINCE_ANALYSIS/temp_graphs/graph{plate_id}.pickle"
    source = path_tot
    upload(source, target)
    # del graphs
    # del saved_infos
    # gc.collect()
