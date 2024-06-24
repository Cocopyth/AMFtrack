
from amftrack.pipeline.functions.transport_processing.high_mag_videos.temporal_graph_util import *
from amftrack.pipeline.functions.transport_processing.high_mag_videos.add_BC import *

path_root = f"/projects/0/einf914/graph_stacks"
plates = [
    "441_20230807",
    "449_20230807",
    "310_20230830"
]
for plate_id in plates:
    path_tot = os.path.join(path_root,f"graph{plate_id}.pickle")
    print('loading_graph')
    spatial_temporal_graph,folders = load(path_tot)
    exp = make_exp(spatial_temporal_graph,folders,make_pixel_list=True)
    spatial_temporal_graph = simplify(spatial_temporal_graph)
    fix_attributes(spatial_temporal_graph)
    exp = make_exp(spatial_temporal_graph,folders)


    refs = {
        "310_20230830": {
            "20230901_Plate310": "20230901_0719_Plate06",
            "20230902_Plate310": "20230902_1343_Plate07",
            "20230903_Plate310": "20230903_1143_Plate07",
            "20230904_Plate310": "20230904_0942_Plate07",
            "20230905_Plate310": "20230905_1345_Plate07",
            # "20230906_Plate310" : "20230906_1220_Plate07",
        },
        "441_20230807": {
            "20230810_Plate441": "20230810_1005_Plate14",
            "20230811_Plate441": "20230811_1605_Plate14",
            "20230812_Plate441": "20230812_1006_Plate14",
            # "20230813_Plate441": "20230813_1618_Plate14",
        },
        "449_20230807": {
            "20230813_Plate449": "20230813_1606_Plate10",
            "20230814_Plate449": "20230814_1019_Plate10",
            "20230815_Plate449": "20230815_1021_Plate10",
            # "20230816_Plate449": "20230816_1027_Plate10",
            # "20230818_Plate449": "20230818_1107_Plate10",
        },
    }
    r0 = 3

    indexes = refs[plate_id]
    for plate_id_video in list(indexes.keys()):
        index0 = np.where(folders["folder"] == indexes[plate_id_video])[0][0]
        index1 = index0+1
        exp1 = add_fluxes(exp,index0,index1,folders)
        for edge in exp1.nx_graph[0].edges:
            for attribute in ["QBC_net","QBC_tot","speed_backflow","water_flux","speed_heaton","water_flux_heaton","speed_backflow2","water_flux2","speed_backflow_phase"]:
                spatial_temporal_graph[edge[0]][edge[1]][str(index0)][attribute] = exp1.nx_graph[0][edge[0]][edge[1]][attribute]
        # break

    spatial_temporal_graph.folder_infos = [folders.transpose()]
    path_tot = os.path.join(path_root,f"graph{plate_id}_flux.pickle")
    pickle.dump(spatial_temporal_graph, open(path_tot, 'wb'))
    target = f"/DATA/CocoTransport/graphs/graph{plate_id}_flux.pickle"
    source = path_tot
    upload(source, target)