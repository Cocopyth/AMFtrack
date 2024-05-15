
from amftrack.pipeline.functions.transport_processing.high_mag_videos.temporal_graph_util import *
from amftrack.pipeline.functions.transport_processing.high_mag_videos.add_BC import *

path_root = f"/scratch-shared/amftrack/graph_stacks"
plates = [
    "441_20230807", "449_20230807", "310_20230830"
]
plate_id = plates[0]
path_tot = os.path.join(path_root,f"graph{plate_id}.pickle")
spatial_temporal_graph,folders = load(path_tot)
exp = make_exp(spatial_temporal_graph,folders,make_pixel_list=True)
spatial_temporal_graph = simplify(spatial_temporal_graph)
exp = make_exp(spatial_temporal_graph,folders)
for edge in spatial_temporal_graph.edges:
    spatial_temporal_graph[edge[0]][edge[1]]["QBC_net"] = {}
    spatial_temporal_graph[edge[0]][edge[1]]["QBC_tot"] = {}
    spatial_temporal_graph[edge[0]][edge[1]]["water_flux"] = {}

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
        "20230816_Plate449": "20230816_1027_Plate10",
        # "20230818_Plate449": "20230818_1107_Plate10",
    },
}
r0 = 3

indexes = refs[plate_id]
for plate_id_video in list(indexes.keys()):
    index0 = np.where(folders["folder"] == indexes[plate_id_video])[0][0]
    index1 = index0+1
    print("folder length",len(folders),index0)

    weights, nodes_exp = get_growing_nodes(exp,index0,index1)
    nodes_source = [node for node in nodes_exp if weights[node] / (np.pi * r0 ** 2) * 3600 > 10]
    nodes_source = [node for node in nodes_source if weights[node] / (np.pi * r0 ** 2) * 3600 <= 1000]
    G0 = create_subgraph_by_attribute(spatial_temporal_graph, "activation", index0)
    components = nx.connected_components(G0)
    largest_component = max(components, key=len)
    largest_component_graph = create_subgraph_from_nodelist(G0, largest_component)

    exp1 = make_exp(largest_component_graph, folders)
    nodes = get_all_nodes(exp1, 0)
    nodes_sink = [node for node in nodes if get_min_activation(largest_component_graph, node.label) <= index0]
    nodes_sink = find_lowest_nodes(nodes_sink, 0, 20)
    nodes_source = [node for node in nodes_source if node in nodes]
    print("adding_lipid")

    add_lipid_flux(exp1.nx_graph[0], nodes_source, nodes_sink, weights)
    for edge in exp1.nx_graph[0].edges:
        spatial_temporal_graph[edge[0]][edge[1]]["QBC_net"][indexes[plate_id_video]] = \
        exp1.nx_graph[0][edge[0]][edge[1]]["QBC_net"]
        spatial_temporal_graph[edge[0]][edge[1]]["QBC_tot"][indexes[plate_id_video]] = \
            exp1.nx_graph[0][edge[0]][edge[1]]["QBC_tot"]
    print("adding_flows")
    add_flows(exp1.nx_graph[0], nodes_source, nodes_sink, weights)
    for edge in exp1.nx_graph[0].edges:
        spatial_temporal_graph[edge[0]][edge[1]]["water_flux"][indexes[plate_id_video]] = \
        exp1.nx_graph[0][edge[0]][edge[1]]["water_flux"]
    # break

spatial_temporal_graph.folder_infos = [folders.transpose()]
path_tot = os.path.join(path_root,f"graph{plate_id}_flux.pickle")
pickle.dump(spatial_temporal_graph, open(path_tot, 'wb'))
target = f"/DATA/CocoTransport/graphs/graph{plate_id}_flux.pickle"
source = path_tot
upload(source, target)