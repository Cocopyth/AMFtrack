import networkx as nx

from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Node,
    Edge, Experiment,
)
from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_all_edges,
    get_all_nodes, get_timedelta_second,
)
from amftrack.pipeline.functions.post_processing.extract_study_zone import (
    load_study_zone,
)
from amftrack.pipeline.functions.post_processing.util import (
    is_in_study_zone,
    is_in_ROI_node,
)
import numpy as np
import pandas as pd
from scipy import sparse

from amftrack.pipeline.functions.transport_processing.high_mag_videos.register_videos import (
    add_attribute,
)

# hyphae = pd.read_excel(
#     "/home/cbisot/pycode/AMFtrack/amftrack/notebooks/transport/hyphae.xlsx"
# )


def add_hyphal_attributes(edge_data_csv, edge, mapping, t, exp, unique_id):
    hyphae_plate = hyphae[hyphae["plate"] == unique_id]
    for index, row in hyphae_plate.iterrows():
        begin, end = row["begin"], row["end"]
        fun = lambda edge: get_abcisse(edge, begin, end, t, exp)
        add_attribute(edge_data_csv, edge, fun, f"abcisse_{begin}_{end}", mapping)


def find_lowest_nodes(nodes, t):
    positions = [node.pos(t) for node in nodes]

    # 1. Find the range of x positions
    min_x = min(positions, key=lambda p: p[1])[1]
    max_x = max(positions, key=lambda p: p[1])[1]

    # 2. Calculate the size of each part
    part_size = (max_x - min_x) / 10

    # 3. Group nodes by x range and select the node with the lowest y in each group
    selected_nodes = []
    for i in range(10):
        start_x = min_x + i * part_size
        end_x = start_x + part_size

        # Find nodes within the current x range
        nodes_in_range = [node for node in nodes if start_x <= node.pos(t)[1] < end_x]

        # 4. Select the node with the lowest y in the current range
        if nodes_in_range:
            lowest_y_node = max(nodes_in_range, key=lambda p: p.pos(t)[0])
            selected_nodes.append(lowest_y_node)
    return selected_nodes


Vmax = 1


def get_weight(node, t):
    weight = 0
    for edge in node.edges(t):
        weight += Vmax * np.pi * edge.width(t) / 2
        # weight += Vmax * np.pi / 2

    return weight


def get_shortest_path_edges(node2, shortest):
    exp = node2.experiment
    if node2.label in shortest.keys():
        nodes = shortest[node2.label]
    else:
        nodes = []
    edges = []
    for i in range(len(nodes) - 1):
        nodea = Node(nodes[i], exp)
        nodeb = Node(nodes[i + 1], exp)
        edges.append(Edge(nodea, nodeb, exp))
    return edges


def get_quantitative_BC_dic(exp, t, nodes_sink, nodes_source,weight_fun = get_weight):
    edges = get_all_edges(exp, t)
    edge_flux = {edge: 0 for edge in edges}
    for node1 in nodes_sink:
        shortest = nx.single_source_dijkstra_path(
            exp.nx_graph[t], node1.label, weight="length"
        )
        for node2 in nodes_source:
            w = weight_fun(node2,t)
            # print("here",w)
            path = get_shortest_path_edges(node2, shortest)
            # print(path)
            for edge in path:
                # print("here",w)
                edge_flux[edge] += w / len(nodes_sink)
    edge_flux_final = {(edge.begin.label,edge.end.label) : edge_flux[edge] for edge in edges}
    return(edge_flux_final)


def add_betweenness_QP(exp, t):
    exp.save_location = ""

    load_study_zone(exp)
    nodes = get_all_nodes(exp, t)
    nodes_source = [node for node in nodes if is_in_ROI_node(node, t)]
    nodes_sink = [node for node in nodes if is_in_ROI_node(node, t)]
    nodes_sink = find_lowest_nodes(nodes_sink, t)
    print("len sourcesink",len(nodes_sink),len(nodes_source))
    fluxes = get_quantitative_BC_dic(exp, t, nodes_sink, nodes_source)
    # print("fluxes",fluxes)
    for edge in exp.nx_graph[t].edges:
        if edge not in fluxes.keys() and (edge[1], edge[0]) not in fluxes.keys():
            fluxes[edge] = 0
    nx.set_edge_attributes(exp.nx_graph[t], fluxes, "betweenness_QP")


def add_betweenness(exp, t):
    print(f"compute BC{t}")
    exp.save_location = ""

    load_study_zone(exp)
    edges = get_all_edges(exp, t)
    nodes = get_all_nodes(exp, t)
    nodes_source = [
        node
        for node in nodes
        if not is_in_study_zone(node, t, 1000, 100)[1]
        and is_in_study_zone(node, t, 1000, 100)[0]
    ]
    nodes_source = nodes_source
    # nodes_sink = [
    #     node
    #     for node in nodes
    #     if is_in_study_zone(node, t, 1000, 150)[1]
    # ]
    nodes_sink = [
        node
        for node in nodes
        if is_in_study_zone(node, t, 1000, 150)[1]
        if node.degree(t) == 1
    ]
    weights = {(edge.begin.label, edge.end.label): edge.length_um(t) for edge in edges}
    nx.set_edge_attributes(exp.nx_graph[t], weights, "length")
    weights = {
        (edge.begin.label, edge.end.label): 1 / edge.length_um(t) for edge in edges
    }
    nx.set_edge_attributes(exp.nx_graph[t], weights, "1/length")
    G = exp.nx_graph[t]
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    final_betweeness = {}

    for g in S:
        source = [node.label for node in nodes_source if node.label in g]
        sink = [node.label for node in nodes_sink if node.label in g]

        betweeness = nx.edge_betweenness_centrality_subset(
            g, source, sink, normalized=True, weight="length"
        )
        for edge in betweeness.keys():
            final_betweeness[edge] = betweeness[edge]

    for edge in exp.nx_graph[t].edges:
        if (
            edge not in final_betweeness.keys()
            and (edge[1], edge[0]) not in final_betweeness.keys()
        ):
            final_betweeness[edge] = 0
    print("adding BC",t)
    nx.set_edge_attributes(exp.nx_graph[t], final_betweeness, "betweenness")

def get_abcisse(edge, begin, end, t, exp):
    begin = Node(begin, exp).get_pseudo_identity(t).label
    try:
        if Node(end, exp).is_in(t):
            nodes = nx.shortest_path(exp.nx_graph[t], begin, end, weight="weight")
            nodes = [Node(node, exp) for node in nodes]
            edges = [Edge(nodes[i], nodes[i + 1], exp) for i in range(len(nodes) - 1)]
            lengths = [edge.length_um(t) for edge in edges]
            abciss = np.cumsum(lengths)
            if edge in edges:
                i = edges.index(edge)
                return abciss[i]
    except nx.NetworkXNoPath:
        return -1
    return -1


def get_derivative(edge_d, t, fun):
    edges_begin = edge_d.begin.edges(t)
    edges_begin.remove(edge_d)
    edges_end = edge_d.end.edges(t)
    edges_end.remove(edge_d)
    weight_begin = np.sum([fun(edge) for edge in edges_begin])
    weight_end = np.sum([fun(edge) for edge in edges_end])
    return weight_end - weight_begin

def get_segment_centers(exp):
    #From Amin code, to update when new functions are available
    last_index = 1
    # Size of the segment in pixels
    segments_length = 5

    final_graph = exp.nx_graph[last_index]
    node_not_in_ROI = []
    for node in final_graph:
        if not is_in_ROI_node(Node(node, exp), last_index):
            node_not_in_ROI.append(node)
    final_graph.remove_nodes_from(node_not_in_ROI)

    label = max(final_graph.nodes) + 1
    graph_segemented_final = nx.empty_graph()
    nodes_pos = {}
    edges_indexes = {}
    segments_index = {}
    segments_center_final = []

    for edge in final_graph.edges:
        e = Edge(Node(edge[0], exp), Node(edge[1], exp), exp)
        edges_indexes[f"{edge[0]},{edge[1]}"] = []
        pixels = e.pixel_list(last_index)
        length = len(pixels)
        if length < segments_length:
            graph_segemented_final.add_edge(edge[0], edge[1])
            segments_index[f"{edge[0]},{edge[1]}"] = len(segments_center_final)
            edges_indexes[f"{edge[0]},{edge[1]}"].append(len(segments_center_final))
            central_point = np.mean(np.array(pixels), axis=0)
            segments_center_final.append(central_point)
            nodes_pos[edge[0]] = pixels[0]
            nodes_pos[edge[1]] = pixels[-1]
            continue

        for i in range(0, length, segments_length):
            sub_list = pixels[i:i + segments_length]
            if i == 0:
                graph_segemented_final.add_edge(edge[0], label)
                segments_index[f"{edge[0]},{label}"] = len(segments_center_final)
                edges_indexes[f"{edge[0]},{edge[1]}"].append(len(segments_center_final))
                central_point = np.mean(np.array(sub_list), axis=0)
                segments_center_final.append(central_point)
                nodes_pos[edge[0]] = sub_list[0]
                nodes_pos[label] = sub_list[-1]
                label += 1
            elif i + segments_length >= length:
                graph_segemented_final.add_edge(label - 1, edge[1])
                segments_index[f"{label - 1},{edge[1]}"] = len(segments_center_final)
                edges_indexes[f"{edge[0]},{edge[1]}"].append(len(segments_center_final))
                central_point = np.mean(np.array(sub_list), axis=0)
                segments_center_final.append(central_point)
                nodes_pos[edge[1]] = sub_list[-1]
            else:
                graph_segemented_final.add_edge(label - 1, label)
                segments_index[f"{label - 1},{label}"] = len(segments_center_final)
                edges_indexes[f"{edge[0]},{edge[1]}"].append(len(segments_center_final))
                central_point = np.mean(np.array(sub_list), axis=0)
                segments_center_final.append(central_point)
                nodes_pos[label] = sub_list[-1]
                label += 1

    array_segments_center_final = np.array(segments_center_final)
    shape_segments_center = array_segments_center_final.shape
    return(array_segments_center_final,shape_segments_center,
           final_graph,edges_indexes,
           graph_segemented_final,nodes_pos,segments_index)


def closest_point(point, points):
    dist_square = np.sum((points - point) ** 2, axis=1)
    min_index = np.argmin(dist_square)
    return points[min_index], dist_square[min_index]

def get_exp2(exp,segment_length = 5):
    last_index = 1
    segments_length = segment_length
    #From Amin code, to update when new functions are available
    array_segments_center_final, shape_segments_center,\
    final_graph,edges_indexes,graph_segemented_final,nodes_pos,segments_index = get_segment_centers(exp)
    threshold = (2*segment_length) ** 2

    segments_centers = []
    segments_min_distances = []
    array_segments_center = array_segments_center_final.copy()
    for time in reversed(range(last_index + 1)):
        print(f"Process time {time}")
        rows = []
        cols = []
        previous_edges = get_all_edges(exp, time)
        for edge in previous_edges:
            p_list = edge.pixel_list(time)
            row, col = zip(*p_list)
            rows.extend(row)
            cols.extend(col)

        data = np.ones(len(rows))
        points_matrix = sparse.csr_matrix((data, (rows, cols)))

        centers_distance = []
        new_centers = array_segments_center.copy()
        for index, center in enumerate(array_segments_center):
            xc, yc = center
            xc, yc = int(xc), int(yc)

            min_x, max_x = max(0, xc - 4 * segments_length), xc + 4 * segments_length
            min_y, max_y = max(0, yc - 4 * segments_length), yc + 4 * segments_length
            coords = points_matrix[min_x:max_x, min_y:max_y].nonzero()
            coords = np.column_stack(coords)
            if not coords.shape[0]:
                centers_distance.append(32 * (segments_length ** 2))
                continue

            xc -= min_x
            yc -= min_y

            new_center, min_dist = closest_point([xc, yc], coords)
            centers_distance.append(min_dist)
            if min_dist < threshold:
                new_centers[index] = new_center + np.array([min_x, min_y])

        array_segments_center = new_centers
        segments_centers.append(new_centers)
        segments_min_distances.append(centers_distance)

    segments_min_distances.reverse()
    # Index t are the centers of the segments at time t
    segments_centers.reverse()
    # Amount of segment to look for at in an edge to get the date at which the edge encounter the node
    # Depends of how big segments are and what threshold you use
    amount_of_border_segment = 7

    segments_min_distances_array = np.array(segments_min_distances)
    segments_min_distances_array = np.where(segments_min_distances_array < threshold, 1, 0)
    segments_time = segments_min_distances_array.argmax(axis=0)

    edges_time_interval = {}

    for e in final_graph.edges:
        edge = Edge(Node(e[0], exp), Node(e[1], exp), exp)
        segments_indexes = edges_indexes[f"{edge.begin.label},{edge.end.label}"]
        segments_times = np.array([segments_time[index] for index in segments_indexes])

        begin = np.median(segments_times[:amount_of_border_segment])
        if len(segments_times) > amount_of_border_segment:
            end = np.median(segments_times[-amount_of_border_segment:])
        else:
            end = np.median(segments_times)

        edges_time_interval[f"{edge.begin.label},{edge.end.label}"] = (begin, end)
    exp2 = Experiment(exp.directory)
    # i = indexes[plate_id_video]
    # i = np.where(folders['folder'] == indexes[plate_id_video])[0][0]
    # selection = folders[folders['folder'].isin(indexes.values())]
    i = 0
    exp2.load(exp.folders.iloc[1: 2], suffix="_labeled2")
    exp2.nx_graph = [graph_segemented_final]
    exp2.positions = [nodes_pos]
    pixel_lists = {}
    lengths = {}

    t = 0
    for key, edge in enumerate(graph_segemented_final.edges):
        begin, end = edge
        (y1, x1), (y2, x2) = nodes_pos[begin], nodes_pos[end]
        pixel_lists[edge] = [(y1, x1), (y2, x2)]
        pixel_conversion_factor = 1.725
        lengths[edge] = np.sqrt((y1-y2)**2+(x1-x2)**2)*pixel_conversion_factor

    nx.set_edge_attributes(exp2.nx_graph[t], pixel_lists, "pixel_list")
    edges = get_all_edges(exp2, 0)
    ages = {(edge.begin.label, edge.end.label): get_age(edge,segments_time,segments_index) for edge in edges}
    nx.set_edge_attributes(exp2.nx_graph[t], ages, "age")
    nx.set_edge_attributes(exp2.nx_graph[t], lengths, "length")
    return(exp2)

def get_age(edge,segments_time,segments_index):
    begin,end = edge.begin.label,edge.end.label
    index = segments_index.get(f"{begin},{end}")
    if index is None:
        index = segments_index[f"{end},{begin}"]
    time = segments_time[index]
    return(time)

def get_nodes_source_C(exp):
    assert len(exp.folders)==2
    time_delta = get_timedelta_second(exp,0,1)
    exp2 = get_exp2(exp)
    G = exp2.nx_graph[0]
    subgraph_age_0 = nx.Graph([e for e in G.edges(data=True) if e[2]['age'] == 0])
    subgraph_age_1 = nx.Graph([e for e in G.edges(data=True) if e[2]['age'] == 1])
    components_age_1 = list(nx.connected_components(subgraph_age_1))
    nodes = get_all_nodes(exp2, 0)
    weights = {node: 0 for node in nodes}
    # For each edge in subgraph_age_1, find its component
    for edge in subgraph_age_1.edges(data=True):
        node_u, node_v, data = edge
        for component in components_age_1:
            if node_u in component or node_v in component:
                connected_nodes = set()
                for node in component:
                    if node in subgraph_age_0:
                        connected_nodes.add(node)
                        #to fix, should be quantitative (include radius and time etc...)
                for node in connected_nodes:
                    weights[Node(node, exp2)] += edge[2]["length"]/len(connected_nodes)/time_delta
    nodes = get_all_nodes(exp2, 0)

    nodes_exp2 = [node for node in nodes if weights[node] >= 1]
    t = 0
    nodes_exp = {find_pseudo_identity(node_exp2, t, exp): weights[node_exp2] for node_exp2 in nodes_exp2}
    print("tot_growth",np.sum(list(nodes_exp.values())))
    nodes_source = list(nodes_exp.keys())
    return(nodes_source,nodes_exp)

def get_weight_C(node,t,nodes_exp):
    if node in nodes_exp.keys():
        return(nodes_exp[node])
    else:
        return(0)


def add_betweenness_QC(exp, t):
    load_study_zone(exp)
    nodes = get_all_nodes(exp, t)
    nodes_source,nodes_exp = get_nodes_source_C(exp)
    nodes_sink = [node for node in nodes if is_in_ROI_node(node, t)]
    # nodes_sink = [
    #     node
    #     for node in nodes
    #     if is_in_study_zone(node, t, 1000, 150)[1]
    # ]
    nodes_sink = find_lowest_nodes(nodes_sink, t)
    fluxes = get_quantitative_BC_dic(exp, t, nodes_sink, nodes_source,lambda node,t : get_weight_C(node,t,nodes_exp))
    # print("fluxes",fluxes)
    for edge in exp.nx_graph[t].edges:
        if edge not in fluxes.keys() and (edge[1], edge[0]) not in fluxes.keys():
            fluxes[edge] = 0
    nx.set_edge_attributes(exp.nx_graph[t], fluxes, "betweenness_QC")

def find_pseudo_identity(node_exp2,t,exp):
    identifier = 0
    mini = np.inf
    poss = exp.positions[t]
    pos_root = node_exp2.pos(0)
    for node in exp.nx_graph[t]:
        distance = np.linalg.norm(poss[node] - pos_root)
        if distance < mini:
            mini = distance
            identifier = node
    return Node(identifier, exp)
