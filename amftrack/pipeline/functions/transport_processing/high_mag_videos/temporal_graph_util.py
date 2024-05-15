import pickle
import os

from amftrack.util.sys import (

    update_plate_info,

    get_current_folders,
)

from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment, Node, Edge,
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
from sthype import SpatialGraph, HyperGraph
from sthype.hypergraph.hypergraph_from_spatial_graphs import spatial_temporal_graph_from_spatial_graphs
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
import pandas as pd
from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_all_edges,
    get_all_nodes, get_timedelta_second,
)
from matplotlib import cm
from amftrack.pipeline.functions.transport_processing.high_mag_videos.plotting import *

downsizing = 5

# def interpolate_points(p1, p2, num_points=10):
#     y1, x1 = p1
#     y2, x2 = p2
#     interpolated_points = [(y1 + i / num_points * (y2 - y1), x1 + i / num_points * (x2 - x1)) for i in range(num_points + 1)]
#     return interpolated_points

def interpolate_points(p1, p2, spacing=1.0):
    y1, x1 = p1
    y2, x2 = p2

    # Calculate the distance between points
    distance = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    # Calculate the number of intervals based on the spacing
    num_points = int(distance / spacing)+1

    # Create interpolated points
    interpolated_points = [(y1 + i / num_points * (y2 - y1), x1 + i / num_points * (x2 - x1)) for i in range(num_points + 1)]

    return interpolated_points

def load(path):
    spatial_temporal_graph = pickle.load(open(path, 'rb'))
    folders = pd.concat(spatial_temporal_graph.folder_infos, axis=1).transpose()
    return(spatial_temporal_graph,folders)

def make_exp(spatial_temporal_graph,folders,make_pixel_list = False,spacing = 1):
    exp = Experiment("")
    exp.folders = folders
    exp.nx_graph = [spatial_temporal_graph]
    exp.positions = [nx.get_node_attributes(spatial_temporal_graph, 'position')]
    exp.dimX_dimY = (3000, 4096)
    dico = nx.get_node_attributes(spatial_temporal_graph, 'position')
    positions = {node: (dico[node].x, dico[node].y) for node in dico.keys()}
    exp.positions = [positions]
    pixel_lists = {}
    if make_pixel_list:
        for node1, node2, edge_data in spatial_temporal_graph.edges(data=True):
            position1, position2 = exp.positions[0][node1], exp.positions[0][node2]
            (y1, x1), (y2, x2) = position1, position2
            edge = node1, node2
            p1 = (y1, x1)
            p2 = (y2, x2)

            # Interpolate points
            interpolated_points = interpolate_points(p1, p2,spacing = spacing)

            # Store them in pixel_lists[edge]
            pixel_lists[edge] = interpolated_points
            # pixel_lists[edge] = [(y1, x1), (y2, x2)]
            # print([(y1, x1), (y2, x2)])

        nx.set_edge_attributes(exp.nx_graph[0], pixel_lists, "pixel_list")
    return(exp)

def activation_continuity(G,node):
    edges = tuple(G.edges(node,data=True))
    return(edges[0][2]['activation']==edges[1][2]['activation'])

def get_min_activation(G,node):
    edges = list(G.edges(node,data=True))
    return(min([edge[2]['activation'] for edge in edges]))

def simplify(nx_graph):
    pos = nx.get_node_attributes(nx_graph, 'position')
    degree_2_nodes = [node for node in nx_graph.nodes if
                      nx_graph.degree(node) == 2 and activation_continuity(nx_graph, node)]
    list_attributes = ["width","activation"]
    while len(degree_2_nodes) > 0:
        print(len(degree_2_nodes))
        for node in degree_2_nodes:
            # node = degree_2_nodes.pop()
            if nx_graph.degree(node) == 2:
                neighbours = list(nx_graph.neighbors(node))
                right_n = neighbours[0]
                left_n = neighbours[1]
                right_edge = nx_graph.get_edge_data(node, right_n)["pixel_list"]
                left_edge = nx_graph.get_edge_data(node, left_n)["pixel_list"]
                if np.any(np.round(right_edge[0]) != np.round(np.array((pos[node].x, pos[node].y)))):
                    right_edge = list(reversed(right_edge))
                if np.any(np.round(left_edge[-1]) != np.round(np.array((pos[node].x, pos[node].y)))):
                    left_edge = list(reversed(left_edge))
                pixel_list = left_edge + right_edge[1:]
                info = {"weight": len(pixel_list), "pixel_list": pixel_list}

                for attribute in list_attributes:
                    right_edge_attribute = nx_graph.get_edge_data(node, right_n)[attribute]
                    left_edge_attribute = nx_graph.get_edge_data(node, left_n)[attribute]

                    attribute_new = (
                                            right_edge_attribute * len(right_edge) + left_edge_attribute * len(left_edge)
                                    ) / (len(right_edge) + len(left_edge))
                    if attribute == "activation":
                        attribute_new = int(attribute_new)
                        # print(right_edge_attribute,left_edge_attribute,attribute_new)
                    info[attribute] = attribute_new

                nx_graph.add_edges_from([(right_n, left_n, info)])
                nx_graph.remove_node(node)
        degree_2_nodes = [node for node in nx_graph.nodes if
                          nx_graph.degree(node) == 2 and activation_continuity(nx_graph, node)]
    degree_0_nodes = [node for node in nx_graph.nodes if
                      nx_graph.degree(node) <= 1]
    for node in degree_0_nodes:
        nx_graph.remove_node(node)
    return(nx_graph)

def get_growing_nodes(exp,index0,index1):
    exp.dates = list(exp.folders["datetime"])
    k = 0
    edges = get_all_edges(exp, k)

    weights = {(edge.begin.label, edge.end.label): edge.length_um(k) for edge in edges}
    nx.set_edge_attributes(exp.nx_graph[k], weights, "length")
    r0 = 3
    time_delta = get_timedelta_second(exp, index0, index1)
    G = exp.nx_graph[0]
    subgraph_age_0 = nx.Graph([e for e in G.edges(data=True) if e[2]['activation'] <= index0])
    subgraph_age_1 = nx.Graph(
        [e for e in G.edges(data=True) if e[2]['activation'] <= index1 and e[2]['activation'] > index0])
    components_age_1 = list(nx.connected_components(subgraph_age_1))
    nodes = get_all_nodes(exp, 0)
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
                        # to fix, should be quantitative (include radius and time etc...)
                for node in connected_nodes:
                    weights[Node(node, exp)] += np.pi * r0 ** 2 * edge[2]["length"] / len(connected_nodes) / time_delta
    nodes = get_all_nodes(exp, 0)
    nodes_exp = [node for node in nodes if weights[node] / (np.pi * r0 ** 2) * 3600 > 0]
    return(weights, nodes_exp)

def create_subgraph_by_attribute(G, attribute, threshold):
    """
    Creates a subgraph from a given graph `G` that includes only nodes with a specified
    attribute value below a certain threshold.

    Parameters:
    - G (nx.Graph): The original graph.
    - attribute (str): The node attribute to check.
    - threshold (float): The threshold value for the attribute.

    Returns:
    - nx.Graph: A new graph containing only the nodes (and their edges) for which
                the specified attribute is below the given threshold.
    """
    # Initialize a new graph of the same type as the original
    new_graph = SpatialGraph(nx.Graph())

    for u, v, attrs in G.edges(data=True):
        # Check if the attribute exists and is below the threshold
        if attrs[attribute] <= threshold:
            # Add both nodes and the edge to the new graph
            if not new_graph.has_node(u):
                new_graph.add_node(u, **G.nodes[u])
            if not new_graph.has_node(v):
                new_graph.add_node(v, **G.nodes[v])
            new_graph.add_edge(u, v, **attrs)
    isolated_nodes = [node for node in new_graph.nodes() if new_graph.degree(node) == 0]
    new_graph.remove_nodes_from(isolated_nodes)
    return new_graph

def create_subgraph_from_nodelist(G, node_list):
    """
    Creates a subgraph from a given graph `G` that includes only nodes with a specified
    attribute value below a certain threshold.

    Parameters:
    - G (nx.Graph): The original graph.
    - attribute (str): The node attribute to check.
    - threshold (float): The threshold value for the attribute.

    Returns:
    - nx.Graph: A new graph containing only the nodes (and their edges) for which
                the specified attribute is below the given threshold.
    """
    # Initialize a new graph of the same type as the original
    new_graph = SpatialGraph(nx.Graph())

    for u, v, attrs in G.edges(data=True):
        # Check if the attribute exists and is below the threshold
        if u in node_list:
            # Add both nodes and the edge to the new graph
            if not new_graph.has_node(u):
                new_graph.add_node(u, **G.nodes[u])
            if not new_graph.has_node(v):
                new_graph.add_node(v, **G.nodes[v])
            new_graph.add_edge(u, v, **attrs)
    isolated_nodes = [node for node in new_graph.nodes() if new_graph.degree(node) == 0]
    new_graph.remove_nodes_from(isolated_nodes)
    return new_graph


def convert_to_directed(undirected_graph):
    # Create a new directed graph
    directed_graph = nx.DiGraph()

    # Define constant values for dynamic viscosity (mu)
    mu = 0.1  # Assumed for example purposes

    # Add edges to the directed graph with direction based on node label ordering
    for u, v, attr in undirected_graph.edges(data=True):
        if u < v:
            directed_graph.add_edge(u, v, **attr)
        else:
            directed_graph.add_edge(v, u, **attr)

    # Calculate the resistance and potential for each edge in the directed graph
    for u, v, attr in directed_graph.edges(data=True):
        attr['radius'] = max(attr['width'] / 2, 0.5)
        attr['v0'] = 0

        attr['Res'] = 8 * mu * attr["length"] / (np.pi * attr['radius'] ** 4)
        attr['pot'] = -8 * mu * attr["length"] * attr['v0'] / (np.pi * attr['radius'] ** 2)

    return directed_graph


def build_matrix_system(G, ext_flows):
    node_index = {node: idx for idx, node in enumerate(list(G.nodes)[:-1])}
    edge_index = {(u, v): idx for idx, (u, v) in enumerate(G.edges())}

    num_nodes = len(node_index)
    num_edges = len(edge_index)
    print(len(num_nodes,num_edges))
    A = np.zeros((num_nodes, num_edges))
    b = np.zeros(num_nodes)

    # Node equations for mass conservation
    for (u, v), idx in edge_index.items():
        if u in node_index:
            A[node_index[u], idx] -= 1
        if v in node_index:
            A[node_index[v], idx] += 1

    for node, idx in node_index.items():
        b[idx] = ext_flows.get(node, 0)

    # Loop equations for energy conservation
    cycles = nx.cycle_basis(G.to_undirected())
    print('found all cycles')
    for cycle in tqdm(cycles, desc='Processing Cycles'):  # Add tqdm progress bar
        if len(cycle) > 2:  # ensuring it's a simple cycle
            new_row = np.zeros(num_edges)
            tot_pot = 0
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if (u, v) in edge_index:
                    idx = edge_index[(u, v)]
                    new_row[idx] += G[u][v]['Res']
                    tot_pot += G[u][v]['pot']
                elif (v, u) in edge_index:
                    idx = edge_index[(v, u)]
                    new_row[idx] -= G[v][u]['Res']
                    tot_pot -= G[v][u]['pot']

            A = np.vstack([A, new_row])
            b = np.append(b, tot_pot)

    return A, b


def solve_flows(A, b):
    return np.linalg.solve(A, b)

def add_flows(G,nodes_source,nodes_sink,weights):
    DG = convert_to_directed(G)
    ext_flows_in = {node_source.label: -weights[node_source] for node_source in
                    nodes_source}  # external flows: positive into the network, negative out
    for node in nodes_sink:
        ext_flows_in[node.label] = 0
    tot_flow = np.sum(list(ext_flows_in.values()))
    ext_flows_out = {node_sink.label: -tot_flow / len(nodes_sink) for node_sink in
                     nodes_sink}  # external flows: positive into the network, negative out
    ext_flows = {**ext_flows_in, **ext_flows_out}
    print("net_flow",np.sum(list(ext_flows.values())))
    A, b = build_matrix_system(DG, ext_flows)
    flows = solve_flows(A, b)
    edge_flows = {edge: flow for edge, flow in zip(DG.edges(), flows)}
    nx.set_edge_attributes(G, edge_flows, "water_flux")
    for edge in G.edges:
        G[edge[0]][edge[1]]["water_flux_abs"] = abs(G[edge[0]][edge[1]]["water_flux"])
        G[edge[0]][edge[1]]["radius"] = max(G[edge[0]][edge[1]]["width"] / 2, 1)

        G[edge[0]][edge[1]]["speed"] = G[edge[0]][edge[1]]["water_flux"] / (np.pi * G[edge[0]][edge[1]]["radius"] ** 2)

def add_lipid_flux(G,nodes_source,nodes_sink,weights):
    edge_flux1 = {edge: 0 for edge in G.edges}
    edge_flux2 = {(edge[1],edge[0]): 0 for edge in G.edges}
    edge_flux = {**edge_flux1, **edge_flux2}

    shortests = {}
    for node1 in nodes_sink:
        shortest = nx.single_source_dijkstra_path(
            G, node1.label, weight="length"
        )
        shortests[node1] = shortest

    for node2 in nodes_source:
        w = weights[node2]
        len_path = []
        for node1 in nodes_sink:
            shortest = shortests[node1]
            path = get_shortest_path_edges(node2, shortest)
            len_path.append(len(path))
            # print(path)
        node1 = nodes_sink[np.argmin(len_path)]
        shortest = shortests[node1]
        path = get_shortest_path_edges(node2, shortest)
        for edge in path:
            # print("here",w)
            edge_flux[edge] += w
    for edge in edge_flux.keys():
        if edge[0] > edge[1]:
            G[edge[0]][edge[1]]["QBC_p"] = edge_flux[edge]  # Attribute for edge from A to B
        else:
            G[edge[0]][edge[1]]["QBC_m"] = edge_flux[edge]  # Attribute for edge from A to B
    for edge in G.edges:
        G[edge[0]][edge[1]]["QBC_net"] = G[edge[0]][edge[1]]["QBC_p"] - G[edge[0]][edge[1]]["QBC_m"]
        G[edge[0]][edge[1]]["QBC_tot"] = G[edge[0]][edge[1]]["QBC_p"] + G[edge[0]][edge[1]]["QBC_m"]


def get_shortest_path_edges(node2, shortest):
    if node2.label in shortest.keys():
        nodes = shortest[node2.label]
    else:
        nodes = []
    edges = []
    for i in range(len(nodes) - 1):
        edges.append((nodes[i],nodes[i+1]))
    return edges

def plot_region(exp0,region,attribute,vmax):
    fig, ax,f = plot_edge_color_value_3(
        exp0,
        0,
        lambda edge : abs(edge.get_attribute(attribute,0)),
        cmap=cm.get_cmap("viridis", 100),
        plot_cmap=True,
        show_background=False,
        dilation=10,
        figsize=(5, 3),
        alpha = 1,
        downsizing = downsizing,
        region = region,
        v_min = 0,
        v_max = vmax
    )
    edges_network = get_all_edges(exp0,0)

    for edge in edges_network:
        pixels = edge.pixel_list(0)

        if is_in_bounding_box(pixels[0], region) or is_in_bounding_box(pixels[-1], region):
            if len(pixels) > 100:
                begin_arrow = np.array(f(pixels[20]))
                end_arrow = np.array(f(pixels[-20]))
                if abs(edge.get_attribute(attribute, 0))>0:
                    relative_flux = edge.get_attribute(attribute, 0) / edge.get_attribute("QBC_tot", 0)
                    orientation = 1 - 2 * (relative_flux > 0)
                    orientation *= (1-2 * (edge.begin.label > edge.end.label))
                    color = 'red'
                    if orientation<0:
                        begin_arrow,end_arrow = end_arrow,begin_arrow
                    plot_arrows_along_edge(ax, begin_arrow, end_arrow, color)
    plt.xticks([])  # Removes
    plt.yticks([])  # Removes x-ticks and x-tick labelsx-ticks and x-tick labels

def plot_arrows_along_edge(ax, begin, end, color, spacing=60):
    # Calculate the vector and its magnitude
    vector = end - begin
    distance = np.linalg.norm(vector)
    unit_vector = vector / distance

    # Calculate the number of arrows to create based on the spacing
    num_arrows = int(distance / spacing)
    if num_arrows<2:
        spacing= distance/2
        num_arrows = 1
    # Plot each arrow along the edge
    for i in range(1, num_arrows + 1):
        point = begin + unit_vector * (i * spacing)
        ax.annotate('', xytext=(point[1], point[0]), xy=(point[1] + unit_vector[1], point[0] + unit_vector[0]),
                    arrowprops=dict(arrowstyle="->", color=color))