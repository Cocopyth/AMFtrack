import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas
from PIL import Image
import numpy as np
from scipy import sparse
from pymatreader import read_mat
import networkx as nx
import pandas as pd
import csv
import ast
from collections import Counter


def orient(pixel_list, root_pos):
    if np.all(root_pos == pixel_list[0]):
        return pixel_list
    else:
        return list(reversed(pixel_list))


def dic_to_sparse(dico):
    indptr = dico["jc"]
    indices = dico["ir"]
    datar = dico["data"]
    return sparse.csc_matrix((datar, indices, indptr))


def sparse_to_doc(sparse_mat):
    doc_mat = {}
    nonzeros = sparse_mat.nonzero()
    for i, x in enumerate(nonzeros[0]):
        doc_mat[x, nonzeros[1][i]] = 1
    return doc_mat


def order_pixel(pixel_begin, pixel_end, pixel_list):
    def get_neighbours(pixel):
        x = pixel[0]
        y = pixel[1]
        primary_neighbours = {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)}
        secondary_neighbours = {
            (x + 1, y - 1),
            (x + 1, y + 1),
            (x - 1, y + 1),
            (x - 1, y - 1),
        }
        num_neighbours = 0
        actual_neighbours = set()
        for neighbour in primary_neighbours:
            if neighbour in pixel_list:
                xp = neighbour[0]
                yp = neighbour[1]
                primary_neighboursp = {
                    (xp + 1, yp),
                    (xp - 1, yp),
                    (xp, yp + 1),
                    (xp, yp - 1),
                }
                for neighbourp in primary_neighboursp:
                    secondary_neighbours.discard(neighbourp)
                actual_neighbours.add(neighbour)
        for neighbour in secondary_neighbours:
            if neighbour in pixel_list:
                actual_neighbours.add(neighbour)
        return actual_neighbours

    ordered_list = [pixel_begin]
    current_pixel = pixel_begin
    precedent_pixel = pixel_begin
    while current_pixel != pixel_end:
        neighbours = get_neighbours(current_pixel)
        neighbours.discard(precedent_pixel)
        precedent_pixel = current_pixel
        current_pixel = neighbours.pop()
        ordered_list.append(current_pixel)
    return ordered_list


def extract_branches(doc_skel):
    def get_neighbours(pixel):
        x = pixel[0]
        y = pixel[1]
        primary_neighbours = {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)}
        secondary_neighbours = {
            (x + 1, y - 1),
            (x + 1, y + 1),
            (x - 1, y + 1),
            (x - 1, y - 1),
        }
        num_neighbours = 0
        actual_neighbours = []
        for neighbour in primary_neighbours:
            if neighbour in non_zero_pixel:
                num_neighbours += 1
                xp = neighbour[0]
                yp = neighbour[1]
                primary_neighboursp = {
                    (xp + 1, yp),
                    (xp - 1, yp),
                    (xp, yp + 1),
                    (xp, yp - 1),
                }
                for neighbourp in primary_neighboursp:
                    secondary_neighbours.discard(neighbourp)
                actual_neighbours.append(neighbour)
        for neighbour in secondary_neighbours:
            if neighbour in non_zero_pixel:
                num_neighbours += 1
                actual_neighbours.append(neighbour)
        return (actual_neighbours, num_neighbours)

    pixel_branch_dic = {pixel: set() for pixel in doc_skel.keys()}
    is_node = {pixel: False for pixel in doc_skel.keys()}
    pixel_set = set(doc_skel.keys())
    non_zero_pixel = doc_skel
    new_index = 1
    non_explored_direction = set()
    while len(pixel_set) > 0:
        is_new_start = len(non_explored_direction) == 0
        if is_new_start:
            pixel = pixel_set.pop()
        else:
            pixel = non_explored_direction.pop()
        actual_neighbours, num_neighbours = get_neighbours(pixel)
        if is_new_start:
            if num_neighbours == 2:
                new_index += 1
                pixel_branch_dic[pixel] = {new_index}
        is_node[pixel] = num_neighbours in [0, 1, 3, 4]
        pixel_set.discard(pixel)
        #!!! This is to solve the two neighbours nodes problem
        if is_node[pixel]:
            for neighbour in actual_neighbours:
                if is_node[neighbour]:
                    new_index += 1
                    pixel_branch_dic[pixel].add(new_index)
                    pixel_branch_dic[neighbour].add(new_index)
            continue
        else:
            for neighbour in actual_neighbours:
                if neighbour in pixel_set:
                    non_explored_direction.add(neighbour)
                pixel_branch_dic[neighbour] = pixel_branch_dic[neighbour].union(
                    pixel_branch_dic[pixel]
                )
    return (pixel_branch_dic, is_node, new_index)


def from_sparse_to_graph(doc_skel):
    column_names = ["origin", "end", "pixel_list"]
    graph = pd.DataFrame(columns=column_names)
    pixel_branch_dic, is_node, new_index = extract_branches(doc_skel)
    nodes = []
    edges = {}
    for pixel in pixel_branch_dic:
        for branch in pixel_branch_dic[pixel]:
            right_branch = branch
            if right_branch not in edges.keys():
                edges[right_branch] = {"origin": [], "end": [], "pixel_list": [[]]}
            if is_node[pixel]:
                if len(edges[right_branch]["origin"]) == 0:
                    edges[right_branch]["origin"] = [pixel]
                else:
                    edges[right_branch]["end"] = [pixel]
            edges[right_branch]["pixel_list"][0].append(pixel)
    for branch in edges:
        if len(edges[branch]["origin"]) > 0 and len(edges[branch]["end"]) > 0:
            # TODO(FK): Use pandas.concat instead (Frame.append soon deprecated)
            # graph = graph.append(pd.DataFrame(edges[branch]), ignore_index=True)
            graph = pandas.concat([graph, pd.DataFrame(edges[branch])])
    for index, row in graph.iterrows():
        row["pixel_list"] = order_pixel(row["origin"], row["end"], row["pixel_list"])
    return graph


def from_nx_to_tab(nx_graph, pos):
    column_names = [
        "origin_label",
        "end_label",
        "origin_pos",
        "end_pos",
        "pixel_list",
        "width",
    ]
    tab = pd.DataFrame(columns=column_names)
    for edge in nx_graph.edges:
        origin_label = edge[0]
        end_label = edge[1]
        origin_pos = pos[origin_label]
        end_pos = pos[end_label]
        pixel_list = orient(nx_graph.get_edge_data(*edge)["pixel_list"], origin_pos)
        width = nx_graph.get_edge_data(*edge)["width"]
        new_line = pd.DataFrame(
            {
                "origin_label": [origin_label],
                "end_label": [end_label],
                "origin_pos": [origin_pos],
                "end_pos": [end_pos],
                "pixel_list": [pixel_list],
                "width": [width],
            }
        )
        tab = tab.append(new_line, ignore_index=True)
    return tab


def from_nx_to_tab_matlab(nx_graph, pos):
    column_names = [
        "origin_label",
        "end_label",
        "origin_posx",
        "origin_posy",
        "end_posx",
        "end_posy",
        "pixel_list",
    ]
    tab = pd.DataFrame(columns=column_names)
    for edge in nx_graph.edges:
        origin_label = edge[0]
        end_label = edge[1]
        origin_posx = pos[origin_label][0]
        origin_posy = pos[origin_label][1]
        end_posx = pos[end_label][0]
        end_posy = pos[end_label][1]
        pixel_list = orient(
            nx_graph.get_edge_data(*edge)["pixel_list"], pos[origin_label]
        )
        new_line = pd.DataFrame(
            {
                "origin_label": [origin_label],
                "end_label": [end_label],
                "origin_posx": [origin_posx],
                "origin_posy": [origin_posy],
                "end_posx": [end_posx],
                "end_posy": [end_posy],
                "pixel_list": [
                    ",".join([f"{pixel[0]},{pixel[1]}" for pixel in pixel_list])
                ],
            }
        )
        tab = tab.append(new_line, ignore_index=True)
    return tab


def generate_set_node(graph_tab):
    nodes = set()
    for index, row in graph_tab.iterrows():
        nodes.add(row["origin"])
        nodes.add(row["end"])
    return sorted(nodes)


def generate_nx_graph(graph_tab, labeled=False):
    G = nx.Graph()
    pos = {}
    if not labeled:
        nodes = generate_set_node(graph_tab)
    for index, row in graph_tab.iterrows():
        if labeled:
            identifier1 = row["origin_label"]
            identifier2 = row["end_label"]
            pos[identifier1] = np.array(row["origin_pos"]).astype(np.int32)
            pos[identifier2] = np.array(row["end_pos"]).astype(np.int32)
        else:
            identifier1 = nodes.index(row["origin"])
            identifier2 = nodes.index(row["end"])
            pos[identifier1] = np.array(row["origin"]).astype(np.int32)
            pos[identifier2] = np.array(row["end"]).astype(np.int32)
        info = {"weight": len(row["pixel_list"]), "pixel_list": row["pixel_list"]}
        G.add_edges_from([(identifier1, identifier2, info)])
    return (G, pos)


def clean_degree_4(nx_graph, pos, thresh=30):
    nx_graph_clean = nx.Graph.copy(nx_graph)
    remaining_to_fuse = True
    print("cleaning, number of nodes before", len(nx_graph_clean.nodes))
    while remaining_to_fuse:
        remaining_to_fuse = False
        to_fuse = []
        for edge in nx_graph_clean.edges:
            if (
                nx_graph_clean.get_edge_data(*edge)["weight"] <= thresh
                and nx_graph_clean.degree(edge[0]) == 3
                and nx_graph_clean.degree(edge[1]) == 3
            ):
                to_fuse.append(edge)
        nodes_to_fuse = [edge[0] for edge in to_fuse] + [edge[1] for edge in to_fuse]
        occurence_count = Counter(nodes_to_fuse)
        difficult_cases = []
        for edge in to_fuse:
            node1 = edge[0]
            node2 = edge[1]
            if occurence_count[node1] != 1 and occurence_count[node2] != 1:
                difficult_cases.append(edge)
            else:
                remaining_to_fuse = True
                if occurence_count[node1] == 1:
                    pivot = node2
                    fuser = node1
                else:
                    pivot = node1
                    fuser = node2
                neighbours = list(nx_graph_clean.neighbors(fuser))
                for neighbour in neighbours:
                    right_n = pivot
                    left_n = neighbour
                    right_edge = nx_graph_clean.get_edge_data(fuser, right_n)[
                        "pixel_list"
                    ]
                    left_edge = nx_graph_clean.get_edge_data(fuser, left_n)[
                        "pixel_list"
                    ]
                    if np.any(right_edge[0] != pos[fuser]):
                        right_edge = list(reversed(right_edge))
                    if np.any(left_edge[-1] != pos[fuser]):
                        left_edge = list(reversed(left_edge))
                    pixel_list = left_edge + right_edge[1:]
                    info = {"weight": len(pixel_list), "pixel_list": pixel_list}
                    if right_n != left_n:
                        connection_data = nx_graph_clean.get_edge_data(right_n, left_n)
                        if (
                            connection_data is None
                            or connection_data["weight"] >= info["weight"]
                        ):
                            if not connection_data is None:
                                nx_graph_clean.remove_edge(right_n, left_n)
                            nx_graph_clean.add_edges_from([(right_n, left_n, info)])
                nx_graph_clean.remove_node(fuser)
        print("number of unsolved cases", len(difficult_cases))
    print("end cleaning, number of nodes after", len(nx_graph_clean.nodes))
    return (nx_graph_clean, difficult_cases)


# def generate_skeleton(nx_graph, dim=(3000, 4096)):
#     skel = sparse.dok_matrix(dim, dtype=bool)
#     for edge in nx_graph.edges.data("pixel_list"):
#         for pixel in edge[2]:
#             skel[pixel] = True
#     return skel


def prune_graph(nx_graph, threshold):
    # should implement threshold!
    S = [nx_graph.subgraph(c).copy() for c in nx.connected_components(nx_graph)]
    selected = [g for g in S if g.size(weight="weight") / 10**6 >= threshold]
    len_connected = [
        (nx_graph.size(weight="weight") / 10**6) for nx_graph in selected
    ]
    print(len_connected)
    G = selected[0]
    for g in selected[1:]:
        G = nx.compose(G, g)
    return G


def clean(skeleton):
    skeleton_doc = sparse.dok_matrix(skeleton)
    graph_tab = from_sparse_to_graph(skeleton_doc)
    nx_graph, pos = generate_nx_graph(graph_tab)
    nx_graph = prune_graph(nx_graph)
    return generate_skeleton(nx_graph).todense()


def generate_graph_tab_from_skeleton(skelet):
    dok_skel = sparse.dok_matrix(skelet)
    graph_tab = from_sparse_to_graph(dok_skel)
    return graph_tab


def generate_nx_graph_from_skeleton(skelet):
    #     dok_skel = sparse.dok_matrix(skelet)
    dok_skel = skelet
    graph_tab = from_sparse_to_graph(dok_skel)
    return generate_nx_graph(graph_tab)


def transform_list(position):
    c = position[1:-1].split()
    return np.array(c).astype(np.int)


def sub_graph(nx_graph, pos, xbegin, xend, ybegin, yend):
    sub_nx_graph = nx.Graph.copy(nx_graph)
    for node in nx_graph.nodes:
        if (
            pos[node][0] >= xend
            or pos[node][0] <= xbegin
            or pos[node][1] >= yend
            or pos[node][1] <= ybegin
        ):
            sub_nx_graph.remove_node(node)
    return sub_nx_graph


def generate_skeleton(nx_graph, dim=(30000, 60000), shift=(0, 0)) -> sparse.dok_matrix:
    """
    Generate a sparse binary image of the whole squeleton.
    Dimension should be the dimension of full stiched image.
    """
    skel = sparse.dok_matrix(dim, dtype=bool)
    for edge in nx_graph.edges.data("pixel_list"):
        for pixel in edge[2]:
            skel[(pixel[0] - shift[0], pixel[1] - shift[1])] = True
    return skel


def from_nx_to_simple_csv(nx_graph, pos, path, suffix):
    array = []
    for edge in nx_graph.edges:
        origin_label = edge[0]
        end_label = edge[1]
        origin_posx = pos[origin_label][0]
        origin_posy = pos[origin_label][1]
        end_posx = pos[end_label][0]
        end_posy = pos[end_label][1]
        pixel_list = orient(
            nx_graph.get_edge_data(*edge)["pixel_list"], pos[origin_label]
        )
        new_line = (
            [origin_label]
            + [end_label]
            + [int(origin_posx)]
            + [int(origin_posy)]
            + [int(end_posx)]
            + [int(end_posy)]
            + [int(coordinate) for pixel in pixel_list for coordinate in pixel]
        )
        array.append(new_line)
    #     print(type(array[0][0]),type(array[0][1]),type(array[0][2]),type(array[0][10]))
    #     array=np.asarray(array,dtype=np.int)
    #         print(type(new_line))
    with open(path + suffix, "w+") as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(array)


def connections_pixel_list_to_tab(origin_tips, pattern_growth):
    column_names = ["tip_origin", "nodes_from_tip", "growth_pattern"]
    tab = pd.DataFrame(columns=column_names)
    pattern_growth = {
        tip: [
            [(pixel[0], pixel[1]) for pixel in branch] for branch in pattern_growth[tip]
        ]
        for tip in pattern_growth.keys()
    }
    for tip in origin_tips.keys():
        new_line = pd.DataFrame(
            {
                "tip_origin": [tip],
                "nodes_from_tip": [origin_tips[tip]],
                "growth_pattern": [pattern_growth[tip]],
            }
        )
        tab = tab.append(new_line, ignore_index=True)
    return tab


def from_connection_tab(connect_tab):
    from_tip = {}
    growth_pattern = {}
    for i in range(len(connect_tab["tip_origin"])):
        tip = connect_tab["tip_origin"][i]
        growth_pattern[tip] = ast.literal_eval(connect_tab["growth_pattern"][i])
        from_tip[tip] = ast.literal_eval(connect_tab["nodes_from_tip"][i])
    return (from_tip, growth_pattern)


def get_neighbours(pixel, non_zero_pixel):
    x = pixel[0]
    y = pixel[1]
    primary_neighbours = {(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)}
    secondary_neighbours = {
        (x + 1, y - 1),
        (x + 1, y + 1),
        (x - 1, y + 1),
        (x - 1, y - 1),
    }
    num_neighbours = 0
    actual_neighbours = []
    for neighbour in primary_neighbours:
        if neighbour in non_zero_pixel:
            num_neighbours += 1
            xp = neighbour[0]
            yp = neighbour[1]
            primary_neighboursp = {
                (xp + 1, yp),
                (xp - 1, yp),
                (xp, yp + 1),
                (xp, yp - 1),
            }
            for neighbourp in primary_neighboursp:
                secondary_neighbours.discard(neighbourp)
            actual_neighbours.append(neighbour)
    for neighbour in secondary_neighbours:
        if neighbour in non_zero_pixel:
            num_neighbours += 1
            actual_neighbours.append(neighbour)
    return (actual_neighbours, num_neighbours)


def get_degree3_nodes(skel):
    deg_3 = []
    non_zero = skel.keys()
    for pixel in non_zero:
        n, num = get_neighbours(pixel, non_zero)
        if num == 3:
            deg_3.append(pixel)
    return deg_3
