from shapely.geometry import Polygon, shape,Point
from scipy import spatial
import networkx as nx
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Edge, Node
from amftrack.pipeline.functions.post_processing.util import get_length_um_edge, is_in_study_zone
import numpy as np

def get_hulls(exp,ts):
    hulls = []
    for t in ts:
        nx_graph = exp.nx_graph[t]
        threshold = 0.1
        S = [nx_graph.subgraph(c).copy() for c in nx.connected_components(nx_graph)]
        selected = [
            g for g in S if g.size(weight="weight") * len(g.nodes) / 10 ** 6 >= threshold
        ]
        if len(selected)>=0:
            area_max = 0
            for g in selected:
                nodes = np.array([node.pos(t) for node in exp.nodes if node.is_in(t) and np.all(is_in_study_zone(node,t,1000,150)) and (node.label in g.nodes)])
                if len(nodes)>3:
                    hull=spatial.ConvexHull(nodes)
                    poly = Polygon([nodes[vertice] for vertice in hull.vertices])
                    area_hull = poly.area* 1.725**2/(1000**2)
                    if area_hull>=area_max:
                        area_max=area_hull
                        select_poly  = poly
        else:
            select_poly = Polygon()
        hulls.append(select_poly)
    return(hulls)

def ring_area(hull1,hull2):
    return(hull2.area* 1.725**2/(1000**2)-hull1.area* 1.725**2/(1000**2))

def get_nodes_in_ring(hull1,hull2,t,exp):
    nodes = [node for node in exp.nodes if node.is_in(t) and hull2.contains(Point(node.pos(t))) and not hull1.contains(Point(node.pos(t))) and np.all(is_in_study_zone(node,t,1000,200))]
    return(nodes)

def get_length_in_ring(hull1,hull2,t,exp):
    nodes = get_nodes_in_ring(hull1,hull2,t,exp)
    edges = {edge for node in nodes for edge in node.edges(t)}
    tot_length = np.sum([np.linalg.norm(edge.end.pos(t)-edge.begin.pos(t))*1.725 for edge in edges])
    return(tot_length)

def get_regular_hulls(num,exp,ts):
    hulls = get_hulls(exp,ts)
    areas = [hull.area* 1.725**2/(1000**2) for hull in hulls]
    area_incr = areas[-1]-areas[0]
    length_incr = np.sqrt(area_incr)
    incr = length_incr /num
    regular_hulls = [hulls[0]]
    init_area = areas[0]
    indexes = [0]
    current_length = incr
    for i in range(num-1):
        current_area = init_area + current_length**2
        index = min([i for i in range(len(areas)) if areas[i]>=current_area])
        indexes.append(index)
        current_length += incr
        regular_hulls.append(hulls[index])
    return(regular_hulls,indexes)

def get_regular_hulls_area_ratio(num,exp,ts):
    hulls = get_hulls(exp,ts)
    areas = [hull.area* 1.725**2/(1000**2) for hull in hulls]
    area_incr = areas[-1]-areas[0]
    incr = area_incr /num
    regular_hulls = [hulls[0]]
    init_area = areas[0]
    indexes = [0]
    current_length = incr
    current_area = init_area
    for i in range(num-1):
        current_area += incr
        index = min([i for i in range(len(areas)) if areas[i]>=current_area])
        indexes.append(index)
        regular_hulls.append(hulls[index])
    return(regular_hulls,indexes)

def get_regular_hulls_area_fixed(exp,ts,incr,num):
    hulls = get_hulls(exp,ts)
    areas = [hull.area* 1.725**2/(1000**2) for hull in hulls]
    regular_hulls = [hulls[0]]
    init_area = areas[0]
    indexes = [0]
    current_area = init_area
    for i in range(num-1):
        current_area += incr
        index = min([i for i in range(len(areas)) if areas[i]>=current_area])
        indexes.append(index)
        regular_hulls.append(hulls[index])
    return(regular_hulls,indexes)


def get_density_in_ring(exp,t,args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls,indexes = get_regular_hulls_area_fixed(exp,range(exp.ts),incr,i+2)
    hull1,hull2 = regular_hulls[i],regular_hulls[i+1]
    length = get_length_in_ring(hull1,hull2,t,exp)
    area= ring_area(hull1,hull2)
    return(f"ring_density_incr-{incr}_index-{i}",length/area)