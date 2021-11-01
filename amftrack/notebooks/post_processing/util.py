import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
import numpy as np
from amftrack.util import get_dates_datetime
import pickle
from amftrack.pipeline.functions.experiment_class_surf import Edge,Node
from random import choice
import networkx as nx
from amftrack.pipeline.paths.directory import directory_scratch, path_code
from scipy import sparse
from amftrack.pipeline.functions.hyphae_id_surf import get_pixel_growth_and_new_children
from shapely.geometry import Polygon, shape
from scipy import spatial
from amftrack.pipeline.functions.hyphae_id_surf import get_pixel_growth_and_new_children

def get_area(exp,t,args):
    nodes = np.array([node.pos(t) for node in exp.nodes if node.is_in(t)])
    hull=spatial.ConvexHull(nodes)
    poly = Polygon([nodes[vertice] for vertice in hull.vertices])
    area = poly.area* 1.725**2/(1000**2)
    return('area',area)
def get_num_tips(exp,t,args):
    return('num_tips',len([node for node in exp.nodes if node.is_in(t) and node.degree(t)==1]))
def get_num_nodes(exp,t,args):
    return('num_tips',len([node for node in exp.nodes if node.is_in(t)]))
def get_length(exp,t,args):
    length=0
    for edge in exp.nx_graph[t].edges:
        edge_obj= Edge(Node(edge[0],exp),Node(edge[1],exp),exp)
        length+=get_length_um_edge(edge_obj,t)
    return('tot_length',length)
def get_length_um(seg):
    pixel_conversion_factor = 1.725
    pixels = seg
    length_edge = 0
    for i in range(len(pixels) // 10 + 1):
        if i * 10 <= len(pixels) - 1:
            length_edge += np.linalg.norm(
                np.array(pixels[i * 10])
                - np.array(pixels[min((i + 1) * 10, len(pixels) - 1)])
            )
    #         length_edge+=np.linalg.norm(np.array(pixels[len(pixels)//10-1*10-1])-np.array(pixels[-1]))
    return length_edge * pixel_conversion_factor

def get_length_um_edge(edge,t):
    pixel_conversion_factor = 1.725
    length_edge = 0
    pixels = edge.pixel_list(t)
    for i in range(len(pixels) // 10 + 1):
        if i * 10 <= len(pixels) - 1:
            length_edge += np.linalg.norm(
                np.array(pixels[i * 10])
                - np.array(pixels[min((i + 1) * 10, len(pixels) - 1)])
            )
#             length_edge+=np.linalg.norm(np.array(pixels[len(pixels)//10-1*10-1])-np.array(pixels[-1]))
    return length_edge * pixel_conversion_factor

def get_length_um_node_list(node_list,exp,t):
    total_length = 0
    for i in range(len(node_list)-1):
        nodea=Node(node_list[i],exp)
        nodeb=Node(node_list[i+1],exp)
        edge_obj =Edge(nodea,nodeb,exp)
        total_length += get_length_um_edge(edge_obj,t)
    return(total_length)

def get_time(hypha,t,tp1,args):
    seconds = (hypha.experiment.dates[tp1]-exp.dates[t]).total_seconds()
    return("time",seconds/3600)
def get_speed(hypha,t,tp1,args):
    try:
        pixels,nodes = get_pixel_growth_and_new_children(hypha,t,tp1)
        speed = np.sum([get_length_um(seg) for seg in pixels])/get_time(hypha,t,tp1,None)[1]
        return('speed',speed)
    except:
        print('not_connected',hypha.end.label,hypha.get_root(tp1).label)
        return('speed',None)