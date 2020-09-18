import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import numpy as np
from scipy import sparse
from pymatreader import read_mat
import networkx as nx
import pandas as pd
def dic_to_sparse(dico):
    indptr=dico['jc']
    indices=dico['ir']
    datar=dico['data']
    return(sparse.csc_matrix((datar,indices,indptr))) 

def order_pixel(pixel_begin,pixel_end,pixel_list):
    def get_neighbours(pixel):
        x=pixel[0]
        y=pixel[1]
        primary_neighbours = {(x+1,y),(x-1,y),(x,y+1),(x,y-1)}
        secondary_neighbours = {(x+1,y-1),(x+1,y+1),(x-1,y+1),(x-1,y-1)}
        num_neighbours = 0
        actual_neighbours = set()
        for neighbour in primary_neighbours:
            if neighbour in pixel_list:
                xp=neighbour[0]
                yp=neighbour[1]
                primary_neighboursp = {(xp+1,yp),(xp-1,yp),(xp,yp+1),(xp,yp-1)}
                for neighbourp in primary_neighboursp:
                    secondary_neighbours.discard(neighbourp)
                actual_neighbours.add(neighbour)
        for neighbour in secondary_neighbours:
            if neighbour in pixel_list:
                actual_neighbours.add(neighbour)
        return(actual_neighbours)
    ordered_list=[pixel_begin]
    current_pixel=pixel_begin
    precedent_pixel=pixel_begin
    while current_pixel!=pixel_end:
        neighbours=get_neighbours(current_pixel)
        neighbours.discard(precedent_pixel)
        precedent_pixel = current_pixel
        current_pixel = neighbours.pop()
        ordered_list.append(current_pixel)
    return(ordered_list)

def extract_branches(doc_skel):
    def get_neighbours(pixel):
        x=pixel[0]
        y=pixel[1]
        primary_neighbours = {(x+1,y),(x-1,y),(x,y+1),(x,y-1)}
        secondary_neighbours = {(x+1,y-1),(x+1,y+1),(x-1,y+1),(x-1,y-1)}
        num_neighbours = 0
        actual_neighbours = []
        for neighbour in primary_neighbours:
            if neighbour in non_zero_pixel:
                num_neighbours +=1
                xp=neighbour[0]
                yp=neighbour[1]
                primary_neighboursp = {(xp+1,yp),(xp-1,yp),(xp,yp+1),(xp,yp-1)}
                for neighbourp in primary_neighboursp:
                    secondary_neighbours.discard(neighbourp)
                actual_neighbours.append(neighbour)
        for neighbour in secondary_neighbours:
            if neighbour in non_zero_pixel:
                num_neighbours +=1
                actual_neighbours.append(neighbour)
        return(actual_neighbours,num_neighbours)
    pixel_branch_dic ={pixel : set() for pixel in doc_skel.keys()}
    is_node = {pixel : False for pixel in doc_skel.keys()}
    pixel_set=set(doc_skel.keys())
    non_zero_pixel = doc_skel
    new_index=1
    non_explored_direction=set()
    while len(pixel_set)>0:
        is_new_start=len(non_explored_direction)==0
        if is_new_start:
            pixel=pixel_set.pop()
        else:
            pixel = non_explored_direction.pop()
        actual_neighbours, num_neighbours = get_neighbours(pixel)
        if is_new_start:
            if num_neighbours ==2:
                new_index+=1
                pixel_branch_dic[pixel]={new_index}
        is_node[pixel]=num_neighbours in [0,1,3,4]
        pixel_set.discard(pixel)
        #!!! This is to solve the two neighbours nodes problem
        if is_node[pixel]:
            for neighbour in actual_neighbours:
                if is_node[neighbour]:
                    new_index+=1
                    pixel_branch_dic[pixel].add(new_index)
                    pixel_branch_dic[neighbour].add(new_index)
            continue
        else:
            for neighbour in actual_neighbours:
                if neighbour in pixel_set:
                    non_explored_direction.add(neighbour)
                pixel_branch_dic[neighbour]=pixel_branch_dic[neighbour].union(pixel_branch_dic[pixel])
    return(pixel_branch_dic,is_node,new_index)  

def from_sparse_to_graph(doc_skel):
    column_names = ["origin", "end", "pixel_list"]
    graph = pd.DataFrame(columns=column_names)
    pixel_branch_dic,is_node,new_index = extract_branches(doc_skel)
    nodes=[]
    edges={}
    for pixel in pixel_branch_dic:
        for branch in pixel_branch_dic[pixel]:
            right_branch=branch
            if right_branch not in edges.keys():
                edges[right_branch]={"origin":[], "end":[], "pixel_list":[[]]}
            if is_node[pixel]:
                if len(edges[right_branch]['origin'])==0:
                    edges[right_branch]['origin']=[pixel]
                else:
                    edges[right_branch]['end']=[pixel]
            edges[right_branch]['pixel_list'][0].append(pixel)
    for branch in edges:
        if len(edges[branch]['origin'])>0 and len(edges[branch]['end'])>0:
            graph=graph.append(pd.DataFrame(edges[branch]),ignore_index=True)
    for index, row in graph.iterrows():
        row['pixel_list']=order_pixel(row['origin'],row['end'],row['pixel_list'])
    return(graph)

def generate_set_node(graph_tab):
    nodes=set()
    for index, row in graph_tab.iterrows():
        nodes.add(row['origin'])
        nodes.add(row['end'])
    return(sorted(nodes))

def generate_nx_graph(graph_tab):
    G = nx.Graph()
    pos={}
    nodes=generate_set_node(graph_tab)
    for index, row in graph_tab.iterrows():
        identifier1=nodes.index(row['origin'])
        identifier2=nodes.index(row['end'])
        pos[identifier1]=np.array(row['origin'])
        pos[identifier2]=np.array(row['end'])
        info={'weight':len(row['pixel_list']),'pixel_list':row['pixel_list']}
        G.add_edges_from([(identifier1,identifier2,info)])
    return(G,pos)

def generate_skeleton(nx_graph,dim=(3000,4096)):
    skel = sparse.dok_matrix(dim, dtype=bool)
    for edge in nx_graph.edges.data('pixel_list'):
        for pixel in edge[2]:
            skel[pixel]=True
    return(skel)

def prune_graph(nx_graph,threshold=150):
    S = [nx_graph.subgraph(c).copy() for c in nx.connected_components(nx_graph)]
    for s in S:
        if s.size(weight='weight')<threshold:
            nx_graph.remove_nodes_from(s.nodes)
    to_remove=[]
    for edge in nx_graph.edges:
        if nx_graph.degree(edge[0])==1 and nx_graph.degree(edge[1])==1:
            to_remove.append(edge[0])
            to_remove.append(edge[1])
    nx_graph.remove_nodes_from(to_remove)
    return(nx_graph)

def clean(skeleton):
    skeleton_doc=sparse.dok_matrix(skeleton)
    graph_tab=from_sparse_to_graph(skeleton_doc)
    nx_graph,pos=generate_nx_graph(graph_tab)
    nx_graph=prune_graph(nx_graph)
    return(generate_skeleton(nx_graph).todense())

def generate_graph_tab_from_skeleton(skelet):
    dok_skel = sparse.dok_matrix(skelet)
    graph_tab = from_sparse_to_graph(dok_skel)
    return(graph_tab)


def generate_nx_graph_from_skeleton(skelet):
    dok_skel = sparse.dok_matrix(skelet)
    graph_tab = from_sparse_to_graph(dok_skel)
    return(generate_nx_graph(graph_tab))
