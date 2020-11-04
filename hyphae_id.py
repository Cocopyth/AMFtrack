from util import get_path
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from extract_graph import generate_nx_graph, transform_list, generate_skeleton, generate_nx_graph_from_skeleton, from_connection_tab
from node_id import whole_movement_identification, second_identification
import ast
from plotutil import plot_t_tp1, compress_skeleton
from scipy import sparse
from sparse_util import dilate, zhangSuen
from realign import realign
from datetime import datetime,timedelta
from node_id import orient
import pickle
from matplotlib.widgets import CheckButtons
import scipy.io as sio
import imageio
from pymatreader import read_mat
import os
from matplotlib import colors
from random import choice
from experiment_class import Experiment,clean_exp_with_hyphaes, Node

def resolve_ambiguity(hyphaes):
#     problems=[]
#     safe=[]
#     for hyph in hyphaes:
#         if len(hyph.root.ts())<len(hyph.ts):
#             problems.append(hyph)
#         else:
#             safe.append(hyph)
    safe=hyphaes
    ambiguities=[]
    connection={hyph : [] for hyph in safe}
    for hyph in safe:
        root = hyph.root
        for hyph2 in safe:
            if hyph2.root == root and hyph2.end != hyph.end and (hyph2,hyph) not in ambiguities:
                ambiguities.append((hyph,hyph2))
#         t0=hyph.ts[0]
#         nodes = hyph.get_nodes_within(t0)
#         nodes_within_initial[hyph.end]=nodes
#     for hyph in safe:
#         nodes = nodes_within_initial[hyph.end]
#         root,first = nodes[0],nodes[1]
#         for hyph2 in safe:
#             if hyph2.end != hyph.end:
#                 nodes2 = nodes_within_initial[hyph2.end]
#                 if root in nodes2 and first in nodes2:
#                     ambiguities.append(hyph,hyph2)
    for ambig in ambiguities:
        common_ts = sorted(set(ambig[0].ts).intersection(set(ambig[1].ts)))
        if len(common_ts)>=1:
            continue
        else:
            hyph1 = ambig[0]
            hyph2 = ambig[1]
            if hyph1.ts[-1]<=hyph2.ts[0]:
                t1 = hyph1.ts[-1]
                t2 = hyph2.ts[0]
            else:
                t1 = hyph1.ts[0]
                t2 = hyph2.ts[-1]
            if np.linalg.norm(hyph1.end.pos(t1)-hyph2.end.pos(t2))<=300:
                connection[hyph1].append(hyph2)
    equ_classes = []
    put_in_class=set()
    for hyph in connection.keys():
        if not hyph in put_in_class:
            equ={hyph}
            full_equ_class = False
            while not full_equ_class:
                full_equ_class = True
                for hypha in list(equ):
                    for hyph2 in connection[hypha]:
                        if hyph2 not in equ:
                            equ.add(hyph2)
                            full_equ_class = False
            if not np.any([hyphaa in put_in_class for hyphaa in equ]):
                for hyphaa in equ:
                    put_in_class.add(hyphaa)
                equ_classes.append(equ)
    connect={}
    for hyph in safe:
        found = False
        for equ in equ_classes:
            if hyph in equ:
                found = True
                connect[hyph.end.label]=np.min([hyphaa.end.label for hyphaa in equ])
        if not found:
            connect[hyph.end.label]=hyph.end.label
    return(equ_classes,ambiguities,connect)

def relabel_nodes_after_amb(corresp,nx_graph_list,pos_list):
    new_poss = [{} for i in range(len(nx_graph_list))]
    new_graphs=[]
    all_nodes = set()
    for nx_graph in nx_graph_list:
        all_nodes=all_nodes.union(set(nx_graph.nodes))
    all_nodes=all_nodes.union(set(corresp.keys()))
    all_nodes=all_nodes.union(set(corresp.values()))
    maxi=max(all_nodes)+1
    def mapping(node):
        if node in corresp.keys():
            return(int(corresp[node]))
        else:
            return (node)
    for i,nx_graph in enumerate(nx_graph_list):
        for node in nx_graph.nodes:
            pos=pos_list[i][node]
            new_poss[i][mapping(node)]=pos
        new_graphs.append(nx.relabel_nodes(nx_graph,mapping,copy=True))
    return(new_graphs,new_poss)

def clean_and_relabel(exp):
    exp_clean= clean_exp_with_hyphaes(exp)
    equ_class,ambig,connection=resolve_ambiguity(exp_clean.hyphaes)
    new_graph,newposs = relabel_nodes_after_amb(connection,exp_clean.nx_graph,exp_clean.positions)
    exp_clean.nx_graph = new_graph
    exp_clean.positions = newposs
    exp_clean.nodes = []
    labels = {int(node) for g in exp_clean.nx_graph for node in g}
    for label in labels:
        exp_clean.nodes.append(Node(label,exp_clean))
    exp_clean_relabeled= clean_exp_with_hyphaes(exp_clean)
    return(exp_clean_relabeled)

def get_mother(hyphaes):
    nodes_within={hyphae.end : {} for hyphae in hyphaes}
    for  i,hyphae in enumerate(hyphaes):
        if i%500==0:
            print(i/len(hyphaes))
        mothers=[]
        t0 = hyphae.ts[0]
        for hyph in hyphaes:
            if t0 in hyph.ts and hyph.end != hyphae.end:
                if t0 in nodes_within[hyph.end].keys():
                    nodes_within_hyph = nodes_within[hyph.end][t0]
                else:
                    nodes_within_hyph = hyph.get_nodes_within(t0)[0]
                    nodes_within[hyph.end][t0]= nodes_within_hyph
                if hyphae.root.label in nodes_within_hyph:
                    mothers.append(hyph)
        hyphae.mother = mothers
        
def get_pixel_growth_and_new_children(hyphae,t1,t2):
    assert t1<t2, "t1 should be strictly inferior to t2"
    edges = hyphae.get_nodes_within(t2)[1]
    mini = np.inf
    if t1 not in hyphae.ts:
        pixels = []
        nodes = [hyphae.root]
        for edge in edges:
            pixels.append(edge.pixel_list(t2))
            nodes.append(edge.end)
        return(pixels,nodes)
    else : 
        if len(edges)==0:
            print(hyphae.root,hyphae.end)
            return([],[])
        for i,edge in enumerate(edges):
            distance = np.min(np.linalg.norm(hyphae.end.pos(t1)-np.array(edge.pixel_list(t2)),axis=1))
            if distance < mini :
                index = i
                mini = distance
                last_edge = edge
                index_nearest_pixel=np.argmin(np.linalg.norm(hyphae.end.pos(t1)-np.array(edge.pixel_list(t2)),axis=1))
        pixels = [last_edge.pixel_list(t2)[index_nearest_pixel:]]
        nodes=[-1,last_edge.end]
        for edge in edges[index+1:]:
            pixels.append(edge.pixel_list(t2))
            nodes.append(edge.end)
        return(pixels,nodes)
    
    
    
def save_hyphaes(exp,path = 'Data/'):
    column_names_hyphaes = ['end','root', 'ts','mother']
    column_names_growth_info = ['hyphae','t','t+1','nodes_in_hyphae', 'segment_of_growth_t_t+1','node_list_t_t+1']
    hyphaes = pd.DataFrame(columns=column_names_hyphaes)
    growth_info = pd.DataFrame(columns = column_names_growth_info)
    for hyph in exp.hyphaes:
        new_line_hyphae=pd.DataFrame({'end' : [hyph.end.label],'root' :[hyph.root.label], 'ts': [hyph.ts],'mother' : [-1 if len(hyph.mother)==0 else hyph.mother[0].end.label]}) #index 0 for 
        #mothers need to be modified to resolve multi mother issue
        hyphaes=hyphaes.append(new_line_hyphae,ignore_index=True)
        for index in range(len(hyph.ts[:-1])):
            t = hyph.ts[index]
            tp1 = hyph.ts[index+1]
            pixels,nodes = get_pixel_growth_and_new_children(hyph,t,tp1)
            new_line_growth=pd.DataFrame({'hyphae' : [hyph.end],'t' : [t],'t+1' : [tp1],'nodes_in_hyphae' :[hyph.get_nodes_within(t)], 'segment_of_growth_t_t+1' : [pixels],'node_list_t_t+1':[nodes]})
            growth_info.append(new_line_growth,ignore_index = True)
    path='Data/'
    hyphaes.to_csv(path + f'hyphaes_{exp.plate}_{exp.dates[0]}_{exp.dates[-1]}.csv')
    growth_info.to_csv(path + f'growth_info_{exp.plate}_{exp.dates[0]}_{exp.dates[-1]}.csv')
    sio.savemat(path+f'hyphaes_{exp.plate}_{exp.dates[0]}_{exp.dates[-1]}.mat', {name: col.values for name, col in hyphaes.items()})
    sio.savemat(path+f'growth_info_{exp.plate}_{exp.dates[0]}_{exp.dates[-1]}.mat', {name: col.values for name, col in growth_info.items()})
    return(hyphaes,growth_info)