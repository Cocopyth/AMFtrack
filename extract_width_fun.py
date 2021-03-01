from util import get_path
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from extract_graph import generate_nx_graph, transform_list, generate_skeleton, generate_nx_graph_from_skeleton, from_connection_tab
from node_id import reconnect_degree_2
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
from experiment_class_surf import Experiment,clean_exp_with_hyphaes
from hyphae_id import clean_and_relabel, get_mother, save_hyphaes, resolve_ambiguity_two_ends, solve_degree4, clean_obvious_fake_tips
from extract_graph import prune_graph
from skimage.measure import profile_line
import math
from Analysis.util import *
from directory import directory, path_code
from Analysis.data_info import *

def get_source_image(experiment,pos,t,local,force_selection = None):
    x,y=pos[0],pos[1]
    ims,posimg=experiment.find_image_pos(x,y,t,local)
    if force_selection is None:
        dist_border=[min([posimg[1][i],3000-posimg[1][i],posimg[0][i],4096-posimg[0][i]]) for i in range(posimg[0].shape[0])]
        j=np.argmax(dist_border)
    else:
        dist_last=[np.linalg.norm(np.array((posimg[1][i],posimg[0][i])) - np.array(force_selection)) for i in range(posimg[0].shape[0])]
        j=np.argmin(dist_last)
    return(ims[j],(posimg[1][j],posimg[0][j]))

def get_width_pixel(edge,index,im,pivot,before,after,t,size = 20,width_factor = 60,averaging_size = 100,threshold_averaging = 10):
    imtab=im
#     print(imtab.shape)
#     print(int(max(0,pivot[0]-averaging_size)),int(pivot[0]+averaging_size))
    threshold = np.mean(imtab[int(max(0,pivot[0]-averaging_size)):int(pivot[0]+averaging_size),int(max(0,pivot[1]-averaging_size)):int(pivot[1]+averaging_size)]-threshold_averaging)
    orientation=np.array(before)-np.array(after)
    perpendicular = [1,-orientation[0]/orientation[1]] if orientation[1]!=0 else [0,1]
    perpendicular_norm=np.array(perpendicular)/np.sqrt(perpendicular[0]**2+perpendicular[1]**2)
    point1=np.around(np.array(pivot)+width_factor*perpendicular_norm)
    point2=np.around(np.array(pivot)-width_factor*perpendicular_norm)
    point1=point1.astype(int)
    point2=point2.astype(int)
    p = profile_line(imtab, point1, point2,mode='constant')
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot(p)
#     derivative = [p[i+1]-p[i] for i in range(len(p)-1)]
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.plot([np.mean(derivative[5*i:5*i+5]) for i in range(len(derivative)//5)])
    problem=False
    arg = len(p)//2
    if p[arg]>threshold:
        arg = np.argmin(p)
#     we_plot=randrange(1000)
    while  p[arg]<=threshold:
        if arg<=0:
#             we_plot=50
            problem=True
            break
        arg-=1
    begin = arg
    arg = len(p)//2
    if p[arg]>threshold:
        arg = np.argmin(p)
    while  p[arg]<=threshold:
        if arg>=len(p)-1:
#             we_plot=50
            problem=True
            break
        arg+=1
    end = arg
#     print(end-begin,threshold)
    return(np.linalg.norm(point1-point2)*(end-begin)/len(p))

def get_width_edge(edge,resolution,t,local=False, threshold_averaging = 10):
    pixel_conversion_factor = 1.725
    pixel_list = edge.pixel_list(t)
    pixels = []
    indexes = []
    source_images = []
    poss = []
    widths={}
    if len(pixel_list)>3*resolution:
        for i in range(0,len(pixel_list)//resolution):
            index = i*resolution
            indexes.append(index)
            pixel = pixel_list[index]
            pixels.append(pixel)
            source_img,pos = get_source_image(edge.experiment,pixel,t,local)
            source_images.append(source_img)
            poss.append(pos)
    else:
        indexes = [0,len(pixel_list)//2,len(pixel_list)-1]
        for index in indexes:
            pixel = pixel_list[index]
            pixels.append(pixel)
            source_img,pos = get_source_image(edge.experiment,pixel,t,local)
            source_images.append(source_img)
            poss.append(pos)
#     print(indexes)
    for i, index in enumerate(indexes[1:-1]):
        source_img = source_images[i+1]
        pivot = poss[i+1]
        _,before = get_source_image(edge.experiment,pixels[i],t,local,pivot)
        _,after = get_source_image(edge.experiment,pixels[i+2],t,local,pivot)
#         plot_t_tp1([0,1,2],[],{0 : pivot,1 : before, 2 : after},None,source_img,source_img)
        width = get_width_pixel(edge,index,source_img,pivot,before,after,t,threshold_averaging = threshold_averaging)
#         print(width*pixel_conversion_factor)
        widths[pixel_list[index]]=width*pixel_conversion_factor
    edge.experiment.nx_graph[t].get_edge_data(edge.begin.label,edge.end.label)['width'] = widths
    return(widths)      

def get_width_info(experiment,t,resolution = 50):
    edge_width={}
    graph = experiment.nx_graph[t]
#     print(len(list(graph.edges)))
    for edge in graph.edges:
#         print(edge)
        edge_exp = Edge(Node(edge[0],experiment),Node(edge[1],experiment),experiment)
        mean = np.mean(list(get_width_edge(edge_exp,resolution,t).values()))
#         print(np.mean(list(get_width_edge(edge_exp,resolution,t).values())))
        edge_width[edge]=mean
    return(edge_width)