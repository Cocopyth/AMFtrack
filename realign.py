import pandas as pd
import networkx as nx
import numpy as np
from extract_graph import generate_nx_graph, transform_list, generate_skeleton, generate_nx_graph_from_skeleton
from node_id import whole_movement_identification
import ast
from plotutil import plot_t_tp1
from scipy import sparse
from sparse_util import dilate, zhangSuen
from sparse_util import dilate
from scipy.optimize import minimize

def find_common_group_nodes(nx_grapha,nx_graphb,posa,posb,window=500,maxdist=50):
    degree3_nodesa = [node for node in nx_grapha if nx_grapha.degree(node)==3]
    degree3_nodesb = [node for node in nx_graphb if nx_graphb.degree(node)==3]
    common_nodes_a = []
    common_nodes_b = []
    common_centroida = []
    common_centroidb = []
    for node in degree3_nodesa:
        posanchor=posa[node]
        surrounding_nodesa=[node for node in nx_grapha.nodes if 
                            (posanchor[0]-window<posa[node][0]<posanchor[0]+window and posanchor[1]-window<posa[node][1]<posanchor[1]+window 
                             and nx_grapha.degree(node)>=3)]
        surrounding_nodesb=[node for node in nx_graphb.nodes if 
                    (posanchor[0]-window<posb[node][0]<posanchor[0]+window and posanchor[1]-window<posb[node][1]<posanchor[1]+window 
                     and nx_graphb.degree(node)>=3)]
        if len(surrounding_nodesa)==len(surrounding_nodesb):
            possurroundinga=[posa[node] for node in surrounding_nodesa]
            possurroundingb=[posb[node] for node in surrounding_nodesb]
            centroida= np.mean(possurroundinga,axis=0)
            centroidb= np.mean(possurroundingb,axis=0)
            if np.linalg.norm(centroida-centroidb)<=maxdist:
                common_centroida.append(centroida)
                common_centroidb.append(centroidb)
    return(common_centroida,common_centroidb)

def transform_skeleton(skeleton_doc,Rot,trans):
    transformed_skeleton=sparse.dok_matrix(skeleton_doc.shape, dtype=bool)
    transformed_keys = np.round(np.transpose(np.dot(Rot,np.transpose(np.array(list(skeleton_doc.keys())))))+trans).astype(np.int)
    maxx=skeleton_doc.shape[0]
    maxy = skeleton_doc.shape[1]
    i=0
    for pixel in list(transformed_keys):
        i+=1
        if maxx>pixel[0]>0 and maxy>pixel[1]>0:
            transformed_skeleton[(pixel[0],pixel[1])]=1
        if i%100000==0:
            print(i)
    return(transformed_skeleton)

def realign(skeleton1,skeleton2):
    nx_graphA,posA=generate_nx_graph_from_skeleton(skeleton1) 
    nx_graphB,posB=generate_nx_graph_from_skeleton(skeleton2) 
    listeA,listeB = find_common_group_nodes(nx_graphA,nx_graphB,posA,posB,maxdist=30)
    H=np.dot(np.transpose(np.array(listeA)-np.mean(listeA,axis=0)),np.array(listeB)-np.mean(listeB,axis=0))
    U,S,V=np.linalg.svd(H)
    R=np.dot(V,np.transpose(U))
    t=np.mean(listeB,axis=0)-np.dot(R,np.mean(listeA,axis=0))
    skeleton_transformed=transform_skeleton(skeleton1,R,t)
    skeleton_transformed=dilate(skeleton_transformed)
    skeleton_transformed=zhangSuen(skeleton_transformed)
    return(skeleton_transformed)


def shift(skeleton1,skeleton2):
    skeleton1_dilated = dilate(dilate(skeleton1)).astype(np.float)
    skeleton2_dilated = dilate(dilate(skeleton2)).astype(np.float)
    def distance(shift):
        distance=0
#         print(shift)
        for pixel in skeleton1_dilated.keys():
#             print(pixel[0]+shift[0],pixel[1]+shift[1])
            if (skeleton2_dilated.shape[0]>np.ceil(pixel[0]+shift[0])>=0 and skeleton2_dilated.shape[1]>np.ceil(pixel[1]+shift[1])>=0):
                shifted_pixel = (int(pixel[0]+shift[0]),int(pixel[1]+shift[1]))
                shifted_pixel_next = (np.ceil(pixel[0]+shift[0]),np.ceil(pixel[1]+shift[1]))
#                 print(shifted_pixel)
                prop=1/2*(pixel[0]+shift[0]-int(pixel[0]+shift[0])+pixel[1]+shift[1]-int(pixel[1]+shift[1]))
                float_value=(1-prop)*skeleton2_dilated[shifted_pixel[0],shifted_pixel[1]]+prop*(skeleton2_dilated[shifted_pixel_next[0],shifted_pixel_next[1]])
                distance+=abs(skeleton1_dilated[pixel]-float_value)
            else:
                distance+=1
#         for pixel in skeleton2_dilated.keys():
#             if (skeleton2_dilated.shape[0]>pixel[0]-shift[0]>=0 and skeleton2_dilated.shape[1]>pixel[1]-shift[1]>=0):
#                 shifted_pixel = (int(pixel[0]-shift[0]),int(pixel[1]-shift[1]))
#                 distance+=abs(skeleton1_dilated[shifted_pixel[0],shifted_pixel[1]]^skeleton2_dilated[pixel])
#             else:
#                 distance+=1
#         print(distance)
        return distance
    return(minimize(distance,np.array([10,10]), method='nelder-mead',options={'xatol': 1, 'disp': True,'fatol':0.1}))