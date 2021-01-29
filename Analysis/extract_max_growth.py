import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
from util import get_path, get_dates_datetime
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from extract_graph import generate_nx_graph, transform_list, generate_skeleton, generate_nx_graph_from_skeleton, from_connection_tab, from_nx_to_tab
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
from matplotlib import colors
from copy import deepcopy,copy
from collections import Counter
import cv2 
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi, meijering
from skimage.morphology import thin
from skimage import data, filters
from random import choice
import scipy.sparse
import os
from time import time
from extract_graph import dic_to_sparse, from_sparse_to_graph, generate_nx_graph, prune_graph, from_nx_to_tab, from_nx_to_tab_matlab,sparse_to_doc, connections_pixel_list_to_tab, transform_list, clean_degree_4
from time import sleep
from skimage.feature import hessian_matrix_det
from experiment_class_surf import Experiment,clean_exp_with_hyphaes
from hyphae_id_surf import clean_and_relabel, get_mother, save_hyphaes, resolve_ambiguity_two_ends,solve_degree4, clean_obvious_fake_tips, get_pixel_growth_and_new_children
from subprocess import call
import open3d as o3d
import sklearn
from Analysis.util import *
from directory import directory, path_code
from Analysis.data_info import *
from scipy import stats
from scipy.ndimage.filters import uniform_filter1d

for treatment in treatments.keys():
    insts = treatments[treatment]
    for inst in insts:
        result = estimate_growth(inst, criter)
        pickle.dump(result, open(f'{path_code}/MscThesis/Results/maxgrowth_{inst}.pick', "wb"))