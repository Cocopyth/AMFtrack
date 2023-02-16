import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    clean_degree_4,
)
import scipy
from amftrack.pipeline.functions.image_processing.node_id import remove_spurs
from amftrack.pipeline.functions.image_processing.extract_skel import remove_component, remove_holes
import numpy as np
from amftrack.pipeline.development.high_mag_videos.high_mag_videos_fun import *
from scipy import signal
from amftrack.pipeline.functions.image_processing.extract_skel import (
    extract_skel_new_prince,
    run_back_sub,
    bowler_hat,
)
from scipy.interpolate import griddata

from skimage.morphology import skeletonize
from amftrack.util.sys import temp_path
import pandas as pd
from PIL import Image
from scipy.optimize import curve_fit
from amftrack.pipeline.functions.image_processing.extract_skel import (
    extract_skel_new_prince,
    run_back_sub,
    bowler_hat,
)

images_path = r"F:/AMOLF_Data/STORAGE/20221109_Plate462_04"
# images_path = r"/mnt/sun/home-folder/cargill/20221109_Plate462/20221109_Plate462_006"
fps = 20
time_pixel_size = 1/fps #s.pixel
binning = 2
magnification = 50
space_pixel_size = 2*1.725/(magnification)*binning #um.pixel
video_name = images_path.split('/')[-1]
kymos_path = '/'.join(images_path.split('/')[:-1]+["_".join((video_name,'kymos'))])
if not os.path.exists(kymos_path):
    os.mkdir(kymos_path)
files = os.listdir(images_path)
images_total_path = [os.path.join(images_path,file) for file in files]
images_total_path.sort()

selection_file = images_total_path
selection_file.sort()
begin  = 1
end = -1
image = imageio.imread(selection_file[end])
image2 = imageio.imread(selection_file[begin])
selection_file = selection_file[begin:end]

segmented,nx_graph_pruned,pos = segment_fluo(image,thresh = 5e-07)

weight = 0.05

edges = list(nx_graph_pruned.edges)
edge_oriented = []
for edge in edges:
    if pos[edge[0]][0]>pos[edge[1]][0]:
        edge_oriented.append(edge)
    else:
        edge_oriented.append((edge[1],edge[0]))
edges = edge_oriented

edge_oriented

np.linalg.norm(pos[edge[0]]-pos[edge[1]])
bound1 = 0
bound2 = 1
step=30
target_length=130
resolution = 1
fig, ax = plt.subplots()
for edge in edges:
    offset=int(np.linalg.norm(pos[edge[0]]-pos[edge[1]]))//4
    slices, segments = extract_section_profiles_for_edge(
    edge,
    pos,
    image,
    nx_graph_pruned,
    resolution=resolution,
    offset=offset,
    step=step,
    target_length=target_length,
    bound1=bound1,
    bound2=bound2)
    # plot_segments_on_image(segments,ax,color=None)
    plot_segments_on_image(segments,ax, bound1=bound1,
    bound2=bound2,color = 'white',alpha = 0.1)
    ax.plot([pos[edge[0]][1],pos[edge[1]][1]],[pos[edge[0]][0],pos[edge[1]][0]])
    ax.text(*np.flip((1-weight) * pos[edge[0]]+weight*pos[edge[1]]),str(edge[0]),color="white")
    ax.text(*np.flip((1-weight) * pos[edge[1]]+weight*pos[edge[0]]),str(edge[1]),color="white")
save_path_temp = os.path.join(kymos_path, f"extraction.png")
plt.savefig(save_path_temp)

kymos = {edge:get_kymo(edge,pos,selection_file,nx_graph_pruned, resolution=1,
    offset=offset,
    step=step,
    target_length=target_length,
    bound1=bound1,
    bound2=bound2) for edge in edges}

