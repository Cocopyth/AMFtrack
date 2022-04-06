from amftrack.pipeline.functions.image_processing.extract_width_fun import *

import os

from pymatreader import read_mat
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
from amftrack.plotutil import plot_t_tp1
from amftrack.notebooks.analysis.util import directory_scratch
import imageio
from amftrack.transfer.functions.transfer import upload, zip_file
import matplotlib
from amftrack.util import *
from amftrack.pipeline.functions.post_processing.util import get_length_um_edge, is_in_study_zone
from matplotlib import cm
import matplotlib as mpl
from shapely.geometry import Polygon, shape,Point
from shapely.affinity import affine_transform, rotate
from scipy import spatial
import geopandas as gpd
from amftrack.pipeline.functions.post_processing.area_hulls import *
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment, save_graphs, load_graphs, load_skel, plot_raw_plus

API = str(np.load(os.getenv('HOME') + '/pycode/API_drop.npy'))
dir_drop = 'prince_data'


def make_movie_raw(directory,folders,plate_num,exp,id_unique,args=None):
    skels = []
    ims = []
    kernel = np.ones((5, 5), np.uint8)
    itera = 1
    for folder in folders:
        directory_name = folder
        path_snap = directory + directory_name
        skel_info = read_mat(path_snap + '/Analysis/skeleton_pruned_compressed.mat')
        skel = skel_info['skeleton']
        skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
        im = read_mat(path_snap + '/Analysis/raw_image.mat')['raw']
        ims.append(im)
    start = 0
    finish = len(exp.dates)
    for i in range(start, finish):
        plt.close('all')
        clear_output(wait=True)
        plot_t_tp1([], [], None, None, skels[i], ims[i], save=f'{directory_scratch}temp/{plate_num}_im{i}.png',
                   time=f't = {int(get_time(exp, 0, i))}h')
        plt.close('all')

    img_array = []
    for t in range(start, finish):
        img = cv2.imread(f'{directory_scratch}temp/{plate_num}_im{t}.png')
        img_array.append(img)

    path_movie = f'{directory_scratch}temp/{plate_num}.gif'
    imageio.mimsave(path_movie, img_array, duration=1)
    upload(API, path_movie, f'/{dir_drop}/{id_unique}/movie_raw.gif', chunk_size=256 * 1024 * 1024)
    path_movie = f'{directory_scratch}temp/{plate_num}.mp4'
    imageio.mimsave(path_movie, img_array)
    upload(API, path_movie, f'/{dir_drop}/{id_unique}/movie_raw.mp4', chunk_size=256 * 1024 * 1024)

def make_movie_aligned(directory,folders,plate_num,exp,id_unique,args=None):
    skels = []
    ims = []
    kernel = np.ones((5,5),np.uint8)
    itera = 2
    for folder in folders:
        directory_name=folder
        path_snap=directory+directory_name
        skel_info = read_mat(path_snap+'/Analysis/skeleton_realigned_compressed.mat')
        skel = skel_info['skeleton']
        skels.append(cv2.dilate(skel.astype(np.uint8),kernel,iterations = itera))
    start=0
    finish = len(exp.dates)
    for i in range(start,finish):
        plt.close('all')
        clear_output(wait=True)
        plot_t_tp1([], [], None, None, skels[i], skels[i], save=f'{directory_scratch}temp/{plate_num}_im{i}',
                   time=f't = {int(get_time(exp,0,i))}h')
        plt.close('all')
    img_array = []
    for t in range(start,finish):
        img = cv2.imread(f'{directory_scratch}temp/{plate_num}_im{t}.png')
        img_array.append(img)
    path_movie = f'{directory_scratch}temp/{plate_num}.gif'
    imageio.mimsave(path_movie, img_array,duration = 1)
    upload(API,path_movie,f'/{dir_drop}/{id_unique}/movie_aligned.gif',chunk_size=256 * 1024 * 1024)
    path_movie = f'{directory_scratch}temp/{plate_num}.mp4'
    imageio.mimsave(path_movie, img_array)
    upload(API,path_movie,f'/{dir_drop}/{id_unique}/movie_aligned.mp4',chunk_size=256 * 1024 * 1024)

def make_movie_RH_BAS(directory,folders,plate_num,exp,id_unique,args=None):
    time_plate_info, global_hypha_info, time_hypha_info = get_data_tables(redownload=True)
    table = global_hypha_info.loc[global_hypha_info['Plate']==plate_num].copy()
    table['log_length'] = np.log10((table['tot_length_C'] + 1).astype(float))
    table['is_rh'] = (table['log_length'] >= 3.36).astype(int)
    table = table.set_index('hypha')
    for t in range(exp.ts):
        plt.close('all')
        clear_output(wait=True)
        segs = []
        colors = []
        hyphaes = table.loc[(table['strop_track'] >= t) & (table['timestep_init_growth'] <= t) &
                            ((table['out_of_ROI'].isnull()) | (table['out_of_ROI'] > t))].index
        for hyph in exp.hyphaes:
            if t in hyph.ts and hyph.end.label in hyphaes:
                try:
                    nodes, edges = hyph.get_nodes_within(t)
                    color = "red" if np.all(table.loc[table.index == hyph.end.label]['is_rh']) else "blue"
                    # color = 'green' if np.all(table.loc[table.index == hyph.end.label]['is_small']) else color
                    for edge in edges:
                        origin, end = edge.end.get_pseudo_identity(t).pos(t), edge.begin.get_pseudo_identity(t).pos(t)
                        segs.append((origin, end))
                        colors.append(color)
                except nx.exception.NetworkXNoPath:
                    pass
        segs = [(np.flip(origin) // 5, np.flip(end) // 5) for origin, end in segs]
        skels = []
        ims = []
        kernel = np.ones((5, 5), np.uint8)
        itera = 2
        folders = list(exp.folders['folder'])
        folders.sort()
        for folder in folders[t:t + 1]:
            directory_name = folder
            path_snap = directory + directory_name
            skel_info = read_mat(path_snap + '/Analysis/skeleton_realigned_compressed.mat')
            skel = skel_info['skeleton']
            skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
        i = 0
        ln_coll = matplotlib.collections.LineCollection(segs, colors=colors)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(skels[i])
        ax.add_collection(ln_coll)
        plt.draw()
        right = 0.80
        top = 0.80
        fontsize = 20
        ax.text(
            right,
            top,
            f't = {int(get_time(exp,0,t))}h',
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
            color="white",
            fontsize=fontsize,
        )
        plt.savefig(f'{directory_scratch}temp/{plate_num}_rhbas_im{t}')
        plt.close(fig)
    img_array = []
    for t in range(exp.ts):
        img = cv2.imread(f'{directory_scratch}temp/{plate_num}_rhbas_im{t}.png')
        img_array.append(img)
    path_movie = f'{directory_scratch}temp/{plate_num}_rhbas.gif'
    imageio.mimsave(path_movie, img_array,duration = 1)
    upload(API,path_movie,f'/{dir_drop}/{id_unique}/movie_rhbas.gif',chunk_size=256 * 1024 * 1024)
    path_movie = f'{directory_scratch}temp/{plate_num}.mp4'
    imageio.mimsave(path_movie, img_array)
    upload(API,path_movie,f'/{dir_drop}/{id_unique}/movie_rhbas.mp4',chunk_size=256 * 1024 * 1024)

def make_movie_dens(directory,folders,plate_num,exp,id_unique,args=None):
    time_plate_info, global_hypha_info, time_hypha_info = get_data_tables(redownload=True)
    ts = range(exp.ts)
    incr = 100
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, ts, incr)
    plate = plate_num
    table = time_plate_info.loc[time_plate_info["Plate"] == plate]
    table = table.fillna(-1)
    my_cmap = cm.Greys
    my_cmap.set_under('k', alpha=0)
    table = table.set_index('t')
    num_hulls = len(regular_hulls) - 4
    for t in ts:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        skels = []
        ims = []
        kernel = np.ones((5, 5), np.uint8)
        itera = 2
        folders = list(exp.folders['folder'])
        folders.sort()
        for folder in folders[t:t + 1]:
            directory_name = folder
            path_snap = directory + directory_name
            skel_info = read_mat(path_snap + '/Analysis/skeleton_realigned_compressed.mat')
            skel = skel_info['skeleton']
            skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
        for index in range(num_hulls):
            polygon = regular_hulls[num_hulls - 1 - index]
            column = f"ring_density_incr-100_index-{num_hulls - 1 - index}"
            p = affine_transform(polygon, [0.2, 0, 0, -0.2, 0, 0])
            p = rotate(p, 90, origin=(0, 0))
            density = table[column][t]
            if density != -1:
                p = gpd.GeoSeries(p)
                _ = p.plot(ax=ax, color=cm.cool(density / 5000), alpha=0.9)
        polygon = regular_hulls[0]
        p = affine_transform(polygon, [0.2, 0, 0, -0.2, 0, 0])
        p = rotate(p, 90, origin=(0, 0))
        p = gpd.GeoSeries(p)
        _ = p.plot(ax=ax, color="black")
        norm = mpl.colors.Normalize(vmin=0, vmax=3000)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.cool), ax=ax, orientation='horizontal')
        ax.imshow(skels[0], vmin=0.00000001, cmap=my_cmap, zorder=30, interpolation=None)
        right = 0.90
        top = 0.90
        fontsize = 10
        text = ax.text(
            right,
            top,
            f"time = {int(table['time_since_begin'][t])}h",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax.transAxes,
            color="white",
            fontsize=fontsize,
        )
        plt.close(fig)
        plt.savefig(f'{directory_scratch}temp/{plate_num}_dens_im{t}')
    img_array = []
    for t in range(exp.ts):
        img = cv2.imread(f'{directory_scratch}temp/{plate_num}_dens_im{t}.png')
        img_array.append(img)
    path_movie = f'{directory_scratch}temp/{plate_num}_dens.gif'
    imageio.mimsave(path_movie, img_array,duration = 1)
    upload(API,path_movie,f'/{dir_drop}/{id_unique}/movie_dens.gif',chunk_size=256 * 1024 * 1024)
    path_movie = f'{directory_scratch}temp/{plate_num}_dens.mp4'
    imageio.mimsave(path_movie, img_array)
    upload(API,path_movie,f'/{dir_drop}/{id_unique}/movie_dens.mp4',chunk_size=256 * 1024 * 1024)

def plot_anastomosis(directory,folders,plate_num,exp,id_unique,args=None):
    for t in range(exp.ts):
        nodes = [node for node in exp.nodes if node.is_in(t) and np.all(is_in_study_zone(node, t, 1000, 200))]
        tips = [node for node in nodes if node.degree(t) == 1 and node.is_in(t + 1)
                and len(node.ts()) > 2]
        growing_tips = [node for node in tips if np.linalg.norm(node.pos(t) - node.pos(node.ts()[-1])) >= 40]
        growing_rhs = [node for node in growing_tips
                       if np.linalg.norm(node.pos(node.ts()[0]) - node.pos(node.ts()[-1])) >= 1500]

        anas_tips = [tip for tip in growing_tips
                     if tip.degree(t) == 1 and tip.degree(t + 1) ==
                     3 and 1 not in [tip.degree(t) for t in [tau for tau in tip.ts() if tau > t]]]
        for tip in anas_tips:
            node_list = [tip.label]
            fig, ax, center, radius = plot_raw_plus(exp, t, node_list, radius_imp=200)
            fig, ax, center, radius = plot_raw_plus(exp, t + 1, node_list, fig=fig, ax=ax,
                                                    center=center, radius_imp=radius,
                                                    n=2)
            fig, ax, center, radius = plot_raw_plus(exp, t + 4, node_list, fig=fig, ax=ax,
                                                    center=center, radius_imp=radius,
                                                    n=4)
            path_save = f'{directory_scratch}temp/{plate_num}_anas_im{t}_tip{tip.label}'
            plt.savefig(path_save+'.png')
            plt.close(fig)
            upload(API, path_save+'.png', f'/{dir_drop}/{id_unique}/anastomosis/{t}_tip{tip.label}.png',
                   chunk_size=256 * 1024 * 1024)