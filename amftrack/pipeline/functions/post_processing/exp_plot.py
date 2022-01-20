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