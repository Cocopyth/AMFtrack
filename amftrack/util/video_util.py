import cv2
import numpy as np
import matplotlib.pyplot as plt
from amftrack.util.dbx import upload
import imageio
from time import time_ns
import os
from amftrack.util.sys import temp_path
from amftrack.pipeline.functions.image_processing.experiment_util import (
    plot_full_image_with_features,
    get_all_edges,
    get_all_nodes,

)
from random import choice

def make_video(paths,texts,resize,save_path=None,upload_path=None,fontScale=3,color = (0, 255, 255)):
    if resize is None:
        imgs = [cv2.imread(path,cv2.IMREAD_COLOR) for path in paths]
    else:
        imgs = [cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR),resize) for path in paths]
    for i,img in enumerate(imgs):
        anchor =img.shape[0]//10,img.shape[1]//10
        cv2.putText(img=img, text=texts[i],org = anchor, fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=fontScale, color=color,thickness=3)
    if not save_path is None:
        imageio.mimsave(save_path, imgs)
    if not upload_path is None:
        if save_path is None:
            time = time_ns()
            save_path_temp = os.path.join(temp_path,f'{time}.mp4')
            imageio.mimsave(save_path_temp, imgs)
        else:
            save_path_temp = save_path
        upload(save_path_temp,upload_path)
        if save_path is None:
            os.remove(save_path_temp)
    return(imgs)

def make_video_tile(paths_list,texts,resize,save_path=None,upload_path=None,fontScale=3,color = (0, 255, 255)):
    """"""
    if resize is None:
        imgs_list = [[cv2.imread(path, cv2.IMREAD_COLOR) for path in paths] for paths in paths_list]
    else:
        imgs_list = [[cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR),resize) for path in paths] for paths in paths_list]
    for i,imgs in enumerate(imgs_list):
        for j,img in enumerate(imgs):
            anchor =img.shape[0]//10,img.shape[1]//10
            cv2.putText(img=img, text=texts[i][j],org = anchor, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=fontScale, color=color,thickness=3)
    if len(imgs[0])%2==0:
        imgs_final = [[cv2.vconcat(imgs[::2]),cv2.vconcat(imgs[1::2])] for imgs in imgs_list]
        imgs_final = [cv2.hconcat(imgs) for imgs in imgs_final]
    else:
        imgs_final = [cv2.hconcat(imgs) for imgs in imgs_list]
    if not save_path is None:
        imageio.mimsave(save_path, imgs_final)
    if not upload_path is None:
        if save_path is None:
            time = time_ns()
            save_path_temp = os.path.join(temp_path,f'{time}.mp4')
            imageio.mimsave(save_path_temp, imgs_final)
        else:
            save_path_temp = save_path
        upload(save_path_temp,upload_path)
        if save_path is None:
            os.remove(save_path_temp)
    return(imgs)

def make_images_track(exp):
    nodes = get_all_nodes(exp, 0)
    nodes = [node for node in nodes if node.is_in(0) and
             np.linalg.norm(node.pos(0) - node.pos(node.ts()[-1])) > 1000 and
             len(node.ts()) > 3]

    paths_list = []
    num_tiles = 4
    node_select_list = [choice(nodes) for k in range(num_tiles)]
    for t in range(exp.ts):
        exp.load_tile_information(t)
        paths = []
        for k in range(num_tiles):
            node_select = node_select_list[k]
            pos = node_select.pos(0)
            window = 1500
            region = [[pos[0] - window, pos[1] - window], [pos[0] + window, pos[1] + window]]
            path = f"plot_nodes_{time_ns()}"
            path = os.path.join(temp_path, path)
            paths.append(path + '.png')
            plot_full_image_with_features(
                exp,
                t,
                region=region,
                downsizing=5,
                nodes=[node for node in get_all_nodes(exp, 0) if
                       node.is_in(t) and np.linalg.norm(node.pos(t) - pos) <= window],
                edges=get_all_edges(exp, t),
                dilation=5,
                prettify=False,
                save_path=path,
            )
        paths_list.append(paths)
    return(paths_list)