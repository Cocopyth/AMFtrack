import cv2
import numpy as np
import matplotlib.pyplot as plt
from amftrack.util.dbx import upload
import imageio
from time import time_ns
import os
from amftrack.util.sys import temp_path

def make_video(paths,texts,resize,save_path=None,upload_path=None,fontScale=3,color = (0, 255, 255)):
    imgs = [cv2.imread(path,cv2.IMREAD_COLOR) for path in paths]
    if not resize is None:
        imgs = [cv2.resize(img,resize) for img in imgs]
    for i,img in enumerate(imgs):
        anchor =img.shape[0]//10,img.shape[1]//10
        cv2.putText(img=img, text=texts[i],org = anchor, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=fontScale, color=color,thickness=3)
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
    if resize is None:
        imgs_list = [[cv2.imread(path, cv2.IMREAD_COLOR) for path in paths] for paths in paths_list]
    else:
        imgs_list = [[cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR),resize) for path in paths] for paths in paths_list]
    for i,imgs in enumerate(imgs_list):
        for j,img in enumerate(imgs):
            anchor =img.shape[0]//10,img.shape[1]//10
            cv2.putText(img=img, text=texts[i][j],org = anchor, fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=fontScale, color=color,thickness=3)
    imgs_final = [cv2.vconcat(imgs) for imgs in imgs_list]
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