from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
from amftrack.util import get_dirname
import pandas as pd
import ast
from scipy import sparse
from datetime import datetime
from amftrack.pipeline.functions.node_id import orient
import scipy.io as sio
import cv2
import imageio
import numpy as np
from skimage.filters import frangi
from skimage import filters
import scipy.sparse
import os
from time import time
from skimage.feature import hessian_matrix_det
from amftrack.pipeline.functions.extract_graph import from_sparse_to_graph, generate_nx_graph
from bresenham import bresenham
def streline(linelen,degrees):
    theta = degrees%180*np.pi/180
    ray = (linelen-1)/2;
    x = int(np.round((linelen-1)/2*np.cos(theta)))
    y = -int(np.round((linelen-1)/2*np.sin(theta)))
    points = np.array(list(bresenham(0, 0, x, y)))
    c,r=  np.concatenate((-np.flip(points[:,0]),[0],points[:,0])),np.concatenate((-np.flip(points[:,1]),[0],points[:,1]))
    M = 2*np.max(np.abs(r))+1
    N = 2*np.max(np.abs(c))+1
    line = np.zeros((M,N))
    x0=np.expand_dims((r+np.max(np.abs(r))),1)
    y0=np.expand_dims((c+np.max(np.abs(c))),1)
    line[x0,y0] = 1
    return(line)

def stredisk(radius):
    return(cv.getStructuringElement(cv.MORPH_ELLIPSE,(2*radius-1,2*radius-1)))

def bowler_hat(im,no,si):
    o = np.linspace(0,180,no)
    imol = np.zeros((im.shape[0],im.shape[1],len(si),no))
    imod = np.zeros((im.shape[0],im.shape[1],len(si)))
    for i in range (0,len(si)):
        for j in range(0,no):
            se = streline(si[i],o[j]).astype(np.uint8)
            imol[:,:,i,j] = cv.morphologyEx(im, cv.MORPH_OPEN, se)
        se = stredisk(int(np.round(si[i]/2))).astype(np.uint8)
        imod[:,:,i] = cv.morphologyEx(im, cv.MORPH_OPEN, se)
    imd = np.zeros((im.shape[0],im.shape[1],len(si)))
    imr = np.zeros((im.shape[0],im.shape[1],len(si)))
    imm = np.zeros((im.shape[0],im.shape[1],len(si)))
    triv = imod ==0
    for i in range (len(si)):
        imm[:,:,i] = np.max(np.squeeze(imol[:,:,i,:]),axis =2)
        imd[:,:,i] = imm[:,:,i] - imod[:,:,i]
    imr[triv] = 0
    imda = np.max(imd,axis=2)
    imda = np.double(imda)
    imda = (imda - np.min(imda[:]))/(np.max(imda[:])-np.min(imda[:]))
    return(imda)

def extract_skel_bowler_hat(im,no,si,low,high):
    bowled_hat = bowler_hat(-im, no, si)
    transformed = 255 - bowled_hat
    hyst = filters.apply_hysteresis_threshold(transformed, low, high)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv.dilate(hyst.astype(np.uint8) * 255, kernel, iterations=1)
    for i in range(3):
        dilation = cv.erode(dilation.astype(np.uint8) * 255, kernel, iterations=1)
        dilation = cv.dilate(dilation.astype(np.uint8) * 255, kernel, iterations=1)
    dilated = dilation > 0

    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(dilated.astype(np.uint8), connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 4000

    # your answer image
    img2 = np.zeros((dilated.shape))
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 1
    return(img2)

def extract_skel_tip_ext(im,low,high,dist):
    im_cropped = im
    im_blurred =cv2.blur(im_cropped, (200, 200))
    im_back_rem = (im_cropped)/((im_blurred==0)*np.ones(im_blurred.shape)+im_blurred)*120
    im_back_rem[im_back_rem>=130]=130
    # # im_back_rem = im_cropped*1.0
    # # # im_back_rem = cv2.normalize(im_back_rem, None, 0, 255, cv2.NORM_MINMAX)
    frangised = frangi(im_back_rem,sigmas=range(1,20,4))*255
    # # frangised = cv2.normalize(frangised, None, 0, 255, cv2.NORM_MINMAX)
    hessian = hessian_matrix_det(im_back_rem,sigma = 20)
#     transformed = (frangised+cv2.normalize(blur_hessian, None, 0, 255, cv2.NORM_MINMAX)-im_back_rem+120)*(im_blurred>=35)
#     transformed = (frangised+cv2.normalize(abs(hessian), None, 0, 255, cv2.NORM_MINMAX)-im_back_rem+120)*(im_blurred>=35)
    transformed = (frangised-im_back_rem+120)*(im_blurred>=35)
#     low = 20
#     high = 100
    hyst = filters.apply_hysteresis_threshold(transformed, low, high)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(hyst.astype(np.uint8) * 255,kernel,iterations = 1)
    for i in range(3):
        dilation=cv2.erode(dilation.astype(np.uint8) * 255,kernel,iterations = 1)
        dilation = cv2.dilate(dilation.astype(np.uint8) * 255,kernel,iterations = 1)
    dilated = dilation>0

    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(dilated.astype(np.uint8), connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = 4000  

    #your answer image
    img2 = np.zeros((dilated.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 1
    skeletonized = cv2.ximgproc.thinning(np.array(255*img2,dtype=np.uint8))
    nx_g = generate_nx_graph(from_sparse_to_graph(scipy.sparse.dok_matrix(skeletonized)))
    g,pos= nx_g
    tips = [node for node in g.nodes if g.degree(node)==1]
    dilated_bis = np.copy(img2)
    for tip in tips:
        branch = np.array(orient(g.get_edge_data(*list(g.edges(tip))[0])['pixel_list'],pos[tip]))
        orientation = branch[0]-branch[min(branch.shape[0]-1,20)]
        orientation = orientation/(np.linalg.norm(orientation))
        window = 20
        x,y = pos[tip][0],pos[tip][1]
        if x-window>=0 and x+window< dilated.shape[0] and y-window>=0 and y+window< dilated.shape[1]:
            shape_tip = dilated[x-window:x+window,y-window:y+window]
#             dist = 20
            for i in range(dist):
                pixel = (pos[tip]+orientation*i).astype(int)
                xp,yp = pixel[0],pixel[1]
                if xp-window>=0 and xp+window< dilated.shape[0] and yp-window>=0 and yp+window< dilated.shape[1]:
                    dilated_bis[xp-window:xp+window,yp-window:yp+window]+=shape_tip
    dilation = cv2.dilate(dilated_bis.astype(np.uint8) * 255,kernel,iterations = 1)
    for i in range(3):
        dilation=cv2.erode(dilation.astype(np.uint8) * 255,kernel,iterations = 1)
        dilation = cv2.dilate(dilation.astype(np.uint8) * 255,kernel,iterations = 1)
        return(dilation>0)