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
from amftrack.pipeline.functions.extract_skel import extract_skel_tip_ext
from amftrack.util import get_dates_datetime, get_dirname, get_plate_number, get_postion_number

i = int(sys.argv[-1])
k = int(sys.argv[-2])
plate = int(sys.argv[1])
low = int(sys.argv[2])
high = int(sys.argv[3])
dist = int(sys.argv[4])
directory = str(sys.argv[5])

dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
dates = dates_datetime
date = dates[i]
directory_name = get_dirname(date, plate)
path_snap=directory+directory_name
path_tile=path_snap+'/Img/TileConfiguration.txt.registered'
try:
    tileconfig = pd.read_table(path_tile,sep=';',skiprows=4,header=None,converters={2 : ast.literal_eval},skipinitialspace=True)
except:
    print('error_name')
    path_tile=path_snap+'/Img/TileConfiguration.registered.txt'
    tileconfig = pd.read_table(path_tile,sep=';',skiprows=4,header=None,converters={2 : ast.literal_eval},skipinitialspace=True)
dirName = path_snap+'/Analysis'
shape = (3000,4096)
try:
    os.mkdir(path_snap+'/Analysis') 
    print("Directory " , dirName ,  " Created ")
except FileExistsError:
    print("Directory " , dirName ,  " already exists")  
t=time()
xs =[c[0] for c in tileconfig[2]]
ys =[c[1] for c in tileconfig[2]]
dim = (int(np.max(ys)-np.min(ys))+4096,int(np.max(xs)-np.min(xs))+4096)
ims = []
for name in tileconfig[0][k]:
    imname = '/Img/'+name.split('/')[-1]
#     ims.append(imageio.imread('//sun.amolf.nl/shimizu-data/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE'+date_plate+plate_str+'/Img/'+name))
    ims.append(imageio.imread(directory+directory_name+imname))
for index,im in enumerate(ims):
    print(index)
#     im_cropped = im
#     im_blurred =cv2.blur(im_cropped, (200, 200))
#     im_back_rem = (im_cropped)/((im_blurred==0)*np.ones(im_blurred.shape)+im_blurred)*120
#     im_back_rem[im_back_rem>=130]=130
#     hessian = hessian_matrix_det(im_back_rem,sigma = 20)
#     blur_hessian = cv2.blur(abs(hessian), (20, 20))
# #     transformed = (frangised+cv2.normalize(blur_hessian, None, 0, 255, cv2.NORM_MINMAX)-im_back_rem+120)*(im_blurred>=35)
# #     transformed = (frangised+cv2.normalize(abs(hessian), None, 0, 255, cv2.NORM_MINMAX)-im_back_rem+120)*(im_blurred>=35)
#     #for every component in the image, you keep it only if it's above min_size
#     laplacian = cv2.Laplacian((im_cropped<=15).astype(np.uint8),cv2.CV_64F)
#     points = laplacian>=4
    segmented = extract_skel_tip_ext(im,low,high,dist)
#     np.save(f'Temp\dilated{tileconfig[0][i]}',dilated)
    boundaries = int(tileconfig[2][index][0]-np.min(xs)),int(tileconfig[2][index][1]-np.min(ys))
    skel[boundaries[1]:boundaries[1]+shape[0],boundaries[0]:boundaries[0]+shape[1]] += segmented
    # contour[boundaries[1]:boundaries[1]+shape[0],boundaries[0]:boundaries[0]+shape[1]] += points
    # if index<=80:
    #     half_circle[boundaries[1]:boundaries[1]+shape[0],boundaries[0]:boundaries[0]+shape[1]] += points
# print(len(skel.nonzero()[0]))
# skelet = sparse_to_doc(skel)
print("number to reduce : ", np.sum(skel>0),np.sum(skel<=0))
skeletonized = cv2.ximgproc.thinning(np.array(255*(skel>0),dtype=np.uint8))
# skeletonized = cv2.ximgproc.thinning(np.array(255*(skel>0),dtype=np.uint8))
# skeletonized = zhangSuen(skelet)
skel_sparse = sparse.lil_matrix(skel)
sio.savemat(path_snap+'/Analysis/dilated.mat',{'dilated' : skel_sparse})
# sio.savemat(path_snap+'/Analysis/skeleton.mat',{'skeleton' : scipy.sparse.csc_matrix(skeletonized),'contour' : scipy.sparse.csc_matrix(contour),'half_circle' : half_circle})
sio.savemat(path_snap+'/Analysis/skeleton.mat',{'skeleton' : scipy.sparse.csc_matrix(skeletonized)})
compressed = cv2.resize(skeletonized,(dim[1]//5,dim[0]//5))
sio.savemat(path_snap+'/Analysis/skeleton_compressed.mat',{'skeleton' : compressed})
print('time=',time()-t)
