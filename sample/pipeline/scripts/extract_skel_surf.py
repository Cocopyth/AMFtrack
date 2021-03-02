from util import get_dirname
import pandas as pd
import ast
from scipy import sparse
from datetime import datetime
from node_id import orient
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
import sys
from sample.pipeline.scripts.extract_graph import from_sparse_to_graph, generate_nx_graph

i = int(sys.argv[-1])
plate = int(sys.argv[1])
low = int(sys.argv[2])
high = int(sys.argv[3])
dist = int(sys.argv[4])
directory = str(sys.argv[5])

listdir=os.listdir(directory) 
list_dir_interest=[name for name in listdir if name.split('_')[-1]==f'Plate{0 if plate<10 else ""}{plate}']
ss=[name.split('_')[0] for name in list_dir_interest]
ff=[name.split('_')[1] for name in list_dir_interest]
dates_datetime=[datetime(year=int(ss[i][:4]),month=int(ss[i][4:6]),day=int(ss[i][6:8]),hour=int(ff[i][0:2]),minute=int(ff[i][2:4])) for i in range(len(list_dir_interest))]
dates_datetime.sort()
dates_datetime_chosen=dates_datetime
dates = dates_datetime_chosen
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
for name in tileconfig[0]:
    imname = '/Img/'+name.split('/')[-1]
#     ims.append(imageio.imread('//sun.amolf.nl/shimizu-data/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE'+date_plate+plate_str+'/Img/'+name))
    ims.append(imageio.imread(directory+directory_name+imname))
skel = np.zeros(dim,dtype=np.uint8)
contour = scipy.sparse.lil_matrix(dim,dtype=np.uint8)
half_circle = scipy.sparse.lil_matrix(dim,dtype=np.uint8)
for index,im in enumerate(ims):
    print(index)
    im_cropped = im
    im_blurred =cv2.blur(im_cropped, (200, 200))
    im_back_rem = (im_cropped+1)/(im_blurred+1)*120
    im_back_rem[im_back_rem>=130]=130
    # # im_back_rem = im_cropped*1.0
    # # # im_back_rem = cv2.normalize(im_back_rem, None, 0, 255, cv2.NORM_MINMAX)
    frangised = frangi(im_back_rem,sigmas=range(1,20,4))*255
    # # frangised = cv2.normalize(frangised, None, 0, 255, cv2.NORM_MINMAX)
    hessian = hessian_matrix_det(im_back_rem,sigma = 20)
    blur_hessian = cv2.blur(abs(hessian), (20, 20))
#     transformed = (frangised+cv2.normalize(blur_hessian, None, 0, 255, cv2.NORM_MINMAX)-im_back_rem+120)*(im_blurred>=35)
#     transformed = (frangised+cv2.normalize(abs(hessian), None, 0, 255, cv2.NORM_MINMAX)-im_back_rem+120)*(im_blurred>=35)
    transformed = (frangised-im_back_rem+120)*(im_blurred>=35)
#     low = 20
#     high = 100
    lowt = (transformed > low).astype(int)
    hight = (transformed > high).astype(int)
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
#     skeletonized = cv2.ximgproc.thinning(np.array(255*dilated_bis,dtype=np.uint8))
    print('number threshold : ', np.sum(dilated_bis))
    laplacian = cv2.Laplacian((im_cropped<=15).astype(np.uint8),cv2.CV_64F)
    points = laplacian>=4
#     np.save(f'Temp\dilated{tileconfig[0][i]}',dilated)
    boundaries = int(tileconfig[2][index][0]-np.min(xs)),int(tileconfig[2][index][1]-np.min(ys))
    skel[boundaries[1]:boundaries[1]+shape[0],boundaries[0]:boundaries[0]+shape[1]] += dilation>0
    contour[boundaries[1]:boundaries[1]+shape[0],boundaries[0]:boundaries[0]+shape[1]] += points
    if index<=80:
        half_circle[boundaries[1]:boundaries[1]+shape[0],boundaries[0]:boundaries[0]+shape[1]] += points
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
