import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2 
import numpy as np


def show_im(matrix,alpha=None,cmap='gray',interpolation='none'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(matrix, cmap=cmap, interpolation=interpolation,alpha=alpha)
    numrows, numcols = matrix.shape[0],matrix.shape[1]
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = matrix[row,col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f'%(y, x, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(y, x)

    ax.format_coord = format_coord
    plt.show()

def show_im_rgb(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(matrix, cmap='gray', interpolation='nearest')
    numrows, numcols = matrix.shape[0],matrix.shape[1]
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            return 'x=%1.4f, y=%1.4f'%(y, x)
        else:
            return 'x=%1.4f, y=%1.4f'%(y, x)

    ax.format_coord = format_coord
    plt.show()

def overlap(skel,raw):
    kernel = np.ones((3,3),np.uint8)
#     dilated = cv2.dilate(skel,kernel,iterations = 4)
    fig = plt.figure()
    matrix=skel
    ax = fig.add_subplot(111)
    ax.imshow(raw, cmap='gray',interpolation='none')
    ax.imshow(skel, cmap='jet', alpha=0.5,interpolation='none')
    numrows, numcols = matrix.shape[0],matrix.shape[1]
    def format_coord(x, y):
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = matrix[row,col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f'%(y, x, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(y, x)

    ax.format_coord = format_coord
    plt.show()
    
def plot_nodes(nx_graph,pos,im):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    bbox_props = dict(boxstyle="circle", fc="white")
    for node in nx_graph.nodes:
        t = ax.text(pos[node][1], pos[node][0], str(node), ha="center", va="center",
                    size=5,
                    bbox=bbox_props)
    plt.show()
    
def plot_nodes_from_list(node_list,pos,im):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    bbox_props = dict(boxstyle="circle", fc="white")
    for node in node_list:
        t = ax.text(pos[node][1], pos[node][0], str(node), ha="center", va="center",
                    size=5,
                    bbox=bbox_props)
    plt.show()
    
def plot_t_tp1(node_list_t,node_list_tp1,pos_t,pos_tp1,imt,imtp1,relabel_t=lambda x:x,relabel_tp1=lambda x:x, shift=(0,0),compress=1,save='',time=None):
    right = 0.90
    top = 0.90
    if len(save)>=1:
        fig=plt.figure(figsize=(14,12))
        size = 10
    else:
        fig = plt.figure()
        size = 5
    ax = fig.add_subplot(111)
    ax.imshow(imtp1, cmap='gray',interpolation='none')
    ax.imshow(imt, cmap='jet', alpha=0.5,interpolation='none')
    bbox_props1 = dict(boxstyle="circle", fc="grey")
    bbox_props2 = dict(boxstyle="circle", fc="white")
    ax.text(right, top, time,
        horizontalalignment='right',
        verticalalignment='bottom',
        transform=ax.transAxes,color='white')
    for node in node_list_t:
        t = ax.text((pos_t[node][1]-shift[1])//compress, (pos_t[node][0]-shift[0])//compress, str(relabel_t(node)), ha="center", va="center",
                    size=size,
                    bbox=bbox_props1)
    for node in node_list_tp1:
        if node in pos_tp1.keys():
            t = ax.text((pos_tp1[node][1]-shift[1])//compress, (pos_tp1[node][0]-shift[0])//compress, str(relabel_tp1(node)), ha="center", va="center",
                        size=size,
                        bbox=bbox_props2)
    if len(save)>=1:
        plt.savefig(save)
        plt.close(fig)
    else:
        plt.show()
        
def compress_skeleton(skeleton_doc,factor):
    shape=skeleton_doc.shape
    final_picture = np.zeros(shape=(shape[0]//factor,shape[1]//factor))
    for pixel in skeleton_doc.keys():
        x=min(round(pixel[0]/factor),shape[0]//factor-1)
        y=min(round(pixel[1]/factor),shape[1]//factor-1)
        final_picture[x,y]+=1
    return(final_picture)