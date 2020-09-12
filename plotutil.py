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
    dilated = cv2.dilate(skel,kernel,iterations = 4)
    fig = plt.figure()
    plt.imshow(raw, cmap='gray',interpolation='none')
    plt.imshow(dilated, cmap='jet', alpha=0.5,interpolation='none')
    plt.show()