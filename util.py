from scipy import sparse
import numpy as np

def get_path(date,plate,skeleton,row=None,column=None,extension=".mat"):
    def get_number(number):
        if number<10:
            return(f'0{number}')
        else:
            return(str(number))
    root_path = r'//sun.amolf.nl/shimizu-data/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE'
    date_plate = f'/2020{date}'
    plate = f'_Plate{plate}'
    if skeleton:
        end='/Analysis/Skeleton'+extension
    else:
        end='/Img'+f'/Img_r{get_number(row)}_c{get_number(column)}.tif'
    return (root_path+date_plate+plate+end)

def shift_skeleton(skeleton,shift):
    shifted_skeleton=sparse.dok_matrix(skeleton.shape, dtype=bool)
    for pixel in skeleton.keys():
#             print(pixel[0]+shift[0],pixel[1]+shift[1])
            if (skeleton.shape[0]>np.ceil(pixel[0]+shift[0])>0 and skeleton.shape[1]>np.ceil(pixel[1]+shift[1])>0):
                shifted_pixel = (np.round(pixel[0]+shift[0]),np.round(pixel[1]+shift[1]))
                shifted_skeleton[shifted_pixel]=1
    return(shifted_skeleton)

