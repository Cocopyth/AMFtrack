from pymatreader import read_mat
from scipy import sparse
import numpy as np
import os
from datetime import datetime, timedelta
import pandas
from amftrack.pipeline.functions.extract_graph import from_sparse_to_graph, generate_nx_graph, sparse_to_doc
import cv2
import json
import pandas as pd

path_code = "/home/cbisot/pycode/MscThesis/"
# path_code = r'C:\Users\coren\Documents\PhD\Code\AMFtrack'
# plate_info = pandas.read_excel(path_code+r'/plate_info/SummaryAnalizedPlates.xlsx',engine='openpyxl',header=3,)


def get_path(date, plate, skeleton, row=None, column=None, extension=".mat"):
    def get_number(number):
        if number < 10:
            return f"0{number}"
        else:
            return str(number)

    root_path = (
        r"//sun.amolf.nl/shimizu-data/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE"
    )
    date_plate = f"/2020{date}"
    plate = f"_Plate{plate}"
    if skeleton:
        end = "/Analysis/Skeleton" + extension
    else:
        end = "/Img" + f"/Img_r{get_number(row)}_c{get_number(column)}.tif"
    return root_path + date_plate + plate + end


def get_dates_datetime(directory, plate):
    listdir = os.listdir(directory)
    list_dir_interest = [
        name
        for name in listdir
        if name.split("_")[-1] == f'Plate{0 if plate<10 else ""}{plate}'
    ]
    ss = [name.split("_")[0] for name in list_dir_interest]
    ff = [name.split("_")[1] for name in list_dir_interest]
    dates_datetime = [
        datetime(
            year=int(ss[i][:4]),
            month=int(ss[i][4:6]),
            day=int(ss[i][6:8]),
            hour=int(ff[i][0:2]),
            minute=int(ff[i][2:4]),
        )
        for i in range(len(list_dir_interest))
    ]
    dates_datetime.sort()
    return dates_datetime

def get_dirname(date,plate):
    return(f'{date.year}{0 if date.month<10 else ""}{date.month}{0 if date.day<10 else ""}{date.day}_{0 if date.hour<10 else ""}{date.hour}{0 if date.minute<10 else ""}{date.minute}_Plate{0 if plate<10 else ""}{plate}')

# def get_plate_number(position_number,date):
#     for index,row in plate_info.loc[plate_info['Position #']==position_number].iterrows():
#         if type(row['crossed date'])==datetime:
#             date_crossed = row['crossed date']
#             date_harvest = row['harvest date']+timedelta(days=1)
#         else:
#             date_crossed = datetime.strptime(row['crossed date'], "%d.%m.%Y")
#             date_harvest = datetime.strptime(row['harvest date'], "%d.%m.%Y")+timedelta(days=1)
#         if date>= date_crossed and date<= date_harvest:
#             return(row['Plate #'])
        
# def get_postion_number(plate_number):
#     for index,row in plate_info.loc[plate_info['Plate #']==plate_number].iterrows():
#             return(row['Position #'])

# def get_begin_index(plate_number,directory):
#     plate = get_postion_number(plate_number)
#     dates_datetime = get_dates_datetime(directory,plate)
#     plate_number_found = get_plate_number(plate,dates_datetime[0])
#     print(0,plate_number)
#     for i in range(len(dates_datetime)):
#         new_plate_number = get_plate_number(plate,dates_datetime[i])
#         if plate_number_found!=new_plate_number:
#             plate_number_found=new_plate_number
#             print(i,plate_number_found)
#         if plate_number_found == plate_number:
#             return(i,dates_datetime[i])

def shift_skeleton(skeleton, shift):
    shifted_skeleton = sparse.dok_matrix(skeleton.shape, dtype=bool)
    for pixel in skeleton.keys():
        #             print(pixel[0]+shift[0],pixel[1]+shift[1])
        if (
            skeleton.shape[0] > np.ceil(pixel[0] + shift[0]) > 0
            and skeleton.shape[1] > np.ceil(pixel[1] + shift[1]) > 0
        ):
            shifted_pixel = (
                np.round(pixel[0] + shift[0]),
                np.round(pixel[1] + shift[1]),
            )
            shifted_skeleton[shifted_pixel] = 1
    return shifted_skeleton

def transform_skeleton_final_for_show(skeleton_doc,Rot,trans):
    skeleton_transformed={}
    transformed_keys = np.round(np.transpose(np.dot(Rot,np.transpose(np.array(list(skeleton_doc.keys())))))+trans).astype(np.int)
    i=0
    for pixel in list(transformed_keys):
        i+=1
        skeleton_transformed[(pixel[0],pixel[1])]=1
    skeleton_transformed_sparse=sparse.lil_matrix((27000, 60000))
    for pixel in list(skeleton_transformed.keys()):
        i+=1
        skeleton_transformed_sparse[(pixel[0],pixel[1])]=1
    return(skeleton_transformed_sparse)

def get_skeleton(exp,boundaries,t,directory):
    i = t
    plate = exp.plate   
    listdir=os.listdir(directory) 
    dates = exp.dates
    date =dates [i]
    directory_name = get_dirname(date, plate)
    path_snap=directory+directory_name
    skel = read_mat(path_snap+'/Analysis/skeleton_pruned_realigned.mat')
    skelet = skel['skeleton']
    skelet = sparse_to_doc(skelet)
    Rot= skel['R']
    trans = skel['t']
    skel_aligned = transform_skeleton_final_for_show(skelet,np.array([[1,0],[0,1]]),np.array([0,0]))
    output = skel_aligned[boundaries[2]:boundaries[3],boundaries[0]:boundaries[1]].todense()
    kernel = np.ones((5,5),np.uint8)
    output = cv2.dilate(output.astype(np.uint8),kernel,iterations = 2)
    return(output,Rot,trans)

def get_param(folder,directory): #Very ugly but because interfacing with Matlab so most elegant solution.
    path_snap=directory+folder
    file1 = open(path_snap + "/param.m", 'r')
    Lines = file1.readlines()
    ldict = {}
    for line in Lines:
        exec(line.split(';')[0],globals(),ldict)
    files = [ '/Img/TileConfiguration.txt.registered', '/Analysis/skeleton_compressed.mat', 
         '/Analysis/skeleton_masked_compressed.mat',
         '/Analysis/skeleton_pruned_compressed.mat', '/Analysis/transform.mat',
         '/Analysis/transform_corrupt.mat',
         '/Analysis/skeleton_realigned_compressed.mat']
    for file in files:
        ldict[file] = os.path.isfile(path_snap + file)
    return(ldict)

def update_plate_info(directory):
    listdir = os.listdir(directory)
    plate_info = json.load(open('data_info.json', 'r')) if os.path.isfile('data_info.json') else {}
    for folder in listdir:
        path_snap=directory+folder
        if os.path.isfile(path_snap + "/param.m"):
            params = get_param(folder,directory)
            ss = folder.split("_")[0]
            ff = folder.split("_")[1]
            date = datetime(
                    year=int(ss[:4]),
                    month=int(ss[4:6]),
                    day=int(ss[6:8]),
                    hour=int(ff[0:2]),
                    minute=int(ff[2:4]),
                )
            params['date'] = datetime.strftime(date, "%d.%m.%Y, %H:%M:")
            plate_info[folder] = params
    with open('data_info.json', 'w') as jsonf:
        json.dump(plate_info, jsonf,  indent=4)
        
def get_data_info():
    data_info = pd.read_json('data_info.json',
   convert_dates=True).transpose()
    data_info.index.name = 'folder'
    data_info.reset_index(inplace=True)
    return(data_info)

def get_current_folders(directory):
    plate_info = get_data_info()
    listdir = os.listdir(directory)
    return(plate_info.loc[np.isin(plate_info['folder'],listdir)])

def get_folders_by_plate_id(plate_id, begin = 0, end = -1, directory = None):
    data_info = get_data_info() if directory is None else get_current_folders(directory)
    folders = data_info.loc[10**8*data_info['Plate']+data_info['CrossDate']== plate_id]
    dates_datetime = [datetime.strptime(row['date'],"%d.%m.%Y, %H:%M:") for index,row in folders.iterrows()]
    dates_datetime.sort()
    dates_datetime_select = dates_datetime[begin: end]
    dates_str = [datetime.strftime(date, "%d.%m.%Y, %H:%M:") for date in dates_datetime_select]
    select_folders = folders.loc[np.isin(folders['date'],dates_str)]
    return(select_folders)