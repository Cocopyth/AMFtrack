from scipy import sparse
import numpy as np
import os
from datetime import datetime, timedelta
import pandas
path_code = "/home/cbisot/pycode/"
plate_info = pandas.read_excel(path_code + 'MscThesis/plate_info/SummaryAnalizedPlates.xlsx',engine='openpyxl',header=3,)


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

def get_plate_number(position_number,date):
    for index,row in plate_info.loc[plate_info['Position #']==position_number].iterrows():
        if type(row['crossed date'])==datetime:
            date_crossed = row['crossed date']
            date_harvest = row['harvest date']+timedelta(days=1)
        else:
            date_crossed = datetime.strptime(row['crossed date'], "%d.%m.%Y")
            date_harvest = datetime.strptime(row['harvest date'], "%d.%m.%Y")+timedelta(days=1)
        if date>= date_crossed and date<= date_harvest:
            return(row['Plate #'])
        
def get_postion_number(plate_number):
    for index,row in plate_info.loc[plate_info['Plate #']==plate_number].iterrows():
            return(row['Position #'])



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
