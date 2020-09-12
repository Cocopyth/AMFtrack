def get_path(date,plate,skeleton,row=None,column=None):
    def get_number(number):
        if number<10:
            return(f'0{number}')
        else:
            return(str(number))
    root_path = r'//sun.amolf.nl/shimizu-data/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE'
    date_plate = f'/2020{date}'
    plate = f'_Plate{plate}'
    if skeleton:
        end='/Analysis/Skeleton.mat'
    else:
        end='/Img'+f'/Img_r{get_number(row)}_c{get_number(column)}.tif'
    return (root_path+date_plate+plate+end)