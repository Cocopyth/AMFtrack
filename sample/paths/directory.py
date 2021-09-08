from datetime import datetime
from subprocess import call
from sample.util import get_dates_datetime, get_dirname
import os
from copy import copy

# directory = "/scratch/shared/AMF914/old/from_cartesius/"
# directory = "/scratch/shared/mrozemul/Fiji.app/"
directory_scratch = "/scratch/shared/AMF914/Fiji.app/"
directory_project = "/projects/0/einf914/data/"



path_job = "/home/cbisot/bash/job.sh"
path_code = "/home/cbisot/pycode/"


def run_parallel(code, args, begin, end, num_parallel, time, name):
    begin_skel = begin // num_parallel
    end_skel = (end) // num_parallel + 1
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    arg_str_out = "_".join([str(arg) for arg in args if type(arg)!=str])
    for j in range(begin_skel, end_skel):
        start = num_parallel * j + begin % num_parallel
        stop = num_parallel * j + num_parallel - 1 + begin % num_parallel
        ide = int(datetime.now().timestamp())
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH -N 1 \n#SBATCH -t {time}\n#SBATCH -p normal\n"
        )
        my_file.write(
            f'#SBATCH -o "{path_code}MscThesis/slurm/{name}_{arg_str_out}_{start}_{stop}_{ide}.out" \n'
        )
        my_file.write(f"for i in `seq {start} {stop}`; do\n")
        my_file.write(f"\t python {path_code}MscThesis/sample/pipeline/scripts/{code} {arg_str} $i &\n")
        my_file.write("done\n")
        my_file.write("wait\n")
        my_file.close()
        call(f"sbatch {path_job}", shell=True)
        
def check_state(plate,begin,end,file,directory):
    not_exist=[]
    dates_datetime = get_dates_datetime(directory,plate)
    dates_datetime_chosen = dates_datetime[begin:end+1]
    dates = dates_datetime_chosen
    for i,date in enumerate(dates):
        directory_name = get_dirname(date,plate)
        path_snap=directory+directory_name
        stage = os.path.exists(path_snap+file)
        if not stage:
            not_exist.append((date,i+begin))
    return(not_exist)

def make_stitching_loop(directory,dirname,index):
    a_file = open(f'{path_code}MscThesis/sample/pipeline/scripts/stitching_loops/stitching_loop.ijm',"r")

    list_of_lines = a_file.readlines()

    list_of_lines[4] = f'mainDirectory = \u0022{directory}\u0022 ;\n'
    list_of_lines[29] = f'\t if(startsWith(list[i],\u0022{dirname}\u0022)) \u007b\n'
    file_name = f'{path_code}MscThesis/sample/pipeline/scripts/stitching_loops/stitching_loop{index}.ijm'
    a_file = open(file_name, "w")

    a_file.writelines(list_of_lines)

    a_file.close()
    
def run_parallel_stitch(plate, directory, begin, end, num_parallel, time):
    begin_skel = begin // num_parallel
    end_skel = (end) // num_parallel + 1
    listdir = os.listdir(directory)
    list_dir_interest = [name for name in listdir if name.split('_')[-1]==f'Plate{0 if plate<10 else ""}{plate}']
    dates_datetime = get_dates_datetime(directory,plate)
    for j in range(begin_skel, end_skel):
        start = num_parallel * j + begin % num_parallel
        stop = num_parallel * j + num_parallel + begin % num_parallel
        for k in range(start,stop):
            make_stitching_loop(directory,get_dirname(dates_datetime[k], plate),k)
        ide = int(datetime.now().timestamp())
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH -N 1 \n#SBATCH -t {time}\n#SBATCH -p normal\n"
        )
        my_file.write(
            f'#SBATCH -o "{path_code}MscThesis/slurm/stitching__{start}_{stop}_{ide}.out" \n'
        )
        for k in range(start,stop):
            my_file.write(f"~/Fiji.app/ImageJ-linux64 --headless -macro  {path_code}MscThesis/sample/pipeline/scripts/stitching_loops/stitching_loop{k}.ijm &\n")
        my_file.write("wait\n")
        my_file.close()
        call(f"sbatch {path_job}", shell=True)

def find_state(plate,begin,end,directory,include_stitch=True):
    files = [ '/Analysis/skeleton_compressed.mat', '/Analysis/skeleton_masked_compressed.mat',
             '/Analysis/skeleton_pruned_compressed.mat', '/Analysis/transform.mat',
             '/Analysis/skeleton_realigned_compressed.mat']
    if include_stitch:
        files = ['/Img/TileConfiguration.txt.registered'] + files
    not_present2 = check_state(plate,begin+1,end,'/Analysis/transform_corrupt.mat', directory)
    for file in files:
        if file == '/Analysis/transform.mat':
            not_present = check_state(plate,begin+1,end,file,directory)
            to_check = copy(not_present)
            for datetme in to_check:
                if datetme not in not_present2:
                    print(datetme,'alignment failed')
                    not_present.remove(datetme)
        else:
            not_present = check_state(plate,begin,end,file,directory)
        if len(not_present)>0:
            return(file,not_present)
    return("skeletonization is complete")

def find_state_extract(plate,begin,end,directory):
    files = [ '/Analysis/nx_graph_pruned.p', '/Analysis/nx_graph_pruned_width.p','/Analysis/nx_graph_pruned_labeled.p']
    for file in files:
        not_present = check_state(plate,begin,end,file,directory)
        if len(not_present)>0:
            return(file,not_present)
    return("extration is complete")
