from datetime import datetime, timedelta
from subprocess import call
from util import get_path, get_dates_datetime, get_dirname
import os

# directory = "/scratch/shared/AMF914/old/from_cartesius/"
# directory = "/scratch/shared/mrozemul/Fiji.app/"
directory = "/scratch/shared/AMF914/Fiji.app/"


path_job = "/home/cbisot/bash/job.sh"
path_code = "/home/cbisot/pycode/"


def run_parallel(code, args, begin, end, num_parallel, time, name):
    begin_skel = begin // num_parallel
    end_skel = (end) // num_parallel + 1
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    arg_str_out = "_".join(args_str)
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
        my_file.write(f"\t python {path_code}MscThesis/{code} {arg_str} $i &\n")
        my_file.write("done\n")
        my_file.write("wait\n")
        my_file.close()
        call(f"sbatch {path_job}", shell=True)
        
def check_state(plate,begin,end,file):
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

def find_state(plate,begin,end):
    files = ['/Img/TileConfiguration.txt.registered', '/Analysis/skeleton_compressed.mat', '/Analysis/skeleton_masked_compressed.mat',
             '/Analysis/skeleton_pruned_compressed.mat', '/Analysis/transform.mat',
             '/Analysis/skeleton_realigned_compressed.mat']
    for file in files:
        if file == '/Analysis/transform.mat':
            not_present = check_state(plate,begin+1,end,file)
        else:
            not_present = check_state(plate,begin,end,file)
        if len(not_present)>0:
            return(not_present,file)
    return("skeletonization is complete")
