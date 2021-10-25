from datetime import datetime
from subprocess import call
from amftrack.util import get_dates_datetime, get_dirname
import os
from copy import copy
from time import time_ns

# directory = "/scratch/shared/AMF914/old/from_cartesius/"
# directory = "/scratch/shared/mrozemul/Fiji.app/"
directory_scratch = "/scratch-shared/amftrack/"
directory_project = "/projects/0/einf914/data/"



path_job = "/home/cbisot/bash/job.sh"
path_code = "/home/cbisot/pycode/MscThesis/"
# path_job = r'C:\Users\coren\Documents\PhD\Code\bash\job.sh'
# path_code = r'C:\Users\coren\Documents\PhD\Code\AMFtrack/'

def run_parallel(code, args, folders, num_parallel, time, name):
    op_id = time_ns()
    folders.to_json(f'{directory_scratch}temp/{op_id}.json')# temporary file
    length = len(folders)
    begin_skel = 0
    end_skel = length // num_parallel + 1
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    arg_str_out = "_".join([str(arg) for arg in args if type(arg)!=str])
    for j in range(begin_skel, end_skel):
        start = num_parallel * j
        stop = num_parallel * j + num_parallel - 1
        ide = time_ns()
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH --nodes=1 \n#SBATCH -t {time}\n #SBATCH --ntask=1 \n#SBATCH --cpus-per-task=128\n#SBATCH -p thin \n"
        )
        my_file.write(
            f'#SBATCH -o "{path_code}slurm/{name}_{arg_str_out}_{start}_{stop}_{ide}.out" \n'
        )
        my_file.write(f"source /home/cbisot/miniconda3/etc/profile.d/conda.sh\n")
        my_file.write(f"conda activate amftrack\n")
        my_file.write(f"for i in `seq {start} {stop}`; do\n")
        my_file.write(f"\t python {path_code}amftrack/pipeline/scripts/{code} {arg_str} {op_id} $i &\n")
        my_file.write("done\n")
        my_file.write("wait\n")
        my_file.close()
        call(f"sbatch {path_job}", shell=True)
        
def run_parallel_skelet(low, high, dist, op_id, i):
    op_id = time_ns()
    folders.to_json(f'{directory_scratch}temp/{op_id}.json')# temporary file
    length = len(folders)
    begin_skel = 0
    end_skel = length // num_parallel + 1
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    arg_str_out = "_".join([str(arg) for arg in args if type(arg)!=str])
    for j in range(begin_skel, end_skel):
        start = num_parallel * j
        stop = num_parallel * j + num_parallel - 1
        ide = time_ns()
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH --nodes=1 \n#SBATCH -t {time}\n #SBATCH --ntask=1 \n#SBATCH --cpus-per-task=128\n#SBATCH -p thin \n"
        )
        my_file.write(
            f'#SBATCH -o "{path_code}slurm/{name}_{arg_str_out}_{start}_{stop}_{ide}.out" \n'
        )
        my_file.write(f"source /home/cbisot/miniconda3/etc/profile.d/conda.sh\n")
        my_file.write(f"conda activate amftrack\n")
        my_file.write(f"for i in `seq {start} {stop}`; do\n")
        my_file.write(f"\t python {path_code_dir}/amftrack/pipeline/scripts/extract_skel_indiv.py {low} {high} {dist} {op_id} {k} $i &\n")
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

def make_stitching_loop(directory,dirname,op_id):
    a_file = open(f'{path_code}amftrack/pipeline/scripts/stitching_loops/stitching_loop.ijm',"r")

    list_of_lines = a_file.readlines()

    list_of_lines[4] = f'mainDirectory = \u0022{directory}\u0022 ;\n'
    list_of_lines[29] = f'\t if(startsWith(list[i],\u0022{dirname}\u0022)) \u007b\n'
    file_name = f'{directory_scratch}stitching_loops/stitching_loop{op_id}.ijm'
    a_file = open(file_name, "w")

    a_file.writelines(list_of_lines)

    a_file.close()
    
def run_parallel_stitch(directory, folders, num_parallel, time):
    folder_list = list(folders['folder'])
    folder_list.sort()    
    length = len(folders)
    begin_skel = 0
    end_skel = length // num_parallel + 1
    for j in range(begin_skel, end_skel):
        op_ids = []
        start = num_parallel * j
        stop = num_parallel * j + num_parallel
        for k in range(start,min(stop,len(folder_list))):
            op_id = time_ns()
            make_stitching_loop(directory,folder_list[k],op_id)
            op_ids.append(op_id)
        ide = time_ns()
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH --nodes=1 \n#SBATCH -t {time}\n #SBATCH --ntask=1 \n#SBATCH --cpus-per-task=128\n#SBATCH -p thin \n"
        )
        my_file.write(
            f'#SBATCH -o "{path_code}slurm/stitching__{folder_list[start]}_{ide}.out" \n'
        )
        for k in range(0,min(stop,len(folder_list))-start):
            op_id = op_ids[k]
            my_file.write(f"~/Fiji.app/ImageJ-linux64 --headless -macro  {directory_scratch}stitching_loops/stitching_loop{op_id}.ijm &\n")
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