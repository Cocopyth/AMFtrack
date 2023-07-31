from datetime import datetime
from subprocess import call, DEVNULL, check_output
from typing import List
from amftrack.util.sys import path_code, temp_path, slurm_path, slurm_path_transfer,conda_path
import os
from copy import copy
from time import time_ns
import pickle
import imageio
import sys
from time import sleep
directory_scratch = "/scratch-shared/amftrack/"
directory_project = "/projects/0/einf914/data/"
directory_archive = "/archive/cbisot/"
directory_sun = "/run/user/357100554/gvfs/smb-share:server=sun.amolf.nl,share=shimizu-data,user=bisot/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE/"


if os.getenv("HOME") is not None:
    path_bash = os.getenv("HOME") + "/bash/"
else:
    print("This is not a linux system, I am lost")

path_stitch = f"{temp_path}/stitching_loops/"
if not os.path.isdir(path_stitch):
    os.mkdir(path_stitch)


def get_queue_size():
    return len(check_output(["squeue"]).decode(sys.stdout.encoding).split("\n")) - 2


def call_code(path_job, dependency):
    len_queue = get_queue_size()
    while len_queue > 100:
        sleep(120)
        len_queue = get_queue_size()
    if not dependency:
        call(f"sbatch {path_job}", shell=True)
    else:
        call(f"sbatch --dependency=singleton {path_job}", shell=True)
    sleep(1)


def run_parallel(
    code,
    args,
    folders,
    num_parallel,
    time,
    name,
    cpus=128,
    node="thin",
    dependency=None,
    name_job="job.sh",
):
    path_job = f"{path_bash}{name_job}"
    op_id = time_ns()
    folders.to_json(f"{temp_path}/{op_id}.json")  # temporary file
    length = len(folders)
    begin_skel = 0
    end_skel = length // num_parallel + 1
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    arg_str_out = "_".join([str(arg) for arg in args if type(arg) != str])
    for j in range(begin_skel, end_skel):
        start = num_parallel * j
        if j == end_skel - 1:
            stop = length
        else:
            stop = num_parallel * j + num_parallel - 1
        ide = time_ns()
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH --nodes=1 \n#SBATCH -t {time}\n #SBATCH --ntask=1 \n#SBATCH --cpus-per-task={cpus}\n#SBATCH -p {node} \n"
        )
        my_file.write(
            f'#SBATCH -o "{slurm_path}/{name}_{arg_str_out}_{start}_{stop}_{ide}.out" \n'
        )
        my_file.write(f"source {os.path.join(conda_path,'etc/profile.d/conda.sh')}\n")
        my_file.write(f"conda activate amftrack\n")
        my_file.write(f"for i in `seq {start} {stop}`; do\n")
        my_file.write(
            f"\t python {path_code}pipeline/scripts/image_processing/{code} {arg_str} {op_id} $i &\n"
        )
        my_file.write("done\n")
        my_file.write("wait\n")
        my_file.close()
        call_code(path_job, dependency)


def run_parallel_flows(
    code,
    args,
    folders,
    num_parallel,
    time,
    name,
    cpus=128,
    node="thin",
    dependency=None,
    name_job="job.sh",
):
    path_job = f"{path_bash}{name_job}"
    print(f"{path_job}")
    op_id = time_ns()
    print(f'Sending jobs with id {op_id}')
    folders.to_json(f"{temp_path}/{op_id}.json")  # temporary file
    length = len(folders)
    if length >= num_parallel:
      num_jobs = length // num_parallel
    else:
      num_jobs = length
    print(length)
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    arg_str_out = "_".join([str(arg) for arg in args if type(arg) != str])
    for j in range(num_jobs):
        start = num_parallel * j
        if j == num_jobs - 1:
            stop = length
        else:
            stop = num_parallel * j + num_parallel - 1
        ide = time_ns()
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH --nodes=1 \n#SBATCH -t {time}\n #SBATCH --ntask={num_jobs} \n#SBATCH --cpus-per-task=1\n#SBATCH -p {node} \n"
        )
        my_file.write(
            f'#SBATCH -o "{slurm_path}/{name}_{arg_str_out}_{start}_{stop}_{ide}.out" \n'
        )
        my_file.write(f"module load 2021 \n")
        my_file.write(f"module load Python/3.9.5-GCCcore-10.3.0 \n")
        my_file.write(f"for i in `seq {start} {stop}`; do\n")
        my_file.write(
            f"\t python {path_code}pipeline/scripts/flow_processing/{code} {arg_str} {op_id} $i &\n"
        )
        my_file.write("done\n")
        my_file.write("wait\n")
        my_file.close()
        call_code(path_job, dependency)


def run_parallel_all_time(
    code,
    args,
    folders,
    num_parallel,
    time,
    name,
    cpus=128,
    node="thin",
    dependency=None,
    name_job="job.sh",
):
    path_job = f"{path_bash}{name_job}"
    op_id = time_ns()
    folders.to_json(f"{temp_path}/{op_id}.json")  # temporary file
    plates = set(folders["Plate"].values)
    length = len(plates)
    begin_skel = 0
    end_skel = length // num_parallel + 1
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    arg_str_out = "_".join([str(arg) for arg in args if type(arg) != str])
    for j in range(begin_skel, end_skel):
        start = num_parallel * j
        stop = num_parallel * j + num_parallel - 1
        ide = time_ns()
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH --nodes=1 \n#SBATCH -t {time}\n #SBATCH --ntask=1 \n#SBATCH --cpus-per-task={cpus}\n#SBATCH -p {node} \n"
        )
        my_file.write(
            f'#SBATCH -o "{slurm_path}/{name}_{arg_str_out}_{start}_{stop}_{ide}.out" \n'
        )
        my_file.write(f"source {os.path.join(conda_path,'etc/profile.d/conda.sh')}\n")
        my_file.write(f"conda activate amftrack\n")
        my_file.write(f"for i in `seq {start} {stop}`; do\n")
        my_file.write(
            f"\t python {path_code}pipeline/scripts/image_processing/{code} {arg_str} {op_id} $i &\n"
        )
        my_file.write("done\n")
        my_file.write("wait\n")
        my_file.close()
        call_code(path_job, dependency)


def run_parallel_post(
    code,
    list_f,
    list_args,
    args,
    folders,
    num_parallel,
    time,
    name,
    cpus=128,
    node="thin",
    name_job="post",
    dependency=False,
):
    path_job = f"{path_bash}{name_job}"
    op_id = time_ns()
    folders.to_json(f"{temp_path}/{op_id}.json")  # temporary file
    pickle.dump((list_f, list_args), open(f"{temp_path}/{op_id}.pick", "wb"))
    length = len(folders)
    begin_skel = 0
    end_skel = length // num_parallel + 1
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    arg_str_out = "_".join([str(arg) for arg in args if type(arg) != str])
    for j in range(begin_skel, end_skel):
        start = num_parallel * j
        stop = num_parallel * j + num_parallel - 1
        ide = time_ns()
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH --nodes=1 \n#SBATCH -t {time}\n #SBATCH --ntask=1 \n#SBATCH --cpus-per-task={cpus}\n#SBATCH -p {node} \n"
        )
        my_file.write(
            f'#SBATCH -o "{slurm_path}/{name}_{arg_str_out}_{start}_{stop}_{ide}.out" \n'
        )
        my_file.write(f"source {os.path.join(conda_path,'etc/profile.d/conda.sh')}\n")
        my_file.write(f"conda activate amftrack\n")
        my_file.write(f"for i in `seq {start} {stop}`; do\n")
        my_file.write(
            f"\t python {path_code}pipeline/scripts/post_processing/{code} {arg_str} {op_id} $i &\n"
        )
        my_file.write("done\n")
        my_file.write("wait\n")
        my_file.close()
        call_code(path_job, dependency)


def make_stitching_loop(directory, dirname, op_id, is_mini_PRINCE=False):
    if is_mini_PRINCE:
        a_file = open(
            f"{path_code}pipeline/scripts/stitching_loops/stitching_loop_miniPRINCE.ijm",
            "r",
        )
    else:
        a_file = open(
            f"{path_code}pipeline/scripts/stitching_loops/stitching_loop.ijm", "r"
        )

    list_of_lines = a_file.readlines()

    list_of_lines[4] = f"mainDirectory = \u0022{directory}\u0022 ;\n"
    list_of_lines[29] = f"\t if(startsWith(list[i],\u0022{dirname}\u0022)) \u007b\n"
    file_name = f"{temp_path}/stitching_loops/stitching_loop{op_id}.ijm"
    a_file = open(file_name, "w")

    a_file.writelines(list_of_lines)

    a_file.close()


def run_parallel_stitch(
    directory,
    folders,
    num_parallel,
    time,
    cpus=128,
    node="thin",
    name_job="stitch",
    dependency=False,
    is_mini_PRINCE=False,
):
    folder_list = list(folders["folder"])
    folder_list.sort()
    length = len(folders)
    begin_skel = 0
    end_skel = length // num_parallel + 1
    folder_list = list(folders["folder"])
    folder_list.sort()
    path_job = f"{path_bash}{name_job}"
    size_x = 14 if is_mini_PRINCE else 11
    size_y = 10 if is_mini_PRINCE else 16

    for folder in folder_list:
        path_im_copy = f"{directory}/{folder}/Img/Img_r03_c05.tif"
        if os.path.isfile(path_im_copy) and os.path.getsize(path_im_copy) >= 1e6:
            # im = imageio.imread(path_im_copy)
            for x in range(1, size_x):
                for y in range(1, size_y):
                    strix = str(x) if x >= 10 else f"0{x}"
                    striy = str(y) if y >= 10 else f"0{y}"
                    path = f"{directory}/{folder}/Img/Img_r{strix}_c{striy}.tif"
                    if not os.path.isfile(path):
                        f = open(path, "w")
                    if os.path.getsize(path) == 0:
                        im = imageio.imread(path_im_copy)
                        imageio.imwrite(path, im * 0)
                        print(path)
    for j in range(begin_skel, end_skel):
        op_ids = []
        start = num_parallel * j
        stop = num_parallel * j + num_parallel
        for k in range(start, min(stop, len(folder_list))):
            op_id = time_ns()
            make_stitching_loop(directory, folder_list[k], op_id, is_mini_PRINCE)
            op_ids.append(op_id)
        ide = time_ns()
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH --nodes=1 \n#SBATCH -t {time}\n #SBATCH --ntask=1 \n#SBATCH --cpus-per-task={cpus}\n#SBATCH -p {node} \n"
        )
        my_file.write(
            f'#SBATCH -o "{slurm_path}/stitching__{folder_list[start]}_{ide}.out" \n'
        )
        for k in range(0, min(stop, len(folder_list)) - start):
            op_id = op_ids[k]
            my_file.write(
                f"~/Fiji.app/ImageJ-linux64 --headless -macro  {temp_path}/stitching_loops/stitching_loop{op_id}.ijm &\n"
            )
        my_file.write("wait\n")
        my_file.close()
        call_code(path_job, dependency)


def run_parallel_transfer(
    code,
    args,
    folders,
    num_parallel,
    time,
    name,
    cpus=1,
    node="staging",
    name_job="transfer.sh",
    dependency=False,
):
    path_job = f"{path_bash}{name_job}"
    op_id = time_ns()
    folders.to_json(f"{temp_path}/{op_id}.json")  # temporary file
    length = len(folders)
    begin_skel = 0
    end_skel = length // num_parallel + 1
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    arg_str_out = "_".join([str(arg) for arg in args if type(arg) != str])
    for j in range(begin_skel, end_skel):
        start = num_parallel * j
        stop = num_parallel * j + num_parallel - 1
        ide = time_ns()
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH --nodes=1 \n#SBATCH -t {time}\n #SBATCH --ntask=1 \n#SBATCH --cpus-per-task={cpus}\n#SBATCH -p {node} \n"
        )
        my_file.write(
            f'#SBATCH -o "{slurm_path_transfer}/{name}_{arg_str_out}_{start}_{stop}_{ide}.out" \n'
        )
        if os.path.isfile(os.path.join(conda_path,'envs','amftrack')):
          my_file.write(f"source {os.path.join(conda_path,'etc/profile.d/conda.sh')}\n")
          my_file.write(f"conda activate amftrack\n")
        else:
          my_file.write(f"module load 2021 \n")
          my_file.write(f"module load Python/3.9.5-GCCcore-10.3.0 \n")

        my_file.write(f"for i in `seq {start} {stop}`; do\n")
        my_file.write(
            f"\t python {path_code}transfer/scripts/{code} {arg_str} {op_id} $i \n"
        )
        my_file.write("done\n")
        my_file.write("wait\n")
        my_file.close()
        call_code(path_job, dependency)


def run_launcher(
    code,
    args,
    plates: List[str],
    time,
    cpus=1,
    name="launcher",
    node="staging",
    name_job="one_shot.sh",
    dependency=False,
):
    """
    :param plates: ['1015_20220504',"1023_20220502"]
    """
    path_job = f"{path_bash}{name_job}"
    args_str = [str(arg) for arg in args] + [str(plate) for plate in plates]
    arg_str = " ".join(args_str)
    arg_str_out = "_".join([str(arg) for arg in args if type(arg) != str])
    my_file = open(path_job, "w")
    my_file.write(
        f"#!/bin/bash \n#Set job requirements \n#SBATCH --nodes=1 \n#SBATCH -t {time}\n #SBATCH --ntask=1 \n#SBATCH --cpus-per-task={cpus}\n#SBATCH -p {node} \n"
    )
    my_file.write(f'#SBATCH -o "{slurm_path_transfer}/{name}_{arg_str_out}.out" \n')
    my_file.write(f"source {os.path.join(conda_path, 'etc/profile.d/conda.sh')}\n")
    my_file.write(f"conda activate amftrack\n")
    my_file.write(
        f"python {path_code}pipeline/launching/launcher_scripts/{code} {arg_str}  &\n"
    )
    my_file.write("wait\n")
    my_file.close()
    call_code(path_job, dependency)


def run_parallel_transfer_to_archive(
    folders,
    directory,
    time,
    name,
    cpus=1,
    node="staging",
    dependency="transfer_archive.sh",
):
    path_job = f"{path_bash}{dependency}"
    plates = set(folders["Plate"].values)
    length = len(plates)
    folders = folders.copy()
    folders["unique_id"] = (
        folders["Plate"].astype(str) + "_" + folders["CrossDate"].astype(str)
    )
    folder_list = list(folders["folder"])
    folder_list.sort()
    for directory_name in folder_list:
        path_info = f'{os.getenv("HOME")}/archinfo/{directory_name}_info.json'
        line = folders.loc[folders["folder"] == directory_name]
        line.to_json(path_info)

    for plate in plates:
        id_unique = folders.loc[folders["Plate"] == plate]["unique_id"].iloc[0]
        folder = f"{directory}{id_unique}"
        ide = time_ns()
        my_file = open(path_job, "w")
        my_file.write(
            f"#!/bin/bash \n#Set job requirements \n#SBATCH --nodes=1 \n#SBATCH -t {time}\n #SBATCH --ntask=1 \n#SBATCH --cpus-per-task={cpus}\n#SBATCH -p {node} \n"
        )
        my_file.write(f'#SBATCH -o "{slurm_path}/{name}_{ide}.out" \n')
        my_file.write(f"source {os.path.join(conda_path,'etc/profile.d/conda.sh')}\n")
        my_file.write(f"conda activate amftrack\n")
        my_file.write(
            f"tar cvf /archive/cbisot/prince_data/{id_unique}.tar {folder}*\n"
        )
        my_file.write("wait\n")
        my_file.close()
        call(f"sbatch --dependency=singleton {path_job}", shell=True)

