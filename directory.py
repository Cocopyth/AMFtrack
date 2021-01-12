from datetime import datetime,timedelta
from subprocess import call
# directory = "/scratch/shared/AMF914/old/from_cartesius/" 
# directory = "/scratch/shared/mrozemul/Fiji.app/" 
directory = "/scratch/shared/AMF914/Fiji.app/" 


path_job = "/home/cbisot/bash/job.sh"
path_code = "/home/cbisot/pycode/"
def run_parallel(code,args,begin,end,num_parallel,time,name):
    begin_skel = begin//num_parallel
    end_skel = (end)//num_parallel+1
    args_str = [str(arg) for arg in args]
    arg_str = ' '.join(args_str)
    arg_str_out = '_'.join(args_str)
    for j in range(begin_skel,end_skel):
        start = num_parallel*j+begin%num_parallel
        stop = num_parallel*j+num_parallel-1+begin%num_parallel
        ide=int(datetime.now().timestamp())
        my_file = open(path_job, "w")
        my_file.write(f'#!/bin/bash \n#Set job requirements \n#SBATCH -N 1 \n#SBATCH -t {time}\n#SBATCH -p normal\n')
        my_file.write(f'#SBATCH -o "{path_code}MscThesis/slurm/{name}_{arg_str_out}_{start}_{stop}_{ide}.out" \n')
        my_file.write(f'for i in `seq {start} {stop}`; do\n')
        my_file.write(f'\t python {path_code}MscThesis/{code} {arg_str} $i &\n')
        my_file.write('done\n')
        my_file.write('wait\n')
        my_file.close()
        call(f'sbatch {path_job}', shell=True)