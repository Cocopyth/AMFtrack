import sys
from amftrack.util.sys import (
    update_analysis_info,
    get_analysis_info,
)
from amftrack.pipeline.launching.run_super import run_parallel_post
from amftrack.pipeline.functions.post_processing.global_plate import *
from amftrack.pipeline.functions.post_processing.time_plate import *
from amftrack.pipeline.functions.post_processing.global_hypha import *
from amftrack.pipeline.functions.post_processing.area_hulls import *
from amftrack.pipeline.functions.post_processing.P_regions import *

from amftrack.pipeline.launching.run_super import run_parallel, run_launcher
from amftrack.pipeline.functions.post_processing.exp_plot import *

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
plates = sys.argv[4:]
update_analysis_info(directory_targ)
analysis_info = get_analysis_info(directory_targ)
analysis_folders = analysis_info.loc[analysis_info["unique_id"].isin(plates)]
directory = directory_targ
print(len(analysis_folders))


time = "6:40:00"
directory = directory
max_ind = 20
incr = 100

list_f = [
    get_length_density_in_region
]*18
list_args = [{'i': i} for i in range(18)]
overwrite = False
num_parallel = 6
run_parallel_post(
    "time_plate_post_process.py",
    list_f,
    list_args,
    [directory, overwrite],
    analysis_folders,
    num_parallel,
    time,
    "time_plate_post_process",
    cpus=32,
    name_job=name_job,
    node="fat",
)
if stage>0:
    run_launcher(
        "analysis_uploader.py",
        [directory_targ, name_job],
        plates,
        "12:00:00",
        name="upload_analysis",
        dependency=True,
        name_job=name_job,
    )
