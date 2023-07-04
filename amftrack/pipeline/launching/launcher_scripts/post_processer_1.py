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
# time = "10:00"
# list_f = [
#     num_hypha,
#     prop_lost_tracks_junction,
#     prop_lost_tracks_tips,
#     prop_lost_tracks_junction,
#     prop_lost_tracks_tips,
#     prop_inconsistent_root,
#     number_of_timepoints_withing_boundaries,
#     number_of_timepoints,
# ]
# list_args = [None, [1], [1], [10], [10]] + [[]] * len(list_f)
# overwrite = True
# num_parallel = 6
# run_parallel_post(
#     "global_plate_post_process.py",
#     list_f,
#     list_args,
#     [directory, overwrite],
#     analysis_folders,
#     num_parallel,
#     time,
#     "global_plate_post_process",
#     cpus=32,
#     name_job=name_job,
#     node="fat",
# )

time = "6:40:00"
directory = directory
max_ind = 20
incr = 100

list_f = [
    get_num_trunks,
    get_area,
    get_area_separate_connected_components,
    get_num_tips,
    get_num_nodes,
    get_area_study_zone,
    get_num_tips_study_zone,
    get_num_nodes_study_zone,
    get_num_edges,
    get_length_tot,
    get_length_study_zone,
    get_is_out_study,
    get_mean_edge_straight,
    get_spore_volume,
    get_num_spores,
    get_tot_biovolume_study,
    get_tot_biovolume,
]
list_args = [{}] * len(list_f)
overwrite = True
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

time = "24:00:00"
directory = directory
max_ind = 20
incr = 100

fs = [
    get_biovolume_density_in_ring,
    get_density_in_ring,
    # get_density_anastomose_in_ring,
    # get_density_branch_rate_in_ring,
    # get_density_stop_rate_in_ring,
    get_density_active_tips_in_ring,
]
# fs = [get_mean_speed_in_ring]

list_f = []
list_args = []

for f in fs:
    list_f += [f] * max_ind

    list_args += [{"incr": incr, "i": i, "rh_only": True} for i in range(max_ind)]
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
    dependency=True,
)

time = "24:00:00"
list_f = [plot_hulls, plot_tracking, plot_anastomosis]
# list_f = [plot_tracking]
# list_f = [plot_anastomosis]

list_args = [[]] * len(list_f)
overwrite = True
num_parallel = 6
run_parallel_post(
    "exp_plot.py",
    list_f,
    list_args,
    [directory, overwrite],
    analysis_folders,
    num_parallel,
    time,
    "global_plate_post_process",
    cpus=128,
    name_job=name_job,
    node="fat",
)
#
time = "12:00:00"
list_f = [plot_get_hull_nodes]

list_args = [[]] * len(list_f)
overwrite = True
num_parallel = 30
run_parallel_post(
    "exp_plot.py",
    list_f,
    list_args,
    [directory, overwrite],
    analysis_folders,
    num_parallel,
    time,
    "global_plate_post_process",
    cpus=128,
    name_job=name_job,
    node="fat",
)
time = "3:40:00"
directory = directory
# list_f = [get_width_f,get_tot_length_C_f,get_tot_growth_C_f]
# list_f = [get_timestep_anastomosis]
list_f = [
    gets_out_of_ROI,
    get_width_f,
    get_tot_length_C_f,
    get_tot_growth_C_f,
    get_tot_length_pp_f,
    get_tot_growth_pp_f,
    get_timestep_stop_growth,
    get_time_stop_growth,
    get_time_init_growth,
    get_mean_speed_growth,
    get_stop_track,
    get_timestep_anastomosis,
    get_timestep_biological_stop_growth,
]
# # list_f = [get_num_branch]
# # list_f = [gets_out_of_ROI]
list_args = [{}] * len(list_f)
overwrite = True
num_parallel = 32
run_parallel_post(
    "global_hypha_post_process.py",
    list_f,
    list_args,
    [directory, overwrite],
    analysis_folders,
    num_parallel,
    time,
    "global_hypha_post_process",
    cpus=32,
    name_job=name_job,
    node="fat",
)

if stage > 0:
    run_launcher(
        "post_processer_3.py",
        [directory_targ, name_job, stage - 1],
        plates,
        "12:00:00",
        dependency=True,
        name_job=name_job,
    )
    run_launcher(
        "post_processer_2.py",
        [directory_targ, name_job, stage - 1],
        plates,
        "12:00:00",
        dependency=True,
        name_job=name_job,
    )

elif stage == 0:
    run_launcher(
        "dropbox_uploader.py",
        [directory_targ, name_job],
        plates,
        "12:00:00",
        dependency=True,
        name_job=name_job,
    )
