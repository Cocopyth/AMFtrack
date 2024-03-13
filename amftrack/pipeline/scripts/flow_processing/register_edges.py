from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment
from amftrack.pipeline.functions.transport_processing.high_mag_videos.loading import load_video_dataset
from amftrack.pipeline.functions.transport_processing.high_mag_videos.register_videos import register_dataset
from amftrack.util.sys import update_plate_info, get_current_folders

plate_id = "310_20230830"
plate_id_video = "20230903_Plate310"
videos_folder = "/projects/0/einf914/videos/"

analysis_folder = "/projects/0/einf914/analysis_videos/CocoTransport/"
analysis_folder_root = "/projects/0/einf914/analysis_videos/"

data_obj = load_video_dataset(plate_id_video,analysis_folder,analysis_folder_root)

# directory_targ = os.path.join(directory_scratch, "stitch_temp2") + "/"
directory_targ = '/projects/0/einf914/transport/'
update_plate_info(directory_targ, local=True)
all_folders = get_current_folders(directory_targ, local=True)
folders = all_folders.loc[all_folders["unique_id"] == plate_id]
folders = folders.loc[folders["/Analysis/nx_graph_pruned_labeled.p"] == True]
folders = folders.sort_values(by="datetime")

exp = Experiment(directory_targ)
i = 44
exp.load(folders.iloc[i : i + 1], suffix="_width")
for t in range(exp.ts):
    exp.load_tile_information(t)

data_obj.video_objs = sorted(data_obj.video_objs,key = lambda video : video.dataset['video_int'])
register_dataset(data_obj,exp,t)