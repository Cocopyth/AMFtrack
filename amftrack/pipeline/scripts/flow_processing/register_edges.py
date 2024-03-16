from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment
from amftrack.pipeline.functions.transport_processing.high_mag_videos.loading import load_video_dataset
from amftrack.pipeline.functions.transport_processing.high_mag_videos.register_videos import register_dataset
from amftrack.util.sys import update_plate_info, get_current_folders

plate_id = "310_20230830"
indexes = {
"20230901_Plate310" : 20,
"20230902_Plate310" : 33,
"20230903_Plate310" : 42,
"20230904_Plate310" : 52,
"20230905_Plate310" : 64,
"20230906_Plate310" : 73,
}
videos_folder = "/projects/0/einf914/videos/"

analysis_folder = "/projects/0/einf914/analysis_videos/CocoTransport/"
analysis_folder_root = "/projects/0/einf914/analysis_videos/"


# directory_targ = os.path.join(directory_scratch, "stitch_temp2") + "/"
directory_targ = '/projects/0/einf914/transport/'
update_plate_info(directory_targ, local=True)
all_folders = get_current_folders(directory_targ, local=True)
folders = all_folders.loc[all_folders["unique_id"] == plate_id]
folders = folders.loc[folders["/Analysis/nx_graph_pruned_labeled.p"] == True]
folders = folders.sort_values(by="datetime")
for plate_id_video in list(indexes.keys()):
    print(plate_id_video)
    data_obj = load_video_dataset(plate_id_video, videos_folder, analysis_folder, analysis_folder_root)

    exp = Experiment(directory_targ)
    i = indexes[plate_id_video]
    exp.load(folders.iloc[i : i + 1], suffix="_labeled")
    for t in range(exp.ts):
        exp.load_tile_information(t)

    data_obj.video_objs = sorted(data_obj.video_objs,key = lambda video : video.dataset['video_int'])
    register_dataset(data_obj,exp,t)
