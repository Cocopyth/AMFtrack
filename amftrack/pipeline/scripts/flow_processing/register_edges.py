from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment
from amftrack.pipeline.functions.transport_processing.high_mag_videos.loading import load_video_dataset
from amftrack.pipeline.functions.transport_processing.high_mag_videos.register_videos import register_dataset
from amftrack.util.sys import update_plate_info, get_current_folders
import numpy as np
refs = {
    "310_20230830": {
    "20230901_Plate310" : "20230901_0719_Plate06",
    "20230902_Plate310" : "20230902_1343_Plate07",
    "20230903_Plate310" : "20230903_1143_Plate07",
    "20230904_Plate310" : "20230904_0942_Plate07",
    "20230905_Plate310" : "20230905_1345_Plate07",
    # "20230906_Plate310" : "20230906_1220_Plate07",
    },
    "441_20230807":
            {
    "20230810_Plate441" : "20230810_1005_Plate14",
    "20230811_Plate441" : "20230811_1605_Plate14",
    "20230812_Plate441" : "20230812_1006_Plate14",
    "20230813_Plate441" : "20230813_1618_Plate14",
    },

    "449_20230807":
        {
    "20230813_Plate449" : "20230813_1606_Plate10",
    "20230814_Plate449" : "20230814_1019_Plate10",
    "20230815_Plate449" : "20230815_1021_Plate10",
    "20230816_Plate449" : "20230816_1027_Plate10",
    "20230818_Plate449" : "20230818_1107_Plate10",
    }
}
for plate_id in refs.keys():
    indexes = refs[plate_id]
    analysis_folder = "/projects/0/einf914/analysis_videos/CocoTransport/"
    analysis_folder_root = "/projects/0/einf914/analysis_videos/"
    videos_folder = "/projects/0/einf914/videos/"

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
        # i = indexes[plate_id_video]
        i = np.where(folders['folder'] == indexes[plate_id_video])[0][0]

        exp.load(folders.iloc[i : i + 1], suffix="_labeled")
        for t in range(exp.ts):
            exp.load_tile_information(t)

        data_obj.video_objs = sorted(data_obj.video_objs,key = lambda video : video.dataset['video_int'])
        register_dataset(data_obj,exp,t)
