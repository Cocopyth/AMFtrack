from PIL import Image, ImageDraw, ImageFont

from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Edge,
    Node,
)
from amftrack.pipeline.functions.image_processing.extract_width_fun import (
    extract_section_profiles_for_edge_exp,
)
from amftrack.pipeline.functions.transport_processing.high_mag_videos.register_videos import (
    check_hasedges,
)
import os
import numpy as np
import imageio.v2 as imageio

font_size = 24  # Change this to your desired font size
font = ImageFont.truetype(
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf", font_size
)  # 'arial.ttf' might need to be adjusted based on your system


def make_images(data_obj, videos_folder):
    for vid_obj in data_obj.video_objs:
        if vid_obj.dataset["mode"] == "BF":
            path_images = os.path.join(videos_folder, vid_obj.dataset["folder"])
            images = os.listdir(path_images)
            images = sorted(images)
            first_frame = imageio.imread(os.path.join(path_images, images[0]))
            first_frame_img = Image.fromarray(first_frame)
            draw = ImageDraw.Draw(first_frame_img)

            # Optionally, specify a font for the text
            # font = ImageFont.truetype("path_to_font.ttf", font_size)

            for edge in vid_obj.edge_objs:
                begin = edge.edge_name.split(",")[0][1:]
                end = edge.edge_name.split(",")[1][1:-1]
                ypos_1 = edge.edge_infos["edge_ypos_1"]
                xpos_1 = edge.edge_infos["edge_xpos_1"]
                ypos_2 = edge.edge_infos["edge_ypos_2"]
                xpos_2 = edge.edge_infos["edge_xpos_2"]

                # Calculate positions for the start and end texts
                start_pos = (0.9 * ypos_1 + 0.1 * ypos_2, 0.9 * xpos_1 + 0.1 * xpos_2)
                end_pos = (0.9 * ypos_2 + 0.1 * ypos_1, 0.9 * xpos_2 + 0.1 * xpos_1)

                # Draw the start and end texts on the image
                draw.text(
                    start_pos, str(begin), fill="white", font=font
                )  # , font=font) if using a custom font
                draw.text(
                    end_pos, str(end), fill="white", font=font
                )  # , font=font) if using a custom font

            # Save the modified image or display it
            unique_id = vid_obj.dataset["unique_id"]
            path = f"/scratch-shared/amftrack/ML_dataset/{unique_id}.png"
            first_frame_img.save(path)


def make_profile(data_obj, exp, t):
    for index, vid_obj in enumerate(data_obj.video_objs):
        if check_hasedges(vid_obj) and vid_obj.dataset["mode"] == "BF":
            for edge in vid_obj.edge_objs:
                if "network_begin" in edge.mean_data.keys():
                    edge_begin = int(edge.mean_data["network_begin"])
                    edge_end = int(edge.mean_data["network_end"])
                    network_edge = Edge(Node(edge_begin, exp), Node(edge_end, exp), exp)
                    (
                        profiles,
                        transects,
                        new_section_coord_list,
                    ) = extract_section_profiles_for_edge_exp(
                        exp,
                        t,
                        network_edge,
                        resolution=5,
                        offset=10,
                        step=3,
                        target_length=120,
                    )
                    unique_id = vid_obj.dataset["unique_id"]

                    os.makedirs(
                        f"/scratch-shared/amftrack/ML_dataset/{unique_id}/",
                        exist_ok=True,
                    )
                    for index in [5, 10, 15]:
                        if index < len(profiles):
                            path = f"/scratch-shared/amftrack/ML_dataset/{unique_id}/{edge.edge_name}_{index}.npy"
                            np.save(path, profiles[index])
