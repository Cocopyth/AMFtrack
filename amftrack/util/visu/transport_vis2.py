import dash
import dash_html_components as html
import dash_leaflet as dl
from dash.dependencies import Input, Output
from jupyter_dash import JupyterDash
from PIL import Image
import os
from IPython.display import clear_output
import re
from amftrack.pipeline.development.high_mag_videos.plot_data import (
    plot_summary,
    save_raw_data,
)
from amftrack.pipeline.functions.transport_processing.high_mag_videos.kymo_class import (
    KymoVideoAnalysis,
    KymoEdgeAnalysis,
)
import os
import pandas as pd
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
from tifffile import imwrite
from tqdm import tqdm
import scipy
import matplotlib as mpl
from pathlib import Path
from amftrack.pipeline.launching.run import (
    run_transfer,
)
from amftrack.pipeline.launching.run_super import run_parallel_transfer
from subprocess import call
import logging
import datetime
import glob
import json
from PIL import Image
import os
from PIL import Image, ImageDraw

analysis_folder = r"C:\Users\coren\AMOLF-SHIMIZU Dropbox\DATA\CocoTransport\Analysis"

img_infos = glob.glob(f"{analysis_folder}/**/video_data_network.json", recursive=True)
vid_anls_frame = pd.DataFrame()
for address in tqdm(img_infos):
    add_info = pd.read_json(address, orient="index").T
    vid_anls_frame = pd.concat([vid_anls_frame, add_info], ignore_index=True)

vid_frame = vid_anls_frame.sort_values("unique_id").reset_index(drop=True)
####################################################################################
### This is where you can apply the filters. Only those videos will be analyzed. ###
####################################################################################
plate_id = "20230813_Plate441"
analysis_frame = vid_frame.loc[vid_frame["plate_id"] == plate_id]
####################################################################################
### Below code will prepare for those videos to be downloaded to videos_folder.  ###
####################################################################################
analysis_frame["plate_int"] = [
    entry.split("_")[-1] for entry in analysis_frame["plate_id"]
]
analysis_frame["video_int"] = [
    entry.split("_")[-1] for entry in analysis_frame["unique_id"]
]
# analysis_frame
video_paths = [
    os.path.join(
        analysis_folder,
        row["plate_id"],
        row["video_int"],
        "Img",
        f"{row['plate_id']}_{row['video_int']}_video.mp4",
    )
    for index, row in analysis_frame.iterrows()
]
video_pos = [
    ((row["ypos_network"]) / 5, (row["xpos_network"]) / 5)
    for index, row in analysis_frame.iterrows()
]
# video_pos = [((row['xpos']+22)*1000/5,(row['ypos']+29)*1000/5) for index,row in analysis_frame.iterrows()]

detected_edges_path = [
    os.path.join(
        analysis_folder, row["plate_id"], row["video_int"], "Img", f"Detected edges.png"
    )
    for index, row in analysis_frame.iterrows()
]
output_path = os.path.join(analysis_folder, plate_id)


def generate_dash_leaflet_app(image_path, video_pos, video_list, assets_folder="asset"):
    app = dash.Dash(__name__, assets_folder=assets_folder)
    image_path_global = image_path
    # Convert the PNG image to JPG and save it in the assets directory
    image_path = os.path.relpath(image_path, assets_folder)
    video_list = [
        "/assets/" + os.path.relpath(video, assets_folder).replace("\\", "/")
        for video in video_list
    ]
    print(video_list[0])
    with Image.open(image_path_global) as img:
        image_width, image_height = img.size
        image_center = [image_height / 2, image_width / 2]
    markers = []
    for i, (x, y) in enumerate(video_pos):
        marker = dl.Marker(position=[image_height - y, x], id=f"marker-{i}")
        markers.append(marker)
    app.layout = html.Div(
        [
            dl.Map(
                children=[
                    dl.ImageOverlay(
                        url=f"/assets/{image_path}",
                        bounds=[[0, 0], [image_height, image_width]],
                    ),
                    dl.LayerGroup(markers),
                ],
                id="map",
                style={"width": "50%", "height": "50vh", "display": "block"},
                crs="Simple",  # Set the CRS to Simple
                center=image_center,
                zoom=-4,  # Initial zoom level
                minZoom=-4,
            ),
            html.Div(
                [
                    html.Video(
                        src=video,
                        controls=True,
                        id=f"video-{i}",
                        style={"width": "50%", "display": "none"},
                        preload=None,
                    )
                    for i, video in enumerate(video_list)
                ]
            ),
        ]
    )

    @app.callback(
        [Output(f"video-{i}", "style") for i in range(len(video_list))],
        [Input(f"marker-{i}", "n_clicks") for i in range(len(video_list))],
    )
    def display_video(*clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        clicked_marker_id = ctx.triggered[0]["prop_id"].split(".")[0]
        marker_index = int(clicked_marker_id.split("-")[1])
        styles = [
            {"display": "block" if i == marker_index else "none", "width": "50%"}
            for i in range(len(video_list))
        ]
        return styles

    app.run_server(debug=False)


image_path = os.path.join(analysis_folder, plate_id, "network_overlay.png")
video_list = video_paths

generate_dash_leaflet_app(image_path, video_pos, video_list, assets_folder=output_path)
