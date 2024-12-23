import dash
from dash import html
import dash_leaflet as dl
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
from dash import html, dcc
import dash_bootstrap_components as dbc
from tqdm import tqdm

import glob
import os
from PIL import Image, ImageDraw
from flask import send_file


def read_video_data(analysis_folder):
    img_infos = glob.glob(
        f"{analysis_folder}/**/video_data_network.json", recursive=True
    )
    vid_anls_frame = pd.DataFrame()
    for address in tqdm(img_infos):
        add_info = pd.read_json(address, orient="index").T
        vid_anls_frame = pd.concat([vid_anls_frame, add_info], ignore_index=True)
    return vid_anls_frame.sort_values("unique_id").reset_index(drop=True)


def load_data(vid_frame, plate_id):
    analysis_frame = vid_frame
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
    analysis_frame = analysis_frame.fillna(0)
    video_pos = [
        ((row["ypos_network"]) / 5, (row["xpos_network"]) / 5)
        for index, row in analysis_frame.iterrows()
    ]
    # video_pos = [((row['xpos']+22)*1000/5,(row['ypos']+29)*1000/5) for index,row in analysis_frame.iterrows()]

    modes = [(row["mode"]) for index, row in analysis_frame.iterrows()]
    displays = [
        (row["plate_id"] == plate_id) for index, row in analysis_frame.iterrows()
    ]
    image_paths = [
        os.path.join(
            analysis_folder,
            row["plate_id"],
            row["video_int"],
            "Img",
            "Detected edges.png",
        )
        for index, row in analysis_frame.iterrows()
    ]
    print(image_paths)
    return video_pos, video_paths, modes, displays, image_paths


def adjust_position(video_pos, modes):
    adjusted_positions = []
    for i, (x, y) in enumerate(video_pos):
        OFFSET = np.random.random() * 10
        if modes[i] == "F":
            x += OFFSET
            y += OFFSET
        adjusted_positions.append((x, y))
    return adjusted_positions


def generate_map_children(plate_id):
    video_pos, video_list, modes, displays, image_paths = load_data(vid_frame, plate_id)
    video_pos = adjust_position(video_pos, modes)

    # Your existing logic to set up markers, circles, image overlay, etc.
    image_path_global = os.path.join(analysis_folder, plate_id, "network_overlay.png")
    with Image.open(image_path_global) as img:
        image_width, image_height = img.size
    markers = []
    circles = []
    not_shown = []
    for i, (x, y) in enumerate(video_pos):
        display = displays[i]
        color = (
            "#3498db" if modes[i] == "F" else "#e74c3c"
        )  # These are shades of blue and red.
        if display:
            circle = dl.Circle(
                center=[image_height - y, x],
                radius=5,
                color=color,
                fillColor=color,
                fillOpacity=1.0,
            )
            marker = dl.Marker(position=[image_height - y, x], id=f"marker-{i}")
            markers.append(marker)
            circles.append(circle)
        else:
            not_shown.append(html.Div([], id=f"marker-{i}", style={"display": "none"}))
    image_overlay_url = f"/images/{plate_id}/{os.path.basename(image_path_global)}"

    return [
        dl.ImageOverlay(
            url=image_overlay_url, bounds=[[0, 0], [image_height, image_width]]
        ),
        dl.LayerGroup(markers + circles),
    ], not_shown


def generate_dash_leaflet_app(vid_frame):
    app = dash.Dash(__name__)

    @app.server.route("/videos/<plate_id>/<video_name>")
    def serve_video(plate_id, video_name):
        print(video_name)
        video_path = os.path.join(
            analysis_folder, plate_id, video_name.split("_")[-2], "Img", video_name
        )
        return send_file(video_path, mimetype="video/mp4")

    @app.server.route("/images/<plate_id>/<image_name>")
    def serve_image(plate_id, image_name):
        return send_file(
            os.path.join(analysis_folder, plate_id, image_name), mimetype="image.png"
        )

    @app.server.route("/images_edges/<plate_id>/<video_name>")
    def serve_image_edge(plate_id, video_name):
        return send_file(
            os.path.join(
                analysis_folder,
                plate_id,
                video_name.split("_")[-2],
                "Img",
                "arrows.png",
            ),
            mimetype="image/png",
        )

    @app.server.route("/images_edges2/<plate_id>/<video_name>")
    def serve_image_edge_flux(plate_id, video_name):
        return send_file(
            os.path.join(
                analysis_folder, plate_id, video_name.split("_")[-2], "Img", "flux.png"
            ),
            mimetype="image/png",
        )

    app.layout = html.Div(
        [
            dcc.Store(id="display-state", data={"show_map": False}),
            # Screen 1: Selection screen
            html.Div(
                [
                    html.H2("Select unique_id"),
                    dcc.Dropdown(
                        id="unique_id_dropdown",
                        options=[
                            {"label": uid, "value": uid}
                            for uid in vid_frame["plate_id"].unique()
                        ],
                    ),
                    html.Button("Select", id="select_button"),
                ],
                id="content-div",
                style={"display": "block"},
            ),
            dcc.Store(id="video-data-store", data={}),
        ]
    )

    @app.callback(
        [
            Output("video-data-store", "data"),
            Output("content-div", "children"),
        ],
        [
            State("unique_id_dropdown", "value"),
            Input("display-state", "data"),  # Add this line
        ],
    )
    def toggle_screen(selected_id, display_data):
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        # Use the display_data to decide if we should show the map or the selection screen
        show_map = display_data["show_map"]

        # If the "Select" button was clicked and a plate_id was selected from the dropdown, show the map
        if show_map and selected_id:
            print(f"Selected ID: {selected_id}")
            map_children, not_shown = generate_map_children(selected_id)
            image_path_global = os.path.join(
                analysis_folder, selected_id, "network_overlay.png"
            )
            with Image.open(image_path_global) as img:
                image_width, image_height = img.size
                image_center = [image_height / 2, image_width / 2]
            video_pos, video_list, modes, displays, image_paths = load_data(
                vid_frame, selected_id
            )
            video_data = {
                "video_list": video_list,
                "selected_id": selected_id,
                "image_paths": image_paths,
            }
            video_components = [
                html.Video(
                    id=f"video",
                    controls=True,
                    style={"width": "50%", "display": "none"},
                    preload=None,
                )
            ]
            image_components = [
                html.Img(
                    id=f"image",
                    style={"max-width": "100%", "height": "auto", "display": "none"},
                )
            ]
            image_components += [
                html.Img(
                    id=f"image2",
                    style={"max-width": "100%", "height": "auto", "display": "none"},
                )
            ]

            childr = [
                html.Div(
                    [
                        html.Div(
                            [
                                dl.Map(
                                    children=map_children,
                                    id="map",
                                    style={"width": "100%", "height": "100%"},
                                    crs="Simple",
                                    zoom=-4,
                                    minZoom=-4,
                                    center=image_center,
                                ),
                            ],
                            style={"width": "50%", "height": "100%"},
                        ),
                        html.Div(
                            id="video-container",
                            children=video_components,
                            style={"width": "50%", "height": "100%"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "flex-direction": "row",
                        "height": "60vh",
                        "width": "100%",
                    },
                ),
                html.Div(
                    id="image-container",
                    children=image_components,
                    style={
                        "height": "40vh",
                        "display": "flex",
                        "align-items": "flex-start",
                        "justify-content": "flex-start",
                    },
                ),
                html.Button("Go Back", id="select_button", style={"display": "block"}),
                dcc.Dropdown(
                    id="unique_id_dropdown",
                    options=[
                        {"label": uid, "value": uid}
                        for uid in vid_frame["plate_id"].unique()
                    ],
                    style={"display": "none"},
                ),
                html.Div(not_shown, style={"display": "none"}),
            ]

            childr = html.Div(
                childr,
                id="map_screen",
                style={"display": "flex", "flex-direction": "column"},
            )
            return video_data, childr
        else:
            childr = html.Div(
                [
                    html.H2("Select unique_id"),
                    dcc.Dropdown(
                        id="unique_id_dropdown",
                        options=[
                            {"label": uid, "value": uid}
                            for uid in vid_frame["plate_id"].unique()
                        ],
                    ),
                    html.Button("Select", id="select_button"),
                ],
                id="content-div",
                style={"display": "flex", "flex-direction": "column"},
            )
            return {"video_list": [], "selected_id": None, "image_paths": []}, childr

    @app.callback(
        Output("display-state", "data"),
        Input("select_button", "n_clicks"),
        State("display-state", "data"),
        prevent_initial_call=True,
    )
    def toggle_display(n_clicks, display_data):
        print(display_data)
        display_data["show_map"] = 1 - display_data["show_map"]
        return display_data

    @app.callback(
        [Output(f"video", "style")]
        + [Output(f"image", "style")]
        + [Output(f"image2", "style")]
        + [Output(f"video", "src")]
        + [Output(f"image", "src")]
        + [Output(f"image2", "src")],
        [
            Input(f"marker-{i}", "n_clicks")
            for i in range(len(vid_frame["unique_id"].unique()))
        ],
        [State("video-data-store", "data")],
    )
    def display_video(*args):
        data = args[-1]
        if not data or "video_list" not in data:
            raise dash.exceptions.PreventUpdate
        ctx = dash.callback_context
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate

        video_list = data["video_list"]
        image_paths = data["image_paths"]

        clicked_marker_id = ctx.triggered[0]["prop_id"].split(".")[0]
        marker_index = int(clicked_marker_id.split("-")[1])
        styles = [{"display": "flex", "height": "60vh", "width": "auto"}]
        styles2 = [{"display": "flex", "height": "40vh", "width": "auto"}]

        # Setting src for the video that was clicked.
        sources = [
            f"/videos/{data['selected_id']}/{os.path.basename(video_list[marker_index])}"
        ]
        image_sources = [
            f"/images_edges/{data['selected_id']}/{os.path.basename(video_list[marker_index])}"
        ]
        image_sources2 = [
            f"/images_edges2/{data['selected_id']}/{os.path.basename(video_list[marker_index])}"
        ]
        return styles + styles2 + styles2 + sources + image_sources + image_sources2

    return app


if __name__ == "__main__":
    analysis_folder = (
        r"C:\Users\coren\AMOLF-SHIMIZU Dropbox\DATA\CocoTransport\KymoSpeeDExtract"
    )
    vid_frame = read_video_data(analysis_folder)
    app = generate_dash_leaflet_app(vid_frame)
    app.run_server(debug=False)
