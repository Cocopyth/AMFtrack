import dash
from dash import html
import dash_leaflet as dl
from dash.dependencies import Input, Output, State
import numpy as np
import pandas as pd
from dash import html, dcc
from tqdm import tqdm

import glob
import os
from PIL import Image, ImageDraw
from flask import send_file


def read_video_data(analysis_folder):

    img_infos = glob.glob(f"{analysis_folder}/**/video_data_network.json", recursive=True)
    vid_anls_frame = pd.DataFrame()
    for address in tqdm(img_infos):
        add_info = pd.read_json(address, orient='index').T
        vid_anls_frame = pd.concat([vid_anls_frame, add_info], ignore_index=True)
    return vid_anls_frame.sort_values('unique_id').reset_index(drop=True)



plate_id = "20230813_Plate441"

analysis_folder = r"C:\Users\coren\AMOLF-SHIMIZU Dropbox\DATA\CocoTransport\Analysis"
vid_frame = read_video_data(analysis_folder)
image_path_global = os.path.join(analysis_folder, plate_id, 'network_overlay.png')
image_overlay_url = f"/images/{plate_id}/{os.path.basename(image_path_global)}"


def generate_dash_leaflet_app():
    app = dash.Dash(__name__)

    @app.server.route('/images/<plate_id>/<image_name>')
    def serve_image(plate_id, image_name):
        return send_file(os.path.join(analysis_folder, plate_id, image_name), mimetype='image/png')

    app.layout = html.Div([
        dcc.Store(id="display-state", data={"show_map": False}),  # Using Store to manage display state

        html.Div([
            html.H2("Select unique_id"),
            html.Button("Select", id="select_button"),
        ], id="content-div")  # Setting initial content to the selection screen
    ])

    @app.callback(
        Output("content-div", "children"),
        Input("display-state", "data"),
        prevent_initial_call=True
    )
    def update_display(data):
        if data["show_map"]:
            return html.Div([
                dl.Map(children=dl.ImageOverlay(url=image_overlay_url, bounds=[[0, 0], [5587, 11379]]),
                       id='map',
                       style={'width': '50%', 'height': '50vh', 'display': 'flex'},
                       crs="Simple",
                       zoom=-4,
                       minZoom=-4,
                       center=[5587 / 2, 11379 / 2]),
            ], id="map_screen")
        else:
            return html.Div([
                html.H2("Select unique_id"),
                html.Button("Select", id="select_button"),
            ], id="selection_screen")

    @app.callback(
        Output("display-state", "data"),
        Input("select_button", "n_clicks"),
        prevent_initial_call=True
    )
    def toggle_display(n_clicks):
        return {"show_map": True}
    return(app)

if __name__ == "__main__":
    app = generate_dash_leaflet_app()
    app.run_server(debug=False)
