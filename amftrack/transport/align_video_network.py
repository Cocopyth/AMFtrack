import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets


from amftrack.pipeline.launching.run_super import (
    run_launcher,
    directory_scratch,
    directory_project,
    directory_project,
    run_parallel_stitch,
    run_parallel_transfer,
)
import os
from amftrack.util.sys import (
    get_dates_datetime,
    get_dirname,
    temp_path,
    get_data_info,
    update_plate_info,
    update_analysis_info,
    get_analysis_info,
    get_current_folders,
    get_folders_by_plate_id,
)

from time import time_ns
from amftrack.util.dbx import upload_folders, load_dbx, download, get_dropbox_folders
from datetime import datetime
from amftrack.pipeline.launching.run_super import (
    run_parallel,
    directory_scratch,
    directory_project,
    run_parallel_stitch,
)
from amftrack.util.dbx import read_saved_dropbox_state,get_dropbox_folders
import sys
import os

from amftrack.util.sys import get_dirname, temp_path
import pandas as pd
import ast
from scipy import sparse
import scipy.io as sio
import cv2
import imageio.v2 as imageio
import numpy as np
import scipy.sparse
import os
from time import time
from amftrack.pipeline.functions.image_processing.extract_skel import (
    extract_skel_new_prince,
    run_back_sub,
    bowler_hat,
)

from amftrack.util.sys import get_dates_datetime, get_dirname
import shutil
import matplotlib.pyplot as plt
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    save_graphs,
    load_graphs,
    Edge,
    Node
)
from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_random_edge,
    distance_point_edge,
    plot_edge,
    plot_edge_cropped,
    find_nearest_edge,
    get_edge_from_node_labels,
    plot_full_image_with_features,
    get_all_edges,
    get_all_nodes,
    find_neighboring_edges,
    reconstruct_image,
    reconstruct_skeletton_from_edges,
    reconstruct_skeletton_unicolor,
    reconstruct_image_from_general,
    plot_full,
    plot_edge_color_value
)
from matplotlib import cm
from matplotlib.patches import Rectangle
selected_rectangle = None
def identify_nodes(exp,t):
    vmax = 9
    vmin = 3
    region = None
    nodes = get_all_nodes(exp, t)
    downsizing = 5

    fig, ax = plot_edge_color_value(
        exp,
        t,
        lambda edge: edge.width(t),
        region=region,
        # nodes = nodes,
        cmap=cm.get_cmap("viridis", 100),
        v_min=vmin,
        v_max=vmax,
        plot_cmap=True,
        show_background=True,
        dilation=10,
        figsize=(16, 12),
        alpha=0.3,
        downsizing=downsizing

    )
    # fig,ax = plt.subplots()
    points = np.transpose([(node.pos(t)[1] / downsizing, node.pos(t)[0] / downsizing) for node in nodes])

    scatter_plot = ax.scatter(points[0], points[1], s=5)

    # Variable to store the selected point information
    selected_point_info = {}

    # Create a text box for input
    text_box = widgets.Text(
        value='',
        placeholder='Type point name here',
        description='Point name:',
        disabled=False
    )
    display(text_box)
    dicopoint = {}
    selected_rectangle = None

    def on_text_box_submit(sender):
        selected_point_info[text_box.value] = selected_point
        print(f'Saved point {text_box.value} with coordinates {selected_point}')
        dicopoint[text_box.value] = selected_point
        text_box.value = ''  # clear the text box

    text_box.on_submit(on_text_box_submit)

    def onclick(event):
        global selected_point, selected_rectangle
        distances = (points[0] - event.xdata) ** 2 + (points[1] - event.ydata) ** 2
        closest_point_index = np.argmin(distances)
        selected_point = points[:, closest_point_index]
        print(f"You clicked closest to point at coordinates ({selected_point[0]}, {selected_point[1]})")

        # Draw a rectangle around the selected point, and remove the previous one
        if selected_rectangle is not None:
            selected_rectangle.remove()
        selected_rectangle = Rectangle((selected_point[0] - 2.5, selected_point[1] - 2.5), 5, 5, fill=False,
                                       color='red')
        ax.add_patch(selected_rectangle)
        fig.canvas.draw()

    # Connect the click event with the callback function
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    return cid, fig