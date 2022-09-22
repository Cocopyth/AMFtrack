import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
from time import time_ns
import os
from random import choice
from PIL import Image
from typing import List

from amftrack.util.dbx import upload
from amftrack.util.sys import temp_path
from amftrack.util.aliases import coord
from amftrack.pipeline.functions.image_processing.experiment_util import (
    plot_full_image_with_features,
    get_all_edges,
    get_all_nodes,
    plot_hulls_skelet,
    plot_full,
    reconstruct_image_from_general,
)
from amftrack.pipeline.functions.image_processing.hyphae_id_surf import (
    get_anastomosing_hyphae,
)

from amftrack.pipeline.functions.post_processing.area_hulls import is_in_study_zone
from amftrack.pipeline.functions.post_processing.extract_study_zone import (
    load_study_zone,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    save_graphs,
    load_graphs,
    load_skel,
    Node,
)
from amftrack.pipeline.functions.post_processing.area_hulls import (
    get_regular_hulls_area_fixed,
)
from amftrack.util.geometry import (
    distance_point_pixel_line,
    get_closest_line_opt,
    get_closest_lines,
    format_region,
    intersect_rectangle,
    get_overlap,
    get_bounding_box,
    expand_bounding_box,
    is_in_bounding_box,
    centered_bounding_box,
)


def make_video(
    paths,
    texts,
    resize,
    save_path=None,
    upload_path=None,
    fontScale=3,
    color=(0, 255, 255),
):
    if resize is None:
        imgs = [cv2.imread(path, cv2.IMREAD_COLOR) for path in paths]
    else:
        imgs = [
            cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), resize) for path in paths
        ]
    for i, img in enumerate(imgs):
        anchor = img.shape[0] // 10, img.shape[1] // 10
        cv2.putText(
            img=img,
            text=texts[i],
            org=anchor,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=fontScale,
            color=color,
            thickness=3,
        )
    if not save_path is None:
        imageio.mimsave(save_path, imgs)
    if not upload_path is None:
        if save_path is None:
            time = time_ns()
            save_path_temp = os.path.join(temp_path, f"{time}.mp4")
            imageio.mimsave(save_path_temp, imgs)
        else:
            save_path_temp = save_path
        upload(save_path_temp, upload_path)
        if save_path is None:
            os.remove(save_path_temp)
    return imgs


def make_video_tile(
    paths_list,
    texts,
    resize,
    save_path=None,
    upload_path=None,
    fontScale=3,
    color=(0, 255, 255),
):
    """
    This function makes a video out of a list of list of paths.
    paths_list is of the form [['path1','path2','path3','path4'],['path5','path6','path7','path8']...]
    where 'path1','path2','path3','path4' correspond to the path of images
    of a single timesteps that one want to tile together.
    If odd number they will all be arranged horizontally
    If even number they will be arranged in 2 columns.
    The final images of the movie are returned
    :param texts: a text list of the same dimension as the paths_list to indicate texts
    to be written at the top left of the single image of the tile
    :param resize: a format in which to resize each individual image of the tile
    :save_path: the path where to save the final movie, if None, the final movie is not saved
    :upload_path: the dropbox format path to upload the movie, if None it's not uploaded
    :fontScale: the font of the text
    :color: the color of the text
    """
    if resize is None:
        imgs_list = [
            [cv2.imread(path, cv2.IMREAD_COLOR) for path in paths]
            for paths in paths_list
        ]
    else:
        imgs_list = [
            [cv2.resize(cv2.imread(path, cv2.IMREAD_COLOR), resize) for path in paths]
            for paths in paths_list
        ]
    for i, imgs in enumerate(imgs_list):
        for j, img in enumerate(imgs):
            anchor = img.shape[0] // 10, img.shape[1] // 10
            cv2.putText(
                img=img,
                text=texts[i][j],
                org=anchor,
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=fontScale,
                color=color,
                thickness=3,
            )
    if len(imgs_list[0]) % 2 == 0:
        imgs_final = [
            [cv2.vconcat(imgs[::2]), cv2.vconcat(imgs[1::2])] for imgs in imgs_list
        ]
        imgs_final = [cv2.hconcat(imgs) for imgs in imgs_final]
    elif len(imgs_list[0]) == 1:
        imgs_final = [imgs[0] for imgs in imgs_list]
    else:
        imgs_final = [cv2.hconcat(imgs) for imgs in imgs_list]
    if not save_path is None:
        imageio.mimsave(save_path, imgs_final)
    if not upload_path is None:
        if save_path is None:
            time = time_ns()
            save_path_temp = os.path.join(temp_path, f"{time}.mp4")
            imageio.mimsave(save_path_temp, imgs_final)
        else:
            save_path_temp = save_path
        upload(save_path_temp, upload_path)
        if save_path is None:
            os.remove(save_path_temp)
    return imgs


def make_images_track(exp, num_tiles=4):
    """
    This function makes images centered on the initial position of some random nodes,
    plots the skeleton on top of the raw image, the label of the nodes at different timesteps
    it returns the paths_list of those plotted image in the format for tile video making
    :param exp:
    :param num_tiles: number of such images to tile together
    """
    nodes = get_all_nodes(exp, 0)
    nodes = [
        node
        for node in nodes
        if node.is_in(0)
        and np.linalg.norm(node.pos(0) - node.pos(node.ts()[-1])) > 1000
        and len(node.ts()) > 3
    ]

    paths_list = []
    node_select_list = [choice(nodes) for k in range(num_tiles)]
    for t in range(exp.ts):
        exp.load_tile_information(t)
        paths = []
        for k in range(num_tiles):
            node_select = node_select_list[k]
            pos = node_select.pos(0)
            window = 1500
            region = [
                [pos[0] - window, pos[1] - window],
                [pos[0] + window, pos[1] + window],
            ]
            path = f"plot_nodes_{time_ns()}"
            path = os.path.join(temp_path, path)
            paths.append(path + ".png")
            plot_full_image_with_features(
                exp,
                t,
                region=region,
                downsizing=5,
                nodes=[
                    node
                    for node in get_all_nodes(exp, 0)
                    if node.is_in(t) and np.linalg.norm(node.pos(t) - pos) <= window
                ],
                edges=get_all_edges(exp, t),
                dilation=5,
                prettify=False,
                save_path=path,
            )
        paths_list.append(paths)
    return paths_list


def make_images_track(exp, num_tiles=4):
    """
    This function makes images centered on the initial position of some random nodes,
    plots the skeleton on top of the raw image, the label of the nodes at different timesteps
    it returns the paths_list of those plotted image in the format for tile video making
    :param exp:
    :param num_tiles: number of such images to tile together
    """
    nodes = get_all_nodes(exp, 0)
    nodes = [
        node
        for node in nodes
        if node.is_in(0)
        and np.linalg.norm(node.pos(0) - node.pos(node.ts()[-1])) > 1000
        and len(node.ts()) > 3
    ]

    paths_list = []
    node_select_list = [choice(nodes) for k in range(num_tiles)]
    for t in range(exp.ts):
        exp.load_tile_information(t)
        paths = []
        for k in range(num_tiles):
            node_select = node_select_list[k]
            pos = node_select.pos(0)
            window = 1500
            region = centered_bounding_box(pos, size=3000)
            path = f"plot_nodes_{time_ns()}"
            path = os.path.join(temp_path, path)
            paths.append(path + ".png")
            plot_full(
                exp,
                t,
                region=region,
                downsizing=5,
                nodes=[
                    node
                    for node in get_all_nodes(exp, 0)
                    if node.is_in(t) and np.linalg.norm(node.pos(t) - pos) <= window
                ],
                edges=get_all_edges(exp, t),
                dilation=5,
                prettify=False,
                save_path=path,
            )
        paths_list.append(paths)
    return paths_list


def make_images_track2(exp):
    """
    This function makes images centered on the initial position of some random nodes,
    plots the skeleton on top of the raw image, the label of the nodes at different timesteps
    it returns the paths_list of those plotted image in the format for tile video making
    :param exp:
    :param num_tiles: number of such images to tile together
    """
    t0 = choice(range(exp.ts))
    nodes = get_all_nodes(exp, t0)
    node_select = choice(nodes)
    paths = []
    for t in range(exp.ts):
        exp.load_tile_information(t)
        pos = node_select.pos(t0)
        window = 600
        region = centered_bounding_box(pos, size=int(1.5 * window))
        path = f"plot_nodes_{time_ns()}"
        path = os.path.join(temp_path, path)
        paths.append([path + ".png"])
        plot_full(
            exp,
            t,
            region=region,
            downsizing=1,
            nodes=[
                node
                for node in get_all_nodes(exp, t)
                if node.is_in(t) and np.linalg.norm(node.pos(t) - pos) <= window
            ],
            edges=get_all_edges(exp, t),
            dilation=4,
            prettify=False,
            save_path=path,
        )
    return paths


def make_images_track3(exp):
    """
    This function makes images centered on the initial position of some random nodes,
    plots the skeleton on top of the raw image, the label of the nodes at different timesteps
    it returns the paths_list of those plotted image in the format for tile video making
    :param exp:
    :param num_tiles: number of such images to tile together
    """
    paths = []
    ends = [hypha.end for hypha in exp.hyphaes if len(hypha.end.ts()) > 0]
    to_choose = [node for node in ends if node.ts()[0] < 100 and len(node.ts()) > 4]
    node_select = choice(to_choose)
    t0 = node_select.ts()[-1]
    pos = node_select.pos(t0)
    for t in range(exp.ts):
        # for t in node_select.ts():
        window = 600
        region = centered_bounding_box(pos, size=int(1.5 * window))

        nodes = [
            node
            for node in ends
            if node.is_in(t) and np.linalg.norm(node.pos(t) - pos) <= window
        ]

        if len(nodes) > 0:
            exp.load_tile_information(t)
            path = f"plot_nodes_{time_ns()}"
            path = os.path.join(temp_path, path)
            paths.append([path + ".png"])
            plot_full(
                exp,
                t,
                region=region,
                downsizing=1,
                nodes=nodes,
                edges=get_all_edges(exp, t),
                dilation=4,
                prettify=False,
                save_path=path,
            )
    return paths


def make_images_track4(exp):
    """
    This function makes images centered on the initial position of some random nodes,
    plots the skeleton on top of the raw image, the label of the nodes at different timesteps
    it returns the paths_list of those plotted image in the format for tile video making
    :param exp:
    :param num_tiles: number of such images to tile together
    """
    paths = []
    anastomosing_hyphae = get_anastomosing_hyphae(exp)
    ends = [result[0].end for result in anastomosing_hyphae]
    # ends = [hypha.end for hypha in exp.hyphaes]

    node_select = choice(ends)
    t0 = node_select.ts()[-1]
    pos = node_select.pos(t0)
    for t in range(exp.ts):
        exp.load_tile_information(t)
        window = 600
        region = centered_bounding_box(pos, size=int(1.5 * window))
        path = f"plot_nodes_{time_ns()}"
        path = os.path.join(temp_path, path)
        paths.append([path + ".png"])
        plot_full(
            exp,
            t,
            region=region,
            downsizing=1,
            nodes=[
                node
                for node in ends
                if node.is_in(t) and np.linalg.norm(node.pos(t) - pos) <= window
            ],
            edges=get_all_edges(exp, t),
            dilation=4,
            prettify=False,
            save_path=path,
        )
    return paths


def make_images_track_hypha(exp, hypha):
    """
    This function makes images centered on the initial position of some random nodes,
    plots the skeleton on top of the raw image, the label of the nodes at different timesteps
    it returns the paths_list of those plotted image in the format for tile video making
    :param exp:
    :param num_tiles: number of such images to tile together
    """
    paths = []
    node_select = Node(hypha, exp)
    for t in node_select.ts():
        pos = node_select.pos(t)
        exp.load_tile_information(t)
        window = 600
        region = centered_bounding_box(pos, size=int(1.5 * window))
        path = f"plot_nodes_{time_ns()}"
        path = os.path.join(temp_path, path)
        paths.append([path + ".png"])
        plot_full(
            exp,
            t,
            region=region,
            downsizing=1,
            nodes=[node_select],
            edges=get_all_edges(exp, t),
            dilation=4,
            prettify=False,
            save_path=path,
        )
    return paths


def make_video_targeted(
    exp: Experiment, coordinate: coord, directory_path: str, size=400
) -> List[str]:
    """
    Given a `coordinate` in the GENERAL referential, extract images centered in this
    location at each timestep of `exp` and store the images in the `directory_path`.
    :param coordinate: [1233, 3243]
    :param size: size of the squared image around the location
    :return: list of paths to the generated images
    :WARNING: the plate must have been already processed and loaded
    """
    path_list = []
    region = centered_bounding_box(coordinate, size=size)
    format = "%Y%m%d-%H%M"
    for t in range(exp.ts):
        exp.load_tile_information(t)
        name = (
            f"{exp.dates[t].strftime(format)}.png"  # TODO(FK): get timestep name here
        )
        path = os.path.join(directory_path, name)
        path_list.append(path)
        image, _ = reconstruct_image_from_general(
            exp, t, region=region, downsizing=1, prettify=False
        )
        im_pil = Image.fromarray(image)
        im_pil.save(path)
    return path_list


def make_hull_images(exp, ts_plot):
    """
    TODO
    """
    ts = range(exp.ts)
    incr = 100
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, ts, incr)
    paths_list = []
    for t in ts_plot:
        path = f"plot_nodes_{time_ns()}.png"
        path = os.path.join(temp_path, path)
        plot_hulls_skelet(exp, t, regular_hulls, save_path=path)
        paths_list.append([path])
    return paths_list


def make_anastomosis_images(exp, t0):
    """
    TODO
    """
    paths_list = []

    nodes = [
        node
        for node in exp.nodes
        if node.is_in(t0) and np.all(is_in_study_zone(node, t0, 1000, 200))
    ]
    tips = [
        node
        for node in nodes
        if node.degree(t0) == 1 and node.is_in(t0 + 1) and len(node.ts()) > 2
    ]
    growing_tips = [
        node
        for node in tips
        if np.linalg.norm(node.pos(t0) - node.pos(node.ts()[-1])) >= 40
    ]
    growing_rhs = [
        node
        for node in growing_tips
        if np.linalg.norm(node.pos(node.ts()[0]) - node.pos(node.ts()[-1])) >= 1500
    ]

    anas_tips = [
        tip
        for tip in growing_rhs
        if tip.degree(t0) == 1
        and tip.degree(t0 + 1) == 3
        and 1 not in [tip.degree(t) for t in [tau for tau in tip.ts() if tau > t]]
    ]
    if len(anas_tips) > 0:
        ts = [t0, t0 + 1, t0 + 3]
        for t in ts:
            exp.load_tile_information(t)
        node = choice(anas_tips)
        pos = node.pos(t0)
        region = centered_bounding_box(pos, size=3000)
        path = f"plot_anas_{time_ns()}.png"
        path = os.path.join(temp_path, path)
        for t in ts:

            # region = centered_bounding_box(pos, size=3000)
            plot_full(
                exp,
                t,
                region=region,
                downsizing=5,
                nodes=[node],
                edges=get_all_edges(exp, t),
                dilation=5,
                prettify=False,
                save_path=path,
            )
