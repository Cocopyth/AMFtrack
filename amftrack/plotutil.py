import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import numpy as np
from amftrack.util.sys import get_skeleton


def get_time(exp, t, tp1):  # redefined here to avoid loop in import
    """Return time between date t and date tp1"""
    seconds = (exp.dates[tp1] - exp.dates[t]).total_seconds()
    return seconds / 3600


def show_im(matrix, alpha=None, cmap="gray", interpolation="none"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(matrix, cmap=cmap, interpolation=interpolation, alpha=alpha)
    numrows, numcols = matrix.shape[0], matrix.shape[1]

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = matrix[row, col]
            return "x=%1.4f, y=%1.4f, z=%1.4f" % (y, x, z)
        else:
            return "x=%1.4f, y=%1.4f" % (y, x)

    ax.format_coord = format_coord
    plt.show()


def show_im_rgb(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(matrix, cmap="gray", interpolation="nearest")
    numrows, numcols = matrix.shape[0], matrix.shape[1]

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            return "x=%1.4f, y=%1.4f" % (y, x)
        else:
            return "x=%1.4f, y=%1.4f" % (y, x)

    ax.format_coord = format_coord
    plt.show()


def overlap(skel, raw):
    kernel = np.ones((3, 3), np.uint8)
    #     dilated = cv2.dilate(skel,kernel,iterations = 4)
    fig = plt.figure()
    matrix = skel
    ax = fig.add_subplot(111)
    ax.imshow(raw, cmap="gray", interpolation="none")
    ax.imshow(skel, cmap="jet", alpha=0.5, interpolation="none")
    numrows, numcols = matrix.shape[0], matrix.shape[1]

    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = matrix[row, col]
            return "x=%1.4f, y=%1.4f, z=%1.4f" % (y, x, z)
        else:
            return "x=%1.4f, y=%1.4f" % (y, x)

    ax.format_coord = format_coord
    plt.show()


def plot_nodes(nx_graph, pos, im):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    bbox_props = dict(boxstyle="circle", fc="white")
    for node in nx_graph.nodes:
        t = ax.text(
            pos[node][1],
            pos[node][0],
            str(node),
            ha="center",
            va="center",
            size=5,
            bbox=bbox_props,
        )
    plt.show()


def plot_nodes_from_list(node_list, pos, im):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im)
    bbox_props = dict(boxstyle="circle", fc="white")
    for node in node_list:
        t = ax.text(
            pos[node][1],
            pos[node][0],
            str(node),
            ha="center",
            va="center",
            size=5,
            bbox=bbox_props,
        )
    plt.show()


def plot_t_tp1(
    node_list_t,
    node_list_tp1,
    pos_t,
    pos_tp1,
    imt,
    imtp1,
    relabel_t=lambda x: x,
    relabel_tp1=lambda x: x,
    shift=(0, 0),
    compress=1,
    save="",
    time=None,
    gray=False,
    fontsize=25,
):
    right = 0.90
    top = 0.90
    if len(save) >= 1:
        fig = plt.figure(figsize=(14, 12))
        size = 10
    else:
        fig = plt.figure()
        size = 5
    ax = fig.add_subplot(111)
    ax.imshow(imtp1, cmap="gray", interpolation="none")
    if gray:
        ax.imshow(imt, cmap="gray", alpha=0.5, interpolation="none")
    else:
        ax.imshow(imt, cmap="jet", alpha=0.5, interpolation="none")

    bbox_props1 = dict(boxstyle="circle", fc="grey")
    bbox_props2 = dict(boxstyle="circle", fc="white")
    ax.text(
        right,
        top,
        time,
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        color="white",
        fontsize=fontsize,
    )
    for node in node_list_t:
        t = ax.text(
            (pos_t[node][1] - shift[1]) // compress,
            (pos_t[node][0] - shift[0]) // compress,
            str(relabel_t(node)),
            ha="center",
            va="center",
            size=size,
            bbox=bbox_props1,
        )
    for node in node_list_tp1:
        if node in pos_tp1.keys():
            t = ax.text(
                (pos_tp1[node][1] - shift[1]) // compress,
                (pos_tp1[node][0] - shift[0]) // compress,
                str(relabel_tp1(node)),
                ha="center",
                va="center",
                size=size,
                bbox=bbox_props2,
            )
    if len(save) >= 1:
        plt.savefig(save)
        plt.close(fig)
    else:
        plt.show()


def compress_skeleton(skeleton_doc, factor, shape=None):
    if shape is None:
        shape = skeleton_doc.shape
    final_picture = np.zeros(shape=(shape[0] // factor, shape[1] // factor))
    for pixel in skeleton_doc.keys():
        x = min(round(pixel[0] / factor), shape[0] // factor - 1)
        y = min(round(pixel[1] / factor), shape[1] // factor - 1)
        final_picture[x, y] += 1
    return final_picture


def plot_node_skel(node, t0, ranges=1000, save="", anchor=None):
    t = t0
    exp = node.experiment
    anchor_time = t0 if (anchor is None) else anchor
    center = node.pos(anchor_time)[1], node.pos(anchor_time)[0]
    window = (
        center[0] - ranges,
        center[0] + ranges,
        center[1] - ranges,
        center[1] + ranges,
    )
    skelet, rot, trans = get_skeleton(
        node.experiment, window, t, node.experiment.directory
    )
    #     im_stitched = get_im_stitched(exp,window,t,directory)
    tips = [
        node.label
        for node in exp.nodes
        if t in node.ts()
        and node.degree(t) == 1
        and node.pos(t)[1] >= window[0]
        and node.pos(t)[1] <= window[1]
        and node.pos(t)[0] >= window[2]
        and node.pos(t)[0] <= window[3]
    ]
    junction = [
        node.label
        for node in exp.nodes
        if t in node.ts()
        and node.degree(t) >= 2
        and node.pos(t)[1] >= window[0]
        and node.pos(t)[1] <= window[1]
        and node.pos(t)[0] >= window[2]
        and node.pos(t)[0] <= window[3]
    ]
    _ = plot_t_tp1(
        junction,
        tips,
        exp.positions[t],
        exp.positions[t],
        skelet,
        skelet,
        shift=(window[2], window[0]),
        save=save,
        time=f"t={int(get_time(exp,0,t))}h",
    )
