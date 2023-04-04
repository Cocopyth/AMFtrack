import cv2
import numpy as np
from time import time_ns
import matplotlib.pyplot as plt
import os
from amftrack.util.sys import temp_path
from amftrack.util.dbx import upload


def detect_blobs(im):
    im2 = 255 - im
    im2_bw = im2 >= 200
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 5
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 50000

    params.filterByCircularity = True
    params.minCircularity = 0.4
    img = (1 - im2_bw).astype(np.uint8) * 255
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    spores = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        r = keypoint.size // 2
        spores.append((x, y, r))
    return spores


def plot_blobs_upload(im):
    im2 = 255 - im
    im2_bw = im2 >= 200
    kernel_size = (10, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    closing = cv2.morphologyEx(im2_bw.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 60000

    params.filterByCircularity = False
    # params.minCircularity = 0.8
    img = (1 - im2_bw).astype(np.uint8) * 255
    # img = im2.astype(np.uint8)
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    # print(keypoints)
    # blank = np.zeros((1, 1))
    # blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    fig, ax = plt.subplots()
    ax.imshow(im)
    for keypoint in keypoints:
        x, y = keypoint.pt
        r = keypoint.size
        if r > 0:
            print(r // 2, np.pi * (r // 2) ** 2, x, y)
            c = plt.Circle(
                (x, y), r // 1.8, color="red", linewidth=1, fill=False, alpha=0.3
            )
            ax.add_patch(c)
    time = time_ns()
    name_save = f"spore{time}.png"
    path = os.path.join(temp_path, name_save)
    plt.savefig(path, dpi=600)
    dir_drop = "/DATA/PRINCE_OVERVIEW/SPORES"
    target_drop = f"{dir_drop}/{name_save}"
    upload(path, target_drop, catch_exception=False)
    os.remove(path)
