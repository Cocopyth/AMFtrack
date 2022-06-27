import cv2
import numpy as np

def detect_blobs(im):
    im2 = 255 - im
    im2_bw = im2 >= 200
    kernel_size = (10, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 500

    params.filterByCircularity = True
    params.minCircularity = 0.8
    img = (1 - im2_bw).astype(np.uint8) * 255
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    spores = []
    for keypoint in keypoints:
        x, y = keypoint.pt
        r = keypoint.size//2
        spores.append((x,y,r))
    return(spores)



