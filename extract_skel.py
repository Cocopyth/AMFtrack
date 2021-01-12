import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi
from skimage.morphology import thin
from skimage import data, filters
from util import get_path


def extract_skeleton(date, plate, row, column):
    im = imageio.imread(get_path(date, plate, False, row, column))
    im_cropped = im
    im_blurred = cv2.blur(im_cropped, (200, 200))
    im_back_rem = im_cropped / (im_blurred + 1)
    im_back_rem = cv2.normalize(im_back_rem, None, 0, 255, cv2.NORM_MINMAX)
    im_back_rem_inv = cv2.normalize(255 - im_back_rem, None, 0, 255, cv2.NORM_MINMAX)
    frangised = frangi(im_back_rem, sigmas=range(1, 20, 4))
    frangised = cv2.normalize(frangised, None, 0, 255, cv2.NORM_MINMAX)
    transformed = cv2.normalize(frangised - im_back_rem, None, 0, 255, cv2.NORM_MINMAX)
    low = 100
    high = 200
    lowt = (transformed > low).astype(int)
    hight = (transformed > high).astype(int)
    hyst = filters.apply_hysteresis_threshold(transformed, low, high)
    kernel = np.ones((3, 3), np.uint8)
    for i in range(3):
        dilation = cv2.dilate(hyst.astype(np.uint8) * 255, kernel, iterations=1)
        dilation = cv2.erode(dilated2.astype(np.uint8) * 255, kernel, iterations=1)
    dilated = dilation > 0
    skeletonized = thin(dilated)
    np.save(f"Data/imbackrem_{date}_{plate}_{row}_{column}", im_back_rem)
    np.save(f"Data/frangised_{date}_{plate}_{row}_{column}", frangised)
    np.save(f"Data/dilated_{date}_{plate}_{row}_{column}", dilated)
    np.save(f"Data/skeletonized_{date}_{plate}_{row}_{column}", skeletonized)
    return skeletonized


def extract_width(date, plate, row, column, size=10, width_factor=60):
    threshold = imtab.mean() - 20
    width_doc = sparse.dok_matrix(imtab.shape, dtype=np.float32)
    problem_doc = sparse.dok_matrix(imtab.shape, dtype=np.float32)
    for index, row in graph_tab.iterrows():
        pixel_list_ex = row["pixel_list"]
        for index in range(len(pixel_list_ex)):
            sub_list = pixel_list_ex[max(0, index - size) : index + size]
            orientation = np.array(sub_list[0]) - np.array(sub_list[-1])
            perpendicular = (
                [1, -orientation[0] / orientation[1]] if orientation[1] != 0 else [0, 1]
            )
            perpendicular_norm = np.array(perpendicular) / np.sqrt(
                perpendicular[0] ** 2 + perpendicular[1] ** 2
            )
            pivot = pixel_list_ex[index]
            point1 = np.around(np.array(pivot) + width_factor * perpendicular_norm)
            point2 = np.around(np.array(pivot) - width_factor * perpendicular_norm)
            point1 = point1.astype(int)
            point2 = point2.astype(int)
            p = profile_line(imtab, point1, point2, mode="constant")
            problem = False
            arg = len(p) // 2
            we_plot = randrange(1000)
            while p[arg] <= threshold:
                if arg <= 0:
                    #             we_plot=50
                    problem = True
                    break
                arg -= 1
            begin = arg
            arg = len(p) // 2
            while p[arg] <= threshold:
                if arg >= len(p) - 1:
                    #             we_plot=50
                    problem = True
                    break
                arg += 1
            end = arg
            width_doc[pivot] = math.dist(point1, point2) * (end - begin) / len(p)
            if problem:
                problem_doc[pivot] = True
    #                 print(pixel_list_ex[index])
    #                 print(point1,point2)
    #                 plt.plot(p)
    #                 plt.axvline(x=begin,color ="red")
    #                 plt.axvline(x=end,color="red")
    #                 plt.show()
    values = list(width_doc.values())
    mean = np.mean(values)
    std = np.std(values)
    mini = min(values)
    maxi = max(values)
    width_doc_normalised = sparse.dok_matrix(imtab.shape, dtype=np.float32)
    for key in width_doc.keys():
        width_doc_normalised[key] = (width_doc[key] - mini) / (maxi - mini) * 255
    return (width_doc_normalised, width_doc, problem_doc, mini, maxi)
