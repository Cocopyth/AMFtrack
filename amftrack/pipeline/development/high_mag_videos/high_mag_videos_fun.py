import os
import imageio
import matplotlib.pyplot as plt
import cv2

from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    clean_degree_4,
)
import scipy
from amftrack.pipeline.functions.image_processing.node_id import remove_spurs
from amftrack.pipeline.functions.image_processing.extract_skel import remove_component, remove_holes
import numpy as np
from amftrack.pipeline.functions.image_processing.extract_width_fun import generate_pivot_indexes, compute_section_coordinates
from skimage.measure import profile_line
from amftrack.pipeline.functions.image_processing.experiment_class_surf import orient

def get_length_um_edge(edge, nx_graph,space_pixel_size):
    pixel_conversion_factor = space_pixel_size
    length_edge = 0
    pixels = nx_graph.get_edge_data(*edge)['pixel_list']
    for i in range(len(pixels) // 10 + 1):
        if i * 10 <= len(pixels) - 1:
            length_edge += np.linalg.norm(
                np.array(pixels[i * 10])
                - np.array(pixels[min((i + 1) * 10, len(pixels) - 1)])
            )
    #             length_edge+=np.linalg.norm(np.array(pixels[len(pixels)//10-1*10-1])-np.array(pixels[-1]))
    return length_edge * pixel_conversion_factor

def calcGST(inputIMG, w):
    img = inputIMG.astype(np.float32)
    imgDiffX = cv2.Sobel(img, cv2.CV_32F, 1, 0, 3)
    imgDiffY = cv2.Sobel(img, cv2.CV_32F, 0, 1, 3)
    imgDiffXY = cv2.multiply(imgDiffX, imgDiffY)

    imgDiffXX = cv2.multiply(imgDiffX, imgDiffX)
    imgDiffYY = cv2.multiply(imgDiffY, imgDiffY)
    J11 = cv2.boxFilter(imgDiffXX, cv2.CV_32F, (w, w))
    J22 = cv2.boxFilter(imgDiffYY, cv2.CV_32F, (w, w))
    J12 = cv2.boxFilter(imgDiffXY, cv2.CV_32F, (w, w))
    tmp1 = J11 + J22
    tmp2 = J11 - J22
    tmp2 = cv2.multiply(tmp2, tmp2)
    tmp3 = cv2.multiply(J12, J12)
    tmp4 = np.sqrt(tmp2 + 4.0 * tmp3)
    lambda1 = 0.5 * (tmp1 + tmp4)  # biggest eigenvalue
    lambda2 = 0.5 * (tmp1 - tmp4)  # smallest eigenvalue
    imgCoherencyOut = cv2.divide(lambda1 - lambda2, lambda1 + lambda2)
    imgOrientationOut = cv2.phase(J22 - J11, 2.0 * J12, angleInDegrees=True)
    imgOrientationOut = 0.5 * imgOrientationOut
    return imgCoherencyOut, imgOrientationOut

def segment(images_adress):
    images = [imageio.imread(file) for file in images_adress]
    images = [cv2.resize(image, np.flip(images[0].shape)) for image in images]
    average_proj = np.mean(np.array(images),axis=0)
    segmented = average_proj>10
    segmented = remove_holes(segmented)
    segmented = segmented.astype(np.uint8)
    connected = remove_component(segmented)
    connected = connected.astype(np.uint8)
    skeletonized = cv2.ximgproc.thinning(np.array(connected, dtype=np.uint8))
    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph, pos = remove_spurs(nx_graph, pos,threshold = 20)
    # nx_graph = clean_degree_4(nx_graph, pos)[0]
    return(skeletonized,nx_graph,pos)

def extract_section_profiles_for_edge(
    edge: tuple,
    pos: dict,
    raw_im : np.array,
    nx_graph,
    resolution=5,
    offset=4,
    step=15,
    target_length=120,
) -> np.array:
    """
    Main function to extract section profiles of an edge.
    Given an Edge of Experiment at timestep t, returns a np array
    of dimension (target_length, m) where m is the number of section
    taken on the hypha.
    :param resolution: distance between two measure points along the hypha
    :param offset: distance at the end and the start where no point is taken
    :param step: step in pixel to compute the tangent to the hypha
    :target_length: length of the section extracted in pixels
    :return: np.array of sections, list of segments in TIMESTEP referential
    """
    pixel_list = orient(nx_graph.get_edge_data(*edge)['pixel_list'],pos[edge[0]])
    offset = max(
        offset, step
    )  # avoiding index out of range at start and end of pixel_list
    pixel_indexes = generate_pivot_indexes(
        len(pixel_list), resolution=resolution, offset=offset
    )
    list_of_segments = compute_section_coordinates(
        pixel_list, pixel_indexes, step=step, target_length=target_length + 1
    )  # target_length + 1 to be sure to have length all superior to target_length when cropping
    # TODO (FK): is a +1 enough?
    images = {}
    l = []
    for i, sect in enumerate(list_of_segments):
        im = raw_im
        point1 = np.array([sect[0][0], sect[0][1]])
        point2 = np.array([sect[1][0], sect[1][1]])
        profile = profile_line(im, point1, point2, mode="constant")[:target_length]
        profile = profile.reshape((1, len(profile)))
        # TODO(FK): Add thickness of the profile here
        l.append(profile)
    return np.concatenate(l, axis=0), list_of_segments

def plot_segments_on_image(segments,ax):
    for (point1, point2) in segments:
        ax.plot(
                [point1[1], point2[1]],  # x1, x2
                [point1[0], point2[0]],  # y1, y2
                color="red",
                linewidth=2,
            )

def get_kymo(edge,pos,images_adress,nx_graph_pruned):
    kymo = []
    for image_adress in images_adress:
        image = imageio.imread(image_adress)
        slices, segments = extract_section_profiles_for_edge(
            edge,
            pos,
            image,
            nx_graph_pruned,
            resolution=1,
            offset=4,
            step=15,
            target_length=10,
        )
        kymo_line = np.mean(slices,axis=1)
        kymo.append(kymo_line)
    return(np.array(kymo))

def filter_kymo(kymo):
    A = kymo[:,:]
    B = np.flip(A,axis=0)
    C = np.flip(A,axis=1)
    D = np.flip(B,axis=1)
    tiles = [[D,B,D],[C,A,C],[D,B,D]]
    tiles = [cv2.hconcat(imgs) for imgs in tiles]
    tiling_for_fourrier = cv2.vconcat(tiles)
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(tiling_for_fourrier))
    coordinates_middle = np.array(dark_image_grey_fourier.shape)//2
    LT_quadrant = np.s_[:coordinates_middle[0],:coordinates_middle[1]]
    LB_quadrant = np.s_[coordinates_middle[0]+1:,:coordinates_middle[1]]
    RB_quadrant = np.s_[coordinates_middle[0]+1:,coordinates_middle[1]:]
    RT_quadrant = np.s_[:coordinates_middle[0],coordinates_middle[1]:]

    filtered_fourrier  = dark_image_grey_fourier
    filtered_fourrier[LT_quadrant] = 0
    filtered_fourrier[RB_quadrant] = 0
    filtered = np.fft.ifft2(filtered_fourrier)
    shape_v,shape_h = filtered.shape
    shape_v,shape_h = shape_v//3,shape_h//3
    middle_slice = np.s_[shape_v:2*shape_v,shape_h:2*shape_h]
    middle = filtered[middle_slice]
    filtered_left = A-np.abs(middle)
    filtered_fourrier  = dark_image_grey_fourier
    filtered_fourrier[RT_quadrant] = 0
    filtered_fourrier[LB_quadrant] = 0
    filtered = np.fft.ifft2(filtered_fourrier)
    middle = filtered[middle_slice]
    filtered_right= A-np.abs(middle)
    return(filtered_left,filtered_right)


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]