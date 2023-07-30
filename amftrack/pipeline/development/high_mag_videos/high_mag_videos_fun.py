import imageio
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage as ndi

from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
)
import scipy
from amftrack.pipeline.functions.image_processing.node_id import remove_spurs
from amftrack.pipeline.functions.image_processing.extract_skel import (
    remove_component,
    remove_holes,
)
import numpy as np
from amftrack.pipeline.functions.image_processing.extract_width_fun import (
    generate_pivot_indexes,
    compute_section_coordinates,
)
from skimage.measure import profile_line
from amftrack.pipeline.functions.image_processing.experiment_class_surf import orient
from skimage.filters import frangi
from skimage.morphology import skeletonize
import itertools,operator
from scipy.ndimage.filters import generic_filter



def get_length_um_edge(edge, nx_graph, space_pixel_size):
    pixel_conversion_factor = space_pixel_size
    length_edge = 0
    pixels = nx_graph.get_edge_data(*edge)["pixel_list"]
    for i in range(len(pixels) // 10 + 1):
        if i * 10 <= len(pixels) - 1:
            length_edge += np.linalg.norm(
                np.array(pixels[i * 10])
                - np.array(pixels[min((i + 1) * 10, len(pixels) - 1)])
            )
    #             length_edge+=np.linalg.norm(np.array(pixels[len(pixels)//10-1*10-1])-np.array(pixels[-1]))
    return length_edge * pixel_conversion_factor


def calcGST(inputIMG, w):
    """
    Calculates the Image orientation and the image coherency. Image orientation is merely a guess, and image coherency gives an idea how sure that guess is.
    inputIMG:   The input image
    w:          The window size of the various filters to use. Large boxes catch higher order structures.
    """

    # The idea here is to perceive any patch of the image as a transformation matrix.
    # Such a matrix will have some eigenvalues, which describe the direction of uniform transformation.
    # If the largest eigenvalue is much bigger than the smallest eigenvalue, that indicates a strong orientation.

    img = inputIMG.astype(np.float32)
    imgDiffX = cv2.Sobel(img, cv2.CV_32F, 1, 0, -1)
    imgDiffY = cv2.Sobel(img, cv2.CV_32F, 0, 1, -1)
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


def segment(images_adress, threshold=10):
    images = [imageio.imread(file) for file in images_adress]
    images = [cv2.resize(image, np.flip(images[0].shape)) for image in images]
    average_proj = np.mean(np.array(images), axis=0)
    segmented = average_proj > threshold
    segmented = remove_holes(segmented)
    segmented = segmented.astype(np.uint8)
    connected = remove_component(segmented)
    connected = connected.astype(np.uint8)
    skeletonized = cv2.ximgproc.thinning(np.array(connected, dtype=np.uint8))
    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph, pos = remove_spurs(nx_graph, pos, threshold=20)
    # nx_graph = clean_degree_4(nx_graph, pos)[0]
    return (skeletonized, nx_graph, pos)


def extract_section_profiles_for_edge(
        edge: tuple,
        pos: dict,
        raw_im: np.array,
        nx_graph,
        resolution=5,
        offset=4,
        step=15,
        target_length=120,
        bounds=(0, 1),
) -> np.array:
    """
    Main function to extract section profiles of an edge.
    Given an Edge of Experiment at timestep t, returns a np array
    of dimension (target_length, m) where m is the number of section
    taken on the hypha.
    :param resolution:  distance between two measure points along the hypha
    :param offset:      distance at the end and the start where no point is taken
    :param step:        step in pixel to compute the tangent to the hypha
    :target_length:     length of the section extracted in pixels
    :return:            np.array of sections, list of segments in TIMESTEP referential
    """
    pixel_list = orient(nx_graph.get_edge_data(*edge)["pixel_list"], pos[edge[0]])
    offset = max(
        offset, step
    )  # avoiding index out of range at start and end of pixel_list
    pixel_indexes = generate_pivot_indexes(
        len(pixel_list), resolution=resolution, offset=offset
    )

    # WARNING: Below function is the bottleneck if extract_section_profiles_for_edge is called multiple times for the
    # same edge. Consider implementing this code on a higher level.
    list_of_segments = compute_section_coordinates(
        pixel_list, pixel_indexes, step=step, target_length=target_length + 1
    )  # target_length + 1 to be sure to have length all superior to target_length when cropping
    # TODO (FK): is a +1 enough?
    images = {}
    l = []
    im = raw_im

    for i, sect in enumerate(list_of_segments):
        point1 = np.array([sect[0][0], sect[0][1]])
        point2 = np.array([sect[1][0], sect[1][1]])
        profile = profile_line(im, point1, point2, mode="constant")[
                  int(bounds[0] * target_length): int(bounds[1] * target_length)
                  ]
        profile = profile.reshape((1, len(profile)))
        # TODO(FK): Add thickness of the profile here
        l.append(profile)
    return np.concatenate(l, axis=0), list_of_segments


def plot_segments_on_image(segments, ax, color="red", bounds=(0, 1), alpha=1, adj=1):
    for (point1_pivot, point2_pivot) in segments:
        point1 = (1 - bounds[0]) * point1_pivot + bounds[0] * point2_pivot
        point2 = (1 - bounds[1]) * point1_pivot + bounds[1] * point2_pivot
        ax.plot(
            [point1[1]*adj, point2[1]*adj],  # x1, x2
            [point1[0]*adj, point2[0]*adj],  # y1, y2
            color=color,
            linewidth=2,
            alpha=alpha,
        )


# OLD FUNCTION. Use get_kymo_new
def get_kymo(
        edge,
        pos,
        images_adress,
        nx_graph_pruned,
        resolution=1,
        offset=4,
        step=15,
        target_length=10,
        bound1=0,
        bound2=1,
):
    kymo = []
    for image_adress in images_adress:
        image = imageio.imread(image_adress)
        slices, segments = extract_section_profiles_for_edge(
            edge,
            pos,
            image,
            nx_graph_pruned,
            resolution=resolution,
            offset=offset,
            step=step,
            target_length=target_length,
            bound1=bound1,
            bound2=bound2,
        )
        kymo_line = np.mean(slices, axis=1)
        kymo.append(kymo_line)
    return np.array(kymo)


def filter_kymo_left_old(kymo):
    A = kymo[:, :]
    B = np.flip(A, axis=0)
    C = np.flip(A, axis=1)
    D = np.flip(B, axis=1)
    tiles = [[D, B, D], [C, A, C], [D, B, D]]
    tiles = [cv2.hconcat(imgs) for imgs in tiles]
    tiling_for_fourrier = cv2.vconcat(tiles)
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(tiling_for_fourrier))
    coordinates_middle = np.array(dark_image_grey_fourier.shape) // 2
    LT_quadrant = np.s_[: coordinates_middle[0], : coordinates_middle[1]]
    LB_quadrant = np.s_[coordinates_middle[0] + 1:, : coordinates_middle[1]]
    RB_quadrant = np.s_[coordinates_middle[0] + 1:, coordinates_middle[1]:]
    RT_quadrant = np.s_[: coordinates_middle[0], coordinates_middle[1]:]

    filtered_fourrier = dark_image_grey_fourier
    filtered_fourrier[LT_quadrant] = 0
    filtered_fourrier[RB_quadrant] = 0
    filtered = np.fft.ifft2(filtered_fourrier)
    shape_v, shape_h = filtered.shape
    shape_v, shape_h = shape_v // 3, shape_h // 3
    middle_slice = np.s_[shape_v: 2 * shape_v, shape_h: 2 * shape_h]
    middle = filtered[middle_slice]
    filtered_left = A - np.abs(middle)
    return filtered_left


def tile_image(img):
    A = img[:, :]
    B = np.flip(A, axis=0)
    C = np.flip(A, axis=1)
    D = np.flip(B, axis=1)
    tiles = [[D, B, D], [C, A, C], [D, B, D]]
    tiles = [cv2.hconcat(imgs) for imgs in tiles]
    tiling_for_fourrier = cv2.vconcat(tiles)
    return tiling_for_fourrier


def filter_kymo_left(kymo, nr_tiles=1, static_offset=1, static_angle=1000, plots=False, perc_dim=6):
    """
    This is a complicated function, with a lot of considerations. The idea is to make a fourier transform of a
    kymograph, then remove certain parts of it to end up with a spectrum where only forward or backward moving
    particles are shown. Preferably, the DC term of the fourier transform is also removed. Intensity is recovered by
    comparing the dimmest particles of the resulting filtered kymos and the original image.
    kymo:           The input kymograph
    nr_tiles:       The number of tilings that is done to avoid Fourier ringing. Currently only set to one
    static_offset:  How many horizontal bands in the fourier spectrum are removed from the middle. These correspond to vertical lines in real-space.
    static_angle:   The angle of the two triangles that flare from the middle of the spectrum. This to remove static particles from the kymograph that have negligible movement.
    """
    if plots:
        fig, ax = plt.subplots(4, 2, figsize=(9, 18))
        img_max = np.max(kymo.flatten())

    # Final shifting of the filtered kymo is based on the 3% of dimmest pixels. The mean of these have to match up.
    dim_pixls = kymo < np.percentile(kymo, perc_dim)
    dim_val = np.mean(kymo[dim_pixls].flatten())

    # The kymograph is tiled to prevent ringing. This creates a 3x3 tile

    img = kymo.copy()
    for i in range(nr_tiles):
        img = tile_image(img)
    tiling_for_fourrier = img
    shape_v, shape_h = tiling_for_fourrier.shape

    # We use orthogonal normalization in the fourier transforms to try and get proper intensities back
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(tiling_for_fourrier, norm='ortho'))
    coordinates_middle = np.array(dark_image_grey_fourier.shape) // 2

    # Within the fourier transform, we have four quadrants, where we remove two of them. This in addition to the DC term
    LT_quadrant = np.s_[: coordinates_middle[0] + 0, : coordinates_middle[1]]
    LB_quadrant = np.s_[coordinates_middle[0] - 1:, : coordinates_middle[1]]
    RB_quadrant = np.s_[coordinates_middle[0] + 1:, coordinates_middle[1] + 1:]
    RT_quadrant = np.s_[: coordinates_middle[0], coordinates_middle[1]:]

    filtered_fourrier = dark_image_grey_fourier.copy()

    # Calculation of the static part of the fourier spectrum. Looks like a horizontal band with vertically flared ends.
    v_axis = np.arange(0, shape_v) - (shape_v // 2)
    h_axis = np.arange(0, shape_h) - (shape_h // 2)
    v_array = np.array([v_axis for i in range(shape_h)]).transpose()
    h_array = np.array([h_axis for i in range(shape_v)])
    stat_array = 1 * (abs(v_array) <= (static_offset + abs(h_array) / static_angle))

    filtered_fourrier[LT_quadrant] = 0
    filtered_fourrier[RB_quadrant] = 0
    filtered_fourrier *= (1 - stat_array)
    # filtered_fourrier[coordinates_middle[0], coordinates_middle[1]] = 0

    # stat_array[coordinates_middle[0], coordinates_middle[1]] = 0
    stat_filt = np.fft.ifft2(np.fft.ifftshift(dark_image_grey_fourier * stat_array), norm='ortho')
    filtered = np.fft.ifft2(np.fft.ifftshift(filtered_fourrier), norm='ortho')

    shape_v, shape_h = shape_v // 3, shape_h // 3
    middle_slice = np.s_[shape_v: 2 * shape_v, shape_h: 2 * shape_h]
    middle_square = np.s_[coordinates_middle[0] - 10:coordinates_middle[0] + 50,
                    coordinates_middle[1] - 50:coordinates_middle[1] + 10]
    middle = filtered[middle_slice]
    filtered_left = kymo - middle.real

    if plots:
        col = [None] * 6
        col[0] = ax[0][0].imshow(tiling_for_fourrier)
        ax[0][0].set_title("tiled image")
        col[1] = ax[0][1].imshow(stat_array[middle_square], vmin=0)
        ax[0][1].set_title("Fourier filter")
        col[2] = ax[1][0].imshow(filtered[middle_slice].real, vmin=0, vmax=img_max)
        ax[1][0].set_title("forw + stat")
        col[3] = ax[1][1].imshow(filtered_left, vmin=0, vmax=img_max)
        ax[1][1].set_title("back")
        col[4] = ax[2][0].imshow(filtered[middle_slice].real + filtered_left, vmin=0, vmax=img_max)
        ax[2][0].set_title("forw + back + stat")
        col[5] = ax[2][1].imshow(abs(stat_filt[middle_slice]), vmin=0, vmax=img_max)
        ax[2][1].set_title("stat")
        ax[3][0].imshow(np.log(abs(dark_image_grey_fourier))[middle_square])
        ax[3][0].set_title("Full 2D Fourier transform")
        ax[3][1].imshow(np.log(abs(filtered_fourrier))[middle_square])
        ax[3][1].set_title("Filtered Transform")
        for i in range(6):
            plt.colorbar(col[i], fraction=0.056, pad=0.02)
        fig.tight_layout()

    # print(np.sum(abs(stat_filt[middle_slice]).flatten()))

    # img_out = np.abs(middle) - abs(stat_filt[middle_slice])

    # Here the intensity of the image is restored by setting equal the 3 percentile dimmest particles in the image.
    # This works best when there is a decent chunk of video where no particles are present.
    img_out = middle.real
    out_dim_pixls = img_out < np.percentile(img_out, perc_dim)
    out_dim_value = np.mean(img_out[out_dim_pixls].flatten())
    DC_value = dim_val - out_dim_value

    return (img_out + DC_value).real

    """
    Simple function that filters the kymograph and outputs the forward and backward filters.
    """


def filter_kymo(kymo):
    filtered_left = filter_kymo_left(kymo)
    filtered_right = np.flip(filter_kymo_left(np.flip(kymo, axis=1)), axis=1)
    return (filtered_left, filtered_right)


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


def get_speeds(kymo, W, C_Thr, fps, binning, magnification):
    time_pixel_size = 1 / fps  # s.pixel

    space_pixel_size = 2 * 1.725 / (magnification) * binning  # um.pixel
    imgCoherency, imgOrientation = calcGST(kymo, W)
    nans = np.empty(imgOrientation.shape)
    nans.fill(np.nan)

    real_movement = np.where(imgCoherency > C_Thr, imgOrientation, nans)
    speed = (
            np.tan((real_movement - 90) / 180 * np.pi) * space_pixel_size / time_pixel_size
    )  # um.s-1


def get_width_from_graph_im(edge, pos, image, nx_graph_pruned, slice_length=400):
    bound1 = 0
    bound2 = 1
    offset = 100
    step = 30
    target_length = slice_length
    resolution = 1
    slices, _ = extract_section_profiles_for_edge(
        edge,
        pos,
        image,
        nx_graph_pruned,
        resolution=resolution,
        offset=offset,
        step=step,
        target_length=target_length,
        bounds=(bound1, bound2)
    )
    return get_width(slices)


def get_width(slices, avearing_window=50, num_std=2):
    widths = []
    for index in range(len(slices)):
        thresh = np.mean(
            (
                np.mean(slices[index, :avearing_window]),
                np.mean(slices[index, -avearing_window:]),
            )
        )
        std = np.std(
            (
                np.concatenate(
                    (slices[index, :avearing_window], slices[index, -avearing_window:])
                )
            )
        )
        try:
            deb = np.min(np.argwhere(slices[index, :] < thresh - num_std * std))
            end = np.max(np.argwhere(slices[index, :] < thresh - num_std * std))
            print(deb)
            width = end - deb
            widths.append(width)
        except ValueError:
            continue
    return np.median(widths)


def segment_brightfield(image, thresh=0.5e-6, frangi_range=np.arange(70, 170, 30), seg_thresh = 11, binning=2):
    """
    Segmentation method for brightfield video, uses vesselness filters to get result.
    image:          Input image
    thresh:         Value close to zero such that the function will output a boolean array
    frangi_range:   Range of values to use a frangi filter with. Frangi filter is very good for brightfield vessel segmentation

    """
    frangi_range = frangi_range * 2 / binning
    smooth_im_blur = cv2.blur(-image, (11, 11))
    smooth_im = frangi(-smooth_im_blur, frangi_range)
    smooth_im = np.array(smooth_im * (255/np.max(smooth_im)), dtype=np.uint8)
    _, segmented = cv2.threshold(smooth_im, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
#     seg_shape = smooth_im.shape

#     for i in range(1, 100):
#         _, segmented = cv2.threshold(smooth_im, i, 255, cv2.THRESH_BINARY)
#         coverage = 100 * np.sum(1 * segmented.flatten()) / (255 * seg_shape[0] * seg_shape[1])
#         if coverage < seg_thresh:
#             break
    
    skeletonized = skeletonize(segmented > seg_thresh)
    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph_pruned, pos = remove_spurs(nx_graph, pos, threshold=200)
    return (segmented, nx_graph_pruned, pos)


def get_kymo_new(
        edge,
        pos,
        images_adress,
        nx_graph_pruned,
        resolution=1,
        offset=4,
        step=15,
        target_length=10,
        bounds=(0, 1),
        x_len=10,
        order=None
):
    """
    The new get_kymo function. The old one had some inefficiencies that lead to it being much slower than this one.
    edge:               The edge in question. Will pull from skeleton the positional arguments
    pos:                Array of positions of nodes that are associated with the skeleton
    images_adress:      misspelled variable that gives the address of the image folder housing the video
    nx_graph_pruned:    Graph array that houses the skeleton of the hypha structure in the video
    resolution:         No idea what this does, probably important
    offset:             Measure of the length of the hypha
    step:               How large the step is between two points on the edge for tangent calculation.
                        Large step means smooth kymo, but less accuracy, small step means more jittery kymo, but closer to the hypha
    target_length:      Target width of the hypha. Adjust this to fit the edge of the hypha neatly into an edge video.
    bounds:             tuple describing what fraction of the total hypha length to use. Range is 0.0 (start) to 1.0 (end)
    order:              Important variable that is changed in the function as well. No idea what it does.
    """

    """
    Following section calculates the section coordinates for the skeleton of the hypha. 
    These coordinates are used for all images in the video. Generally we don't expect hyphae to move.
    """
    pixel_list = orient(nx_graph_pruned.get_edge_data(*edge)["pixel_list"], pos[edge[0]])
    offset = max(offset, step)  # avoiding index out of range at start and end of pixel_list
    pixel_indexes = generate_pivot_indexes(len(pixel_list), resolution=resolution, offset=offset)
    list_of_segments = compute_section_coordinates(pixel_list, pixel_indexes, step=step,
                                                   target_length=target_length + 1)

    perp_lines = []
    kymo = []
    for i, sect in enumerate(list_of_segments):
        point1 = np.array([sect[0][0], sect[0][1]])
        point2 = np.array([sect[1][0], sect[1][1]])
        perp_lines.append(extract_perp_lines(point1, point2))

    for image_adress in images_adress:
        im = imageio.imread(image_adress)
        order = validate_interpolation_order(im.dtype, order)
        l = []
        for perp_line in perp_lines:
            pixels = ndi.map_coordinates(im, perp_line, prefilter=order > 1, order=order, mode='reflect', cval=0.0)
            pixels = np.flip(pixels, axis=1)
            pixels = pixels[int(bounds[0] * target_length): int(bounds[1] * target_length)]
            pixels = pixels.reshape((1, len(pixels)))
            l.append(pixels)

        slices = np.concatenate(l, axis=0)
        kymo_line = np.sum(slices, axis=1) / (target_length)
        kymo.append(kymo_line)
    return np.array(kymo)


def get_edge_image(edge, pos, images_address,
                   nx_graph_pruned,
                   resolution=1,
                   offset=4,
                   step=30,
                   target_length=10,
                   img_frame=0,
                   bounds=(0, 1),
                   order=None,
                   logging=False):
    slices_list = []
    pixel_list = orient(nx_graph_pruned.get_edge_data(*edge)["pixel_list"], pos[edge[0]])
    offset = max(
        offset, step
    )  # avoiding index out of range at start and end of pixel_list
    pixel_indexes = generate_pivot_indexes(
        len(pixel_list), resolution=resolution, offset=offset
    )
    list_of_segments = compute_section_coordinates(
        pixel_list, pixel_indexes, step=step, target_length=target_length + 1)
    perp_lines = []

    for i, sect in enumerate(list_of_segments):
        point1 = np.array([sect[0][0], sect[0][1]])
        point2 = np.array([sect[1][0], sect[1][1]])
        perp_lines.append(extract_perp_lines(point1, point2))

    if np.ndim(img_frame) == 0:
        im_list = [imageio.imread(images_address[img_frame])]
        order = validate_interpolation_order(im_list[0].dtype, order)
    else:
        im_list = [imageio.imread(images_address[frame]) for frame in img_frame]
        order = validate_interpolation_order(im_list[0].dtype, order)
    for im in im_list:
        l = []
        for perp_line in perp_lines:
            pixels = ndi.map_coordinates(im, perp_line, prefilter=order > 1, order=order, mode='reflect', cval=0.0)
            pixels = np.flip(pixels, axis=1)
            pixels = pixels[int(bounds[0] * target_length): int(bounds[1] * target_length)]
            pixels = pixels.reshape((1, len(pixels)))
            # TODO(FK): Add thickness of the profile here
            l.append(pixels)
        slices = np.concatenate(l, axis=0)
        slices_list.append(slices)

    if np.ndim(img_frame) == 0:
        return slices_list[0]
    else:
        return np.array(slices_list)
    
def get_edge_widths(edge, pos, segmented,
                   nx_graph_pruned,
                   resolution=1,
                   offset=4,
                   step=30,
                   target_length=10,
                   bounds=(0, 1),
                   order=None,
                   logging=False):
    target_length = target_length*2
    slices_list = []
    pixel_list = orient(nx_graph_pruned.get_edge_data(*edge)["pixel_list"], pos[edge[0]])
    offset = max(
        offset, step
    )  # avoiding index out of range at start and end of pixel_list
    pixel_indexes = generate_pivot_indexes(
        len(pixel_list), resolution=resolution, offset=offset
    )
    list_of_segments = compute_section_coordinates(
        pixel_list, pixel_indexes, step=step, target_length=target_length + 1)
    perp_lines = []

    for i, sect in enumerate(list_of_segments):
        point1 = np.array([sect[0][0], sect[0][1]])
        point2 = np.array([sect[1][0], sect[1][1]])
        perp_lines.append(extract_perp_lines(point1, point2))
    
    order = validate_interpolation_order(segmented.dtype, order)
    im = segmented
    l = []
    for perp_line in perp_lines:
        pixels = ndi.map_coordinates(im, perp_line, prefilter=order > 1, order=order, mode='reflect', cval=0.0)
        pixels = np.flip(pixels, axis=1)
        pixels = pixels[int(bounds[0] * target_length): int(bounds[1] * target_length)]
        pixels = pixels.reshape((1, len(pixels)))
        l.append(pixels)
    slices = np.concatenate(l, axis=0)
    slices_list = [max((sum(1 for _ in group) for value, group in itertools.groupby(pixel_row) if value == 0), default=0) for pixel_row in slices]
        # TODO(FK): Add thickness of the profile here
#         slices_list.append(max(len(list(y)) for (c,y) in itertools.groupby(np.array(pixels)) if c==1))
    
    return np.array(slices_list)
    


def extract_perp_lines(src, dst, linewidth=1):
    src_row, src_col = src = np.asarray(src, dtype=float)
    dst_row, dst_col = dst = np.asarray(dst, dtype=float)
    d_row, d_col = dst - src
    theta = np.arctan2(d_row, d_col)

    length = int(np.ceil(np.hypot(d_row, d_col) + 1))
    # we add one above because we include the last point in the profile
    # (in contrast to standard numpy indexing)
    line_col = np.linspace(src_col, dst_col, length)
    line_row = np.linspace(src_row, dst_row, length)

    # we subtract 1 from linewidth to change from pixel-counting
    # (make this line 3 pixels wide) to point distances (the
    # distance between pixel centers)
    col_width = (linewidth - 1) * np.sin(-theta) / 2
    row_width = (linewidth - 1) * np.cos(theta) / 2
    perp_rows = np.stack([np.linspace(row_i - row_width, row_i + row_width,
                                      linewidth) for row_i in line_row])
    perp_cols = np.stack([np.linspace(col_i - col_width, col_i + col_width,
                                      linewidth) for col_i in line_col])
    return np.stack([perp_rows, perp_cols])


def validate_interpolation_order(image_dtype, order):
    """Validate and return spline interpolation's order.

    Parameters
    ----------
    image_dtype : dtype
        Image dtype.
    order : int, optional
        The order of the spline interpolation. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.

    Returns
    -------
    order : int
        if input order is None, returns 0 if image_dtype is bool and 1
        otherwise. Otherwise, image_dtype is checked and input order
        is validated accordingly (order > 0 is not supported for bool
        image dtype)

    """

    if order is None:
        return 0 if image_dtype == bool else 1

    if order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the "
                         "range 0-5.")

    if image_dtype == bool and order != 0:
        raise ValueError(
            "Input image dtype is bool. Interpolation is not defined "
            "with bool data type. Please set order to 0 or explicitely "
            "cast input image to another data type.")

    return order


def segment_fluo(image, thresh=0.5e-7, seg_thresh=4.5, k_size=11, magnif = 50, binning=2, test_plot=False):
    kernel = np.ones((k_size, k_size), np.uint8)
    kernel_2 = np.ones((10, 10), np.uint8)
    smooth_im = cv2.GaussianBlur(image, (5, 5), 0)
    if magnif < 30:
        smooth_im = cv2.morphologyEx(smooth_im, cv2.MORPH_TOPHAT, kernel)
    im_canny = cv2.Canny(smooth_im, 0, 20)
    im_canny_smooth = cv2.GaussianBlur(im_canny, (5, 5), 0)
    smooth_im_close = cv2.morphologyEx(smooth_im, cv2.MORPH_CLOSE, kernel)
    std_im = generic_filter(image, np.std, size=10)
    
    if magnif < 30:
        k_data = np.float32(np.array([std_im.reshape(-1), im_canny_smooth.reshape(-1)]).T)
    else:
        k_data = np.float32(np.array([std_im.reshape(-1), smooth_im.reshape(-1)]).T)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    retval, labels, centers = cv2.kmeans(k_data, [4, 2][magnif<30], None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers=np.uint8(centers)
    labels=np.array(labels.T[0])
    centers = [center[0] for center in centers]
    segmented_data  = np.array([centers[label] for label in labels])
    segmented_image = segmented_data.reshape((smooth_im.shape))
    segmented_image = np.uint8(segmented_image>np.min(centers))
    segmented = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel_2)

    skeletonized = skeletonize(segmented > thresh)

    if test_plot:
        fig, ax = plt.subplots(7, figsize=(9, 25))
        ax[0].imshow(im_canny)
        ax[0].set_title("Smooth")
        ax[1].imshow(std_im)
        ax[1].set_title("open")
        ax[2].imshow(smooth_im)
        ax[2].set_title("smooth_im")
        ax[3].imshow(segmented)
        ax[3].set_title("segmented")
        ax[4].imshow(skeletonized)
        ax[5].hist(smooth_im_close.flatten(), log=True, bins=50)
        ax[6].plot(smooth_im_close[1000])
        fig.tight_layout()

    skeleton = scipy.sparse.dok_matrix(skeletonized)
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph_pruned, pos = remove_spurs(nx_graph, pos, threshold=200)
    return (segmented > thresh, nx_graph, pos)


def find_thresh_fluo(blurred_image, thresh_guess=40, fold_thresh=0.005):
    histr = cv2.calcHist([blurred_image], [0], None, [256], [0, 256])
    histr_sum = np.sum(histr[0:thresh_guess])
    for i in range(thresh_guess, 1, -1):
        diff = (histr[i] - histr[i - 1]) / histr_sum
        if -1 * diff > fold_thresh:
            break
    return i
