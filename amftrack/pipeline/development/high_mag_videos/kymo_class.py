import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    clean_degree_4,
)
import scipy
from amftrack.pipeline.functions.image_processing.node_id import remove_spurs
from amftrack.pipeline.functions.image_processing.extract_skel import remove_component, remove_holes
import numpy as np
from amftrack.pipeline.development.high_mag_videos.high_mag_videos_fun import *
from scipy import signal
from amftrack.pipeline.functions.image_processing.extract_skel import (
    extract_skel_new_prince,
    run_back_sub,
    bowler_hat,
)
from scipy.interpolate import griddata

from skimage.morphology import skeletonize
from amftrack.util.sys import temp_path
import pandas as pd
from PIL import Image
from scipy.optimize import curve_fit
from amftrack.pipeline.functions.image_processing.extract_skel import (
    extract_skel_new_prince,
    run_back_sub,
    bowler_hat,
)


class Kymo_video_analysis(object):
    def __init__(self,
                 imgs_address,
                 format='tiff',
                 logging=False,
                 fps=20,
                 binning=2,
                 magnification=50,
                 im_range=(0, -1),
                 thresh=5e-07,
                 filter_step=30,
                 vid_type='BRIGHT'
                 ):
        self.imgs_address = os.path.join(imgs_address, 'Img')
        self.fps = fps
        self.vid_type = vid_type
        self.logging = logging
        self.imformat = format
        self.im_range = im_range
        self.time_pixel_size = 1 / self.fps
        self.binning = binning
        self.magnification = magnification
        self.video_name = imgs_address.split("/")[-1]
        self.kymos_path = "/".join(
            imgs_address.split("/")[:-1] + ["".join((self.video_name, "Analysis"))]
        )
        if not os.path.exists(self.kymos_path):
            os.makedirs(self.kymos_path)
        if self.logging:
            print('Kymos file created, address is at {}'.format(self.kymos_path))
        self.files = os.listdir(self.imgs_address)
        self.images_total_path = [os.path.join(self.imgs_address, file) for file in self.files]
        self.images_total_path.sort()
        self.selection_file = self.images_total_path
        self.selection_file.sort()
        self.selection_file = self.selection_file[self.im_range[0]:self.im_range[1]]
        # print(self.selection_file[0])
        if self.logging:
            print('Using image selection {} to {}'.format(self.im_range[0], self.im_range[1]))
        self.pos = []

        self.create_skeleton(filter_step, thresh)

    def create_skeleton(self, filter_step, thresh):
        if self.vid_type == 'BRIGHT':
            self.segmented, self.nx_graph_pruned, self.pos = segment_brightfield(
                imageio.imread(self.selection_file[self.im_range[0]]), thresh=thresh)
        elif self.vid_type == 'FLUO':
            self.segmented, self.nx_graph_pruned, self.pos = segment_fluo(
                imageio.imread(self.selection_file[self.im_range[0]]), thresh=thresh)
        else:
            print("I don't have a valid flow_processing type!!! Using fluo thresholding.")
            self.segmented, self.nx_graph_pruned, self.pos = segment_fluo(
                imageio.imread(self.selection_file[self.im_range[0]]), thresh=thresh)
        self.edges = list(self.nx_graph_pruned.edges)

        for i, edge in enumerate(self.edges):
            if self.pos[edge[0]][0] > self.pos[edge[1]][0]:
                self.edges[i] = edge
            else:
                self.edges[i] = (edge[1], edge[0])

        if self.logging:
            print('Succesfully extracted the skeleton. Did you know there is a skeleton inside inside you right now?')

        self.edge_objects = [self.createEdge(edge) for edge in self.edges]
        self.edge_objects = self.filter_edges(filter_step)

    def filter_edges(self, step):
        filt_edge_objects = []
        for edge in self.edge_objects:
            offset = int(np.linalg.norm(self.pos[edge.edge_name[0]] - self.pos[edge.edge_name[1]])) // 4
            if offset >= step:
                filt_edge_objects.append(edge)
        return filt_edge_objects

    def plot_start_end(self):
        image = imageio.imread(self.selection_file[self.im_range[0]])
        image2 = imageio.imread(self.selection_file[self.im_range[1]])
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        ax.imshow(image2, alpha=0.5)
        plt.show()
        return None

    def plot_extraction_img(self, weight=0.05, bounds=(0.0, 1.0), step=30, target_length=130, resolution=1,
                            save_img=True, logging=False):
        fig, ax = plt.subplots()
        image = imageio.imread(self.selection_file[self.im_range[0]])
        ax.imshow(image)
        for edge in self.edge_objects:
            if logging:
                print('Working on edge {}, sir!'.format(edge.edge_name))
            offset = int(np.linalg.norm(self.pos[edge.edge_name[0]] - self.pos[edge.edge_name[1]])) // 4
            if offset >= step:
                segments = edge.create_segments(self.pos, image, self.nx_graph_pruned, resolution, offset, step,
                                                target_length, bounds)
                plot_segments_on_image(
                    segments, ax, bounds=bounds, color="white", alpha=0.1
                )
                ax.plot(
                    [self.pos[edge.edge_name[0]][1], self.pos[edge.edge_name[1]][1]],
                    [self.pos[edge.edge_name[0]][0], self.pos[edge.edge_name[1]][0]]
                )
                ax.text(
                    *np.flip((1 - weight) * self.pos[edge.edge_name[0]] + weight * self.pos[edge.edge_name[1]]),
                    str(edge.edge_name[0]),
                    color="white",
                )
                ax.text(
                    *np.flip((1 - weight) * self.pos[edge.edge_name[1]] + weight * self.pos[edge.edge_name[0]]),
                    str(edge.edge_name[1]),
                    color="white",
                )
        plt.show()
        if save_img:
            save_path_temp = os.path.join(self.kymos_path, f"Detected edges.png")
            plt.savefig(save_path_temp)
            print("Just saved an image, sir!")
        return None

    def createEdge(self, edge):
        return Kymo_edge_analysis(self, edge)


class Kymo_edge_analysis(object):
    def __init__(self, video_analysis=None, edge_name=None, kymo=None):
        if kymo is None:
            self.video_analysis = video_analysis
            self.edge_name = edge_name
            self.offset = int(np.linalg.norm(
                self.video_analysis.pos[self.edge_name[0]] - self.video_analysis.pos[self.edge_name[1]])) // 4

            self.kymo = []
            self.kymos = []
        else:
            if len(kymo.shape) == 2:
                self.kymo = kymo
            self.kymos = [kymo]
            self.edge_name = (-1, -1)
        self.filtered_left = []
        self.filtered_right = []
        self.slices = []
        self.segments = []
        self.bounds = (0, 1)
        self.edge_array = []

    def view_edge(self,
                  resolution=1,
                  step=30,
                  target_length=130,
                  save_im=True,
                  bounds=(0, 1),
                  img_frame=0):
        self.edge_array = get_edge_image(self.edge_name,
                                         self.video_analysis.pos,
                                         self.video_analysis.selection_file,
                                         self.video_analysis.nx_graph_pruned,
                                         resolution,
                                         self.offset,
                                         step,
                                         target_length,
                                         img_frame,
                                         bounds)
        # fig, ax = plt.subplots()
        # ax.imshow(self.edge_array)
        if save_im:
            im = Image.fromarray(self.edge_array.astype(np.uint8))
            save_path_temp = os.path.join(self.video_analysis.kymos_path, f"{self.edge_name} edge.png")
            im.save(save_path_temp)
            if self.video_analysis.logging:
                print('Saved the image')
        return self.edge_array

    def create_segments(self, pos, image, nx_graph_pruned, resolution, offset, step, target_length, bounds):
        self.slices, self.segments = extract_section_profiles_for_edge(
            self.edge_name,
            pos,
            image,
            nx_graph_pruned,
            resolution=resolution,
            offset=offset,
            step=step,
            target_length=target_length,
            bounds=bounds
        )
        return self.segments

    def extract_multi_kymo(self,
                           bin_nr,
                           resolution=1,
                           step=30,
                           target_length=130,
                           save_array=True,
                           save_im=True,
                           bounds=(0, 1)):
        bin_edges = np.linspace(bounds[0], bounds[1], bin_nr + 1)
        self.kymos = [self.extract_kymo(bounds=(bin_edges[i],
                                                bin_edges[i + 1]),
                                        resolution=resolution,
                                        step=step,
                                        target_length=target_length,
                                        save_array=save_array,
                                        save_im=save_im,
                                        img_suffix=str(i))
                      for i in range(bin_nr)]
        return self.kymos

    def extract_kymo(self,
                     resolution=1,
                     step=30,
                     target_length=130,
                     save_array=True,
                     save_im=True,
                     bounds=(0, 1),
                     img_suffix=""):
        self.kymo = get_kymo_new(self.edge_name,
                                 self.video_analysis.pos,
                                 self.video_analysis.selection_file,
                                 self.video_analysis.nx_graph_pruned,
                                 resolution,
                                 self.offset,
                                 step,
                                 target_length,
                                 bounds)
        if save_array:
            save_path_temp = os.path.join(self.video_analysis.kymos_path, f"{self.edge_name} {img_suffix} kymo.npy")
            np.save(save_path_temp, self.kymo)
            if self.video_analysis.logging:
                print('Saved the array')
        if save_im:
            im = Image.fromarray(self.kymo.astype(np.uint8))
            save_path_temp = os.path.join(self.video_analysis.kymos_path, f"{self.edge_name} {img_suffix} kymo.png")
            im.save(save_path_temp)
            if self.video_analysis.logging:
                print('Saved the image')
        return self.kymo

    def fourier_kymo(self):
        if len(self.kymos) > 0:
            self.filtered_left = [filter_kymo_left(kymo) for kymo in self.kymos]
            self.filtered_right = [np.flip(filter_kymo_left(np.flip(kymo, axis=1)), axis=1) for kymo in self.kymos]
        return self.filtered_left, self.filtered_right

    def extract_speeds(self, speed_thresh=20, plots=False, speedplot=False, w=3, c_thr=0.95, klen=25,
                       magnification=2 * 1.725, binning=1, fps=1):
        speeds_tot = []
        times = []
        kernel = np.ones((klen, klen)) / klen ** 2
        space_pixel_size = 2 * 1.725 / (magnification) * binning  # um.pixel
        time_pixel_size = 1 / fps  # s.pixel

        speed_dataframe = pd.DataFrame()

        if speedplot:
            fig, ax = plt.subplots(len(self.kymos))

        for j, kymo in enumerate(self.kymos):
            nans = np.empty(kymo.shape)
            nans.fill(np.nan)
            speeds = [[], []]
            for i in [0, 1]:
                kymo_interest = [self.filtered_left[j], self.filtered_right[j]][i]
                imgCoherency, imgOrientation = calcGST(kymo_interest, w)
                real_movement = np.where(imgCoherency > c_thr, imgOrientation, nans)
                speed = (
                        np.tan((real_movement - 90) / 180 * np.pi)
                        * space_pixel_size
                        / time_pixel_size
                )
                speed = np.where(speed < speed_thresh, speed, nans)
                speed = np.where(speed > -1*speed_thresh, speed, nans)
                z1 = scipy.signal.convolve2d(imgCoherency, kernel, mode="same")
                speed = np.where(z1 > 0.8, speed, nans)
                if i == 0:
                    times.append(np.array(range(kymo.shape[0])) * time_pixel_size)

                label = self.edge_name if i == 0 else None
                speeds[i] = speed
                direction = np.array(["root" if i == 0 else "tip" for k in range(speed.shape[0])])
                edges_list = np.array([str(self.edge_name) for k in range(speed.shape[0])])

                data = pd.DataFrame(
                    np.transpose((times[j], np.nanmean(speeds[i], axis=1), edges_list, direction)),
                    columns=["time (s)", "speed (um.s-1)", "edge", "direction"],
                )
                speed_dataframe = pd.concat((speed_dataframe, data))
            if speedplot:
                if len(self.kymos) == 1:
                    p = ax.imshow(
                        np.nansum([speeds[0], speeds[1]], 0),
                        label=self.edge_name if i == 0 else None,
                        cmap='bwr'
                    )
                elif len(self.kymos) > 1:
                    p = ax[j].imshow(
                        speeds[i],
                        label=self.edge_name if i == 0 else None,
                        cmap='bwr'
                    )
                ax.set_ylabel("speed($\mu m.s^{-1}$)")
                ax.set_xlabel("time ($s$)")
            # np.concatenate((speeds_tot, speeds))
            speeds_tot.append(speeds)
        if speedplot:
            ax.legend()
            fig.tight_layout()

        return np.array(speeds_tot), times


