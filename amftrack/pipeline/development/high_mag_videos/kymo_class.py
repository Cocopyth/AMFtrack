from pathlib import Path
import imageio.v2 as imageio
from amftrack.pipeline.development.high_mag_videos.high_mag_videos_fun import *
from scipy import signal
import pandas as pd
from PIL import Image



class Kymo_video_analysis(object):
    def __init__(self,
                 imgs_address,
                 format='tiff',
                 fps=None,
                 binning=2,
                 magnification=50,
                 vid_type='BRIGHT',
                 im_range=(0, -1),
                 thresh=5e-07,
                 filter_step=30,
                 seg_thresh=20,
                 logging=False,
                 segment_plots=False,
                 show_seg=False
                 ):
        """
        imgs_address:   This is the folder address containing an "Img" file and potentially an Analysis file.
        format:         The file format of the raw images, will always be tiff unless we do different experiments
        fps:            The frames per second of the video that is analyzed. Will be extracted from the csv, unless manually assigned.
        binning:        The level of binning done during measurement. 1 means the video is in 4X resolution, 2 means 2X resolution.
        magnification:  The type of objective used during measurement. Typical values are 50 and 4
        vid_type:       Either "BRIGHT" or "FLUO", should be extracted from the csv, changes filters for skeletonisation
        im_range:       The range of images used in measurement, default is all of them
        thresh:         Threshold for the skeletonization function, which makes a binary image for scipy skeletonize
        filter_step:    Threshold for the filter step of the skeletonization function, which removes edges smaller than filter_step
        logging:        Boolean on whether progress is printed to the terminal

        """

        # First we're assigning values, and extracting data about the video if there is a .csv (or .xlsx) available.
        # self.imgs_address = os.path.join(imgs_address, 'Img')
        self.imgs_address = Path(imgs_address) / "Img"
        self.video_nr = int(imgs_address[:-1].split("_")[-1])
        parent_files = self.imgs_address.parents[1]
        self.binning = binning
        self.magnification = magnification
        self.vid_type = vid_type
        self.fps = fps
        self.logging = logging
        self.imformat = format
        self.im_range = im_range
        self.segment_plots= segment_plots
        self.back_fit = [0,0,0]
        self.back_offset = 0

        ### Extracts the video parameters from the nearby csv.
        self.csv_path = next(Path(parent_files).glob("*.csv"), None)
        self.xlsx_path = next(Path(parent_files).glob("*.xlsx"), None)
        if self.csv_path is not None or self.xlsx_path is not None:
            if self.csv_path is not None and self.xlsx_path is not None:
                print("Found both a csv and xlsx file, will use csv data.")
            if self.csv_path is not None:
                if logging:
                    print("Found a csv file, using that data")
                videos_data = pd.read_csv(self.csv_path)


                print(videos_data)

                self.video_data = videos_data.loc[videos_data['video'] == self.video_nr]
                # print(self.video_data["Illumination"][0])
                self.vid_type = ["FLUO", "BRIGHT"][self.video_data["Illumination"].iloc[0] == "BF"]
                self.fps = float(self.video_data["fps"].iloc[0])
                self.binning = (self.video_data["Binned"].iloc[0] * 2) + 1
                self.magnification = self.video_data["Lens"].iloc[0]
            elif self.xlsx_path is not None:
                if logging:
                    print("Found an xlsx file, using that data")
                videos_data = pd.read_excel(self.xlsx_path)
                self.video_data = videos_data.loc[videos_data.iloc[:, 0] == imgs_address[:-1].split('/')[-1]]
                self.vid_type = ["FLUO", "BRIGHT"][self.video_data.iloc[0, 9] == "BF"]
                self.fps = float(self.video_data["FPS"].iloc[0])
                self.binning = [1, 2][self.video_data["Binned (Y/N)"].iloc[0] == "Y"]
                self.magnification = self.video_data["Magnification"].iloc[0]

            if logging:
                print(f"Analysing {self.vid_type} video of {self.magnification}X zoom, with {self.fps} fps")

        self.time_pixel_size = 1 / self.fps
        self.space_pixel_size = 2 * 1.725 / (self.magnification) * self.binning  # um.pixel
        self.pos = []
        self.kymos_path = "/".join(
            imgs_address.split("/")[:-1] + ["".join((imgs_address.split("/")[-1], "Analysis"))]
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
        self.target_length=130
        self.x_length = self.space_pixel_size * self.target_length
        # print(self.selection_file[0])
        if self.logging:
            print(
                'Using image selection {} to {}'.format(self.im_range[0], len(self.selection_file) + self.im_range[0]))

        ###Skeleton creation, we segment the image using either brightfield or fluo segmentation methods.
        if self.vid_type == 'BRIGHT':
            self.segmented, self.nx_graph_pruned, self.pos = segment_brightfield(
                imageio.imread(self.selection_file[self.im_range[0]]), thresh=thresh, segment_plots=self.segment_plots)
        elif self.vid_type == 'FLUO':
            self.segmented, self.nx_graph_pruned, self.pos = segment_fluo(
                imageio.imread(self.selection_file[self.im_range[0]]), thresh=thresh, segment_plots=self.segment_plots, seg_thresh=seg_thresh)
        else:
            print("I don't have a valid flow_processing type!!! Using fluo thresholding.")
            self.segmented, self.nx_graph_pruned, self.pos = segment_fluo(
                imageio.imread(self.selection_file[self.im_range[0]]), thresh=thresh, seg_thresh=seg_thresh, segment_plots=self.segment_plots)
        self.edges = list(self.nx_graph_pruned.edges)
        for i, edge in enumerate(self.edges):
            if self.pos[edge[0]][0] > self.pos[edge[1]][0]:
                self.edges[i] = edge
            else:
                self.edges[i] = (edge[1], edge[0])
        self.edges = self.filter_edges(filter_step)
        self.edge_objects = [self.createEdge(edge) for edge in self.edges]

        if self.logging:
            print('Succesfully extracted the skeleton. Did you know there is a skeleton inside inside you right now?')

        if show_seg:
            fig, ax = plt.subplots()
            ax.imshow(self.segmented)
            seg_shape = self.segmented.shape
            ax.set_title(f'Segmentation ({ 100* np.sum(1*self.segmented.flatten()) / (seg_shape[0]*seg_shape[1]):.4} $\%$ coverage)')
            fig.tight_layout()
    def filter_edges(self, step):
        """
        Uses the edge name as a measure of distance, filters edges that are too short.
        """
        filt_edge_objects = []
        for edge in self.edges:
            offset = int(np.linalg.norm(self.pos[edge[0]] - self.pos[edge[1]])) // 4
            if offset >= step:
                filt_edge_objects.append(edge)
        return filt_edge_objects

    #
    # def plot_start_end(self):
    #     image = imageio.imread(self.selection_file[self.im_range[0]])
    #     image2 = imageio.imread(self.selection_file[self.im_range[1]])
    #     fig, ax = plt.subplots()
    #     ax.imshow(image, cmap="gray")
    #     ax.imshow(image2, alpha=0.5)
    #     plt.show()
    #     return None

    def plot_extraction_img(self, weight=0.05, bounds=(0.0, 1.0), target_length=130, resolution=1,
                            save_img=True, logging=False):
        """
        Sadly an essential function, that makes each edge calculate its own edges.
        Will output a chart of all edges with node points to select hypha from.
        """
        self.target_length = target_length
        self.x_length = self.target_length*self.space_pixel_size
        fig, ax = plt.subplots(1,2, figsize=(10,5))
        image = imageio.imread(self.selection_file[self.im_range[0]])
        ax[0].imshow(self.segmented)
        
        ax[0].set_title("Segmentation and skeleton")
        ax[1].imshow(image)
        for edge in self.edge_objects:
            if logging:
                print('Working on edge {}, sir!'.format(edge.edge_name))
            offset = int(np.linalg.norm(self.pos[edge.edge_name[0]] - self.pos[edge.edge_name[1]])) // 4
            segments = edge.create_segments(self.pos, image, self.nx_graph_pruned, resolution, offset,
                                            target_length, bounds)
            plot_segments_on_image(
                segments, ax[1], bounds=bounds, color="white", alpha=0.1
            )
            ax[1].plot(
                [self.pos[edge.edge_name[0]][1], self.pos[edge.edge_name[1]][1]],
                [self.pos[edge.edge_name[0]][0], self.pos[edge.edge_name[1]][0]]
            )
            ax[1].text(
                *np.flip((1 - weight) * self.pos[edge.edge_name[0]] + weight * self.pos[edge.edge_name[1]]),
                str(edge.edge_name[0]),
                color="white",
            )
            ax[1].text(
                *np.flip((1 - weight) * self.pos[edge.edge_name[1]] + weight * self.pos[edge.edge_name[0]]),
                str(edge.edge_name[1]),
                color="white",
            )
            ax[1].set_title("Extracted Edges")
        fig.tight_layout()
        plt.show()
        if save_img:
            save_path_temp = os.path.join(self.kymos_path, f"Detected edges.png")
            plt.savefig(save_path_temp)
            print("Saved the extracted edges")
        return None

    def makeVideo(self, resize_ratio = 4):
        vid_out = os.path.join(self.kymos_path, f"{self.video_nr}_video.mp4")

        if os.path.exists(vid_out):
            os.remove(vid_out)

        with imageio.get_writer(vid_out, mode='I', fps=self.fps, quality=6) as writer:
            for img_path in self.selection_file:
                img = imageio.imread(img_path)
                image_resize = img[::resize_ratio, ::resize_ratio]
                assert img.dtype == np.uint8, "Must be uint8 array!"
                writer.append_data(image_resize)
        # video_array = [imageio.imread(img) for img in self.selection_file]
        # imageio.mimwrite(os.path.join(self.kymos_path, f"{self.video_nr}_video.mp4"), video_array, quality=6,
        #                  fps=self.fps)

        return None

    def createEdge(self, edge):
        """
        Creates an edge analysis object
        """
        return Kymo_edge_analysis(self, edge_name=edge)

    def fit_backgr(self, img_seq, plots=False):
        # backgr_segm = np.invert(self.segmented)
        backgr_segm = self.segmented
        video_array = [imageio.imread(img) for img in self.selection_file]
        backgr_ints = [np.mean(img[backgr_segm].flatten()) for img in video_array]

        """
        Fit function for photobleaching. First assumes an exponential function with offset. 
        If the fitted function returns a positive exponent, will do a linear fit instead. 
        """
        s_fit = np.zeros((len(backgr_ints)))
        for i in range(len(backgr_ints)):
            if i > 0:
                s_fit[i] = s_fit[i-1] + 0.5*(backgr_ints[i] + backgr_ints[i-1])
        t_fit = np.arange(len(backgr_ints))
        matr_1 = np.array([[np.sum(s_fit**2), np.sum(s_fit * t_fit), np.sum(s_fit)],
                           [np.sum(s_fit * t_fit), np.sum(t_fit**2), np.sum(t_fit)],
                           [np.sum(s_fit), np.sum(t_fit), len(backgr_ints)]])
        vect_1 = np.array([np.sum(s_fit*backgr_ints),
                           np.sum(t_fit*backgr_ints),
                           np.sum(backgr_ints)])
        answ_1 = np.linalg.inv(matr_1).dot(vect_1)
        if answ_1[0] > 0:
            print("Using linear fit")
            (m,b) = np.polyfit(t_fit, backgr_ints, 1)
            self.back_fit = m*t_fit + b
        else:
            matr_2 = np.array([[len(backgr_ints), np.sum(np.exp(answ_1[0]*t_fit))],
                               [np.sum(np.exp(answ_1[0]*t_fit)), np.sum(np.exp(2*answ_1[0]*t_fit))]])
            vect_2 = np.array([np.sum(backgr_ints), np.sum(np.exp(answ_1[0]*t_fit) * backgr_ints)])
            answ_2 = np.linalg.inv(matr_2).dot(vect_2)

            self.back_fit = [answ_2[0], answ_2[1], answ_1[0]]
            self.back_fit = answ_2[0] + answ_2[1]*np.exp(answ_1[0]*t_fit)
            self.back_offset = answ_2[0]
            print(f'{answ_2[0]:.4} + {answ_2[1]:.4} * exp({answ_1[0]:.4}x)')
        if plots:
            fig, ax = plt.subplots()
            ax.plot(backgr_ints, label="Average backgr intens")
            ax.plot(self.back_fit, label="Fitted exp w/ offset")
            ax.set_title("Intensity of background over time")
            ax.set_ylabel("Intensity")
            ax.set_xlabel("Time (frames)")
            ax.legend()
            fig.tight_layout()
        return self.back_fit


class Kymo_edge_analysis(object):
    def __init__(self, video_analysis=None, edge_name=None, kymo=None):
        """
        Initialises an edge object, usually done by kymo_analysis object, but can also be initialised with a kymo image.
        Will contain a copy of the video analysis object for video parameters.
        """
        if kymo is None:
            self.video_analysis = video_analysis
            self.edge_name = edge_name
            self.offset = int(np.linalg.norm(
                self.video_analysis.pos[self.edge_name[0]] - self.video_analysis.pos[self.edge_name[1]])) // 4
            self.kymo = []
            self.kymos = []
            self.edge_path = os.path.join(self.video_analysis.kymos_path, f"edge {self.edge_name}")
            if not os.path.exists(self.edge_path):
                os.makedirs(self.edge_path)

            self.space_pixel_size = None
            self.time_pixel_size = None
        else:
            if len(kymo.shape) == 2:
                self.kymo = kymo
            self.kymos = [kymo]
            self.edge_name = (-1, -1)
            self.space_pixel_size = 1.0
            self.time_pixel_size = 1.0
        self.filtered_left = []
        self.filtered_right = []
        self.slices = []
        self.segments = []
        self.bounds = (0, 1)
        self.edge_array = []
        self.speeds_tot = []
        self.times = []
        self.imshow_extent = []

    # def extract_photobleach(self):
    #     edge_vid_intensity = [np.sum(imageio.imread(img).flatten()) for img in self.video_analysis.selection_file]
    #     bleach_trend = np.polyfit(np.arange(len(self.video_analysis.selection_file)), np.log(edge_vid_intensity), 1)
    #     return bleach_trend[0]
    #     # for frame_address in self.video_analysis.selection_file:
    #     #     edge_vid_intensity = [np.sum()]

    def view_edge(self,
                  resolution=1,
                  step=30,
                  target_length=130,
                  save_im=True,
                  bounds=(0, 1),
                  img_frame=0,
                  quality=6):
        """
        Function to either output a snaphot of an edge or a video of an edge. Image will output to a plt plot.
        Video will be saved in the analysis folder.
        This function recalculates the segments,
        so make sure the target length is equal to that of the edge extraction plot.

        resolution:     Not sure what it does, but it generates pivot indices.
        step:           Used to calculate orientation of profiles, smaller steps will produce more wobbly images.
        target_length:  Width of eventual image or video
        save_im:        Boolean whether the image or video should be saved
        bounds:         Fraction-based cutoff for where to start imaging. Probably interferes with target-length.
        img_frame:      Which frame index to use for imaging. Will produce a video if this is an array.
        quality:        Imageio parameter on video quality. 6 is a good balance of compression and file size.
        """
        self.edge_array = get_edge_image(self.edge_name,
                                         self.video_analysis.pos,
                                         self.video_analysis.selection_file,
                                         self.video_analysis.nx_graph_pruned,
                                         resolution,
                                         self.offset,
                                         step,
                                         target_length,
                                         img_frame,
                                         bounds,
                                         logging=self.video_analysis.logging)
        # fig, ax = plt.subplots()
        # ax.imshow(self.edge_array)

        if save_im:
            if np.ndim(img_frame) == 0:
                fig, ax = plt.subplots()
                ax.imshow(self.edge_array, aspect=1,
                          extent=[0, self.video_analysis.space_pixel_size * self.edge_array.shape[1], 0,
                                  self.video_analysis.space_pixel_size * self.edge_array.shape[0]])
                ax.set_title(f"Edge snapshot {self.edge_name}")
                ax.set_xlabel("space ($\mu m$)")
                ax.set_ylabel("space ($\mu m$)")

                im = Image.fromarray(self.edge_array.astype(np.uint8))
                save_path_temp = os.path.join(self.edge_path, f"{self.edge_name} snapshot.png")
                im.save(save_path_temp)
                if self.video_analysis.logging:
                    print('Saved an image of the edge')
                return [self.edge_array]
            elif np.ndim(img_frame) == 1:
                imageio.mimwrite(os.path.join(self.edge_path,
                                              f'{self.edge_name} edge_video.mp4'),
                                 self.edge_array,
                                 quality=quality,
                                 fps=self.video_analysis.fps)
            else:
                print("Input image sequence has a weird amount of dimensions. This will probably crash")
        return self.edge_array

    def create_segments(self, pos, image, nx_graph_pruned, resolution, offset, target_length, bounds):
        """
        Internal function which generates coordinates for kymograph and edge videos
        """
        self.slices, self.segments = extract_section_profiles_for_edge(
            self.edge_name,
            pos,
            image,
            nx_graph_pruned,
            resolution=resolution,
            offset=offset,
            target_length=target_length,
            bounds=bounds
        )

        return self.segments

    def extract_multi_kymo(self,
                           bin_nr,
                           resolution=1,
                           step=30,
                           target_length=130,
                           subtr_backg = False,
                           kymo_adj= False,
                           save_array=True,
                           save_im=True,
                           bounds=(0, 1),
                           plots=False):
        """
        Creates kymograph for the edge. Uses bin_nr to divide the width into evenly distributed strips.
        Kymographs are stored in the object, and images will be stored in the analysis folder.

        bin_nr:         Number of evenly distributed width strips to make kymogrpahs from
        resolution:     Something related to section creation
        step:           Quality parameter in calculating the orientation of the edges
        target_length:  Width of edge to make kymographs from, really should be just the internal length.
        save_array:     Boolean on whether to make a .npy file of the array
        save_im:        Boolean on whether to save the kymographs as images
        bounds:         Fraction-based limit on edge width, probably interferes with target_length.
        """
        bin_edges = np.linspace(bounds[0], bounds[1], bin_nr + 1)
        if self.space_pixel_size is None:
            space_pixel_size = self.video_analysis.space_pixel_size  # um.pixel
            time_pixel_size = self.video_analysis.time_pixel_size  # s.pixel
        else:
            space_pixel_size = self.space_pixel_size
            time_pixel_size = self.time_pixel_size
        self.kymos = np.array([self.extract_kymo(bounds=(bin_edges[i],
                                                bin_edges[i + 1]),
                                        resolution=resolution,
                                        step=step,
                                        target_length=target_length,
                                        save_array=save_array,
                                        save_im=save_im,
                                        img_suffix=str(bin_nr) + ' ' + str(i + 1),
                                        kymo_adjust=kymo_adj,
                                        x_len= space_pixel_size)
                      for i in tqdm(range(bin_nr))])
        if subtr_backg:
            for kymo in self.kymos:
                dimmest_pix = kymo < np.percentile(kymo, 1)
                dim_value = np.mean(kymo[dimmest_pix].flatten())
                kymo -= dim_value
        if plots:
            if self.space_pixel_size is None:
                space_pixel_size = self.video_analysis.space_pixel_size  # um.pixel
                time_pixel_size = self.video_analysis.time_pixel_size  # s.pixel
                self.imshow_extent = [0, self.video_analysis.space_pixel_size * self.kymos[0].shape[1],
                                      self.video_analysis.time_pixel_size * self.kymos[0].shape[0], 0]
            else:
                space_pixel_size = self.space_pixel_size
                time_pixel_size = self.time_pixel_size
                self.imshow_extent = [0, self.space_pixel_size * self.kymos[0].shape[1],
                                      self.time_pixel_size * self.kymos[0].shape[0], 0]
            fig, ax = plt.subplots(1, bin_nr, figsize=(8, 8), sharey='row')
            bin_space = np.linspace(0, 1, bin_nr + 1)
            for j in range(bin_nr):
                if bin_nr == 1:
                    ax = [ax]
                ax[j].imshow(self.kymos[j], aspect='auto', extent=self.imshow_extent, vmin=0)
                ax[j].set_title(f"Kymo [{bin_space[j]}-{bin_space[j + 1]}]")
                ax[j].set_xlabel("space ($\mu m$)")
                ax[j].set_ylabel("time ($s$)")
            fig.tight_layout()
        return self.kymos

    def extract_kymo(self,
                     resolution=1,
                     step=30,
                     target_length=130,
                     save_array=True,
                     save_im=True,
                     bounds=(0, 1),
                     img_suffix="",
                     x_len=10,
                     kymo_adjust = False):
        self.kymo = get_kymo_new(self.edge_name,
                                 self.video_analysis.pos,
                                 self.video_analysis.selection_file,
                                 self.video_analysis.nx_graph_pruned,
                                 resolution,
                                 self.offset,
                                 step,
                                 target_length,
                                 bounds,
                                 x_len)
        if kymo_adjust:
            adj_params = self.video_analysis.back_fit
            norm_exp   = self.video_analysis.back_fit / self.video_analysis.back_fit[0]
            self.kymo -= self.video_analysis.back_offset
            self.kymo  = np.array([self.kymo[i] / norm_exp[i] for i in range(self.kymo.shape[0])])

        if save_array:
            save_path_temp = os.path.join(self.edge_path, f"{self.edge_name} {img_suffix} kymo.npy")
            np.save(save_path_temp, self.kymo)
        if save_im:
            im = Image.fromarray(self.kymo.astype(np.uint8))
            save_path_temp = os.path.join(self.edge_path, f"{self.edge_name} {img_suffix} kymo.png")
            im.save(save_path_temp)
        return self.kymo

    def fourier_kymo(self, bin_nr, hor_lines=1, return_self=True, plots=False, test_plots=False, save_im=True, save_array=True):
        """
        Internal function which takes a kymograph and produces a forward and backward filtered kymograph.
        These are stored in the object.
        """

        if len(self.kymos) > 0:
            self.filtered_left = np.array([filter_kymo_left(kymo, plots=test_plots) for kymo in self.kymos])
            self.filtered_right = np.array(
                [np.flip(filter_kymo_left(np.flip(kymo, axis=1), plots=test_plots), axis=1) for kymo in self.kymos])
        if plots:
            fig, ax = plt.subplots(4, bin_nr, figsize=(8, 32), sharey='row')

            if bin_nr == 1:
                for i in range(4):
                    ax[i] = [ax[i]]
            for i in range(bin_nr):
                ax[0][i].imshow(self.kymos[i], aspect='auto', extent=self.imshow_extent, vmin=0)
                ax[0][i].set_title(f"Kymo of edge {self.edge_name}")
                ax[0][i].set_xlabel("space ($\mu m$)")
                ax[0][i].set_ylabel("time ($s$)")

                ax[1][i].imshow(self.filtered_left[i], aspect='auto', extent=self.imshow_extent, vmin=0)
                ax[1][i].set_title("Backward filter")
                ax[1][i].set_xlabel("space ($\mu m$)")
                ax[1][i].set_ylabel("time ($s$)")

                ax[2][i].imshow(self.filtered_right[i], aspect='auto', extent=self.imshow_extent, vmin=0)
                ax[2][i].set_title("Forward filter")
                ax[2][i].set_xlabel("space ($\mu m$)")
                ax[2][i].set_ylabel("time ($s$)")

                ax[3][i].imshow(self.filtered_left[i] + self.filtered_right[i], aspect='auto',
                                extent=self.imshow_extent, vmin=0)
                ax[3][i].set_title("Forw+Backw")
                ax[3][i].set_xlabel("space ($\mu m$)")
                ax[3][i].set_ylabel("time ($s$)")

                if save_array:
                    save_path_temp = os.path.join(self.edge_path, f"{self.edge_name} {bin_nr} {i + 1} kymo left.npy")
                    np.save(save_path_temp, self.filtered_left[i])
                    save_path_temp = os.path.join(self.edge_path, f"{self.edge_name} {bin_nr} {i + 1} kymo right.npy")
                    np.save(save_path_temp, self.filtered_right[i])
                    save_path_temp = os.path.join(self.edge_path,
                                                  f"{self.edge_name} {bin_nr} {i + 1} kymo filtered.npy")
                    np.save(save_path_temp, self.filtered_right[i] + self.filtered_left[i])

                if save_im:
                    im = Image.fromarray(
                        cv2.normalize(self.filtered_left[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(
                            np.uint8))
                    save_path_temp = os.path.join(self.edge_path, f"{self.edge_name} {bin_nr} {i + 1} kymo left.png")
                    im.save(save_path_temp)
                    im = Image.fromarray(cv2.normalize(self.filtered_right[i], None, alpha=0, beta=255,
                                                       norm_type=cv2.NORM_MINMAX).astype(np.uint8))
                    save_path_temp = os.path.join(self.edge_path, f"{self.edge_name} {bin_nr} {i + 1} kymo right.png")
                    im.save(save_path_temp)
                    im = Image.fromarray(
                        cv2.normalize(self.filtered_left[i] + self.filtered_right[i], None, alpha=0, beta=255,
                                      norm_type=cv2.NORM_MINMAX).astype(np.uint8))
                    save_path_temp = os.path.join(self.edge_path,
                                                  f"{self.edge_name} {bin_nr} {i + 1} kymo filtered.png")
                    im.save(save_path_temp)
            fig.tight_layout()

        if return_self:
            return self.filtered_left, self.filtered_right
        else:
            return None

    def test_GST(self, w_size, w_start=3, C_thresh=0.90, blur_size=7, C_thresh_falloff=0.02, speed_thresh=10, preblur=True, plots=False):

        """Initialize the speed array and time array, as well as the bin values for the window size, and fourier filters. if not already done so"""
        times = []
        speeds_tot = []
        discrete_bounds = np.linspace(w_start, w_size*2 + w_start, w_size + 1)
        w=0
        if len(self.filtered_left) == 0:
            self.fourier_kymo(1, return_self=False)

        """Get the real pixel space and time sizes either from the video, or set them as one if testing"""
        if self.space_pixel_size is None:
            space_pixel_size = self.video_analysis.space_pixel_size  # um.pixel
            time_pixel_size = self.video_analysis.time_pixel_size  # s.pixel
            self.imshow_extent = [0, self.video_analysis.space_pixel_size * self.kymos[0].shape[1],
                                  self.video_analysis.time_pixel_size * self.kymos[0].shape[0], 0]
        else:
            space_pixel_size = self.space_pixel_size
            time_pixel_size = self.time_pixel_size
            self.imshow_extent = [0, self.space_pixel_size * self.kymos[0].shape[1],
                                  self.time_pixel_size * self.kymos[0].shape[0], 0]

        """Iterate over each kymograph, over the forward and backward directions, and over the window sizes"""
        for j, kymo in enumerate(self.kymos):
            if plots:
                fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            """Set up the times axis, should be identical every iteration, also the forward/backward speed arrays"""
            times.append(np.array(range(kymo.shape[0])) * time_pixel_size)
            spds_both = [[], []]
            for i in [0, 1]:

                """Set up speed array for this iteration, also select either forward or backward filtered kymograph"""
                spd_stack = []
                kymo_interest = [self.filtered_right[j], self.filtered_left[j]][i]

                """Preblur gets rid of some gaussian noise in the kymograph"""
                if preblur:
                    kymo_interest = cv2.GaussianBlur(kymo_interest, (blur_size, blur_size), 0)

                """
                imgCoherency is an array with all the coherency and orientations of any given window size,
                imgCoherencySum is meant to document at each pixel which window size provides the highest image coherency
                imgGSTMax is supposed to contain the orientations of the pixels with the highest intensity,
                In the setting up phase, we put nans where the threshold is lower than desired.
                
                Then in the window iterations, we check for what the max image coherency is, 
                and change the image orientation to that.
                """

                imgCoherency = np.array([calcGST(kymo_interest, w) for w in range(w_start, w_size * 2 + w_start, 2)])
                imgCoherencySum = 1*np.greater(imgCoherency[0][0], C_thresh)
                imgCoherencySum = np.where(imgCoherencySum==1, 0, np.nan)
                imgGSTMax = np.where(imgCoherencySum == 0, imgCoherency[0][1], np.nan)
                
                for w in range(1, w_size):
                    C_thresh_current = C_thresh - C_thresh_falloff * w
                    coherency_interest = np.where(imgCoherency[w][0] > C_thresh_current, imgCoherency[w][0], np.nan)
                    imgCoherencySum = np.where(coherency_interest > imgCoherency[w-1][0], w, imgCoherencySum)
                    newValues = np.isnan(imgCoherencySum) * (np.invert(np.isnan(coherency_interest)))
                    imgCoherencySum = np.where(newValues, w, imgCoherencySum)
                    imgGSTMax = np.where(imgCoherencySum == w, imgCoherency[w][1], imgGSTMax)

                """
                The speed is thresholded based on the max speed threshold, and that we don't expect forward speeds in the 
                backwards filtered image and vice versa.
                """

                speed_unthr = (
                        np.tan((imgGSTMax - 90) / 180 * np.pi)
                        * space_pixel_size
                        / time_pixel_size
                        )
                speed = np.where(speed_unthr < speed_thresh, speed_unthr, np.nan)
                speed = np.where(speed > -1 * speed_thresh, speed, np.nan)
                spd_interest = np.where([speed < 0, speed > 0][i], speed, np.nan)
                spds_both[i] = spd_interest
                spd_stack.append(spd_interest)

                if plots:
                    ax[1][i].imshow(spds_both[i], aspect='auto', vmin=-speed_thresh, vmax=speed_thresh, cmap='bwr')
                    ax[1][i].set_title("Speeds")
                    ax[1][i].set_xlabel("space ($\mu m $)")
                    ax[1][i].set_ylabel("time ($s$)")
                    lay_plot = ax[0][i].imshow(kymo_interest, aspect='auto', vmin=0)
                    ax[0][i].set_title(
                        f"Filtered Fourier")
                    ax[0][i].set_xlabel("space ($\mu m $)")
                    ax[0][i].set_ylabel("time ($s$)")

            speeds_tot.append(spds_both)
            self.speeds_tot = np.array(speeds_tot)
            self.times = times

            if plots:
                fig.tight_layout()
        return np.array(speeds_tot), times

    def extract_speeds(self,
                       speed_thresh=20.0,
                       plots=False,
                       speedplot=False,
                       save_im=True,
                       save_array=True,
                       preblur=True,
                       limit_filter=False,
                       w=3,
                       c_thr=0.95,
                       klen=25,
                       magnification=2 * 1.725,
                       binning=1,
                       fps=1,
                       padding=0):
        """
        Creates graphs of speeds from kymographs.

        speed_thresh:       Maximum speed threshold in micrometers per second. Replaces higher speeds with NaNs.
        plots:              Boolean on whether to plot speeds, image coherency and other debug plots
        speedplot:          Boolean on whether to just plot speed
        preblur:            Blur filter option. Needs work
        w:                  kernel size window when calculating the Gradient Speed Tensor
        c_thr:              Lower threshold on image coherency. Coherency values lower than this will be NaNs
        klen:               kernel size which blurs image coherency, will create vignette in speed
        magnification:      Magnification of imaging setup
        binning:            Binning of imaging setup
        fps:                Frames per second of the imaging setup
        """
        if len(self.filtered_left) == 0:
            self.fourier_kymo(return_self=False)
        speeds_tot = []
        times = []
        kernel = np.ones((klen, klen)) / klen ** 2
        speed_dataframe = pd.DataFrame()

        if self.space_pixel_size is None:
            space_pixel_size = self.video_analysis.space_pixel_size  # um.pixel
            time_pixel_size = self.video_analysis.time_pixel_size  # s.pixel
            self.imshow_extent = [0, self.video_analysis.space_pixel_size * self.kymos[0].shape[1],
                                  self.video_analysis.time_pixel_size * self.kymos[0].shape[0], 0]
        else:
            space_pixel_size = self.space_pixel_size
            time_pixel_size = self.time_pixel_size
            self.imshow_extent = [0, self.space_pixel_size * self.kymos[0].shape[1],
                                  self.time_pixel_size * self.kymos[0].shape[0], 0]

        for j, kymo in enumerate(self.kymos):
            if plots:
                fig1, ax1 = plt.subplots(5, 2, figsize=(8, 8 * 4))
            nans = np.empty(kymo.shape)
            nans.fill(np.nan)
            speeds = [[], []]
            for i in [0, 1]:
                if i == 0:
                    times.append(np.array(range(kymo.shape[0])) * time_pixel_size)
                kymo_interest = [self.filtered_left[j], self.filtered_right[j]][i]

                # Increases image coherency a bit, while decreasing noise
                if preblur:
                    kymo_interest = cv2.GaussianBlur(kymo_interest, (7, 7), 0)

                # Measure how much the pixels adhere to a structure, and select high coherence
                imgCoherency, imgOrientation = calcGST(kymo_interest, w)
                real_movement = np.where(imgCoherency > c_thr, imgOrientation, nans)
                speed_unthr = (
                        np.tan((real_movement - 90) / 180 * np.pi)
                        * space_pixel_size
                        / time_pixel_size
                )

                # Filter based on expected speed values
                speed = np.where(speed_unthr < speed_thresh, speed_unthr, nans)
                speed = np.where(speed > -1 * speed_thresh, speed, nans)
                if limit_filter:
                    speed = np.where([speed < 0, speed > 0][i], speed, nans)

                # Add vignette to speed
                z1 = scipy.signal.convolve2d(imgCoherency, kernel, mode="same")
                speed = np.where(z1 > 0.6, speed, nans)
                if padding > 0:
                    speed_interp = pd.DataFrame(speed)
                    speed_interp = speed_interp.interpolate(limit=padding)
                    speed = speed_interp.to_numpy()

                # Create pandas dataframe with data
                label = self.edge_name if i == 0 else None
                speeds[i] = speed
                direction = np.array(["root" if i == 0 else "tip" for k in range(speed.shape[0])])
                edges_list = np.array([str(self.edge_name) for k in range(speed.shape[0])])
                data = pd.DataFrame(
                    np.transpose((times[j], np.nanmean(speeds[i], axis=1), edges_list, direction)),
                    columns=["time (s)", "speed (um.s-1)", "edge", "direction"],
                )
                speed_dataframe = pd.concat((speed_dataframe, data))

                if plots:
                    ax1[0][i].imshow(kymo_interest, aspect='auto', extent=self.imshow_extent)
                    ax1[0][i].set_title("Kymo{} {} {}".format(self.edge_name, str(j + 1), ["back", "forward"][i]))
                    ax1[0][i].set_xlabel("space ($\mu m$)")
                    ax1[0][i].set_ylabel("time ($s$)")

                    ax1[1][i].imshow(imgCoherency, vmin=0.0, vmax=1.0, aspect='auto', extent=self.imshow_extent)
                    ax1[1][i].set_title(f"Coherency {self.edge_name} {str(j + 1)}, {['back', 'forward'][i]}")
                    ax1[1][i].set_xlabel("space ($\mu m $)")
                    ax1[1][i].set_ylabel("time ($s$)")

                    sp_unthr = ax1[2][i].imshow(speed_unthr, vmin=-1 * speed_thresh, vmax=speed_thresh, aspect='auto',
                                                extent=[-speed_thresh / 2, speed_thresh / 2, int(times[j][-1]), 0],
                                                cmap='bwr')
                    ax1[2][i].plot(np.nanmean(speed_unthr, axis=1), times[j])
                    ax1[2][i].set_title(
                        "Speed_unthr {} {} {}".format(self.edge_name, str(j + 1), ["back", "forward"][i]))
                    ax1[2][i].grid(True)
                    ax1[2][i].set_xlabel("speed ($\mu m / s$)")
                    ax1[2][i].set_ylabel("time ($s$)")

                    sp_thr = ax1[3][i].imshow(speed, vmin=-speed_thresh, vmax=speed_thresh, aspect='auto',
                                              extent=[-speed_thresh / 2, speed_thresh / 2, int(times[j][-1]), 0],
                                              cmap='bwr')
                    ax1[3][i].plot(np.nanmean(speed, axis=1), times[j])
                    ax1[3][i].set_title("Speed_thr {} {} {}".format(self.edge_name, str(j + 1), ["back", "forward"][i]))
                    ax1[3][i].grid(True)
                    ax1[3][i].set_xlabel("speed ($\mu m / s$)")
                    ax1[3][i].set_ylabel("time ($s$)")

            if plots:
                sp_tot = ax1[4][0].imshow(np.nansum(speeds, axis=0), vmin=-speed_thresh, vmax=speed_thresh,
                                          aspect='auto',
                                          extent=[-speed_thresh / 2, speed_thresh / 2, int(times[j][-1]), 0],
                                          cmap='bwr')
                ax1[4][0].plot(np.nanmean(np.nansum(speeds, axis=0), axis=1), times[j])
                ax1[4][0].grid(True)
                ax1[4][0].set_title(f"Summed speeds {self.edge_name} {(j + 1)}")
                ax1[4][0].set_xlabel("speed ($\mu m / s$)")
                ax1[4][0].set_ylabel("time ($s$)")

                ax1[4][1].imshow(self.filtered_left[j] + self.filtered_right[j], aspect='auto',
                                 extent=self.imshow_extent)
                ax1[4][1].set_title(f"Added filters {self.edge_name} {(j + 1)}")
                ax1[4][1].set_xlabel("space ($\mu m$)")
                ax1[4][1].set_ylabel("time ($s$)")
                plt.colorbar(sp_unthr)
                plt.colorbar(sp_thr)
                plt.colorbar(sp_tot)

            speeds_tot.append(speeds)
            if plots:
                fig1.tight_layout()
            if save_im:
                fig1.savefig(f'{self.edge_path}/{self.edge_name} {len(self.kymos)} {j + 1} speedfig.png')

        # if speedplot:
        #     fig2.tight_layout()
        self.speeds_tot = np.array(speeds_tot)
        self.times = times

        return np.array(speeds_tot), times

    def extract_transport(self,
                          noise_thresh=0.01,
                          GST_window=5,
                          speed_thresh=20,
                          c_thresh=0.95,
                          margin=25,
                          plots=False,
                          histos=False,
                          save_filters=True,
                          save_speeds=True,
                          save_im=True,
                          save_flux_array=True,
                          photobleach_adjust=True):
        """
        Function that extracts the net transport of internal kymographs.

        noise_thresh: Simple threshold below which everything is considered noise, and set to zero
        GST_window:     Parameter for speed extraction, decides how big the GST kernel is.
        speed_thresh:   Parameter for speed extraction, decides maximum absolute speed.
        c_thresh:       Parameter for speed extraction, removes all datapoints where image coherency is below it
        margin:         Cutoff in pixels to account for the vignette created during speed extraction
        plots:          Boolean on whether to create plots
        save_filters:   Boolean on whether to save the filtered kymographs as an image
        save_speeds:    Boolean on whether to save the speed data as a pd dataframe
        photobleach_adjust: Boolean on whether to calculate a fluorescence falloff curve, and to adjust kymographs based on it.
        """

        if self.space_pixel_size is None:
            space_pixel_size = self.video_analysis.space_pixel_size  # um.pixel
            time_pixel_size = self.video_analysis.time_pixel_size  # s.pixel
            self.imshow_extent = [0, self.video_analysis.space_pixel_size * self.kymos[0].shape[1],
                                  self.video_analysis.time_pixel_size * self.kymos[0].shape[0], 0]
        else:
            space_pixel_size = self.space_pixel_size
            time_pixel_size = self.time_pixel_size
            self.imshow_extent = [0, self.space_pixel_size * self.kymos[0].shape[1],
                                  self.time_pixel_size * self.kymos[0].shape[0], 0]
        if self.video_analysis.vid_type == "BRIGHT":
            photobleach_adjust = False
        if len(self.filtered_left) == 0:
            print("Filtering kymos")
            self.fourier_kymo(return_self=False)
        if self.speeds_tot is None:
            print("Collecting speeds")
            self.extract_speeds(speed_thresh=speed_thresh, c_thr=c_thresh, plots=plots, w=GST_window, preblur=True)

        kernel = np.ones((5, 5), np.uint8) / 5 ** 2
        spd_max = np.nanmax(abs(self.speeds_tot.flatten()))

        if photobleach_adjust:
            img_seq = np.arange(len(self.video_analysis.selection_file))
            edge_video = self.view_edge(img_frame=img_seq, save_im=False, quality=6)
            edge_vid_intensity = [np.sum(frame.flatten()) for frame in edge_video]
            bleach_trend = np.polyfit(img_seq, np.log(edge_vid_intensity), 1)[0]
            bleach_plot = np.exp(bleach_trend * img_seq)
            inv_bleach_plot = 1 / bleach_plot
            if bleach_plot[-1] < 0:
                print("WEIRD ERROR IN BLEACH NORMALIZATION, FINAL VALUE IS LESS THAN ZERO")
            if plots:
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.plot(self.times[0], edge_vid_intensity / edge_vid_intensity[0], label='total intensity sum')
                ax.plot(self.times[0], bleach_plot, label='exp fit')
                ax.set_title(f"Photobleaching effect {self.edge_name}")
                ax.set_xlabel("time ($s$)")
                ax.set_ylabel("Normalized falloff")
                ax.legend()
                fig.tight_layout()

        for k, kymo in enumerate(self.kymos):
            back_thresh, forw_thresh = (self.filtered_right[k], self.filtered_left[k])
            forw, back = forw_thresh, back_thresh
            forw_back_thresh = np.add(forw_thresh, back_thresh)

            if photobleach_adjust:
                kymo_adj = np.array([kymo[i] * inv_bleach_plot[i] for i in range(len(inv_bleach_plot))])
                back_thresh = np.array([back_thresh[i] * inv_bleach_plot[i] for i in range(len(inv_bleach_plot))])
                forw_thresh = np.array([forw_thresh[i] * inv_bleach_plot[i] for i in range(len(inv_bleach_plot))])
                forw_back_thresh = np.array(
                    [forw_back_thresh[i] * inv_bleach_plot[i] for i in range(len(inv_bleach_plot))])
            else:
                kymo_adj = kymo

            if plots:
                img_max = np.max(kymo_adj.flatten())

                fig, ax = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
                fig.suptitle(f"kymo {self.edge_name} {k+1} of {len(self.kymos)}")

                ax[0][0].imshow(kymo_adj, aspect='auto', extent=self.imshow_extent, vmin=0, vmax=img_max)
                ax[0][0].set_title(f"Original")
                ax[0][0].set_xlabel("space ($\mu m$)")
                ax[0][0].set_ylabel("time ($s$)")

                ax[0][1].imshow(forw_back_thresh, aspect='auto', extent=self.imshow_extent, vmin=0, vmax=img_max)
                ax[0][1].set_title("Removed static")
                ax[0][1].set_xlabel("space ($\mu m$)")
                ax[0][1].set_ylabel("time ($s$)")

                ax[1][0].imshow(back_thresh, aspect='auto', extent=self.imshow_extent, vmin=0, vmax=img_max)
                ax[1][0].set_title("Back filter")
                ax[1][0].set_xlabel("space ($\mu m$)")
                ax[1][0].set_ylabel("time ($s$)")

                ax[1][1].imshow(forw_thresh, aspect='auto', extent=self.imshow_extent, vmin=0, vmax=img_max)
                ax[1][1].set_title("Forward filter")
                ax[1][1].set_xlabel("space ($\mu m$)")
                ax[1][1].set_ylabel("time ($s$)")

                fig.tight_layout()

            speeds = self.speeds_tot

            spds_back = speeds[k][0]
            spds_forw = speeds[k][1]

            iters = 1

            spds_tot = np.nansum(np.dstack((spds_back, spds_forw)), 2)
            flux_tot = np.nansum((np.prod((spds_forw, forw_thresh), 0), np.prod((spds_back, back_thresh), 0)), 0)
            flux_max = np.max(abs(flux_tot.flatten()))

            forw_back_thresh_int = np.sum(forw_back_thresh, axis=0)
            net_trans = np.array([np.nancumsum(flux_tot.transpose()[i][margin:-margin]) for i in
                                  range(margin, flux_tot.shape[1] - 1 * margin)]).transpose()

            if save_filters:
                im_left = Image.fromarray((forw_thresh * 255 / np.max(forw_thresh)).astype(np.uint8))
                im_right = Image.fromarray((back_thresh * 255 / np.max(back_thresh)).astype(np.uint8))
                im_full = Image.fromarray((forw_back_thresh * 255 / np.max(forw_back_thresh)).astype(np.uint8))
                save_path_temp = os.path.join(self.edge_path, f"{self.edge_name} {k} kymo_left.png")
                im_left.save(save_path_temp)
                save_path_temp = os.path.join(self.edge_path, f"{self.edge_name} {k} kymo_right.png")
                im_right.save(save_path_temp)
                save_path_temp = os.path.join(self.edge_path, f"{self.edge_name} {k} kymo_filtered.png")
                im_full.save(save_path_temp)

            if plots:
                if histos:
                    fig, ax = plt.subplots(3, figsize=(8, 8), sharey=True, sharex=True)
                    ax[0].hist(back.flatten(), bins=50, log=True, label="pre-shift")
                    ax[0].hist(back_thresh.flatten(), bins=50, log=True, label="post-shift", alpha=0.5)
                    ax[0].set_title("Backward hist")
                    ax[1].hist(forw.flatten(), bins=50, log=True, label="pre-shift")
                    ax[1].hist(forw_thresh.flatten(), bins=50, log=True, label="post-shift", alpha=0.5)
                    ax[1].set_title("Forward hist")
                    ax[2].hist(kymo_adj.flatten(), bins=50, label='original', log=True)
                    ax[2].hist(forw_back_thresh.flatten(), bins=50, alpha=0.5, label='filtered', log=True)
                    ax[2].set_title("Total hist comparison")
                    for i in range(3):
                        ax[i].set_xlabel("Pixel intensity")
                        ax[i].set_ylabel("Log frequency")
                        ax[i].legend()
                    fig.tight_layout()

                fig, ax = plt.subplots(2, 2, figsize=(9, 9), sharey='row')
                ax[0][0].imshow(np.prod((spds_back, back_thresh), 0), vmin=-flux_max, vmax=flux_max, aspect='auto',
                                extent=self.imshow_extent, cmap='bwr')
                ax[0][1].imshow(np.prod((spds_forw, forw_thresh), 0), vmin=-flux_max, vmax=flux_max, aspect='auto',
                                extent=self.imshow_extent, cmap='bwr')
                for i in [0, 1]:
                    ax[0][i].set_title(f"flux {['backward', 'forward'][i]}")
                    ax[0][i].set_xlabel("space ($\mu m$)")
                    ax[0][i].set_ylabel("time ($s$)")

                ax[1][0].imshow(net_trans / forw_back_thresh_int[margin:-margin], vmin=-1, vmax=1, cmap='bwr',
                                aspect='auto', extent=self.imshow_extent)
                ax[1][0].set_xlabel("space ($\mu m$)")
                ax[1][0].set_ylabel("time ($s$)")
                ax[1][0].set_title("Net transport")
                ax[1][1].plot(np.sum(net_trans / forw_back_thresh_int[margin: -margin], axis=1) / (space_pixel_size * net_trans.shape[1]),
                              self.times[0][margin:-margin])
                ax[1][1].set_title("Mean net transport")

                fig.tight_layout()

                if save_im:
                    fig.savefig(f'{self.edge_path}/{self.edge_name} {len(self.kymos)} {k + 1} fluxfig')

            if save_flux_array:
                np.save(f'{self.edge_path}/{self.edge_name} {len(self.kymos)} {k + 1} fluxback.npy',
                        np.prod((spds_back, back_thresh), 0))
                np.save(f'{self.edge_path}/{self.edge_name} {len(self.kymos)} {k + 1} fluxforw.npy',
                        np.prod((spds_forw, forw_thresh), 0))

                # if len(self.kymos) > 1:
                #     ax_spd[0][k].imshow(spds_tot,
                #                         vmin=-1*abs(np.max(spds_tot.flatten())),
                #                         vmax=abs(np.max(spds_tot.flatten())),
                #                         cmap='bwr')
                #     ax_spd[1][k].imshow(flux_tot,
                #                         vmin=-1 * abs(np.max(flux_tot.flatten())),
                #                         vmax=abs(np.max(flux_tot.flatten())),
                #                         cmap='bwr'
                #                         )
                #
                #     ax_trans[k].plot((net_trans[-1] - net_trans[0])/forw_back_thresh_int[margin:-margin])
                #     ax_trans[k].set_xlabel("space (x)")

        return speeds
