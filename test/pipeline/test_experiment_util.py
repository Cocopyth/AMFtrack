import os
import numpy as np
import random
import unittest
import matplotlib.pyplot as plt
from amftrack.util.geometry import expand_bounding_box, get_bounding_box
from amftrack.util.sys import (
    update_plate_info_local,
    get_current_folders_local,
    test_path,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Node,
    Edge,
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
    plot_edge_width,
)
from amftrack.util.sys import test_path
from test import helper
from PIL import Image


@unittest.skipUnless(helper.has_test_plate(), "No plate to run the tests..")
class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = helper.make_experiment_object()

    def test_get_random_edge(self):
        get_random_edge(self.exp)
        get_random_edge(self.exp)

    def test_get_edge_from_node_labels(self):
        # Valid edge
        edge = get_random_edge(self.exp)
        self.assertIsNotNone(
            get_edge_from_node_labels(self.exp, 0, edge.begin.label, edge.end.label)
        )
        self.assertIsNotNone(
            get_edge_from_node_labels(self.exp, 0, edge.end.label, edge.begin.label)
        )
        # Invalid case
        self.assertIsNone(get_edge_from_node_labels(self.exp, 0, 100, 100))

    def test_plot_edge(self):
        edge = get_random_edge(self.exp)
        plot_edge(edge, 0)
        plt.close()

    def test_plot_edge_save(self):
        edge = get_random_edge(self.exp)
        plot_edge(edge, 0, save_path=os.path.join(test_path, "plot_edge_1"))

    def test_plot_edge_cropped(self):
        edge = get_random_edge(self.exp)
        plot_edge_cropped(edge, 0, save_path=os.path.join(test_path, "plot_edge_2"))

    def test_plot_full_image_with_features(self):
        plot_full_image_with_features(
            self.exp,
            0,
            downsizing=10,
            region=None,
            points=[[11191, 39042], [11923, 45165]],
            segments=[[[11191, 39042], [11923, 45165]]],
            nodes=[Node(10, self.exp), Node(100, self.exp), Node(200, self.exp)],
            edges=[get_random_edge(self.exp), get_random_edge(self.exp)],
            dilation=1,
            save_path=os.path.join(test_path, "plot_full"),
            prettify= False
        )

    def test_plot_full_image_with_features_2(self):
        plot_full_image_with_features(
            self.exp,
            0,
            downsizing=5,
            region=[[0, 0], [20000, 55000]],
            points=[[11191, 39042], [11923, 45165]],
            segments=[[[11191, 39042], [11923, 45165]]],
            nodes=[Node(10, self.exp), Node(100, self.exp), Node(200, self.exp)],
            edges=get_all_edges(self.exp, 0),
            dilation=5,
            save_path=os.path.join(test_path, "plot1"),
        )
        plot_full_image_with_features(
            self.exp,
            0,
            downsizing=5,
            region=[[0, 0], [20000, 55000]],
            nodes=[Node(10, self.exp), Node(23, self.exp), Node(1, self.exp)],
            edges=get_all_edges(self.exp, 0)[10:],
            dilation=9,
            save_path=os.path.join(test_path, "plot2"),
        )

        plot_full_image_with_features(
            self.exp,
            0,
            region=[[10000, 20000], [25000, 35000]],
            downsizing=3,
            nodes=[Node(10, self.exp), Node(23, self.exp), Node(1, self.exp)],
            edges=get_all_edges(self.exp, 0)[10:],
            dilation=4,
            save_path=os.path.join(test_path, "plot3"),
        )

        plot_full_image_with_features(
            self.exp,
            0,
            region=[[16000, 20000], [25000, 30000]],
            downsizing=1,
            nodes=[Node(10, self.exp), Node(23, self.exp), Node(1, self.exp)],
            edges=get_all_edges(self.exp, 0),
            dilation=20,
            save_path=os.path.join(test_path, "plot4"),
        )

    def test_plot_full_image_with_features_3(self):

        plot_full_image_with_features(
            self.exp,
            0,
            region=[[18000, 20000], [25000, 30000]],
            downsizing=1,
            nodes=[Node(10, self.exp), Node(23, self.exp), Node(1, self.exp)],
            edges=get_all_edges(self.exp, 0),
            dilation=40,
            save_path=os.path.join(test_path, "plot5"),
        )

    def test_plot_full_image_with_features_4(self):

        plot_full_image_with_features(
            self.exp,
            0,
            region=[[18000, 20000], [25000, 30000]],
            downsizing=1,
            nodes=[Node(i, self.exp) for i in range(100)],
            edges=get_all_edges(self.exp, 0),
            dilation=40,
            save_path=os.path.join(test_path, "plot6"),
        )
    def test_plot_full_image_with_features_5(self):

        plot_full_image_with_features(
            self.exp,
            0,
            region=[[18000, 20000], [25000, 30000]],
            downsizing=1,
            nodes=[Node(10, self.exp), Node(23, self.exp), Node(1, self.exp)],
            edges=get_all_edges(self.exp, 0),
            dilation=40,
            save_path=os.path.join(test_path, "plot7"),
            prettify=True,

        )

    def test_get_all_edges(self):
        get_all_edges(self.exp, t=0)

    def get_nearest_edge(self):
        edge = get_random_edge(self.exp)
        edge_coords = edge.pixel_list(0)
        middle_coord = edge_coords[len(edge_coords) // 2]

        found_edge = find_nearest_edge(middle_coord, self.exp, 0)
        found_edges = find_neighboring_edges(
            middle_coord, self.exp, 0, n_nearest=5, step=50
        )

        self.assertEqual(edge, found_edge)
        self.assertIn(edge, found_edges)

    def test_reconstruct_image(self):
        region = [[10000, 20000], [20000, 40000]]

        # Full image downsized without region
        im, _ = reconstruct_image(self.exp, 0, downsizing=13)
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(test_path, "reconstruct_full_downsized.png"))

        # Region precise
        im, _ = reconstruct_image(self.exp, 0, region=region)
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(test_path, "reconstruct_region.png"))

        # Region downsized
        im, _ = reconstruct_image(self.exp, 0, region=region, downsizing=15)
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(test_path, "reconstruct_region_downsized.png"))

        # Backgroud check
        im, _ = reconstruct_image(self.exp, 0, downsizing=40, white_background=True)
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(test_path, "reconstruct_white_bg.png"))
        im, _ = reconstruct_image(self.exp, 0, downsizing=40, white_background=False)
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(test_path, "reconstruct_black_bg.png"))

    def test_reconstruct_skeletton_from_edges(self):
        # Several edges
        edges = [get_random_edge(self.exp) for _ in range(20)]
        im, _ = reconstruct_skeletton_from_edges(
            self.exp,
            0,
            edges=edges,
            region=None,
            color_seeds=None,
            downsizing=5,
            dilation=10,
        )
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(test_path, "reconstruct_squeletton_0.png"))
        # All edges
        im, _ = reconstruct_skeletton_from_edges(
            self.exp,
            0,
            edges=get_all_edges(self.exp, 0),
            region=None,
            color_seeds=None,
            downsizing=5,
            dilation=10,
        )
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(test_path, "reconstruct_squeletton_1.png"))
        # Try coloring
        all_edges = get_all_edges(self.exp, 0)
        im, _ = reconstruct_skeletton_from_edges(
            self.exp,
            0,
            edges=all_edges,
            region=None,
            color_seeds=[random.randint(0, 2) for i in range(len(all_edges))],
            downsizing=5,
            dilation=10,
        )
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(test_path, "reconstruct_squeletton_2.png"))
        # Try with region
        all_edges = get_all_edges(self.exp, 0)
        im, _ = reconstruct_skeletton_from_edges(
            self.exp,
            0,
            region=[[10000, 20000], [25000, 35000]],
            edges=all_edges,
            color_seeds=[random.randint(0, 2) for i in range(len(all_edges))],
            downsizing=10,
            dilation=20,
        )
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(test_path, "reconstruct_squeletton_3.png"))

    def test_squeletton_unicolor(self):
        all_edges = get_all_edges(self.exp, 0)
        im, f = reconstruct_skeletton_unicolor(
            [edge.pixel_list(0) for edge in all_edges],
            region=None,
            downsizing=10,
            dilation=10,
            foreground=255,
        )
        im_pil = Image.fromarray(im)
        im_pil.save(os.path.join(test_path, "reconstruct_squeletton_unicolor.png"))

    def test_reconstruct_image_2(self):
        # Verify that the ploting function works
        random.seed(6)  # 6, 11
        edge = get_random_edge(self.exp, 0)
        pixel_coord_ts = [
            self.exp.general_to_timestep(coord, 0) for coord in edge.pixel_list(0)
        ]
        region = expand_bounding_box(get_bounding_box(pixel_coord_ts), margin=100)
        im, f = reconstruct_image(
            self.exp, 0, downsizing=5, region=region, white_background=False
        )
        plt.imshow(im)
        for i, coord in enumerate(pixel_coord_ts):
            if i % 10 == 0:
                coord = f(coord)
                plt.plot(coord[1], coord[0], marker="x", color="red")
        plt.savefig(os.path.join(test_path, "test_plot_function.png"))

    def test_plot_edge_width(self):

        # With widths
        def f(edge):
            return random.random() * 10

        plot_edge_width(
            self.exp,
            0,
            width_fun=f,
            region=None,
            downsizing=5,
            nodes=[Node(10, self.exp), Node(23, self.exp), Node(1, self.exp)],
            dilation=10,
            save_path=os.path.join(test_path, "plot_width_with"),
        )

        # With no widths, interesting region
        plot_edge_width(
            self.exp,
            0,
            width_fun=lambda x: -2,
            region=[[12000, 15000], [26000, 35000]],
            downsizing=5,
            nodes=[Node(10, self.exp), Node(23, self.exp), Node(1, self.exp)],
            dilation=10,
            save_path=os.path.join(test_path, "plot_width_without"),
        )

    def test_plot_edge_width_2(self):
        # Try plotting all the nodes
        plot_edge_width(
            self.exp,
            0,
            width_fun=lambda x: -2,
            region=[[12000, 15000], [26000, 35000]],
            downsizing=5,
            nodes=get_all_nodes(self.exp, 0),
            dilation=10,
            save_path=os.path.join(test_path, "plot_width_and_nodes"),
        )

    def test_get_all_nodes(self):
        get_all_nodes(self.exp, 0)


class TestExperimentLight(unittest.TestCase):
    def test_distance_point_edge(self):

        edge = helper.EdgeMock([[2, 3], [3, 3], [3, 4], [4, 5], [5, 5], [6, 6], [7, 7]])
        self.assertEqual(
            distance_point_edge([2, 3], edge, 0, step=1),
            0,
        )
