import os
import unittest
from test.util import helper

from amftrack.pipeline.launching.run import run_function
import os
from amftrack.util.sys import (
    get_dates_datetime,
    get_dirname,
    temp_path,
    get_data_info,
    update_plate_info,
    update_analysis_info,
    get_analysis_info,
    get_current_folders,
    get_folders_by_plate_id,
)

from time import time_ns

from amftrack.util.dbx import upload_folders, load_dbx, download
from datetime import datetime
from amftrack.pipeline.launching.run_super import (
    run_parallel,
    directory_scratch,
    directory_project,
    run_parallel_stitch,
)
from amftrack.util.dbx import (
    read_saved_dropbox_state,
    get_dropbox_folders_prince,
    get_dropbox_folders_prince,
    save_dropbox_state,
    get_dropbox_folders_general_recursive,
)
from amftrack.pipeline.scripts.image_processing import (
    mask_skel,
    extract_skel_2,
    detect_blob,
    prune_skel,
    extract_nx_graph,
    extract_width,
    track_nodes,
    make_labeled_graphs,
    extract_skel_no_external,
    final_alignment_new,
    realign_new
)


class TestImageAnalysis(unittest.TestCase):
    """Tests that need only a static plate with one timestep"""

    @classmethod
    def setUpClass(cls):
        cls.folders = helper.make_folders()
        cls.directory = helper.test_path

    def test_create_script(self):
        helper.create_script_function("extract_skel_2.py")

    def test_skeletonize(self):
        hyph_width = 30
        perc_low = 85
        perc_high = 99.5
        minlow = 10
        minhigh = 70

        args = [None, hyph_width, perc_low, perc_high, minlow, minhigh, self.directory]
        run_function(extract_skel_2.process, args, self.folders)

    def test_skeletonize_no_exeternal(self):
        hyph_width = 30
        perc_low = 85
        perc_high = 99.5
        minlow = 10
        minhigh = 70

        args = [None, hyph_width, perc_low, perc_high, minlow, minhigh, self.directory]
        run_function(extract_skel_no_external.process, args, self.folders)

    def test_spore(self):
        args = [None, self.directory]
        run_function(detect_blob.process, args, self.folders)

    def test_mask(self):
        thresh = 40
        args = [None, thresh, self.directory]
        run_function(mask_skel.process, args, self.folders)

    def test_prune(self):
        threshold = 0.01 / 1.725
        skip = False
        args = [None, threshold, skip, self.directory]
        run_function(prune_skel.process, args, self.folders[:1])

    def test_realign(self):
        args = [None, self.directory]
        run_function(
            final_alignment_new.process,
            args,
            self.folders[:4],
            sequential_process = True
        )
    def test_create_realign(self):
        args = [None, self.directory]
        run_function(
            realign_new.process,
            args,
            self.folders[:4],
            per_unique_id = True
        )
    def test_graph_extract(self):
        args = [None, self.directory]
        run_function(extract_nx_graph.process, args, self.folders[:1])

    def test_width_extract(self):
        skip = "False"
        resolution = "10"
        args = [None, self.directory, skip, resolution]
        run_function(extract_width.process, args, self.folders[:1])

    def test_node_id(self):
        args = [None, self.directory]
        run_function(track_nodes.process, args, self.folders)

    def test_make_labeled(self):
        args = [None, self.directory]
        run_function(
            make_labeled_graphs.process, args, self.folders, per_unique_id=True
        )
