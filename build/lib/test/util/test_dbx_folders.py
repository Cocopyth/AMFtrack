import unittest
from amftrack.util.dbx import upload_folders

from amftrack.util.sys import (
    test_path,
    update_plate_info_local,
    get_current_folders_local,
)


class TestFolder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = test_path + "/"  # TODO(FK): fix this error
        # create an sparse or empty folder structure to avoid test running forever
        plate_name = "20220330_2357_Plate19_empty"
        update_plate_info_local(directory)
        folder_df = get_current_folders_local(directory)
        selected_df = folder_df.loc[folder_df["folder"] == plate_name]
        cls.selected_df = selected_df

    def test_upload(self):
        upload_folders(self.selected_df, "test", catch_exception=False)
