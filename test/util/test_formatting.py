import unittest
from amftrack.util.formatting import str_to_coord_list, coord_list_to_str
from test.util import helper


class TestFormatting(unittest.TestCase):
    def test_str_to_coord_list(self):
        list1 = [[23, 12], [1.232134, 0.0]]
        list1_ = str_to_coord_list(coord_list_to_str(list1))
        list2 = [[2, 2], [1, 0]]
        list2_ = str_to_coord_list(coord_list_to_str(list2))
        self.assertTrue(helper.is_equal_seq(list1, list1_))
        self.assertTrue(helper.is_equal_seq(list2, list2_))
