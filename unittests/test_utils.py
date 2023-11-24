import unittest

import time
import os

import src.utils.utils as utils

class UtilsTestCase(unittest.TestCase):
    def test_get_root_path(self):
        actual_root_path = utils.get_parent_path(os.path.abspath(__file__), 2)
        self.assertEqual(actual_root_path, utils.root_path())

    def test_get_embedding_dim(self):
        actual_dim = 50
        start_time = time.time()
        obtained_dim = utils.get_embedding_dim()
        end_time = time.time()

        print("Time elapsed: {}".format(end_time - start_time))
        self.assertEqual(actual_dim, obtained_dim)



if __name__ == '__main__':
    unittest.main()
