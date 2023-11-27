import unittest

import time
import os

import src.utils.utils as utils
from src.data_processor import DataProcessor
from src.utils.utils import root_path, embedding_path

DATA_DIR_NAME = 'data'

class UtilsTestCase(unittest.TestCase):
    BATCH_SIZE = 32

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

    def test_embed_vocab(self):
        data_processor = DataProcessor(os.path.join(root_path(), DATA_DIR_NAME), self.BATCH_SIZE)
        utils.embed_vocab(embed_path=embedding_path(), vocab=data_processor.vocab)

    def test_convert_milliseconds_to_hms(self):
        actual_time = (0, 0, 0.001)
        self.assertEqual(actual_time, utils.convert_milliseconds_to_hms(1))



if __name__ == '__main__':
    unittest.main()
