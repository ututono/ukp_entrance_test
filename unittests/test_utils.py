import sys
import unittest

import time
import os

import src.utils_.utils as utils
from src.data_processor import DataProcessor
from src.utils_.global_variables import OUTPUT_DIR_NAME, VAL_CM_NAME
from src.utils_.utils import root_path, embedding_path

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

    def test_convert_second_to_hms(self):
        actual_time = (0, 0, 1)
        self.assertEqual(actual_time, utils.convert_second_to_hms(1))

        SLEEP_TIME = 1
        start_time = time.time()
        time.sleep(SLEEP_TIME)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.assertAlmostEqual(elapsed_time, SLEEP_TIME, delta=1)
        date_time = utils.convert_second_to_hms(elapsed_time)
        self.assertEqual("Elapsed time: 0h:0m:1s", f"Elapsed time: {date_time[0]}h:{date_time[1]}m:{date_time[2]}s")

    def test_get_ckpt_dir(self):
        mock_ckpt_dir = os.path.join(root_path(), OUTPUT_DIR_NAME, 'checkpoints', '2023_11_27-17_16_04')
        self.assertEqual(mock_ckpt_dir, utils.get_ckpt_dir(date_time='2023_11_27-17_16_04'))

    def test_visualize_cm(self):
        ckpt_dir = utils.get_ckpt_dir(date_time='2023_11_28-03_37_47')
        val_cm = utils.json2dict(os.path.join(ckpt_dir, VAL_CM_NAME))
        utils.visualize_cm('macro_f1', val_cm, os.path.join(ckpt_dir, 'val_marco_f1.png'))


if __name__ == '__main__':
    unittest.main()
