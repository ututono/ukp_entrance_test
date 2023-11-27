import os
import unittest
import random

from src.data_processor import DataProcessor
from src.utils_.global_variables import TRAIN_FILENAME
from src.utils_.utils import root_path

DATA_DIR_NAME = 'data'


class DataProcessorTestCase(unittest.TestCase):
    BATCH_SIZE = 32

    @classmethod
    def setUpClass(cls):
        data_processor = DataProcessor(os.path.join(root_path(), DATA_DIR_NAME), cls.BATCH_SIZE)
        cls.data_processor = data_processor

    def test_get_dataset(self):
        TRAIN_SENTENCE_NUM = 7389

        train_dataset = self.data_processor.get_dataset(TRAIN_FILENAME)
        random_index = random.randint(0, TRAIN_SENTENCE_NUM - 1)

        self.assertEqual(len(train_dataset), TRAIN_SENTENCE_NUM)

        # The padding length of the sentences should be the same
        self.assertEqual(train_dataset[2][0].shape, train_dataset[1][0].shape, train_dataset[random_index][0].shape)

        # The padding length of the labels should be the same
        self.assertEqual(train_dataset[2][2].shape, train_dataset[1][2].shape, train_dataset[random_index][2].shape)

    def test_get_dataloader(self):
        train_dataloader = self.data_processor.get_dataloader(TRAIN_FILENAME, 0)

        self.assertEqual(len(train_dataloader), 230)

        train_features, sent_lengths, train_labels = next(iter(train_dataloader))
        self.assertEqual(train_features.shape, train_labels.shape)
        self.assertEqual(train_features.shape[0], self.BATCH_SIZE)

    def test_vectorize_labels(self):
        LABELS_NUM = 9
        self.assertEqual(LABELS_NUM, self.data_processor.num_tags)

    def test_dataset_size(self):
        TRAIN_SENTENCE_NUM = 2
        train_dataset = self.data_processor.get_dataset(TRAIN_FILENAME, num_samples=TRAIN_SENTENCE_NUM)
        self.assertEqual(len(train_dataset), TRAIN_SENTENCE_NUM)


if __name__ == '__main__':
    unittest.main()
