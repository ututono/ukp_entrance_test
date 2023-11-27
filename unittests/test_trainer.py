import os
import unittest
from datetime import datetime

import torch

from src.data_processor import DataProcessor
from src.utils_.utils import root_path, get_embedding_dim, embed_vocab, embedding_path, get_ckpt_dir, seed_random_generators
from src.utils_.global_variables import DATA_DIR_NAME, TRAIN_FILENAME, DEV_FILENAME, MODEL_NAME, SEED
from src.trainer import Trainer
from src.model import VanillaBiLSTMTagger


class TrainerTestCase(unittest.TestCase):
    BATCH_SIZE = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = SEED

    @classmethod
    def setUpClass(cls):
        data_processor = DataProcessor(os.path.join(root_path(), DATA_DIR_NAME), cls.BATCH_SIZE)
        seed_random_generators(cls.seed)
        cls.data_processor = data_processor

    def test_model_train(self):
        train_dataloader = self.data_processor.get_dataloader(TRAIN_FILENAME, 0, num_samples=10)
        dev_dataloader = self.data_processor.get_dataloader(DEV_FILENAME, 0, num_samples=10)

        embedding_dim = get_embedding_dim()
        embedding = embed_vocab(embed_path=embedding_path(), vocab=self.data_processor.vocab)
        tags_size = self.data_processor.num_tags

        embedding = embedding.to(self.device)

        model = VanillaBiLSTMTagger(embeddings=embedding, embedding_dim=embedding_dim, hidden_dim=100,
                                    tagset_size=tags_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criteria = torch.nn.CrossEntropyLoss()
        epochs = 1

        trainer = Trainer()
        trainer.train(model, train_dataloader, dev_dataloader, optimizer, criteria, epochs, self.data_processor.labels,
                      self.device)


if __name__ == '__main__':
    unittest.main()
