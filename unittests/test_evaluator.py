import os
import unittest
import torch

from src.data_processor import DataProcessor
from src.evaluator import Evaluator
from src.model import VanillaBiLSTMTagger
from src.utils_.utils import root_path, seed_random_generators, get_embedding_dim, embed_vocab, embedding_path, \
    get_ckpt_dir
from src.utils_.global_variables import DATA_DIR_NAME, DEV_FILENAME, MODEL_NAME, SEED, TEST_FILENAME, LOGGING_LEVEL, \
    TRAIN_FILENAME
from src.utils_.basic_logger import setup_logger

logger = setup_logger(__name__, level=LOGGING_LEVEL)


class EvaluatorTestCase(unittest.TestCase):
    BATCH_SIZE = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = SEED
    ckpt = '2023_11_27-17_16_04'

    @classmethod
    def setUpClass(cls):
        data_processor = DataProcessor(os.path.join(root_path(), DATA_DIR_NAME), cls.BATCH_SIZE)
        seed_random_generators(cls.seed)
        cls.data_processor = data_processor
        cls.labels = data_processor.labels

    def test_load_model(self):
        embedding_dim = get_embedding_dim()
        embedding = embed_vocab(embed_path=embedding_path(), vocab=self.data_processor.vocab)
        tags_size = self.data_processor.num_tags

        embedding = embedding.to(self.device)

        model = VanillaBiLSTMTagger(embeddings=embedding, embedding_dim=embedding_dim, hidden_dim=100,
                                    tagset_size=tags_size)

        ckpt_dir = get_ckpt_dir(self.ckpt)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, MODEL_NAME)))

    def test_evaluate_model(self):
        labels = self.labels
        train_dataloader = self.data_processor.get_dataloader(TRAIN_FILENAME, 0, num_samples=10)
        dev_dataloader = self.data_processor.get_dataloader(DEV_FILENAME, 0, num_samples=10)
        test_dataloader = self.data_processor.get_dataloader(TEST_FILENAME, 0, num_samples=10)

        embedding_dim = get_embedding_dim()
        embedding = embed_vocab(embed_path=embedding_path(), vocab=self.data_processor.vocab)
        tags_size = self.data_processor.num_tags

        embedding = embedding.to(self.device)

        model = VanillaBiLSTMTagger(embeddings=embedding, embedding_dim=embedding_dim, hidden_dim=100,
                                    tagset_size=tags_size)

        ckpt_dir = get_ckpt_dir(self.ckpt)
        model.load_state_dict(torch.load(os.path.join(ckpt_dir, MODEL_NAME)))

        evaluator = Evaluator()
        cm = evaluator.run(model, train_dataloader, labels, self.device, criteria=None)
        logger.info(f"Confusion matrix for train dataloader:\n{cm.get_all_metrics()}")

        cm = evaluator.run(model, dev_dataloader, labels, self.device, criteria=None)
        logger.info(f"Confusion matrix for dev dataloader:\n{cm.get_all_metrics()}")

        cm = evaluator.run(model, test_dataloader, labels, self.device, criteria=None)
        logger.info(f"Confusion matrix for test dataloader:\n{cm.get_all_metrics()}")


if __name__ == '__main__':
    unittest.main()
