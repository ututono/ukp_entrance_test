import os
from datetime import datetime

import torch

from src.data_processor import DataProcessor
from src.model import VanillaBiLSTMTagger
from src.trainer import Trainer
from src.utils.global_variables import SEED, DATA_DIR_NAME, TRAIN_FILENAME, DEV_FILENAME, MODEL_NAME
from src.utils.utils import root_path, seed_random_generators, embedding_path, get_embedding_dim, embed_vocab, \
    create_ckpt_dir


def train_model():
    BATCH_SIZE = 2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = SEED
    
    data_processor = DataProcessor(os.path.join(root_path(), DATA_DIR_NAME), BATCH_SIZE)
    seed_random_generators(seed)
    train_dataloader = data_processor.get_dataloader(TRAIN_FILENAME, 0)
    dev_dataloader = data_processor.get_dataloader(DEV_FILENAME, 0)

    embedding_dim = get_embedding_dim()
    embedding = embed_vocab(embed_path=embedding_path(), vocab=data_processor.vocab)
    tags_size = data_processor.num_tags

    embedding = embedding.to(device)

    model = VanillaBiLSTMTagger(embeddings=embedding, embedding_dim=embedding_dim, hidden_dim=100,
                                tagset_size=tags_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criteria = torch.nn.CrossEntropyLoss()
    epochs = 20

    trainer = Trainer()
    trainer.train(model, train_dataloader, dev_dataloader, optimizer, criteria, epochs, data_processor.labels,
                  device)

    ckpt_dir = create_ckpt_dir(date_time=datetime.now())
    save_path = os.path.join(ckpt_dir, MODEL_NAME)
    trainer.save_model(save_path)
    
    
    
def main():
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.current_device())
    train_model()


if __name__ == '__main__':
    main()