import os
from datetime import datetime
import sys
import torch

from src.data_processor import DataProcessor
from src.evaluator import Evaluator
from src.model import VanillaBiLSTMTagger
from src.trainer import Trainer
from src.utils_.global_variables import SEED, DATA_DIR_NAME, TRAIN_FILENAME, DEV_FILENAME, MODEL_NAME, \
    TRAIN_CONFIG_NAME, LOGGING_LEVEL, TEST_FILENAME
from src.utils_.utils import root_path, seed_random_generators, embedding_path, get_embedding_dim, embed_vocab, \
    get_ckpt_dir, get_train_params, update_param_dict, dict2json, get_eval_params
from src.settings import parse_arguments
from src.utils_.basic_logger import setup_logger

logger = setup_logger(__name__, LOGGING_LEVEL)
logger.info(f"Current working directory: {os.getcwd()}")


def pipeline_train(train_params):
    batch_size = train_params["batch_size"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = train_params["seed"]
    learning_rate = train_params["learning_rate"]
    loss_name = train_params["loss"]
    optimizer_name = train_params["optimizer"]
    epochs = train_params["epochs"]

    # Set random seed
    seed_random_generators(seed)

    # Instantiate data processor
    data_processor = DataProcessor(os.path.join(root_path(), DATA_DIR_NAME), batch_size)
    tags_size = data_processor.num_tags

    # Get dataloader
    train_dataloader = data_processor.get_dataloader(TRAIN_FILENAME, 0)
    dev_dataloader = data_processor.get_dataloader(DEV_FILENAME, 0)

    # Get embedding layer
    embedding_dim = get_embedding_dim()
    embedding = embed_vocab(embed_path=embedding_path(), vocab=data_processor.vocab)
    embedding = embedding.to(device)

    # Instantiate model
    model = VanillaBiLSTMTagger(embeddings=embedding, embedding_dim=embedding_dim, hidden_dim=100,
                                tagset_size=tags_size)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError(f"{optimizer_name} is not implemented yet")
    if loss_name == "cross_entropy":
        criteria = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"{loss_name} is not implemented yet")

    # Instantiate trainer to train the model
    trainer = Trainer()
    trainer.train(model, train_dataloader, dev_dataloader, optimizer, criteria, epochs, data_processor.labels,
                  device)

    # Create checkpoint directory and save it to train_params
    ckpt_dir = get_ckpt_dir(date_time=datetime.now())
    update_param_dict(train_params, checkpoint=ckpt_dir, device=device)

    # Save model and training parameters
    model_save_path = os.path.join(ckpt_dir, MODEL_NAME)
    config_save_path = os.path.join(ckpt_dir, TRAIN_CONFIG_NAME)
    trainer.save_model(model_save_path)
    dict2json(train_params, config_save_path)
    logger.info(f"Model is saved to {model_save_path}")
    logger.info(f"Training parameters are saved to {config_save_path}")
    logger.info(f"Training is finished.")

def pipeline_test(eval_params):
    batch_size = eval_params["batch_size"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = eval_params["seed"]
    ckpt = eval_params["checkpoint"]

    # Set random seed
    seed_random_generators(seed)

    # Instantiate data processor
    data_processor = DataProcessor(os.path.join(root_path(), DATA_DIR_NAME), batch_size)

    # Get dataloader
    test_dataloader = data_processor.get_dataloader(TEST_FILENAME, 0)

    # Get embedding layer
    embedding_dim = get_embedding_dim()
    embedding = embed_vocab(embed_path=embedding_path(), vocab=data_processor.vocab)
    embedding = embedding.to(device)

    # Instantiate model
    model = VanillaBiLSTMTagger(embeddings=embedding, embedding_dim=embedding_dim, hidden_dim=100,
                                tagset_size=data_processor.num_tags)

    # Load model
    ckpt_dir = get_ckpt_dir(ckpt)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, MODEL_NAME)))

    # Instantiate evaluator to evaluate the model
    evaluator = Evaluator()
    cm = evaluator.run(model, test_dataloader, data_processor.labels, device, criteria=None)
    logger.info(f"Confusion matrix for test dataloader:\n{cm.get_all_metrics()}")



def main():
    args = parse_arguments()
    if args.mode == "train":
        train_params = get_train_params(args)
        pipeline_train(train_params)
    elif args.mode == "test":
        eval_params = get_eval_params(args)
        pipeline_test(eval_params)
    else:
        raise ValueError(f"Mode {args.mode} is not supported")


if __name__ == '__main__':
    main()
