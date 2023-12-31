import json
import os
from typing import Tuple

import pandas as pd
from torch import nn
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

from src.utils_.global_variables import ENCODING, NEGLECT_TAGS, DATA_COL_NAMES, START_TAG, STOP_TAG, SEED, \
    OUTPUT_DIR_NAME


def root_path():
    return get_parent_path(os.path.abspath(__file__), 3)


def embedding_path():
    return os.path.join(root_path(), "pretrained", "glove.6B.50d.txt")


def get_parent_path(path, levels=1):
    """
    Returns the parent folder path of a given path, up a certain number of levels.

    :param path: The original file or folder path.
    :param levels: Number of levels up to move. Default is 1.
    :return: The parent folder path.
    """
    for _ in range(levels):
        path = os.path.dirname(path)
    return path


def embed_vocab(embed_path: str, vocab, uniform_bounding=0.25):
    """
    Embedding the vocabulary using the pretrained embeddings
    :param embed_path:
    :param vocab: dict. Mapping from word to index
    :param uniform_bounding:
    :return: torch.nn.Embedding
    """

    def init_embeddings_weight(vocab_size, embedding_dim, bounding):
        """Initializing neural network weights randomly"""
        return np.random.uniform(-bounding, bounding, (vocab_size, embedding_dim))

    with open(embed_path, 'r', encoding=ENCODING) as f:
        lines = f.readlines()
        embed_dim = len(lines[0].strip().split()) - 1

        weights = init_embeddings_weight(len(lines), embed_dim, uniform_bounding)

        # According to the pretrained embedding lookup table, the embedding of the words in the vocabulary(existing in the provided dataset) to the weights
        for line in lines:
            word, *vector = line.strip().split()
            if word in vocab:
                weights[vocab[word]] = np.array(vector, dtype=np.float32)

    # Set padding tag to zero
    for padding_tag in NEGLECT_TAGS:
        if padding_tag in vocab:
            weights[vocab[padding_tag]] = np.zeros(embed_dim)

    embeddings = nn.Embedding(num_embeddings=weights.shape[0], embedding_dim=weights.shape[1])
    embeddings.weight = nn.Parameter(torch.from_numpy(weights))

    return embeddings


def get_embedding_dim(embed_path: str = None):
    embed_path = embed_path if embed_path else embedding_path()
    check_path_exists(embed_path)
    with open(embed_path, 'r', encoding=ENCODING) as f:
        lines = f.readline()  # One line is enough
        embed_dim = len(lines.strip().split()) - 1
    return embed_dim


def check_path_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist")


def read_csv_file(file_path: str, option: str):
    """
    Read a csv file using different arguments according to the option, return a pandas dataframe
    :param file_path:
    :param option: str. ["data-conll"]
    :return:
    """
    check_path_exists(file_path)
    if option == "data-conll":
        df = pd.read_csv(file_path, delimiter='\t', encoding=ENCODING, names=DATA_COL_NAMES, skiprows=[0])
        df = df.dropna()
    else:
        raise ValueError("The option is invalid")
    return df


def get_label2index(labels):
    """
    Get the mapping from label to index
    :param labels: list of labels
    :return: dict. Mapping from label to index
    """
    # Map the NEG tag to the number
    label2index = {l: i for i, l in enumerate(labels)}
    return label2index


def get_ckpt_dir(date_time):
    if not isinstance(date_time, str):
        timestamp = date_time.strftime('%Y_%m_%d-%H_%M_%S')
    else:
        timestamp = date_time
    ckpt_dir = os.path.join(root_path(), OUTPUT_DIR_NAME, 'checkpoints', timestamp)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    else:
        if not isinstance(date_time, str):
            raise FileExistsError(f"{ckpt_dir} already exists")
    return ckpt_dir


def extract_valid_labels(mask, labels) -> Tuple:
    """Remove padding from prediction, only use valid labels for loss calculation"""
    valid = (mask.sum(dim=1))
    result = labels[mask].split(valid.tolist())
    return result


def transfer_set_tensors_to_numpy(set_tensors):
    """
    Transfer a set of tensors to numpy
    :param set_tensors:
    :return:
    """
    return [tensor.cpu().int().numpy() for tensor in set_tensors]


def seed_random_generators(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def convert_milliseconds_to_hms(milliseconds):
    seconds = milliseconds / 1000
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return hours, minutes, seconds


def convert_second_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return hours, minutes, seconds


def get_train_params(args, **kwargs) -> dict:
    train_params = {
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "device": None,
        "optimizer": args.optimizer,
        "loss": args.loss,
        "checkpoint": None,
        'mode': args.mode,
        "hidden_dim": args.hidden_dim,
        "num_samples": args.num_samples,
    }
    train_params.update(kwargs)
    return train_params


def get_eval_params(args, **kwargs) -> dict:
    eval_params = {
        "checkpoint": args.checkpoint,
        "batch_size": args.batch_size,
        "device": None,
        "loss": args.loss,
        "mode": args.mode,
        "seed": args.seed,
        "num_samples": args.num_samples,
    }
    eval_params.update(kwargs)
    return eval_params


def update_param_dict(train_params, **kwargs):
    train_params.update(kwargs)
    return train_params


def dict2json(obj: dict, save_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Attempt to write the dictionary to a JSON file
    try:
        with open(save_path, 'w', encoding=ENCODING) as f:
            json.dump(obj, f, indent=4, separators=(',', ': '))
    except IOError as e:
        # Handle the error (e.g., permission error, disk full)
        raise IOError(f"An error occurred while writing to the file: {e}")
    except TypeError as e:
        # Handle the error (e.g., object is not serializable)
        raise TypeError(f"An error occurred while dumping JSON: {e}")


def json2dict(json_path):
    # Attempt to read the JSON file
    try:
        with open(json_path, 'r', encoding=ENCODING) as f:
            obj = json.load(f)
    except IOError as e:
        # Handle the error (e.g., file not found)
        raise IOError(f"An error occurred while reading the file: {e}")
    except ValueError as e:
        # Handle the error (e.g., JSON syntax error)
        raise ValueError(f"An error occurred while parsing JSON: {e}")
    return obj


def permute_sequence_by_length(sequence, sequence_lengths):
    """
    Sort a sequence by length
    :param sequence: list
    :param sequence_lengths: list
    :return:
    """
    sorted_sequence_lengths, sorted_indices = sequence_lengths.sort(0, descending=True)
    sorted_sequence = sequence[sorted_indices]
    return sorted_sequence, sorted_sequence_lengths, sorted_indices


def init_cm_result_dict():
    """
    Initialize the confusion matrix result dictionary where stores the metrics during the training
    :param labels: list of labels
    :return: dict. The confusion matrix result dictionary
    """
    cm_result = {
        "acc": [],
        "mean_precision": [],
        "mean_recall": [],
        "macro_f1": [],
        "weighted_precision": [],
        "weighted_recall": [],
        "weighted_f1": [],
    }
    return cm_result


def visualize_cm(key: str, cms_result: dict, title, save_path: str = None):
    """
    Visualize the confusion matrix result
    :param title: str. The title of the plot
    :param key: str. The key of the confusion matrix result
    :param cms_result: dict. The confusion matrix result dictionary
    :return:
    """
    fig, axs = plt.subplots(1, 1)
    axs.plot(cms_result[key])
    axs.set_title(title)
    axs.set_xlabel('Epoch')
    axs.set_ylabel(key)
    if save_path:
        plt.savefig(save_path)
    plt.show()
