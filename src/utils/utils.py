import os

import pandas as pd
from torch import nn
import numpy as np
import torch

from src.utils.global_variables import ENCODING, NEGLECT_TAGS, DATA_COL_NAMES


def root_path():
    return get_parent_path(os.path.abspath(__file__), 3)


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

    :param embed_path:
    :param vocab:
    :param uniform_bounding:
    :return: torch.nn.Embedding, embedding_dim
    """

    def init_embeddings(vocab_size, embedding_dim, bounding):
        """Initializing neural network weights randomly"""
        return np.random.uniform(-bounding, bounding, (vocab_size, embedding_dim))

    with open(embed_path, 'r') as f:
        lines = f.readlines()
        embed_dim = len(lines[0].strip().split()) - 1

        weights = init_embeddings(len(lines), embed_dim, uniform_bounding)

        for line in lines:
            word, *vector = line.strip().split()
            weights[vocab[word]] = np.array(vector, dtype=np.float32)

    # Set padding tag to zero
    for padding_tag in NEGLECT_TAGS:
        if padding_tag in vocab:
            weights[vocab[padding_tag]] = np.zeros(embed_dim)

    embeddings = nn.Embedding(num_embeddings=weights.shape[0], embedding_dim=weights.shape[1])
    embeddings.weight = nn.Parameter(torch.from_numpy(weights))

    return embeddings, weights.shape[1]


def get_embedding_dim(embed_path: str = None):
    embed_path = embed_path if embed_path else os.path.join(root_path(), "pretrained", "glove.6B.50d.txt")
    check_path_exists(embed_path)
    with open(embed_path, 'r') as f:
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
    if option == "data-conll":
        df = pd.read_csv(file_path, delimiter='\t', encoding=ENCODING, names=DATA_COL_NAMES, skiprows=[0])
        df = df.dropna()
    else:
        raise ValueError("The option is invalid")
    return df
