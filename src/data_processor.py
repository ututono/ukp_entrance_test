import os
from typing import List

import pandas as pd
import numpy as np
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils.global_variables import TEST_FILENAME, TRAIN_FILENAME, DEV_FILENAME, ENCODING, DATA_COL_NAMES, WORD_COL, \
    NER_COL, NEGLECT_TAGS, ENDING_PUNCTUATIONS, DATA_OPT
from src.utils.utils import root_path, get_embedding_dim, read_csv_file
from src.utils.basic_logger import setup_logger

logger = setup_logger(__name__, level='DEBUG')


def extract_sent_labels(data_df: pd.DataFrame):
    """
    Extract sentences and labels from data_df. A sentence is **a list of words**, and a label is a list of tags.
    :param data_df: DataFrame read from the data file
    :return: sentences, labels
    """
    sentences, sentence = [], []
    labels, label = [], []
    for word, tag in zip(data_df[WORD_COL].tolist(), data_df[NER_COL].tolist()):
        if word in ENDING_PUNCTUATIONS:
            sentence.append(word)
            label.append(tag)
            sentences.append(sentence)
            labels.append(label)
            sentence, label = [], []
        else:
            sentence.append(word)
            label.append(tag)
    return sentences, labels


class DataProcessor:
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=4, vectorize=None, dtype=np.float32,
                 device=None):
        self._data_dir = data_dir
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._num_workers = num_workers
        self.EMBEDDING_DIM = get_embedding_dim()
        self._vocab = None
        self._labels = None
        self._label2index = None  # Mapping from label to index
        self._vectorize_word = vectorize if vectorize else self.default_vectorize_word
        self._dtype = dtype
        self._tensor_type = torch.long
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if device is None else device

        # Initialize vocab and labels
        self._init_vocab()

    def _init_vocab(self, min_freq=1):
        """
        Initialize the vocab and labels. vocab is a dict, whose key is the word and value is the index of the word.
        labels is a list of all the tags from NER column. label2index is initialized as well.
        :param min_freq: Word with frequency less than min_freq will be neglected
        """
        x_data = Counter()
        y_target = Counter()
        for data_df in self._load_raw_data():
            x_data.update(data_df[WORD_COL].tolist())
            y_target.update(data_df[NER_COL].tolist())

        x_data = dict(filter(lambda x: x[1] >= min_freq, x_data.items()))

        alpha = list(x_data.keys())

        self._vocab = {w: i + 1 for i, w in enumerate(alpha)}
        for tag in NEGLECT_TAGS:
            if tag in self.vocab:
                self.vocab[tag] = 0

        self._labels = list(y_target.keys())

        label2index = {l: i + 1 for i, l in enumerate(self._labels)}
        label2index['PAD'] = 0
        self._label2index = label2index

    def load_embedding(self):
        raise NotImplementedError

    def default_vectorize_word(self, sentence: List[str]):
        """
        Convert a sentence to a tensor
        :param sentence:
        :return:
        """
        default_vector = np.zeros(self.EMBEDDING_DIM, self._dtype)
        return [self.vocab.get(word, default_vector) for word in sentence]

    def process_data(self):
        """
        Load data from data_dir and process it, finally return a dataset
        :return:
        """
        raise NotImplementedError

    def get_dataset(self, filename: str) -> TensorDataset:
        self._check_attributes_none(['vocab'])

        labels_tensors = []  # Each element is a list of indices of labels from a sentence
        sentences_tensors = []  #
        sentences_lengths = []  # Each element is the length of a sentence

        file_df = read_csv_file(os.path.join(self._data_dir, filename), DATA_OPT)

        sentences, labels_list = extract_sent_labels(file_df)

        for sent, labels in zip(sentences, labels_list):
            label_vec = self._vectorize_label_set(labels)
            words_vec = self._vectorize_word(sent)
            sent_length = len(sent)
            sentences_lengths.append(torch.tensor(sent_length, dtype=self._tensor_type, device=self._device))
            sentences_tensors.append(torch.tensor(words_vec, dtype=self._tensor_type, device=self._device))
            labels_tensors.append(torch.tensor(label_vec, dtype=self._tensor_type, device=self._device))

        sentences_lengths = torch.stack(sentences_lengths)
        x_tensor = torch.nn.utils.rnn.pad_sequence(sentences_tensors, batch_first=True)
        y_tensor = torch.nn.utils.rnn.pad_sequence(labels_tensors, batch_first=True)

        dataset = TensorDataset(x_tensor, sentences_lengths, y_tensor)
        return dataset

    def get_dataloader(self, filename: str, num_workers=None) -> DataLoader:
        num_workers = self._num_workers if num_workers is None else num_workers
        dataset = self.get_dataset(filename)
        return DataLoader(dataset, batch_size=self._batch_size, shuffle=self._shuffle, drop_last=True)

    def _vectorize_label_set(self, labels: List[str]):
        return list(self._label2index.get(label) for label in labels)

    def _load_raw_data(self):
        for filename in [TRAIN_FILENAME, DEV_FILENAME, TEST_FILENAME]:
            df = read_csv_file(os.path.join(self._data_dir, filename), DATA_OPT)
            yield df

    def _check_attributes_none(self, attribute_names: List[str]):
        for attr in attribute_names:
            if getattr(self, attr) is None:
                raise ValueError(f"The attribute {attr} is None")

    @property
    def vocab(self):
        return self._vocab
