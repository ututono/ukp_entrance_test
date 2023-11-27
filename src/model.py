import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils_.global_variables import START_TAG, STOP_TAG, LOGGING_LEVEL
from src.utils_.basic_logger import setup_logger

logger = setup_logger(__name__, LOGGING_LEVEL)


class VanillaBiLSTMTagger(nn.Module):
    def __init__(self, embeddings, embedding_dim, hidden_dim, tagset_size, batch_size=1):
        super(VanillaBiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = embeddings

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim,  hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.double()

    def forward(self, inputs):
        """

        :param inputs: include vectorised sentence and its length
        :return:
        """
        # Shape of one_hots: (batch_size, padded_sentence_length)
        one_hots, lengths = inputs
        embeds = self.word_embeddings(one_hots)
        embeds = embeds.transpose(0, 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths.tolist())

        # The shape of lstm out is (max seq_len in this batch, batch_size, hidden_dim)
        lstm_out, hidden = self.lstm(packed)
        lstm_outs, lstm_outs_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        tag_space = self.hidden2tag(lstm_outs)
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_space, lstm_outs_lengths