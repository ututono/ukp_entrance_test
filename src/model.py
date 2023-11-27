import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from src.utils.global_variables import START_TAG, STOP_TAG, LOGGING_LEVEL
from src.utils.utils import log_sum_exp, argmax
from src.utils.basic_logger import setup_logger

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


class LSTMClassifierCRF(nn.Module):
    def __init__(self, embeddings, num_classes, embed_dim, rnn_units, tag_to_ix, batch_size, rnn_layers=1, dropout=0.1,
                 hidden_units=[]):
        super().__init__()
        # self.embedding_dim = embedding_dim
        self.hidden_dim = rnn_units
        # self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix  # dictionary of tags
        self.tagset_size = len(tag_to_ix)
        self.batch_size = batch_size

        self.embeddings = embeddings
        self.lstm = nn.LSTM(embed_dim, rnn_units // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(rnn_units, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, inputs):
        one_hots, lengths = inputs
        self.hidden = self.init_hidden()
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        embeds = self.embeddings(one_hots)
        embeds = embeds.transpose(0, 1)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embeds, lengths.tolist())
        lstm_out, hidden = self.lstm(packed, self.hidden)
        # print(lstm_out.shape())
        lstm_outs, lstm_outs_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_outs)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long),
                          tags])  # Add START_TAG ahead of the origin tags
        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, inputs, tags):
        feats = self._get_lstm_features(inputs)

        feats = feats.view(-1, self.tagset_size)

        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, inputs):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(inputs)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
