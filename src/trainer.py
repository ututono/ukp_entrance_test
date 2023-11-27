import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import time
import sys
from src.utils_.confusion_matrix import ConfusionMatrix
from src.utils_.basic_logger import setup_logger
from src.utils_.global_variables import LOGGING_LEVEL
from src.utils_.utils import extract_valid_labels, transfer_set_tensors_to_numpy, convert_milliseconds_to_hms, \
    convert_second_to_hms

logger = setup_logger(__name__, level=LOGGING_LEVEL)


def process_bar(current, total, msg=None):
    """
    Display a process bar
    :param current: current progress
    :param total: total progress
    :param msg: message to display
    :return:
    """
    rate = float(current) / total
    ratenum = int(100 * rate)
    r = '\rModel training:[{}{}]{}%{}'.format('*' * ratenum, ' ' * (100 - ratenum), ratenum, msg)
    sys.stdout.write(r)
    sys.stdout.flush()


class Trainer:
    def __init__(self):
        self._model = None

    def train(self, model, train_dataloader, dev_dataloader, optimizer, loss_function, epochs, labels, device):
        """
        Train the model
        :param labels:
        :param model: model to train
        :param train_dataloader: dataloader for training data
        :param dev_dataloader: dataloader for dev data
        :param optimizer: optimizer
        :param loss_function: loss function
        :param epochs: number of epochs
        :param device: device to train on
        :return:
        """
        model = model.to(device)
        self._train_vanilla_model(model, train_dataloader, dev_dataloader, optimizer, loss_function, epochs, labels,
                                  device)

    def _train_vanilla_model(self, model, train_dataloader, dev_dataloader, optimizer, loss_function, epochs, labels,
                             device):

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            logger.info('-' * 100)
            self._train_one_epoch(model, train_dataloader, optimizer, loss_function, device, self._update_vanilla,
                                  labels)
            # logger.info(f"Dev confusion matrix:\n{self.evaluate(model, dev_dataloader, loss_function, device, labels)}")

        self._model = model

    def save_model(self, path):
        torch.save(self._model.state_dict(), path)

    def _train_one_epoch(self, model, train_dataloader, optimizer, loss_function, device, update_func, labels):
        log_interval = max(2, int(len(train_dataloader) * 0.01))
        cm = ConfusionMatrix(labels)
        len_train_dataloader = len(train_dataloader)
        epoch_loss = 0.
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        start_time.record()
        for batch_idx, batch in enumerate(train_dataloader):
            loss_value, y_pred, y_actual = update_func(model, loss_function, optimizer, batch, device)
            epoch_loss += loss_value
            yt = transfer_set_tensors_to_numpy(y_actual)
            yp = transfer_set_tensors_to_numpy(y_pred)
            cm.add_batch(yt, yp)
            if batch_idx % log_interval == 0 or batch_idx == len_train_dataloader - 1:
                if batch_idx != len_train_dataloader - 1:
                    msg = f" - loss: {loss_value:.6f}"
                else:
                    average_loss = epoch_loss / len_train_dataloader
                    end_time.record()
                    torch.cuda.synchronize()
                    elapsed_time = start_time.elapsed_time(end_time)
                    date_time = convert_milliseconds_to_hms(elapsed_time)
                    msg = f" - Mean loss: {average_loss:.6f} - Elapsed time: {date_time[0]}h:{date_time[1]}m:{date_time[2]}s\n"
                process_bar(batch_idx, len_train_dataloader, msg)

        logger.info(f"\nTraining confusion matrix:\n{cm.get_all_metrics()}")

    def _update_vanilla(self, model, criteria, optimizer, batch, device):
        """
        Perform a single training update on a given model batch for vanilla model.

        This function executes the forward pass, computes the loss, performs backpropagation, and updates the model parameters using the given optimizer.

        :return: loss, y_pred, y_target
        """
        optimizer.zero_grad()
        features, sent_lengths, labels = batch
        batch_size = features.shape[0]

        # Sort the features and labels by their length, the longest first in this batch
        sent_lengths, perm_idx = sent_lengths.sort(0, descending=True)
        features_sorted = features[perm_idx]
        labels_sorted = labels[perm_idx]

        y_ = labels_sorted.to(device)
        x_ = (features_sorted.to(device), sent_lengths.to(device))

        # Used for confusion matrix calculation
        tags_scores, tags_scores_lengths = model(x_)

        # Remove padding from prediction, only use valid labels for loss calculation
        # y_ = extract_valid_labels(mask=(y_ != 0), labels=y_)

        batch_loss = 0.
        y_pred = []
        y_target = []

        # Compute loss for each sentence in the batch
        for i in range(batch_size):
            golden_labels_sent_i = y_[i, :tags_scores_lengths[i]]
            tags_scores_sent_i = tags_scores[:tags_scores_lengths[i], i, :]
            predicted_labels_sent_i = torch.argmax(tags_scores_sent_i, dim=1)
            y_pred.append(predicted_labels_sent_i)
            y_target.append(golden_labels_sent_i)
            batch_loss += criteria(tags_scores_sent_i, golden_labels_sent_i)

        loss = batch_loss / batch_size
        loss.backward()
        optimizer.step()
        return loss.item(), y_pred, y_target
