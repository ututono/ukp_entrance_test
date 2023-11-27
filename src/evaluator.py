import torch
import torch.nn as nn

from src.utils_.confusion_matrix import ConfusionMatrix
from src.utils_.utils import transfer_set_tensors_to_numpy, permute_sequence_by_length
from src.utils_.basic_logger import setup_logger
from src.utils_.global_variables import LOGGING_LEVEL

logger = setup_logger(__name__, level=LOGGING_LEVEL)


class Evaluator:
    def run(self, model: nn.Module, data_loader, labels, device, criteria=None, print_cm=False):
        """

        :param print_cm: print the cm if True
        :param criteria: Only used for training while computing val_loss
        :param data_loader:
        :param labels:
        :param device:
        :return:
        """
        cm = ConfusionMatrix(labels)
        model.to(device)
        for batch_idx, batch in enumerate(data_loader):
            loss, y_pred, y_actual = self.inference(model, batch, device, criteria)
            yt = transfer_set_tensors_to_numpy(y_actual)
            yp = transfer_set_tensors_to_numpy(y_pred)
            cm.add_batch(yt, yp)
        if not criteria and print_cm:
            logger.info(f"Confusion matrix:\n{cm.get_all_metrics()}")

        return cm


    def inference(self, model: nn.Module, batch, device, criteria=None):
        """

        :param criteria:
        :param model:
        :param batch:
        :return: predictions, targets
        """
        model.eval()
        with torch.no_grad():
            features, sent_lengths, labels = batch
            batch_size = features.shape[0]

            # Sort the features and labels by their length, the longest first in this batch
            features_sorted, sent_lengths, perm_idx = permute_sequence_by_length(features, sent_lengths)
            labels_sorted = labels[perm_idx]

            y_ = labels_sorted.to(device)
            x_ = (features_sorted.to(device), sent_lengths.to(device))

            tags_scores, tags_scores_lengths = model(x_)

            y_pred = []
            y_target = []
            batch_loss = 0.

            for i in range(batch_size):
                golden_labels_sent_i = y_[i, :tags_scores_lengths[i]]
                tags_scores_sent_i = tags_scores[:tags_scores_lengths[i], i, :]
                predicted_labels_sent_i = torch.argmax(tags_scores_sent_i, dim=1)
                y_pred.append(predicted_labels_sent_i)
                y_target.append(golden_labels_sent_i)
                if criteria:
                    batch_loss += criteria(tags_scores_sent_i, golden_labels_sent_i)

            loss = (batch_loss / batch_size).item() if criteria else 0

            return loss, y_pred, y_target






