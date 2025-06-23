import torch
from torch import nn
import torch.nn.functional as F

from model.layers.span_representation import SpanAttention
from model.losses import BinaryFocalLoss


class MentionDetector(nn.Module):
    """
    Module for detecting mentions in text.

    Args:
        hidden_size (int): The size of the hidden layer. Default is 768.

    Attributes:
        span_rep_layer (SpanAttention): The span attention layer used for generating span representations.
        span_decoder (nn.Sequential): The sequential layer used for decoding span representations.

    """

    def __init__(self, hidden_size=768, max_width=4):
        super(MentionDetector, self).__init__()
        self.span_rep_layer = SpanAttention(
            hidden_size=hidden_size, max_width=max_width
        )
        self.span_decoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        # self.loss = nn.BCEWithLogitsLoss()
        self.loss = BinaryFocalLoss(alpha=0.25, gamma=2.0)
        self.threshold = 0.5

    def forward(self, embeddings, span_idx, span_mask, span_labels, entity_pos):
        """
        Forward pass.

        Args:
            embeddings (torch.Tensor): The input embeddings.
            span_idx (torch.Tensor): The indices of the spans.
            span_mask (torch.Tensor): The mask indicating valid spans.
            span_labels (torch.Tensor): The labels for the spans.

        Returns:
            torch.Tensor: The logits.
            torch.Tensor: The loss.
            float: The precision.
            float: The recall.
            float: The F1 score.

        """
        span_idx = span_idx * span_mask.unsqueeze(-1)
        span_representations = self.span_rep_layer(embeddings, span_idx)
        logits = self.span_decoder(span_representations)
        logits = logits.squeeze(-1)

        logits = logits.view(span_labels.size())

        predicted_labels = (torch.sigmoid(logits) >= 0.5).long()
        predicted_entity_pos = self.get_predicted_entity_pos(predicted_labels, span_idx)
        if self.threshold != 0.5:
            joint_predicted_labels = (torch.sigmoid(logits) >= self.threshold).long()
            predicted_entity_pos = self.get_predicted_entity_pos(
                joint_predicted_labels, span_idx
            )
        logits = logits.view(-1)
        span_labels = span_labels.view(-1)
        loss = self.loss(logits.float(), span_labels.float())
        predicted_labels = predicted_labels.view(-1)
        return predicted_labels, predicted_entity_pos, loss

    def get_predicted_entity_pos(self, predicted_labels, span_idx):
        """
        Get the predicted entity positions based on the given logits and span indices.

        Args:
            logits (torch.Tensor): The logits tensor.
            span_idx (torch.Tensor): The tensor containing the span indices.

        Returns:
            List[List[Tuple[int, int]]]: A list of lists of tuples representing the predicted entity positions.
        """
        predicted_entity_pos = []

        for b in range(predicted_labels.size(0)):
            b_predicted_entity_pos = []
            for predicted_label, span in zip(
                predicted_labels[b].tolist(), span_idx[b].tolist()
            ):
                if predicted_label == 1:
                    b_predicted_entity_pos.append((span[0], span[1]))
            predicted_entity_pos.append(b_predicted_entity_pos)
        return predicted_entity_pos

    def compute_metrics(self, predicted_entity_pos, entity_pos):
        """
        Compute recall, precision, and F1 score between predicted entity clusters and ground truth entity clusters.

        Args:
            predicted_clusters (list): List of predicted entity clusters.
            ground_truth_clusters (list): List of ground truth entity clusters.

        Returns:
            float: Recall score.
            float: Precision score.
            float: F1 score.

        """
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for i in range(len(predicted_entity_pos)):
            for pos in predicted_entity_pos[i]:
                if pos in entity_pos[i]:
                    true_positives += 1
                else:
                    false_positives += 1

            for pos in entity_pos[i]:
                if pos not in predicted_entity_pos[i]:
                    false_negatives += 1

        if true_positives == 0:
            return 0, 0, 0
        else:
            recall = true_positives / (true_positives + false_negatives)
            precision = true_positives / (true_positives + false_positives)
            f1 = 2 * (precision * recall) / (precision + recall)

        return precision, recall, f1
