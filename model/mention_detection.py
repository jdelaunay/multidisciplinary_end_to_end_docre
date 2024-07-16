from torch import nn
import torch.nn.functional as F

from model.layers.span_representation import SpanAttention
from model.metrics import compute_metrics_md


class MentionDetector(nn.Module):
    """
    Module for detecting mentions in text.

    Args:
        hidden_size (int): The size of the hidden layer. Default is 768.

    Attributes:
        span_rep_layer (SpanAttention): The span attention layer used for generating span representations.
        span_decoder (nn.Sequential): The sequential layer used for decoding span representations.

    """

    def __init__(self, hidden_size=768):
        super(MentionDetector, self).__init__()
        self.span_rep_layer = SpanAttention(hidden_size=hidden_size, max_width=4)
        self.span_decoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.Linear(512, 2),
        )

    def forward(self, embeddings, span_idx, span_mask, span_labels):
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

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            span_labels.view(-1),
            ignore_index=-1,
            reduction="mean",
        )
        precision, recall, f1 = compute_metrics_md(logits, span_labels)
        return logits, loss, precision, recall, f1
