import torch
from torch import nn
import torch.nn.functional as F


class EntityClassifier(nn.Module):
    """
    EntityClassifier is a PyTorch module for classifying entity types based on entity embeddings.

    Args:
        hidden_size (int): The size of the hidden layer.
        num_labels (int): The number of entity labels.

    Attributes:
        hidden_size (int): The size of the hidden layer.
        num_labels (int): The number of entity labels.
        classifier (nn.Sequential): The classifier module.

    Methods:
        forward(entity_embeddings, entity_types): Performs forward pass of the entity classifier.

    """

    def __init__(self, hidden_size=768, num_labels=2):
        super(EntityClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, entity_embeddings, entity_types):
        """
        Performs forward pass of the entity classifier.

        Args:
            entity_embeddings (torch.Tensor): The input entity embeddings.
            entity_types (torch.Tensor): The target entity types.

        Returns:
            tuple: A tuple containing the loss, precision, recall, and F1 score.

        """
        entity_embeddings = torch.cat(entity_embeddings, dim=0)

        entity_logits = self.classifier(entity_embeddings)
        entity_types = torch.cat(entity_types, dim=0).to(entity_logits.device)

        loss = F.cross_entropy(
            entity_logits.float(),
            entity_types.float(),
            reduction="mean",
        )
        predicted_entity_types = torch.argmax(entity_logits, dim=1)
        return predicted_entity_types, loss
