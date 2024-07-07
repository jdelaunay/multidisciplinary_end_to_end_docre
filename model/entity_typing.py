import torch
from torch import nn
import torch.nn.functional as F

from model.metrics import compute_metrics_multi_class


class EntityClassifier(nn.Module):
    def __init__(self, hidden_size=768, num_labels=2):
        super(EntityClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 512),
            nn.Linear(512, num_labels),
        )

    def forward(self, entity_embeddings, entity_types):
        entity_logits = self.classifier(entity_embeddings)
        print("Entity logits size:", entity_logits.size())

        entity_types = torch.cat(entity_types, dim=0).to(entity_logits.device)

        loss = F.cross_entropy(
            entity_logits.float(),
            entity_types.float(),
            reduction="mean",
        )
        precision, recall, f1 = compute_metrics_multi_class(entity_logits, entity_types)
        return loss, precision, recall, f1
