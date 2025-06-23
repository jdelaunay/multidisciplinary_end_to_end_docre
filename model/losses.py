import torch
import torch.nn as nn
import torch.nn.functional as F


class AutomaticWeightedLoss(nn.Module):
    """
    Automatically weighted multi-task loss.

    Params:
        num: int
            The number of loss functions to combine.
        x: tuple
            A tuple containing multiple task losses.

    Examples:
        loss1 = 1
        loss2 = 2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        # Initialize parameters for weighting each loss, with gradients enabled
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)

    def forward(self, *losses):
        """
        Forward pass to compute the combined loss.

        Args:
            *losses: Variable length argument list of individual loss values.

        Returns:
            torch.Tensor: The combined weighted loss.
        """
        loss_sum = 0
        for i, loss in enumerate(losses):
            # Compute the weighted loss component for each task
            weighted_loss = 0.5 / (self.params[i] ** 2) * loss
            # Add a regularization term to encourage the learning of useful weights
            regularization = torch.log(1 + self.params[i] ** 2)
            # Sum the weighted loss and the regularization term
            loss_sum += weighted_loss + regularization

        return loss_sum


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        BCE_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        ce_loss = F.cross_entropy(logits.float(), labels, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        if len(labels.size()) == 1:
            th_label[0] = 1.0
            labels[0] = 0.0
        else:
            th_label[:, 0] = 1.0
            labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = logits > th_logit
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.0).to(logits)
        return output

    def get_score(self, logits, num_labels=-1):

        if num_labels > 0:
            return torch.topk(logits, num_labels, dim=1)
        else:
            return logits[:, 1] - logits[:, 0], 0
