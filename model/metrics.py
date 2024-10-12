import torch
import torch.nn.functional as F


def compute_metrics_md(logits, targets):
    """
    Compute precision, recall, and F1 score for mention detection.

    Args:
        logits (torch.Tensor): The predicted logits from the model.
        targets (torch.Tensor): The target labels.

    Returns:
        tuple: A tuple containing the precision, recall, and F1 score.

    """
    predicted_labels = torch.sigmoid(logits)
    predicted_labels = (predicted_labels >= 0.5).long()
    predicted_labels = predicted_labels.view(-1)
    l = [predicted_label for predicted_label in predicted_labels if predicted_label == 1]
    true_labels = targets.view(-1)
    t = [true_label for true_label in true_labels if true_label == 1]


    true_positive = torch.sum((predicted_labels == 1) & (true_labels == 1)).item()
    false_positive = torch.sum((predicted_labels == 1) & (true_labels == 0)).item()
    false_negative = torch.sum((predicted_labels == 0) & (true_labels == 1)).item()

    if true_positive == 0:
        precision = 0
        recall = 0
        f1_score = 0
    else:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score


def compute_metrics_multi_class(logits, targets):
    """
    Compute precision, recall, and F1 score for multi-class classification.

    Args:
        logits (torch.Tensor): The predicted logits from the model.
        targets (torch.Tensor): The true labels.

    Returns:
        tuple: A tuple containing the precision, recall, and F1 score.

    """
    predicted_labels = torch.softmax(logits, dim=-1)
    predicted_labels = torch.argmax(predicted_labels, dim=-1).view(-1)
    true_labels = torch.argmax(targets, dim=-1).view(-1)
    assert predicted_labels.size() == true_labels.size()

    true_positive = torch.sum(
        (predicted_labels == true_labels) & (true_labels != 0)
    ).item()
    false_positive = torch.sum(
        (predicted_labels != true_labels) & (predicted_labels != 0)
    ).item()
    false_negative = torch.sum((predicted_labels == 0) & (true_labels != 0)).item()

    if true_positive == 0:
        precision = 0
        recall = 0
        f1_score = 0
    else:
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score
    
