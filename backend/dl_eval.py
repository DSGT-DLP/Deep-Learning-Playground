import torch
import torch.nn as nn


def compute_accuracy(predicted, actual):
    """
    Given a prediction (usually in logit form for classification problem), identify the
    most likely label (probabilistically). Usually, for multiclass (more than 2 classes),
    Softmax is applied at the end. For binary, apply Sigmoid activation at the last

    Args:
        predicted (torch.Tensor): For each row, what's the probability that the instance belongs to each of K classes
        actual (torch.Tensor): actual class label

    NOTE: Since we have our training data in "batch form", we will be getting an accuracy for each batch in the dataloader
    """
    prediction = torch.argmax(
        predicted, dim=1
    )  # identify index of the most likely class

    performance = torch.where(
        (prediction == actual), torch.Tensor([1.0]), torch.Tensor([0.0])
    )  # for each row, did you predict correctly?

    batch_accuracy = torch.sum(performance) / prediction.size()[0]

    return batch_accuracy
