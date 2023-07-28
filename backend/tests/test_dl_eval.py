import torch
import pytest
import torch.nn as nn
from dl.dl_eval import compute_correct, compute_accuracy


@pytest.mark.parametrize(
    "predicted, actual, expected_correct",
    [
        # Test case: All correct predictions
        (torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]), torch.tensor([2, 1]), 2),
        # Test case: Some correct predictions
        (torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.4, 0.4]]), torch.tensor([0, 0]), 1),
        # Test case: No correct predictions
        (torch.tensor([[0.9, 0.1, 0.0], [0.7, 0.2, 0.1]]), torch.tensor([2, 1]), 0),
    ],
)
def test_compute_correct(predicted, actual, expected_correct):
    # Compute the number of correct predictions
    correct = compute_correct(predicted, actual)

    # Check if the number of correct predictions is correct
    assert correct == expected_correct


@pytest.mark.parametrize(
    "predicted, actual, expected_accuracy",
    [
        # Test case: Accuracy of 1
        (torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]), torch.tensor([2, 1]), 1),
        # Test case: Accuracy between 0 and 1
        (torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.4, 0.4]]), torch.tensor([0, 0]), 0.5),
        # Test case: Accuracy of 0
        (torch.tensor([[0.9, 0.1, 0.0], [0.7, 0.2, 0.1]]), torch.tensor([2, 1]), 0),
    ],
)
def test_compute_accuracy(predicted, actual, expected_accuracy):
    # Compute the accuracy
    accuracy = compute_accuracy(predicted, actual)

    # Check if the accuracy is correct
    assert accuracy == expected_accuracy
