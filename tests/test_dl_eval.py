import torch
import pytest
import torch.nn as nn
from backend.dl.dl_eval import compute_correct, compute_accuracy


def test_compute_correct():
    # Create dummy tensors
    predicted = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    actual = torch.tensor([2, 1])

    # Compute the number of correct predictions
    correct = compute_correct(predicted, actual)

    # Check if the number of correct predictions is correct
    assert correct == 2


def test_compute_accuracy():
    # Create dummy tensors
    predicted = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]])
    actual = torch.tensor([2, 1])

    # Compute the accuracy
    accuracy = compute_accuracy(predicted, actual)

    # Check if the accuracy is correct
    assert accuracy == 1
    
     # Test case: Accuracy of 0
    predicted_none_correct = torch.tensor([[0.9, 0.1, 0.0], [0.7, 0.2, 0.1]])
    actual_none_correct = torch.tensor([2, 1])
    accuracy_none_correct = compute_accuracy(predicted_none_correct, actual_none_correct)
    assert accuracy_none_correct == 0

    # Test case: Accuracy between 0 and 1
    predicted_mixed = torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.4, 0.4]])
    actual_mixed = torch.tensor([0,0])
    accuracy_mixed = compute_accuracy(predicted_mixed, actual_mixed)
    assert accuracy_mixed == 0.5
