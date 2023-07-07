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