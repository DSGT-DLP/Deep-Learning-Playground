from typing import Callable
import torch

OPTIMIZER_MAPPING: dict[str, Callable[..., torch.optim.Optimizer]] = {
    "SGD": torch.optim.SGD,
    "ADAM": torch.optim.Adam,
}


def getOptimizer(model: torch.nn.Module, optimizer_name: str, learning_rate: float):
    """
    Given an optimizer name, instantiate the object

    Args:
        model (nn.Module): pytorch model to train
        optimizer_name (str): name of optimizer
        learning_rate (float): learning rate
    """
    return OPTIMIZER_MAPPING[optimizer_name](model.parameters(), lr=learning_rate)
