import torch
import torch.nn as nn
from torch.autograd import Variable

def get_optimizer(model, optimizer_name, learning_rate):
    """
    Given an optimizer name, instantiate the object

    Args:
        model (nn.Module): pytorch model to train
        optimizer_name (str): name of optimizer
        learning_rate (float): learning rate
    """
    if (optimizer_name.upper() == "SGD"):
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif (optimizer_name.upper() == "ADAM"):
        return torch.optim.Adam(model.parameters(), lr=learning_rate)