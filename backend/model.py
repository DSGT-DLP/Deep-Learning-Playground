import torch
import torch.nn as nn
from torch.autograd import Variable

class DLModel(nn.Module):
    def __init__(self, layer_list):
        """
        Function to initialize Deep Learning model
        given a user specified layer list

        Args:
            layer_list (list): list of nn.Module layers from parser.py
        """
        super().__init__()
        self.model = nn.Sequential(*layer_list)
    
    def forward(self, x: torch.Tensor):
        return self.model(x) #apply model on input x