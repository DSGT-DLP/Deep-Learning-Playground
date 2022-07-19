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
        self.model = self.build_model(layer_list)

    def build_model(self, layer_list):
        model = nn.Sequential()
        ctr = 1
        for layer in layer_list:
            model.add_module(f"layer #{ctr}: {str(layer.__class__.__name__)}", layer)
            ctr += 1
        return model

    def forward(self, x: torch.Tensor):
        pred = self.model(x)  # apply model on input x
        return pred
