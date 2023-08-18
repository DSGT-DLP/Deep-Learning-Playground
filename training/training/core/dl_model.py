from typing import Any, Callable
from ninja import Schema
import torch
import torch.nn as nn
from typing import TYPE_CHECKING

from training.routes.tabular.schemas import LayerParams


class DLModel(nn.Module):
    LAYERS_MAPPING: dict[str, Callable[..., nn.Module]] = {
        "LINEAR": nn.Linear,
        "RELU": nn.ReLU,
        "TANH": nn.Tanh,
        "SOFTMAX": nn.Softmax,
        "SIGMOID": nn.Sigmoid,
        "LOGSOFTMAX": nn.LogSoftmax,
    }

    def __init__(self, layer_list: list[nn.Module]):
        """
        Function to initialize Deep Learning model
        given a user specified layer list

        Args:
            layer_list (list): list of nn.Module layers from parser.py
        """
        super().__init__()
        self.model = self.build_model(layer_list)

    @classmethod
    def fromLayerParamsList(cls, layer_params_list: list[LayerParams]):
        layer_list = []
        for layer_params in layer_params_list:
            if layer_params.value not in cls.LAYERS_MAPPING:
                raise Exception(f"Layer ${layer_params.value} not supported")
            linear_layer = cls.LAYERS_MAPPING[layer_params.value](
                *layer_params.parameters
            )
            layer_list.append(linear_layer)
        return cls(layer_list)

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
