from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.naive_bayes import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms


LAYERS_NN_MAPPING = {
    "LINEAR": nn.Linear,
    "RELU": nn.ReLU,
    "TANH": nn.Tanh,
    "SOFTMAX": nn.Softmax,
    "SIGMOID": nn.Sigmoid,
    "LOGSOFTMAX": nn.LogSoftmax,
}


def parse_deep_user_architecture(layers):
    """
    Given a list of user_models in a certain order,
    each layer should contain name and the values of each
    parameter

    Eg: layers = [{ 'value': 'LINEAR', 'parameters': [10, 3] }] should become [nn.Linear(10, 3)]

    Args:
        layers (list): string form where you have the model name and the values for each parameter
    Returns:
        parsed_modules (list): list of nn.Module objects parsed
    """

    # JSON file will contain name of model along with what hyperparameters need to be specified
    converted_data = []

    for item in layers:
        value = item['value']
        parameters = item['parameters']

        if value not in LAYERS_NN_MAPPING:
            raise Exception(f"Layer ${value} not supported")
        
        linear_layer = LAYERS_NN_MAPPING[value](*parameters)
        converted_data.append(linear_layer)

    return converted_data



if __name__ == "__main__":
    print("starting")
    print(
        parse_deep_user_architecture(
            [
                "nn.Linear(in_features=50, out_features=10)",
                "nn.Conv2d(in_channels=3, out_channels=36, kernel_size=3, stride=2)",
            ]
        )
    )
