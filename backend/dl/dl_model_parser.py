from sklearn.tree import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.svm import *
from sklearn.naive_bayes import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms


def parse_deep_user_architecture(user_model):
    """
    Given a list of user_models in a certain order,
    each layer should contain name and the values of each
    parameter

    Eg: ["Linear(in_features=50, out_features=10)"] should become [nn.Linear(in_features=50, out_features=10)]

    ["nn.Linear(in_features=50, out_features=10)", "nn.Conv2d(in_channels=16, out_channels=33, kernel_size=3, stride=2)"] should become
    [nn.Linear(in_features=50, out_features=10), nn.Conv2d(in_channels=16, out_channels=33, kernel_size=3, stride=2)]

    Args:
        user_model (list): string form where you have the model name and the values for each parameter
    Returns:
        parsed_modules (list): list of nn.Module objects parsed
    """

    # JSON file will contain name of model along with what hyperparameters need to be specified
    parsed_modules = []
    for element in user_model:
        layer = get_object(element)
        parsed_modules.append(layer)
    return parsed_modules


def get_object(element):
    """
    Given a string representation of a model layer or object, return the properly
    instantiated object (nn.Module instance in most scenarios for deep learning)

    Args:
        element (string): string representation of torch module

    return:
        Instantiated object behind the "string representation of the instance"
    """
    return eval(
        element.replace("'", "")
    )  # takes in the string representation and returns the "instantiated object"


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
