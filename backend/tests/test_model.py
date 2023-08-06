import pytest
import torch.nn as nn
from torch.autograd import Variable
from dl.dl_model import *


@pytest.mark.parametrize(
    "input_list",
    [
        ([nn.Linear(10, 5), nn.Linear(5, 3)]),
        ([nn.Linear(0, 0), nn.Linear(0, 0)]),
        ([nn.Linear(100, 50), nn.Linear(5, 3)]),
    ],
)
def test_dlmodel(input_list):
    print("input_list: " + str(input_list) + " is of type " + str(type(input_list)))
    my_model = DLModel(input_list)
    print("my_model: " + str(my_model) + " is of type " + str(type(my_model)))
    print(
        "[module for module in my_model.model.modules() if not isinstance(module, nn.Sequential)]: "
        + str(
            [
                module
                for module in my_model.model.modules()
                if not isinstance(module, nn.Sequential)
            ]
        )
    )
    assert [
        module
        for module in my_model.model.modules()
        if not isinstance(module, nn.Sequential)
    ] == input_list
