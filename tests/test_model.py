import pytest
import torch.nn as nn
from torch.autograd import Variable
from backend.model_parser import *

@pytest.mark.parametrize(
    "input_list,expected",
    [
        (
            ["nn.Linear(10,40)", "nn.Linear(40,3)"],
            [nn.Linear(10, 40), nn.Linear(40, 3)],
        ),
        (["nn.Linear(0,0)", "nn.Linear(0,0)"], [nn.Linear(0, 0), nn.Linear(0, 0)]),
    ],
)
def test_DLModel(input_list, expected):
    print(
        "parse_user_architecture(user_model): "
        + str(parse_user_architecture(user_model))
    )
    print("expected: " + str(expected))
    assert [i == j for i, j in zip(parse_user_architecture(user_model), expected)]
