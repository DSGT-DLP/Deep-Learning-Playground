import pytest
import torch.nn as nn
from torch.autograd import Variable
from dl.dl_model_parser import *

# come up with the expected parsing
# 2 linear layers in a row, you should know what it should look like


@pytest.mark.parametrize(
    "user_model,expected",
    [
        (
            [
                {"value": "LINEAR", "parameters": [10, 40]},
                {"value": "LINEAR", "parameters": [40, 3]},
            ],
            [nn.Linear(10, 40), nn.Linear(40, 3)],
        ),
        (
            [
                {"value": "LINEAR", "parameters": [0, 0]},
                {"value": "LINEAR", "parameters": [0, 0]},
            ],
            [nn.Linear(0, 0), nn.Linear(0, 0)],
        ),
    ],
)
def test_parse_user_architecture(user_model, expected):
    print(
        "parse_user_architecture(user_model): "
        + str(parse_deep_user_architecture(user_model))
    )
    print("expected: " + str(expected))
    assert [i == j for i, j in zip(parse_deep_user_architecture(user_model), expected)]
