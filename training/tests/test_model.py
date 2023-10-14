import pytest
import torch.nn as nn
from torch.autograd import Variable
from training.core.dl_model import DLModel


@pytest.mark.parametrize(
    "input_list",
    [
        ([nn.Linear(10, 5), nn.Linear(5, 3)]),
        ([nn.Linear(0, 0), nn.Linear(0, 0)]),
        ([nn.Linear(100, 50), nn.Linear(5, 3)]),
    ],
)
def test_dlmodel(input_list):
    my_model = DLModel(input_list)
    assert [
        module
        for module in my_model.model.modules()
        if not isinstance(module, nn.Sequential)
    ] == input_list
