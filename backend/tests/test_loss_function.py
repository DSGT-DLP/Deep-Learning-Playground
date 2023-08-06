import pytest
import torch
import torch.nn as nn
from common.loss_functions import compute_loss, LossFunctions


"""
Unit tests to check that the loss function is being computed correctly
"""


# initialize some tensors
zero_tensor = torch.zeros(10, 1, 1)
one_tensor = torch.ones(10, 1, 1)
tensor_vstack_one = torch.vstack(
    [torch.tensor([[2.5], [56.245], [2342.68967]]), torch.tensor([[3], [4], [5]])]
).reshape((6, 1, 1))
tensor_vstack_two = torch.vstack(
    [torch.tensor([[5646456], [634767], [37647346]]), torch.tensor([[6], [7], [8]])]
).reshape((6, 1, 1))


@pytest.mark.parametrize(
    "loss_function_name, output, labels, expected_number",
    [
        ("L1LOSS", zero_tensor, one_tensor, 1.0),
        ("L1LOSS", tensor_vstack_one, tensor_vstack_two, 7321426),
    ],
)
def test_l1_loss_computation_correct(
    loss_function_name, output, labels, expected_number
):
    assert pytest.approx(expected_number) == compute_loss(
        loss_function_name, output, labels
    )


@pytest.mark.parametrize(
    "loss_function_name, output, labels, expected_number",
    [
        ("MSELOSS", zero_tensor, one_tensor, 1.0),
        ("MSELOSS", tensor_vstack_one, tensor_vstack_two, 15543340),
    ],
)
def test_mse_loss_computation_correct(
    loss_function_name, output, labels, expected_number
):
    print(torch.sqrt(compute_loss(loss_function_name, output, labels)))
    assert pytest.approx(expected_number) == torch.sqrt(
        compute_loss(loss_function_name, output, labels)
    )


@pytest.mark.parametrize(
    "loss_function_name, output, labels, expected_number",
    [("BCELOSS", zero_tensor.reshape((10, 1)), one_tensor.reshape(10), 100)],
)
def test_bce_loss_computation_correct(
    loss_function_name, output, labels, expected_number
):
    assert pytest.approx(expected_number) == compute_loss(
        loss_function_name, output, labels
    )
