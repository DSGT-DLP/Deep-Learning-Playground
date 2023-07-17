from collections import Counter
import torch
import torch.nn as nn
import os

from typing import Union
from enum import Enum

from common.constants import UNZIPPED_DIR_NAME


class LossFunctions(Enum):
    # Some common loss functions
    L1LOSS = nn.L1Loss()
    MSELOSS = nn.MSELoss()
    BCELOSS = nn.BCELoss()
    BCEWITHLOGITSLOSS = nn.BCEWithLogitsLoss()
    CELOSS = nn.CrossEntropyLoss(reduction="mean")
    WCELOSS = nn.CrossEntropyLoss(
        reduction="mean"
    )  # Will not use this, just to prevent errors

    def get_loss_obj(self):
        return self.value


def compute_loss(loss_function_name, output, labels):
    """
    Function to compute the loss. Postprocessing of output or labels depends on the loss object used

    Args:
        loss_function_name (str): Valid name from LossFunctions Enum
        output (_type_): _description_
        labels (_type_): _description_

    Return:
        loss(float): computed loss
    """
    postprocess_output = output.clone()
    postprocess_label = labels.clone()
    if loss_function_name in LossFunctions._member_names_:
        loss_obj = LossFunctions.get_loss_obj(LossFunctions[loss_function_name])
        if (
            loss_function_name.upper() == "BCELOSS"
            or loss_function_name.upper() == "BCEWITHLOGITSLOSS"
        ):
            # If target is say [20] but output is [20, 1], you need to unsqueeze target to be [20, 1] dimension
            return loss_obj(
                postprocess_output, postprocess_label.unsqueeze(1)
            )  # get the dimensions to match up.
        elif (
            loss_function_name.upper() == "MSELOSS"
            or loss_function_name.upper() == "L1LOSS"
        ):
            postprocess_output = torch.reshape(
                postprocess_output,
                (postprocess_output.shape[0], postprocess_output.shape[2]),
            )

            # print(f"output dims = {postprocess_output.size()}")
            # print(f"label dims = {postprocess_label.size()}")
            return loss_obj(postprocess_output, postprocess_label)  # compute the loss
        else:
            postprocess_output = torch.reshape(
                postprocess_output,
                (postprocess_output.shape[0], postprocess_output.shape[2]),
            )
            postprocess_label = postprocess_label.squeeze_()
            return loss_obj(
                postprocess_output, postprocess_label.long()
            )  # compute the loss
    raise Exception(
        "Invalid loss function name provided. Please contact admin to request addition of it. Provide documentation of this loss function"
    )


def compute_img_loss(criterion, pred, ground_truth, weights_counter):
    """
    Computes CE and WCE loss. pred and y are processed to different shapes supported by the corresponding functions.
    """
    loss_obj = LossFunctions.get_loss_obj(LossFunctions[criterion])

    if criterion == LossFunctions.CELOSS.name:
        return loss_obj(pred, ground_truth.squeeze())
    if criterion == "WCELOSS":
        weight_list = [0] * len(pred[0])

        for i in range(len(weight_list)):
            if weights_counter[i] != 0:
                weight_list[i] = 1 / weights_counter[i]
        # Weighting the class with least representation in dataset with maximum weight

        loss = nn.CrossEntropyLoss(
            weight=torch.FloatTensor(weight_list), reduction="mean"
        )
        return loss(pred, ground_truth.squeeze())
