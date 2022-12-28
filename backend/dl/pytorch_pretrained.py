import torch
import timm
import torch.nn as nn
import numpy as np
import os

from backend.dl.dl_trainer import train_deep_image_classification
from ..common.dataset import loader_from_zipped
from ..common.optimizer import get_optimizer


def pytorch_pretrained(
    n_class,
    model_name,
    num_epochs,
    device,
    in_chan,
    loss_func,
    train_loader,
    valid_loader,
    optimizer_name="ADAM",
    # send_progress=lambda x: print("send progerss", x),
):

    model = GetModel(model_name, n_class, in_chan)
    print("got model")
    model.to(device)

    optimizer = get_optimizer(model, optimizer_name, learning_rate=0.05)

    print("optimizer")

    return train_deep_image_classification(
        model,
        train_loader,
        valid_loader,
        optimizer,
        loss_func,
        num_epochs,
        device,
        # send_progress,
    )


class GetModel(nn.Module):
    def __init__(self, model_name, N_CLASS, in_chan, pretrained=True):
        super().__init__()

        self.model_name = model_name
        self.cnn = timm.create_model(
            self.model_name,
            pretrained=pretrained,
            num_classes=N_CLASS,
            in_chans=in_chan,
        )

    def forward(self, x):
        x = self.cnn(x)
        return x


if __name__ == "__main__":
    file = os.path.join("tests", "zip_files", "better_zipped.zip")

    train_loader, test_loader = loader_from_zipped(file, batch_size=2, shuffle=False)
    pytorch_pretrained(
        n_class=2,
        model_name="inception_v3",
        num_epochs=2,
        device=torch.device("cpu"),
        in_chan=3,
        loss_func="CELOSS",
        train_loader=train_loader,
        valid_loader=test_loader,
    )
