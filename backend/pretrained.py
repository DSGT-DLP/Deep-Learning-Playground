# TODO:
# (Enhancement) Allow other git repo feature

import timm
import torch.nn as nn
import torch.hub
import torchvision
import os

from fastai.data.core import DataLoaders
from fastai.vision.learner import has_pool_type
from fastai.vision.learner import _update_first_layer
from fastai.vision.all import *
from fastai.callback.progress import CSVLogger
from wwf.vision.timm import *
from torchvision.models import *
from torchvision import models

try:
    from dataset import dataset_from_zipped
    from constants import DEFAULT_TRANSFORM, ONNX_MODEL
except:
    from backend.dataset import dataset_from_zipped
    from backend.constants import DEFAULT_TRANSFORM, ONNX_MODEL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(
    zipped_file,
    model_name,
    batch_size,
    loss_func,
    n_epochs,
    shuffle=False,
    optimizer=Adam,
    lr=1e-3,
    cut=None,
    n_classes=10,
    train_transform=DEFAULT_TRANSFORM,
    valid_transform=DEFAULT_TRANSFORM,
    chan_in=3,
):
    """
    Args:
        zipped_file(str Path) : Path to the zipped file which contains data in the correct format
        model_name (str) : name of the model
        batch_size (int) : batch_size for the dataloaders
        n_epochs (int) : number of epochs to train for
        n_classes (int) : number of classes
        chan_in (int) : number of input channels
    """

    train_dataset, valid_dataset = dataset_from_zipped(
        zipped_file, valid_transform=valid_transform, train_transform=train_transform
    )

    dls = DataLoaders.from_dsets(
        train_dataset, valid_dataset, device=device, shuffle=shuffle, bs=batch_size
    )

    if is_timm(model_name):
        learner = local_timm_learner(
            dls,
            model_name,
            lr=lr,
            opt_func=optimizer,
            n_out=n_classes,
            cut=cut,
            normalize=False,
            n_in=chan_in,
            loss_func=loss_func,
        )

    elif is_pytorch(model_name):
        model = eval("torchvision.models.{}".format(model_name))
        learner = vision_learner(
            dls,
            model,
            lr=lr,
            opt_func=optimizer,
            pretrained=True,
            cut=cut,
            normalize=False,
            loss_func=loss_func,
            n_out=n_classes,
            device= device
        )

    learner.fit(n_epochs, cbs=[CSVLogger(fname="dl_results.csv")])
    learner.export(ONNX_MODEL)
    return learner


def get_num_features(body):
    """
    Helper method which returns the number of out_features in a model by identifying the last linear layer and returning its out_features
    """
    try:
        return num_features_model(nn.Sequential(*body.children()))
    except:
        for i in range(len(body)):
            layer = body[-i + 1]
            if isinstance(layer, torch.nn.modules.linear.Linear):
                return layer.out_features
            if isinstance(layer, torch.nn.Sequential):
                for block in layer:
                    for sublayer in block.children():
                        if isinstance(sublayer, torch.nn.modules.linear.Linear):
                            return sublayer.out_features
                        if isinstance(
                            sublayer, timm.models.vision_transformer.Attention
                        ):
                            for ll in sublayer.children():
                                if isinstance(ll, torch.nn.modules.linear.Linear):
                                    return ll.out_features


def create_timm_body(arch: str, pretrained=True, cut=None, n_in=3):
    """
    Creates a body from any model in the `timm` library.
    Code adapted from https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
    """
    model = timm.create_model(arch, pretrained=pretrained, num_classes=10)
    _update_first_layer(model, n_in, pretrained)
    if cut is None:
        try:
            ll = list(enumerate(model.children()))
            cut = next(
                i for i, o in reversed(ll) if has_pool_type(o)
            )  ## i is the layer number and o is the type
        except StopIteration:
            cut = -1
            pass
    if isinstance(cut, int):
        if cut == 0:
            cut = -1
        return nn.Sequential(*list(model.children())[:cut])
    elif callable(cut):
        return cut(model)
    else:
        raise NameError("cut must be either integer or function")


def create_timm_model(
    arch: str,
    n_out,
    cut=None,
    pretrained=True,
    n_in=3,
    init=nn.init.kaiming_normal_,
    custom_head=None,
    concat_pool=True,
    **kwargs
):
    """
    Create custom architecture using `arch`, `n_in` and `n_out` from the `timm` library
    Code adapted from https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
    """
    body = create_timm_body(arch, pretrained, None, n_in)
    if custom_head is None:
        nf = get_num_features(body)
        head = create_head(nf, n_out, concat_pool=concat_pool, **kwargs)
    else:
        head = custom_head
    model = nn.Sequential(body, head)
    if init is not None:
        apply_init(model[1], init)
    return model


def local_timm_learner(
    dls,
    arch: str,
    loss_func=None,
    pretrained=True,
    cut=None,
    splitter=None,
    y_range=None,
    config=None,
    n_out=None,
    n_in=3,
    normalize=False,
    **kwargs
):
    """
    Code adapted from https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
    """
    "Build a convnet style learner from `dls` and `arch` using the `timm` library"
    if config is None:
        config = {}
    if n_out is None:
        n_out = get_c(dls)
    assert (
        n_out
    ), "`n_out` is not defined, and could not be inferred from data, set `dls.c` or pass `n_out`"

    if y_range is None and "y_range" in config:
        y_range = config.pop("y_range")
    model = create_timm_model(
        arch, n_out, default_split, pretrained, y_range=y_range, n_in=n_in, **config
    )
    learn = Learner(dls, model, loss_func=loss_func, splitter=default_split, **kwargs)
    if pretrained:
        learn.freeze()
    return learn


def is_timm(model_name):
    """
    Checks if the model_name is present in the timm models catalogue
    """
    for i in range(len(timm.list_models(pretrained=True))):
        if model_name == timm.list_models(pretrained=True)[i]:
            return True
    return False


def is_pytorch(model_name):
    """
    Checks if the model_name is in torchvision.models catalogue
    """
    for i in dir(models):
        if i == model_name:
            return True
    return False


# TODO: Can possibly add this feature??
## The feature below allows you to use a pretrained model from any source repository

# def is_pytorch(model_name, github_repo="pytorch/vision"):
#     for i in torch.hub.list(github=github_repo):
#         if model_name == i:
#             return True
#     return False


# local testing
if __name__ == "__main__":

    train(
        "../tests/zip_files/double_zipped.zip",
        "resnet34",
        2,
        torch.nn.CrossEntropyLoss(),
        3,
        shuffle=False,
        optimizer=SGD,
        lr=3e-4,
        n_classes=2,
    )
