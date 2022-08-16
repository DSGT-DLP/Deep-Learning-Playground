"""The weights, biases and architectures are obtained through [timm](https://github.com/rwightman/pytorch-image-models/tree/master/timm/models) and [pytorch](https://pytorch.org/vision/stable/models.html) libraries. The models are trained through fastai (another library). Fastai has a direct compatibility with pytorch models, however wwf is used to connect timm models with fastai library."""

import timm
import torch.nn as nn
import torch.hub
import os
import torchvision
from ..common.utils import generate_acc_plot, generate_confusion_matrix, generate_loss_plot, generate_AUC_ROC_CURVE

from fastai.data.core import DataLoaders
from fastai.learner import save_model
from fastai.vision.learner import has_pool_type
from fastai.vision.learner import _update_first_layer
from fastai.vision.all import *
from fastai.callback.progress import CSVLogger

from wwf.vision.timm import *
from torchvision.models import *
from torchvision import models
from fastai.callback.hook import num_features_model
from backend.common.loss_functions import LossFunctions
from backend.common.dataset import dataset_from_zipped
from backend.common.constants import SAVED_MODEL, DEEP_LEARNING_RESULT_CSV_PATH
import pandas as pd

class PretrainCallback(Callback):
    "Log the results displayed in `learn.path/fname`"
    order=60
    def __init__(self, n_epochs, send_progress):
        self.n_epochs, self.send_progress = n_epochs, send_progress


    def after_epoch(self):
        self.send_progress((self.learn.epoch+1)/self.n_epochs*100)
        print("percent done: ", (self.learn.epoch + 1) / self.n_epochs*100)

from backend.common.constants import DEFAULT_TRANSFORM, ONNX_MODEL, SAVED_MODEL


def train(
    train_dataset,
    test_dataset,
    model_name,
    batch_size,
    loss_func,
    n_epochs,
    send_progress=lambda a: print("Warning: instantiate send progress"),
    default=None,
    shuffle=False,
    optimizer_name="Adam",
    lr=1e-3,
    cut=None,
    n_classes=10,
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
        cut (int or Callable) : indices where you want to cut the model and create head and body
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    img_shape = 0
    dls = DataLoaders.from_dsets(
        train_dataset, test_dataset, device=device, shuffle=shuffle, bs=batch_size
    )
    setattr(dls, "device", device)

    for i in dls:
        for j in i:
            img_shape = j[0][0].shape
            break
        break

    loss_func = LossFunctions.get_loss_obj(LossFunctions[loss_func])
    train_accuaracy = train_acc
    if is_pytorch(model_name.lower()):
        print("torch model")
        model = eval("torchvision.models.{}".format(model_name.lower()))
        learner = vision_learner(
            dls,
            model,
            lr=lr,
            opt_func=eval(optimizer_name),
            pretrained=True,
            cut=cut,
            normalize=False,
            loss_func=loss_func,
            n_out=n_classes,
            n_in=chan_in,
            metrics=[train_accuaracy, accuracy]
        )
    elif is_timm(model_name.lower()):
        print("made the zipped file")
        learner = local_timm_learner(
            dls,
            model_name.lower(),
            lr=lr,
            opt_func= eval(optimizer_name),
            n_out=n_classes,
            cut=cut,
            normalize=False,
            n_in=chan_in,
            loss_func=loss_func,
        )

    backend_dir = (
        ""
        if (os.getcwd()).split("\\")[-1].split("/")[-1] == "backend"
        else "backend"
        if ("backend" in os.listdir(os.getcwd()))
        else "../backend"
    )

    # saved_model = (
    #     SAVED_MODEL
    #     if not "frontend" in os.listdir(os.getcwd())
    #     else "/".join(SAVED_MODEL.split("/")[1:])
    # )

    learner.fit(
        n_epochs, cbs=[CSVLogger(fname=os.path.join(backend_dir, "dl_results.csv")), PretrainCallback(n_epochs, send_progress)]
    )

    preds, target = learner.get_preds()

    y_pred_list = []
    y_pred_list.append(to_np(preds))
    auxiliary_outputs = {}

    confusion_matrix = generate_confusion_matrix(labels_last_epoch=to_np(target), y_pred_last_epoch=y_pred_list)
    auxiliary_outputs["confusion_matrix"] = confusion_matrix

    AUC_ROC_curve_data = generate_AUC_ROC_CURVE(labels_last_epoch=to_np(target), y_pred_last_epoch=y_pred_list)
    auxiliary_outputs["AUC_ROC_curve_data"] = AUC_ROC_curve_data

    val = pd.read_csv(DEEP_LEARNING_RESULT_CSV_PATH)
    val.columns = ['epoch','train_loss','test_loss','train_acc','val/test acc','time']
    val.to_csv(DEEP_LEARNING_RESULT_CSV_PATH)
    generate_loss_plot(DEEP_LEARNING_RESULT_CSV_PATH)
    generate_acc_plot(DEEP_LEARNING_RESULT_CSV_PATH)

    save_model(SAVED_MODEL, learner.model, getattr(learner, 'opt', None))
    torch.onnx.export(
        learner.model, torch.randn(1, chan_in, img_shape[2], img_shape[1]), ONNX_MODEL)

    return auxiliary_outputs, learner

def train_acc(batch_preds, batch_targ):
    y_pred, y_true = torch.argmax(batch_preds, axis=1), batch_targ.long().squeeze()
    train_correct = (y_pred == y_true).type(torch.float).sum().item()
    return train_correct


def get_all():
    """
    Returns the names of all possible models to be used
    """
    list = dir(models)[29:-1] + (timm.list_models(pretrained=True))
    list.sort()
    return list


def local_create_timm_body(arch: str, pretrained=True, cut=None, n_in=3, n_classes=10):
    """
    Creates a body from any model in the `timm` library.
    Code adapted from https://github.com/fastai/fastai/blob/master/fastai/vision/learner.py
    """
    print("here as well")
    model = timm.create_model(arch, pretrained=pretrained, num_classes=n_classes)
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
        layer_list = []
        for x in model.children():
            layer_list.append(x)
        return nn.Sequential(*layer_list[:cut])
    elif callable(cut):
        return cut(model)
    else:
        raise NameError("cut must be either integer or function")


def local_create_timm_model(
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

    body = local_create_timm_body(arch, pretrained, None, n_in, n_classes=n_out)
    if custom_head is None:
        nf = num_features_model(body)
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
    model = local_create_timm_model(
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
    if "pit" in model_name:
        return False
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


if __name__ == "__main__":
    train_dataset, test_dataset = dataset_from_zipped(
        "./tests/zip_files/double_zipped.zip"
    )
    train(train_dataset, test_dataset, "resnet18", 2, "CELOSS", 2, n_classes=2)
