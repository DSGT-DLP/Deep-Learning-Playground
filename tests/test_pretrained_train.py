import pytest
import torch.nn
from backend.dl.pretrained import train, get_all
import os
import pandas as pd
from backend.common.dataset import dataset_from_zipped
from backend.common.constants import DEFAULT_TRANSFORM

train_dir = "" if (os.getcwd()).split("\\")[-1].split("/")[-1] == "tests" else "tests"
backend_dir = (
    "../backend"
    if (os.getcwd()).split("\\")[-1].split("/")[-1] == "tests"
    else "backend"
)
double_zipped = os.path.join(train_dir, "zip_files/double_zipped.zip")
valid_2 = os.path.join(train_dir, "zip_files/valid_2.zip")

# timm: adv_inception_v3, resnet50, vit_small_r26_s32_224, tf_mobilenetv3_small_075
# torch: others
@pytest.mark.parametrize(
    "path_to_file,model_name",
    [
        (double_zipped, "adv_inception_v3"),
        (double_zipped, "resnet50"),
        (
            double_zipped,
            "vit_small_r26_s32_224",
        ),
        (
            double_zipped,
            "tf_mobilenetv3_small_075",
        ),
        (double_zipped, "efficientnet_b2"),
        (double_zipped, "googlenet"),
        (double_zipped, "vgg19"),
        (double_zipped, "wide_resnet50_2"),
    ],
)
def test_train_valid_input_diff_models(path_to_file, model_name):
    train_dataset, test_dataset = dataset_from_zipped(
                path_to_file, test_transform=DEFAULT_TRANSFORM, train_transform=DEFAULT_TRANSFORM
            )

    train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_name=model_name,
        batch_size=2,
        loss_func="CELOSS",
        n_epochs=3,
        n_classes=2,
        lr=1e-2,
    )

    val = pd.read_csv(os.path.join(backend_dir, "dl_results.csv"))
    if val["train_loss"].isnull().any():
        assert False
    elif val["valid_loss"].isnull().any():
        assert False
    assert val.shape[0] == 3
    assert True

@pytest.mark.parametrize(
    "path_to_file,model_name, n_classes",
    [
        (double_zipped, "resnet50", 2),
        (valid_2, "resnet50", 2),
        (os.path.join(train_dir, "zip_files/valid_3.zip"), "resnet50", 3),
    ],
)
def test_train_diff_valid_input_files(path_to_file, model_name, n_classes):

    train_dataset, test_dataset = dataset_from_zipped(path_to_file, DEFAULT_TRANSFORM, DEFAULT_TRANSFORM)
    train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_name=model_name,
        batch_size=2,
        loss_func="CELOSS",
        n_epochs=2,
        lr=1e-3,
        n_classes=n_classes,
    )

    val = pd.read_csv(os.path.join(backend_dir, "dl_results.csv"))
    if val["train_loss"].isnull().any():
        assert False
    elif val["valid_loss"].isnull().any():
        assert False
    assert val.shape[0] == 2  # n_epochs
    assert True

@pytest.mark.parametrize(
    "path_to_file,model_name",
    [
        ("zip_files/not_zip", "resnet50"),
        ("zip_files/different_folders.zip", "resnet50"),
        ("zip_files/empty_folder.zip", "resnet50"),
    ],
)
def test_train_invalid_path(path_to_file, model_name):
    with pytest.raises(ValueError):
        dataset_from_zipped(path_to_file, DEFAULT_TRANSFORM, DEFAULT_TRANSFORM)
