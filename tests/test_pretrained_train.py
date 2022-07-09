import pytest
import torch.nn
from backend.pretrained import get_num_features, train
from fastai.vision.all import Adam, SGD
import os
import pandas as pd
import timm

main_dir = "" if (os.getcwd()).split("\\")[-1].split("/")[-1] == "tests" else "tests"
double_zipped = os.path.join(main_dir, "zip_files/double_zipped.zip")

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
        (double_zipped, "EfficientNet"),
        (double_zipped, "googlenet"),
        (double_zipped, "vgg19"),
        (double_zipped, "wide_resnet50_2"),
    ],
)
def test_train_valid_input_diff_models(path_to_file, model_name):
    learner = train(
        path_to_file, model_name, 8, torch.nn.CrossEntropyLoss(), 3, n_classes=2
    )

    assert learner.epoch == 2
    assert type(learner.loss_func) is torch.nn.CrossEntropyLoss
    assert get_num_features(learner) == 2

    val = pd.read_csv('../backend/dl_results.csv')
    if val['train_loss'].isnull().any():
        assert False
    elif val['valid_loss'].isnull().any():
        assert False
    assert True


@pytest.mark.parametrize(
    "path_to_file,model_name",
    [
        (double_zipped, "resnet50"),
        (os.path.join(main_dir, "zip_files/valid_2.zip"), "resnet50"),
        (os.path.join(main_dir, "zip_files/valid_3.zip"), "resnet50"),
    ],
)
def test_train_diff_valid_input_files(path_to_file, model_name):
    learner = train(
        path_to_file, model_name, 2, torch.nn.CrossEntropyLoss(), 3, n_classes=2
    )

    assert learner.epoch == 2
    assert type(learner.loss_func) is torch.nn.CrossEntropyLoss
    assert get_num_features(learner) == 2

    val = pd.read_csv('../backend/dl_results.csv')
    if val['train_loss'].isnull().any():
        assert False
    elif val['valid_loss'].isnull().any():
        assert False
    assert True


@pytest.mark.parametrize(
    "model_name,batch_size,loss_func,n_epochs,shuffle,optimizer,lr,n_classes",
    [
        ("tf_mobilenetv3_small_075", 2, torch.nn.CrossEntropyLoss(), 3, True, Adam, 1e-4, 3),
        ("tv_resnet152", 1, torch.nn.CrossEntropyLoss(), 3, False, Adam, 3e-4, 2),
        ("xcit_small_12_p8_224_dist", 1, torch.nn.CrossEntropyLoss(), 4, True, Adam, 1e-4, 3),
        (
            "pit_xs_distilled_224",
            2,
            torch.nn.CrossEntropyLoss(),
            1,
            True,
            SGD,
            1e-2,
            10,
        ),
    ],
)
def test_train_valid_input_with_params(
    model_name, batch_size, loss_func, n_epochs, shuffle, optimizer, lr, n_classes
):
    train_loader, learner = train(
        double_zipped,
        model_name,
        batch_size,
        loss_func,
        n_epochs,
        shuffle,
        optimizer,
        None,
        lr,
        n_classes=n_classes,
    )

    assert learner.opt_func.__name__ == optimizer.__name__
    assert learner.epoch == n_epochs - 1
    assert type(learner.loss_func) is type(loss_func)
    assert get_num_features(learner) == n_classes
    assert learner.lr == lr

    val = pd.read_csv('../backend/dl_results.csv')
    if val['train_loss'].isnull().any():
        assert False
    elif val['valid_loss'].isnull().any():
        assert False
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
        train(path_to_file, model_name, 8, None, 1)
