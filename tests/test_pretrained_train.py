import pytest
import torch.nn
from backend.pretrained import get_num_features, train
from fastai.vision.all import Adam, SGD
import os
import timm

main_dir = '' if (os.getcwd()).split('\\')[-1].split('/')[-1] == 'tests' else 'tests'

# timm: adv_inception_v3, resnet50, vit_small_r26_s32_224, tf_mobilenetv3_small_075
# torch: others
@pytest.mark.parametrize(
    "path_to_file,model_name",
    [
        (os.path.join(main_dir, "zip_files/double_zipped.zip"), "adv_inception_v3"),
        (os.path.join(main_dir, "zip_files/double_zipped.zip", "resnet50")),
        (os.path.join(main_dir, "zip_files/double_zipped.zip"), "vit_small_r26_s32_224"),
        (os.path.join(main_dir, "zip_files/double_zipped.zip"), "tf_mobilenetv3_small_075"),
        (os.path.join(main_dir, "zip_files/double_zipped.zip"), "EfficientNet"),
        (os.path.join(main_dir, "zip_files/double_zipped.zip", "googlenet")),
        (os.path.join(main_dir, "zip_files/double_zipped.zip", "vgg19")),
        (os.path.join(main_dir, "zip_files/double_zipped.zip"), "wide_resnet50_2"),
    ],
)
def test_train_valid_input_diff_models(path_to_file, model_name):
    train_loader, learner = train(
        path_to_file, model_name, 8, torch.nn.CrossEntropyLoss(), 3, n_out=2
    )

    assert learner.epoch == 2
    assert type(learner.loss_func) is torch.nn.CrossEntropyLoss
    assert get_num_features(learner) == 2

    assert train_loader.batch_size == 8


@pytest.mark.parametrize(
    "path_to_file,model_name",
    [
        (os.path.join(main_dir, "zip_files/double_zipped.zip"), "resnet50"),
        (os.path.join(main_dir, "zip_files/valid_2.zip"), "resnet50"),
        (os.path.join(main_dir, "zip_files/valid_3.zip"), "resnet50"),
    ],
)
def test_train_diff_valid_input_files(path_to_file, model_name):
    train_loader, learner = train(
        path_to_file, model_name, 8, torch.nn.CrossEntropyLoss(), 3, n_out=2
    )

    assert learner.epoch == 2
    assert type(learner.loss_func) is torch.nn.CrossEntropyLoss
    assert get_num_features(learner) == 2

    assert train_loader.batch_size == 8


@pytest.mark.parametrize(
    "model_name,batch_size,loss_func,n_epochs,shuffle,optimizer,lr,n_out",
    [
        ("tf_mobilenetv3_small_075", 24, torch.nn.MSELoss(), 3, True, Adam, 1e-4, 3),
        ("tv_resnet152", 4, torch.nn.MSELoss(), 3, False, Adam, 3e-4, 2),
        ("xcit_small_12_p8_224_dist", 8, torch.nn.KLDivLoss(), 4, True, Adam, 1e-4, 3),
        (
            "pit_xs_distilled_224",
            12,
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
    model_name, batch_size, loss_func, n_epochs, shuffle, optimizer, lr, n_out
):
    train_loader, learner = train(
        os.path.join(main_dir,"zip_files/double_zipped.zip"),
        model_name,
        batch_size,
        loss_func,
        n_epochs,
        shuffle,
        optimizer,
        None,
        lr,
        n_out=n_out,
    )

    assert learner.opt_func.__name__ == optimizer.__name__
    assert learner.epoch == n_epochs - 1
    assert type(learner.loss_func) is type(loss_func)
    assert get_num_features(learner) == n_out
    assert learner.lr == lr

    assert train_loader.batch_size == batch_size


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
