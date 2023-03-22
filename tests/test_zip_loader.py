import torchvision.transforms as transforms
import pytest
from pathlib import Path
from backend.common.dataset import loader_from_zipped, errorMessage
from backend.common.constants import UNZIPPED_DIR_NAME
from filecmp import dircmp
import torch
import os

different_folder = "zip_files/different_folders.Zip"
empty_folder = "zip_files/empty.zip"
double_zipped = "zip_files/double_zipped.zip"
not_zip = "zip_files/not_zip"
num_classes = "zip_files/num_classes.zip"

dir_in = "" if (os.getcwd()).split("\\")[-1].split("/")[-1] == "tests" else "tests"


@pytest.mark.parametrize(
    "filepath,expected",
    [
        (not_zip, errorMessage.CHECK_FILE_STRUCTURE.value),
        (different_folder, errorMessage.CHECK_FILE_STRUCTURE.value),
    ],
)
def test_invalid_file_structure(filepath, expected):
    # filepath = str(Path(filepath).parent.absolute()) + "/" + filepath.split("/")[-1]

    filepath = os.path.join(dir_in, filepath)

    print("os.cwd(): ", os.getcwd())
    print("dir_in: ", dir_in)
    print("filepath: ", filepath)
    with pytest.raises(ValueError) as e:
        loader_from_zipped(filepath)
    assert str(e.value) == expected


@pytest.mark.parametrize(
    "filepath, relative_output_path",
    [(double_zipped, f"{UNZIPPED_DIR_NAME}/input/double_zipped")],
)
def test_load_correct_file_structure(filepath, relative_output_path):
    try:
        expected_filename = filepath.split("/")[-1]

        filepath = os.path.join(dir_in, "zip_files")
        filepath = filepath + "/" + expected_filename
        loader_from_zipped(
            filepath,
            train_transform=[
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
            ],
        )
        # print("passed the loader from zipped function without exception")
        expected_filename = expected_filename.replace(".zip", "")
        print("expected/{}".format(expected_filename))
        print(relative_output_path)
        print(os.path.exists("expected/{}".format(expected_filename)))
        print(os.path.exists(relative_output_path))
        dcmf2 = os.path.join(dir_in, "expected")
        dcmp = dircmp(relative_output_path, dcmf2 + "/" + expected_filename)

        assert len(dcmp.diff_files) == 0
    except Exception:
        assert False


@pytest.mark.parametrize(
    "train_transform, valid_transform, filepath",
    [
        (None, [transforms.GaussianBlur(kernel_size=3)], "double_zipped.zip"),
        (
            [
                transforms.GaussianBlur(kernel_size=3),
                transforms.RandomHorizontalFlip(p=0.4),
                transforms.ToTensor(),
            ],
            None,
            double_zipped,
        ),
        (
            [transforms.RandomHorizontalFlip(p=0.9), transforms.ToTensor()],
            [
                transforms.RandomVerticalFlip(p=0.3),
                transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
            ],
            double_zipped,
        )
        # ([transforms.Normalize(0, 1), transforms.ToTensor()], None, double_zipped)
    ],
)
def check_diff_transforms(train_transform, valid_transform, filepath):
    filepath = os.path.join(dir_in, filepath)
    train_loader, valid_loader = loader_from_zipped(
        filepath, test_transform=valid_transform, train_transform=train_transform
    )

    for data, index in train_loader:
        if index == 0:
            train_data_val = data
            print(train_data_val)
            break

    for data, index in valid_loader:
        if index == 0:
            valid_data_val = data
            break

    assert not torch.equal(train_data_val, valid_data_val)


@pytest.mark.parametrize(
    "filepath, tensor_array",
    [
        (double_zipped, [transforms.Normalize(0, 1)]),  ## Applying Tensor only
        (
            double_zipped,
            [transforms.RandomVerticalFlip(p=0.3), transforms.ToTensor()],
        ),
        (
            double_zipped,
            [
                transforms.ToTensor(),
                transforms.RandomChoice(transforms=[transforms.Resize((256, 256))]),
            ],
        ),
    ],
)
def check_ordered_transforms(filepath, tensor_array):
    with pytest.raises(ValueError) as e:
        train_loader, valid_loader = loader_from_zipped(
            filepath, train_transform=tensor_array
        )
    assert str(e.value) == errorMessage.CHECK_TRANSFORM.value
