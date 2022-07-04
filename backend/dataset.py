import pandas as pd
import urllib.request as request
import csv
import traceback
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import shutil
from enum import Enum
from torch.utils.data import DataLoader
from zipfile import ZipFile, is_zipfile
from os import mkdir, path

try:
    from backend.constants import *
except:
    from constants import *


def read_dataset(url):
    """
    Given a url to a CSV dataset, read it and build temporary csv file

    Args:
        url (str): URL to dataset
    """
    try:
        r = request.urlopen(url).read().decode("utf8").split("\n")
        reader = csv.reader(r)
        with open(CSV_FILE_PATH, mode="w", newline="") as f:
            csvwriter = csv.writer(f)
            for line in reader:
                csvwriter.writerow(line)
    except Exception as e:
        traceback.print_exc()
        raise Exception(
            "Reading Dataset from URL failed. Might want to check the validity of the URL"
        )


def read_local_csv_file(file_path):
    """
    Given a local file path or the "uploaded CSV file", read it in pandas dataframe

    Args:
        file_path (str): file path

    """
    try:
        with open(file_path, mode="r") as data_file:
            with open(CSV_FILE_PATH, mode="w", newline="") as f:
                csvwriter = csv.writer(f)
                for line in data_file:
                    csvwriter.writerow(line)
    except Exception as e:
        traceback.print_exc()
        raise Exception(
            "Reading Local CSV failed. Might want to check that you uploaded a proper CSV file"
        )


class errorMessage(Enum):
    TRAIN_AND_VALID_VOID = "The zip file doesn't contain train and valid folders"
    TRAIN_VOID = "The zip file doesn't contain train folder"
    VALID_VOID = "The zip file doesn't contain valid folder"
    NOT_ZIP = "The given file is not a zip file"
    UNEQUAL_CLASSES = "Train and valid datasets have different number of classes"
    TRAIN_NO_FILES = "Train folder has no files"
    VALID_NO_FILES = "Valid folder has no files"
    FILE_NOT_EXIST = "The file doesn't exist"
    INVALID_TRANSFORM = "The sequence of transformations is invalid"


def get_unzipped(zipped_file):
    """
    Creates and returns a train and valid directory path from a zipped file
    """

    if not path.exists(zipped_file):
        raise ValueError(errorMessage.FILE_NOT_EXIST.value)

    if not is_zipfile(zipped_file):
        raise ValueError(errorMessage.NOT_ZIP.value)

    if path.exists(UNZIPPED_DIR_NAME):
        shutil.rmtree(UNZIPPED_DIR_NAME)

    mkdir(UNZIPPED_DIR_NAME)
    mkdir(os.path.join(UNZIPPED_DIR_NAME, "input"))

    files = ZipFile(zipped_file, "r")
    files.extractall(os.path.join(UNZIPPED_DIR_NAME, "input"))
    files.close()

    name = os.path.splitext(zipped_file)[0]
    name = name.split("/")[-1]
    name = name.split("\\")[-1]

    def check_inside_file(dir_name):
        if not path.exists(dir_name):
            return None, None

        train_dir = None
        valid_dir = None

        for filename in os.listdir(dir_name):
            if filename == "train":
                train_dir = os.path.join(dir_name, filename)
                if valid_dir is not None:
                    return train_dir, valid_dir
            elif filename == "valid":
                valid_dir = os.path.join(dir_name, filename)
                if train_dir is not None:
                    return train_dir, valid_dir

        return train_dir, valid_dir

    train_dir, valid_dir = check_inside_file(os.path.join(UNZIPPED_DIR_NAME, "input"))

    if not train_dir or valid_dir == None:
        train_dir, valid_dir = check_inside_file(
            os.path.join(UNZIPPED_DIR_NAME, "input", name)
        )

    ## All cases of errors
    if train_dir is None and valid_dir is None:
        shutil.rmtree(UNZIPPED_DIR_NAME)
        raise ValueError(errorMessage.TRAIN_AND_VALID_VOID.value)
    elif train_dir is None:
        shutil.rmtree(UNZIPPED_DIR_NAME)
        raise ValueError(errorMessage.TRAIN_VOID.value)
    elif valid_dir is None:
        shutil.rmtree(UNZIPPED_DIR_NAME)
        raise ValueError(errorMessage.VALID_VOID.value)

    if len(os.listdir(train_dir)) == 0:
        shutil.rmtree(UNZIPPED_DIR_NAME)
        raise ValueError(errorMessage.TRAIN_NO_FILES.value)

    if len(os.listdir(valid_dir)) == 0:
        shutil.rmtree(UNZIPPED_DIR_NAME)
        raise ValueError(errorMessage.VALID_NO_FILES.value)

    if not len(os.listdir(train_dir)) == len(os.listdir(valid_dir)):
        shutil.rmtree(UNZIPPED_DIR_NAME)
        raise ValueError(errorMessage.UNEQUAL_CLASSES.value)

    return train_dir, valid_dir


def dataset_from_zipped(
    zipped_folder, train_transform=DEFAULT_TRANSFORM, valid_transform=DEFAULT_TRANSFORM
):

    """
    Returns a dataset from a zipped folder applying train and valid transformation (if they are legal).
    Calls get_unzipped to create unzipped folder and obtain train and valid directories
    """

    train_dir, valid_dir = get_unzipped(zipped_folder)

    def check_valid_transform(transform):
        if not transform:
            raise ValueError(errorMessage.INVALID_TRANSFORM.value)
        if transform == DEFAULT_TRANSFORM:
            return True
        to_tensor_idx = -1
        idx = 0
        for x in transform:
            if isinstance(x, transforms.ToTensor):
                to_tensor_idx = idx
            else:
                if to_tensor_idx == -1:
                    for y in TENSOR_ONLY_TRANSFORMS:
                        if isinstance(x, y):
                            raise ValueError(errorMessage.INVALID_TRANSFORM.value)
                else:
                    for y in PIL_ONLY_TRANSFORMS:
                        if isinstance(x, y) and to_tensor_idx != -1:
                            raise ValueError(errorMessage.INVALID_TRANSFORM.value)
            idx += 1
        if to_tensor_idx == -1:
            raise ValueError(errorMessage.INVALID_TRANSFORM.value)
        else:
            return True

    def create_transform(transform):
        """
        Converts a list of transform to a valid transform
        Assumes the list to be valid
        """
        if check_valid_transform(transform):
            return transforms.Compose([x for x in transform])

    train_transform = create_transform(train_transform)
    valid_transform = create_transform(valid_transform)

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=valid_transform)

    return train_dataset, valid_dataset


def loader_from_zipped(
    zipped_file,
    batch_size=32,
    shuffle=False,
    train_transform=DEFAULT_TRANSFORM,
    valid_transform=DEFAULT_TRANSFORM,
):
    """
    Creates a dataloader from a zipped file

    Args:
        zipped_file: the file to be unzipped
        train_transform: transformer settings for the train dataset
        valid_transform: transformer setting for the validation dataset

    Returns:
        train_dataloader, valid_dataloader
    """

    train_dataset, valid_dataset = dataset_from_zipped(
        zipped_file, train_transform=train_transform, valid_transform=valid_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )

    return train_loader, valid_loader


if __name__ == "__main__":
    read_dataset("https://raw.githubusercontent.com/karkir0003/dummy/main/job.csv")
    read_local_csv_file("test.csv")

    # local testing
    train_loader, valid_loader = loader_from_zipped(
        "../tests/zip_files/double_zipped.Zip",
        train_transform=[
            transforms.ToTensor(),
            transforms.transforms.RandomChoice(transforms=[transforms.ToTensor()]),
        ],
    )
