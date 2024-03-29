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
import boto3
from botocore.exceptions import ClientError
from io import StringIO
from common.constants import *
import requests
from io import BytesIO
from aws_helpers.s3_utils.s3_client import write_to_bucket

FILE_UPLOAD_BUCKET_NAME = "dlp-upload-bucket"


def read_dataset(url):
    """
    Given a url to a CSV dataset, read it and build a temporary csv file

    Args:
        url (str): URL to dataset
    """
    try:
        r = requests.get(url).text
        csv_data = StringIO(r)

        # Define the name of the temporary CSV file
        csv_file_name = "data.csv"

        # Write the StringIO data to a local temporary file
        with open(csv_file_name, mode="w", encoding="utf-8") as f:
            f.write(csv_data.getvalue())

        # Upload the temporary CSV file to the file upload S3 bucket
        write_to_bucket(csv_file_name, FILE_UPLOAD_BUCKET_NAME, csv_file_name)

        # Remove the local temporary CSV file after uploading
        os.remove(csv_file_name)

    except Exception as e:
        traceback.print_exc()
        raise Exception(
            "Reading Dataset from URL and uploading to S3 failed. Please check the validity of the URL."
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
    CHECK_FILE_STRUCTURE = "The file doesn't have correct structure"
    CHECK_TRANSFORM = "The transforms applied are not valid"


def get_unzipped(zipped_file):
    """
    Creates and returns a train and valid directory path from a zipped file
    """

    try:
        if path.exists(UNZIPPED_DIR_NAME):
            shutil.rmtree(UNZIPPED_DIR_NAME)

        mkdir(UNZIPPED_DIR_NAME)
        mkdir(os.path.join(UNZIPPED_DIR_NAME, "input"))

        files = ZipFile(zipped_file, "r")
        files.extractall(os.path.join(UNZIPPED_DIR_NAME, "input"))
        files.close()

        train_dir = os.path.join(UNZIPPED_DIR_NAME, "input", "train")
        test_dir = os.path.join(UNZIPPED_DIR_NAME, "input", "test")

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            name = os.path.splitext(zipped_file)[0]
            name = name.split("/")[-1]
            name = name.split("\\")[-1]
            train_dir = os.path.join(UNZIPPED_DIR_NAME, "input", name, "train")
            test_dir = os.path.join(UNZIPPED_DIR_NAME, "input", name, "test")

        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            raise ValueError(errorMessage.CHECK_FILE_STRUCTURE.value)

        if len(os.listdir(train_dir)) != len(os.listdir(test_dir)):
            raise ValueError(errorMessage.CHECK_FILE_STRUCTURE.value)

        return train_dir, test_dir
    except:
        raise ValueError(errorMessage.CHECK_FILE_STRUCTURE.value)


def dataset_from_zipped(
    zipped_folder, train_transform=DEFAULT_TRANSFORM, test_transform=DEFAULT_TRANSFORM
):
    """
    Returns a dataset from a zipped folder applying train and test transformation (if they are legal).
    Calls get_unzipped to create unzipped folder and obtain train and test directories
    """

    try:
        train_dir, test_dir = get_unzipped(zipped_folder)

        train_transform = transforms.Compose([x for x in train_transform])
        test_transform = transforms.Compose([x for x in test_transform])

        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

        return train_dataset, test_dataset
    except Exception as e:
        error = (
            errorMessage.CHECK_FILE_STRUCTURE
            if (str(e) == errorMessage.CHECK_FILE_STRUCTURE.value)
            else errorMessage.CHECK_TRANSFORM
        )
        raise ValueError(error.value)


def loader_from_zipped(
    zipped_file,
    batch_size=32,
    shuffle=False,
    train_transform=DEFAULT_TRANSFORM,
    test_transform=DEFAULT_TRANSFORM,
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
    try:
        train_dataset, test_dataset = dataset_from_zipped(
            zipped_file, train_transform=train_transform, test_transform=test_transform
        )

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True
        )  ### DROP LAST drops the last non full batch. 97 data points and batchsize = 10 would lead to creation of only 9 datasets, 7 extra data points will be lost

        return train_loader, test_loader
    except Exception as e:
        raise ValueError(e)


if __name__ == "__main__":
    read_dataset("https://raw.githubusercontent.com/karkir0003/dummy/main/job.csv")
    read_local_csv_file("test.csv")

    train_loader, valid_loader = loader_from_zipped(
        "./tests/zip_files/double_zipped.Zip",
        train_transform=[
            transforms.ToTensor(),
            transforms.transforms.RandomChoice(transforms=[transforms.ToTensor()]),
        ],
    )

    train_dataset, valid_dataset = dataset_from_zipped(
        zipped_folder="./tests/zip_files/double_zipped.Zip"
    )
