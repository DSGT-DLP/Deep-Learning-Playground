import pandas as pd
import numpy as np
from sklearn.datasets import *
from enum import Enum
from backend.common.constants import DEFAULT_DATASETS
import torchvision
import torch
import ssl
from sklearn.model_selection import train_test_split

ssl._create_default_https_context = ssl._create_unverified_context

torchvision.datasets.MNIST.mirrors = [torchvision.datasets.MNIST.mirrors[1]]   ## torchvision default MNIST route causes 503 error sometimes

def get_default_dataset_header(dataset):
    """
    If user doesn't specify dataset
    Args:
        dataset (str): Which default dataset are you using (built in functions like load_boston(), load_iris())
    Returns:
        X: input (default dataset)
        y: target (default dataset)
    """
    try:
        if dataset not in DEFAULT_DATASETS:
            raise Exception(
                f"The {dataset} file does not currently exist in our inventory. Please submit a request to the contributors of the repository"
            )
        else:
            raw_data = eval(
                DEFAULT_DATASETS[dataset]
            )  # get raw data from sklearn.datasets
            default_dataset = pd.DataFrame(
                data=np.c_[raw_data["data"], raw_data["target"]],
                columns=raw_data["feature_names"] + ["target"],
            )
            default_dataset.dropna(how="all", inplace=True)  # remove any empty lines
            return default_dataset.columns
    except Exception:
        raise Exception(f"Unable to load the {dataset} file into Pandas DataFrame")

def get_default_dataset(dataset, target = None, features = None):
    """
    If user doesn't specify dataset
    Args:
        dataset (str): Which default dataset are you using (built in functions like load_boston(), load_iris())
    Returns:
        X: input (default dataset)
        y: target (default dataset)
    """
    try:
        if dataset not in DEFAULT_DATASETS:
            raise Exception(
                f"The {dataset} file does not currently exist in our inventory. Please submit a request to the contributors of the repository"
            )
        else:
            raw_data = eval(
                DEFAULT_DATASETS[dataset]
            )  # get raw data from sklearn.datasets
            default_dataset = pd.DataFrame(
                data=np.c_[raw_data["data"], raw_data["target"]],
                columns=raw_data["feature_names"] + ["target"],
            )
            default_dataset.dropna(how="all", inplace=True)  # remove any empty lines
            if(features and target):
                y = default_dataset[target]
                X = default_dataset[features]
            else:
                y = default_dataset["target"]
                X = default_dataset.drop("target", axis=1)
            print(default_dataset.head())
            return X, y

    except Exception:
        raise Exception(f"Unable to load the {dataset} file into Pandas DataFrame")


def get_img_default_dataset_loaders(
    datasetname, test_transform, train_transform, batch_size, shuffle
):
    """
    Returns dataloaders from default datasets
    Args:
        datasetname (str) : Name of dataset
        test_transform (list) : list of transforms
        train_transform (list) : list of transforms
        batch_size (int) : batch_size
    """
    train_transform = torchvision.transforms.Compose([x for x in train_transform])
    test_transform = torchvision.transforms.Compose([x for x in test_transform])

    dataset = eval(
        f"torchvision.datasets.{datasetname}(root='./backend/image_data_uploads', train=True, download=True, transform=train_transform)"
    )

    train_set, test_set = train_test_split(dataset, train_size=0.8, shuffle=False)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    return train_loader, test_loader

def get_img_default_dataset(
    datasetname, test_transform, train_transform
):
    """
    Returns dataloaders from default datasets
    Args:
        datasetname (str) : Name of dataset
        test_transform (list) : list of transforms
        train_transform (list) : list of transforms
        batch_size (int) : batch_size
    """
    train_transform = torchvision.transforms.Compose([x for x in train_transform])
    test_transform = torchvision.transforms.Compose([x for x in test_transform])

    train_set = eval(
        f"torchvision.datasets.{datasetname}(root='./backend/image_data_uploads', train=True, download=True, transform=train_transform)"
    )
    test_set = eval(
        f'torchvision.datasets.{datasetname}(root="./backend/image_data_uploads", train=False, download=True, transform=test_transform)'
    )
    return train_set, test_set