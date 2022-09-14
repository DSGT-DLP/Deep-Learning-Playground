import pandas as pd
import numpy as np
from sklearn.datasets import *
from enum import Enum
from backend.common.constants import DEFAULT_DATASETS
from backend.common.audio_utils import create_transform, make_collate_fn
import torchvision
import torchaudio
import torchaudio.transforms as T
import torch

# torchvision default MNIST route causes 503 error sometimes
torchvision.datasets.MNIST.mirrors = [torchvision.datasets.MNIST.mirrors[1]]

def get_default_dataset(dataset):
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
            # remove any empty lines
            default_dataset.dropna(how="all", inplace=True)
            y = default_dataset["target"]
            X = default_dataset.drop("target", axis=1)
            print(default_dataset.head())
            return X, y

    except Exception:
        raise Exception(f"Unable to load the {dataset} file into Pandas DataFrame")

def get_img_default_dataset_loaders(
    datasetname, train_transform, test_transform, batch_size, shuffle
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
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle, drop_last=True
    )
    return train_loader, test_loader

def get_audio_default_dataset_loaders(
    datasetname, train_transform, test_transform, batch_size, shuffle, sample_rate, max_sec
):
    """
    Returns dataloaders from default datasets
    Args:
        datasetname (str) : Name of dataset
        test_transform (list) : list of transforms
        train_transform (list) : list of transforms
        batch_size (int) : batch_size
    """
    
    train_set = eval(f"torchaudio.datasets.{datasetname}")(root='./backend/audio_data_uploads', subset='training', download=False)
    print('train_set downloaded')
    test_set = eval(f"torchaudio.datasets.{datasetname}")(root="./backend/audio_data_uploads", subset='validation', download=False)
    print('test_set downloaded')
    
    train_transform = create_transform(train_transform)
    test_transform = create_transform(test_transform)
    
    train_collate_fn = make_collate_fn(datasetname, sample_rate, max_sec, train_transform)
    test_collate_fn = make_collate_fn(datasetname, sample_rate, max_sec, test_transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, collate_fn=train_collate_fn, shuffle=shuffle)
    print('train_loader set')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, collate_fn=test_collate_fn, shuffle=shuffle)
    print('test_loader set')
    print('returning\n')
    return train_loader, test_loader


if __name__ == '__main__':
    transforms = [T.MelSpectrogram(), T.AmplitudeToDB()]
    train_loader, test_loader = get_audio_default_dataset_loaders('SPEECHCOMMANDS', transforms, transforms, 41, False, 16000, 0.9)
    for train_batch in train_loader:
        print('random train_loader batch:', train_batch)
        break
    for test_batch in test_loader:
        print('random test_loader batch:', test_batch)
        break