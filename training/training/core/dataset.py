from abc import ABC, abstractmethod
from typing import Callable, Optional, Union, cast

from numpy import ndarray
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.conftest import fetch_california_housing
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from enum import Enum
import os
import shutil
import torchaudio
import soundata


class TrainTestDatasetCreator(ABC):
    """
    Creator that creates train and test PyTorch datasets from a given dataset.

    This class serves as an abstract base class for creating training and testing
    datasets compatible with PyTorch's dataset structure. Implementations should
    define specific methods for dataset processing and loading.
    """

    @abstractmethod
    def createTrainDataset(self) -> Dataset:
        pass

    @abstractmethod
    def createTestDataset(self) -> Dataset:
        pass


class SklearnDatasetCreator(TrainTestDatasetCreator):
    DEFAULT_DATASETS: dict[
        str, Callable[[], Union[Bunch, tuple[Bunch, tuple], tuple[ndarray, ndarray]]]
    ] = {
        "IRIS": load_iris,
        "BREAST_CANCER": load_breast_cancer,
        "CALIFORNIA_HOUSING": fetch_california_housing,
        "DIABETES": load_diabetes,
        "WINE": load_wine,
    }

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        shuffle: bool,
        category_list: Optional[list[str]],
    ) -> None:
        super().__init__()
        self._category_list = category_list
        self._X_train, self._X_test, self._y_train, self._y_test = cast(
            tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
            train_test_split(X, y, test_size=test_size, shuffle=shuffle),
        )

    @classmethod
    def getDefaultDataset(cls, name: str) -> pd.DataFrame:
        raw_data = cls.DEFAULT_DATASETS[name]()
        default_dataset = pd.DataFrame(
            data=np.c_[raw_data["data"], raw_data["target"]],  # type: ignore
            columns=raw_data["feature_names"] + ["target"],  # type: ignore
        )

        default_dataset.dropna(how="all", inplace=True)
        return default_dataset

    @classmethod
    def fromDefault(cls, name: str, test_size: float, shuffle: bool):
        raw_data = cls.DEFAULT_DATASETS[name]()
        default_dataset = cls.getDefaultDataset(name)
        y = default_dataset["target"]
        X = default_dataset.drop("target", axis=1)
        return cls(X, y, test_size, shuffle, list(raw_data.target_names) if hasattr(raw_data, "target_names") else None)  # type: ignore

    def createTrainDataset(self) -> Dataset:
        X_train_tensor = Variable(torch.Tensor(self._X_train.to_numpy()))
        X_train_tensor = torch.reshape(
            X_train_tensor, (X_train_tensor.size()[0], 1, X_train_tensor.size()[1])
        )
        X_train_tensor.requires_grad_(True)

        y_train_tensor = Variable(torch.Tensor(self._y_train.to_numpy()))
        y_train_tensor = torch.reshape(y_train_tensor, (y_train_tensor.size()[0], 1))
        return TensorDataset(X_train_tensor, y_train_tensor)

    def createTestDataset(self) -> Dataset:
        X_test_tensor = Variable(torch.Tensor(self._X_test.to_numpy()))
        X_test_tensor = torch.reshape(
            X_test_tensor, (X_test_tensor.size()[0], 1, X_test_tensor.size()[1])
        )
        X_test_tensor.requires_grad_(True)

        y_test_tensor = Variable(torch.Tensor(self._y_test.to_numpy()))
        y_test_tensor = torch.reshape(y_test_tensor, (y_test_tensor.size()[0], 1))
        return TensorDataset(X_test_tensor, y_test_tensor)

    def getCategoryList(self) -> list[str]:
        if self._category_list is None:
            raise Exception("Category list not available")
        return self._category_list


class DefaultImageDatasets(Enum):
    MNIST = "MNIST"
    FASHION_MNIST = "FashionMNIST"
    KMNIST = "KMNIST"
    CIFAR10 = "CIFAR10"


class ImageDefaultDatasetCreator(TrainTestDatasetCreator):
    def __init__(
        self,
        dataset_name: str,
        train_transform: None,
        test_transform: None,
        batch_size: int = 32,
        shuffle: bool = True,
    ):
        if dataset_name not in DefaultImageDatasets.__members__:
            raise Exception(
                f"The {dataset_name} file does not currently exist in our inventory. Please submit a request to the contributors of the repository"
            )

        self.dataset_dir = "./training/image_data_uploads"
        self.train_transform = train_transform or transforms.Compose(
            [transforms.ToTensor()]
        )

        self.test_transform = test_transform or transforms.Compose(
            [transforms.ToTensor()]
        )
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Ensure the directory exists
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Load the datasets

        self.train_set = datasets.__dict__[dataset_name](
            root=self.dataset_dir,
            train=True,
            download=True,
            transform=self.train_transform,
        )
        self.test_set = datasets.__dict__[dataset_name](
            root=self.dataset_dir,
            train=False,
            download=True,
            transform=self.test_transform,
        )

    @classmethod
    def fromDefault(
        cls,
        dataset_name: str,
        train_transform=None,
        test_transform=None,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> "ImageDefaultDatasetCreator":
        return cls(dataset_name, train_transform, test_transform, batch_size, shuffle)

    def delete_datasets_from_directory(self):
        if os.path.exists(self.dataset_dir):
            try:
                shutil.rmtree(self.dataset_dir)
                print(f"Successfully deleted {self.dataset_dir}")
            except Exception as e:
                print(f"Failed to delete {self.dataset_dir} with error: {e}")

    def createTrainDataset(self) -> DataLoader:
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
        )
        self.delete_datasets_from_directory()  # Delete datasets after loading
        return train_loader

    def createTestDataset(self) -> DataLoader:
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=True,
        )
        self.delete_datasets_from_directory()  # Delete datasets after loading
        return test_loader

    def getCategoryList(self) -> list[str]:
        return self.train_set.classes if hasattr(self.train_set, "classes") else []

class UrbanSoundDataset(Dataset):
    
    def __init__(self, dataset, folderList):
        # Initialize lists to hold file names, labels, and folder numbers
        self.file_names = []
        self.labels = []
        self.folders = []
        self.dataset = dataset
        
        # Loop through the dataset and only add entries from folders in the folder list
        for clip in dataset.clip_ids:
            clip_metadata = dataset.clip(clip)
            fold = int(clip_metadata.subset.split('_')[1])  # Extract fold number
            if fold in folderList:
                self.file_names.append(clip_metadata.audio_path)
                self.labels.append(clip_metadata.tags[0])  # Assuming the first tag is the label
                self.folders.append(fold)
        
        self.folderList = folderList
        
    def __getitem__(self, index):
        # Load the file
        path = self.file_names[index]
        sound, sample_rate = torchaudio.load(path)
        
        # Convert to mono if stereo. Otherwise, just take the first channel. 
        # we need this bc the UrbanSound8k dataset is stereo, but torchaudio defaults to mono
        if sound.size(0) > 1:
            sound = torch.mean(sound, dim=0, keepdim=True)
        
        # Normalize the audio. Mean and std are calculated from the training set
        sound = (sound - sound.mean()) / sound.std()

        # Downsample the audio to ~8kHz
        tempData = torch.zeros([160000, 1])  # tempData accounts for audio clips that are too short
        if sound.numel() < 160000:
            tempData[:sound.numel()] = sound[:]
        else:
            tempData[:] = sound[:160000]
        
        soundData = tempData
        soundFormatted = torch.zeros([32000, 1])
        soundFormatted[:32000] = soundData[::5]  # Take every fifth sample of soundData
        soundFormatted = soundFormatted.permute(1, 0)
        return soundFormatted, self.labels[index]
    
    def __len__(self):
        return len(self.file_names)

class UrbanSoundDatasetCreator(TrainTestDatasetCreator):
    def __init__(self, folderList_train, folderList_test, batch_size=128, shuffle=True):
        # Initialize and download the dataset using soundata
        self.dataset = soundata.initialize('urbansound8k')
        self.dataset.download()  # Download the dataset
        
        self.folderList_train = folderList_train
        self.folderList_test = folderList_test
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_set = UrbanSoundDataset(self.dataset, self.folderList_train)
        self.test_set = UrbanSoundDataset(self.dataset, self.folderList_test)
    
    @classmethod
    def fromDefault(cls, batch_size=128, shuffle=True):
        folderList_train = range(1, 10)
        folderList_test = [10]
        return cls(folderList_train, folderList_test, batch_size, shuffle)
    
    def createTrainDataset(self) -> DataLoader:
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=self.shuffle)
        return train_loader

    def createTestDataset(self) -> DataLoader:
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=self.shuffle)
        return test_loader