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


class TrainTestDatasetCreator(ABC):
    "Creator that creates train and test PyTorch datasets"

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

        # remove any empty lines
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
