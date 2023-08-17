"""
URL configuration for training project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
import datetime
from typing import Callable, Literal, Optional, Union, cast
from django.contrib import admin
from django.http import HttpRequest
from django.urls import path
from ninja import NinjaAPI, Path, Schema
from numpy import ndarray
import pandas as pd
from sklearn.utils import Bunch
import torch
from torch.utils.data import DataLoader
from training.criterion import getCriterionHandler
from training.dataset import SklearnDatasetCreator
from training.dl_model import DLModel, LayerParams
from training.optimizer import getOptimizer
from training.trainer import ClassificationTrainer, RegressionTrainer, Trainer

api = NinjaAPI()


class TabularParams(Schema):
    target: str
    features: list[str]
    name: str
    problem_type: Literal["CLASSIFICATION", "REGRESSION"]
    default: Optional[str]
    criterion: str
    optimizer_name: str
    shuffle: bool
    epochs: int
    test_size: float
    batch_size: int
    user_arch: list[LayerParams]


@api.post("/tabular")
def tabularTrain(request: HttpRequest, tabularParams: TabularParams):
    if tabularParams.default:
        dataCreator = SklearnDatasetCreator.fromDefault(
            tabularParams.default, tabularParams.test_size, tabularParams.shuffle
        )
        train_loader = DataLoader(
            dataCreator.createTrainDataset(),
            batch_size=tabularParams.batch_size,
            shuffle=False,
            drop_last=True,
        )

        test_loader = DataLoader(
            dataCreator.createTestDataset(),
            batch_size=tabularParams.batch_size,
            shuffle=False,
            drop_last=True,
        )

        model = DLModel.fromLayerParamsList(tabularParams.user_arch)
        optimizer = getOptimizer(model, tabularParams.optimizer_name, 0.05)
        criterionHandler = getCriterionHandler(tabularParams.criterion)
        if tabularParams.problem_type == "CLASSIFICATION":
            trainer = ClassificationTrainer(
                train_loader,
                test_loader,
                model,
                optimizer,
                criterionHandler,
                tabularParams.epochs,
                dataCreator.getCategoryList(),
            )
            for epoch_result in trainer:
                print(epoch_result)
            print(trainer.labels_last_epoch, trainer.y_pred_last_epoch)
            print(trainer.generate_confusion_matrix())
            print(trainer.generate_AUC_ROC_CURVE())
            return trainer.generate_AUC_ROC_CURVE()
        else:
            trainer = RegressionTrainer(
                train_loader,
                test_loader,
                model,
                optimizer,
                criterionHandler,
                tabularParams.epochs,
            )
            for epoch_result in trainer:
                print(epoch_result)


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", api.urls),  # type: ignore
]
