from datetime import date
from celery import Celery
import django
from django.conf import settings

settings.configure()
django.setup()

import boto3


from training.constants import DLP_EXECUTIONS_BUCKET_NAME
from training.core.celery.criterion import getCriterionHandler
from training.core.celery.dataset import SklearnDatasetCreator
from training.core.celery.dataset import ImageDefaultDatasetCreator
from training.core.celery.dl_model import DLModel
from torch.utils.data import DataLoader
from training.core.celery.optimizer import getOptimizer
from training.core.celery.trainer import ClassificationTrainer, RegressionTrainer
from training.routes.tabular.schemas import TabularParams
from training.routes.image.schemas import ImageParams
from training.routes.training.results.schemas import (
    DetailedTrainResultsData,
    TrainResultsData,
)

from training import celeryconfig

celery_app = Celery("celery")
celery_app.config_from_object(celeryconfig)


def saveDetailedTrainResultsDataToS3(
    detailedTrainResultsData: DetailedTrainResultsData,
):
    s3 = boto3.resource("s3")
    s3.Object(
        DLP_EXECUTIONS_BUCKET_NAME,
        f"{detailedTrainResultsData.basic_info.trainspaceId}.json",
    ).put(Body=detailedTrainResultsData.json())


def collectTrainingResults(trainer, basic_info, is_classification):
    trainTestLoss = [
        {
            "x_name": "Epoch",
            "y_name": "Train loss",
            "x_values": [],
            "y_values": [],
        },
        {
            "x_name": "Epoch",
            "y_name": "Test loss",
            "x_values": [],
            "y_values": [],
        },
    ]
    for epoch_result in trainer:
        trainTestLoss[0]["x_values"].append(epoch_result.epoch_num)
        trainTestLoss[0]["y_values"].append(epoch_result.train_loss)
        trainTestLoss[1]["x_values"].append(epoch_result.epoch_num)
        trainTestLoss[1]["y_values"].append(epoch_result.test_loss)

    all_metrics = [
        {
            "name": "Train and test loss vs epoch",
            "time_series": trainTestLoss,
            "graph_index": 0,
            "chart_type": "LINE",
        }
    ]
    if is_classification:
        confusionMatrix = trainer.generate_confusion_matrix()
        aucRocCurve = trainer.generate_AUC_ROC_CURVE()
        all_metrics.append(
            {
                "name": "Confusion matrix",
                "values": confusionMatrix.tolist(),
                "chart_type": "CONFUSION_MATRIX",
                "graph_index": 1,
            }
        )
        all_metrics.append(
            {
                "name": "AUC/ROC curve",
                "values": aucRocCurve,
                "chart_type": "AUC/ROC",
                "graph_index": 2,
            }
        )

    detailedTrainResultsData = DetailedTrainResultsData(
        **{"basic_info": basic_info, "all_metrics": all_metrics}
    )
    return detailedTrainResultsData


@celery_app.task(name="tabularTrainTask")
def tabularTrainTask(input: dict, trainspaceId: str, uid: str):
    tabularParams = TabularParams(**input)
    basic_info = TrainResultsData(
        **{
            "name": tabularParams.name,
            "trainspaceId": trainspaceId,
            "dataSource": "TABULAR",
            "status": "SUCCESS",
            "created": date.today(),
            "step": "step",
            "uid": uid,
        }
    )

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
        else:
            trainer = RegressionTrainer(
                train_loader,
                test_loader,
                model,
                optimizer,
                criterionHandler,
                tabularParams.epochs,
            )
        detailedTrainResultsData = collectTrainingResults(
            trainer, basic_info, tabularParams.problem_type == "CLASSIFICATION"
        )

        # save detailedTrainResultsData
        saveDetailedTrainResultsDataToS3(detailedTrainResultsData)


@celery_app.task(name="imageTrainTask")
def imageTrainTask(input: dict, trainspaceId: str, uid: str):
    imageParams = ImageParams(**input)
    basic_info = TrainResultsData(
        **{
            "name": imageParams.name,
            "trainspaceId": trainspaceId,
            "dataSource": "IMAGE",
            "status": "SUCCESS",
            "created": date.today(),
            "step": "step",
            "uid": uid,
        }
    )

    if imageParams.default:
        dataCreator = ImageDefaultDatasetCreator.fromDefault(imageParams.default)
        train_loader = dataCreator.createTrainDataset()
        test_loader = dataCreator.createTestDataset()
        model = DLModel.fromLayerParamsList(imageParams.user_arch)
        optimizer = getOptimizer(model, imageParams.optimizer_name, 0.05)
        criterionHandler = getCriterionHandler(imageParams.criterion)
        if imageParams.problem_type == "CLASSIFICATION":
            trainer = ClassificationTrainer(
                train_loader,
                test_loader,
                model,
                optimizer,
                criterionHandler,
                imageParams.epochs,
                dataCreator.getCategoryList(),
            )
            detailedTrainResultsData = collectTrainingResults(trainer, basic_info, True)

            # save detailedTrainResultsData
            saveDetailedTrainResultsDataToS3(detailedTrainResultsData)
