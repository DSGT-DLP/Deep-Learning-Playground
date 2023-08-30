from typing import Literal, Optional
from django.http import HttpRequest
from ninja import Router, Schema
from training.core.criterion import getCriterionHandler
from training.core.dataset import SklearnDatasetCreator
from training.core.dl_model import DLModel
from torch.utils.data import DataLoader
from training.core.optimizer import getOptimizer
from training.core.trainer import ClassificationTrainer, RegressionTrainer
from training.routes.tabular.schemas import TabularParams
from training.core.authenticator import FirebaseAuth

router = Router()


@router.post("", auth=FirebaseAuth())
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
