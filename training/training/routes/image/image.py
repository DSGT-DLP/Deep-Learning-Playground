from typing import Literal, Optional
from django.http import HttpRequest
from ninja import Router, Schema
from training.core.criterion import getCriterionHandler
from training.core.dl_model import DLModel
from training.core.dataset import ImageDefaultDatasetCreator
from torch.utils.data import DataLoader
from training.core.optimizer import getOptimizer
from training.core.trainer import ClassificationTrainer
from training.routes.image.schemas import ImageParams
from training.core.authenticator import FirebaseAuth

router = Router()


@router.post("", auth=FirebaseAuth())
def imageTrain(request: HttpRequest, imageParams: ImageParams):
    if imageParams.default:
        dataCreator = ImageDefaultDatasetCreator.fromDefault(
            imageParams.default
        )
        print(vars(dataCreator))
        train_loader = dataCreator.createTrainDataset()
        test_loader = dataCreator.createTestDataset()
        # train_loader = DataLoader(
        #     dataCreator.createTrainDataset(),
        #     batch_size=imageParams.batch_size,
        #     shuffle=False,
        #     drop_last=True,
        # )

        # test_loader = DataLoader(
        #     dataCreator.createTestDataset(),
        #     batch_size=imageParams.batch_size,
        #     shuffle=False,
        #     drop_last=True,
        # )

        model = DLModel.fromLayerParamsList(imageParams.user_arch)
        print(f'model is: {model}')
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
            for epoch_result in trainer:
                print(epoch_result)
            print(trainer.labels_last_epoch, trainer.y_pred_last_epoch)
            print(trainer.generate_confusion_matrix())
            print(trainer.generate_AUC_ROC_CURVE())
            return trainer.generate_AUC_ROC_CURVE()
