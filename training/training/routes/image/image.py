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
import torchvision.transforms as transforms
import json

router = Router()


@router.post("", auth=FirebaseAuth())
def imageTrain(request: HttpRequest, imageParams: ImageParams):
    transforms = json.loads(imageParams.transforms)
    train_transorms = transformParser(transforms["train_transforms"]) if transforms["train_transforms"] else transforms.ToTensor()
    test_transforms = transformParser(transforms["test_transforms"]) if transforms["test_transforms"] else transforms.ToTensor()
    
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
            for epoch_result in trainer:
                print(epoch_result)
            print(trainer.labels_last_epoch, trainer.y_pred_last_epoch)
            print(trainer.generate_confusion_matrix())
            print(trainer.generate_AUC_ROC_CURVE())
            return trainer.generate_AUC_ROC_CURVE()

def transformParser(transformArray):
    transformsToReturn = transforms.ToTensor()
    for x in transformArray:
        if (x["type"] == "CenterCrop"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.CenterCrop(x["parameters"]["size"])
            )
        elif (x["type"] == "ColorJitter"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.ColorJitter(x["parameters"]["brightness"], x["parameters"]["contrast"], x["parameters"]["saturation"], x["parameters"]["hue"])
            )
        elif (x["type"] == "FiveCrop"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.FiveCrop(x["parameters"]["size"])
            )
        elif (x["type"] == "Grayscale"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.Grayscale(x["parameters"]["num_output_channels"])
            )
        elif (x["type"] == "Pad"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.Pad(x["parameters"]["padding"])
            )
        elif (x["type"] == "RandomAffine"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.RandomAffine(x["parameters"]["degrees"], x["parameters"]["translate"], x["parameters"]["scale"], x["parameters"]["shear"])
            )
        elif (x["type"] == "RandomCrop"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.RandomCrop(x["parameters"]["size"], x["parameters"]["padding"])
            )
        elif (x["type"] == "RandomGrayscale"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.RandomGrayscale(x["parameters"]["p"])
            )
        elif (x["type"] == "RandomHorizontalFlip"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.RandomHorizontalFlip(x["parameters"]["p"])
            )
        elif (x["type"] == "Resize"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.Resize(x["parameters"]["size"])
            )
        elif (x["type"] == "RandomPerspective"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.RandomPerspective(x["parameters"]["distortion_scale"], x["parameters"]["p"])
            )
        elif (x["type"] == "RandomResizedCrop"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                #What do we want to do for other non-single value params
                transforms.RandomResizedCrop(x["parameters"]["size"])
            )
        elif (x["type"] == "RandomVerticalFlip"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.RandomVerticalFlip(x["parameters"]["p"])
            )
        elif (x["type"] == "RandomRotation"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.RandomRotation(x["parameters"]["degrees"])
            )
        elif (x["type"] == "TenCrop"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.TenCrop(x["parameters"]["size"])
            )
        elif (x["type"] == "GaussianBlur"):
            transformsToReturn = transforms.Compose(
                transformsToReturn,
                transforms.GaussianBlur(x["parameters"]["kernel_size"])
            )
    return transformsToReturn