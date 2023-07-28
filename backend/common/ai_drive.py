import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_california_housing
from aws_helpers.dynamo_db_utils.trainspace_db import TrainspaceData
from aws_helpers.s3_utils.s3_bucket_names import FILE_UPLOAD_BUCKET_NAME
from aws_helpers.s3_utils.s3_client import (
    make_train_bucket_path,
    read_df_from_bucket,
    read_from_bucket,
)

from common.constants import (
    IMAGE_FILE_DOWNLOAD_TMP_PATH,
    LOGGER_FORMAT,
    ONNX_MODEL,
)
from common.dataset import read_dataset, loader_from_zipped
from common.default_datasets import (
    get_default_dataset,
    get_img_default_dataset_loaders,
)
from common.optimizer import get_optimizer
from common.utils import *

from dl.dl_model import DLModel
from dl.dl_model_parser import parse_deep_user_architecture
from dl.dl_trainer import train_deep_model, train_deep_image_classification

from ml.ml_trainer import train_classical_ml_model
from ml.ml_model_parser import get_object_ml
from dlp_logging import logger


def dl_tabular_drive(trainspace_data: TrainspaceData):
    """
    Driver function/entrypoint into backend for deep learning model. Onnx file is generated containing model architecture for user to visualize in netron.app
    Args:
        user_arch (list): list that contains user defined deep learning architecture
        fileURL (str): URL of the dataset file, if provided by user
        params (dict): dictionary containing all the parameters for the model, e.g. criterion and problem type
        json_csv_data_str (str, optional): json string of the dataset, if provided by user. Defaults to "".
        customModelName (str, optional): name of the custom model. Defaults to None.
    :return: a dictionary containing the epochs, train and test accuracy and loss results, each in a list

    NOTE:
         CSV_FILE_NAME is the data csv file for the torch model. Assumed that you have one dataset file
    """

    """
    Params:
        criterion (str): What loss function to use
        optimizer (str): What optimizer does the user wants to use (Adam or SGD for now, but more support in later iterations)
        problem type (str): "classification" or "regression" problem
        target (str): name of target column
        features (list): list of columns in dataframe for the feature based on user selection
        default (str, optional): the default dataset chosen by the user. Defaults to None.
        test_size (float, optional): size of test set in train/test split (percentage). Defaults to 0.2.
        epochs (int, optional): number of epochs/rounds to run model on
        shuffle (bool, optional): should the dataset be shuffled prior to train/test split
    """
    params = trainspace_data.parameters_data

    target = params.get("target_col", None)
    features = params.get("features", None)
    problem_type = params["problem_type"]
    optimizer_name = params["optimizer_name"]
    criterion = params["criterion"]
    default = None
    if trainspace_data.dataset_data.get("is_default_dataset"):
        default = trainspace_data.dataset_data["name"]
    epochs = params.get("epochs", 5)
    shuffle = params.get("shuffle", True)
    test_size = params.get("test_size", 0.2)
    batch_size = params.get("batch_size", 20)

    user_arch = params["layers"]

    category_list = []

    if default and problem_type.upper() == "CLASSIFICATION":
        X, y, category_list = get_default_dataset(default.upper(), target, features)
    elif default and problem_type.upper() == "REGRESSION":
        X, y, _ = get_default_dataset(default.upper(), target, features)
    else:
        input_df = read_df_from_bucket(
            FILE_UPLOAD_BUCKET_NAME, make_train_bucket_path(trainspace_data)
        )
        y = input_df[target]
        X = input_df[features]

        if problem_type.upper() == "CLASSIFICATION" and not default:
            category_list = []
            target_categories = input_df[target]
            for category in target_categories:
                if category not in category_list:
                    category_list.append(category)

    if len(y) * test_size < batch_size or len(y) * (1 - test_size) < batch_size:
        raise ValueError("reduce batch size, not enough values in dataframe")

    if problem_type.upper() == "CLASSIFICATION":
        # label encode the categorical values to numbers
        y = y.astype("category")
        y = y.cat.codes
        logger.info(y.head())

    # Convert to tensor
    if shuffle and problem_type.upper() == "CLASSIFICATION":
        # using stratify only for classification problems to ensure correct AUC calculation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle
        )

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_tensors(
        X_train, X_test, y_train, y_test
    )
    # Build the Deep Learning model that the user wants
    model = DLModel(parse_deep_user_architecture(user_arch))
    logger.info(f"model: {model}")

    optimizer = get_optimizer(model, optimizer_name=optimizer_name, learning_rate=0.05)

    logger.info(f"loss criterion: {criterion}")
    train_loader, test_loader = get_dataloaders(
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor,
        batch_size=batch_size,
    )

    train_loss_results = train_deep_model(
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        epochs,
        problem_type,
        category_list,
    )
    torch.onnx.export(model, X_train_tensor, ONNX_MODEL)

    return train_loss_results


def dl_img_drive(trainspace_data: TrainspaceData):
    params = trainspace_data.parameters_data

    optimizer_name = params["optimizer_name"]
    criterion = params["criterion"]
    default = (
        trainspace_data.dataset_data["name"]
        if trainspace_data.dataset_data.get("is_default_dataset")
        else None
    )
    epochs = params.get("epochs", 5)
    shuffle = params.get("shuffle", True)
    batch_size = params.get("batch_size", 20)
    train_transform = params["train_transform"]
    test_transform = params["test_transform"]

    user_arch = params["layers"]

    model = DLModel(parse_deep_user_architecture(user_arch))

    train_transform = parse_deep_user_architecture(train_transform)
    test_transform = parse_deep_user_architecture(test_transform)

    if not default:
        uid = trainspace_data.uid
        filename = trainspace_data.dataset_data["name"]
        read_from_bucket(
            FILE_UPLOAD_BUCKET_NAME,
            make_train_bucket_path(trainspace_data),
            filename,
            IMAGE_FILE_DOWNLOAD_TMP_PATH,
        )
        zip_file = os.path.join(IMAGE_FILE_DOWNLOAD_TMP_PATH, filename)
        train_loader, test_loader = loader_from_zipped(
            zip_file, batch_size, shuffle, train_transform, test_transform
        )
    else:
        train_loader, test_loader = get_img_default_dataset_loaders(
            default, train_transform, test_transform, batch_size, shuffle
        )

    logger.info("got data loaders")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(
        device
    )  # model should go to GPU before initializing optimizer  https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least/66096687#66096687

    optimizer = get_optimizer(model, optimizer_name=optimizer_name, learning_rate=0.05)

    train_loss_results = train_deep_image_classification(
        model, train_loader, test_loader, optimizer, criterion, epochs, device
    )
    return train_loss_results


def ml_drive(trainspace_data: TrainspaceData):
    """
    Driver function/endpoint into backend for training a classical ML model (eg: SVC, SVR, DecisionTree, Naive Bayes, etc)

    Args:
        user_model (str): What ML model and parameters does the user want
        problem_type (str): "CLASSIFICATION" or "REGRESSION" problem
        target (str, optional): name of target column. Defaults to None.
        features (list, optional): list of columns in dataframe for the feature based on user selection. Defaults to None.
        default (bool, optional): use the iris dataset for default classification or california housing for default regression. Defaults to False.
        test_size (float, optional): size of test set in train/test split (percentage). Defaults to 0.2.
        shuffle (bool, optional): should the dataset be shuffled prior to train/test split
    """
    params = trainspace_data.parameters_data

    target = params.get("target_col", None)
    features = params.get("features", None)
    problem_type = params["problem_type"]
    default = (
        trainspace_data.dataset_data["name"]
        if trainspace_data.dataset_data.get("is_default_dataset")
        else None
    )
    shuffle = params.get("shuffle", True)
    test_size = params.get("test_size", 0.2)

    if len(params["layers"]) != 1:
        raise Exception("Only one layer is supported for classical ML models")

    user_model = params["layers"][0]

    try:
        if default and problem_type.upper() == "CLASSIFICATION":
            X, y, target_names = get_default_dataset(default.upper(), target, features)
            logger.info(y.head())
        elif default and problem_type.upper() == "REGRESSION":
            X, y, target_names = get_default_dataset(default.upper(), target, features)
        else:
            input_df = read_df_from_bucket(
                FILE_UPLOAD_BUCKET_NAME, make_train_bucket_path(trainspace_data)
            )
            input_df[target] = input_df[target].astype("category").cat.codes
            y = input_df[target]
            X = input_df[features]

        if shuffle and problem_type.upper() == "CLASSIFICATION":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=True, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, shuffle=shuffle
            )
        model = get_object_ml(user_model)
        train_ml_results = train_classical_ml_model(
            model, X_train, X_test, y_train, y_test, problem_type=problem_type
        )
        return train_ml_results
    except Exception as e:
        raise e
