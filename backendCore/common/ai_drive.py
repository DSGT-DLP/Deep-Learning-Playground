import json
import traceback
import urllib.request as request
import csv
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from backendCore.common.constants import CSV_FILE_PATH, DEFAULT_DATASETS, ONNX_MODEL
from backendCore.dl.dl_model import DLModel
from backendCore.dl.dl_model_parser import parse_deep_user_architecture
from backendCore.dl.dl_trainer import train_deep_model


def read_dataset(url):
    """
    Given a url to a CSV dataset, read it and build temporary csv file

    Args:
        url (str): URL to dataset
    """
    try:
        r = request.urlopen(url).read().decode("utf8").split("\n")
        reader = csv.reader(r)
        with open(CSV_FILE_PATH, mode="w", newline="") as f:
            csvwriter = csv.writer(f)
            for line in reader:
                csvwriter.writerow(line)
    except Exception as e:
        traceback.print_exc()
        raise Exception(
            "Reading Dataset from URL failed. Might want to check the validity of the URL"
        )


def get_default_dataset(dataset, target=None, features=None):
    """
    If user doesn't specify dataset
    Args:
        dataset (str): Which default dataset are you using (built in functions like load_boston(), load_iris())
    Returns:
        X: input (default dataset)
        y: target (default dataset)
        target_names: a list of strings representing category names (default dataset)
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
            target_names = []
            try:
                target_names = list(raw_data["target_names"])
            except:
                pass
            default_dataset = pd.DataFrame(
                data=np.c_[raw_data["data"], raw_data["target"]],
                columns=raw_data["feature_names"] + ["target"],
            )
            # remove any empty lines
            default_dataset.dropna(how="all", inplace=True)
            if features and target:
                y = default_dataset[target]
                X = default_dataset[features]
            else:
                y = default_dataset["target"]
                X = default_dataset.drop("target", axis=1)
            print(default_dataset.head())
            return X, y, target_names
    except Exception:
        raise Exception(f"Unable to load the {dataset} file into Pandas DataFrame")


def get_tensors(X_train, X_test, y_train, y_test):
    """
    Helper function to convert X_train, X_test, y_train, y_test
    into tensors for dataloader
    Args:
        X_train (pd.DataFrame): X_train (train set)
        X_test (pd.DataFrame): X_test (test set)
        y_train (pd.Series): label/value of target for each row of X_train
        y_test (pd.Series): label/value of target for each row of X_test

    Return:
        X_train, X_test, y_train, y_test in the form of tensor
    """
    X_train_tensor = Variable(torch.Tensor(X_train.to_numpy()))
    y_train_tensor = Variable(torch.Tensor(y_train.to_numpy()))
    X_train_tensor = torch.reshape(
        X_train_tensor, (X_train_tensor.size()[0], 1, X_train_tensor.size()[1])
    )
    y_train_tensor = torch.reshape(y_train_tensor, (y_train_tensor.size()[0], 1))

    X_test_tensor = Variable(torch.Tensor(X_test.to_numpy()))
    y_test_tensor = Variable(torch.Tensor(y_test.to_numpy()))
    X_test_tensor = torch.reshape(
        X_test_tensor, (X_test_tensor.size()[0], 1, X_test_tensor.size()[1])
    )
    y_test_tensor = torch.reshape(y_test_tensor, (y_test_tensor.size()[0], 1))

    X_train_tensor.requires_grad_(True)
    X_test_tensor.requires_grad_(True)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def get_optimizer(model, optimizer_name, learning_rate):
    """
    Given an optimizer name, instantiate the object

    Args:
        model (nn.Module): pytorch model to train
        optimizer_name (str): name of optimizer
        learning_rate (float): learning rate
    """
    if optimizer_name.upper() == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name.upper() == "ADAM":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)


def get_dataloaders(
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size
):
    """
    Helper function to get Dataloaders for X_train, y_train, X_test, y_test
    Args:
        X_train_tensor (torch.Tensor)
        y_train_tensor (torch.Tensor)
        X_test_tensor (torch.Tensor)
        y_test_tensor (torch.Tensor)
        batch_size (int)
    """
    train = TensorDataset(X_train_tensor, y_train_tensor)
    test = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


def dl_tabular_drive(
    user_arch: list,
    fileURL: str,
    params: dict,
    json_csv_data_str: str = "",
    customModelName: str = None,
):
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
    target = params.get("target", None)
    features = params.get("features", None)
    problem_type = params["problem_type"]
    optimizer_name = params["optimizer_name"]
    criterion = params["criterion"]
    default = params.get("default", None)
    epochs = params.get("epochs", 5)
    shuffle = params.get("shuffle", True)
    test_size = params.get("test_size", 0.2)
    batch_size = params.get("batch_size", 20)

    category_list = []
    if not default:
        if fileURL:
            read_dataset(fileURL)
        elif json_csv_data_str:
            pass
        else:
            raise ValueError("Need a file input")

    if default and problem_type.upper() == "CLASSIFICATION":
        X, y, category_list = get_default_dataset(default.upper(), target, features)
    elif default and problem_type.upper() == "REGRESSION":
        X, y, discard = get_default_dataset(default.upper(), target, features)
    else:
        if json_csv_data_str:
            input_df = pd.read_json(json_csv_data_str, orient="records")

        y = input_df[target]
        X = input_df[features]

    if len(y) * test_size < batch_size or len(y) * (1 - test_size) < batch_size:
        raise ValueError("reduce batch size, not enough values in dataframe")

    if problem_type.upper() == "CLASSIFICATION":
        # label encode the categorical values to numbers
        y = y.astype("category")
        y = y.cat.codes
        print(y.head())

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
    print(f"model: {model}")

    optimizer = get_optimizer(model, optimizer_name=optimizer_name, learning_rate=0.05)

    print(f"loss criterion: {criterion}")
    train_loader, test_loader = get_dataloaders(
        X_train_tensor,
        y_train_tensor,
        X_test_tensor,
        y_test_tensor,
        batch_size=batch_size,
    )
    if problem_type.upper() == "CLASSIFICATION" and not default:
        category_list = []
        json_data = json.loads(json_csv_data_str)
        pandas_data = pd.DataFrame.from_dict(json_data)
        target_categories = pandas_data[target]
        for category in target_categories:
            if category not in category_list:
                category_list.append(category)
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
