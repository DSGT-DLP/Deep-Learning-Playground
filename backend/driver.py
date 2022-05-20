from inspect import trace
import pandas as pd
import traceback
import os
from flask import Flask, json, request, jsonify

from utils import *
from enum import Enum
from constants import CSV_FILE_NAME, ONNX_MODEL
from loss import LossFunctions
from optimizer import get_optimizer
from model_parser import parse_deep_user_architecture, get_object
from dl_trainer import train_deep_model, get_deep_predictions
from ml_trainer import train_classical_ml_model
from model import DLModel
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split

app = Flask(__name__)


def get_default_dataset(dataset):
    """
    If user doesn't specify dataset
    Args:
        dataset_name (str): Which default dataset are you using (built in functions like load_boston(), load_iris())
    Returns:
        X: input (default dataset)
        y: target (default dataset)
    """
    input_df = pd.DataFrame(dataset.data)
    input_df["class"] = dataset.target
    input_df.columns = dataset.feature_names + ["class"]
    input_df.dropna(how="all", inplace=True)  # remove any empty lines
    y = pd.Series(dataset.target)
    X = input_df[dataset.feature_names]
    print(f"iris dataset = {input_df.head()}")
    return X, y


def ml_drive(user_model, problem_type, target=None, features=None, default=False, test_size=0.2, shuffle=True):
    """
    Driver function/endpoint into backend for training a classical ML model (eg: SVC, SVR, DecisionTree, Naive Bayes, etc) 

    Args:
        user_model (str): What ML model and parameters does the user want
        problem_type (str): "classification" or "regression" problem
        target (str, optional): name of target column. Defaults to None.
        features (list, optional): list of columns in dataframe for the feature based on user selection. Defaults to None.
        default (bool, optional): use the iris dataset for default classifiction or california housing for default regression. Defaults to False.
        test_size (float, optional): size of test set in train/test split (percentage). Defaults to 0.2.
        shuffle (bool, optional): should the dataset be shuffled prior to train/test split
    """
    try:
        if (default and problem_type.upper() == "CLASSIFICATION"):
            dataset = load_iris()
            X, y = get_default_dataset(dataset)
            print(y.head())
        elif default and problem_type.upper() == "REGRESSION":
            # If the user specifies no dataset, use california housing as default regression
            dataset = fetch_california_housing()
            X, y = get_default_dataset(dataset)
        else:
            input_df = pd.read_csv(CSV_FILE_NAME)
            y = input_df[target]
            X = input_df[features]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0, shuffle=shuffle)
        model = get_object(user_model)
        train_classical_ml_model(
            model, X_train, X_test, y_train, y_test, problem_type=problem_type)
    except Exception:
        traceback.print_exc()
        return traceback.format_exc(limit=1)


def dl_drive(
    user_arch,
    criterion,
    optimizer_name,
    problem_type,
    target=None,
    features=None,
    default=False,
    test_size=0.2,
    epochs=5,
    shuffle=True,
):
    """
    Driver function/entrypoint into backend for deep learning model. Onnx file is generated containing model architecture for user to visualize in netron.app
    Args:
        user_arch (list): list that contains user defined deep learning architecture
        criterion (str): What loss function to use
        optimizer (str): What optimizer does the user wants to use (Adam or SGD for now, but more support in later iterations)
        problem type (str): "classification" or "regression" problem
        target (str): name of target column
        features (list): list of columns in dataframe for the feature based on user selection
        default (bool, optional): use the iris dataset for default classifiction or california housing for default regression. Defaults to False.
        test_size (float, optional): size of test set in train/test split (percentage). Defaults to 0.2.
        epochs (int, optional): number of epochs/rounds to run model on
        shuffle (bool, optional): should the dataset be shuffled prior to train/test split

    NOTE:
         CSV_FILE_NAME is the data csv file for the torch model. Assumed that you have one dataset file
    """
    try:
        if default and problem_type.upper() == "CLASSIFICATION":
            # If the user specifies no dataset, use iris as the default classification
            dataset = load_iris()
            X, y = get_default_dataset(dataset)
            print(y.head())
        elif default and problem_type.upper() == "REGRESSION":
            # If the user specifies no dataset, use california housing as default regression
            dataset = fetch_california_housing()
            X, y = get_default_dataset(dataset)
        else:
            input_df = pd.read_csv(CSV_FILE_NAME)
            y = input_df[target]
            X = input_df[features]

        if problem_type.upper() == "CLASSIFICATION":
            # label encode the categorical values to numbers
            y = y.astype("category")
            y = y.cat.codes
            print(y.head())

        # Convert to tensor
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0, shuffle=shuffle
        )
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_tensors(
            X_train, X_test, y_train, y_test
        )

        # Build the Deep Learning model that the user wants
        model = DLModel(parse_deep_user_architecture(user_arch))
        print(f"model: {model}")
        optimizer = get_optimizer(
            model, optimizer_name=optimizer_name, learning_rate=0.05
        )
        criterion = LossFunctions.get_loss_obj(LossFunctions[criterion])
        print(f"loss criterion: {criterion}")
        train_loader, test_loader = get_dataloaders(
            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size=20
        )
        train_deep_model(
            model, train_loader, test_loader, optimizer, criterion, epochs, problem_type
        )
        pred, ground_truth = get_deep_predictions(model, test_loader)
        torch.onnx.export(model, X_train_tensor, ONNX_MODEL)

    except Exception:
        return traceback.format_exc(limit=1)  # give exception in string format


@app.route('/run', methods=['GET', 'POST'])
def hello():
    request_data = json.loads(request.data)
    user_arch = request_data['user_arch']
    criterion = request_data['criterion']
    optimizer_name = request_data['optimizer_name']
    problem_type = request_data['problem_type']
    default = request_data['default']
    epochs = request_data['epochs']

    if request.method == 'POST':
        print(
            dl_drive(
                user_arch=user_arch,
                criterion="CELOSS",
                optimizer_name="SGD",
                problem_type=problem_type,
                default=default,
                epochs=epochs,
            )
        )
        return jsonify({"success": True, "message": "Contact deleted successfully 22"}), 201

    return jsonify({"success": True}), 200


if __name__ == "__main__":
    # TODO Faris complete frontend implementation and visualization
    # print(
    #     dl_drive(
    #         ["nn.Linear(4, 10)", "nn.ReLU()", "nn.Linear(10, 3)", "nn.Softmax()"],
    #         "CELOSS",
    #         "SGD",
    #         problem_type="classification",
    #         default=True,
    #         epochs=10,
    #     )
    # )


    # TODO Faris to implement the frontend for this
    # print(ml_drive("DecisionTreeClassifier(max_depth=3, random_state=15)",
    #       problem_type="classification", default=True))
    app.run(debug=True)
