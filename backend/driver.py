import pandas as pd
import traceback
import os
from flask import Flask, json, request, jsonify

from utils import *
from constants import CSV_FILE_NAME, ONNX_MODEL
from dataset import read_local_csv_file, read_dataset
from optimizer import get_optimizer
from model_parser import parse_deep_user_architecture, get_object
from dl_trainer import train_deep_model, get_deep_predictions
from ml_trainer import train_classical_ml_model
from model import DLModel
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from default_datasets import get_default_dataset
from flask_cors import CORS
from email_notifier import send_email
from flask import send_from_directory

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(
    os.getcwd()), 'frontend', 'playground-frontend', 'build'))
CORS(app)


def ml_drive(
    user_model,
    problem_type,
    target=None,
    features=None,
    default=False,
    test_size=0.2,
    shuffle=True,
):
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
        if default and problem_type.upper() == "CLASSIFICATION":
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
            X, y, test_size=test_size, random_state=0, shuffle=shuffle
        )
        model = get_object(user_model)
        train_classical_ml_model(
            model, X_train, X_test, y_train, y_test, problem_type=problem_type
        )
    except Exception as e:
        raise e


def dl_drive(
    user_arch,
    criterion,
    optimizer_name,
    problem_type,
    target=None,
    features=None,
    default=None,
    test_size=0.2,
    epochs=5,
    shuffle=True,
    json_csv_data_str="",
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
        default (str, optional): the default dataset chosen by the user. Defaults to None.
        test_size (float, optional): size of test set in train/test split (percentage). Defaults to 0.2.
        epochs (int, optional): number of epochs/rounds to run model on
        shuffle (bool, optional): should the dataset be shuffled prior to train/test split
    :return: a dictionary containing the epochs, train and test accuracy and loss results, each in a list

    NOTE:
         CSV_FILE_NAME is the data csv file for the torch model. Assumed that you have one dataset file
    """
    try:
        if default and problem_type.upper() == "CLASSIFICATION":
            X, y = get_default_dataset(default.upper())
            print(y.head())
        elif default and problem_type.upper() == "REGRESSION":
            X, y = get_default_dataset(default.upper())
        else:
            if json_csv_data_str:
                input_df = pd.read_json(json_csv_data_str, orient="records")

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
        # criterion = LossFunctions.get_loss_obj(LossFunctions[criterion])
        print(f"loss criterion: {criterion}")
        train_loader, test_loader = get_dataloaders(
            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size=20
        )
        train_loss_results = train_deep_model(
            model, train_loader, test_loader, optimizer, criterion, epochs, problem_type
        )
        pred, ground_truth = get_deep_predictions(model, test_loader)
        torch.onnx.export(model, X_train_tensor, ONNX_MODEL)

        return train_loss_results

    except Exception as e:
        raise e


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def root(path):
    if path != "" and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


@app.route("/run", methods=["GET", "POST"])
def train_and_output():
    print("Hi")
    request_data = json.loads(request.data)

    user_arch = request_data["user_arch"]
    criterion = request_data["criterion"]
    optimizer_name = request_data["optimizer_name"]
    problem_type = request_data["problem_type"]
    target = request_data["target"]
    features = request_data["features"]
    default = request_data["default"]
    test_size = request_data["test_size"]
    epochs = request_data["epochs"]
    shuffle = request_data["shuffle"]
    csvDataStr = request_data["csvData"]
    fileURL = request_data["fileURL"]
    email = request_data["email"]
    if request.method == "POST":
        if not default:
            if fileURL:
                read_dataset(fileURL)
            elif csvDataStr:
                pass
            else:
                raise ValueError("Need a file input")
                return

        try:
            train_loss_results = dl_drive(
                user_arch=user_arch,
                criterion=criterion,
                optimizer_name=optimizer_name,
                problem_type=problem_type,
                target=target,
                features=features,
                default=default,
                test_size=test_size,
                epochs=epochs,
                shuffle=shuffle,
                json_csv_data_str=csvDataStr
            )
            # If the length of the email is greater than 0 then that means a valid email has been
            # inputted for the ONNX file to be sent to the user.
            if len(email) != 0:
                send_email(
                    email,
                    "Your ONNX file and visualizations from Deep Learning Playground",
                    "Attached is the ONNX file and visualizations that you just created in Deep Learning Playground. Please notify us if there are any problems.",
                    [ONNX_MODEL, LOSS_VIZ, ACC_VIZ],
                )
            return (
                jsonify(
                    {
                        "success": True,
                        "message": "Dataset trained and results outputted successfully",
                        "dl_results": csv_to_json(),
                    }
                ),
                200,
            )

        except Exception:
            print(traceback.format_exc())
            return (
                jsonify(
                    {"success": False, "message": traceback.format_exc(limit=1)}),
                400,
            )

    return jsonify({"success": False}), 500


@app.route("/sendemail", methods=["POST"])
def send_email_route():
    request_data = json.loads(request.data)

    # extract data
    required_params = ["email_address", "subject", "body_text"]
    for required_param in required_params:
        if required_param not in request_data:
            return jsonify(
                {"success": False, "message": "Missing parameter " + required_param}
            )

    email_address = request_data["email_address"]
    subject = request_data["subject"]
    body_text = request_data["body_text"]
    if "attachment_array" in request_data:
        attachment_array = request_data["attachment_array"]
        if not isinstance(attachment_array, list):
            return jsonify(
                {
                    "success": False,
                    "message": "Attachment array must be a list of filepaths",
                }
            )
    else:
        attachment_array = None

    # try to send email
    try:
        send_email(email_address, subject, body_text, attachment_array)
        return jsonify({"success": True, "message": "Sent email to " + email_address})
    except Exception:
        print(traceback.format_exc())
        return jsonify({"success": False}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
