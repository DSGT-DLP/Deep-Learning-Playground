import pandas as pd
import traceback
import os
from flask import Flask, request, copy_current_request_context
from werkzeug.utils import secure_filename
import shutil

from backend.common.utils import *
from backend.common.constants import CSV_FILE_NAME, ONNX_MODEL, UNZIPPED_DIR_NAME
from backend.common.dataset import (
    loader_from_zipped,
    read_local_csv_file,
    read_dataset,
    dataset_from_zipped,
)
from backend.common.optimizer import get_optimizer
from backend.dl.dl_model_parser import parse_deep_user_architecture, get_object
from backend.dl.dl_trainer import (
    train_deep_classification_model,
    train_deep_model,
    get_deep_predictions,
    train_deep_image_classification,
)
from backend.ml.ml_trainer import train_classical_ml_model
from backend.dl.dl_model import DLModel
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from backend.common.default_datasets import (
    get_default_dataset,
    get_img_default_dataset_loaders,
    get_img_default_dataset,
)
from flask_cors import CORS
from backend.common.email_notifier import send_email
from flask import send_from_directory
from flask_socketio import SocketIO
import eventlet
import datetime, threading
from backend.dl.pretrained import train

app = Flask(
    __name__,
    static_folder=os.path.join(
        os.path.dirname(os.getcwd()), "frontend", "playground-frontend", "build"
    ),
)
CORS(app)
socket = SocketIO(app, cors_allowed_origins="*", ping_timeout=600, ping_interval=15)


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

        if shuffle and problem_type.upper() == "CLASSIFICATION":
            # using stratify only for classification problems to ensure correct AUC calculation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0, shuffle=True, stratify=y
            )
        else:
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
    send_progress,
    target=None,
    features=None,
    default=None,
    test_size=0.2,
    epochs=5,
    shuffle=True,
    json_csv_data_str="",
    batch_size=20,
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
                X, y, test_size=test_size, random_state=0, shuffle=True, stratify=y
            )
        else:
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
            send_progress,
        )
        pred, ground_truth = get_deep_predictions(model, test_loader)
        torch.onnx.export(model, X_train_tensor, ONNX_MODEL)

        return train_loss_results

    except Exception as e:
        raise e


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def root(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


@socket.on("frontendLog")
def frontend_log(log):
    app.logger.info(f'"frontend: {log}"')


@socket.on("img-run")
def testing(request_data):
    try:
        print("backend started")
        IMAGE_UPLOAD_FOLDER = "./backend/image_data_uploads"
        # request_data = json.loads(request.data)
        train_transform = request_data["train_transform"]
        test_transform = request_data["test_transform"]
        user_arch = request_data["user_arch"]
        criterion = request_data["criterion"]
        optimizer_name = request_data["optimizer_name"]
        default = request_data["using_default_dataset"]
        epochs = request_data["epochs"]
        batch_size = request_data["batch_size"]
        shuffle = request_data["shuffle"]

        print(train_transform)
        print(test_transform)

        # upload()
        print(user_arch)
        print("sdsakdnasjfk", request_data["user_arch"])
        model = DLModel(parse_deep_user_architecture(user_arch))

        train_transform = parse_deep_user_architecture(train_transform)
        test_transform = parse_deep_user_architecture(test_transform)
        print(train_transform)
        print(test_transform)
        if not default:
            for x in os.listdir(IMAGE_UPLOAD_FOLDER):
                if x != ".gitkeep":
                    zip_file = os.path.join(os.path.abspath(IMAGE_UPLOAD_FOLDER), x)
                    break
            train_loader, test_loader = loader_from_zipped(
                zip_file, batch_size, shuffle, train_transform, test_transform
            )
        else:
            train_loader, test_loader = get_img_default_dataset_loaders(
                default, test_transform, train_transform, batch_size, shuffle
            )

        print("got data loaders")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if(torch.cuda.is_available()):
            print("cuda")
        else:
            print("cpu")
        model.to(
            device
        )  # model should go to GPU before initializing optimizer  https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least/66096687#66096687

        optimizer = get_optimizer(
            model, optimizer_name=optimizer_name, learning_rate=0.05
        )

        train_loss_results = train_deep_image_classification(
            model,
            train_loader,
            test_loader,
            optimizer,
            criterion,
            epochs,
            device,
            send_progress=send_progress,
        )

        print("training successfully finished")

        socket.emit(
            "trainingResult",
            {
                "success": True,
                "message": "Dataset trained and results outputted successfully",
                "dl_results": csv_to_json(),
                "auxiliary_outputs": train_loss_results,
                "status": 200,
            },
        )
    except Exception as e:
        print(traceback.format_exc())
        socket.emit(
            "trainingResult",
            {"success": False, "message": traceback.format_exc(limit=1), "status": 400},
        )
    finally:
        for x in os.listdir(IMAGE_UPLOAD_FOLDER):
            if x != ".gitkeep":
                file_rem = os.path.join(os.path.abspath(IMAGE_UPLOAD_FOLDER), x)
                if os.path.isdir(file_rem):
                    shutil.rmtree(file_rem)
                else:
                    os.remove(file_rem)
        if os.path.exists(UNZIPPED_DIR_NAME):
            shutil.rmtree(UNZIPPED_DIR_NAME)


@socket.on("runTraining")
def train_and_output(request_data):
    user_arch = request_data["user_arch"]
    criterion = request_data["criterion"]
    optimizer_name = request_data["optimizer_name"]
    problem_type = request_data["problem_type"]
    target = request_data["target"]
    features = request_data["features"]
    default = request_data["using_default_dataset"]
    test_size = request_data["test_size"]
    batch_size = request_data["batch_size"]
    epochs = request_data["epochs"]
    shuffle = request_data["shuffle"]
    csvDataStr = request_data["csv_data"]
    fileURL = request_data["file_URL"]

    try:
        if not default:
            if fileURL:
                read_dataset(fileURL)
            elif csvDataStr:
                pass
            else:
                raise ValueError("Need a file input")

        train_loss_results = dl_drive(
            user_arch=user_arch,
            criterion=criterion,
            optimizer_name=optimizer_name,
            problem_type=problem_type,
            send_progress=send_progress,
            target=target,
            features=features,
            default=default,
            test_size=test_size,
            epochs=epochs,
            shuffle=shuffle,
            json_csv_data_str=csvDataStr,
            batch_size=batch_size,
        )

        socket.emit(
            "trainingResult",
            {
                "success": True,
                "message": "Dataset trained and results outputted successfully",
                "dl_results": csv_to_json(),
                "auxiliary_outputs": train_loss_results,
                "status": 200,
            },
        )

    except Exception:
        print(traceback.format_exc())
        socket.emit(
            "trainingResult",
            {"success": False, "message": traceback.format_exc(limit=1), "status": 400},
        )


@socket.on("pretrain-run")
def train_pretrained(request_data):
    try:
        print("backend started")
        IMAGE_UPLOAD_FOLDER = "./backend/image_data_uploads"
        # request_data = json.loads(request.data)
        train_transform = request_data["train_transform"]
        test_transform = request_data["test_transform"]
        criterion = request_data["criterion"]
        optimizer_name = request_data["optimizer_name"]
        default = request_data["using_default_dataset"]
        epochs = request_data["epochs"]
        batch_size = request_data["batch_size"]
        shuffle = request_data["shuffle"]
        model_name = request_data["model_name"]
        # train_transform.append("transforms.Lambda(lambda x: x.repeat(3, 1, 1) )")
        # test_transform.append("transforms.Lambda(lambda x: x.repeat(3, 1, 1) )")
        train_transform = parse_deep_user_architecture(train_transform)
        test_transform = parse_deep_user_architecture(test_transform)
        if not default:
            zip_file = "tests/zip_files/double_zipped.zip"
            train_dataset, test_dataset = dataset_from_zipped(
                zip_file, test_transform=test_transform, train_transform=train_transform
            )
        else:
            train_dataset, test_dataset = get_img_default_dataset(
                default, test_transform, train_transform
            )

        print("got datasets")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loss_results, learner = train(
            train_dataset,
            test_dataset,
            model_name,
            batch_size,
            criterion,
            epochs,
            optimizer_name=optimizer_name,
            shuffle=shuffle,
            chan_in=train_dataset[0][0].shape[0],
            n_classes=len(train_dataset.classes)
        )
        print("training successfully finished")
        socket.emit(
            "trainingResult",
            {
                "success": True,
                "message": "Dataset trained and results outputted successfully",
                "dl_results": csv_to_json(),
                "auxiliary_outputs": train_loss_results,
                "status": 200,
            },
        )
    except Exception as e:
        print(traceback.format_exc())
        socket.emit(
            "trainingResult",
            {"success": False, "message": traceback.format_exc(limit=1), "status": 400},
        )
    finally:
        for x in os.listdir(IMAGE_UPLOAD_FOLDER):
            if x != ".gitkeep":
                file_rem = os.path.join(os.path.abspath(IMAGE_UPLOAD_FOLDER), x)
                if os.path.isdir(file_rem):
                    shutil.rmtree(file_rem)
                else:
                    os.remove(file_rem)
        if os.path.exists(UNZIPPED_DIR_NAME):
            shutil.rmtree(UNZIPPED_DIR_NAME)


@app.route("/upload", methods=["POST"])
def upload():
    @copy_current_request_context
    def save_file(closeAfterWrite):
        print(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + " dropzone is working"
        )
        f = request.files["file"]
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(
            basepath, "image_data_uploads", secure_filename(f.filename)
        )
        f.save(upload_path)
        closeAfterWrite()
        print(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            + " dropzone has finished its task"
        )

    def passExit():
        pass

    if request.method == "POST":
        f = request.files["file"]
        normalExit = f.stream.close
        f.stream.close = passExit
        t = threading.Thread(target=save_file, args=(normalExit,))
        t.start()
        return "200"
    return "200"


def send_progress(progress):
    socket.emit("trainingProgress", progress)
    eventlet.greenthread.sleep(
        0
    )  # to prevent logs from being grouped and sent together at the end of training


if __name__ == "__main__":
    socket.run(app, debug=True, host="0.0.0.0", port=8000)
