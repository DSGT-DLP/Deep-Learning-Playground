import shutil
import traceback

from flask import Blueprint
from flask import request

from backend.aws_helpers.dynamo_db_utils.execution_db import (
    createUserExecutionsData,
    updateStatus,
)
from backend.common.ai_drive import dl_tabular_drive, dl_img_drive, ml_drive
from backend.common.constants import UNZIPPED_DIR_NAME
from backend.common.utils import *
from backend.dl.detection import detection_img_drive
from backend.endpoints.utils import send_traceback_error

train_bp = Blueprint("train", __name__)


@train_bp.route("/tabular-run", methods=["POST"])
def tabular_run():
    try:
        request_data = json.loads(request.data)

        user_arch = request_data["user_arch"]
        fileURL = request_data["file_URL"]
        uid = request_data["user"]["uid"]
        json_csv_data_str = request_data["csv_data"]
        customModelName = request_data["custom_model_name"]
        execution_id = request_data["execution_id"]

        params = {
            "target": request_data["target"],
            "features": request_data["features"],
            "problem_type": request_data["problem_type"],
            "optimizer_name": request_data["optimizer_name"],
            "criterion": request_data["criterion"],
            "default": request_data["using_default_dataset"],
            "epochs": request_data["epochs"],
            "shuffle": request_data["shuffle"],
            "test_size": request_data["test_size"],
            "batch_size": request_data["batch_size"],
        }

        createUserExecutionsData(
            {
                "execution_id": execution_id,
                "user_id": uid,
                "name": customModelName,
                "data_source": "TABULAR",
                "status": "STARTING",
                "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "progress": 0,
            }
        )
        train_loss_results = dl_tabular_drive(
            user_arch, fileURL, params, json_csv_data_str, customModelName
        )
        train_loss_results["user_arch"] = user_arch
        print(train_loss_results)
        updateStatus(execution_id, "SUCCESS")
        return send_train_results(train_loss_results)

    except Exception:
        updateStatus(execution_id, "ERROR")
        print(traceback.format_exc())
        return send_traceback_error()


@train_bp.route("/ml-run", methods=["POST"])
def ml_run():
    try:
        request_data = json.loads(request.data)

        user_model = request_data["user_arch"]
        problem_type = request_data["problem_type"]
        target = request_data["target"]
        features = request_data["features"]
        json_csv_data_str = request_data["csv_data"]
        default = request_data["using_default_dataset"]
        shuffle = request_data["shuffle"]

        train_loss_results = ml_drive(
            user_model=user_model,
            problem_type=problem_type,
            target=target,
            features=features,
            default=default,
            json_csv_data_str=json_csv_data_str,
            shuffle=shuffle,
        )
        train_loss_results["user_arch"] = user_model
        print(train_loss_results)
        return send_train_results(train_loss_results)

    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@train_bp.route("/img-run", methods=["POST"])
def img_run():
    IMAGE_UPLOAD_FOLDER = "./backend/image_data_uploads"
    try:
        request_data = json.loads(request.data)

        train_transform = request_data["train_transform"]
        test_transform = request_data["test_transform"]
        user_arch = request_data["user_arch"]
        criterion = request_data["criterion"]
        optimizer_name = request_data["optimizer_name"]
        default = request_data["using_default_dataset"]
        epochs = request_data["epochs"]
        batch_size = request_data["batch_size"]
        shuffle = request_data["shuffle"]
        customModelName = request_data["custom_model_name"]
        uid = request_data["user"]["uid"]
        execution_id = request_data["execution_id"]
        train_loss_results = dl_img_drive(
            train_transform,
            test_transform,
            user_arch,
            criterion,
            optimizer_name,
            default,
            epochs,
            batch_size,
            shuffle,
            IMAGE_UPLOAD_FOLDER,
        )
        train_loss_results["user_arch"] = user_arch
        print("training successfully finished")
        return send_train_results(train_loss_results)

    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()

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


@train_bp.route("/object-detection", methods=["POST"])
def object_detection_run():
    IMAGE_UPLOAD_FOLDER = "./backend/image_data_uploads"
    try:
        request_data = json.loads(request.data)
        problem_type = request_data["problem_type"]
        detection_type = request_data["detection_type"]
        transforms = request_data["transforms"]
        image = detection_img_drive(
            IMAGE_UPLOAD_FOLDER, detection_type, problem_type, transforms
        )
        return send_detection_results(image)
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()
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
