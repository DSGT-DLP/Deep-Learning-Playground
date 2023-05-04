import time
import json
import traceback
import os
import shutil

import backend.aws_helpers.sqs_utils.sqs_client as sqs_helper
import backend.aws_helpers.s3_utils.s3_client as s3_helper
from backend.common.utils import *
from backend.aws_helpers.s3_utils.s3_bucket_names import (
    FILE_UPLOAD_BUCKET_NAME,
    EXECUTION_BUCKET_NAME,
)
from backend.common.constants import (
    UNZIPPED_DIR_NAME,
    ONNX_MODEL,
    SAVED_MODEL_DL,
    SAVED_MODEL_ML,
    DEEP_LEARNING_RESULT_CSV_PATH,
    IMAGE_DETECTION_RESULT_CSV_PATH,
)
from backend.common.utils import csv_to_json
from backend.common.ai_drive import dl_tabular_drive, ml_drive, dl_img_drive
from backend.dl.detection import detection_img_drive
from backend.aws_helpers.dynamo_db_utils.execution_db import updateStatus


def router(msg):
    """
    Routes the message from the SQS Training queue to the appropriate training function.
    
    Args: 
      - msg: request info from the SQS queue
    """
    print("Message received")
    execution_id = msg["execution_id"]
    print(f"{execution_id} is marked as STARTING")
    entryData = {
        "execution_id": msg["execution_id"],
        "user_id": msg["user"]["uid"],
        "name": msg["custom_model_name"],
        "data_source": msg["data_source"],
        "status": "STARTING",
        "timestamp": get_current_timestamp(),
        "progress": 0,
    }
    updateStatus(execution_id=execution_id, status="STARTING", entryData=entryData)
    if msg["route"] == "tabular-run":
        result = tabular_run_route(msg)
        if result[1] != 200:
            print("Error in tabular run route: result is", result)
            updateStatus(execution_id=execution_id, status="ERROR", entryData=entryData)
            return

        updateStatus(execution_id=execution_id, status="SUCCESS", entryData=entryData)
        s3_helper.write_to_bucket(
            SAVED_MODEL_DL,
            EXECUTION_BUCKET_NAME,
            f"{execution_id}/{os.path.basename(SAVED_MODEL_DL)}",
        )
        s3_helper.write_to_bucket(
            ONNX_MODEL,
            EXECUTION_BUCKET_NAME,
            f"{execution_id}/{os.path.basename(ONNX_MODEL)}",
        )
        s3_helper.write_to_bucket(
            DEEP_LEARNING_RESULT_CSV_PATH,
            EXECUTION_BUCKET_NAME,
            f"{execution_id}/{os.path.basename(DEEP_LEARNING_RESULT_CSV_PATH)}",
        )
    elif msg["route"] == "ml-run":
        result = ml_run_route(msg)
        if result[1] != 200:
            updateStatus(execution_id=execution_id, status="ERROR", entryData=entryData)
            return

        updateStatus(execution_id=execution_id, status="SUCCESS", entryData=entryData)
        s3_helper.write_to_bucket(
            SAVED_MODEL_ML,
            EXECUTION_BUCKET_NAME,
            f"{execution_id}/{os.path.basename(SAVED_MODEL_ML)}",
        )
    elif msg["route"] == "img-run":
        print("Running Img run route")
        result = img_run_route(msg)
        print(result)
        if result[1] != 200:
            updateStatus(execution_id=execution_id, status="ERROR", entryData=entryData)
            return

        updateStatus(execution_id=execution_id, status="SUCCESS", entryData=entryData)
        print("execution id status updated after img run complete")
        s3_helper.write_to_bucket(
            SAVED_MODEL_DL,
            EXECUTION_BUCKET_NAME,
            f"{execution_id}/{os.path.basename(SAVED_MODEL_DL)}",
        )
        s3_helper.write_to_bucket(
            ONNX_MODEL,
            EXECUTION_BUCKET_NAME,
            f"{execution_id}/{os.path.basename(ONNX_MODEL)}",
        )
        s3_helper.write_to_bucket(
            DEEP_LEARNING_RESULT_CSV_PATH,
            EXECUTION_BUCKET_NAME,
            f"{execution_id}/{os.path.basename(DEEP_LEARNING_RESULT_CSV_PATH)}",
        )
        print("img run result files successfully uploaded to s3")
    elif msg["route"] == "object-detection":
        result = object_detection_route(msg)
        if result[1] != 200:
            updateStatus(execution_id=execution_id, status="ERROR", entryData=entryData)
            return

        updateStatus(execution_id=execution_id, status="SUCCESS", entryData=entryData)
        s3_helper.write_to_bucket(
            IMAGE_DETECTION_RESULT_CSV_PATH,
            EXECUTION_BUCKET_NAME,
            f"{execution_id}/{os.path.basename(IMAGE_DETECTION_RESULT_CSV_PATH)}",
        )


# Wrapper for dl_tabular_drive() function
def tabular_run_route(request_data):
    """
    Route to run train for a tabular train request

    Args:
        request_data (json): Request metadata for tabular train

    Returns:
        train_loss_results: json encapsulating the train result
    """
    try:
        user_arch = request_data["user_arch"]
        fileURL = request_data["file_URL"]
        uid = request_data["user"]["uid"]
        json_csv_data_str = request_data["csv_data"]
        customModelName = request_data["custom_model_name"]

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

        train_loss_results = dl_tabular_drive(
            user_arch, fileURL, params, json_csv_data_str, customModelName
        )

        print(train_loss_results)
        return send_train_results(train_loss_results)

    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


# Wrapper for ml_drive() function
def ml_run_route(request_data):
    """
    Route to run train for a classical ML request

    Args:
        request_data (json): Request metadata for classical ML train

    Returns:
        train_loss_results: json encapsulating the train result
    """
    try:
        user_model = request_data["user_arch"]
        problem_type = request_data["problem_type"]
        target = request_data["target"]
        uid = request_data["user"]["uid"]
        features = request_data["features"]
        default = request_data["using_default_dataset"]
        shuffle = request_data["shuffle"]

        train_loss_results = ml_drive(
            user_model=user_model,
            problem_type=problem_type,
            target=target,
            features=features,
            default=default,
            shuffle=shuffle,
        )
        print(train_loss_results)
        return send_train_results(train_loss_results)

    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


# Wrapper for dl_img_drive() function
def img_run_route(request_data):
    """
    Route to run train for a image request

    Args:
        request_data (json): Request metadata for image train

    Returns:
        train_loss_results: json encapsulating the train result
    """
    IMAGE_UPLOAD_FOLDER = "./backend/image_data_uploads"
    try:
        train_transform = request_data["train_transform"]
        test_transform = request_data["test_transform"]
        user_arch = request_data["user_arch"]
        criterion = request_data["criterion"]
        optimizer_name = request_data["optimizer_name"]
        default = request_data["using_default_dataset"]
        epochs = request_data["epochs"]
        batch_size = request_data["batch_size"]
        shuffle = request_data["shuffle"]
        uid = request_data["user"]["uid"]
        customModelName = request_data["custom_model_name"]

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

        print("training successfully finished")
        return send_train_results(train_loss_results)

    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


# Wrapper for detection_img_drive() function
def object_detection_route(request_data):
    """
    Route to run train for a object detection request

    Args:
        request_data (json): Request metadata for object detection train

    Returns:
        train_loss_results: json encapsulating the train result
    """
    IMAGE_UPLOAD_FOLDER = "./backend/image_data_uploads"
    try:
        problem_type = request_data["problem_type"]
        detection_type = request_data["detection_type"]
        uid = request_data["user"]["uid"]
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


def send_success(results: dict):
    return (json.dumps({"success": True, **results}), 200)


def send_error(message: str):
    return (json.dumps({"success": False, "message": message}), 400)


def send_train_results(train_loss_results: dict):
    return send_success(
        {
            "message": "Dataset trained and results outputted successfully",
            "dl_results": csv_to_json(),
            "auxiliary_outputs": train_loss_results,
        }
    )


def send_detection_results(object_detection_results: dict):
    return send_success(
        {
            "message": "Detection worked successfully",
            "dl_results": object_detection_results["dl_results"],
            "auxiliary_outputs": object_detection_results["auxiliary_outputs"],
        }
    )


def send_traceback_error():
    return send_error(traceback.format_exc(limit=1))


def empty_message(message):
    """
    Returns if JSON is empty
    """
    return json.dumps(message) == "{}"


# Polls for messages from the SQS queue, and handles them.
while True:
    # Get message from queue
    print("Polling for messages...\n")
    msg = sqs_helper.receive_message()

    if not empty_message(msg):
        print(msg)
        # Handle data
        router(msg)
    else:
        # No message found
        print("No message found")

    # Check again
    time.sleep(1)
