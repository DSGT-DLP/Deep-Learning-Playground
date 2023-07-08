import time
import json
import traceback
import os
import shutil
from backend.aws_helpers.dynamo_db_utils.constants import TrainStatus
from backend.aws_helpers.dynamo_db_utils.trainspace_db import (
    TrainspaceData,
    updateStatus,
)
import logging
import backend.aws_helpers.sqs_utils.sqs_client as sqs_helper
import backend.aws_helpers.s3_utils.s3_client as s3_helper
from backend.common.utils import *
from backend.aws_helpers.s3_utils.s3_bucket_names import (
    EXECUTION_BUCKET_NAME,
)
from backend.common.constants import (
    IMAGE_FILE_DOWNLOAD_TMP_PATH,
    LOGGER_FORMAT,
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

logging.basicConfig(level=logging.DEBUG, format=LOGGER_FORMAT)
logger = logging.getLogger()

def router(msg):
    """
    Routes the message to the appropriate training function.
    """
    logger.info("Message received")
    request_data = TrainspaceData(**(json.loads(msg)))
    trainspace_id = request_data.trainspace_id
    logger.info(f"{trainspace_id} is marked as STARTING")
    updateStatus(trainspace_id, TrainStatus.STARTING.name)
    data_source = request_data.data_source

    if data_source == "TABULAR":
        result = tabular_run_route(request_data)
        if result[1] != 200:
            logger.warn("Error in tabular run route: result is", result)
            updateStatus(trainspace_id, TrainStatus.ERROR.name)
            return

        updateStatus(trainspace_id, TrainStatus.SUCCESS.name)
        s3_helper.write_to_bucket(
            SAVED_MODEL_DL,
            EXECUTION_BUCKET_NAME,
            f"{trainspace_id}/{os.path.basename(SAVED_MODEL_DL)}",
        )
        s3_helper.write_to_bucket(
            ONNX_MODEL,
            EXECUTION_BUCKET_NAME,
            f"{trainspace_id}/{os.path.basename(ONNX_MODEL)}",
        )
        s3_helper.write_to_bucket(
            DEEP_LEARNING_RESULT_CSV_PATH,
            EXECUTION_BUCKET_NAME,
            f"{trainspace_id}/{os.path.basename(DEEP_LEARNING_RESULT_CSV_PATH)}",
        )
    elif data_source == "CLASSICAL_ML":
        result = ml_run_route(request_data)
        if result[1] != 200:
            updateStatus(trainspace_id, TrainStatus.ERROR.name)
            return

        updateStatus(trainspace_id, TrainStatus.SUCCESS.name)
        s3_helper.write_to_bucket(
            SAVED_MODEL_ML,
            EXECUTION_BUCKET_NAME,
            f"{trainspace_id}/{os.path.basename(SAVED_MODEL_ML)}",
        )
    elif data_source == "IMAGE":
        logger.info("Running Img run route")
        result = img_run_route(request_data)
        logger.info(result)
        if result[1] != 200:
            updateStatus(trainspace_id, TrainStatus.ERROR.name)
            return

        updateStatus(trainspace_id, TrainStatus.SUCCESS.name)
        logger.info("execution id status updated after img run complete")
        s3_helper.write_to_bucket(
            SAVED_MODEL_DL,
            EXECUTION_BUCKET_NAME,
            f"{trainspace_id}/{os.path.basename(SAVED_MODEL_DL)}",
        )
        s3_helper.write_to_bucket(
            ONNX_MODEL,
            EXECUTION_BUCKET_NAME,
            f"{trainspace_id}/{os.path.basename(ONNX_MODEL)}",
        )
        s3_helper.write_to_bucket(
            DEEP_LEARNING_RESULT_CSV_PATH,
            EXECUTION_BUCKET_NAME,
            f"{trainspace_id}/{os.path.basename(DEEP_LEARNING_RESULT_CSV_PATH)}",
        )
        logger.info("img run result files successfully uploaded to s3")
    elif data_source == "OBJECT_DETECTION":
        result = object_detection_route(request_data)
        if result[1] != 200:
            updateStatus(trainspace_id, TrainStatus.ERROR.name)
            return

        updateStatus(trainspace_id, TrainStatus.SUCCESS.name)
        s3_helper.write_to_bucket(
            IMAGE_DETECTION_RESULT_CSV_PATH,
            EXECUTION_BUCKET_NAME,
            f"{trainspace_id}/{os.path.basename(IMAGE_DETECTION_RESULT_CSV_PATH)}",
        )


# Wrapper for dl_tabular_drive() function
def tabular_run_route(trainspace_data: TrainspaceData):
    try:
        train_loss_results = dl_tabular_drive(trainspace_data)

        logger.info(train_loss_results)
        return send_train_results(train_loss_results)

    except Exception:
        logger.warn(traceback.format_exc())
        return send_traceback_error()


# Wrapper for ml_drive() function
def ml_run_route(trainspace_data: TrainspaceData):
    try:
        train_loss_results = ml_drive(trainspace_data)

        logger.info(train_loss_results)
        return send_train_results(train_loss_results)

    except Exception:
        logger.warn(traceback.format_exc())
        return send_traceback_error()


# Wrapper for dl_img_drive() function
def img_run_route(trainspace_data: TrainspaceData):
    try:
        train_loss_results = dl_img_drive(trainspace_data)

        logger.info("training successfully finished")
        return send_train_results(train_loss_results)

    except Exception:
        logger.warn(traceback.format_exc())
        return send_traceback_error()

    finally:
        filename = trainspace_data.dataset_data["name"]
        zip_file = os.path.join(IMAGE_FILE_DOWNLOAD_TMP_PATH, filename)

        os.remove(zip_file)
        if os.path.exists(UNZIPPED_DIR_NAME):
            shutil.rmtree(UNZIPPED_DIR_NAME)


# Wrapper for detection_img_drive function
def object_detection_route(trainspace_data: TrainspaceData):
    try:
        image = detection_img_drive(trainspace_data)
        return send_detection_results(image)
    except Exception:
        logger.warn(traceback.format_exc())
        return send_traceback_error()
    finally:
        filename = trainspace_data.dataset_data["name"]
        img_file = os.path.join(IMAGE_FILE_DOWNLOAD_TMP_PATH, filename)

        os.remove(img_file)
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


def empty_message(message: dict) -> bool:
    """
    Returns if JSON is empty
    """
    return not bool(message)


if __name__ == "__main__":
    # Polls for messages from the SQS queue, and handles them.
    while True:
        # Get message from queue
        logger.info("Polling for messages...\n")
        # msg = sqs_helper.receive_message()
        msg = {}

        if not empty_message(msg):
            logger.info(msg)
            # Handle data
            router(msg)
        else:
            # No message found
            logger.info("No message found")

        # Check again
        time.sleep(1)
