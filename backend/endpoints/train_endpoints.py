import asyncio
from dataclasses import asdict
from decimal import Decimal
import shutil
import traceback
import uuid

from flask import Blueprint
from flask import request
from backend.aws_helpers.dynamo_db_utils.dynamo_db_utils import (
    create_dynamo_item_from_obj,
)

from aws_helpers.dynamo_db_utils.trainspace_db import (
    TrainspaceData,
    createTrainspaceData,
)
from common.ai_drive import dl_tabular_drive, dl_img_drive, ml_drive
from common.constants import (
    UNZIPPED_DIR_NAME,
)
from common.utils import *
from dl.detection import detection_img_drive
from endpoints.utils import (
    send_detection_results,
    send_success,
    send_traceback_error,
    send_train_results,
)
import aiohttp

import boto3

train_bp = Blueprint("train", __name__)


@train_bp.route("/tabular-run", methods=["POST"])
def tabular_run():
    """
    API Endpoint in order to train a DL Model for tabular datasets

    Params:
     - uid: Unique User id
     - name: Name of Trainspace Data the user specifies
     - dataset_data: DatasetData instance
     - parameters_data:
          - target_col: Target col to predict/classify
          - features: Input columns to the model
          - problem_type: Classification or Regression probelm
          - criterion: Loss function to use (eg: MSELoss, CELoss, BCELoss, etc)
          - optimizer_name: What optimizer should the model use during gradient descent (eg: Adam)
          - shuffle: Should the rows be shuffled or order maintained
          - epochs: How many epochs/iterations do we train model for
          - test_size: What percentage of your dataset should be dedicated for testing the performance of the model
          - batch_size: How big should each "batch" of the dataset be. This is for training in batch during the epoch
          - layers: Architecture of Model
     - review_data: ReviewData instance

    Results:
      - 200: Training successful. Show result page
      - 400: Error in training of model. Could come from problems with the user's request or on the server side
    """
    try:
        request_data = json.loads(request.data)
        id = str(uuid.uuid4())
        tabular_data = TrainspaceData(
            trainspace_id=id,
            uid=request.environ["user"]["uid"],
            created=get_current_timestamp(),
            data_source="TABULAR",
            dataset_data=request_data["dataset_data"],
            name=request_data["name"],
            parameters_data=request_data["parameters_data"],
            review_data=request_data["review_data"],
        )

        try:
            create_dynamo_item_from_obj("trainspace", tabular_data)
            print(id)

            async def run():
                async with aiohttp.ClientSession(json_serialize=json.dumps) as session:
                    async with session.post(
                        "http://localhost:8001/api/trainspace/tabular",
                        json={
                            "target": request_data["parameters_data"]["target_col"],
                            "features": request_data["parameters_data"]["features"],
                            "problem_type": request_data["parameters_data"][
                                "problem_type"
                            ],
                            "criterion": request_data["parameters_data"]["criterion"],
                            "default": request_data["dataset_data"]["name"],
                            "optimizer_name": request_data["parameters_data"][
                                "optimizer_name"
                            ],
                            "shuffle": request_data["parameters_data"]["shuffle"],
                            "epochs": request_data["parameters_data"]["epochs"],
                            "test_size": request_data["parameters_data"]["test_size"],
                            "batch_size": request_data["parameters_data"]["batch_size"],
                            "user_arch": request_data["parameters_data"]["layers"],
                            "name": request_data["name"],
                        },
                    ) as response:
                        print(await response.json(content_type=None))

            asyncio.run(run())
            return send_success({"message": "success", "trainspace_id": id})
        except Exception:
            print(traceback.format_exc())
            return send_traceback_error()

        """ train_loss_results = dl_tabular_drive(
            user_arch, fileURL, params, json_csv_data_str, customModelName
        )
        train_loss_results["user_arch"] = user_arch
        print(train_loss_results)
        updateStatus(execution_id, "SUCCESS") """
        # return send_train_results(train_loss_results)

    except Exception:
        # updateStatus(execution_id, "ERROR")
        print(traceback.format_exc())
        return send_traceback_error()


@train_bp.route("/img-run", methods=["POST"])
def img_run():
    """
    API Endpoint to train an image model via Pytorch

    Params:
      - train_transform: Sequence of image transformations to apply to train set
      - test_transform: Sequence of image transformations to apply to test set
      - user_arch: Architecture of image DL model
      - criterion: Loss function (eg: BCELoss, CELoss, etc)
      - optimizer_name: What optimizer to use during training (eg: Adam)
      - using_default_dataset: Is the user using a default/built-in image dataset
      - epochs: How many epochs/iterations to run the model
      - batch_size: How big should each batch be within the dataset
      - shuffle: Should the data be shuffled around before training?
      - custom_model_name: User specified name of model for their convenience
      - uid: unique user id
      - execution_id: Execution Id to keep track of user's training requests

    Results:
      - 200: Image DL model trained successfully
      - 400: Error happened in model training. Could be on user side or server side
    """
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
        uid = request.environ["user"]["uid"]
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
    """
    API Endpoint for running object detection models.

    Params:
      - problem_type: What class of object detection problems the user wants to play with
      - detection_type: What object detection algorithm does the user want to play with (eg: AWS Rekognition, YOLOV3, etc)
      - transforms: Sequence of image transformations to be done before running object detection model

    Returns:
        _type_: _description_
    """
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
