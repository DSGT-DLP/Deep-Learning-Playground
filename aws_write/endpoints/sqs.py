from typing import Literal, Tuple, Union
from flask import Blueprint
from flask import request
import json
from aws_helpers.dynamo_db.trainspace_db import (
    TrainspaceData,
    createTrainspaceData,
)
from aws_helpers.sqs.sqs_utils import (
    add_to_queue,
    add_to_training_queue,
)
import uuid

from endpoints.utils import send_success, send_traceback_error, send_error

sqs_bp = Blueprint("sqs", __name__)


@sqs_bp.route("/writeToQueue", methods=["POST"])
def writeToQueue() -> Tuple[str, Literal[200, 400]]:
    """
    API Endpoint to write training request to SQS queue to be serviced by
    ECS Fargate training cluster

    Params:
      - JSON that contains information about the user's individual training request

    Results:
      - 200: Training request added to SQS Queue successfully
      - 400: Something went wrong in adding user's training request to the queue

    """
    try:
        # add to SQS queue
        request_data = json.loads(request.data)
        trainspace_id = str(uuid.uuid4())
        request_data["trainspace_id"] = trainspace_id
        queue_send_outcome = add_to_training_queue(request_data)
        print(f"sqs outcome: {queue_send_outcome}")
        status_code = queue_send_outcome["ResponseMetadata"]["HTTPStatusCode"]

        # exit if queue send failed
        if status_code != 200:
            return send_error("Your training request couldn't be added to the queue")

        # add to DynamoDB
        uid = request_data["user"]["uid"]
        create_success = createTrainspaceData(TrainspaceData(trainspace_id, uid))

        if create_success:
            return send_success(
                {
                    "message": "Successfully added your training request to the queue",
                    "trainspace_id": trainspace_id,
                }
            )
        else:
            return send_error("Data queued but failed to create trainspace")

    except Exception:
        return send_error("Failed to queue data")
