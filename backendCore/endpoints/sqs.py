from flask import Blueprint
from flask import request
import json
from backendCore.aws_helpers.dynamo_db_utils.trainspace_db import TrainspaceData, createTrainspaceData
from backendCore.aws_helpers.sqs_utils import add_to_queue, add_to_training_queue
import uuid

from backendCore.endpoints.utils import send_success, send_traceback_error, send_error

sqs_bp = Blueprint("sqs", __name__)


@sqs_bp.route("/writeToQueue", methods=["POST"])
def writeToQueue() -> str:
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
        queue_data = json.loads(request.data)
        queue_send_outcome = add_to_training_queue(queue_data)
        print(f"sqs outcome: {queue_send_outcome}")
        status_code = queue_send_outcome["ResponseMetadata"]["HTTPStatusCode"]
        if status_code != 200:
            return send_error("Your training request couldn't be added to the queue")
        else:
            request_data = json.loads(request.data)
            uid = request_data["user"]["uid"]
            trainspace_id = str(uuid.uuid4())
            create_success = createTrainspaceData(TrainspaceData(trainspace_id, uid))
            
            if create_success:
                return send_success(
                    {"message": "Successfully added your training request to the queue", "trainspace_id": trainspace_id}
                )
            else:
                return send_error("Data queued but failed to create trainspace")
    
    except Exception:
        return send_error("Failed to queue data")
