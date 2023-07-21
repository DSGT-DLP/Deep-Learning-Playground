import traceback

from flask import Blueprint
from flask import request
import json

from aws_helpers.dynamo_db.trainspace_db import (
    createTrainspaceData,
    getAllUserTrainspaceData,
    getTrainspaceData,
    updateStatus,
    updateTrainspaceData,
)
from endpoints.utils import send_traceback_error, send_success, send_error

trainspace_bp = Blueprint("trainspace", __name__)


@trainspace_bp.route("/getTrainspaceData", methods=["GET"])
def getTrainspaceDataFromDb():
    """
    API Endpoint to get a "trainspace" item using a trainspace ID


    Params:
      - trainspace_id: Unique trainspace id

    Results:
      - 200: Trainspace retrieved successfully
      - 400: Error in retrieving trainspace
    """
    try:
        request_args = json.loads(request.args)
        trainspace_id = request_args["trainspace_id"]
        trainspace_data = getTrainspaceData(trainspace_id)
        return send_success({"trainspace_data": trainspace_data})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@trainspace_bp.route("/updateTrainspaceData", methods=["POST"])
def updateTrainspaceDataInDb():
    """
    API Endpoint to update a "trainspace" item using a trainspace ID


    Params:
      - trainspace_id: Unique trainspace id
      - requestData: A dictionary containing the other table attributes to be updated, not including trainspace_id

    Results:
      - 200: Trainspace updated successfully
      - 400: Error in updating trainspace
    """
    try:
        request_data = json.loads(request.data)
        trainspace_id = request_data["trainspace_id"]
        requestData = request_data["requestData"]

        success = updateTrainspaceData(trainspace_id, requestData)
        if success:
            return send_success({"message": "Trainspace updated", "success": success})
        else:
            return send_error("Trainspace not updated")
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@trainspace_bp.route("/getAllUserTrainspaceData", methods=["GET"])
def getAllUserTrainspaceDataInDb():
    """
    API Endpoint to get all "trainspace" items of a user ID


    Params:
      - user_id: Unique user ID

    Results:
      - 200: Trainspaces retrieved successfully
      - 400: Error in retrieving trainspaces
    """
    try:
        request_args = json.loads(request.args)
        user_id = request_args["user_id"]

        userTrainspaceData = getAllUserTrainspaceData(user_id)
        return send_success({"userTrainspaceData": userTrainspaceData})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@trainspace_bp.route("/updateStatus", methods=["POST"])
def updateStatusInDb():
    """
    API Endpoint to update status of a trainspace


    Params:
      - trainspace_id: Unique trainspace id
      - status: New status of the trainspace
      - entryData: The entry to be updated (if any)

    Results:
      - 200: Trainspaces retrieved successfully
      - 400: Error in retrieving trainspaces
    """
    try:
        request_data = json.loads(request.data)
        trainspace_id = request_data["trainspace_id"]
        status = request_data["status"]
        entryData = request_data["entryData"]

        success = updateStatus(trainspace_id, status, entryData)
        if success:
            return send_success({"message": "Updated status", "success": success})
        else:
            return send_error("Status not updated")
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@trainspace_bp.route("/createTrainspaceData", methods=["POST"])
def createTrainspaceDataInDb():
    """
    API Endpoint to create a "trainspace" item


    Params:
      - trainspace_data: Trainspace data object

    Results:
      - 200: Trainspaces created successfully
      - 400: Error in creating trainspace
    """
    try:
        request_data = json.loads(request.data)
        trainspace_data = request_data["trainspace_data"]

        success = createTrainspaceData(trainspace_data)
        if success:
            return send_success({"message": "Trainspace created", "success": success})
        else:
            return send_error("Trainspace not created")
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()
