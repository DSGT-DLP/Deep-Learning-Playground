import traceback

from flask import Blueprint
from flask import request

from backend.aws_helpers.dynamo_db_utils.learnmod_db import (
    UserProgressDDBUtil,
    UserProgressData,
)
from backend.aws_helpers.dynamo_db_utils.trainspace_db import createTrainspaceData, getAllUserTrainspaceData
from backend.common.constants import (
    AWS_REGION,
    USERPROGRESS_TABLE_NAME,
    POINTS_PER_QUESTION,
)
from backend.common.utils import *
from backend.endpoints.utils import send_traceback_error, send_success

trainspace_bp = Blueprint("trainspace", __name__)


@trainspace_bp.route("/api/create-trainspace", methods=["POST"])
def create_trainspace():
    try:
        request_data = json.loads(request.data)
        uid = request_data["user"]["uid"]
        trainspace_id = createTrainspaceData()
        return {"trainspace_id": trainspace_id}
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@trainspace_bp.route("/api/getTrainspaceData", methods=["POST"])
def trainspace_table():
    try:
        request_data = json.loads(request.data)
        user_id = request_data["user"]["uid"]
        record = getAllUserTrainspaceData(user_id)
        return send_success({"record": record})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@trainspace_bp.route("/api/getUserProgressData", methods=["POST"])
def getUserProgressData():
    dynamoTable = UserProgressDDBUtil(USERPROGRESS_TABLE_NAME, AWS_REGION)
    user_id = json.loads(request.data)["user_id"]
    print(user_id)
    try:
        return dynamoTable.get_record(user_id).progressData
    except ValueError:
        newRecord = UserProgressData(user_id, "{}")
        dynamoTable.create_record(newRecord)
        return "{}"


@trainspace_bp.route("/api/updateUserProgressData", methods=["POST"])
def updateUserProgressData():
    requestData = json.loads(request.data)
    uid = requestData["user_id"]
    moduleID = str(requestData["moduleID"])
    sectionID = str(requestData["sectionID"])
    questionID = str(requestData["questionID"])
    dynamoTable = UserProgressDDBUtil(USERPROGRESS_TABLE_NAME, AWS_REGION)

    # get most recent user progress data
    updatedRecord = json.loads(dynamoTable.get_record(uid).progressData)

    if moduleID not in updatedRecord:
        updatedRecord[moduleID] = {
            "modulePoints": POINTS_PER_QUESTION,
            sectionID: {
                "sectionPoints": POINTS_PER_QUESTION,
                questionID: POINTS_PER_QUESTION,
            },
        }
    else:
        if sectionID not in updatedRecord[moduleID]:
            updatedRecord[moduleID][sectionID] = {
                "sectionPoints": POINTS_PER_QUESTION,
                questionID: POINTS_PER_QUESTION,
            }
            updatedRecord[moduleID]["modulePoints"] += POINTS_PER_QUESTION
        else:
            if questionID not in updatedRecord[moduleID][sectionID]:
                updatedRecord[moduleID]["modulePoints"] += POINTS_PER_QUESTION
                updatedRecord[moduleID][sectionID][questionID] = POINTS_PER_QUESTION
                updatedRecord[moduleID][sectionID][
                    "sectionPoints"
                ] += POINTS_PER_QUESTION

    updatedRecordAsString = json.dumps(updatedRecord)

    dynamoTable.update_record(uid, progressData=updatedRecordAsString)
    return '{"status": "success"}'
