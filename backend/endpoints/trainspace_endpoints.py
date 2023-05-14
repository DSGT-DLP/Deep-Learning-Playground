import traceback

from flask import Blueprint
from flask import request

from backend.aws_helpers.dynamo_db_utils.learnmod_db import (
    UserProgressDDBUtil,
    UserProgressData,
)
from backend.aws_helpers.dynamo_db_utils.trainspace_db import (
    createTrainspaceData,
    getAllUserTrainspaceData,
)
from backend.common.constants import (
    AWS_REGION,
    USERPROGRESS_TABLE_NAME,
    POINTS_PER_QUESTION,
)
from backend.common.utils import *
from backend.endpoints.utils import send_traceback_error, send_success

trainspace_bp = Blueprint("trainspace", __name__)


@trainspace_bp.route("/create-trainspace", methods=["POST"])
def create_trainspace():
    """
    API Endpoint to create a "trainspace". Trainspace is a new concept/data structure
    we introduce to track user's training requests. Concept similar to execution_id.


    Params:
      - uid: Unique User id

    Results:
      - 200: Trainspace created successfully
      - 400: Error in creating trainspace
    """
    try:
        request_data = json.loads(request.data)
        uid = request_data["user"]["uid"]
        trainspace_id = createTrainspaceData()
        return {"trainspace_id": trainspace_id}
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@trainspace_bp.route("/getTrainspaceData", methods=["POST"])
def trainspace_table():
    """
    API Endpoint to identify all trainspaces for a given user id

    This endpoint will go into the trainspace Dynamo DB and query by user id all trainspace data objects

    Params:
      - uid: Unique User id

    Results:
      - 200: Able to query and retrieve trainspace objects belonging to a user
      - 400: Error in querying trainspace data for a given uid. Could be on the client side or server side
    """
    try:
        request_data = json.loads(request.data)
        user_id = request_data["user"]["uid"]
        record = getAllUserTrainspaceData(user_id)
        return send_success({"record": record})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@trainspace_bp.route("/getUserProgressData", methods=["POST"])
def getUserProgressData():
    """
    Utility function to get user progress data for the Learning Modules feature
    of DLP.

    Params:
      - uid: Unique User id

    Results:
      - 200: Able to query and load user progress data for a given user that visits the Learning Modules surface on DLP
      - 400: Error in retrieving this data
    """
    dynamoTable = UserProgressDDBUtil(USERPROGRESS_TABLE_NAME, AWS_REGION)
    user_id = json.loads(request.data)["user_id"]
    print(user_id)
    try:
        return dynamoTable.get_record(user_id).progressData
    except ValueError:
        newRecord = UserProgressData(user_id, "{}")
        dynamoTable.create_record(newRecord)
        return "{}"


@trainspace_bp.route("/updateUserProgressData", methods=["POST"])
def updateUserProgressData():
    """
    API Endpoint to update user progress data as the user progresses through the Learning Modules feature. We can identify
    here if a user gets a question correct or not and update that progress within Dynamo Db

    Params:
      - user_id: Unique User id
      - moduleId: What module did the user interact with
      - sectionId: What section within the module did the user interact with
      - questionId: What question did the user interact with

    Results:
      - 200: Dynamo DB update successful
      - 400: Something went wrong in updating the user progress in learning modules
    """
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
