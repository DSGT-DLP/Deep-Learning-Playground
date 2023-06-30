import traceback

from flask import Blueprint
from flask import request
import uuid

from backend.aws_helpers.dynamo_db_utils.userprogress_db import (
    UserProgressData,
    createUserProgressData,
    getAllUserProgressData,
    updateUserProgressData,
)
from backend.aws_helpers.dynamo_db_utils.trainspace_db import (
    TrainspaceData,
    createTrainspaceData,
    getAllUserTrainspaceData,
)
from backend.common.constants import (
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
        uid = request.environ["user"]["uid"]
        trainspace_id = str(uuid.uuid4())
        trainspace_id = createTrainspaceData(TrainspaceData(trainspace_id, uid))
        return {"trainspace_id": trainspace_id}
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@trainspace_bp.route("/getTrainspaceData", methods=["GET"])
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
        user_id = request.environ["user"]["uid"]
        record = getAllUserTrainspaceData(user_id)
        return send_success({"record": record})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@trainspace_bp.route("/getUserProgressData", methods=["GET"])
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

    try:
        user_id = request.environ["user"]["uid"]
        return getAllUserProgressData(user_id)["progressData"]
    except ValueError:
        newRecord = UserProgressData(user_id, {})
        createUserProgressData(newRecord)
        return {}


@trainspace_bp.route("/updateOneUserProgressData", methods=["POST"])
def updateOneUserProgressData():
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
    try:
        requestData = json.loads(request.data)
        uid = request.environ["user_id"]
        moduleID = str(requestData["moduleID"])
        sectionID = str(requestData["sectionID"])
        questionID = str(requestData["questionID"])

        # get most recent user progress data
        updatedRecord = getAllUserProgressData(uid)["progressData"]
    except ValueError:
        print(traceback.format_exc())
        return send_traceback_error()

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

    updateUserProgressData(uid, {"progressData": updatedRecordAsString})
    return '{"status": "success"}'
