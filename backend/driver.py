import os
import traceback
import datetime
from werkzeug.utils import secure_filename
import shutil

from flask import Flask, request, send_from_directory
from backend.aws_helpers.s3_utils.s3_bucket_names import EXECUTION_BUCKET_NAME
from backend.aws_helpers.s3_utils.s3_client import get_presigned_url_from_exec_file, write_to_bucket
from backend.middleware import middleware
from flask_cors import CORS

from backend.common.ai_drive import dl_tabular_drive, dl_img_drive, ml_drive
from backend.common.constants import ONNX_MODEL, SAVED_MODEL_DL, UNZIPPED_DIR_NAME
from backend.common.default_datasets import get_default_dataset_header
from backend.common.email_notifier import send_email
from backend.common.utils import *
from backend.firebase_helpers.firebase import init_firebase
from backend.aws_helpers.dynamo_db_utils.learnmod_db import UserProgressDDBUtil, UserProgressData
from backend.aws_helpers.sqs_utils.sqs_client import add_to_training_queue
from backend.aws_helpers.dynamo_db_utils.execution_db import ExecutionDDBUtil, ExecutionData, createUserExecutionsData, getAllUserExecutionsData, updateStatus
from backend.common.constants import EXECUTION_TABLE_NAME, AWS_REGION, USERPROGRESS_TABLE_NAME, POINTS_PER_QUESTION
from backend.dl.detection import detection_img_drive

from backend.aws_helpers.lambda_utils.lambda_client import invoke

init_firebase()

PORT = os.getenv("PORT")
if PORT is not None:
    PORT = int(PORT)
else:
    PORT = 8000

app = Flask(
    __name__,
    static_folder=os.path.join(
        os.getcwd(), "frontend", "playground-frontend", "build"
    ),
)
CORS(app)

app.wsgi_app = middleware(app.wsgi_app)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def root(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

@app.route("/test")
def verify_backend_alive():
    return {"Status": "Backend is alive"}


@app.route("/api/tabular-run", methods=["POST"])
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
                "progress": 0
            }
        )
        train_loss_results = dl_tabular_drive(user_arch, fileURL, params,
            json_csv_data_str, customModelName)
        train_loss_results["user_arch"] = user_arch
        write_to_bucket(SAVED_MODEL_DL, EXECUTION_BUCKET_NAME, f"{execution_id}/{os.path.basename(SAVED_MODEL_DL)}")
        write_to_bucket(ONNX_MODEL, EXECUTION_BUCKET_NAME, f"{execution_id}/{os.path.basename(ONNX_MODEL)}")
        write_to_bucket(DEEP_LEARNING_RESULT_CSV_PATH, EXECUTION_BUCKET_NAME, f"{execution_id}/{os.path.basename(DEEP_LEARNING_RESULT_CSV_PATH)}")
        print(train_loss_results)
        updateStatus(execution_id, "SUCCESS")
        return send_train_results(train_loss_results)

    except Exception:
        updateStatus(execution_id, "ERROR")
        print(traceback.format_exc())
        return send_traceback_error()


@app.route("/api/ml-run", methods=["POST"])
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
            user_model =  user_model,
            problem_type = problem_type,
            target = target,
            features = features,
            default = default,
            json_csv_data_str = json_csv_data_str,
            shuffle = shuffle
            )
        train_loss_results["user_arch"] = user_model
        print(train_loss_results)
        return send_train_results(train_loss_results)

    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@app.route("/api/img-run", methods=["POST"])
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
            if (x != ".gitkeep"):
                file_rem = os.path.join(os.path.abspath(IMAGE_UPLOAD_FOLDER), x)
                if (os.path.isdir(file_rem)):
                    shutil.rmtree(file_rem)
                else:
                    os.remove(file_rem)
        if os.path.exists(UNZIPPED_DIR_NAME):
            shutil.rmtree(UNZIPPED_DIR_NAME)

@app.route("/api/object-detection", methods=["POST"])
def object_detection_run():
    IMAGE_UPLOAD_FOLDER = "./backend/image_data_uploads"
    try:
        request_data = json.loads(request.data)
        problem_type = request_data["problem_type"]
        detection_type = request_data["detection_type"]
        transforms = request_data["transforms"]
        image = detection_img_drive(IMAGE_UPLOAD_FOLDER, detection_type, problem_type, transforms)
        return send_detection_results(image)
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()
    finally:
        for x in os.listdir(IMAGE_UPLOAD_FOLDER):
            if (x != ".gitkeep"):
                file_rem = os.path.join(os.path.abspath(IMAGE_UPLOAD_FOLDER), x)
                if (os.path.isdir(file_rem)):
                    shutil.rmtree(file_rem)
                else:
                    os.remove(file_rem)
        if os.path.exists(UNZIPPED_DIR_NAME):
            shutil.rmtree(UNZIPPED_DIR_NAME)



@app.route("/api/sendEmail", methods=["POST"])
def send_email_route():
    # extract data
    request_data = json.loads(request.data)
    required_params = ["email_address", "subject", "body_text"]
    for required_param in required_params:
        if required_param not in request_data:
            return send_error("Missing parameter " + required_param)

    email_address = request_data["email_address"]
    subject = request_data["subject"]
    body_text = request_data["body_text"]
    if "attachment_array" in request_data:
        attachment_array = request_data["attachment_array"]
        if not isinstance(attachment_array, list):
            return send_error("Attachment array must be a list of filepaths")
    else:
        attachment_array = []

    # try to send email
    try:
        send_email(email_address, subject, body_text, attachment_array)
        return send_success({"message": "Sent email to " + email_address})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()

@app.route("/api/sendUserCodeEval", methods=["POST"])
def send_user_code_eval():
    try:
        request_data = json.loads(request.data)
        data = request_data["data"]
        codeSnippet = request_data["codeSnippet"]
        payload = json.dumps({
            "data": data,
            "code": codeSnippet
        })
        resJson = invoke('preprocess_data', payload)
        if (resJson['statusCode'] == 200):
            send_success({"message": "Preprocessed data", "data": resJson['data'], "columns": resJson['columns']})
        else:
            print(resJson['message'])
            send_error(resJson['message'])
        return resJson
    except Exception:
        print(traceback.format_exc())
        print("error")
        print("Last element: ", send_traceback_error()[0])
        return send_traceback_error()

@app.route("/api/getSignedUploadUrl", methods=["POST"])
def get_signed_upload_url():
    try:
        version = request.form.get("version")
        filename = request.form.get("filename")
        file = request.files.get("file")
        """ s3_client = boto3.client('s3')
        response = s3_client.generate_presigned_post("dlp-upload-bucket", "v" + str(version) + "/" + filename)
        print(requests.put(response['url'], data=response['fields'], files={'file': file.read()})) """
        s3 = boto3.client('s3')
        s3.upload_fileobj(file, "dlp-upload-bucket", "v" + str(version) + "/" + filename)
        return send_success({"message": "Upload successful"})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()

@app.route("/api/defaultDataset", methods=["POST"])
def send_columns():
    try:
        request_data = json.loads(request.data)
        default = request_data["using_default_dataset"]
        header = get_default_dataset_header(default.upper())
        header_list = header.tolist()
        return send_success({"columns": header_list})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()

@app.route("/api/getExecutionsData", methods=["POST"])
def executions_table():
    try:
        request_data = json.loads(request.data)
        user_id = request_data['user']['uid']
        record = getAllUserExecutionsData(user_id)
        return send_success({"record": record})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()

@app.route("/api/getExecutionsFilesPresignedUrls", methods=["POST"])
def executions_files():
    try:
        request_data = json.loads(request.data)
        exec_id = request_data['exec_id']
        dl_results = get_presigned_url_from_exec_file(EXECUTION_BUCKET_NAME, exec_id, "dl_results.csv")
        model_pt = get_presigned_url_from_exec_file(EXECUTION_BUCKET_NAME, exec_id, "model.pt")
        model_onnx = get_presigned_url_from_exec_file(EXECUTION_BUCKET_NAME, exec_id, "my_deep_learning_model.onnx")
        return send_success({"dl_results": dl_results, "model_pt": model_pt, "model_onnx": model_onnx})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()

@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        print(datetime.datetime.now().isoformat() + " upload has started its task")
        file = request.files['file']
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(basepath, 'image_data_uploads', secure_filename(file.filename))
        file.save(upload_path)
        file.stream.close()
        print(datetime.datetime.now().isoformat() + " upload has finished its task")
        return send_success({"message": "upload success"})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()
    
@app.route("/api/writeToQueue", methods=["POST"])
def writeToQueue() -> str:
    """
    API Endpoint to write training request to SQS queue to be serviced by 
    ECS Fargate training cluster
    
    """
    try:
        queue_data = json.loads(request.data)
        queue_send_outcome = add_to_training_queue(queue_data)
        print(f"sqs outcome: {queue_send_outcome}")
        status_code = queue_send_outcome["ResponseMetadata"]["HTTPStatusCode"]
        if (status_code != 200):
            return send_error("Your training request couldn't be added to the queue")
        else:
            createExecution(queue_data)
            return send_success({"message": "Successfully added your training request to the queue"})
    except Exception:
        return send_error("Failed to queue data")

@app.route("/api/getUserProgressData", methods=["POST"])
def getUserProgressData():
    dynamoTable = UserProgressDDBUtil(USERPROGRESS_TABLE_NAME, AWS_REGION)
    try:
        return dynamoTable.get_record(json.loads(request.data)).progressData
    except ValueError:
        newRecord = UserProgressData(json.loads(request.data), "{}")
        dynamoTable.create_record(newRecord)
        return "{}"


@app.route("/api/updateUserProgressData", methods=["POST"])
def updateUserProgressData():
    requestData = json.loads(request.data)
    uid = requestData['user']['uid']
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
                questionID: POINTS_PER_QUESTION
            }  
        }
    else:
        if sectionID not in updatedRecord[moduleID]:
            updatedRecord[moduleID][sectionID] = {
                "sectionPoints": POINTS_PER_QUESTION,
                questionID: POINTS_PER_QUESTION
            }
            updatedRecord[moduleID]["modulePoints"] += POINTS_PER_QUESTION
        else:
            if questionID not in updatedRecord[moduleID][sectionID]:
                updatedRecord[moduleID]["modulePoints"] += POINTS_PER_QUESTION
                updatedRecord[moduleID][sectionID][questionID] = POINTS_PER_QUESTION
                updatedRecord[moduleID][sectionID]["sectionPoints"] += POINTS_PER_QUESTION

    updatedRecordAsString = json.dumps(updatedRecord)
    
    dynamoTable.update_record(uid, progressData=updatedRecordAsString)
    return "{\"status\": \"success\"}"


def createExecution(entryData: dict) -> dict:
    """
    Creates an entry in the `execution-table` DynamoDB table given an `execution_id`. If does not exist, create a new entry corresponding to the given user_id.

    E.g.
    POST request to http://localhost:8000/api/createExecution with body
    {"execution_id": "fsdh", "user_id": "fweadshas"}
    will create a new entry with the given execution_id and other attributes present e.g. user_id (user_id must be present upon creating a new entry)

    @return: A JSON string of the entry created in the table
    """
    entryData = {
        "execution_id": entryData["execution_id"], 
        "user_id": entryData["user"]["uid"], 
        "name": entryData["custom_model_name"], 
        "data_source": entryData["data_source"], 
        "status": "QUEUED", 
        "timestamp": get_current_timestamp(),
        "progress": 0
    }
    createUserExecutionsData(entryData)

def send_success(results: dict):
    return (json.dumps({"success": True, **results}), 200)


def send_error(message: str):
    return (json.dumps({"success": False, "message": message}), 400)


def send_train_results(train_loss_results: dict):
    return send_success({
        "message": "Dataset trained and results outputted successfully",
        "dl_results": csv_to_json(),
        "auxiliary_outputs": train_loss_results,
    })

def send_detection_results(object_detection_results: dict):
    return send_success({
        "message": "Detection worked successfully",
        "dl_results": object_detection_results["dl_results"],
        "auxiliary_outputs": object_detection_results["auxiliary_outputs"],
    })

def send_traceback_error():
    return send_error(traceback.format_exc(limit=1))


if __name__ == "__main__":
    print("Backend starting")
    app.run(debug=True, host="0.0.0.0", port=PORT)
