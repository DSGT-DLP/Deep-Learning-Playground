import traceback
from pathlib import Path

import boto3
from flask import Blueprint
from flask import request
from werkzeug.utils import secure_filename

from backend.aws_helpers.s3_utils.s3_bucket_names import EXECUTION_BUCKET_NAME
from backend.aws_helpers.s3_utils.s3_client import (
    get_presigned_upload_post_from_user_dataset_file,
    get_presigned_url_from_exec_file,
    get_user_dataset_file_objects,
)
from backend.common.utils import *
from backend.endpoints.utils import send_success, send_traceback_error

s3_bp = Blueprint("s3", __name__)


@s3_bp.route("/getSignedUploadUrl", methods=["POST"])
def get_signed_upload_url():
    """
    API Endpoint uploads a user's dataset file to S3 bucket for easy retrieval later on

    Params:
      - version:
      - filename:

    Results:
      - 200: File uploaded successfully to S3
      - 400: Something went wrong in uploading user's dataset file to S3. There could be a problem with the file in general or an issue
      on the server side
    """
    try:
        version = request.form.get("version")
        filename = request.form.get("filename")
        file = request.files.get("file")
        """ s3_client = boto3.client('s3')
        response = s3_client.generate_presigned_post("dlp-upload-bucket", "v" + str(version) + "/" + filename)
        print(requests.put(response['url'], data=response['fields'], files={'file': file.read()})) """
        s3 = boto3.client("s3")
        s3.upload_fileobj(
            file, "dlp-upload-bucket", "v" + str(version) + "/" + filename
        )
        return send_success({"message": "Upload successful"})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@s3_bp.route("/getExecutionsFilesPresignedUrls", methods=["POST"])
def executions_files():
    """
    API Endpoint to use S3 Presigned URLs to retrieve result files from S3 given an execution id

    Most training requests will output 3 file types:
      * dl_results.csv: Performance of user's model over each epoch
      * model.pt: File storing pytorch weights of the DL model
      * my_deep_learning_model.onnx: A convenient file in Pytorch that lets users visualize the architecture of their model

    Params:
      - exec_id: Execution Id

    Results:
      - 200: Successfully retrieved result files
      - 400: Something went wrong in retrieving the training result files. Maybe execution id was specified incorrectly or
      there was a problem with updating the execution db in Dynamo DB
    """
    try:
        request_data = json.loads(request.data)
        exec_id = request_data["exec_id"]
        dl_results = get_presigned_url_from_exec_file(
            EXECUTION_BUCKET_NAME, exec_id, "dl_results.csv"
        )
        model_pt = get_presigned_url_from_exec_file(
            EXECUTION_BUCKET_NAME, exec_id, "model.pt"
        )
        model_onnx = get_presigned_url_from_exec_file(
            EXECUTION_BUCKET_NAME, exec_id, "my_deep_learning_model.onnx"
        )
        return send_success(
            {"dl_results": dl_results, "model_pt": model_pt, "model_onnx": model_onnx}
        )
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@s3_bp.route("/getUserDatasetFileUploadPresignedPostObj", methods=["POST"])
def getUserDatasetFileUploadPresignedPostObj():
    """
    ADD description here

    Returns:
        _type_: _description_
    """
    try:
        request_data = json.loads(request.data)
        post_obj = get_presigned_upload_post_from_user_dataset_file(
            request.environ["user"]["uid"],
            request_data["data_source"],
            request_data["name"],
        )
        return send_success(
            {"message": "File upload success", "presigned_post_obj": post_obj}
        )
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@s3_bp.route("/getUserDatasetFilesData", methods=["POST"])
def getUserDatasetFilesData():
    """
    API Endpoint to retrieve all user dataset files uploaded in S3

    Params:
      - uid: unique user id
      - data_source: What type of training was the user running (eg: TABULAR, PRETRAINED, OBJECT_DETECTION, IMAGE, etc)

    Results:
      - 200: Retrieved user dataset files from S3
      - 400: Error in retrieving user dataset files from S3
    """
    try:
        request_data = json.loads(request.data)
        file_objects = get_user_dataset_file_objects(
            request.environ["user"]["uid"], request_data["data_source"]
        )
        data = list(
            map(
                lambda f: {
                    "name": Path(f["Key"]).name,
                    "type": Path(f["Key"]).suffix,
                    "last_modified": f["LastModified"].isoformat(),
                    "size": f["Size"],
                },
                file_objects,
            )
        )
        return send_success(
            {"message": "Get dataset files data success", "data": json.dumps(data)}
        )
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@s3_bp.route("/upload", methods=["POST"])
def upload():
    try:
        print(datetime.datetime.now().isoformat() + " upload has started its task")
        file = request.files["file"]
        basepath = os.path.dirname(__file__)
        upload_path = os.path.join(
            basepath, "image_data_uploads", secure_filename(file.filename)
        )
        file.save(upload_path)
        file.stream.close()
        print(datetime.datetime.now().isoformat() + " upload has finished its task")
        return send_success({"message": "upload success"})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()
