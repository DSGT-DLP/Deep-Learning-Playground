import os
import random
import traceback
import datetime
from werkzeug.utils import secure_filename
import shutil

from flask import Flask, request, send_from_directory
from backend.middleware import middleware
from flask_cors import CORS

from backend.common.ai_drive import dl_tabular_drive, dl_img_drive
from backend.common.constants import UNZIPPED_DIR_NAME
from backend.common.default_datasets import get_default_dataset_header
from backend.common.email_notifier import send_email
from backend.common.execution_updates import create_execution, end_training_success, end_training_failure, upload_to_start
from backend.common.utils import *
from backend.firebase_helpers.firebase import init_firebase

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
    
@app.route("/api/tabular-run", methods=["POST"])
def tabular_run():   
    try:
        request_data = json.loads(request.data)
        
        execution_id = request_data["execution_id"]
        user_arch = request_data["user_arch"]
        criterion = request_data["criterion"]
        optimizer_name = request_data["optimizer_name"]    
        problem_type = request_data["problem_type"]
        target = request_data["target"]
        features = request_data["features"]
        default = request_data["using_default_dataset"]
        test_size = request_data["test_size"]
        batch_size = request_data["batch_size"]
        epochs = request_data["epochs"]
        shuffle = request_data["shuffle"]
        csvDataStr = request_data["csv_data"]
        fileURL = request_data["file_URL"]
        custom_model_name = request_data["custom_model_name"]
        
        if execution_id:
              upload_to_start(execution_id)
        else:
            execution_id = create_execution(custom_model_name, "temp", "TABULAR", False)

        train_loss_results = dl_tabular_drive(
            execution_id,
            user_arch,
            criterion,
            optimizer_name,
            problem_type,
            fileURL,
            target,
            features,
            default,
            test_size,
            epochs,
            shuffle,
            batch_size,
            csvDataStr
        )
        end_training_success(execution_id)
        print(train_loss_results)
        
        return send_train_results(train_loss_results)

    except Exception:
        print(traceback.format_exc())
        end_training_failure(execution_id)
        return send_traceback_error()

@app.route("/api/img-run", methods=["POST"])
def img_run():
    IMAGE_UPLOAD_FOLDER = "./backend/image_data_uploads"
    try:
        request_data = json.loads(request.data)
        
        execution_id = request_data["execution_id"]
        train_transform = request_data["train_transform"]
        test_transform = request_data["test_transform"]
        user_arch = request_data["user_arch"]
        criterion = request_data["criterion"]
        optimizer_name = request_data["optimizer_name"]
        default = request_data["using_default_dataset"]
        epochs = request_data["epochs"]
        batch_size = request_data["batch_size"]
        shuffle = request_data["shuffle"]
        custom_model_name = request_data["custom_model_name"]
        
        if execution_id:
            upload_to_start(execution_id)
        else:
            execution_id = create_execution(custom_model_name, "temp", "IMAGE", False)

        train_loss_results = dl_img_drive(
            execution_id,
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

        end_training_success(execution_id)
        print("training successfully finished")
        return send_train_results(train_loss_results)
        
    except Exception:
        print(traceback.format_exc())
        end_training_failure(execution_id)
        
        return send_traceback_error()
        
    finally:
        for x in os.listdir(IMAGE_UPLOAD_FOLDER):
            if (x != ".gitkeep"):
                file_rem = os.path.join(os.path.abspath(IMAGE_UPLOAD_FOLDER) , x)
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

@app.route("/api/upload", methods=["POST"])
def upload():
    try:
        execution_id = create_execution(request.form["model_name"], "temp", request.form["data_source"], True)
        
        print(datetime.datetime.now().isoformat() + " upload has started its task")
        file = request.files["file"]
        basepath = os.path.dirname(__file__) 
        upload_path = os.path.join(basepath, "image_data_uploads", secure_filename(file.filename)) 
        file.save(upload_path)
        file.stream.close()
        print(datetime.datetime.now().isoformat() + " upload has finished its task")
        
        return send_success({"message": "upload success", "execution_id": execution_id})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()
    
@app.route("/api/getRecords", methods=["GET"])
def getRecords():
    try:
        pass
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()

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

def send_traceback_error():
    return send_error(traceback.format_exc(limit=1))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=PORT)