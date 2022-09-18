import os
import traceback
import datetime
from werkzeug.utils import secure_filename
import shutil

from flask import Flask, request, copy_current_request_context, send_from_directory
from flask_cors import CORS

from backend.common.ai_drive import dl_tabular_drive, dl_img_drive
from backend.common.constants import UNZIPPED_DIR_NAME
from backend.common.default_datasets import get_default_dataset_header
from backend.common.email_notifier import send_email
from backend.common.utils import *
from backend.firebase_helpers.authenticate import authenticate
from backend.firebase_helpers.firebase import init_firebase

init_firebase()

app = Flask(
    __name__,
    static_folder=os.path.join(
        os.path.dirname(os.getcwd()), "frontend", "playground-frontend", "build"
    ),
)
CORS(app)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def root(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")
    
@app.route("/tabular-run", methods=["POST"])
def tabular_run():    
    try:
        request_data = json.loads(request.data)
        
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
    
        train_loss_results = dl_tabular_drive(
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
            csvDataStr,
            send_progress=lambda x: print(x),
        )
        print(train_loss_results)
        return send_train_results(train_loss_results)

    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()

@app.route("/img-run", methods=["POST"])
def img_run():
    try: 
        print("backend started")
        request_data = json.loads(request.data)
        
        IMAGE_UPLOAD_FOLDER = "./backend/image_data_uploads"
        train_transform = request_data["train_transform"]
        test_transform = request_data["test_transform"]
        user_arch = request_data["user_arch"]
        criterion = request_data["criterion"]
        optimizer_name = request_data["optimizer_name"]
        default = request_data["using_default_dataset"]
        epochs = request_data["epochs"]
        batch_size = request_data["batch_size"]
        shuffle = request_data["shuffle"]

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
            send_progress=lambda x: print(x)
        )

        print("training successfully finished")
        return send_train_results(train_loss_results)
        
    except Exception as e:
        print(traceback.format_exc())
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

@app.route("/sendEmail", methods=["POST"])
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

@app.route("/defaultDataset", methods=["POST"])
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
    
@app.route("/updateUserSettings", methods=["POST"])
def update_user_settings():
    request_data = json.loads(request.data)
    if not authenticate(request_data):
        return send_success({"message": ""})
    user = request.user

@app.route("/upload", methods=["POST"])
def upload():
    @copy_current_request_context
    def save_file(closeAfterWrite):
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " dropzone is working")
        f = request.files['file']
        basepath = os.path.dirname(__file__) 
        upload_path = os.path.join(basepath, 'image_data_uploads',secure_filename(f.filename)) 
        f.save(upload_path)
        closeAfterWrite()
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " dropzone has finished its task")
    def passExit():
        pass
    if request.method == 'POST':
        f= request.files['file']
        normalExit = f.stream.close
        f.stream.close = passExit
        save_file(normalExit)
        # socket.emit('uploadComplete')
        return '200'
    return '200'

def send_success(results):
    return (json.dumps({"success": True, **results}), 200)

def send_error(message):
    return (json.dumps({"success": False, "message": message}), 400)

def send_train_results(train_loss_results):
    return send_success({
        "message": "Dataset trained and results outputted successfully",
        "dl_results": csv_to_json(),
        "auxiliary_outputs": train_loss_results,
    })

def send_traceback_error():
    return send_error(traceback.format_exc(limit=1))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)