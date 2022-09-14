import os
import traceback
import eventlet
import datetime, threading
from werkzeug.utils import secure_filename
import shutil

from flask import Flask, request, copy_current_request_context, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

from backend.common.ai_drive import dl_tabular_drive, dl_img_drive, dl_audio_drive
from backend.common.constants import UNZIPPED_DIR_NAME
from backend.common.email_notifier import send_email
from backend.common.utils import *
from backend.firebase_helpers.authenticate import authenticate
from backend.firebase_helpers.firebase import init_firebase

init_firebase()

init_firebase()

app = Flask(
    __name__,
    static_folder=os.path.join(
        os.path.dirname(os.getcwd()), "frontend", "playground-frontend", "build"
    ),
)
CORS(app)
socket = SocketIO(app, cors_allowed_origins="*", ping_timeout=600, ping_interval=15)

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def root(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


@socket.on("frontendLog")
def frontend_log(log):
    app.logger.info(f'"frontend: {log}"')

@socket.on('tabular-run')
def train_and_output(request_data, socket_id):
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
    
    try:
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
            send_progress=send_progress_helper(socket_id),
        )        
        send_results(train_loss_results, socket_id)

    except Exception:
        print(traceback.format_exc())
        send_error(socket_id)

@socket.on("img-run")
def testing(request_data, socket_id):
    try: 
        print("backend started")
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
            send_progress=send_progress_helper(socket_id)
        )

        print("training successfully finished")
        send_results(train_loss_results, socket_id)
        
    except Exception as e:
        print(traceback.format_exc())
        send_error(socket_id)
        
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
            
@socket.on("audio-run")
def testing(request_data, socket_id):
    try: 
        print("backend started")
        AUDIO_UPLOAD_FOLDER = "./backend/audio_data_uploads"
        train_transform = request_data["train_transform"]
        test_transform = request_data["test_transform"]
        user_arch = request_data["user_arch"]
        criterion = request_data["criterion"]
        optimizer_name = request_data["optimizer_name"]
        default = request_data["using_default_dataset"]
        epochs = request_data["epochs"]
        batch_size = request_data["batch_size"]
        shuffle = request_data["shuffle"]
        
        train_transform = test_transform = []

        train_loss_results = dl_audio_drive(
            train_transform,
            test_transform,
            user_arch,
            criterion,
            optimizer_name, 
            default,
            epochs,
            batch_size,
            shuffle,
            AUDIO_UPLOAD_FOLDER,
            1600,
            0.9,
            send_progress=send_progress_helper(socket_id)
        )

        print("training successfully finished")
        send_results(train_loss_results, socket_id)
        
    except Exception as e:
        print(traceback.format_exc())
        send_error(socket_id)

@socket.on('sendEmail')
def send_email(request_data, socket_id):
    # extract data
    required_params = ["email_address", "subject", "body_text"]
    for required_param in required_params:
        if required_param not in request_data:
            return socket.emit('emailResult',
                {
                    "success": False,
                    "message": "Missing parameter " + required_param
                },
                to=socket_id
            )

    email_address = request_data["email_address"]
    subject = request_data["subject"]
    body_text = request_data["body_text"]
    if "attachment_array" in request_data:
        attachment_array = request_data["attachment_array"]
        if not isinstance(attachment_array, list):
            return socket.emit(
                "emailResult",
                {
                    "success": False,
                    "message": "Attachment array must be a list of filepaths",
                },
                to=socket_id
            )
    else:
        attachment_array = []

    # try to send email
    try:
        send_email(email_address, subject, body_text, attachment_array)
        return socket.emit('emailResult',
            {
                "success": True,
                "message": "Sent email to " + email_address
            },
            to=socket_id
        )
    except Exception:
        print(traceback.format_exc())
        return socket.emit('emailResult',
            {
                "success": False,
                "message": traceback.format_exc(limit=1)
            },
            to=socket_id
        )

@socket.on("updateUserSettings")
def update_user_settings(request_data):
    if not authenticate(request_data):
        return
    user = request.user

@app.route('/upload', methods=['POST'])
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
        socket.emit('uploadComplete')
        return '200'
    return '200'

def send_results(train_loss_results, socket_id):
    socket.emit('trainingResult',
        {
            "success": True,
            "message": "Dataset trained and results outputted successfully",
            "dl_results": csv_to_json(),
            "auxiliary_outputs": train_loss_results,
            "status": 200
        },
        to=socket_id
    )

def send_error(socket_id):
    socket.emit('trainingResult',
        {
            "success": False,
            "message": traceback.format_exc(limit=1),
            "status": 400
        },
        to=socket_id
    )

def send_progress_helper(socket_id):
    def send_progress(progress):
        socket.emit('trainingProgress', progress, to=socket_id)
        eventlet.greenthread.sleep(0)                 # to prevent logs from being grouped and sent together at the end of training
    return send_progress

if __name__ == "__main__":
    socket.run(app, debug=True, host="0.0.0.0", port=8000)