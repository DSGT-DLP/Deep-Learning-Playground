import pandas as pd
import traceback
import os
from flask import Flask, json, request, jsonify, make_response

from backend.common.utils import *
from backend.common.constants import CSV_FILE_NAME, ONNX_MODEL
from backend.common.dataset import read_local_csv_file, read_dataset
from backend.common.optimizer import get_optimizer
from backend.dl.dl_model_parser import parse_deep_user_architecture, get_object
from backend.dl.dl_trainer import train_deep_model, get_deep_predictions
from backend.ml.ml_trainer import train_classical_ml_model
from backend.dl.dl_model import DLModel
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from backend.common.default_datasets import get_default_dataset
from flask_cors import CORS
from backend.common.email_notifier import send_email
from flask import send_from_directory
import boto3
import base64
from botocore.exceptions import ClientError
import pyrebase
from functools import wraps
import firebase_admin
from firebase_admin import credentials, auth

app = Flask(
    __name__,
    static_folder=os.path.join(
        os.path.dirname(os.getcwd()), "frontend", "playground-frontend", "build"
    ),
)
CORS(app)


def ml_drive(
    user_model,
    problem_type,
    target=None,
    features=None,
    default=False,
    test_size=0.2,
    shuffle=True,
):
    """
    Driver function/endpoint into backend for training a classical ML model (eg: SVC, SVR, DecisionTree, Naive Bayes, etc)

    Args:
        user_model (str): What ML model and parameters does the user want
        problem_type (str): "classification" or "regression" problem
        target (str, optional): name of target column. Defaults to None.
        features (list, optional): list of columns in dataframe for the feature based on user selection. Defaults to None.
        default (bool, optional): use the iris dataset for default classifiction or california housing for default regression. Defaults to False.
        test_size (float, optional): size of test set in train/test split (percentage). Defaults to 0.2.
        shuffle (bool, optional): should the dataset be shuffled prior to train/test split
    """
    try:
        if default and problem_type.upper() == "CLASSIFICATION":
            dataset = load_iris()
            X, y = get_default_dataset(dataset)
            print(y.head())
        elif default and problem_type.upper() == "REGRESSION":
            # If the user specifies no dataset, use california housing as default regression
            dataset = fetch_california_housing()
            X, y = get_default_dataset(dataset)
        else:
            input_df = pd.read_csv(CSV_FILE_NAME)
            y = input_df[target]
            X = input_df[features]

        if shuffle and problem_type.upper() == "CLASSIFICATION":
            # using stratify only for classification problems to ensure correct AUC calculation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0, shuffle=True, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0, shuffle=shuffle
            )
        model = get_object(user_model)
        train_classical_ml_model(
            model, X_train, X_test, y_train, y_test, problem_type=problem_type
        )
    except Exception as e:
        raise e


def dl_drive(
    user_arch,
    criterion,
    optimizer_name,
    problem_type,
    target=None,
    features=None,
    default=None,
    test_size=0.2,
    epochs=5,
    shuffle=True,
    json_csv_data_str="",
    batch_size=20,
):
    """
    Driver function/entrypoint into backend for deep learning model. Onnx file is generated containing model architecture for user to visualize in netron.app
    Args:
        user_arch (list): list that contains user defined deep learning architecture
        criterion (str): What loss function to use
        optimizer (str): What optimizer does the user wants to use (Adam or SGD for now, but more support in later iterations)
        problem type (str): "classification" or "regression" problem
        target (str): name of target column
        features (list): list of columns in dataframe for the feature based on user selection
        default (str, optional): the default dataset chosen by the user. Defaults to None.
        test_size (float, optional): size of test set in train/test split (percentage). Defaults to 0.2.
        epochs (int, optional): number of epochs/rounds to run model on
        shuffle (bool, optional): should the dataset be shuffled prior to train/test split
    :return: a dictionary containing the epochs, train and test accuracy and loss results, each in a list

    NOTE:
         CSV_FILE_NAME is the data csv file for the torch model. Assumed that you have one dataset file
    """
    try:
        if default and problem_type.upper() == "CLASSIFICATION":
            X, y = get_default_dataset(default.upper())
            print(y.head())
        elif default and problem_type.upper() == "REGRESSION":
            X, y = get_default_dataset(default.upper())
        else:
            if json_csv_data_str:
                input_df = pd.read_json(json_csv_data_str, orient="records")

            y = input_df[target]
            X = input_df[features]

        if (len(y) * test_size < batch_size or len(y) * (1 - test_size) < batch_size):
            raise ValueError("reduce batch size, not enough values in dataframe")

        if problem_type.upper() == "CLASSIFICATION":
            # label encode the categorical values to numbers
            y = y.astype("category")
            y = y.cat.codes
            print(y.head())

        # Convert to tensor
        if shuffle and problem_type.upper() == "CLASSIFICATION":
            # using stratify only for classification problems to ensure correct AUC calculation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0, shuffle=True, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0, shuffle=shuffle
            )

        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_tensors(
            X_train, X_test, y_train, y_test
        )
        # Build the Deep Learning model that the user wants
        model = DLModel(parse_deep_user_architecture(user_arch))
        print(f"model: {model}")
        optimizer = get_optimizer(
            model, optimizer_name=optimizer_name, learning_rate=0.05
        )
        # criterion = LossFunctions.get_loss_obj(LossFunctions[criterion])
        print(f"loss criterion: {criterion}")
        train_loader, test_loader = get_dataloaders(
            X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size=batch_size
        )
        train_loss_results = train_deep_model(
            model, train_loader, test_loader, optimizer, criterion, epochs, problem_type
        )
        pred, ground_truth = get_deep_predictions(model, test_loader)
        torch.onnx.export(model, X_train_tensor, ONNX_MODEL)

        return train_loss_results

    except Exception as e:
        raise e


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def root(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


@app.route("/run", methods=["POST"])
def train_and_output():
    request_data = json.loads(request.data)

    user_arch = request_data["user_arch"]
    criterion = request_data["criterion"]
    optimizer_name = request_data["optimizer_name"]
    problem_type = request_data["problem_type"]
    target = request_data["target"]
    features = request_data["features"]
    default = request_data["default"]
    test_size = request_data["test_size"]
    batch_size = request_data["batch_size"]
    epochs = request_data["epochs"]
    shuffle = request_data["shuffle"]
    csvDataStr = request_data["csvData"]
    fileURL = request_data["fileURL"]
    email = request_data["email"]
    if request.method == "POST":
        try:
            if not default:
                if fileURL:
                    read_dataset(fileURL)
                elif csvDataStr:
                    pass
                else:
                    raise ValueError("Need a file input")
                    return

            train_loss_results = dl_drive(
                user_arch=user_arch,
                criterion=criterion,
                optimizer_name=optimizer_name,
                problem_type=problem_type,
                target=target,
                features=features,
                default=default,
                test_size=test_size,
                epochs=epochs,
                shuffle=shuffle,
                json_csv_data_str=csvDataStr,
                batch_size=batch_size,
            )
            return (
                jsonify(
                    {
                        "success": True,
                        "message": "Dataset trained and results outputted successfully",
                        "dl_results": csv_to_json(),
                        "auxiliary_outputs": train_loss_results
                    }
                ),
                200,
            )

        except Exception:
            print(traceback.format_exc())
            return (
                jsonify({"success": False, "message": traceback.format_exc(limit=1)}),
                400,
            )

    return jsonify({"success": False}), 500


@app.route("/sendemail", methods=["POST"])
def send_email_route():
    request_data = json.loads(request.data)

    # extract data
    required_params = ["email_address", "subject", "body_text"]
    for required_param in required_params:
        if required_param not in request_data:
            return jsonify(
                {"success": False, "message": "Missing parameter " + required_param}
            )

    email_address = request_data["email_address"]
    subject = request_data["subject"]
    body_text = request_data["body_text"]
    if "attachment_array" in request_data:
        attachment_array = request_data["attachment_array"]
        if not isinstance(attachment_array, list):
            return jsonify(
                {
                    "success": False,
                    "message": "Attachment array must be a list of filepaths",
                }
            )
    else:
        attachment_array = []

    # try to send email
    try:
        send_email(email_address, subject, body_text, attachment_array)
        return jsonify({"success": True, "message": "Sent email to " + email_address})
    except Exception:
        print(traceback.format_exc())
        return jsonify({"success": False}), 500

def get_secret():

    secret_name = "DLP/Firebase"
    region_name = "us-west-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return secret
        else:
            decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])
            return decoded_binary_secret
            
    # Your code goes here. 

def get_admin_sdk():

    secret_name = "DLP/Firebase/Admin_SDK"
    region_name = "us-west-2"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS key.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            return secret
        else:
            decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])
            return decoded_binary_secret
            
    # Your code goes here. 


config = json.loads(get_secret())
config["databaseURL"] = ""
admin_sdk_config = json.loads(get_admin_sdk())
cred = credentials.Certificate(admin_sdk_config)
firebase = firebase_admin.initialize_app(cred)
pb = pyrebase.initialize_app(config)

#Api route to sign up a new user
@app.route('/signup')
def signup():
    email = request.form.get('email')
    password = request.form.get('password')
    if email is None or password is None:
        return {'message': 'Error missing email or password'},400
    try:
        user = auth.create_user(
               email=email,
               password=password
        )
        jwt = user['idToken']
        response = make_response({'msg': 'Successfully created user!'})
        response.status = 200
        response.headers['Access-Control-Allow-Credentials'] = True
        response.set_cookie('access_token', value=jwt,httponly=True)

        return response
    except:
        return {'message': 'Error creating user'},400

#Api route to login a valid user
@app.route('/login')
def login():
    email = request.form.get('email')
    password = request.form.get('password')
    try:
        user = pb.auth().sign_in_with_email_and_password(email, password)
        jwt = user['idToken']

        
        response = make_response({'msg': 'successfully logged in!'})
        response.status = 200
        response.headers['Access-Control-Allow-Credentials'] = True
        response.set_cookie('access_token', value=jwt,httponly=True)

        return response
    except:
       return {'message': 'There was an error logging in'},400

def check_token(f):
    @wraps(f)
    def wrap(*args,**kwargs):
        if not request.cookies.get('access_token'):
            return {'message': 'No token provided'},400
        try:
            user = auth.verify_id_token(request.cookies.get('access_token'))
            request.user = user
        except:
            return {'message':'Invalid token provided.'},400
        return f(*args, **kwargs)
    return wrap

@app.route('/checklogin')
@check_token
def userinfo():
    return {'message':'Authorized'}

@app.route('/logout')
@check_token
def logout():
    response = make_response({'msg': 'successfully logged out!'})
    response.headers['Access-Control-Allow-Credentials'] = True
    response.set_cookie('access_token', '', expires=0)
    response.status = 200

    return response








if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
