import traceback

from flask import Blueprint
from flask import request

from aws_helpers.s3_utils.s3_client import (
    get_column_names,
)
from common.default_datasets import get_default_dataset_header
from common.utils import *
from endpoints.utils import send_success, send_traceback_error

dataset_bp = Blueprint("dataset", __name__)


@dataset_bp.route("/defaultDataset", methods=["POST"])
def send_columns():
    """
    API Endpint to send columns of a user selected default dataset (eg: IRIS, California Housing, Wine, etc)

    Params:
      - using_default_dataset: boolean indicating if a user selected a default dataset

    Results:
      - 200: Columns of default dataset retrieved successfully
      - 400: Something went wrong in retrieving columns of user selected default dataset

    Returns:
        _type_: _description_
    """
    try:
        request_data = json.loads(request.data)
        default = request_data["using_default_dataset"]
        header = get_default_dataset_header(default.upper())
        header_list = header.tolist()
        return send_success({"columns": json.dumps(header_list)})
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()


@dataset_bp.route("/getColumnsFromDatasetFile", methods=["POST"])
def getColumnsFromDatasetFile():
    """
    API Endpoint to retrieve columns from a user uploaded dataset file (eg: column names for a CSV file)


    Methodology: Given the dataset file, go to S3 and run a basic SQL query to get the column names

    Params:
      - uid: user id
      - data_source: What type of training was the user running (eg: TABULAR, PRETRAINED, OBJECT_DETECTION, IMAGE, etc)
      - name: Name of dataset file

    Results:
      - 200: Columns retrieved successfully from S3
      - 400: Something went wrong in retrieving the columns. Maybe on the client side or server side
    """
    try:
        request_data = json.loads(request.data)
        columns = get_column_names(
            request.environ["user"]["uid"],
            request_data["data_source"],
            request_data["name"],
        )
        return send_success(
            {"message": "Get columns success", "columns": json.dumps(columns)}
        )
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()
