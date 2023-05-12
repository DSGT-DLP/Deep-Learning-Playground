import traceback

from flask import Blueprint
from flask import request

from backend.aws_helpers.s3_utils.s3_client import (
    get_column_names,
)
from backend.common.default_datasets import get_default_dataset_header
from backend.common.utils import *
from backend.endpoints.utils import send_success, send_traceback_error

dataset_bp = Blueprint("dataset", __name__)


@dataset_bp.route("/defaultDataset", methods=["POST"])
def send_columns():
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
    try:
        request_data = json.loads(request.data)
        columns = get_column_names(
            request_data["user"]["uid"],
            request_data["data_source"],
            request_data["name"],
        )
        return send_success(
            {"message": "Get columns success", "columns": json.dumps(columns)}
        )
    except Exception:
        print(traceback.format_exc())
        return send_traceback_error()
