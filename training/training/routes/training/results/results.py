from datetime import date
from typing import Literal, Optional
from ninja import Router, Schema
from ninja.errors import AuthenticationError, ValidationError
import boto3, botocore.exceptions
import json

from training.routes.tabular.schemas import TabularParams
from training.core.authenticator import FirebaseAuth, Request
from training.celery_app import celery_app
from training.routes.training.results.schemas import (
    DetailedTrainResultsData,
    TrainResultsData,
)

router = Router()


@router.get(
    "/results/{trainspace_id}", auth=FirebaseAuth(), response=DetailedTrainResultsData
)
def getDetailedTrainResultsData(request: Request, trainspace_id: str):
    s3 = boto3.resource("s3")

    try:
        content_object = s3.Object("dlp-executions", f"{trainspace_id}.json")
        file_content = content_object.get()["Body"].read().decode("utf-8")
        json_content = json.loads(file_content)
        detailedTrainResultsData = DetailedTrainResultsData(**json_content)
        if request.auth["uid"] != detailedTrainResultsData.basic_info.uid:
            raise AuthenticationError("Invalid authorization")

    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            raise ValidationError("Trainspace id is invalid")
        elif e.response["Error"]["Code"] == 403:
            raise AuthenticationError("Invalid authorization")
        else:
            raise

    return 200, detailedTrainResultsData
