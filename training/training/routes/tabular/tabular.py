from typing import Literal, Optional
from ninja import Router, Schema

from training.routes.tabular.schemas import TabularParams
from training.core.authenticator import FirebaseAuth, Request
from training.celery_app import celery_app
import uuid

router = Router()


@router.post("", auth=FirebaseAuth())
def tabularTrain(request: Request, tabularParams: TabularParams):
    trainspaceId = str(uuid.uuid4())
    task = celery_app.send_task(
        "tabularTrainTask", [tabularParams.dict(), trainspaceId, request.auth["uid"]]
    )

    return 200, {"trainspace_id": trainspaceId}
