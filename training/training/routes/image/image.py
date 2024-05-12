from ninja import Router
from training.routes.image.schemas import ImageParams
from training.core.authenticator import FirebaseAuth, Request
from training.celery_app import celery_app

import uuid

router = Router()


@router.post("", auth=FirebaseAuth())
def imageTrain(request: Request, imageParams: ImageParams):
    trainspaceId = str(uuid.uuid4())
    task = celery_app.send_task(
        "imageTrainTask", [imageParams.dict(), request.auth["uid"]]
    )

    return 200, {"trainspace_id": imageParams.trainspace_id}
