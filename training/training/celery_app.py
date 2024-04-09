from celery import Celery

from training import celeryconfig

celery_app = Celery("training")
celery_app.config_from_object(celeryconfig)


@celery_app.task(name="tabularTrainTask")
def tabularTrainTask(tabularParams: dict, trainspaceId: str, uid: str):
    pass


@celery_app.task(name="imageTrainTask")
def imageTrainTask(imageParams: dict, trainspaceId: str, uid: str):
    pass
