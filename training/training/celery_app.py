from celery import Celery

from training import celeryconfig

celery_app = Celery("training")
celery_app.config_from_object(celeryconfig)


@celery_app.task(name="tabularTrainTask")
def tabularTrainTask(tabularParams: dict, uid: str):
    # implementation located in worker.py
    pass


@celery_app.task(name="imageTrainTask")
def imageTrainTask(imageParams: dict, uid: str):
    # implementation located in worker.py
    pass
