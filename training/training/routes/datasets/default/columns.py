from django.http import HttpRequest
from ninja import Router, Schema
from training.core.dataset import SklearnDatasetCreator
from training.routes.datasets.default.schemas import DefaultDatasetResponse
from training.routes.schemas import NotFoundError

router = Router()


@router.get(
    "{name}/columns",
    response={200: DefaultDatasetResponse, 404: NotFoundError},
)
def defaultDatasets(request: HttpRequest, name: str):
    if not name in SklearnDatasetCreator.DEFAULT_DATASETS:
        return 404, {"message": "Dataset not found"}
    dataset = SklearnDatasetCreator.getDefaultDataset(name)
    return 200, {"data": dataset.columns.tolist(), "message": "Success"}
