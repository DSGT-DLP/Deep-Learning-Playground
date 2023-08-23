from django.http import HttpRequest
import firebase_admin
from ninja import Router, Schema
from ninja.security import HttpBearer
from training.core.dataset import SklearnDatasetCreator
from training.routes.datasets.default.schemas import DefaultDatasetResponse
from training.routes.schemas import NotFoundError

router = Router()

class FirebaseAuth(HttpBearer):
    def authenticate(self, request, token):
        if token is None or not token:
            return
        
        try:
            authorization = token[7:]
            firebase_admin.auth.verify_id_token(authorization)
        except Exception as e:
            print(e)
            return None
        return token

@router.get(
    "{name}/columns",
    response={200: DefaultDatasetResponse, 404: NotFoundError},
    auth=FirebaseAuth()
)
def defaultDatasets(request: HttpRequest, name: str):
    if not name in SklearnDatasetCreator.DEFAULT_DATASETS:
        return 404, {"message": "Dataset not found"}
    dataset = SklearnDatasetCreator.getDefaultDataset(name)
    return 200, { "data" : dataset.columns.tolist(), "message" : "Success", "token" : request.auth }
