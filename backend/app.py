from common.constants import LOGGER_FORMAT
from flask import Blueprint, Flask, send_from_directory
from flask_cors import CORS

from common.utils import *
from firebase_helpers.firebase import init_firebase
from middleware import middleware

from endpoints.trainspace_endpoints import trainspace_bp
from endpoints.aws_endpoints import aws_bp
from endpoints.dataset_endpoints import dataset_bp
from endpoints.s3_edpoints import s3_bp
from endpoints.test_endpoints import test_bp
from endpoints.train_endpoints import train_bp
from dlp_logging import logger


init_firebase()

PORT = os.getenv("PORT")
if PORT is not None:
    PORT = int(PORT)
else:
    PORT = 8000

app = Flask(
    __name__,
    static_folder=os.path.join(os.getcwd(), "frontend", "build"),
)
CORS(app)

app.wsgi_app = middleware(
    app.wsgi_app, exempt_paths=["/api/test", "/api/test/", "/", "/api/apidocs"]
)

app_bp = Blueprint("api", __name__)

app_bp.register_blueprint(trainspace_bp, url_prefix="/trainspace")
app_bp.register_blueprint(aws_bp, url_prefix="/aws")
app_bp.register_blueprint(dataset_bp, url_prefix="/dataset")
app_bp.register_blueprint(s3_bp, url_prefix="/s3")
app_bp.register_blueprint(test_bp, url_prefix="/test")
app_bp.register_blueprint(train_bp, url_prefix="/train")

app.register_blueprint(app_bp, url_prefix="/api")


@app.route("/", methods=["GET"])
def root(path):
    if path != "" and os.path.exists(app.static_folder + "/" + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    logger.info("Backend starting")
    logger.debug("Debug mode enabled")
    app.run(debug=True, host="0.0.0.0", port=PORT)
