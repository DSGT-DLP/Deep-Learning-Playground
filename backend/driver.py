from flask import Blueprint, Flask, send_from_directory
from flask_cors import CORS

from backend.common.utils import *
from backend.firebase_helpers.firebase import init_firebase
from backend.middleware import middleware

from backend.endpoints.trainspace_endpoints import trainspace_bp
from backend.endpoints.aws_endpoints import aws_bp
from backend.endpoints.dataset_endpoints import dataset_bp
from backend.endpoints.s3_edpoints import s3_bp
from backend.endpoints.test_endpoints import test_bp
from backend.endpoints.train_endpoints import train_bp

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
    app.wsgi_app, exempt_paths=["/api/test/", "/", "/api/apidocs"]
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
    print("Backend starting")
    app.run(debug=True, host="0.0.0.0", port=PORT)
