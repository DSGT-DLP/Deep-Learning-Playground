import os
from flask import Blueprint, Flask, send_from_directory, redirect
from flask_cors import CORS
from endpoints.sqs import sqs_bp
from endpoints.trainspace import trainspace_bp

PORT = os.getenv("PORT")
if PORT is not None:
    PORT = int(PORT)
else:
    PORT = 8001

app = Flask(
    __name__,
    static_folder=os.path.join(os.getcwd(), "frontend", "build"),
)
CORS(app)

app_bp = Blueprint("api", __name__)

app_bp.register_blueprint(sqs_bp, url_prefix="/sqs")
app_bp.register_blueprint(trainspace_bp, url_prefix="/trainspace")

app.register_blueprint(app_bp, url_prefix="/api")


@app.route("/test", methods=["GET", "PORT"])
def test():
    return {"result": "200 Backend surface test successful"}


@app.route("/")
def root():
    return "Backend surface running"


if __name__ == "__main__":
    print("Backend surface starting")
    app.run(debug=True, host="0.0.0.0", port=PORT)
