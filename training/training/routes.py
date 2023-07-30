from flask import Blueprint


train = Blueprint("train", __name__)


@train.route("/tabular", methods=["POST"])
def tabular():
    return {"result": "200 Backend surface test successful"}


@train.route("/test", methods=["GET"])
def test():
    return {"result": "200 Backend surface test successful"}
