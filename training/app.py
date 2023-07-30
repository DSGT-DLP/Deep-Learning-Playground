from flask import Flask
from training.routes import train

app = Flask(__name__)
app.register_blueprint(train, url_prefix="/api/train")
