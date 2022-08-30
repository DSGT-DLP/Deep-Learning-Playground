#.env file constants
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
from backend.common.constants import AWS_REGION

ENV_KEYS = ["REACT_APP_SECRET_KEY",
            "REACT_APP_CAPTCHA_SITE_KEY", "REACT_APP_FEEDBACK_EMAIL"]

FINAL_ENV_PATH = os.path.abspath(os.path.join(
    os.getcwd(), '..', 'frontend', 'playground-frontend', '.env'))

SECRET_NAME = "frontend_env"
