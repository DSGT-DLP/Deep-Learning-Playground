#.env file constants
import os
import sys

ENV_KEYS = ["REACT_APP_SECRET_KEY",
            "REACT_APP_CAPTCHA_SITE_KEY", "REACT_APP_FEEDBACK_EMAIL"]

FINAL_REACT_ENV_PATH = os.path.abspath(os.path.join(
    os.getcwd(), 'frontend', 'playground-frontend', '.env'
))

FINAL_PORT_ENV_PATH = os.path.abspath(os.path.join(
    os.getcwd(), '.env'
))

SECRET_NAME = "frontend_env"
