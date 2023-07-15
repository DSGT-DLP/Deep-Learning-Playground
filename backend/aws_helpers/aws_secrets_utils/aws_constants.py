# .env file constants
import os
import sys

ENV_KEYS = [
    "REACT_APP_SECRET_KEY",
    "REACT_APP_CAPTCHA_SITE_KEY",
    "REACT_APP_FEEDBACK_EMAIL",
]
cwd = os.getcwd()

directory, last_part = os.path.split(cwd)

directory_without_last_part = directory

FINAL_ENV_PATH = os.path.abspath(
    os.path.join(directory_without_last_part, "frontend", ".env")
)

SECRET_NAME = "frontend_env"
