import os


AWS_REGION = "us-west-2"
CSV_FILE_NAME = "data.csv"
CSV_FILE_PATH = f"{os.path.join(os.getcwd(), 'backend', CSV_FILE_NAME)}"
ONNX_MODEL = "./frontend/src/backend_outputs/my_deep_learning_model.onnx"
DEFAULT_DATASETS = {
    "IRIS": "load_iris()",
    "BREAST CANCER": "load_breast_cancer()",
    "CALIFORNIAHOUSING": "fetch_california_housing()",
    "DIABETES": "load_diabetes()",
    "DIGITS": "load_digits()",
    "WINE": "load_wine()",
}
