import os
from torchvision.transforms import transforms

CSV_FILE_NAME = "data.csv"
CSV_FILE_PATH = f"{os.path.join(os.getcwd(), CSV_FILE_NAME)}"
DEEP_LEARNING_RESULT_CSV_PATH = f"./backend/dl_results.csv"
TENSORBOARD_LOG = "runs/user_experiment"
ONNX_MODEL = "./frontend/src/backend_outputs/my_deep_learning_model.onnx"
LOSS_VIZ = "./frontend/src/backend_outputs/visualization_output/my_loss_plot.png"
ACC_VIZ = "./frontend/src/backend_outputs/visualization_output/my_accuracy_plot.png"
CONFUSION_VIZ = (
    "./frontend/src/backend_outputs/visualization_output/my_confusion_matrix.png"
)
AUC_ROC_VIZ = "./frontend/src/backend_outputs/visualization_output/my_AUC_ROC_Curve.png"
SAVED_MODEL_DL = "./frontend/src/backend_outputs/model.pt"
SAVED_MODEL_ML = "./frontend/src/backend_outputs/model.pkl"
TRAIN_TIME_CSV = "epoch_times.csv"
NETRON_URL = "https://netron.app/"
OPEN_FILE_BUTTON = "open-file-button"
CLASSICAL_ML_CONFUSION_MATRIX = (
    "./frontend/src/backend_outputs/visualization_output/confusion_matrix.png"
)
CLASSICAL_ML_RESULT_CSV_PATH = f"ml_results.csv"
IMAGE_DETECTION_RESULT_CSV_PATH = f"detection_results.csv"
EPOCH = "epoch"
TRAIN_TIME = "train_time"
TRAIN_LOSS = "train_loss"
TEST_LOSS = "test_loss"
TRAIN_ACC = "train_acc"
TEST = "test"
VAL_TEST_ACC = "val/test acc"
DEFAULT_DATASETS = {
    "IRIS": "load_iris()",
    "BREAST CANCER": "load_breast_cancer()",
    "CALIFORNIAHOUSING": "fetch_california_housing()",
    "DIABETES": "load_diabetes()",
    "DIGITS": "load_digits()",
    "WINE": "load_wine()",
}
UNZIPPED_DIR_NAME = "unzipped_data"
SENDER = "DSGT Deep Learning Playground <dlp@datasciencegt.org>"
AWS_REGION = "us-west-2"

CHARSET = "utf-8"
TENSOR_ONLY_TRANSFORMS = [
    transforms.LinearTransformation,
    transforms.Normalize,
    transforms.ConvertImageDtype,
    transforms.RandomErasing,
]

DEFAULT_TRANSFORM = [transforms.Resize((256, 256)), transforms.ToTensor()]

PIL_ONLY_TRANSFORMS = [transforms.RandomChoice, transforms.RandomOrder]

POINTS_PER_QUESTION = 10

IMAGE_FILE_DOWNLOAD_TMP_PATH = os.path.abspath("backend/tmp/image_file_downloads")

LOGGER_FORMAT = "%(asctime)s %(name)s:%(filename)-5s:%(levelname)-5s - %(message)s"
