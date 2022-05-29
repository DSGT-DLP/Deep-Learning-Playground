import os

CSV_FILE_NAME = "data.csv"
CSV_FILE_PATH = f"{os.path.join(os.getcwd(), CSV_FILE_NAME)}"
DEEP_LEARNING_RESULT_CSV_PATH = f"dl_results.csv"
TENSORBOARD_LOG = "runs/user_experiment"
ONNX_MODEL = "../frontend/playground-frontend/src/backend_outputs/my_deep_learning_model.onnx"
LOSS_VIZ = "../frontend/playground-frontend/src/backend_outputs/visualization_output/my_loss_plot.png"
ACC_VIZ = "../frontend/playground-frontend/src/backend_outputs/visualization_output/my_accuracy_plot.png"
TRAIN_TIME_CSV = "epoch_times.csv"
NETRON_URL = "https://netron.app/"
OPEN_FILE_BUTTON = "open-file-button"
CLASSICAL_ML_CONFUSION_MATRIX = "../frontend/playground-frontend/src/visualization_output/confusion_matrix.png"
CLASSICAL_ML_RESULT_CSV_PATH = f"ml_results.csv"
EPOCH = "epoch"
TRAIN_TIME = "train_time"
TRAIN_LOSS = "train_loss"
TEST_LOSS = "test_loss"
TRAIN_ACC = "train_acc"
TEST = "test"
VAL_TEST_ACC = "val/test acc"
DEFAULT_DATASETS = {"IRIS": "load_iris()", "BREAST CANCER": "load_breast_cancer()", "CALIFORNIAHOUSING": "fetch_california_housing()",
                    "DIABETES": "load_diabetes()",  "DIGITS": "load_digits()", "WINE": "load_wine()"}
