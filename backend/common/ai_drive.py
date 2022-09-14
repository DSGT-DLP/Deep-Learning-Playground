import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, fetch_california_housing

from backend.common.constants import ONNX_MODEL, CSV_FILE_NAME
from backend.common.dataset import read_dataset, loader_from_zipped
from backend.common.default_datasets import get_default_dataset, get_img_default_dataset_loaders, get_audio_default_dataset_loaders
from backend.common.optimizer import get_optimizer
from backend.common.utils import *

from backend.dl.dl_model import DLModel
from backend.dl.dl_model_parser import parse_deep_user_architecture
from backend.dl.dl_trainer import train_deep_model, train_deep_image_classification, train_deep_audio_classification
from backend.dl.dl_model_parser import get_object

from backend.ml.ml_trainer import train_classical_ml_model

def dl_tabular_drive(
    user_arch,
    criterion,
    optimizer_name,
    problem_type,
    fileURL,
    target=None,
    features=None,
    default=None,
    test_size=0.2,
    epochs=5,
    shuffle=True,
    batch_size=20,
    json_csv_data_str="",
    send_progress=None
):
    """
    Driver function/entrypoint into backend for deep learning model. Onnx file is generated containing model architecture for user to visualize in netron.app
    Args:
        user_arch (list): list that contains user defined deep learning architecture
        criterion (str): What loss function to use
        optimizer (str): What optimizer does the user wants to use (Adam or SGD for now, but more support in later iterations)
        problem type (str): "classification" or "regression" problem
        target (str): name of target column
        features (list): list of columns in dataframe for the feature based on user selection
        default (str, optional): the default dataset chosen by the user. Defaults to None.
        test_size (float, optional): size of test set in train/test split (percentage). Defaults to 0.2.
        epochs (int, optional): number of epochs/rounds to run model on
        shuffle (bool, optional): should the dataset be shuffled prior to train/test split
    :return: a dictionary containing the epochs, train and test accuracy and loss results, each in a list

    NOTE:
         CSV_FILE_NAME is the data csv file for the torch model. Assumed that you have one dataset file
    """
    if not default:
        if fileURL:
            read_dataset(fileURL)
        elif json_csv_data_str:
            pass
        else:
            raise ValueError("Need a file input")

    if default and problem_type.upper() == "CLASSIFICATION":
        X, y = get_default_dataset(default.upper())
        print(y.head())
    elif default and problem_type.upper() == "REGRESSION":
        X, y = get_default_dataset(default.upper())
    else:
        if json_csv_data_str:
            input_df = pd.read_json(json_csv_data_str, orient="records")

        y = input_df[target]
        X = input_df[features]

    if len(y) * test_size < batch_size or len(y) * (1 - test_size) < batch_size:
        raise ValueError("reduce batch size, not enough values in dataframe")

    if problem_type.upper() == "CLASSIFICATION":
        # label encode the categorical values to numbers
        y = y.astype("category")
        y = y.cat.codes
        print(y.head())

    # Convert to tensor
    if shuffle and problem_type.upper() == "CLASSIFICATION":
        # using stratify only for classification problems to ensure correct AUC calculation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0, shuffle=True, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0, shuffle=shuffle
        )

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_tensors(
        X_train, X_test, y_train, y_test
    )
    # Build the Deep Learning model that the user wants
    model = DLModel(parse_deep_user_architecture(user_arch))
    print(f"model: {model}")
    
    optimizer = get_optimizer(
        model, optimizer_name=optimizer_name, learning_rate=0.05
    )
    # criterion = LossFunctions.get_loss_obj(LossFunctions[criterion])
    print(f"loss criterion: {criterion}")
    train_loader, test_loader = get_dataloaders(
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size=batch_size
    )
    train_loss_results = train_deep_model(
        model, train_loader, test_loader, optimizer, criterion, epochs, problem_type, send_progress
    )
    torch.onnx.export(model, X_train_tensor, ONNX_MODEL)

    return train_loss_results
    
def dl_img_drive(
    train_transform,
    test_transform,
    user_arch,
    criterion,
    optimizer_name, 
    default,
    epochs,
    batch_size,
    shuffle,
    IMAGE_UPLOAD_FOLDER,
    send_progress
):    
    print(user_arch)
    model = DLModel(parse_deep_user_architecture(user_arch))

    train_transform = parse_deep_user_architecture(train_transform)
    test_transform = parse_deep_user_architecture(test_transform)

    if not default:
        for x in os.listdir(IMAGE_UPLOAD_FOLDER):
            if x != ".gitkeep":
                zip_file = os.path.join(os.path.abspath(IMAGE_UPLOAD_FOLDER), x)
                break
        train_loader, test_loader = loader_from_zipped(zip_file, batch_size, shuffle, train_transform, test_transform)
    else:
        train_loader, test_loader = get_img_default_dataset_loaders(default, train_transform, test_transform, batch_size, shuffle)

    print("got data loaders")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # model should go to GPU before initializing optimizer  https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least/66096687#66096687 

    optimizer = get_optimizer(
            model, optimizer_name=optimizer_name, learning_rate=0.05
    )

    train_loss_results= train_deep_image_classification(
        model, train_loader, test_loader, optimizer, criterion, epochs, device, send_progress=send_progress
    )
    return train_loss_results

def dl_audio_drive(
    train_transform,
    test_transform,
    user_arch,
    criterion,
    optimizer_name, 
    default,
    epochs,
    batch_size,
    shuffle,
    AUDIO_UPLOAD_FOLDER,
    sample_rate,
    max_sec,
    send_progress
):    
    print(user_arch)
    model = DLModel(parse_deep_user_architecture(user_arch))

    train_transform = parse_deep_user_architecture(train_transform)
    test_transform = parse_deep_user_architecture(test_transform)

    if not default:
        for x in os.listdir(AUDIO_UPLOAD_FOLDER):
            if x != ".gitkeep":
                zip_file = os.path.join(os.path.abspath(AUDIO_UPLOAD_FOLDER), x)
                break
        # train_loader, test_loader = loader_from_zipped(zip_file, batch_size, shuffle, train_transform, test_transform)
    else:
        train_loader, test_loader = get_audio_default_dataset_loaders(
            default, test_transform, train_transform, batch_size, shuffle, sample_rate, max_sec
        )

    print("got data loaders")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) # model should go to GPU before initializing optimizer  https://stackoverflow.com/questions/66091226/runtimeerror-expected-all-tensors-to-be-on-the-same-device-but-found-at-least/66096687#66096687 

    optimizer = get_optimizer(
        model, optimizer_name=optimizer_name, learning_rate=0.05
    )

    train_loss_results= train_deep_audio_classification(
        model, train_loader, test_loader, optimizer, criterion, epochs, device, send_progress=send_progress
    )
    return train_loss_results
    
def ml_drive(
    user_model,
    problem_type,
    target=None,
    features=None,
    default=False,
    test_size=0.2,
    shuffle=True,
):
    """
    Driver function/endpoint into backend for training a classical ML model (eg: SVC, SVR, DecisionTree, Naive Bayes, etc)

    Args:
        user_model (str): What ML model and parameters does the user want
        problem_type (str): "classification" or "regression" problem
        target (str, optional): name of target column. Defaults to None.
        features (list, optional): list of columns in dataframe for the feature based on user selection. Defaults to None.
        default (bool, optional): use the iris dataset for default classifiction or california housing for default regression. Defaults to False.
        test_size (float, optional): size of test set in train/test split (percentage). Defaults to 0.2.
        shuffle (bool, optional): should the dataset be shuffled prior to train/test split
    """
    try:
        if default and problem_type.upper() == "CLASSIFICATION":
            dataset = load_iris()
            X, y = get_default_dataset(dataset)
            print(y.head())
        elif default and problem_type.upper() == "REGRESSION":
            # If the user specifies no dataset, use california housing as default regression
            dataset = fetch_california_housing()
            X, y = get_default_dataset(dataset)
        else:
            input_df = pd.read_csv(CSV_FILE_NAME)
            y = input_df[target]
            X = input_df[features]

        if shuffle and problem_type.upper() == "CLASSIFICATION":
            # using stratify only for classification problems to ensure correct AUC calculation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0, shuffle=True, stratify=y
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=0, shuffle=shuffle
            )
        model = get_object(user_model)
        train_classical_ml_model(
            model, X_train, X_test, y_train, y_test, problem_type=problem_type
        )
    except Exception as e:
        raise e