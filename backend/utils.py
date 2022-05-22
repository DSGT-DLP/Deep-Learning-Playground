from constants import LOSS_VIZ, ACC_VIZ, TRAIN_TIME_CSV, DEEP_LEARNING_RESULT_CSV_PATH, EPOCH, TRAIN_TIME, TRAIN_LOSS, TEST_LOSS, TRAIN_ACC, TEST, VAL_TEST_ACC

import pandas as pd
import torch
import matplotlib.pyplot as plt
from enum import Enum
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import csv
import json


class ProblemType(Enum):
    # Are we solving a Classification or Regression problem
    CLASSIFICATION = "CLASSIFICATION"
    REGRESSION = "REGRESSION"

    def get_problem_obj(self):
        return self.value


def get_tensors(X_train, X_test, y_train, y_test):
    """
    Helper function to convert X_train, X_test, y_train, y_test
    into tensors for dataloader
    Args:
        X_train (pd.DataFrame): X_train (train set)
        X_test (pd.DataFrame): X_test (test set)
        y_train (pd.Series): label/value of target for each row of X_train
        y_test (pd.Series): label/value of target for each row of X_test

    Return:
        X_train, X_test, y_train, y_test in the form of tensor
    """
    X_train_tensor = Variable(torch.Tensor(X_train.to_numpy()))
    y_train_tensor = Variable(torch.Tensor(y_train.to_numpy()))
    X_train_tensor = torch.reshape(
        X_train_tensor, (X_train_tensor.size()[0], 1, X_train_tensor.size()[1])
    )
    y_train_tensor = torch.reshape(
        y_train_tensor, (y_train_tensor.size()[0], 1))

    X_test_tensor = Variable(torch.Tensor(X_test.to_numpy()))
    y_test_tensor = Variable(torch.Tensor(y_test.to_numpy()))
    X_test_tensor = torch.reshape(
        X_test_tensor, (X_test_tensor.size()[0], 1, X_test_tensor.size()[1])
    )
    y_test_tensor = torch.reshape(y_test_tensor, (y_test_tensor.size()[0], 1))

    X_train_tensor.requires_grad_(True)
    X_test_tensor.requires_grad_(True)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor


def get_dataloaders(
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size
):
    """
    Helper function to get Dataloaders for X_train, y_train, X_test, y_test
    Args:
        X_train_tensor (torch.Tensor)
        y_train_tensor (torch.Tensor)
        X_test_tensor (torch.Tensor)
        y_test_tensor (torch.Tensor)
        batch_size (int)
    """
    train = TensorDataset(X_train_tensor, y_train_tensor)
    test = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(test, batch_size=batch_size,
                             shuffle=False, drop_last=True)
    return train_loader, test_loader


def generate_loss_plot(results_path):
    """
    Given a training result file, plot the loss in a matplotlib plot
    Args:
        results_path(str): path to csv file containing training result
    """
    results = pd.read_csv(results_path)
    train_loss = results[TRAIN_LOSS]
    test_loss = results[TEST_LOSS]
    assert len(train_loss) == len(test_loss)
    plt.figure("Loss Plot")
    plt.clf()
    x_axis = [i for i in range(1, len(train_loss) + 1)]
    plt.scatter(x_axis, train_loss, c="r", label="train loss")
    plt.scatter(x_axis, test_loss, c="b", label="test loss")
    #get unique labels for legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("Train vs. Test loss for your Deep Learning Model")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.savefig(LOSS_VIZ)


def generate_acc_plot(results_path):
    """
    Given training result file, plot the accuracy in a matplotlib plot
    Args:
        results_path(str): path to csv file containing training result
    """
    results = pd.read_csv(results_path)
    train_acc = results[TRAIN_ACC]
    val_acc = results[VAL_TEST_ACC]
    assert len(train_acc) == len(val_acc)
    plt.figure("Accuracy Plot")
    plt.clf()
    x_axis = [i for i in range(1, len(train_acc) + 1)]
    plt.scatter(x_axis, train_acc, c="r", label="train accuracy")
    plt.scatter(x_axis, val_acc, c="b", label="test accuracy")
    #get unique labels for legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("Train vs. Test accuracy for your Deep Learning Model")
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy")
    plt.savefig(ACC_VIZ) 


def generate_train_time_csv(epoch_time):
    """
    Given the time taken to run each epoch, generate CSV of the DataFrame
    Args:
        epoch_time (list): array consisting of train time for each epoch
    """
    epoch = [i for i in range(1, len(epoch_time) + 1)]
    df = pd.DataFrame({"Train Time": epoch_time},
                      index=epoch, columns=["Train Time"])
    df.to_csv(TRAIN_TIME_CSV)


def csv_to_json(csvFilePath: str = DEEP_LEARNING_RESULT_CSV_PATH, jsonFilePath: str = None) -> str:
    """
    Creates a JSON data derived from the input CSV. Will return
    the JSON data and create a JSON file with the data, if a jsonFilePath is
    provided. src: https://pythonexamples.org/python-csv-to-json/

    :param csvFilePath: optional, file path of csv input file
    :param jsonFilePath: optional, output JSON file
    :return: Converted CSV file in JSON format
    """
    jsonArray = []

    # read csv file
    with open(csvFilePath, encoding='utf-8') as csvf:
        # load csv file data using csv library's dictionary reader
        csvReader = csv.DictReader(csvf)

        # convert each csv row into python dict
        for row in csvReader:
            # add this python dict to json array
            jsonArray.append(row)

    # creates the JSON string item and JSON data
    jsonString = json.dumps(jsonArray, indent=4)
    jsonData = json.loads(jsonString)

    # convert python jsonArray to JSON String and write to file, if path is
    # provided
    if jsonFilePath:
        with open(jsonFilePath, 'w', encoding='utf-8') as jsonf:
            jsonf.write(jsonString)

    return jsonData
