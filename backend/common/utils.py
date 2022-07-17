from backend.common.constants import (
    LOSS_VIZ,
    ACC_VIZ,
    TRAIN_TIME_CSV,
    DEEP_LEARNING_RESULT_CSV_PATH,
    EPOCH,
    TRAIN_TIME,
    TRAIN_LOSS,
    TEST_LOSS,
    TRAIN_ACC,
    TEST,
    VAL_TEST_ACC,
    CONFUSION_VIZ,
    AUC_ROC_VIZ
)
import pandas as pd
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from enum import Enum
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import csv
import os
import json

matplotlib.use("Agg")


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
    y_train_tensor = torch.reshape(y_train_tensor, (y_train_tensor.size()[0], 1))

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
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader


def generate_loss_plot(results_path) -> dict[str : list[float]]:
    """
    Given a training result file, plot the loss in a matplotlib plot
    Args:
        results_path(str): path to csv file containing training result
    :return: a dictionary containing the epoch, train, and test loss values
    """
    results = pd.read_csv(results_path)
    train_loss = results[TRAIN_LOSS]
    test_loss = results[TEST_LOSS]

    assert len(train_loss) == len(test_loss)

    plt.figure("Loss Plot")

    plt.clf()  # clear figure
    x_axis = [i for i in range(1, len(train_loss) + 1)]
    plt.scatter(x_axis, train_loss, c="r", label="train loss")
    plt.scatter(x_axis, test_loss, c="b", label="test loss")

    # get unique labels for legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Train vs. Test loss for your Deep Learning Model")
    plt.xlabel("Epoch Number")
    plt.ylabel("Loss")
    plt.savefig(LOSS_VIZ)

    return {
        "epochs": x_axis,
        "train_loss": train_loss.values.tolist(),
        "test_loss": test_loss.values.tolist(),
    }


def generate_acc_plot(results_path) -> dict[str : list[float]]:
    """
    Given training result file, plot the accuracy in a matplotlib plot
    Args:
        results_path(str): path to csv file containing training result
    :return: a dictionary containing the epoch, train, and test accuracy values
    """
    results = pd.read_csv(results_path)
    train_acc = results[TRAIN_ACC]
    val_acc = results[VAL_TEST_ACC]
    assert len(train_acc) == len(val_acc)
    plt.figure("Accuracy Plot")

    plt.clf()  # clear figure
    x_axis = [i for i in range(1, len(train_acc) + 1)]
    plt.scatter(x_axis, train_acc, c="r", label="train accuracy")
    plt.scatter(x_axis, val_acc, c="b", label="test accuracy")

    # get unique labels for legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title("Train vs. Test accuracy for your Deep Learning Model")
    plt.xlabel("Epoch Number")
    plt.ylabel("Accuracy")
    plt.savefig(ACC_VIZ)

    return {
        "epochs": x_axis,
        "train_acc": train_acc.values.tolist(),
        "test_acc": val_acc.values.tolist(),
    }


def generate_train_time_csv(epoch_time):
    """
    Given the time taken to run each epoch, generate CSV of the DataFrame
    Args:
        epoch_time (list): array consisting of train time for each epoch
    """
    epoch = [i for i in range(1, len(epoch_time) + 1)]
    df = pd.DataFrame({"Train Time": epoch_time}, index=epoch, columns=["Train Time"])
    df.to_csv(TRAIN_TIME_CSV)


def generate_confusion_matrix(labels_last_epoch, y_pred_last_epoch):
    """
    Given the prediction results and label, generate confusion matrix (only applicable to classification tasks)
    Args:
        label: array consisting of ground truth values
        y_pred: array consisting of predicted results
        categoryList: list of strings that represent the categories to classify into (this will be used to label the axis)
    Returns: the confusion matrix in a 2D-array format
    """
    label = []
    y_pred = []
    categoryList = []

    for l in labels_last_epoch.tolist():
        label.append(l[0])

    predictions = torch.argmax(
        y_pred_last_epoch, dim=-1
    )
    for prediction in predictions.tolist():
        y_pred.append(prediction[0])

    # generating a category list for confusion matrix axis labels
    if (len(y_pred_last_epoch) > 0 and len(y_pred_last_epoch[0]) > 0):
        for i, x in enumerate(y_pred_last_epoch[0][0]):
            categoryList.append(i)

    plt.clf()
    label_np = np.array(label)
    pred_np = np.array(y_pred)
    cm = confusion_matrix(label_np, pred_np, labels=categoryList)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap='Purples');  #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax.set_xlabel('Predicted');ax.set_ylabel('Actual'); 
    ax.set_title('Confusion Matrix (last Epoch)'); 
    ax.xaxis.set_ticklabels(categoryList); ax.yaxis.set_ticklabels(categoryList);
    plt.savefig(CONFUSION_VIZ)
    return cm.tolist()


def generate_AUC_ROC_CURVE(labels_last_epoch, y_pred_last_epoch):
    label_list = []
    y_preds_list = []
    categoryList = []
    plot_data = []  

    # generating a category list for confusion matrix axis labels, and setting up the y_preds_list and label_list for each category
    if (len(y_pred_last_epoch) > 0 and len(y_pred_last_epoch[0]) > 0):
        for i, x in enumerate(y_pred_last_epoch[0][0]):
            categoryList.append(i)
            y_preds_list.append([])
            label_list.append([])

    for label in labels_last_epoch:
        ground_truth = label.item();
        for x in range(len(label_list)):
            # toggle on a 1 for the category that corresponds to the ground truth, this is to produce the 1-vs-all system for multiclass classification
            if x == ground_truth:
                label_list[x].append(1)
            else:
                label_list[x].append(0)

    for row in y_pred_last_epoch:
        for i, tensor in enumerate(row[0]):
            # appending tensor values to each category's probability predicitons in y_preds_list
            y_preds_list[i].append(tensor.item())
    
    # making a AUC/ROC graph for each category's probability predicitons
    try:
        # using matplotlib in addition to plotly so that we can generate graph image in backend and email this to user
        plt.clf()
        plt.title('AUC/ROC Curves for your Deep Learning Model')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot([0, 1], [0, 1], linestyle='--', label=f'baseline')
        for i in range(len(categoryList)):
            pred_prob = np. array(y_preds_list[i])
            y_test = np. array(label_list[i])
            fpr, tpr, _ = roc_curve(y_test, pred_prob)
            auc = roc_auc_score(y_test, pred_prob)
            # this data will be sent to frontend to make interactive plotly graph
            plot_data.append([fpr.tolist(), tpr.tolist(), auc])
            plt.plot(fpr, tpr, linestyle='-', label=f'{i} (AUC: {round(auc,4)})')
        plt.legend()
        plt.savefig(AUC_ROC_VIZ)


    except:
        return []

    return plot_data


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
    with open(csvFilePath, encoding="utf-8") as csvf:
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
        with open(jsonFilePath, "w", encoding="utf-8") as jsonf:
            jsonf.write(jsonString)

    return jsonData
