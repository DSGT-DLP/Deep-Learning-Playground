from collections import Counter
from backend.common.loss_functions import compute_loss, compute_img_loss
from backend.dl.dl_eval import compute_accuracy, compute_correct
from backend.common.utils import (
    generate_acc_plot,
    generate_loss_plot,
    generate_train_time_csv,
    generate_confusion_matrix,
    generate_AUC_ROC_CURVE,
)
from backend.common.utils import ProblemType
from backend.common.constants import (
    DEEP_LEARNING_RESULT_CSV_PATH,
    EPOCH,
    TRAIN_TIME,
    TRAIN_LOSS,
    TEST_LOSS,
    TRAIN_ACC,
    TEST,
    VAL_TEST_ACC,
    SAVED_MODEL_DL,
    ONNX_MODEL,
)
import torch  # pytorch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
import traceback

"""
This file contains helpful functions to aid in training and evaluation
of Pytorch models. 
Links to helpful Resources: 
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
 
"""


def train_deep_classification_model(
    model, train_loader, test_loader, optimizer, criterion, epochs, category_list
):
    """
    Function for training pytorch model for classification. This function also times how long it takes to complete each epoch
    Args:
        model (nn.Module): Torch Model to train
        train_loader (torch.DataLoader): Train Dataset split into batches
        test_loader (torch.DataLoader): Test Dataset split into batches
        optimizer (torch.optim.Optimizer): Optimizer to use when training model
        criterion(str): Loss function
        epochs (int): number of epochs
    :return: a dictionary containing confusion matrix and AUC/ROC plot raw data
    """
    try:
        train_loss = []  # accumulate training loss over each epoch
        test_loss = []  # accumulate testing loss over each epoch
        epoch_time = []  # how much time it takes for each epoch
        train_acc = []  # accuracy of training set
        val_acc = []  # accuracy of test/validation set
        labels_last_epoch = []
        y_pred_last_epoch = []

        num_train_batches = len(train_loader)
        # total number of data points used for training per epoch
        epoch_train_size = train_loader.batch_size * num_train_batches
        num_test_batches = len(test_loader)
        # total number of data points used for testing per epoch
        epoch_test_size = test_loader.batch_size * num_test_batches

        for epoch in range(epochs):
            train_correct = (
                0  # number of correct predictions in training set in current epoch
            )
            test_correct = (
                0  # number of correct predictions in testing set in current epoch
            )
            epoch_batch_loss = 0  # cumulative training/testing loss per epoch

            start_time = time.time()
            model.train(True)  # set model to train model
            for i, data in enumerate(train_loader):
                # each batch is (input, label) pair in dataloader
                input, labels = data
                optimizer.zero_grad()  # zero out gradient for each batch
                output = model(input)  # make prediction on input
                train_correct += compute_correct(output, labels)
                # compute the loss
                loss = compute_loss(criterion, output, labels)
                loss.backward()  # backpropagation
                optimizer.step()  # adjust optimizer weights
                epoch_batch_loss += float(loss.detach())

            epoch_time.append(time.time() - start_time)
            mean_train_acc = train_correct / epoch_train_size
            mean_train_loss = epoch_batch_loss / num_train_batches
            train_acc.append(mean_train_acc)
            train_loss.append(mean_train_loss)

            model.train(False)  # test the model on test set
            epoch_batch_loss = 0
            for i, data in enumerate(test_loader):
                input, labels = data
                test_pred = model(input)
                # currently only preserving the prediction array and label array for the last epoch for
                # confusion matrix calculation
                if epoch == epochs - 1:
                    y_pred_last_epoch.append(test_pred.detach().numpy().squeeze())

                    labels_last_epoch.append(labels.detach().numpy().squeeze())

                test_correct += compute_correct(test_pred, labels)
                loss = compute_loss(criterion, test_pred, labels)
                epoch_batch_loss += float(loss.detach())
            mean_test_acc = test_correct / epoch_test_size
            mean_test_loss = epoch_batch_loss / num_test_batches
            val_acc.append(mean_test_acc)
            test_loss.append(mean_test_loss)

            print(
                f"epoch: {epoch}, train loss: {train_loss[-1]}, test loss: {test_loss[-1]}, train_acc: {mean_train_acc}, val_acc: {mean_test_acc}"
            )
        result_table = pd.DataFrame(
            {
                EPOCH: [i for i in range(1, epochs + 1)],
                TRAIN_TIME: epoch_time,
                TRAIN_LOSS: train_loss,
                TEST_LOSS: test_loss,
                TRAIN_ACC: train_acc,
                VAL_TEST_ACC: val_acc,
            }
        )
        print(result_table.head())
        confusion_matrix, numerical_category_list = generate_confusion_matrix(
            labels_last_epoch, y_pred_last_epoch, category_list
        )
        result_table.to_csv(DEEP_LEARNING_RESULT_CSV_PATH, index=False)
        generate_acc_plot(DEEP_LEARNING_RESULT_CSV_PATH)
        generate_loss_plot(DEEP_LEARNING_RESULT_CSV_PATH)

        # Collecting additional outputs to give to the frontend
        auxiliary_outputs = {}
        auxiliary_outputs["confusion_matrix"] = confusion_matrix
        auxiliary_outputs["numerical_category_list"] = numerical_category_list

        # Generating AUC_ROC curve data to send to frontend to make interactive plot
        AUC_ROC_curve_data, numerical_category_list_AUC = generate_AUC_ROC_CURVE(
            labels_last_epoch, y_pred_last_epoch, category_list
        )
        auxiliary_outputs["AUC_ROC_curve_data"] = AUC_ROC_curve_data
        auxiliary_outputs["numerical_category_list_AUC"] = numerical_category_list_AUC
        auxiliary_outputs["category_list"] = category_list
        torch.save(model, SAVED_MODEL_DL)  # saving model into a pt file
        return auxiliary_outputs
    except Exception:
        raise Exception("Deep Learning classification didn't train properly")


def train_deep_regression_model(
    model, train_loader, test_loader, optimizer, criterion, epochs
):
    """
    Train Regression model in Pytorch. This function also times how long it takes to complete each epoch
    Args:
        model (nn.Module): Torch Model to train
        train_loader (torch.DataLoader): Train Dataset split into batches
        test_loader (torch.DataLoader): Test Dataset split into batches
        optimizer (torch.optim.Optimizer): Optimizer to use when training model
        criterion: Loss function
        epochs (int): number of epochs
    :return: an empty dictionary
    """
    try:
        train_loss = []  # accumulate training loss over each epoch
        test_loss = []  # accumulate testing loss over each epoch
        epoch_time = []  # how much time it takes for each epoch
        num_train_batches = len(train_loader)
        num_test_batches = len(test_loader)
        print(num_train_batches, num_test_batches)
        for epoch in range(epochs):
            start_time = time.time()
            model.train(True)  # set model to train mode
            epoch_batch_loss = 0  # cumulative training/testing loss per epoch
            for i, data in enumerate(train_loader):
                # each batch is (input, label) pair in dataloader
                input, labels = data
                optimizer.zero_grad()  # zero out gradient for each batch
                output = model(input)  # make prediction on input
                # compute the loss
                loss = compute_loss(criterion, output, labels)
                loss.backward()  # backpropagation
                optimizer.step()  # adjust optimizer weights
                epoch_batch_loss += float(loss.detach())
            epoch_time.append(time.time() - start_time)
            train_loss.append(epoch_batch_loss / num_train_batches)

            model.train(False)  # test the model on test set
            epoch_batch_loss = 0
            for i, data in enumerate(test_loader):
                input, labels = data
                test_pred = model(input)
                loss = compute_loss(criterion, test_pred, labels)
                epoch_batch_loss += float(loss.detach())
            test_loss.append(epoch_batch_loss / num_test_batches)
            print(
                f"epoch: {epoch}, train loss: {train_loss[-1]}, test loss = {test_loss[-1]}"
            )
        result_table = pd.DataFrame(
            {
                EPOCH: [i for i in range(1, epochs + 1)],
                TRAIN_TIME: epoch_time,
                TRAIN_LOSS: train_loss,
                TEST_LOSS: test_loss,
            }
        )
        print(result_table.head())
        result_table.to_csv(DEEP_LEARNING_RESULT_CSV_PATH, index=False)
        torch.save(model, SAVED_MODEL_DL)  # saving model into a pt file
        generate_loss_plot(DEEP_LEARNING_RESULT_CSV_PATH)
        return {}

    except Exception:
        raise Exception("Deep learning regression model didn't run properly")


def train_deep_model(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    epochs,
    problem_type,
    category_list=[],
):
    """
    Given train loader, train torch model
    Args:
        model (nn.Module): Torch Model to train
        train_loader (torch.DataLoader): Train Dataset split into batches
        optimizer (torch.optim.Optimizer): Optimizer to use when training model
        criterion: Loss function
        epochs (int): number of epochs
        problem type (str): "classification" or "regression"
    :return: a dictionary containing the epochs, train and test accuracy and loss results, each in a list
    """
    if problem_type.upper() == ProblemType.get_problem_obj(ProblemType.CLASSIFICATION):
        return train_deep_classification_model(
            model,
            train_loader,
            test_loader,
            optimizer,
            criterion,
            epochs,
            category_list,
        )
    elif problem_type.upper() == ProblemType.get_problem_obj(ProblemType.REGRESSION):
        return train_deep_regression_model(
            model, train_loader, test_loader, optimizer, criterion, epochs
        )


def get_deep_predictions(model: nn.Module, test_loader):
    """
    Given a trained torch model and a test loader, get predictions vs. ground truth

    Args:
        model (nn.Module): trained torch model
        test_loader (torch.DataLoader):
    """
    predictions = []
    ground_truth_values = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input, labels = data
            model.eval()  # evaluation mode
            yhat = model(input)
            predictions.append(yhat.detach().numpy().ravel())
            ground_truth_values.append(labels.detach().numpy().ravel())

    prediction_tensor = torch.from_numpy(np.array(predictions).T)
    ground_truth_tensor = torch.from_numpy(np.array(ground_truth_values).T)

    return prediction_tensor, ground_truth_tensor


def train_deep_image_classification(
    model,
    train_loader,
    test_loader,
    optimizer,
    criterion,
    epochs,
    device,
    category_list=[],
):
    try:
        model = model.to(device)
        train_loss = []  # accumulate training loss over each epoch
        test_loss = []  # accumulate testing loss over each epoch
        epoch_time = []  # how much time it takes for each epoch
        train_acc = []  # accuracy of training set
        val_acc = []  # accuracy of test/validation set
        labels_last_epoch = []
        y_pred_last_epoch = []

        num_train_batches = len(train_loader)
        # total number of data points used for training per epoch
        epoch_train_size = train_loader.batch_size * num_train_batches
        num_test_batches = len(test_loader)
        # total number of data points used for testing per epoch
        epoch_test_size = test_loader.batch_size * num_test_batches
        train_weights_count = Counter()
        test_weights_count = Counter()

        input_dim = None

        for epoch in range(epochs):
            model.train(True)

            if epoch == 0 and criterion == "WCELOSS":
                for i, j in train_loader:
                    train_weights_count.update(j.detach().numpy().flatten())
                for i, j in test_loader:
                    test_weights_count.update(j.detach().numpy().flatten())

            loss, train_correct, epoch_batch_loss = 0, 0, 0
            start_time = time.time()
            for x in train_loader:
                y = x[1]  # label for all images in the batch
                x = x[0]  # (C, H, W) image
                input_dim = x.shape
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                pred = model(x)
                loss = compute_img_loss(criterion, pred, y, train_weights_count)

                loss.backward()
                optimizer.step()
                y_pred, y_true = torch.argmax(pred, axis=1), y.long().squeeze()
                train_correct += (y_pred == y_true).type(torch.float).sum().item()
                epoch_batch_loss += float(loss.detach())
            epoch_time.append(time.time() - start_time)
            mean_train_loss = epoch_batch_loss / num_train_batches
            mean_train_acc = train_correct / epoch_train_size
            train_loss.append(mean_train_loss)

            train_acc.append(mean_train_acc)
            model.train(False)
            loss, test_correct = 0, 0
            print("training for this epoch finished, going to validation")

            for x in test_loader:
                y = x[1]
                x = x[0]
                x, y = x.to(device), y.to(device)
                input_dim = x.shape
                pred = model(x)
                loss = compute_img_loss(criterion, pred, y, test_weights_count)
                y_pred, y_true = torch.argmax(pred, axis=1), y.long().squeeze()
                if epoch == epochs - 1:
                    y_pred_last_epoch.append(pred.detach().numpy().squeeze())
                    labels_last_epoch.append(y.detach().numpy().squeeze())

                test_correct += (y_pred == y_true).type(torch.float).sum().item()
                epoch_batch_loss += float(loss.detach())

            mean_test_loss = epoch_batch_loss / num_test_batches
            mean_test_acc = test_correct / epoch_test_size
            test_loss.append(mean_test_loss)
            val_acc.append(mean_test_acc)

            print(
                f"epoch: {epoch}, train loss: {train_loss[-1]}, test loss: {test_loss[-1]}, train_acc: {mean_train_acc}, val_acc: {mean_test_acc}"
            )
        result_table = pd.DataFrame(
            {
                EPOCH: [i for i in range(1, epochs + 1)],
                TRAIN_TIME: epoch_time,
                TRAIN_LOSS: train_loss,
                TEST_LOSS: test_loss,
                TRAIN_ACC: train_acc,
                VAL_TEST_ACC: val_acc,
            }
        )
        print(result_table)

        confusion_matrix, numerical_category_list = generate_confusion_matrix(
            labels_last_epoch, y_pred_last_epoch
        )
        result_table.to_csv(DEEP_LEARNING_RESULT_CSV_PATH, index=False)

        generate_acc_plot(DEEP_LEARNING_RESULT_CSV_PATH)
        generate_loss_plot(DEEP_LEARNING_RESULT_CSV_PATH)

        auxiliary_outputs = {}
        auxiliary_outputs["confusion_matrix"] = confusion_matrix
        auxiliary_outputs["numerical_category_list"] = numerical_category_list

        # Generating AUC_ROC curve data to send to frontend to make interactive plot
        AUC_ROC_curve_data, numerical_category_list_AUC = generate_AUC_ROC_CURVE(
            labels_last_epoch, y_pred_last_epoch, category_list
        )
        auxiliary_outputs["AUC_ROC_curve_data"] = AUC_ROC_curve_data
        auxiliary_outputs["numerical_category_list_AUC"] = numerical_category_list_AUC
        auxiliary_outputs["category_list"] = category_list

        torch.save(model, SAVED_MODEL_DL)  # saving model into a pt file
        torch.onnx.export(model, torch.randn(input_dim), ONNX_MODEL)
        return auxiliary_outputs

    except Exception:
        raise Exception("Deep Learning classification didn't train properly")
