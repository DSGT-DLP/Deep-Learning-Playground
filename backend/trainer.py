from dl_eval import compute_accuracy
from utils import generate_acc_plot, generate_loss_plot, generate_train_time_csv
from utils import ProblemType
from constants import RESULT_CSV_PATH
import torch #pytorch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

"""
This file contains helpful functions to aid in training and evaluation
of Pytorch models. 
Links to helpful Resources: 
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
 
"""


def train_classification_model(
    model, train_loader, test_loader, optimizer, criterion, epochs
):
    """
    Function for training pytorch model for classification. This function also times how long it takes to complete each epoch
    Args:
        model (nn.Module): Torch Model to train
        train_loader (torch.DataLoader): Train Dataset split into batches
        test_loader (torch.DataLoader): Test Dataset split into batches
        optimizer (torch.optim.Optimizer): Optimizer to use when training model
        criterion: Loss function
        epochs (int): number of epochs
    """
    train_loss = [] #accumulate training loss over each epoch
    test_loss = [] #accumulate testing loss over each epoch
    epoch_time = [] #how much time it takes for each epoch
    train_acc = [] #accuracy of training set
    val_acc = [] #accuracy of test/validation set
    for epoch in range(epochs):
        batch_train_acc = [] #find train accuracy for each batch
        batch_test_acc = [] #find test accuracy for each batch
        start_time = time.time()
        model.train(True)  # set model to train mode
        batch_loss = []  # accumulate list of loss per batch
        for i, data in enumerate(train_loader):
            input, labels = data #each batch is (input, label) pair in dataloader
            optimizer.zero_grad() #zero out gradient for each batch
            output = model(input) #make prediction on input
            batch_train_acc.append(compute_accuracy(output, labels))
            # output = torch.argmax(output, dim=2)
            output = torch.reshape(output, (output.shape[0], output.shape[2]))
            labels = labels.squeeze_()
            loss = criterion(output, labels.long())  # compute the loss
            loss.backward()  # backpropagation
            optimizer.step()  # adjust optimizer weights
            batch_loss.append(loss.detach().numpy())
        epoch_time.append(time.time() - start_time)
        mean_train_loss = np.mean(batch_loss)
        mean_train_acc = np.mean(batch_train_acc)
        train_loss.append(mean_train_loss)
        train_acc.append(mean_train_acc)
        
        
        model.train(False) #test the model on test set
        batch_loss = []
        for i, data in enumerate(test_loader):
            input, labels = data 
            test_pred = model(input)
            batch_test_acc.append(compute_accuracy(test_pred, labels))
            batch_loss.append(test_pred.detach().numpy())
        mean_test_loss = np.mean(batch_loss)
        mean_test_acc = np.mean(batch_test_acc)
        test_loss.append(mean_test_loss)
        val_acc.append(mean_test_acc)
        
        print(f"epoch: {epoch}, train loss: {train_loss[-1]}, test loss: {test_loss[-1]}, train_acc: {mean_train_acc}, val_acc: {mean_test_acc}")
    result_table = pd.DataFrame({"epoch": [i for i in range(1, epochs + 1)], "train time": epoch_time, "train_loss": train_loss, "test_loss": test_loss, "train_acc": train_acc, "val/test acc": val_acc})
    print(result_table.head())
    result_table.to_csv(RESULT_CSV_PATH, index=False)
    generate_acc_plot(train_acc, val_acc)
    generate_loss_plot(train_loss, test_loss)


def train_regression_model(
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
    """
    train_loss = []  # accumulate training loss over each epoch
    test_loss = []  # accumulate testing loss over each epoch
    epoch_time = []  # how much time it takes for each epoch
    for epoch in range(epochs):
        start_time = time.time()
        model.train(True)  # set model to train mode
        batch_loss = []  # accumulate list of loss per batch
        for i, data in enumerate(train_loader):
            input, labels = data  # each batch is (input, label) pair in dataloader
            optimizer.zero_grad()  # zero out gradient for each batch
            output = model(input)  # make prediction on input
            loss = criterion(output, labels)  # compute the loss
            loss.backward()  # backpropagation
            optimizer.step()  # adjust optimizer weights
            batch_loss.append(loss.detach().numpy())
        epoch_time.append(time.time() - start_time)
        train_loss.append(np.mean(batch_loss))

        model.train(False)  # test the model on test set
        batch_loss = []
        for i, data in enumerate(test_loader):
            input, labels = data
            loss_test = model(input)
            batch_loss.append(loss_test.detach().numpy())
        test_loss.append(np.mean(batch_loss))
        print(f"epoch: {epoch}, train loss: {train_loss[-1]}, test loss = {test_loss[-1]}")
    generate_loss_plot(train_loss, test_loss)
    result_table = pd.DataFrame({"epoch": [i for i in range(1, epochs + 1)], "train time": epoch_time, "train_loss": train_loss, "test_loss": test_loss, "train_acc": train_acc, "val/test acc": val_acc})
    print(result_table.head())
    result_table.to_csv(RESULT_CSV_PATH, index=False)


def train_model(
    model, train_loader, test_loader, optimizer, criterion, epochs, problem_type
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
    """

    if problem_type.upper() == ProblemType.get_problem_obj(ProblemType.CLASSIFICATION):
        return train_classification_model(
            model, train_loader, test_loader, optimizer, criterion, epochs
        )
    elif problem_type.upper() == ProblemType.get_problem_obj(ProblemType.REGRESSION):
        return train_regression_model(
            model, train_loader, test_loader, optimizer, epochs
        )


def get_predictions(model: nn.Module, test_loader):
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
