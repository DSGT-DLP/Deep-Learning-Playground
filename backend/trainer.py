from utils import ProblemType
import torch #pytorch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

"""
This file contains helpful functions to aid in training and evaluation
of Pytorch models. 
Links to helpful Resources: 
https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
 
"""


def train_classification_model(model, train_loader, test_loader, optimizer, criterion, epochs):
    """
    Function for training pytorch model for classification

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
    for epoch in range(epochs):
        model.train(True) #set model to train mode
        batch_loss = [] #accumulate list of loss per batch
        for i, data in enumerate(train_loader):
            input, labels = data #each batch is (input, label) pair in dataloader
            optimizer.zero_grad() #zero out gradient for each batch
            output = model(input) #make prediction on input
            # output = torch.argmax(output, dim=2)
            output = torch.reshape(output, (output.shape[0], output.shape[2]))
            labels = labels.squeeze_()
            loss = criterion(output, labels.long()) #compute the loss
            loss.backward() #backpropagation
            optimizer.step() #adjust optimizer weights
            batch_loss.append(loss.detach().numpy())
        mean_train_loss = np.mean(batch_loss)
        train_loss.append(mean_train_loss)
        
        model.train(False) #test the model on test set
        batch_loss = []
        for i, data in enumerate(test_loader):
            input, labels = data 
            loss_test = model(input)
            batch_loss.append(loss_test.detach().numpy())
        mean_test_loss = np.mean(batch_loss)
        test_loss.append(mean_test_loss)
        print(f"epoch: {epoch}, train loss: {train_loss[-1]}, test loss = {test_loss[-1]}")
    return train_loss, test_loss

def train_regression_model(model, train_loader, test_loader, optimizer, criterion, epochs):
    """
    Train Regression model in Pytorch

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
    for epoch in range(epochs):
        model.train(True) #set model to train mode
        batch_loss = [] #accumulate list of loss per batch
        for i, data in enumerate(train_loader):
            input, labels = data #each batch is (input, label) pair in dataloader
            optimizer.zero_grad() #zero out gradient for each batch
            output = model(input) #make prediction on input
            loss = criterion(output, labels) #compute the loss
            loss.backward() #backpropagation
            optimizer.step() #adjust optimizer weights
            batch_loss.append(loss.detach().numpy())
        train_loss.append(np.mean(batch_loss))
        
        model.train(False) #test the model on test set
        batch_loss = []
        for i, data in enumerate(test_loader):
            input, labels = data 
            loss_test = model(input)
            batch_loss.append(loss_test.detach().numpy())
        test_loss.append(np.mean(batch_loss))
        print(f"epoch: {epoch}, train loss: {train_loss[-1]}, test loss = {test_loss[-1]}")
    return train_loss, test_loss

def train_model(model, train_loader, test_loader, optimizer, criterion, epochs, problem_type):
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
    
    if (problem_type.upper() == ProblemType.get_problem_obj(ProblemType.CLASSIFICATION)):
        return train_classification_model(model, train_loader, test_loader, optimizer, criterion, epochs)
    elif (problem_type.upper() == ProblemType.get_problem_obj(ProblemType.REGRESSION)):
        return train_regression_model(model, train_loader, test_loader, optimizer, epochs)
            
def get_predictions(model: nn.Module, test_loader):
    """
    Given a trained torch model and a test loader, get the predictions vs. actual value
    (this function will also undo the scaler transform through inverse_transform())
    Args:
        model (nn.Module): trained torch model
        test_loader (torch.DataLoader): 
    """
    predictions = []
    ground_truth_values = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            input, labels = data
            model.eval() #evaluation mode 
            yhat = model(input)
            predictions.append(yhat.detach().numpy().ravel())
            ground_truth_values.append(labels.detach().numpy().ravel())
        
    return np.array(predictions), np.array(ground_truth_values)