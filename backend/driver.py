import pandas as pd
from loss import LossFunctions
from optimizer import get_optimizer
from input_parser import parse_user_architecture
from trainer import train_model
from utils import get_dataloaders
from model import DLModel
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def drive(user_arch, criterion, dataset_dir=None, target=None, default=False, test_size=0.2, epochs=5, shuffle=True):
    """
    Driver function/entrypoint into backend

    Args:
        user_arch (list): list that contains user defined deep learning architecture
        dataset_dir (str): file path to the dataset
        target (str): name of target column
        default (bool, optional): use the iris dataset or not. Defaults to False.
        test_size (float, optional): size of test set in train/test split. Defaults to 0.2.
        epochs (int, optional): number of epochs/rounds to run model on
        shuffle (bool, optional): should the dataset be shuffled prior to train/test split
    """
    if (default):
        dataset = load_iris()
        iris_df = pd.DataFrame(dataset.data)
        iris_df['class']=dataset.target
        iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
        iris_df.dropna(how="all", inplace=True) # remove any empty lines
        y = iris_df["class"]
        X = iris_df.drop("class", axis=1, inplace=False)
    else:
        dataset = pd.read_csv(dataset_dir)
        y = dataset[target]
        X = dataset.drop(target, axis=1, inplace=False)
    
    print(X.shape)
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=shuffle)
    X_train_tensor = Variable(torch.Tensor(X_train.to_numpy()))
    y_train_tensor = Variable(torch.Tensor(y_train.to_numpy()))
    X_train_tensor = torch.reshape(X_train_tensor, (X_train_tensor.size()[0], 1, X_train_tensor.size()[1]))
    y_train_tensor = torch.reshape(y_train_tensor, (y_train_tensor.size()[0], 1))
    
    X_test_tensor = Variable(torch.Tensor(X_test.to_numpy()))
    y_test_tensor = Variable(torch.Tensor(y_test.to_numpy()))
    X_test_tensor = torch.reshape(X_test_tensor, (X_test_tensor.size()[0], 1, X_test_tensor.size()[1]))
    y_test_tensor = torch.reshape(y_test_tensor, (y_test_tensor.size()[0], 1))
    
    model = DLModel(parse_user_architecture(user_arch))
    print(f"model: {model}")
    optimizer = get_optimizer(model, "sgd", learning_rate=0.05)
    criterion = LossFunctions[criterion]
    train_loader, test_loader = get_dataloaders(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size=20)
    train_loss, test_loss = train_model(model, train_loader, test_loader, optimizer, criterion, epochs)
    print(f"train loss: {train_loss}")
    print(f"test loss: {test_loss}")

drive(["nn.Linear(4, 10)", "nn.Linear(10, 1)"], "CELOSS", default=True)
    
    
    
    
    