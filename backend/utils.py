import torch

from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable



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
    X_train_tensor = torch.reshape(X_train_tensor, (X_train_tensor.size()[0], 1, X_train_tensor.size()[1]))
    y_train_tensor = torch.reshape(y_train_tensor, (y_train_tensor.size()[0], 1))
    
    X_test_tensor = Variable(torch.Tensor(X_test.to_numpy()))
    y_test_tensor = Variable(torch.Tensor(y_test.to_numpy()))
    X_test_tensor = torch.reshape(X_test_tensor, (X_test_tensor.size()[0], 1, X_test_tensor.size()[1]))
    y_test_tensor = torch.reshape(y_test_tensor, (y_test_tensor.size()[0], 1))
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

def get_dataloaders(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size):
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
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, test_loader 