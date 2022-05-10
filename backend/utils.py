from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt 

def get_dataloaders(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size):
    """
    Helper function to get Dataloaders for X_train, y_train, X_tes, y_test
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