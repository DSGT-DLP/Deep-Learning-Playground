import pandas as pd
import traceback

from utils import *
from constants import CSV_FILE_NAME, ONNX_MODEL
from loss import LossFunctions
from optimizer import get_optimizer
from model_parser import parse_user_architecture
from trainer import train_model, get_predictions
from model import DLModel
from webdriver import open_onnx_file
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
#from torchviz import make_dot

def drive(user_arch, criterion, optimizer_name, problem_type, target=None, default=False, test_size=0.2, epochs=5, shuffle=True):
    """
    Driver function/entrypoint into backend

    Args:
        user_arch (list): list that contains user defined deep learning architecture
        criterion (str): What loss function to use
        optimizer (str): What optimizer does the user wants to use (Adam or SGD for now, but more support in later iterations)
        problem type (str): "classification" or "regression" problem
        target (str): name of target column
        default (bool, optional): use the iris dataset or not. Defaults to False.
        test_size (float, optional): size of test set in train/test split. Defaults to 0.2.
        epochs (int, optional): number of epochs/rounds to run model on
        shuffle (bool, optional): should the dataset be shuffled prior to train/test split
    
    NOTE:
         CSV_FILE_NAME is the data csv file for the torch model. Assumed that you have one dataset file
    """
    try:
        if (default):
            #If the user specifies no dataset, use iris as the default
            dataset = load_iris()
            input_df = pd.DataFrame(dataset.data)
            input_df['class']=dataset.target
            input_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
            input_df.dropna(how="all", inplace=True) # remove any empty lines
            y = input_df["class"]
            X = input_df.drop("class", axis=1, inplace=False)
        else:
            input_df = pd.read_csv(CSV_FILE_NAME)
            y = input_df[target]
            X = input_df.drop(target, axis=1, inplace=False)
        
        #Convert to tensor
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0, shuffle=shuffle)
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = get_tensors(X_train, X_test, y_train, y_test)
        
        
        #Build the Deep Learning model that the user wants
        model = DLModel(parse_user_architecture(user_arch))
        print(f"model: {model}")
        optimizer = get_optimizer(model, optimizer_name=optimizer_name, learning_rate=0.05)
        criterion = LossFunctions.get_loss_obj(LossFunctions[criterion])
        print(f"loss criterion: {criterion}")
        train_loader, test_loader = get_dataloaders(X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, batch_size=20)
        train_loss, test_loss, epoch_time = train_model(model, train_loader, test_loader, optimizer, criterion, epochs, problem_type)
        print(f"train loss: {train_loss}")
        print(f"test loss: {test_loss}")
        generate_loss_plot(train_loss, test_loss)
        generate_train_time_csv(epoch_time)
        pred, ground_truth = get_predictions(model, test_loader)
        torch.onnx.export(model, X_train_tensor, ONNX_MODEL)
        open_onnx_file(ONNX_MODEL)
        
        #viz = make_dot(pred, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
        #viz = make_dot(pred, params=dict(model.named_parameters()), show_attrs=True, show_saved=True)
        #viz.render("my_viz.png")
        
    except Exception:
        return traceback.format_exc() #give exception in string format

if __name__ == "__main__":
    print(drive(["nn.Linear(4, 10)", "nn.ReLU()", "nn.Linear(10, 3)"], "CELOSS", "SGD", problem_type="classification", default=True, epochs=10))
    
    
    
    
    