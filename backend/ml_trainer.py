"""
Trainer for classical ML models 
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from constants import CLASSICAL_ML_CONFUSION_MATRIX, CLASSICAL_ML_RESULT_CSV_PATH
from utils import ProblemType

def train_classical_ml_classification(model, X_train, X_test, y_train, y_test):
    """
    Train classical ML Classification model

    Args:
        model (sklearn): sklearn model that user wants to train
        X_train (pd.DataFrame): Train dataset features
        X_test (pd.DataFrame): Test dataset features
        y_train (pd.DataFrame): target value corresponding to each row in X_train
        y_test (pd.DataFrame): target value corresponding to each row in X_test
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #confusion matrix logic. Get sense of true positives, false positives, true negatives, false negatives
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    disp.plot()
    plt.savefig(CLASSICAL_ML_CONFUSION_MATRIX)
    
    
    

def train_classical_ml_regression(model, X_train, X_test, y_train, y_test):
    """
    Train classical ML regression model

    Args:
        model (sklearn): sklearn model that user wants to train
        X_train (pd.DataFrame): Train dataset features
        X_test (pd.DataFrame): Test dataset features
        y_train (pd.DataFrame): target value corresponding to each row in X_train
        y_test (pd.DataFrame): target value corresponding to each row in X_test
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    

def train_classical_ml_model(model, X_train, X_test, y_train, y_test, problem_type):
    """
    Endpoint to train classical ML model

    Args:
        model (sklearn): sklearn model that user wants to train
        X_train (pd.DataFrame): Train dataset features
        X_test (pd.DataFrame): Test dataset features
        y_train (pd.DataFrame): target value corresponding to each row in X_train
        y_test (pd.DataFrame): target value corresponding to each row in X_test
        problem_type (str): "classification" or "regression" model
    """
    if (problem_type.upper() == ProblemType.get_problem_obj(ProblemType.CLASSIFICATION)):
        train_classical_ml_classification(model, X_train, X_test, y_train, y_test)
    elif (problem_type.upper() == ProblemType.get_problem_obj(ProblemType.REGRESSION)):
        train_classical_ml_regression(model, X_train, X_test, y_train, y_test)
    