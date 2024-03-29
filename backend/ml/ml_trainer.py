"""
Trainer for classical ML models 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from common.constants import (
    CLASSICAL_ML_CONFUSION_MATRIX,
    CLASSICAL_ML_RESULT_CSV_PATH,
    SAVED_MODEL_ML,
)
from common.utils import (
    generate_acc_plot,
    generate_loss_plot,
    generate_train_time_csv,
    generate_confusion_matrix,
    generate_AUC_ROC_CURVE,
)
from common.utils import ProblemType
import pickle


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

    y_pred = model.predict_proba(X_test)
    # confusion matrix logic. Get sense of true positives, false positives, true negatives, false negatives
    conf_matrix, numerical_category_list = generate_confusion_matrix(
        y_test, y_pred, model_type="ml"
    )
    plt.savefig(CLASSICAL_ML_CONFUSION_MATRIX)

    # Collecting additional outputs to give to the frontend
    auxiliary_outputs = {}
    auxiliary_outputs["confusion_matrix"] = conf_matrix
    auxiliary_outputs["numerical_category_list"] = numerical_category_list

    # generate AUC curve (if soft labels exist)
    return auxiliary_outputs


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
    y_pred_train = model.predict(X_train)

    # create popular metrics
    train_rmse = mean_squared_error(y_true=y_train, y_pred=y_pred_train, squared=False)
    train_mape = mean_absolute_percentage_error(y_true=y_train, y_pred=y_pred_train)
    test_rmse = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
    test_mape = mean_absolute_percentage_error(y_true=y_test, y_pred=y_pred)
    print(
        f"Regression Root Mean Squared Error => test: {test_rmse.round(4)}\t train: {train_rmse.round(4)}"
    )
    print(
        f"Regression Mean Absolute Percentage Error:  => test: {test_mape.round(4)*100}%\t train: {train_mape.round(4)*100}%"
    )

    return {}


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
    outputs = None
    if problem_type.upper() == ProblemType.get_problem_obj(ProblemType.CLASSIFICATION):
        outputs = train_classical_ml_classification(
            model, X_train, X_test, y_train, y_test
        )
    elif problem_type.upper() == ProblemType.get_problem_obj(ProblemType.REGRESSION):
        outputs = train_classical_ml_regression(model, X_train, X_test, y_train, y_test)

    # save model
    with open(SAVED_MODEL_ML, "wb") as files:
        pickle.dump(model, files)

    return outputs
