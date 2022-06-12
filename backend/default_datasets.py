import pandas as pd
import numpy as np
from sklearn.datasets import *
from enum import Enum
from constants import DEFAULT_DATASETS

def get_default_dataset(dataset):
    """
    If user doesn't specify dataset
    Args:
        dataset (str): Which default dataset are you using (built in functions like load_boston(), load_iris())
    Returns:
        X: input (default dataset)
        y: target (default dataset)
    """
    try:
        if (dataset not in DEFAULT_DATASETS):
            raise Exception(f"The {dataset} file does not currently exist in our inventory. Please submit a request to the contributors of the repository")
        else:
            raw_data = eval(DEFAULT_DATASETS[dataset]) #get raw data from sklearn.datasets
            default_dataset = pd.DataFrame(data= np.c_[raw_data['data'], raw_data['target']],
                     columns= raw_data['feature_names'] + ['target'])
            default_dataset.dropna(how="all", inplace=True)  # remove any empty lines
            y = default_dataset['target']
            X = default_dataset.drop('target', axis=1)
            print("****default_dataset.head()**** \n ", default_dataset.head())
            return X, y
    
    except Exception:
        raise Exception(f"Unable to load the {dataset} file into Pandas DataFrame")