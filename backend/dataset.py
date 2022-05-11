import pandas as pd
import urllib.request as request
import csv
from constants import CSV_FILE_NAME

def read_dataset(url):
    """
    Given a url to a CSV dataset, read it and convert to pandas dataframe

    Args:
        url (str): URL to dataset
    """
    r = request.urlopen(url).read().decode('utf8').split("\n")
    reader = csv.reader(r)
    with open(CSV_FILE_NAME, mode="w") as f:
        for line in reader:
            print(f"line: {line}")
            f.write(str(line))

def read_csv_file(file_path):
    """
    Given a file path or the "uploaded CSV file", read it in pandas dataframe

    Args:
        file_path (str): file path
    
    Returns:
        pd.DataFrame
    """
    return pd.read_csv(file_path)

read_dataset("https://raw.githubusercontent.com/karkir0003/dummy/main/job.csv")