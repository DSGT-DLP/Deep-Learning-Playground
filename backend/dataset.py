import pandas as pd
import urllib.request as request
import csv
import traceback
from backend.constants import *

def read_dataset(url):
    """
    Given a url to a CSV dataset, read it and build temporary csv file

    Args:
        url (str): URL to dataset
    """
    try:
        r = request.urlopen(url).read().decode('utf8').split("\n")
        reader = csv.reader(r)
        with open(CSV_FILE_PATH, mode="w", newline="") as f:
            csvwriter = csv.writer(f)
            for line in reader:
                csvwriter.writerow(line)
    except Exception:
        traceback.print_exc()

def read_local_csv_file(file_path):
    """
    Given a local file path or the "uploaded CSV file", read it in pandas dataframe

    Args:
        file_path (str): file path
    
    """
    try:
        with open(file_path, mode="r") as data_file:
            with open(CSV_FILE_PATH, mode="w", newline="") as f:
                csvwriter = csv.writer(f)
                for line in data_file:
                    csvwriter.writerow(line)
    except Exception:
        traceback.print_exc()
        
    

if __name__ == "__main__":
    read_dataset("https://raw.githubusercontent.com/karkir0003/dummy/main/job.csv")
    read_local_csv_file("test.csv")