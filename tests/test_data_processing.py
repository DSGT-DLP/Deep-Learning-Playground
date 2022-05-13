#unit tests for dataset.py file (maybe dummy csv and see if it works properly. Assert 2 files the same)

import pytest
import csv
from backend.constants import *
from backend.dataset import *
import filecmp
import os

@pytest.mark.parametrize('url,path_to_file', [('https://raw.githubusercontent.com/karkir0003/dummy/main/job.csv','./tests/expected/job.csv'), ('https://raw.githubusercontent.com/karkir0003/dummy/main/cars.csv', './tests/expected/cars.csv')])
def test_dataset_url_reading(url, path_to_file):
    read_dataset(url)
    res = filecmp.cmp(CSV_FILE_PATH, path_to_file, shallow=False)
    #print(res)
    
    # with open(CSV_FILE_PATH, mode="r") as f1:
    #     reader1 = csv.reader(f1)
    # with open(path_to_file, mode="r") as f2:
    #     reader2 = csv.reader(f2)
    #     #assert len(reader1) == len(reader2) #same number of lines
    #     for line1, line2 in enumerate(zip(reader1, reader2)):
    #         assert line1 == line2
    
    with open(CSV_FILE_PATH) as f1:
        with open(path_to_file) as f2:
            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)
            assert [line for line in reader1][:-1] == [line for line in reader2]

@pytest.raises()
def test_dataset_invalid_url(url, path_to_file)
    