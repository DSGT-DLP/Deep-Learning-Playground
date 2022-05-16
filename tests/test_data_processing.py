# unit tests for dataset.py file (maybe dummy csv and see if it works properly. Assert 2 files the same)

from urllib.error import URLError
import pytest
import csv
from backend.constants import *
from backend.dataset import *
import filecmp


@pytest.mark.parametrize(
    "url,path_to_file",
    [
        (
            "https://raw.githubusercontent.com/karkir0003/dummy/main/job.csv",
            "./tests/expected/job.csv",
        ),
        (
            "https://raw.githubusercontent.com/karkir0003/dummy/main/cars.csv",
            "./tests/expected/cars.csv",
        ),
    ],
)
def test_dataset_url_reading(url, path_to_file):
    read_dataset(url)

    with open(CSV_FILE_PATH) as f1:
        with open(path_to_file) as f2:
            reader1 = csv.reader(f1)
            reader2 = csv.reader(f2)
            assert [line for line in reader1][:-1] == [line for line in reader2]


# @pytest.mark.parametrize("url,path_to_file", [("https://csv.com/jobs.csv", './tests/expected/jobs.csv'), ("https://csv.com/cars.csv", "./tests/expected/cars.csv")])
# def test_dataset_invalid_url(url, path_to_file):
#     with pytest.raises(Exception):
#         read_dataset(url)


@pytest.mark.parametrize(
    "url,path_to_file",
    [
        ("csv.com/jobs.csv", "./tests/expected/jobs.csv"),
        ("csv.com/cars.csv", "./tests/expected/cars.csv"),
    ],
)
def test_dataset_invalid_url(url, path_to_file):
    with pytest.raises(Exception):
        read_dataset(url)


# def test_read_local_csv_file(file_path):
#     read_local_csv_file(file_path)
#     pass
