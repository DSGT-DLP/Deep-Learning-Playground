import datetime
from dataclasses import dataclass
from typing import List, Literal, Tuple, Union

from ninja import Schema

# keep in sync with trainTypes.ts
DATA_SOURCE = Literal[
    "TABULAR",
    "PRETRAINED",
    "IMAGE",
    "AUDIO",
    "TEXTUAL",
    "CLASSICAL_ML",
    "OBJECT_DETECTION",
]

TRAIN_STATUS = Literal[
    "QUEUED", "STARTING", "UPLOADING", "TRAINING", "SUCCESS", "ERROR"
]


class TrainResultsData(Schema):
    name: str
    trainspaceId: str
    dataSource: DATA_SOURCE
    status: TRAIN_STATUS
    created: datetime.date
    step: str
    uid: str


CHART_TYPE = Literal["LINE", "AUC/ROC", "CONFUSION_MATRIX"]


class TimeSeriesMetric(Schema):
    x_name: str
    y_name: str

    x_values: List[float]
    y_values: List[float]


class TimeSeriesChart(Schema):
    name: str

    time_series: List[TimeSeriesMetric]
    chart_type = "LINE"
    graph_index: int


class AucRocChart(Schema):
    name: str

    values: List[Tuple[List[float], List[float], float]]
    chart_type = "AUC/ROC"
    graph_index: int


class ConfusionMatrixChart(Schema):
    name: str

    values: List[List[float]]

    chart_type = "CONFUSION_MATRIX"
    graph_index: int


Chart = Union[TimeSeriesChart, AucRocChart, ConfusionMatrixChart]


class DetailedTrainResultsData(Schema):
    basic_info: TrainResultsData

    all_metrics: List[Chart]