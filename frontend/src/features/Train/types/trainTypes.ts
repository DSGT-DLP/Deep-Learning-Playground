import { DATA_SOURCE_ARR } from "../constants/trainConstants";

// keep in sync with schemas.py
export type DATA_SOURCE = typeof DATA_SOURCE_ARR[number];

export type TRAIN_STATUS =
  | "QUEUED"
  | "STARTING"
  | "UPLOADING"
  | "TRAINING"
  | "SUCCESS"
  | "ERROR";

export interface BaseTrainspaceData {
  name: string;
  dataSource: DATA_SOURCE;
  step: number;
}

// basic information, used on dashboard
export interface TrainResultsData {
  name: string;
  trainspaceId: string;
  dataSource: DATA_SOURCE;
  status: TRAIN_STATUS;
  created: Date;
  uid: string;
}

export type CHART_TYPE = "LINE" | "AUC/ROC" | "CONFUSION_MATRIX"

export type Chart = TimeSeriesChart | AucRocChart | ConfusionMatrixChart

export interface TimeSeriesMetric {
  x_name: string;
  y_name: string;

  x_values: number[];
  y_values: number[];
}

export interface TimeSeriesChart {
  name: string;

  time_series: TimeSeriesMetric[]
  chart_type: "LINE" 
  graph_index: number;
}

export interface AucRocChart {
  name: string;

  values: [number[], number[], number][];

  chart_type: "AUC/ROC"
  graph_index: number;
}

export interface ConfusionMatrixChart {
  name: string;
  
  values: number[][];

  chart_type: "CONFUSION_MATRIX"
  graph_index: number;
}

// more detailed information, used when viewing a run
export interface DetailedTrainResultsData {
  basic_info: TrainResultsData

  all_metrics: Chart[]
}

export interface FileUploadData {
  name: string;
  lastModified: string;
  contentType: string;
  sizeInBytes: number;
}

export interface DatasetData {
  isDefaultDataset: boolean;
  name: string;
}
