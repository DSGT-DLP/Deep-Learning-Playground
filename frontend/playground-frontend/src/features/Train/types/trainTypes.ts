import {
  DATA_SOURCE_ARR,
  TABULAR_STEPS_ARR,
} from "@/features/Train/constants/trainConstants";

export type DATA_SOURCE = typeof DATA_SOURCE_ARR[number];

export type TRAIN_STATUS =
  | "QUEUED"
  | "STARTING"
  | "UPLOADING"
  | "TRAINING"
  | "SUCCESS"
  | "ERROR";

export interface FileUploadData {
  fileUrl: string;
  name: string;
  lastModified: Date;
  contentType: string;
  sizeInBytes: number;
}

export interface DatasetData {
  isDefaultDataSet: boolean;
}

export interface FileDatasetData extends DatasetData {
  isDefaultDataSet: false;
  fileUrl: string;
}

export interface DefaultDatasetData extends DatasetData {
  isDefaultDataSet: true;
  dataSetName: string;
}

export interface TrainResultsData {
  name: string;
  trainspaceId: number;
  dataSource: DATA_SOURCE;
  status: TRAIN_STATUS;
  created: Date;
  step: string;
  uid: string;
}

export interface TabularTrainResultsData extends TrainResultsData {
  dataSource: "TABULAR";
  tabularData: TabularData<"TRAIN">;
}

export interface BaseTrainspaceData {
  name: string;
  dataSource: DATA_SOURCE;
  step: ALL_STEPS;
}

export type ALL_STEPS = TABULAR_STEPS;
export type TABULAR_STEPS = typeof TABULAR_STEPS_ARR[number];

export interface TabularData<T extends TABULAR_STEPS = TABULAR_STEPS>
  extends BaseTrainspaceData {
  dataSource: "TABULAR";
  step: T;
  datasetData: T extends "PARAMETERS" | "REVIEW" | "TRAIN"
    ? DatasetData
    : undefined;
  parameterData: T extends "REVIEW" | "TRAIN"
    ? TabularParameterData
    : undefined;
  reviewData: T extends "TRAIN" ? ReviewData : undefined;
}

export interface TabularParameterData {
  targetCol: string;
  features: string[];
  problemType: string;
  criterion: string;
  optimizerName: string;
  shuffle: boolean;
  epochs: number;
  testSize: number;
  batchSize: number;
}

export interface ReviewData {
  notificationEmail?: string;
  notificationPhoneNumber?: string;
}
