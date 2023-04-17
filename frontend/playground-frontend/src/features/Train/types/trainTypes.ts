export const DATA_SOURCE_ARR = [
  "TABULAR",
  "PRETRAINED",
  "IMAGE",
  "AUDIO",
  "TEXTUAL",
  "CLASSICAL_ML",
  "OBJECT_DETECTION",
] as const;

export type DATA_SOURCE = typeof DATA_SOURCE_ARR[number];
export type TRAIN_STATUS =
  | "QUEUED"
  | "STARTING"
  | "UPLOADING"
  | "TRAINING"
  | "SUCCESS"
  | "ERROR";

export const TABULAR_STEPS_ARR = [
  "UPLOAD",
  "PARAMETERS",
  "REVIEW",
  "TRAIN",
] as const;
export type TABULAR_STEPS = typeof TABULAR_STEPS_ARR[number];

export interface UploadData {
  isDefaultDataSet: boolean;
}

export interface FileUploadData extends UploadData {
  isDefaultDataSet: false;
  fileUrl: string;
}

export interface DefaultUploadData extends UploadData {
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
}

export interface TabularData<T extends TABULAR_STEPS = TABULAR_STEPS>
  extends BaseTrainspaceData {
  dataSource: "TABULAR";
  step: T;
  uploadData: T extends "PARAMETERS" | "REVIEW" | "TRAIN"
    ? UploadData
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
