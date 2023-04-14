export type DATA_SOURCE =
  | "TABULAR"
  | "PRETRAINED"
  | "IMAGE"
  | "AUDIO"
  | "TEXTUAL"
  | "CLASSICAL_ML"
  | "OBJECT_DETECTION";
export type TRAIN_STATUS =
  | "QUEUED"
  | "STARTING"
  | "UPLOADING"
  | "TRAINING"
  | "SUCCESS"
  | "ERROR";

export const TABULAR_STEPS_ARR = [
  "UPLOAD",
  "PREPROCESS",
  "PARAMETERS",
  "TRAIN",
] as const;
export type TABULAR_STEPS = typeof TABULAR_STEPS_ARR[number];

export interface TrainSpaceData {
  name: string;
  execution_id: number;
  data_source: DATA_SOURCE;
  timestamp: string;
  status: TRAIN_STATUS;
  progress: number;
  step: string;
}

export interface TabularTrainSpaceData extends TrainSpaceData {
  data_source: "TABULAR";
  step: TABULAR_STEPS;
  uploadData: UploadData;
}

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
