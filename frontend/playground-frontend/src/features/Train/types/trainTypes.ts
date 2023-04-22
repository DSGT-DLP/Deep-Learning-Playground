import { DATA_SOURCE_ARR } from "@/features/Train/constants/trainConstants";

export type DATA_SOURCE = typeof DATA_SOURCE_ARR[number];

export type TRAIN_STATUS =
  | "QUEUED"
  | "STARTING"
  | "UPLOADING"
  | "TRAINING"
  | "SUCCESS"
  | "ERROR";

export interface FileUploadData {
  name: string;
  lastModified: string;
  contentType: string;
  sizeInBytes: number;
}

export interface DatasetData {
  isDefaultDataSet: boolean;
}

export interface FileDatasetData extends DatasetData {
  isDefaultDataSet: false;
  name: string;
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

export interface BaseTrainspaceData {
  name: string;
  dataSource: DATA_SOURCE;
  step: string;
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
