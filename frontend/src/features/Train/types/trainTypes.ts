import { DATA_SOURCE_ARR } from "../constants/trainConstants";

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

export interface TrainResultsData {
  name: string;
  trainspaceId: number;
  dataSource: DATA_SOURCE;
  status: TRAIN_STATUS;
  created: Date;
  step: string;
  uid: string;
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

export interface ImageUploadData {
  name: string;
}
