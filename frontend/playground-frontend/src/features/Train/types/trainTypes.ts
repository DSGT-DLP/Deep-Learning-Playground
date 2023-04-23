import { DATA_SOURCE_ARR } from "@/features/Train/constants/trainConstants";
import { TABULAR_STEPS_ARR } from "@/features/Train/features/Tabular/constants/tabularConstants";
import { TabularData } from "../features/Tabular/types/tabularTypes";

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
  step: string;
}

export type TrainspaceSteps<T extends DATA_SOURCE> = T extends "TABULAR"
  ? typeof TABULAR_STEPS_ARR
  : never[];
export type TrainspaceStep<T extends DATA_SOURCE> = TrainspaceSteps<T>[number];
export type TrainspaceData<
  T extends DATA_SOURCE = DATA_SOURCE,
  U extends TrainspaceStep<T> = TrainspaceStep<T>
> = T extends "TABULAR" ? TabularData<U> : BaseTrainspaceData;

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

export interface ReviewData {
  notificationEmail?: string;
  notificationPhoneNumber?: string;
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
