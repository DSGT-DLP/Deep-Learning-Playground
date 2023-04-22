import { TABULAR_STEPS_ARR } from "@/features/Train/constants/trainConstants";
import {
  BaseTrainspaceData,
  DatasetData,
  ReviewData,
  TabularParameterData,
  TrainResultsData,
} from "@/features/Train/types/trainTypes";

export type TrainspaceSteps<T> = T extends "TABULAR"
  ? typeof TABULAR_STEPS_ARR
  : string[];
export type TrainspaceStep<T> = TrainspaceSteps<T>[number];

export interface TabularData<
  T extends TrainspaceStep<"TABULAR"> = TrainspaceStep<"TABULAR">
> extends BaseTrainspaceData {
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

export interface TabularTrainResultsData extends TrainResultsData {
  dataSource: "TABULAR";
  tabularData: TabularData<"TRAIN">;
}
