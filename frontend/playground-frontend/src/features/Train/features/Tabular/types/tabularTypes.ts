import {
  BaseTrainspaceData,
  DatasetData,
  ReviewData,
  TrainResultsData,
  TrainspaceTypes,
} from "@/features/Train/types/trainTypes";
import { TABULAR_PROBLEM_TYPES_ARR } from "@/features/Train/features/Tabular/constants/tabularConstants";

export interface TabularData<
  T extends TrainspaceTypes["TABULAR"]["step"] = TrainspaceTypes["TABULAR"]["step"]
> extends BaseTrainspaceData {
  dataSource: "TABULAR";
  step: T;
  datasetData: T extends "PARAMETERS" | "REVIEW" | "TRAIN"
    ? DatasetData
    : DatasetData | undefined;
  parameterData: T extends "REVIEW" | "TRAIN"
    ? TabularParameterData
    : TabularParameterData | undefined;
  reviewData: T extends "TRAIN" ? ReviewData : ReviewData | undefined;
}

export interface TabularParameterData {
  targetCol: string;
  features: string[];
  problemType: TABULAR_PROBLEM_TYPE;
  criterion: string;
  optimizerName: string;
  shuffle: boolean;
  epochs: number;
  testSize: number;
  batchSize: number;
}

export interface TabularTrainResultsData extends TrainResultsData {
  dataSource: "TABULAR";
  tabularData: TabularData<"TRAIN">;
}

export type TABULAR_PROBLEM_TYPE = typeof TABULAR_PROBLEM_TYPES_ARR[number];
