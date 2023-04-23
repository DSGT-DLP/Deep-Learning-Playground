import {
  BaseTrainspaceData,
  DatasetData,
  ReviewData,
  TrainResultsData,
  TrainspaceStep,
} from "@/features/Train/types/trainTypes";

export interface TabularData<
  T extends TrainspaceStep<"TABULAR"> = TrainspaceStep<"TABULAR">
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
  problemType: string;
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
