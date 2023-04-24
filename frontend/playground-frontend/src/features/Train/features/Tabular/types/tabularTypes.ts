import {
  BaseTrainspaceData,
  DatasetData,
} from "@/features/Train/types/trainTypes";
import {
  STEP_SETTINGS,
  TRAINSPACE_SETTINGS,
} from "@/features/Train/features/Tabular/constants/tabularConstants";

export interface TrainspaceData<
  T extends typeof TRAINSPACE_SETTINGS["steps"][number] = typeof TRAINSPACE_SETTINGS["steps"][number]
> extends BaseTrainspaceData {
  dataSource: "TABULAR";
  steps: typeof TRAINSPACE_SETTINGS["steps"][number];
  datasetData: T extends "PARAMETERS" | "REVIEW" | "TRAIN"
    ? DatasetData
    : DatasetData | undefined;
  parameterData: T extends "REVIEW" | "TRAIN"
    ? ParameterData
    : ParameterData | undefined;
  reviewData: T extends "TRAIN" ? ReviewData : ReviewData | undefined;
}

export interface ParameterData {
  targetCol: string;
  features: string[];
  problemType: typeof STEP_SETTINGS["PARAMETERS"]["problemTypes"][number];
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
