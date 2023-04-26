import {
  BaseTrainspaceData,
  DatasetData,
} from "@/features/Train/types/trainTypes";
import {
  STEP_SETTINGS,
  TRAINSPACE_SETTINGS,
} from "@/features/Train/features/Tabular/constants/tabularConstants";

export interface TrainspaceData<
  T extends (typeof TRAINSPACE_SETTINGS)["steps"][number] = (typeof TRAINSPACE_SETTINGS)["steps"][number]
> extends BaseTrainspaceData {
  dataSource: "TABULAR";
  steps: (typeof TRAINSPACE_SETTINGS)["steps"][number];
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
  problemType: (typeof STEP_SETTINGS)["PARAMETERS"]["problemTypes"][number]["value"];
  criterion: (typeof STEP_SETTINGS)["PARAMETERS"]["criterions"][number]["value"];
  optimizerName: (typeof STEP_SETTINGS)["PARAMETERS"]["optimizers"][number]["value"];
  shuffle: boolean;
  epochs: number;
  testSize: number;
  batchSize: number;
  layers: {
    value: (typeof STEP_SETTINGS)["PARAMETERS"]["layers"][number]["value"];
    parameters: {
      data: number;
      value: (typeof STEP_SETTINGS)["PARAMETERS"]["layers"][number]["parameters"][number]["value"];
    }[];
  }[];
}

type LTypes = number | string | boolean;
export interface LayerParameter<T extends LTypes = LTypes> {
  index: number;
  parameter_name: string;
  min: T extends number ? number : null;
  max: T extends number ? number : null;
  parameter_type: T;
  default?: T extends "number" ? number : string;
  kwarg?: string;
  value?: T extends "number" ? number : string;
}

export interface ReviewData {
  notificationEmail?: string;
  notificationPhoneNumber?: string;
}
