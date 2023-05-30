import {
    BaseTrainspaceData,
    DatasetData,
  } from "@/features/Train/types/trainTypes";
  import {
    STEP_SETTINGS,
    TRAINSPACE_SETTINGS,
  } from "@/features/Train/features/ObjectDetection/constants/detectionConstants";
  
  export interface TrainspaceData<
    T extends (typeof TRAINSPACE_SETTINGS)["steps"][number] | "TRAIN" =
      | (typeof TRAINSPACE_SETTINGS)["steps"][number]
      | "TRAIN"
  > extends BaseTrainspaceData {
    dataSource: "TABULAR";
    imageData: T extends "PARAMETERS" | "REVIEW" | "TRAIN"
      ? DatasetData
      : DatasetData | undefined;
    parameterData: T extends "REVIEW" | "TRAIN"
      ? ParameterData
      : ParameterData | undefined;
    reviewData: T extends "TRAIN" ? ReviewData : ReviewData | undefined;
  }
  
  export interface TrainspaceResultsData extends TrainspaceData<"TRAIN"> {
    trainspaceId: string;
    created: Date;
    status: string;
  }
  
  export interface ParameterData {
    detectionType: (typeof STEP_SETTINGS)["PARAMETERS"]["detectionTypes"][number]["value"];
    detectionProblemType: (typeof STEP_SETTINGS)["PARAMETERS"]["detectionProblemTypes"][number]["value"];
    transforms: {
      value: keyof (typeof STEP_SETTINGS)["PARAMETERS"]["detectionTransforms"];
      parameters: number[];
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
  