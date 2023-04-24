import {
  TABULAR_PROBLEM_TYPES_ARR,
  TABULAR_STEPS_ARR,
} from "@/features/Train/features/Tabular/constants/tabularConstants";
import { TabularData } from "../features/Tabular/types/tabularTypes";
import { DATA_SOURCE_ARR } from "../constants/trainConstants";

export type DATA_SOURCE = typeof DATA_SOURCE_ARR[number];

//export type PROBLEM_TYPE = typeof PROBLEM_TYPES_ARR[number];

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

type BaseDataSourceSettingsType = {
  name: string;
  trainspaceComponent: React.FC;
};

type BaseStepSettingsType<T extends DATA_SOURCE> = {
  name: string;
  optional: boolean;
  stepComponent: React.FC<{
    renderStepperButtons: (
      handleStepSubmit: (data: TrainspaceTypes[T]["dataType"]) => void
    ) => React.ReactNode;
  }>;
};

export type TrainspaceTypes = {
  TABULAR: {
    dataType: TabularData;
    step: typeof TABULAR_STEPS_ARR[number];
    settings: BaseDataSourceSettingsType & {
      steps: typeof TABULAR_STEPS_ARR;
    };
    stepSettings: {
      DATASET: BaseStepSettingsType<"TABULAR"> & {
        defaultDatasets: { label: string; value: string }[];
      };
      PARAMETERS: BaseStepSettingsType<"TABULAR"> & {
        problemTypes: {
          label: string;
          value: typeof TABULAR_PROBLEM_TYPES_ARR;
        }[];
      };
      REVIEW: BaseStepSettingsType<"TABULAR">;
      TRAIN: BaseStepSettingsType<"TABULAR">;
    };
  };
  PRETRAINED: {
    dataType: TrainResultsData;
    settings: BaseDataSourceSettingsType & {
      steps: [];
    };
    stepSettings: Record<string, never>;
  };
  IMAGE: {
    dataType: TabularData;
    settings: BaseDataSourceSettingsType & {
      steps: [];
    };
    stepSettings: Record<string, never>;
  };
  AUDIO: {
    dataType: TabularData;
    settings: BaseDataSourceSettingsType & {
      steps: [];
    };
    stepSettings: Record<string, never>;
  };
  TEXTUAL: {
    dataType: TabularData;
    settings: BaseDataSourceSettingsType & {
      steps: [];
    };
    stepSettings: Record<string, never>;
  };
  CLASSICAL_ML: {
    dataType: TabularData;
    settings: BaseDataSourceSettingsType & {
      steps: [];
    };
    stepSettings: Record<string, never>;
  };
  OBJECT_DETECTION: {
    dataType: TabularData;
    settings: BaseDataSourceSettingsType & {
      steps: [];
    };
    stepSettings: Record<string, never>;
  };
};
