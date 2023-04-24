import { TrainspaceTypes } from "@/features/Train/types/trainTypes";
import TabularDatasetStep from "@/features/Train/features/Tabular/components/TabularDatasetStep";
import TabularTrainspace from "@/features/Train/features/Tabular/components/TabularTrainspace";
import { TABULAR_STEPS_ARR } from "@/features/Train/features/Tabular/constants/tabularConstants";
import TabularParametersStep from "../features/Tabular/components/TabularParametersStep";

export const DATA_SOURCE_ARR = [
  "TABULAR",
  "PRETRAINED",
  "IMAGE",
  "AUDIO",
  "TEXTUAL",
  "CLASSICAL_ML",
  "OBJECT_DETECTION",
] as const;

export const DATA_SOURCE_SETTINGS: {
  [T in keyof TrainspaceTypes]: TrainspaceTypes[T]["settings"] & {
    stepsSettings: {
      [U in keyof TrainspaceTypes[T]["stepSettings"]]: TrainspaceTypes[T]["stepSettings"][U];
    };
  };
} = {
  TABULAR: {
    name: "Tabular",
    trainspaceComponent: TabularTrainspace,
    steps: TABULAR_STEPS_ARR,
    stepsSettings: {
      DATASET: {
        name: "Dataset",
        optional: false,
        stepComponent: TabularDatasetStep,
        defaultDatasets: [
          { label: "Iris", value: "IRIS" },
          { label: "California Housing", value: "CALIFORNIAHOUSING" },
          { label: "Diabetes", value: "DIABETES" },
          { label: "Digits", value: "DIGITS" },
          { label: "Wine", value: "WINE" },
        ],
      },
      PARAMETERS: {
        name: "Parameters",
        optional: false,
        stepComponent: TabularParametersStep,
        problemTypes: [],
      },
      REVIEW: {
        name: "Review",
        optional: false,
        stepComponent: TabularDatasetStep,
      },
      TRAIN: {
        name: "Train",
        optional: false,
        stepComponent: TabularDatasetStep,
      },
    },
  },
  PRETRAINED: {
    name: "Pretrained",
    trainspaceComponent: TabularTrainspace,
    steps: [],
    stepsSettings: {},
  },
  IMAGE: {
    name: "Image",
    trainspaceComponent: TabularTrainspace,
    steps: [],
    stepsSettings: {},
  },
  AUDIO: {
    name: "Audio",
    trainspaceComponent: TabularTrainspace,
    steps: [],
    stepsSettings: {},
  },
  TEXTUAL: {
    name: "Textual",
    trainspaceComponent: TabularTrainspace,
    steps: [],
    stepsSettings: {},
  },
  CLASSICAL_ML: {
    name: "Classical ML",
    trainspaceComponent: TabularTrainspace,
    steps: [],
    stepsSettings: {},
  },
  OBJECT_DETECTION: {
    name: "Object Detection",
    trainspaceComponent: TabularTrainspace,
    steps: [],
    stepsSettings: {},
  },
};
