import { ALL_STEPS, DATA_SOURCE } from "@/features/Train/types/trainTypes";
import DatasetStep from "../components/DatasetStep";

export const DATA_SOURCE_ARR = [
  "TABULAR",
  "PRETRAINED",
  "IMAGE",
  "AUDIO",
  "TEXTUAL",
  "CLASSICAL_ML",
  "OBJECT_DETECTION",
] as const;

export const TABULAR_STEPS_ARR = [
  "DATASET",
  "PARAMETERS",
  "REVIEW",
  "TRAIN",
] as const;

export const DATA_SOURCE_SETTINGS: {
  [T in DATA_SOURCE]: {
    name: string;
    steps: readonly string[];
    defaultDatasets: { label: string; value: string }[];
  };
} = {
  TABULAR: {
    name: "Tabular",
    steps: TABULAR_STEPS_ARR,
    defaultDatasets: [
      { label: "Iris", value: "IRIS" },
      { label: "California Housing", value: "CALIFORNIAHOUSING" },
      { label: "Diabetes", value: "DIABETES" },
      { label: "Digits", value: "DIGITS" },
      { label: "Wine", value: "WINE" },
    ],
  },
  PRETRAINED: {
    name: "Pretrained",
    steps: ["UPLOAD"],
    defaultDatasets: [],
  },
  IMAGE: {
    name: "Image",
    steps: ["UPLOAD"],
    defaultDatasets: [],
  },
  AUDIO: {
    name: "Audio",
    steps: ["UPLOAD"],
    defaultDatasets: [],
  },
  TEXTUAL: {
    name: "Textual",
    steps: ["UPLOAD"],
    defaultDatasets: [],
  },
  CLASSICAL_ML: {
    name: "Classical ML",
    steps: ["UPLOAD"],
    defaultDatasets: [],
  },
  OBJECT_DETECTION: {
    name: "Object Detection",
    steps: ["UPLOAD"],
    defaultDatasets: [],
  },
};

export const STEPS_SETTINGS: {
  [T in ALL_STEPS]: {
    name: string;
    optional: boolean;
    stepComponent: React.FC<{
      renderStepperButtons: (handleStepSubmit: () => void) => React.ReactNode;
    }>;
  };
} = {
  DATASET: { name: "Dataset", optional: false, stepComponent: DatasetStep },
  PARAMETERS: {
    name: "Parameters",
    optional: false,
    stepComponent: DatasetStep,
  },
  REVIEW: { name: "Review", optional: false, stepComponent: DatasetStep },
  TRAIN: { name: "Train", optional: false, stepComponent: DatasetStep },
};
