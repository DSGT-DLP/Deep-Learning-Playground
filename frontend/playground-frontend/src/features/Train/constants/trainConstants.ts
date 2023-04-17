import {
  ALL_STEPS,
  DATA_SOURCE,
  TABULAR_STEPS,
} from "@/features/Train/types/trainTypes";

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
  "UPLOAD",
  "PARAMETERS",
  "REVIEW",
  "TRAIN",
] as const;

export const DATA_SOURCE_SETTINGS: {
  [T in DATA_SOURCE]: { name: string; steps: readonly string[] };
} = {
  TABULAR: {
    name: "Tabular",
    steps: TABULAR_STEPS_ARR,
  },
  PRETRAINED: {
    name: "Pretrained",
    steps: [],
  },
  IMAGE: {
    name: "Image",
    steps: [],
  },
  AUDIO: {
    name: "Audio",
    steps: [],
  },
  TEXTUAL: {
    name: "Textual",
    steps: [],
  },
  CLASSICAL_ML: {
    name: "Classical ML",
    steps: [],
  },
  OBJECT_DETECTION: {
    name: "Object Detection",
    steps: [],
  },
};

export const STEPS_SETTINGS: {
  [T in ALL_STEPS]: { name: string; optional: boolean };
} = {
  UPLOAD: { name: "Upload", optional: false },
  PARAMETERS: { name: "Parameters", optional: false },
  REVIEW: { name: "Review", optional: false },
  TRAIN: { name: "Train", optional: false },
};
