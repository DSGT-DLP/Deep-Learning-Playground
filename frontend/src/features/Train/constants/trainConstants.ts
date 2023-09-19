import {
  TABULAR_STEP_SETTINGS,
  TABULAR_TRAINSPACE_SETTINGS,
} from "../features/Tabular";
import {
  IMAGE_STEP_SETTINGS,
  IMAGE_TRAINSPACE_SETTINGS,
} from "../features/Image";

export const DATA_SOURCE_ARR = [
  "TABULAR",
  "PRETRAINED",
  "IMAGE",
  "AUDIO",
  "TEXTUAL",
  "CLASSICAL_ML",
  "OBJECT_DETECTION",
] as const;

export const ALL_TRAINSPACE_SETTINGS = {
  TABULAR: TABULAR_TRAINSPACE_SETTINGS,
  IMAGE: IMAGE_TRAINSPACE_SETTINGS,
};
export const IMPLEMENTED_DATA_SOURCE_ARR = Object.keys(
  ALL_TRAINSPACE_SETTINGS
) as (keyof typeof ALL_TRAINSPACE_SETTINGS)[];

export const ALL_STEP_SETTINGS = {
  TABULAR: TABULAR_STEP_SETTINGS,
  PRETRAINED: TABULAR_STEP_SETTINGS,
  IMAGE: IMAGE_STEP_SETTINGS,
  AUDIO: TABULAR_STEP_SETTINGS,
  TEXTUAL: TABULAR_STEP_SETTINGS,
  CLASSICAL_ML: TABULAR_STEP_SETTINGS,
  OBJECT_DETECTION: TABULAR_STEP_SETTINGS,
};
