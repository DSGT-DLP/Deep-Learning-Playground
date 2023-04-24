import TabularDatasetStep from "../components/TabularDatasetStep";
import TabularParametersStep from "../components/TabularParametersStep";
import TabularTrainspace from "../components/TabularTrainspace";

export const TRAINSPACE_SETTINGS = {
  name: "Tabular",
  steps: ["DATASET", "PARAMETERS", "REVIEW", "TRAIN"],
  component: TabularTrainspace,
} as const;

export const STEP_SETTINGS = {
  DATASET: {
    name: "Dataset",
    optional: false,
    component: TabularDatasetStep,
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
    problemTypes: ["CLASSIFICATION", "REGRESSION"],
    component: TabularParametersStep,
  },
  REVIEW: {
    name: "Review",
    optional: false,
    component: TabularDatasetStep,
  },
  TRAIN: {
    name: "Train",
    optional: false,
    component: TabularDatasetStep,
  },
} as const;
