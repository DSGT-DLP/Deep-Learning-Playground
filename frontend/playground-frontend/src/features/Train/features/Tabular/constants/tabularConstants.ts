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
    component: TabularParametersStep,
    problemTypes: [
      { label: "Classification", value: "CLASSIFICATION" },
      { label: "Regression", value: "REGRESSION" },
    ],
    criterions: [
      {
        label: "L1 (Absolute Error) Loss",
        value: "L1LOSS",
        objectName: "nn.L1Loss()",
        problemType: "REGRESSION",
      },
      {
        label: "Mean Squared Error Loss",
        value: "MSELOSS",
        objectName: "nn.MSELoss()",
        problemType: "REGRESSION",
      },
      {
        label: "Binary Cross-Entropy Loss",
        value: "BCELOSS",
        objectName: "nn.BCELoss()",
        problemType: "CLASSIFICATION",
      },
      {
        label: "Cross-Entropy Loss",
        value: "CELOSS",
        objectName: "nn.CrossEntropyLoss(reduction='mean')",
        problemType: "CLASSIFICATION",
      },
    ],
    optimizers: [
      { label: "Stochastic Gradient Descent", value: "SGD" },
      { label: "Adam Optimization", value: "Adam" },
    ],
    layers: {
      LINEAR: {
        label: "Linear",
        objectName: "nn.Linear",
        parameters: [
          {
            label: "Input Size",
            min: 1,
            max: 1600,
            required: true,
            type: "number",
          },
          {
            label: "Output Size",
            min: 1,
            max: 1600,
            required: true,
            type: "number",
          },
        ],
      },
      RELU: {
        label: "ReLU",
        objectName: "nn.ReLU",
        parameters: [],
      },
      TANH: {
        label: "Tanh",
        objectName: "nn.Tanh",
        parameters: [],
      },
      SOFTMAX: {
        label: "Softmax",
        objectName: "nn.Softmax",
        parameters: [
          {
            label: "Dimension",
            min: -3,
            max: 2,
            required: true,
            type: "number",
          },
        ],
      },
    },
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
