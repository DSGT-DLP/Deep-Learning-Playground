export const POSSIBLE_LAYERS = [
  {
    display_name: "Linear",
    object_name: "nn.Linear",
    parameters: {
      inputSize: { index: 0, parameter_name: "Input size" },
      outputSize: { index: 1, parameter_name: "Output size" },
    },
  },
  {
    display_name: "ReLU",
    object_name: "nn.ReLU",
    parameters: {},
  },
  {
    display_name: "Softmax",
    object_name: "nn.Softmax",
    parameters: {
      inputSize: { index: 0, parameter_name: "dim" },
    },
  },
];

export const CRITERIONS = [
  {
    label: "L1LOSS",
    value: "L1LOSS",
    object_name: "nn.L1Loss()",
  },
  {
    label: "MSELOSS",
    value: "MSELOSS",
    object_name: "nn.MSELoss()",
  },
  {
    label: "BCELOSS",
    value: "BCELOSS",
    object_name: "nn.BCELoss()",
  },
  {
    label: "CELOSS",
    value: "CELOSS",
    object_name: "nn.CrossEntropyLoss(reduction='mean')",
  },
];

export const PROBLEM_TYPES = [
  { label: "Classification", value: "classification" },
  { label: "Regression", value: "regression" },
];

export const BOOL_OPTIONS = [
  { label: "False", value: false },
  { label: "True", value: true },
];
export const OPTIMIZER_NAMES = [
  { label: "SGD", value: "SGD" },
  { label: "Adam", value: "Adam" },
];
