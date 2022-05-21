export const POSSIBLE_LAYERS = [
  {
    display_name: "Linear",
    object_name: "nn.linear",
    parameters: {
      inputSize: { index: 0, parameter_name: "Input size" },
      outputSize: { index: 1, parameter_name: "Output size" },
    },
  },
  {
    display_name: "ReLU",
    object_name: "nn.ReLU",
    parameters: [],
  },
  {
    display_name: "Softmax",
    object_name: "nn.Softmax",
    parameters: [],
  },
];

export const CRITERIONS = [
  {
    label: "L1LOSS",
    value: 0,
    object_name: "nn.L1Loss()",
  },
  {
    label: "MSELOSS",
    value: 1,
    object_name: "nn.MSELoss()",
  },
  {
    label: "BCELOSS",
    value: 2,
    object_name: "nn.BCELoss()",
  },
  {
    label: "CELOSS",
    value: 3,
    object_name: "nn.CrossEntropyLoss(reduction='mean')",
  },
];

export const PROBLEM_TYPES = [
  { label: "Classification", value: 0 },
  { label: "Regression", value: 1 },
];

export const BOOL_OPTIONS = [
  { label: "False", value: false },
  { label: "True", value: true },
];
export const OPTIMIZER_NAMES = [
  { label: "SGD", value: 0 },
  { label: "Adam", value: 1 },
];
