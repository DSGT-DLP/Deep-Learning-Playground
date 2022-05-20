export const POSSIBLE_LAYERS = [
  {
    display_name: "Linear",
    object_name: "nn.linear",
    parameters: [
      { display_name: "Input size" },
      { display_name: "Output size" },
    ],
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

export const CRITERIONS = [{ label: "CELOSS", value: 0 }];

export const PROBLEM_TYPES = [
  { label: "Classification", value: 0 },
  { label: "Regression", value: 1 },
];

export const DEFAULT_OPTIONS = [
  { label: "False", value: 0 },
  { label: "True", value: 1 },
];
export const OPTIMIZER_NAMES = [{ label: "SGD", value: 0 }];
