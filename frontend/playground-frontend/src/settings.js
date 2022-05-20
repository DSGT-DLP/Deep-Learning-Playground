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
