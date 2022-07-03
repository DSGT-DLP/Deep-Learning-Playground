export const COLORS = {
  input: "#E2E2E2",
  layer: "#CD7BFF",
  addLayer: "#F2ECFF",
  disabled: "#6c757d",
  background: "#F6F6FF",
  dark_blue: "#003058",  // primary
  gold: "#B3A36A",  // secondary
};

export const LAYOUT = {
  row: {
    display: "flex",
    flexDirection: "row",
  },
  column: {
    display: "flex",
    flexDirection: "column",
  },
  centerMiddle: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  },
};

export const GENERAL_STYLES = {
  p: {
    fontFamily: "Arial, Helvetica, sans-serif",
    fontWeight: "bold",
  },
};

export const ITEM_TYPES = {
  NEW_LAYER: "new_layer",
};

export const DEFAULT_ADDED_LAYERS = [
  {
    display_name: "Linear",
    object_name: "nn.Linear",
    parameters: {
      inputSize: { index: 0, parameter_name: "Input size", value: 4 },
      outputSize: { index: 1, parameter_name: "Output size", value: 10 },
    },
  },
  {
    display_name: "ReLU",
    object_name: "nn.ReLU",
    parameters: {},
  },
  {
    display_name: "Linear",
    object_name: "nn.Linear",
    parameters: {
      inputSize: { index: 0, parameter_name: "Input size", value: 10 },
      outputSize: { index: 1, parameter_name: "Output size", value: 3 },
    },
  },
  {
    display_name: "Softmax",
    object_name: "nn.Softmax",
    parameters: {
      inputSize: { index: 0, parameter_name: "dim", value: -1 },
    },
  },
];
