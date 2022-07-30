export const COLORS = {
  input: "#E2E2E2",
  layer: "#CD7BFF",
  addLayer: "#F2ECFF",
  disabled: "#6c757d",
  background: "#F6F6FF",
  dark_blue: "#003058", // primary
  gold: "#B3A36A", // secondary
  visited: "#808080",
};

export const URLs = {
  linkedin: "https://www.linkedin.com/company/dsgt/",
  youtube: "https://www.youtube.com/channel/UC1DGL6Tb9ffwC-aHChadxMw",
  instagram: "https://www.instagram.com/datasciencegt/",
  github: "https://github.com/karkir0003/Deep-Learning-Playground",
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
  error_text: {
    color: "red",
    fontSize: "0.8rem",
    marginLeft: 3,
    marginTop: 5,
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

export const DEFAULT_TRANSFORMS = [
  {
    display_name: "To Tensor",
    object_name: "transforms.ToTensor",
    parameters: {},
    label: "To Tensor",
    value: "TT",
  },
  {
    display_name: "Resize",
    object_name: "transforms.Resize",
    parameters: {
      size: { index: 0, parameter_name: "(H, W)" , value: "(32, 32)"},
    },
    label: "Resize",
    value: "R",
  },
];

export const DEFAULT_IMG_LAYERS = [
  {
    display_name: "Conv2D",
    object_name: "nn.Conv2d",
    parameters: {
      in_channels: {
        index: 0,
        parameter_name: "in channels",
        value: 3,
      },
      out_channels: {
        index: 1,
        parameter_name: "out channels",
        value: 5,
      },
      kernel_size: {
        index: 2,
        parameter_name: "kernel_size",
        value: 3,
      },
      stride: {
        index: 3,
        parameter_name: "stride",
        value: 1,
      },
      padding: {
        index: 4,
        parameter_name: "padding",
        value: 1,
      },
    },
  },
  {
    display_name: "MaxPool2d",
    object_name: "nn.MaxPool2d",
    parameters: {
      kernel_size: { index: 0, parameter_name: "Kernel size", value: 3 },
    },
  },
  {
    display_name: "Flatten",
    object_name: "nn.Flatten",
    parameters: {
      start_dim: { index: 0, parameter_name: "start dim", value: 1 },
      end_dim: { index: 1, parameter_name: "end dim", value: -1 },
    },
  },
  {
    display_name: "Linear",
    object_name: "nn.Linear",
    parameters: {
      inputSize: { index: 0, parameter_name: "Input size", value: 10 * 10 * 5 },
      outputSize: { index: 1, parameter_name: "Output size", value: 2 },
    },
  },
  {
    display_name: "Sigmoid",
    object_name: "nn.Sigmoid",
    parameters: {},
  },
];
