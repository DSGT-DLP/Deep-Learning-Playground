export const COLORS = Object.freeze({
  input: "#E2E2E2",
  layer: "#CD7BFF",
  addLayer: "#F2ECFF",
  disabled: "#6c757d",
  background: "#F6F6FF",
  dark_blue: "#003058", // primary
  gold: "#B3A36A", // secondary
  visited: "#808080",
});
``
export const URLs = Object.freeze({
  donate: "https://buy.stripe.com/9AQ3e4eO81X57y8aEG",
  linkedin: "https://www.linkedin.com/company/dsgt/",
  youtube: "https://www.youtube.com/channel/UC1DGL6Tb9ffwC-aHChadxMw",
  instagram: "https://www.instagram.com/datasciencegt/",
  github: "https://github.com/DSGT-DLP/Deep-Learning-Playground",
});

export const EXPECTED_FAILURE_HTTP_CODES = Object.freeze([
  400, 401, 403, 404, 405, 500,
]);

export const ROUTE_DICT = Object.freeze({
  tabular: "tabular-run",
  image: "img-run",
  pretrained: "pretrain-run",
  classicalml: "ml-run",
  objectdetection: "object-detection",
});

export const LAYOUT = Object.freeze({
  row: {
    display: "flex",
    flexDirection: "row",
  } as React.CSSProperties,
  column: {
    display: "flex",
    flexDirection: "column",
  } as React.CSSProperties,
  centerMiddle: {
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
  } as React.CSSProperties,
});

export const GENERAL_STYLES = Object.freeze({
  p: {
    fontWeight: "bold",
  },
  error_text: {
    color: "red",
    fontSize: "0.8rem",
    marginLeft: 3,
    marginTop: 5,
  },
});

export const ITEM_TYPES = Object.freeze({
  NEW_LAYER: "new_layer",
});

export const DEFAULT_ADDED_LAYERS = Object.freeze([
  {
    display_name: "Linear",
    object_name: "nn.Linear",
    parameters: {
      inputSize: {
        index: 0,
        parameter_name: "Input size",
        value: 4,
        min: 1,
        max: 1600,
        parameter_type: "number",
      },
      outputSize: {
        index: 1,
        parameter_name: "Output size",
        value: 10,
        min: 1,
        max: 1600,
        parameter_type: "number",
      },
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
      inputSize: {
        index: 0,
        parameter_name: "Input size",
        value: 10,
        min: 1,
        max: 1600,
        parameter_type: "number",
      },
      outputSize: {
        index: 1,
        parameter_name: "Output size",
        value: 3,
        min: 1,
        max: 1600,
        parameter_type: "number",
      },
    },
  },
  {
    display_name: "Softmax",
    object_name: "nn.Softmax",
    parameters: {
      inputSize: {
        index: 0,
        parameter_name: "dim",
        value: -1,
        min: -3,
        max: 2,
        parameter_type: "number",
      },
    },
  },
]);

export const DEFAULT_TRANSFORMS = Object.freeze([
  {
    display_name: "Grayscale",
    object_name: "transforms.Grayscale",
    parameters: {},
    label: "Grayscale",
    value: "Grayscale",
  },
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
      height: {
        index: 0,
        parameter_name: "Height",
        value: 32,
        min: 1,
        max: 1000,
        parameter_type: "number",
      },
      width: {
        index: 1,
        parameter_name: "Width",
        value: 32,
        min: 1,
        max: 1000,
        parameter_type: "number",
      },
    label: "Resize",
    value: "R",
    },
  },
]);

export const DEFAULT_IMG_LAYERS = Object.freeze([
  {
    display_name: "Conv2D",
    object_name: "nn.Conv2d",
    parameters: {
      in_channels: {
        index: 0,
        parameter_name: "in channels",
        value: 1,
        min: 1,
        max: 16,
        parameter_type: "number",
      },
      out_channels: {
        index: 1,
        parameter_name: "out channels",
        value: 5,
        min: 1,
        max: 16,
        parameter_type: "number",
      },
      kernel_size: {
        index: 2,
        parameter_name: "kernel_size",
        value: 3,
        min: 1,
        max: 1000,
        parameter_type: "number",
      },
      stride: {
        index: 3,
        parameter_name: "stride",
        value: 1,
        min: 1,
        max: 1000,
        parameter_type: "number",
      },
      padding: {
        index: 4,
        parameter_name: "padding",
        value: 1,
        min: 1,
        max: 1000,
        parameter_type: "number",
      },
    },
  },
  {
    display_name: "MaxPool2d",
    object_name: "nn.MaxPool2d",
    parameters: {
      kernel_size: {
        index: 0,
        parameter_name: "Kernel size",
        value: 3,
        min: 1,
        max: 1000,
        parameter_type: "number",
      },
    },
  },
  {
    display_name: "Flatten",
    object_name: "nn.Flatten",
    parameters: {
      start_dim: {
        index: 0,
        parameter_name: "start dim",
        value: 1,
        min: -4,
        max: 3,
        parameter_type: "number",
      },
      end_dim: {
        index: 1,
        parameter_name: "end dim",
        value: -1,
        min: -4,
        max: 3,
        parameter_type: "number",
      },
    },
  },
  {
    display_name: "Linear",
    object_name: "nn.Linear",
    parameters: {
      inputSize: {
        index: 0,
        parameter_name: "Input size",
        value: 10 * 10 * 5,
        min: 1,
        max: 1600,
        parameter_type: "number",
      },
      outputSize: {
        index: 1,
        parameter_name: "Output size",
        value: 10,
        min: 1,
        max: 1600,
        parameter_type: "number",
      },
    },
  },
  {
    display_name: "Sigmoid",
    object_name: "nn.Sigmoid",
    parameters: {},
  },
]);

export const DEFAULT_DETECTION_TRANSFORMS = Object.freeze([
  {
    display_name: "Grayscale",
    object_name: "transforms.Grayscale",
    parameters: {},
    label: "Grayscale",
    value: "Grayscale",
  },
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
      height: {
        index: 0,
        parameter_name: "Height",
        value: 32,
        min: 1,
        max: 1000,
        parameter_type: "number",
      },
      width: {
        index: 1,
        parameter_name: "Width",
        value: 32,
        min: 1,
        max: 1000,
        parameter_type: "number",
      },
    },
    label: "Resize",
    value: "R",
  },
]);
