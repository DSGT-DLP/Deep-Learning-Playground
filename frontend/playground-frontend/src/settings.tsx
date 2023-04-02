import React from "react";

type LTypes = "number" | "text" | "tuple" | "boolean";
export interface LayerParameter<T extends LTypes = LTypes> {
  index: number;
  parameter_name: string;
  min: T extends "number" | "tuple" ? number : null;
  max: T extends "number" | "tuple" ? number : null;
  parameter_type: T;
  default?: T extends "number" ? number : string;
  kwarg?: string;
  value?: T extends "number" ? number : string;
}

export interface ModelLayer {
  display_name: string;
  object_name: string;
  parameters: { [key: string]: LayerParameter };
  tooltip_info?: JSX.Element;
}

export const POSSIBLE_LAYERS: ModelLayer[] = [
  {
    display_name: "Linear",
    object_name: "nn.Linear",
    parameters: {
      inputSize: {
        index: 0,
        parameter_name: "Input size",
        min: 1,
        max: 1600,
        parameter_type: "number",
      } as LayerParameter<"number">,
      outputSize: {
        index: 1,
        parameter_name: "Output size",
        min: 1,
        max: 1600,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Applies a linear transformation to the incoming data:{" "}
            <i>
              y = xA
              <sup>T</sup> + b
            </i>
            .
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>in_features</em> - size of each input sample
          </li>
          <li>
            <em>out_features</em> - size of each output sample
          </li>
        </ul>

        <a href="https://pytorch.org/docs/stable/generated/torch.nn.Linear.html">
          More info
        </a>
      </>
    ),
  },
  {
    display_name: "ReLU",
    object_name: "nn.ReLU",
    parameters: {},
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Applies the rectified linear unit function element-wise: ReLU
            <i>
              (x) = (x)<sup>+</sup> ={" "}
            </i>
            max(0, <i>x)</i>.
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <p className="info">
          <i>None</i>
        </p>
        <a href="https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html">
          More info
        </a>
      </>
    ),
  },
  {
    display_name: "Softmax",
    object_name: "nn.Softmax",
    parameters: {
      inputSize: {
        index: 0,
        parameter_name: "dim",
        min: -3,
        max: 2,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Applies the Softmax function to an <i>n</i>-dimensional input Tensor
            rescaling them so that the elements of the <i>n</i>-dimensional
            output Tensor lie in the range [0,1] and sum to 1.
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>dim (int)</em> - A dimension along which Softmax will be
            computed (so every slice along dim will sum to 1).
          </li>
        </ul>
        <a href="https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html">
          More info
        </a>
      </>
    ),
  },
  {
    display_name: "Sigmoid",
    object_name: "nn.Sigmoid",
    parameters: {},
    tooltip_info: (
      <>
        <p className="info">
          <strong>Applies the Sigmoid function.</strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <p className="info">
          <i className="info">None</i>
        </p>
        <a href="https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html">
          More info
        </a>
      </>
    ),
  },
  {
    display_name: "Tanh",
    object_name: "nn.Tanh",
    parameters: {},
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Applies the Hyperbolic Tangent (Tanh) function element-wise.
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <p className="info">
          <i>None</i>
        </p>
        <a href="https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html">
          More info
        </a>
      </>
    ),
  },
  {
    display_name: "LogSoftmax",
    object_name: "nn.LogSoftmax",
    parameters: {
      inputSize: {
        index: 0,
        parameter_name: "dim",
        min: -3,
        max: 2,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Applies the log(Softmax(<i>x</i>)) function to an <i>n</i>
            -dimensional input Tensor.
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>dim (int)</em> - A dimension along which LogSoftmax will be
            computed.
          </li>
        </ul>
        <a href="https://pytorch.org/docs/stable/generated/torch.nn.LogSoftmax.html">
          More info
        </a>
      </>
    ),
  },
];

export const ML_MODELS: ModelLayer[] = [
  {
    display_name: "Gaussian Naive Bayes",
    object_name: "sklearn.naive_bayes.GaussianNB",
    parameters: {},
  },
  {
    display_name: "RF Classifier",
    object_name: "sklearn.ensemble.RandomForestClassifier",
    parameters: {
      n_estimators: {
        index: 0,
        parameter_name: "Number of Estimators",
        kwarg: "n_estimators = ",
        default: 100,
        min: 1,
        max: 200,
        parameter_type: "number",
      } as LayerParameter<"number">,
      max_depth: {
        index: 1,
        parameter_name: "Max Depth",
        kwarg: "max_depth = ",
        default: 5,
        min: 1,
        max: 100,
        parameter_type: "number",
      } as LayerParameter<"number">,
      min_samples_split: {
        index: 2,
        parameter_name: "Minimum Samples Split",
        kwarg: "min_samples_split = ",
        default: 2,
        min: 1,
        max: 10,
        parameter_type: "number",
      } as LayerParameter<"number">,
      max_features: {
        index: 3,
        parameter_name: "Max Features",
        kwarg: "max_features = ",
        default: "sqrt",
      } as LayerParameter<"text">,
    },
  },
  {
    display_name: "RF Regressor",
    object_name: "sklearn.ensemble.RandomForestRegressor",
    parameters: {
      n_estimators: {
        index: 0,
        parameter_name: "Number of Estimators",
        kwarg: "n_estimators = ",
        default: 100,
        min: 1,
        max: 200,
        parameter_type: "number",
      } as LayerParameter<"number">,
      max_depth: {
        index: 1,
        parameter_name: "Max Depth",
        kwarg: "max_depth = ",
        default: 5,
        min: 1,
        max: 5,
        parameter_type: "number",
      } as LayerParameter<"number">,
      min_samples_split: {
        index: 2,
        parameter_name: "Minimum Samples Split",
        kwarg: "min_samples_split = ",
        default: 2,
        min: 2,
        max: 10,
        parameter_type: "number",
      } as LayerParameter<"number">,
      max_features: {
        index: 3,
        parameter_name: "Max Features",
        kwarg: "max_features = ",
        default: "sqrt",
        parameter_type: "text",
      } as LayerParameter<"text">,
    },
  },
  {
    display_name: "Logistic Regression",
    object_name: "sklearn.linear_model.LogisticRegression",
    parameters: {
      fit_intercept: {
        index: 0,
        parameter_name: "Intercept",
        kwarg: "fit_intercept = ",
        default: 1,
        min: 0,
        max: 1,
        parameter_type: "number",
      } as LayerParameter<"number">,
      C: {
        index: 1,
        parameter_name: "Regularization Strength (C)",
        kwarg: "C = ",
        default: 1,
        min: 0,
        max: 10,
        parameter_type: "number",
      } as LayerParameter<"number">,
      penslty: {
        index: 2,
        parameter_name: "Penalty",
        default: "l2",
        kwarg: "penalty = ",
        parameter_type: "text",
      } as LayerParameter<"text">,
    },
  },
  {
    display_name: "Linear  Regression",
    object_name: "sklearn.linear_model.LinearRegression",
    parameters: {
      fit_intercept: {
        index: 0,
        parameter_name: "Intercept",
        kwarg: "fit_intercept = ",
        min: 0,
        max: 1,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
  },
  {
    display_name: "Decision Tree Classifier",
    object_name: "sklearn.tree.DecisionTreeClassifier",
    parameters: {
      max_depth: {
        index: 0,
        parameter_name: "Max Depth",
        kwarg: "max_depth = ",
        default: 5,
        min: 1,
        max: 100,
        parameter_type: "number",
      } as LayerParameter<"number">,
      min_samples_split: {
        index: 1,
        parameter_name: "Minimum Sample Splits",
        kwarg: "min_samples_split = ",
        default: 2,
        min: 2,
        max: 10,
        parameter_type: "number",
      } as LayerParameter<"number">,
      max_features: {
        index: 2,
        parameter_name: "Max Features",
        kwarg: "max_features = ",
        default: "sqrt",
        parameter_type: "text",
      } as LayerParameter<"text">,
    },
  },
  {
    display_name: "Decision Tree Regressor",
    object_name: "sklearn.tree.DecisionTreeRegressor",
    parameters: {
      max_depth: {
        index: 0,
        parameter_name: "Max Depth",
        kwarg: "max_depth = ",
        default: 5,
        min: 1,
        max: 5,
        parameter_type: "number",
      } as LayerParameter<"number">,
      min_samples_split: {
        index: 1,
        parameter_name: "Minimum Sample Splits",
        kwarg: "min_samples_split = ",
        default: 2,
        min: 1,
        max: 10,
        parameter_type: "number",
      } as LayerParameter<"number">,
      max_features: {
        index: 2,
        parameter_name: "Max Features",
        kwarg: "max_features = ",
        default: "sqrt",
        parameter_type: "text",
      } as LayerParameter<"text">,
    },
  },
];

export const IMAGE_LAYERS: ModelLayer[] = [
  {
    display_name: "Conv2D",
    object_name: "nn.Conv2d",
    parameters: {
      in_channels: {
        index: 0,
        parameter_name: "in channels",
        min: 1,
        max: 16,
        parameter_type: "number",
      } as LayerParameter<"number">,
      out_channels: {
        index: 1,
        parameter_name: "out channels",
        min: 1,
        max: 16,
        parameter_type: "number",
      } as LayerParameter<"number">,
      kernel_size: {
        index: 2,
        parameter_name: "kernel size",
        min: 1,
        max: 1000,
        parameter_type: "number",
      } as LayerParameter<"number">,
      stride: {
        index: 3,
        parameter_name: "stride",
        min: 1,
        max: 1000,
        parameter_type: "number",
      } as LayerParameter<"number">,
      padding: {
        index: 4,
        parameter_name: "padding",
        min: 1,
        max: 1000,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Applies a 2D convolution over an input signal composed of several
            input planes.
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>in channels (int)</em> Number of channels in the input image. (3
            for RGB, 1 for grayscale)
          </li>
          <li>
            <em>out_channels (int)</em> Number of channels produced by the
            convolution
          </li>
          <li>
            <em>kernel_size (int or tuple)</em> Size of convolving tuple
          </li>
        </ul>
        <a href="https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html">
          More info
        </a>
      </>
    ),
  },
  {
    display_name: "BatchNorm2D",
    object_name: "nn.BatchNorm2d",
    parameters: {
      num_features: {
        index: 0,
        parameter_name: "num features",
        min: 1,
        max: 16,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Applies Batch Normalization over a 4D input (a mini-batch of 2D
            inputs with additional channel dimension) as described in the paper{" "}
            <a href="https://arxiv.org/abs/1502.03167">
              Batch Normalization: Accelerating Deep Network Training by
              Reducing Internal Covariate Shift.
            </a>
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>num features (int)</em> C from an expected input of size (N, C,
            H, W)
          </li>
        </ul>
        <a href="https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html">
          More info
        </a>
      </>
    ),
  },
  {
    display_name: "MaxPool2d",
    object_name: "nn.MaxPool2d",
    parameters: {
      kernel_size: {
        index: 0,
        parameter_name: "Kernel size",
        min: 1,
        max: 1000,
        parameter_type: "number",
      } as LayerParameter<"number">,
      stride: {
        index: 1,
        parameter_name: "stride",
        min: 1,
        max: 1000,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Applies a 2D max pooling over an input signal composed of several
            input planes.
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>Kernel Size (int)</em> - the size of the window to take a max
            over
          </li>
        </ul>

        <a href="https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html">
          More info
        </a>
      </>
    ),
  },
  {
    display_name: "AdaptAvg Pool2d",
    object_name: "nn.AdaptiveAvgPool2d",
    parameters: {
      output_size: {
        index: 0,
        parameter_name: "Output size",
        min: 1,
        max: 16,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Applies a 2D adaptive average pooling over an input signal composed
            of several input planes.
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>Output Size (int or tuple)</em> - the target output size of the
            image of the form H x W. Can be a tuple (H, W) or a single H for a
            square image H x H. H and W can be either a int, or None which means
            the size will be the same as that of the input.
          </li>
        </ul>

        <a href="https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html">
          More info
        </a>
      </>
    ),
  },
  {
    display_name: "Dropout",
    object_name: "nn.Dropout",
    parameters: {
      p: {
        index: 0,
        parameter_name: "Probability",
        min: 0,
        max: 1,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            During training, randomly zeroes some of the elements of the input
            tensor with probability p using samples from a Bernoulli
            distribution. Each channel will be zeroed out independently on every
            forward call.
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>Probability (float)</em> - probability of an element to be
            zeroed. Default: 0.5
          </li>
        </ul>

        <a href="https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html">
          More info
        </a>
      </>
    ),
  },

  {
    display_name: "Flatten",
    object_name: "nn.Flatten",
    parameters: {
      start_dim: {
        index: 0,
        parameter_name: "start dim",
        min: -4,
        max: 3,
        parameter_type: "number",
      } as LayerParameter<"number">,
      end_dim: {
        index: 1,
        parameter_name: "end dim",
        min: -4,
        max: 3,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>Flattens a contiguous range of dims into a tensor.</strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>Start Dim (int)</em> - First dimension to flatten Default: 1
          </li>
          <li>
            <em>End Dim (int)</em> - Last dimension to flatten Default: -1
          </li>
        </ul>

        <a href="https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html">
          More info
        </a>
      </>
    ),
  },
];

interface PossibleTransform extends ModelLayer {
  label: string;
  value: string;
}

export const POSSIBLE_TRANSFORMS: PossibleTransform[] = [
  {
    display_name: "Random Horizontal Flip",
    object_name: "transforms.RandomHorizontalFlip",
    parameters: {
      probability: {
        index: 0,
        parameter_name: "prob",
        min: 0,
        max: 1,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    label: "Random Horizontal Flip",
    value: "RandomHorizontalFlip",
  },

  {
    display_name: "Random Vertical Flip",
    object_name: "transforms.RandomVerticalFlip",
    parameters: {
      p: {
        index: 0,
        parameter_name: "prob",
        min: 0,
        max: 1,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    label: "Random Vertical Flip",
    value: "RandomVerticalFlip",
  },

  {
    display_name: "To Tensor",
    object_name: "transforms.ToTensor",
    parameters: {},
    label: "To Tensor",
    value: "ToTensor",
  },
  {
    display_name: "Resize",
    object_name: "transforms.Resize",
    parameters: {
      size: {
        index: 0,
        parameter_name: "(H, W)",
        min: 1,
        max: 1000,
        default: "(32, 32)",
        parameter_type: "tuple",
      } as LayerParameter<"tuple">,
    },
    label: "Resize",
    value: "Resize",
  },
  {
    display_name: "Gaussian Blur",
    object_name: "transforms.GaussianBlur",
    parameters: {
      kernel_size: {
        index: 0,
        parameter_name: "kernel size",
        min: 1,
        max: 1000,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    label: "Gaussian Blur",
    value: "GaussianBlur",
  },
  {
    display_name: "Grayscale",
    object_name: "transforms.Grayscale",
    parameters: {},
    label: "Grayscale",
    value: "Grayscale",
  },
  {
    display_name: "Normalize",
    object_name: "transforms.Normalize",
    parameters: {
      mean: {
        index: 0,
        parameter_name: "mean",
        min: -1000,
        max: 1000,
        default: 0,
        parameter_type: "number",
      } as LayerParameter<"number">,
      std: {
        index: 1,
        parameter_name: "std",
        min: -1000,
        max: 1000,
        default: 1,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    label: "Normalize",
    value: "Normalize",
  },
];

interface DetectionTransform extends PossibleTransform {
  transform_type?: string;
}

export const DETECTION_TRANSFORMS: DetectionTransform[] = [
  {
    display_name: "Random Horizontal Flip",
    object_name: "transforms.RandomHorizontalFlip",
    parameters: {
      probability: {
        index: 0,
        parameter_name: "prob",
        min: 0,
        max: 1,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Horizontally flip the given image randomly with a given probability.
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>prob</em> - probability of the flip
          </li>
        </ul>

        <a href="https://pytorch.org/vision/main/generated/torchvision.transforms.RandomHorizontalFlip.html">
          More info
        </a>
      </>
    ),
    label: "Random Horizontal Flip",
    value: "RandomHorizontalFlip",
  },

  {
    display_name: "Random Vertical Flip",
    object_name: "transforms.RandomVerticalFlip",
    parameters: {
      p: {
        index: 0,
        parameter_name: "prob",
        min: 0,
        max: 1,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Vertically flip the given image randomly with a given probability.
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>prob</em> - probability of the flip
          </li>
        </ul>

        <a href="https://pytorch.org/vision/main/generated/torchvision.transforms.RandomVerticalFlip.html">
          More info
        </a>
      </>
    ),
    label: "Random Vertical Flip",
    value: "RandomVerticalFlip",
  },

  {
    display_name: "To Tensor",
    object_name: "transforms.ToTensor",
    parameters: {},
    tooltip_info: (
      <>
        <p className="info">
          <strong>Convert PIL Image or numpy.ndarray to tensor.</strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <p className="info">
          <i>None</i>
        </p>
        <a href="https://pytorch.org/vision/main/generated/torchvision.transforms.ToTensor.html">
          More info
        </a>
      </>
    ),
    label: "To Tensor",
    value: "ToTensor",
  },
  {
    display_name: "Resize",
    object_name: "transforms.Resize",
    parameters: {
      size: {
        index: 0,
        parameter_name: "(H, W)",
        min: 1,
        max: 1000,
        default: "(32, 32)",
        parameter_type: "tuple",
      } as LayerParameter<"tuple">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>Resize the input image to the given size.</strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>(H, W)</em> - output size
          </li>
        </ul>

        <a href="https://pytorch.org/vision/main/generated/torchvision.transforms.functional.resize.html">
          More info
        </a>
      </>
    ),
    label: "Resize",
    value: "Resize",
  },
  {
    display_name: "Gaussian Blur",
    object_name: "transforms.GaussianBlur",
    parameters: {
      kernel_size: {
        index: 0,
        parameter_name: "kernel size",
        min: 1,
        max: 1000,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>Blurs image with randomly chosen GaussianBlur.</strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>kernel size</em> - size of Gaussian kernel
          </li>
        </ul>
        <a href="https://pytorch.org/vision/main/generated/torchvision.transforms.GaussianBlur.html">
          More info
        </a>
      </>
    ),
    label: "Gaussian Blur",
    value: "GaussianBlur",
  },
  {
    display_name: "Grayscale",
    object_name: "transforms.Grayscale",
    parameters: {},
    tooltip_info: (
      <>
        <p className="info">
          <strong>Convert image to grayscale.</strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <p className="info">
          <i>None</i>
        </p>
        <a href="https://pytorch.org/vision/main/generated/torchvision.transforms.Grayscale.html">
          More info
        </a>
      </>
    ),
    label: "Grayscale",
    value: "Grayscale",
  },
  {
    display_name: "Normalize",
    object_name: "transforms.Normalize",
    parameters: {
      mean: {
        index: 0,
        parameter_name: "mean",
        min: -1000,
        max: 1000,
        default: 0,
        parameter_type: "number",
      } as LayerParameter<"number">,
      std: {
        index: 1,
        parameter_name: "std",
        min: -1000,
        max: 1000,
        default: 1,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>Normalize image with mean and standard deviation.</strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>mean</em> - mean of normalization
          </li>
          <li>
            <em>mean</em> - standard deviation of normalization
          </li>
        </ul>

        <a href="https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html">
          More info
        </a>
      </>
    ),
    label: "Normalize",
    value: "Normalize",
  },
  {
    display_name: "AdjustContrast",
    object_name: "transforms.functional.adjust_contrast",
    transform_type: "functional",
    parameters: {
      contrast_factor: {
        index: 0,
        parameter_name: "contrast_factor",
        min: 0,
        max: 10000,
        default: 0,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>Adjust contrast of an image.</strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>contrast_factor</em> - how much to adjust the contrast
          </li>
        </ul>
        <a href="https://pytorch.org/vision/main/generated/torchvision.transforms.functional.adjust_contrast.html">
          More info
        </a>
      </>
    ),
    label: "AdjustContrast",
    value: "AdjustContrast",
  },
  {
    display_name: "AdjustBrightness",
    object_name: "transforms.functional.adjust_brightness",
    transform_type: "functional",
    parameters: {
      brightness_factor: {
        index: 0,
        parameter_name: "brightness_factor",
        min: 0,
        max: 10000,
        default: 0,
        parameter_type: "number",
      } as LayerParameter<"number">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>Adjust brightness of an image.</strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>brightness_factor</em> - how much to adjust the brightness
          </li>
        </ul>

        <a href="https://pytorch.org/vision/main/generated/torchvision.transforms.function.adjust_brightness.html">
          More info
        </a>
      </>
    ),
    label: "AdjustBrightness",
    value: "AdjustBrightness",
  },
  {
    display_name: "Affine",
    object_name: "transforms.functional.affine",
    transform_type: "functional",
    parameters: {
      angle: {
        index: 0,
        parameter_name: "angle",
        min: -180,
        max: 180,
        default: 0,
        parameter_type: "number",
      } as LayerParameter<"number">,
      translate: {
        index: 1,
        parameter_name: "translate",
        min: -1000,
        max: 1000,
        default: "(0, 0)",
        parameter_type: "tuple",
      } as LayerParameter<"tuple">,
      scale: {
        index: 2,
        parameter_name: "scale",
        min: 0,
        max: 1000,
        default: 1,
        parameter_type: "number",
      } as LayerParameter<"number">,
      shear: {
        index: 3,
        parameter_name: "shear",
        min: -180,
        max: 180,
        default: "(0, 0)",
        parameter_type: "tuple",
      } as LayerParameter<"tuple">,
    },
    tooltip_info: (
      <>
        <p className="info">
          <strong>
            Apply affine transformation on the image keeping image center
            invariant.
          </strong>
        </p>
        <p className="info">
          <strong>Parameters</strong>
        </p>
        <ul>
          <li>
            <em>angle</em> - clockwise rotation angle between -180 and 180
          </li>
          <li>
            <em>translate</em> - post-rotatation translation
          </li>
          <li>
            <em>scale</em> - overall scale
          </li>
          <li>
            <em>shear</em> - shear angle value between -180 and 180
          </li>
        </ul>

        <a href="https://pytorch.org/vision/main/generated/torchvision.transforms.functional.affine.html">
          More info
        </a>
      </>
    ),
    label: "Affine",
    value: "Affine",
  },
];

const CLASSIFICATION = "classification";
const REGRESSION = "regression";

export type ProblemType = typeof CLASSIFICATION | typeof REGRESSION;

interface BaseCriterion {
  label: string;
  value: string;
  object_name: string;
}

export interface Criterion extends BaseCriterion {
  problem_type: ProblemType[];
}

export const CRITERIONS: Criterion[] = [
  {
    label: "L1LOSS",
    value: "L1LOSS",
    object_name: "nn.L1Loss()",
    problem_type: [REGRESSION],
  },
  {
    label: "MSELOSS",
    value: "MSELOSS",
    object_name: "nn.MSELoss()",
    problem_type: [REGRESSION],
  },
  {
    label: "BCELOSS",
    value: "BCELOSS",
    object_name: "nn.BCELoss()",
    problem_type: [CLASSIFICATION],
  },
  {
    label: "CELOSS",
    value: "CELOSS",
    object_name: "nn.CrossEntropyLoss(reduction='mean')",
    problem_type: [CLASSIFICATION],
  },
];

export const IMAGE_CLASSIFICATION_CRITERION: BaseCriterion[] = [
  {
    label: "CELOSS",
    value: "CELOSS",
    object_name: "nn.CrossEntropyLoss()",
  },
  {
    label: "WCELOSS",
    value: "WCELOSS",
    object_name: "nn.CrossEntropyLoss()", // will define a randomized weights for classes in backend
  },
];

export const PROBLEM_TYPES = Object.freeze([
  { label: "Classification", value: CLASSIFICATION },
  { label: "Regression", value: REGRESSION },
]);

export const OBJECT_DETECTION_PROBLEM_TYPES = Object.freeze([
  { label: "Labels", value: "labels" },
  { label: "Celebrities", value: "celebrities" },
]);

export const DETECTION_TYPES = Object.freeze([
  { label: "Rekognition", value: "rekognition" },
  { label: "YOLO", value: "yolo" },
]);

export const BOOL_OPTIONS = Object.freeze([
  { label: "False", value: false },
  { label: "True", value: true },
]);
export const OPTIMIZER_NAMES = Object.freeze([
  { label: "SGD", value: "SGD" },
  { label: "Adam", value: "Adam" },
]);

export interface DefaultDatasetType {
  label: string;
  value: string | null;
}
export const DEFAULT_DATASETS = Object.freeze([
  { label: "NONE", value: null },
  { label: "IRIS", value: "IRIS" },
  { label: "CALIFORNIAHOUSING", value: "CALIFORNIAHOUSING" },
  { label: "DIABETES", value: "DIABETES" },
  { label: "DIGITS", value: "DIGITS" },
  { label: "WINE", value: "WINE" },
]);

export const IMAGE_DEFAULT_DATASETS = Object.freeze([
  { label: "MNIST", value: "MNIST" },
  { label: "FashionMNIST", value: "FashionMNIST" },
  { label: "CIFAR10", value: "CIFAR10" },
]);

export const PRETRAINED_MODELS = Object.freeze([
  { label: "RESNET18", value: "RESNET18" },
]);
