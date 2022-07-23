export const POSSIBLE_LAYERS = [
  {
    display_name: "Linear",
    object_name: "nn.Linear",
    parameters: {
      inputSize: { index: 0, parameter_name: "Input size" },
      outputSize: { index: 1, parameter_name: "Output size" },
    },
    tooltip_info: (
      <>
        <p>
          <strong>
            Applies a linear transformation to the incoming data:{" "}
            <i>
              y = xA
              <sup>T</sup> + b
            </i>
            .
          </strong>
        </p>
        <p>
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
        <p>
          <strong>
            Applies the rectified linear unit function element-wise: ReLU
            <i>
              (x) = (x)<sup>+</sup> ={" "}
            </i>
            max(0, <i>x)</i>.
          </strong>
        </p>
        <p>
          <strong>Parameters</strong>
        </p>
        <p>
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
      inputSize: { index: 0, parameter_name: "dim" },
    },
    tooltip_info: (
      <>
        <p>
          <strong>
            Applies the Softmax function to an <i>n</i>-dimensional input Tensor
            rescaling them so that the elements of the <i>n</i>-dimensional
            output Tensor lie in the range [0,1] and sum to 1.
          </strong>
        </p>
        <p>
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
        <p>
          <strong>Applies the Sigmoid function.</strong>
        </p>
        <p>
          <strong>Parameters</strong>
        </p>
        <p>
          <i>None</i>
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
        <p>
          <strong>
            Applies the Hyperbolic Tangent (Tanh) function element-wise.
          </strong>
        </p>
        <p>
          <strong>Parameters</strong>
        </p>
        <p>
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
      inputSize: { index: 0, parameter_name: "dim" },
    },
    tooltip_info: (
      <>
        <p>
          <strong>
            Applies the log(Softmax(<i>x</i>)) function to an <i>n</i>
            -dimensional input Tensor.
          </strong>
        </p>
        <p>
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

export const IMAGE_LAYERS = [
  {
    display_name: "Conv2D",
    object_name: "nn.Conv2D",
    parameters: {
      
    }
  }
];

export const POSSIBLE_TRANSFORMS = [
  {
    display_name: "Random Horizontal Flip",
    object_name: "transforms.RandomHorizontalFlip",
    parameters: {
      probability: { index: 0, parameter_name: "Prob" },
    },
    label: "Random Horizontal Flip",
    value: "HF",
  },

  {
    display_name: "Random Vertical Flip",
    object_name: "transforms.RandomVerticalFlip",
    parameters: {
      probability: { index: 0, parameter_name: "Prob" },
    },
    label: "Random Vertical Flip",
    value: "VF",
  },

  {
    display_name: "To Tensor",
    object_name: "transforms.ToTensor",
    parameters: {},
    label: "To Tensor",
    value: "TT",
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

export const IMAGE_CLASSIFICATION_CRITERION = [
  {
    label: "CELOSS",
    value: "CELOSS",
    object_name: "nn.CrossEntropyLoss",
  },
  {
    label: "WCELOSS",
    value: "WCELOSS",
    object_name: "nn.CrossEntropyLoss", //will define a randomized weights for classes in backend
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

export const DEFAULT_DATASETS = [
  { label: "IRIS", value: "IRIS" },
  { label: "CALIFORNIAHOUSING", value: "CALIFORNIAHOUSING" },
  { label: "DIABETES", value: "DIABETES" },
  { label: "DIGITS", value: "DIGITS" },
  { label: "WINE", value: "WINE" },
];

export const IMAGE_DEFAULT_DATASETS = [
  {label: "MNIST", value: "MNIST"},
  {label: "FashionMNIST", value: "FASHIONMNIST"},
  {label: "CIFAR10", value: "CIFAR10"}
]

export const PRETRAINED_MODELS = [
  {label: "RESNET18", value: "RESNET18"}
]