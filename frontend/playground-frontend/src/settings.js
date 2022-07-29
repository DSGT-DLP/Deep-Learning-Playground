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
    object_name: "nn.Conv2d",
    parameters: {
      in_channels: {
        index: 0,
        parameter_name: "in channels",
      },
      out_channels: {
        index: 1,
        parameter_name: "out channels",
      },
      kernel_size: {
        index: 2,
        parameter_name: "kernel size",
      },
      stride: {
        index: 3,
        parameter_name: "stride",
      },
      padding: {
        index: 4,
        parameter_name: "padding",
      },
    },
    tooltip_info: (
      <>
        <p>
          <strong>
            Applies a 2D convolution over an input signal composed of several
            input planes.
          </strong>
        </p>
        <p>
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
      },
    },
    tooltip_info: (
      <>
        <p>
          <strong>
            Applies Batch Normalization over a 4D input (a mini-batch of 2D
            inputs with additional channel dimension) as described in the paper{" "}
            <a href="https://arxiv.org/abs/1502.03167">
              Batch Normalization: Accelerating Deep Network Training by
              Reducing Internal Covariate Shift.
            </a>
          </strong>
        </p>
        <p>
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
      kernel_size: { index: 0, parameter_name: "Kernel size" },
      stride: {
        index: 1,
        parameter_name: "stride",
      },
    },
    tooltip_info: (
      <>
        <p>
          <strong>
            Applies a 2D max pooling over an input signal composed of several
            input planes.
          </strong>
        </p>
        <p>
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
    display_name: "AdaptiveAvgPool2d",
    object_name: "nn.AdaptiveAvgPool2d",
    parameters: {
      output_size: { index: 0, parameter_name: "Output size" },
    },
    tooltip_info: (
      <>
        <p>
          <strong>
            Applies a 2D adaptive average pooling over an input signal composed
            of several input planes.
          </strong>
        </p>
        <p>
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
      p: { index: 0, parameter_name: "Probability" },
    },
    tooltip_info: (
      <>
        <p>
          <strong>
            During training, randomly zeroes some of the elements of the input
            tensor with probability p using samples from a Bernoulli
            distribution. Each channel will be zeroed out independently on every
            forward call.
          </strong>
        </p>
        <p>
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
      start_dim: { index: 0, parameter_name: "start dim" },
      end_dim: { index: 1, parameter_name: "end dim" },
    },
    tooltip_info: (
      <>
        <p>
          <strong>Flattens a contiguous range of dims into a tensor.</strong>
        </p>
        <p>
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
      p: { index: 0, parameter_name: "probability" },
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
  {
    display_name: "Resize",
    object_name: "transforms.Resize",
    parameters: {
      size: { index: 0, parameter_name: "(H, W)" },
    },
    label: "Resize",
    value: "R",
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
    object_name: "nn.CrossEntropyLoss()",
  },
  {
    label: "WCELOSS",
    value: "WCELOSS",
    object_name: "nn.CrossEntropyLoss()", // will define a randomized weights for classes in backend
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