import ImageDatasetStep from "../components/ImageDatasetStep";
import ImageParametersStep from "../components/ImageParametersStep";
import ImageReviewStep from "../components/ImageReviewStep";
import ImageTrainspace from "../components/ImageTrainspace";

export const TRAINSPACE_SETTINGS = {
  name: "Image",
  steps: ["DATASET", "PARAMETERS", "REVIEW"],
  component: ImageTrainspace,
} as const;

export const STEP_SETTINGS = {
  DATASET: {
    name: "Dataset",
    optional: false,
    component: ImageDatasetStep,
    defaultDatasets: [
      { label: "MNIST", value: "MNIST" },
      { label: "FashionMNIST", value: "FashionMNIST" },
      { label: "CIFAR10", value: "CIFAR10" },
    ],
  },
  PARAMETERS: {
    name: "Parameters",
    optional: false,
    component: ImageParametersStep,
    criterions: [
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
    ],
    optimizers: [
      { label: "Stochastic Gradient Descent", value: "SGD" },
      { label: "Adam Optimization", value: "Adam" },
    ],
    layerValues: ["CONV2D", "MAXPOOL2D", "FLATTEN", "LINEAR", "SIGMOID"], 
    layers: {
      CONV2D: {
        label: "Conv2d",
        objectName: "nn.Conv2d",
        parameters: [
          {
            label: "Input Channels",
            min: 1,
            max: 16,
            required: true,
            type: "number",
          },
          {
            label: "Output Channels",
            min: 1,
            max: 16,
            required: true,
            type: "number",
          },
          {
            label: "Kernel Size",
            min: 1,
            max: 1000,
            required: true,
            type: "number",
          },
          {
            label: "Stride",
            min: 1,
            max: 1000,
            required: true,
            type: "number",
          },
          {
            label: "Padding",
            min: 1,
            max: 1000,
            required: true,
            type: "number",
          },
        ],
      },
      MAXPOOL2D: {
        label: "MaxPool2d",
        objectName: "nn.MaxPool2d",
        parameters: [
          {
            label: "Kernel Size",
            min: 1,
            max: 1000,
            required: true,
            type: "number",
          },
          {
            label: "Stride",
            min: 1,
            max: 1000,
            required: true,
            type: "number",
          },
        ],
      },
      FLATTEN: {
        label: "Flatten",
        objectName: "nn.Flatten",
        parameters: [
          {
            label: "Start Dimension",
            min: -4,
            max: 3,
            required: true,
            type: "number",
          },
          {
            label: "End Dimension",
            min: -4,
            max: 3,
            required: true,
            type: "number",
          },
        ],
      },
      LINEAR: {
        label: "Linear",
        objectName: "nn.Linear",
        parameters: [
          {
            label: "Input Size",
            min: 1,
            max: 1600,
            required: true,
            type: "number",
          },
          {
            label: "Output Size",
            min: 1,
            max: 1600,
            required: true,
            type: "number",
          },
        ],
      },
      SIGMOID: {
        label: "Sigmoid",
        objectName: "nn.Sigmoid",
        parameters: [],
      },
    },
    transformValues: ["GAUSSIAN_BLUR", "GRAYSCALE", "NORMALIZE", "RANDOM_HORIZONTAL_FLIP", "RANDOM_VERTICAL_FLIP", "RESIZE", "TO_TENSOR"],
    transforms: {
      GAUSSIAN_BLUR: {
        label: "Gaussian Blur",
        objectName: "transforms.GaussianBlur",
        parameters: [
          {
            label: "Kernel Size",
            min: 1,
            max: 1000,
            required: true,
            type: "number",
          },
        ],
      },
      GRAYSCALE: {
        label: "Grayscale",
        objectName: "transforms.Grayscale",
        parameters: [],
      },
      NORMALIZE: {
        label: "Normalize",
        objectName: "transforms.Normalize",
        parameters: [
          {
            label: "Mean",
            min: -1000,
            max: 1000,
            required: true,
            type: "number",
          },
          {
            label: "Standard Deviation",
            min: -1000,
            max: 1000,
            required: true,
            type: "number",
          },
        ],
      },
      RANDOM_HORIZONTAL_FLIP: {
        label: "Random Horizontal Flip",
        objectName: "transforms.RandomHorizontalFlip",
        parameters: [
          {
            label: "Probability",
            min: 0,
            max: 1,
            required: true,
            type: "number",
          },
        ],
      },
      RANDOM_VERTICAL_FLIP: {
        label: "Random Vertical Flip",
        objectName: "transforms.RandomVerticalFlip",
        parameters: [
          {
            label: "Probability",
            min: 0,
            max: 1,
            required: true,
            type: "number",
          },
        ],
      },
      RESIZE: {
        label: "Resize",
        objectName: "transforms.Resize",
        parameters: [
          {
            label: "Height",
            min: 1,
            max: 1000,
            required: true,
            type: "number",
          },
          {
            label: "Width",
            min: 1,
            max: 1000,
            required: true,
            type: "number",
          },
        ],
      },
      TO_TENSOR: {
        label: "To Tensor",
        objectName: "transforms.ToTensor",
        parameters: [],
      },
    },
},
  REVIEW: {
    name: "Review",
    optional: false,
    component: ImageReviewStep,
  },
} as const;
