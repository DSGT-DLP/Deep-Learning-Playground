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
        description:
          "The `CONV2d` function applies a filter to input data in a sliding window manner. By performing element-wise multiplication and sum of the overlapping portions, it captures local spatial patterns and features, enabling applications such as image recognition, object detection, and semantic segmentation in computer vision tasks.",
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
        description:
          "MaxPool2d function reduces the size of input data by dividing it into non-overlapping rectangular regions and selecting the maximum value from each region. This downsampling operation preserves important features while decreasing spatial dimensions, making it beneficial for tasks like image classification and extracting spatial characteristics.",
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
        description:
          "The flatten operation takes a two-dimensional image representation and transforms it into a one-dimensional vector. This process unravels the image structure by concatenating the rows or columns of pixels, creating a linear sequence of values. By doing so, it allows the network to process the image as a simple list of numbers, facilitating tasks such as image classification or object detection in neural networks.",
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
        description:
          "A linear layer in an image dataset takes the flattened image representation and applies a learned linear transformation to map the input values to a new set of values, allowing the network to learn complex relationships for tasks like image classification or object recognition.",
      },
      SIGMOID: {
        label: "Sigmoid",
        objectName: "nn.Sigmoid",
        parameters: [],
        description:
          "The sigmoid function takes the input values, typically the output of a linear layer, and applies a mathematical function that compresses them into a range between 0 and 1. This transformation is useful for interpreting the output as probabilities, where values closer to 1 indicate higher confidence in the presence of a particular feature or class in the image.",
      },
    },
    transformValues: [
      "GAUSSIAN_BLUR",
      "GRAYSCALE",
      "NORMALIZE",
      "RANDOM_HORIZONTAL_FLIP",
      "RANDOM_VERTICAL_FLIP",
      "RESIZE",
      "TO_TENSOR",
    ],
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

export type ALL_LAYERS = keyof typeof STEP_SETTINGS.PARAMETERS.layers;

export const DEFAULT_LAYERS: {
  MNIST: { value: ALL_LAYERS; parameters: number[] }[];
} = {
  MNIST: [
    { value: "CONV2D", parameters: [1, 5, 3, 1, 1] },
    { value: "MAXPOOL2D", parameters: [3, 1] },
    { value: "FLATTEN", parameters: [1, -1] },
    { value: "LINEAR", parameters: [500, 10] },
    { value: "SIGMOID", parameters: [] },
  ],
};
