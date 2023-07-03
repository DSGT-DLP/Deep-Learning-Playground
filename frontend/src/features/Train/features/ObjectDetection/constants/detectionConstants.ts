import DetectionImageStep from "../components/DetectionImageStep";
import DetectionParametersStep from "../components/DetectionParametersStep";
import DetectionReviewStep from "../components/DetectionReviewStep";
import TabularTrainspace from "../components/DetectionTrainspace";

export const TRAINSPACE_SETTINGS = {
  name: "Detection",
  steps: ["IMAGE", "PARAMETERS", "REVIEW"],
  component: TabularTrainspace,
} as const;

export const STEP_SETTINGS = {
  IMAGE: {
    name: "Image",
    optional: false,
    component: DetectionImageStep,
  },
  PARAMETERS: {
    name: "Parameters",
    optional: false,
    component: DetectionParametersStep,
    detectionTypes: [
      { label: "Rekognition", value: "rekognition" },
      { label: "YOLO", value: "yolo" },
    ],
    detectionProblemTypes: [
      { label: "Labels", value: "labels" },
      { label: "Celebrities", value: "celebrities" },
    ],
    detectionTransformValues: [
      "RandomHorizontalFlip",
      "ToTensor",
      "RandomVerticalFlip",
      "Resize",
      "GaussianBlur",
      "Grayscale",
      "Normalize",
      "AdjustContrast",
      "AdjustBrightness",
      "Affine",
    ],
    detectionTransforms: {
      RandomHorizontalFlip: {
        objectName: "transforms.RandomHorizontalFlip",
        parameters: [
          {
            label: "probability",
            min: 0,
            max: 1,
            type: "number",
          },
        ],
        label: "Random Horizontal Flip",
        value: "RandomHorizontalFlip",
      },

      ToTensor: {
        objectName: "transforms.ToTensor",
        parameters: [],
        label: "To Tensor",
        value: "ToTensor",
      },

      RandomVerticalFlip: {
        objectName: "transforms.RandomVerticalFlip",
        parameters: [
          {
            label: "probability",
            min: 0,
            max: 1,
            type: "number",
          },
        ],
        label: "Random Vertical Flip",
        value: "RandomVerticalFlip",
      },
      Resize: {
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
        label: "Resize",
        value: "Resize",
      },
      GaussianBlur: {
        objectName: "transforms.GaussianBlur",
        parameters: [
          {
            label: "kernel size",
            min: 1,
            max: 1000,
            type: "number",
          },
        ],
        label: "Gaussian Blur",
        value: "GaussianBlur",
      },
      Grayscale: {
        objectName: "transforms.Grayscale",
        parameters: [],
        label: "Grayscale",
        value: "Grayscale",
      },
      Normalize: {
        objectName: "transforms.Normalize",
        parameters: [
          {
            label: "mean",
            min: -1000,
            max: 1000,
            default: 0,
            type: "number",
          },
          {
            label: "std",
            min: -1000,
            max: 1000,
            default: 1,
            type: "number",
          },
        ],
        label: "Normalize",
        value: "Normalize",
      },
      AdjustContrast: {
        objectName: "transforms.functional.adjust_contrast",
        parameters: [
          {
            label: "contrast_factor",
            min: 0,
            max: 10000,
            default: 0,
            type: "number",
          },
        ],
        label: "AdjustContrast",
        value: "AdjustContrast",
      },
      AdjustBrightness: {
        objectName: "transforms.functional.adjust_brightness",
        parameters: [
          {
            label: "brightness_factor",
            min: 0,
            max: 10000,
            default: 0,
            type: "number",
          },
        ],
        label: "AdjustBrightness",
        value: "AdjustBrightness",
      },
      Affine: {
        objectName: "transforms.functional.affine",
        parameters: [
          {
            label: "angle",
            min: -180,
            max: 180,
            default: 0,
            type: "number",
          },
          {
            label: "translate",
            min: -1000,
            max: 1000,
            default: "(0, 0)",
            type: "tuple",
          },
          {
            label: "scale",
            min: 0,
            max: 1000,
            default: 1,
            type: "number",
          },
          {
            label: "shear",
            min: -180,
            max: 180,
            default: "(0, 0)",
            type: "tuple",
          },
        ],
        label: "Affine",
        value: "Affine",
      },
    },
  },
  REVIEW: {
    name: "Review",
    optional: false,
    component: DetectionReviewStep,
  },
} as const;
