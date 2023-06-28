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
    detectionProblemTypes : [
      { label: "Labels", value: "labels" },
      { label: "Celebrities", value: "celebrities" },
    ],
    detectionTransformValues: ["RandomHorizontalFlip", "ToTensor", "RandomVerticalFlip", "Resize", "GaussianBlur", "Grayscale", "Normalize", "AdjustContrast", "AdjustBrightness", "Affine"],
    detectionTransforms: {
      RandomHorizontalFlip: {
        display_name: "Random Horizontal Flip",
        objectName: "transforms.RandomHorizontalFlip",
        parameters: [
          {
            label: "probability",
            index: 0,
            parameter_name: "prob",
            min: 0,
            max: 1,
            type: "number",
          },
        ],
        label: "Random Horizontal Flip",
        value: "RandomHorizontalFlip",
      },
    
      ToTensor: {
        display_name: "To Tensor",
        objectName: "transforms.ToTensor",
        parameters: [],
        label: "To Tensor",
        value: "ToTensor",
      },

      RandomVerticalFlip: {
        display_name: "Random Vertical Flip",
        objectName: "transforms.RandomVerticalFlip",
        parameters: [
          {
            label: "probability",
            index: 0,
            parameter_name: "prob",
            min: 0,
            max: 1,
            type: "number",
          },
        ],
        label: "Random Vertical Flip",
        value: "RandomVerticalFlip",
      },
      Resize: {
        display_name: "Resize",
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
        display_name: "Gaussian Blur",
        objectName: "transforms.GaussianBlur",
        parameters: [
          {
            label: "kernel size",
            index: 0,
            parameter_name: "kernel size",
            min: 1,
            max: 1000,
            type: "number",
          },
        ],
        label: "Gaussian Blur",
        value: "GaussianBlur",
      },
      Grayscale: {
        display_name: "Grayscale",
        objectName: "transforms.Grayscale",
        parameters: [],
        label: "Grayscale",
        value: "Grayscale",
      },
      Normalize: {
        display_name: "Normalize",
        objectName: "transforms.Normalize",
        parameters: [
          {
            label: "mean",
            index: 0,
            parameter_name: "mean",
            min: -1000,
            max: 1000,
            default: 0,
            type: "number",
          },
          {
            label: "std",
            index: 1,
            parameter_name: "std",
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
        display_name: "AdjustContrast",
        objectName: "transforms.functional.adjust_contrast",
        transform_type: "functional",
        parameters: [
          {
            label: "contrast_factor",
            index: 0,
            parameter_name: "contrast_factor",
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
        display_name: "AdjustBrightness",
        objectName: "transforms.functional.adjust_brightness",
        transform_type: "functional",
        parameters: [
          {
            label: "brightness_factor",
            index: 0,
            parameter_name: "brightness_factor",
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
        display_name: "Affine",
        objectName: "transforms.functional.affine",
        transform_type: "functional",
        parameters: [
          {
            label: "angle",
            index: 0,
            parameter_name: "angle",
            min: -180,
            max: 180,
            default: 0,
            type: "number",
          },
          {
            label: "translate",
            index: 1,
            parameter_name: "translate",
            min: -1000,
            max: 1000,
            default: "(0, 0)",
            type: "tuple",
          },
          {
            label: "scale",
            index: 2,
            parameter_name: "scale",
            min: 0,
            max: 1000,
            default: 1,
            type: "number",
          },
          {
            label: "shear",
            index: 3,
            parameter_name: "shear",
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
